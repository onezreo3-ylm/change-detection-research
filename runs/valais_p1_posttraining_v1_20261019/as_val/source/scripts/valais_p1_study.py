"""Prepare, smoke, seal, train and inspect the authorized P1 controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import LEVIRDataset, file_sha256
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import load_robust_config, robust_source_files, run_two_view_epoch, tensor_state_sha256
from pcdc.runtime import setup_runtime, write_json
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.stratified_sampling import StratifiedBatchSampler, load_sampling
from scripts.train_stratified_robust import make_sampled_loader, sampling_identity, trainer_hashes

ROOT = Path(__file__).resolve().parents[1]
STUDY_KEYS = {"schema_version", "name", "output_dir", "python", "runs", "sampling_proposal", "guard_config",
              "protocol_document", "initial_model_tensor_sha256", "minimum_unittests", "benchmark_config",
              "baseline_evaluation", "check_interval_minutes"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    source = Path(path).resolve()
    value = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    if set(value) != STUDY_KEYS or value["schema_version"] != 1 or set(value["runs"]) != {"cs", "as"}:
        raise ValueError("Unexpected P1 study schema")
    paths = {key: (source.parent / value[key]).resolve() for key in
             ("output_dir", "python", "sampling_proposal", "guard_config", "protocol_document",
              "benchmark_config", "baseline_evaluation")}
    paths["output_dir"].relative_to(ROOT / "runs")
    runs = {key: load_robust_config((source.parent / target).resolve()) for key, target in value["runs"].items()}
    cs, augmented = runs["cs"], runs["as"]
    if (cs.mode != "b0e" or augmented.mode != "b1" or cs.base.values != augmented.base.values
            or cs.protocol.fingerprint != augmented.protocol.fingerprint
            or cs.paths["severity_review"] != augmented.paths["severity_review"]):
        raise ValueError("C-S/A-S do not share the frozen base, corruption or review")
    for cfg in runs.values():
        base = cfg.base.values
        cfg.output_dir.relative_to(ROOT / "runs")
        if (base["experiment"]["kind"] != "research" or base["experiment"]["seed"] != 42
                or base["runtime"]["device"] != "cuda" or base["train"]["amp"] is not False
                or base["train"]["epochs"] != 100 or base["train"]["batch_size"] != 4
                or cfg.base.run_test_after_training or base["evaluation"]["threshold"] != 0.5
                or cfg.values["expected_source_tiles"] != 11025
                or cfg.values["expected_batches_per_epoch"] != 2757
                or base["model"]["encoder_weights"] is not None):
            raise ValueError("P1 changed the reviewed frozen training/inference settings")
    outputs = [paths["output_dir"], *[cfg.output_dir for cfg in runs.values()]]
    if len(set(outputs)) != 3 or any(a in b.parents for a in outputs for b in outputs if a != b):
        raise ValueError("Output directories alias or nest")
    if not paths["python"].is_file() or value["check_interval_minutes"] != 90:
        raise ValueError("Missing requested runtime or wrong managed cadence")
    return source, value, paths, runs


def planned_sampling(paths, cfg):
    value, _, positive = load_sampling(paths["sampling_proposal"])
    dataset = LEVIRDataset(cfg.base, "train")
    if len(dataset.tiles) != 11025 or len(dataset.records) != len(dataset.tiles):
        raise ValueError("Unexpected train layout")
    names = [record.name for record in dataset.records]
    sampler = StratifiedBatchSampler(names, positive, value["sampling"])
    rows = []
    for epoch in range(1, 101):
        sampler.set_epoch(epoch)
        row = sampler.plan().summary
        if (row["source_draws"] != 11025 or row["batches"] != 2757
                or row["positive_draws"] != 2757 or row["empty_draws"] != 8268
                or row["unique_positive_sources"] != 1046 or row["unique_empty_sources"] != 8268
                or row["positive_repeat_min"] != 2 or row["positive_repeat_max"] != 3):
            raise ValueError("The exact reviewed exposure/repetition contract failed")
        rows.append(row)
    return rows


def prepare(path):
    source, value, paths, runs = load(path)
    if paths["output_dir"].exists() or any(cfg.output_dir.exists() for cfg in runs.values()):
        raise FileExistsError("P1 output already exists")
    rows = planned_sampling(paths, runs["cs"])
    paths["output_dir"].mkdir(parents=True, exist_ok=False)
    write_json(paths["output_dir"] / "sampling_plan.json", rows)
    write_json(paths["output_dir"] / "preflight.json", {
        "status": "sampling_preflight_passed", "test_accessed": False, "training_started": False,
        "epochs": len(rows), "source_draws_total": 1102500, "updates_total": 275700,
        "positive_draws_total": 275700, "empty_draws_total": 826800,
        "intentional_protocol_change": "train_source_sampling_only",
        "same_source_stream_for_cs_and_as": True,
        "sampling_proposal_sha256": file_sha256(paths["sampling_proposal"]),
    })
    return {"status": "sampling_preflight_passed", "epochs_checked": 100}


def smoke(path):
    source, value, paths, runs = load(path)
    output = paths["output_dir"] / "real_cuda_smoke"
    output.mkdir(exist_ok=False)
    guard = EvaluationTemperatureGuard(load_guard_config(paths["guard_config"]), output)
    results = {}
    try:
        guard(force=True)
        for role, cfg in runs.items():
            device = setup_runtime(cfg.base)
            dataset = LEVIRDataset(cfg.base, "train")
            paired, sampler, _, identity = make_sampled_loader(dataset, cfg, paths["sampling_proposal"])
            indices = sampler.plan().batches[0]
            captured = read(load_sampling(paths["sampling_proposal"])[1]["captured_data_audit"])
            for index in indices:
                record = dataset.records[index]
                expected = captured["splits"]["train"]["file_sha256"][record.name]
                for key in ("image_a", "image_b", "mask"):
                    if file_sha256(getattr(record, key)) != expected[key]:
                        raise ValueError("Smoke train source changed")
            loader = DataLoader(Subset(paired, list(indices)), batch_size=4, num_workers=0)
            model = move_model(build_model(cfg.base), device, cfg.base.values["runtime"]["memory_format"])
            initial = tensor_state_sha256(model.state_dict())
            if initial != value["initial_model_tensor_sha256"]:
                raise ValueError("Smoke initialization differs from the archived baseline")
            counts = {name: int(module.num_batches_tracked) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            settings = cfg.base.values["train"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                         betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            report = run_two_view_epoch(
                model, loader, ChangeDetectionLoss(**cfg.base.values["loss"]), device,
                threshold=0.5, consistency_kind="none", consistency_weight=0.0, amp=False,
                memory_format=cfg.base.values["runtime"]["memory_format"], optimizer=optimizer,
                scaler=torch.cuda.amp.GradScaler(enabled=False), grad_clip_norm=settings["grad_clip_norm"],
                guard_callback=guard)
            if (report["optimizer_steps"] != 1 or report["amp_skipped_steps"] != 0 or report["pair_views"] != 8
                    or any(int(module.num_batches_tracked) != counts[name] + 1
                           for name, module in model.named_modules() if name in counts)):
                raise ValueError("Real CUDA FP32 smoke violated update/view/BN contract")
            results[role] = {
                "status": "real_cuda_fp32_two_view_smoke_passed", "research_result": False,
                "initial_model_tensor_sha256": initial,
                "source_ids": [dataset.records[i].name for i in indices],
                "source_batch_size": 4, "pair_views": 8, "optimizer_steps": 1,
                "one_batchnorm_update": True, "loss_finite": True,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "seconds": time.monotonic() - started,
                "sampling_identity": identity,
            }
            del model, optimizer, loader
            torch.cuda.empty_cache()
        if results["cs"]["source_ids"] != results["as"]["source_ids"]:
            raise ValueError("Smoke source batches do not match")
        guard.finish({"status": "real_cuda_p1_smoke_complete"})
        write_json(output / "report.json", {"status": "real_cuda_p1_smoke_complete", "runs": results,
                   "test_accessed": False, "val_accessed": False, "research_result": False})
        return {"status": "real_cuda_p1_smoke_complete", "roles": list(results)}
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        raise


def seal(path):
    source, value, paths, runs = load(path)
    output = paths["output_dir"]
    verified = read(output / "verification.json")
    log = output / "unittest.log"
    text = log.read_text(encoding="utf-8")
    match = re.search(r"Ran (\d+) tests", text)
    if (verified["status"] != "passed" or not match or int(match[1]) < value["minimum_unittests"]
            or verified["unittest_log_sha256"] != file_sha256(log)
            or not re.search(r"(?m)^OK\s*$", text) or "FAILED" in text or "skipped=" in text):
        raise ValueError("The final complete unittest run did not pass")
    if read(output / "real_cuda_smoke/report.json")["status"] != "real_cuda_p1_smoke_complete":
        raise ValueError("Missing passing real CUDA smoke")
    files = {str(source), *(str(p) for p in robust_source_files().values()),
             *(str(ROOT / name) for name in trainer_hashes()),
             str(Path(__file__).resolve()), str(ROOT / "tests/test_stratified_sampling.py"),
             *(str(paths[key]) for key in ("sampling_proposal", "guard_config", "protocol_document")),
             str(output / "sampling_plan.json"), str(output / "preflight.json"),
             str(output / "verification.json"), str(log), str(output / "real_cuda_smoke/report.json"),
             str(output / "real_cuda_smoke/temperature_guard_state.json"),
             str(ROOT / "scripts/train_robust_guarded.py"), str(ROOT / "scripts/evaluate_guarded.py"),
             str(ROOT / "scripts/temperature_guard.py"), str(ROOT / "scripts/run_valais_clean_guarded.py")}
    _, evidence_paths, _ = load_sampling(paths["sampling_proposal"])
    files.update(str(p) for p in evidence_paths.values())
    for cfg in runs.values():
        files.add(str(cfg.source))
        files.update(str(cfg.paths[key]) for key in ("base_config", "protocol_config", "severity_review"))
    record = {
        "status": "user_authorized_p1_managed_training", "authorized_utc": now(),
        "human_instruction": "继续完善实验，尝试探索创新方向并开始正式训练，开启托管模式，每一个半小时核查下，并尽量节省token",
        "scope": ["C-S_100_epochs", "A-S_100_epochs", "frozen_validation", "train_only_residual_coupling_diagnostic",
                  "evidence_based_innovation_research"],
        "test_accessed": False, "automatic_numeric_or_temperature_retry": False,
        "initial_model_tensor_sha256": value["initial_model_tensor_sha256"],
        "bound_sha256": {name: file_sha256(Path(name)) for name in sorted(files)},
    }
    write_json(output / "authorization.json", record)
    return {"status": record["status"], "sealed_files": len(files)}


def verify_seal(paths):
    auth = read(paths["output_dir"] / "authorization.json")
    if auth["status"] != "user_authorized_p1_managed_training" or auth["test_accessed"] is not False:
        raise ValueError("Wrong managed training authorization")
    for name, expected in auth["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return auth


def check_completion(cfg, paths, initial):
    status = read(cfg.output_dir / "status.json")
    completion = read(cfg.output_dir / "guard_completion.json")
    actual = read(cfg.output_dir / "sampling_actual.json")
    planned = read(paths["output_dir"] / "sampling_plan.json")
    state = read(cfg.output_dir / completion["guard_state_path"])
    if (status["status"] != "completed" or status["epochs"] != 100 or status["optimizer_steps"] != 275700
            or status["amp_skipped_steps"] != 0 or status["pair_views"] != 2205000
            or status["test_evaluated"] is not False or status["initial_model_tensor_sha256"] != initial
            or completion["status"] != "completed"
            or completion["training_status_sha256"] != file_sha256(cfg.output_dir / "status.json")
            or completion["guard_state_sha256"] != file_sha256(cfg.output_dir / completion["guard_state_path"])
            or state["status"] != "completed" or len(actual) != 100):
        raise ValueError("P1 training completion/budget/guard failed")
    for observed, expected in zip(actual, planned):
        if observed != dict(expected, actual_source_stream_verified=True):
            raise ValueError("Completed actual source batches differ from the sealed plan")
    return status


def run(path):
    source, value, paths, runs = load(path)
    verify_seal(paths)
    output = paths["output_dir"]
    owner = output / "owner.json"
    # An exclusive start record prevents both an agent and a scheduled check from launching twice.
    write_json(owner, {"pid": os.getpid(), "creation_ticks": creation_time_ticks(os.getpid()),
                      "command": [sys.executable, "-m", "scripts.valais_p1_study", "run", "--config", str(source)],
                      "started_utc": now(), "automatic_retry": False})
    role, child = "cs", None
    completed = {}
    try:
        for role, cfg in runs.items():
            verify_seal(paths)
            command = [str(paths["python"]), "-u", "-m", "scripts.train_stratified_robust",
                       "--config", str(cfg.source), "--sampling-proposal", str(paths["sampling_proposal"]),
                       "--guard-config", str(paths["guard_config"]), "--initial-sha256",
                       value["initial_model_tensor_sha256"]]
            with (output / f"{role}.stdout.log").open("x", encoding="utf-8") as stdout, \
                    (output / f"{role}.stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                _atomic_json(output / "status.json", {
                    "status": "running", "stage": role + "_training", "updated_utc": now(),
                    "owner_pid": os.getpid(), "child_pid": child.pid,
                    "child_creation_ticks": creation_time_ticks(child.pid), "child_command": command,
                    "completed_roles": list(completed), "test_accessed": False,
                })
                code = child.wait()
            if code:
                raise RuntimeError(f"{role} training exited {code}; preserve failure before recovery")
            completed[role] = check_completion(cfg, paths, value["initial_model_tensor_sha256"])
            write_json(output / f"{role}.exit.json", {"exit_code": code, "completed_utc": now(),
                       "status_sha256": file_sha256(cfg.output_dir / "status.json")})
        if read(runs["cs"].output_dir / "sampling_actual.json") != read(runs["as"].output_dir / "sampling_actual.json"):
            raise ValueError("C-S/A-S actual source streams differ")
        _atomic_json(output / "status.json", {
            "status": "completed_training_ready_for_validation", "completed_roles": list(completed),
            "completed_utc": now(), "test_accessed": False, "matched_source_stream_verified": True,
            "next": "Freeze candidate validation bindings, score original 45 conditions, then train-only residual P0.",
        })
    except BaseException as error:
        _atomic_json(output / "status.json", {
            "status": "failed_preserve_before_recovery", "stage": role + "_training",
            "error": f"{type(error).__name__}: {error}", "updated_utc": now(),
            "owner_pid": os.getpid(), "child_pid": child.pid if child else None,
            "completed_roles": list(completed), "test_accessed": False, "automatic_retry": False,
        })
        raise


def monitor(path):
    _, value, paths, runs = load(path)
    output = paths["output_dir"]
    summary = {"study": value["name"], "checked_utc": now(), "test_accessed": False}
    queue = read(output / "status.json") if (output / "status.json").is_file() else {}
    summary["queue"] = {key: queue[key] for key in
                        ("status", "stage", "completed_roles", "updated_utc", "completed_utc", "error") if key in queue}
    for role, cfg in runs.items():
        state = cfg.output_dir / "status.json"
        status = read(state) if state.is_file() else {"status": "not_started"}
        summary[role] = {key: status[key] for key in
                         ("status", "epoch", "epochs", "best_epoch", "best_val_f1", "error") if key in status}
        log = output / f"{role}.stdout.log"
        if log.is_file():
            # Read only the tail; avoid all checkpoint deserialization and full histories.
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 2048))
                summary[role]["log_tail"] = stream.read().decode("utf-8", errors="replace").splitlines()[-3:]
            summary[role]["log_modified_utc"] = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat()
    try:
        summary["child_identity_alive"] = (
            queue.get("status") == "running"
            and creation_time_ticks(queue["child_pid"]) == queue["child_creation_ticks"])
    except (OSError, RuntimeError, ValueError, KeyError):
        summary["child_identity_alive"] = False
    if queue.get("status") == "running" and not summary["child_identity_alive"]:
        summary["attention"] = "Recorded child ended without queue completion; inspect stderr and preserve before recovery."
    try:
        guard = load_guard_config(paths["guard_config"])
        telemetry = subprocess.run(
            [guard.values["nvidia_smi"], "-i", "0",
             "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout.strip().split(",")
        uuid, temperature, utilization, memory = [item.strip() for item in telemetry]
        if uuid != guard.values["gpu_uuid"]:
            raise ValueError("Monitored GPU UUID differs")
        summary["gpu"] = {"temperature_c": int(temperature), "utilization_percent": int(utilization),
                          "memory_used_mib": int(memory)}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        summary["telemetry_error"] = f"{type(error).__name__}: {error}"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "seal", "run", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.action](args.config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
