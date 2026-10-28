"""One sealed recovery of the interrupted P1 queue; original trainer stays frozen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import LEVIRDataset, file_sha256, seed_worker
from pcdc.engine import move_model, read_checkpoint
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import _digest, _restore_rng, robust_source_hashes, run_two_view_epoch, tensor_state_sha256
from pcdc.runtime import setup_runtime, write_json
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.train_stratified_robust import make_sampled_loader, sampling_identity, trainer_hashes, validate_resume
from scripts import valais_p1_study as parent

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    source = Path(path).resolve()
    value = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    keys = {"schema_version", "parent_study", "output_dir", "expected_saved_epoch",
            "interrupted_epoch", "minimum_unittests", "recovery_limit", "protocol_document"}
    if set(value) != keys or value["schema_version"] != 1 or value["recovery_limit"] != 1:
        raise ValueError("Unexpected one-shot recovery schema")
    if value["interrupted_epoch"] != value["expected_saved_epoch"] + 1:
        raise ValueError("Recovery must restart the first uncommitted epoch")
    paths = {key: (source.parent / value[key]).resolve()
             for key in ("parent_study", "output_dir", "protocol_document")}
    paths["output_dir"].relative_to(ROOT / "runs")
    study, settings, study_paths, runs = parent.load(paths["parent_study"])
    return source, value, paths, settings, study_paths, runs


def old_processes_stopped(study_paths):
    owner = parent.read(study_paths["output_dir"] / "owner.json")
    queue = parent.read(study_paths["output_dir"] / "status.json")
    states = {"owner": process_snapshot(owner["pid"], owner["creation_ticks"]),
              "worker": process_snapshot(queue["child_pid"], queue["child_creation_ticks"])}
    if any(state["owned_active"] for state in states.values()):
        raise RuntimeError("Original owner/worker is still active; do not duplicate GPU work")
    return states


def check_prefix_history(history, actual, planned, epochs):
    if [row["epoch"] for row in history] != list(range(1, epochs + 1)) or len(actual) != epochs:
        raise ValueError("Checkpoint/history prefix is incomplete")
    for row, observed, expected in zip(history, actual, planned[:epochs]):
        if (row["train"]["sampling"] != observed
                or observed != dict(expected, actual_source_stream_verified=True)
                or row["train"]["optimizer_steps"] != 2757
                or row["train"]["amp_skipped_steps"] != 0 or row["train"]["pair_views"] != 22050):
            raise ValueError("Saved actual sampling/update prefix differs from its sealed plan")
    if len(planned) < epochs:
        raise ValueError("Planned sampling prefix is incomplete")


def checkpoint_prefix(value, paths, settings, study_paths, runs):
    backup = paths["output_dir"] / "saved_prefix"
    last = read_checkpoint(backup / "cs/last.pt")
    best = read_checkpoint(backup / "cs/best.pt")
    cfg = runs["cs"]
    _, evidence, _, identity = sampling_identity(study_paths["sampling_proposal"])
    audit = parent.read(evidence["captured_data_audit"])
    torch.set_num_threads(cfg.base.values["runtime"]["num_threads"])
    validate_resume(last, cfg, identity, robust_source_hashes(), trainer_hashes(),
                    _digest(audit), settings["initial_model_tensor_sha256"])
    epochs = value["expected_saved_epoch"]
    if (last["epoch"] != epochs or last["robust_config"] != dict(cfg.values)
            or set(last["file_sha256"]) != {"train", "val"}
            or any("test" in row for row in last["history"])):
        raise ValueError("Wrong checkpoint prefix or data scope")
    if last["history"] != parent.read(backup / "cs/history.json"):
        raise ValueError("Saved history file and checkpoint disagree")
    check_prefix_history(last["history"], parent.read(backup / "cs/sampling_actual.json"),
                         parent.read(backup / "study/sampling_plan.json"), epochs)
    if (best["epoch"] != last["best_epoch"] or best["history"] != last["history"][:best["epoch"]]
            or best["best_val_f1"] != last["best_val_f1"]
            or best["sampling_identity"] != last["sampling_identity"]):
        raise ValueError("Best checkpoint is not the original clean-val selection")
    if set(last["rng"]) != {"python", "numpy", "torch", "cuda", "loader"} or len(last["rng"]["cuda"]) != 1:
        raise ValueError("Incomplete saved RNG state")
    tensors = list(last["model"].values())
    tensors.extend(v for state in last["optimizer"]["state"].values() for v in state.values()
                   if isinstance(v, torch.Tensor))
    if any(torch.is_floating_point(t) and not torch.isfinite(t).all().item() for t in tensors):
        raise FloatingPointError("Saved model/optimizer is non-finite; recovery refused")
    return last


def prepare(path):
    source, value, paths, settings, study_paths, runs = load(path)
    parent.verify_seal(study_paths)
    stopped = old_processes_stopped(study_paths)
    if runs["as"].output_dir.exists():
        raise ValueError("This recovery expects A-S not yet started")
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    backup = output / "saved_prefix"
    files = {
        "study": (study_paths["output_dir"], ["owner.json", "status.json", "authorization.json",
                  "sampling_plan.json", "cs.stdout.log", "cs.stderr.log", "queue.stdout.log", "queue.stderr.log"]),
        "cs": (runs["cs"].output_dir, ["last.pt", "best.pt", "status.json", "history.json",
               "sampling_actual.json", "training_exposure_summary.json", "temperature_guard_state.json",
               "source_sha256.json", "stratified_trainer_source_sha256.json", "sampling_identity.json"]),
    }
    copied = {}
    for label, (folder, names) in files.items():
        (backup / label).mkdir(parents=True)
        for name in names:
            original, target = folder / name, backup / label / name
            shutil.copy2(original, target)
            expected = file_sha256(original)
            if file_sha256(target) != expected:
                raise ValueError("Source changed while preserving the interrupted run")
            copied[str(target)] = expected
    shutil.copytree(runs["cs"].output_dir / "temperature_guard", backup / "cs/temperature_guard")
    guard = parent.read(backup / "cs/temperature_guard_state.json")
    if guard["decision"] != "ok" or any(row["decision"] != "ok" for row in guard["observations"]):
        raise ValueError("A recorded temperature/sensor stop needs separate diagnosis")
    if (backup / "study/cs.stderr.log").stat().st_size or (backup / "study/queue.stderr.log").stat().st_size:
        raise ValueError("Recorded stderr requires diagnosis before this external-exit recovery")
    last = checkpoint_prefix(value, paths, settings, study_paths, runs)
    old_processes_stopped(study_paths)
    record = {
        "status": "preserved_prefix_verified", "recorded_utc": parent.now(), "old_processes": stopped,
        "saved_epoch": last["epoch"], "resume_epoch": last["epoch"] + 1,
        "best_epoch": last["best_epoch"], "best_val_f1": last["best_val_f1"],
        "committed_optimizer_steps": sum(row["train"]["optimizer_steps"] for row in last["history"]),
        "committed_pair_views": sum(row["train"]["pair_views"] for row in last["history"]),
        "guard_last_observation": guard["observations"][-1],
        "guard_observed_peak_c": max(row["temperature_c"] for row in guard["observations"] if row["temperature_c"] is not None),
        "discarded_partial_epoch": value["interrupted_epoch"], "last_logged_partial_step": 1375,
        "exact_discarded_updates": None,
        "cause": "Process termination confirmed; no recorded guard/numeric error. External cause not established.",
        "physical_compute_note": "Partial epoch is replayed from the saved state; discarded physical compute is additional.",
        "bound_backup_sha256": copied, "test_accessed": False,
    }
    write_json(output / "preflight.json", record)
    return {k: record[k] for k in ("status", "saved_epoch", "resume_epoch", "committed_optimizer_steps")}


def smoke(path):
    _, value, paths, settings, study_paths, runs = load(path)
    old_processes_stopped(study_paths)
    last = checkpoint_prefix(value, paths, settings, study_paths, runs)
    output = paths["output_dir"] / "resume_smoke"
    output.mkdir(exist_ok=False)
    guard = EvaluationTemperatureGuard(load_guard_config(study_paths["guard_config"]), output)
    cfg = runs["cs"]
    device = setup_runtime(cfg.base)
    dataset = LEVIRDataset(cfg.base, "train")
    paired, sampler, full_loader, _ = make_sampled_loader(dataset, cfg, study_paths["sampling_proposal"])
    paired.set_epoch(value["interrupted_epoch"])
    sampler.set_epoch(value["interrupted_epoch"])
    plan = sampler.plan()
    expected = parent.read(study_paths["output_dir"] / "sampling_plan.json")[value["expected_saved_epoch"]]
    if plan.summary != expected:
        raise ValueError("First resumed epoch differs from the frozen sampler plan")
    results = []
    try:
        guard(force=True)
        for repeat in range(2):
            loader = DataLoader(Subset(paired, list(plan.batches[0])), batch_size=4,
                                num_workers=cfg.base.values["train"]["num_workers"], pin_memory=True,
                                worker_init_fn=seed_worker, generator=full_loader.generator)
            model = move_model(build_model(cfg.base, initialize_encoder=False), device,
                               cfg.base.values["runtime"]["memory_format"])
            model.load_state_dict(last["model"], strict=True)
            train_settings = cfg.base.values["train"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_settings["learning_rate"],
                                         betas=tuple(train_settings["adam_betas"]), weight_decay=train_settings["weight_decay"])
            optimizer.load_state_dict(last["optimizer"])
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            scaler.load_state_dict(last["scaler"])
            before = {name: int(module.num_batches_tracked) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            _restore_rng(last["rng"], loader)
            report = run_two_view_epoch(
                model, loader, ChangeDetectionLoss(**cfg.base.values["loss"]), device,
                threshold=cfg.base.values["evaluation"]["threshold"],
                consistency_kind="none", consistency_weight=0.0, amp=False,
                memory_format=cfg.base.values["runtime"]["memory_format"],
                optimizer=optimizer, scaler=scaler, grad_clip_norm=train_settings["grad_clip_norm"],
                guard_callback=guard)
            if report["optimizer_steps"] != 1 or report["pair_views"] != 8 or report["amp_skipped_steps"]:
                raise ValueError("Resumed-state smoke has a different update/view budget")
            if any(int(module.num_batches_tracked) != before[name] + 1
                   for name, module in model.named_modules() if name in before):
                raise ValueError("Resumed-state smoke changed BN update semantics")
            state = {f"{index}/{key}": tensor for index, item in optimizer.state_dict()["state"].items()
                     for key, tensor in item.items() if isinstance(tensor, torch.Tensor)}
            results.append({"model_sha256": tensor_state_sha256(model.state_dict()),
                            "optimizer_sha256": tensor_state_sha256(state), "loss": report["loss"]})
            del model, optimizer, loader
            torch.cuda.empty_cache()
        if results[0] != results[1]:
            raise ValueError("Restoring the same saved state did not reproduce the next update")
        guard.finish({"status": "resumed_state_cuda_smoke_passed"})
        record = {"status": "resumed_state_cuda_smoke_passed", "research_result": False,
                  "resume_epoch": value["interrupted_epoch"], "repeated_next_update_exact": True,
                  "source_ids": [dataset.records[i].name for i in plan.batches[0]],
                  "source_batch_size": 4, "pair_views_per_repeat": 8, "repeats": 2,
                  "test_accessed": False, "val_accessed": False, "new_state": results[0]}
        write_json(output / "report.json", record)
        return {key: record[key] for key in ("status", "resume_epoch", "repeated_next_update_exact")}
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        raise


def seal(path):
    source, value, paths, settings, study_paths, runs = load(path)
    old_processes_stopped(study_paths)
    parent.verify_seal(study_paths)
    output = paths["output_dir"]
    log = output / "unittest.log"
    text = log.read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", text)
    if not count or int(count[1]) < value["minimum_unittests"] or not re.search(r"(?m)^OK\s*$", text):
        raise ValueError("Complete regression suite did not pass")
    if "FAILED" in text or "skipped=" in text:
        raise ValueError("Regression failure/skip prevents recovery")
    if parent.read(output / "resume_smoke/report.json")["status"] != "resumed_state_cuda_smoke_passed":
        raise ValueError("Missing passing resumed-state real CUDA smoke")
    paths_to_bind = [source, paths["protocol_document"], Path(__file__),
                     ROOT / "scripts/p1_process_state.py", ROOT / "tests/test_p1_recovery.py", log,
                     output / "preflight.json", output / "resume_smoke/report.json"]
    paths_to_bind += [p for p in (output / "saved_prefix").rglob("*") if p.is_file()]
    write_json(output / "authorization.json", {
        "status": "explicit_user_resume_authorized", "human_instruction": "恢复并继续",
        "authorized_utc": parent.now(), "saved_epoch": value["expected_saved_epoch"], "recovery_limit": 1,
        "next": ["resume_cs_to_100", "fresh_as_100", "frozen_validation_then_train_residual_diagnostic"],
        "test_accessed": False, "bound_sha256": {str(p.resolve()): file_sha256(p) for p in paths_to_bind},
    })
    return {"status": "explicit_user_resume_authorized", "tests": int(count[1])}


def verify_seal(paths, study_paths):
    parent.verify_seal(study_paths)
    auth = parent.read(paths["output_dir"] / "authorization.json")
    if auth["status"] != "explicit_user_resume_authorized" or auth["recovery_limit"] != 1:
        raise ValueError("Unexpected resume authorization")
    for name, expected in auth["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen recovery input changed: {name}")


def run(path):
    source, value, paths, settings, study_paths, runs = load(path)
    verify_seal(paths, study_paths)
    old_processes_stopped(study_paths)
    output = paths["output_dir"]
    write_json(output / "owner.json", {"pid": os.getpid(), "creation_ticks": creation_time_ticks(os.getpid()),
               "started_utc": parent.now(), "config": str(source), "automatic_retry": False})
    prefix_sha = parent.read(output / "preflight.json")["bound_backup_sha256"]
    if file_sha256(runs["cs"].output_dir / "last.pt") != prefix_sha[str(output / "saved_prefix/cs/last.pt")]:
        raise ValueError("Live last.pt changed after preservation")
    role, child, child_ticks, completed = "cs", None, None, []
    try:
        for role, cfg in runs.items():
            verify_seal(paths, study_paths)
            command = [str(study_paths["python"]), "-u", "-m", "scripts.train_stratified_robust",
                       "--config", str(cfg.source), "--sampling-proposal", str(study_paths["sampling_proposal"]),
                       "--guard-config", str(study_paths["guard_config"]), "--initial-sha256",
                       settings["initial_model_tensor_sha256"]]
            if role == "cs":
                command.append("--resume")
            with (output / f"{role}.stdout.log").open("x", encoding="utf-8") as stdout, \
                    (output / f"{role}.stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                child_ticks = creation_time_ticks(child.pid)
                _atomic_json(output / "status.json", {
                    "status": "running", "stage": role + "_training", "updated_utc": parent.now(),
                    "owner_pid": os.getpid(), "child_pid": child.pid, "child_creation_ticks": child_ticks,
                    "completed_roles": list(completed), "test_accessed": False,
                    "saved_cs_epoch": value["expected_saved_epoch"], "child_command": command,
                })
                code = child.wait()
            if code:
                raise RuntimeError(f"{role} exited {code}; this recovery does not retry")
            parent.check_completion(cfg, study_paths, settings["initial_model_tensor_sha256"])
            completed.append(role)
            write_json(output / f"{role}.exit.json", {
                "exit_code": code, "completed_utc": parent.now(),
                "training_status_sha256": file_sha256(cfg.output_dir / "status.json"),
            })
        if parent.read(runs["cs"].output_dir / "sampling_actual.json") != parent.read(runs["as"].output_dir / "sampling_actual.json"):
            raise ValueError("Effective C-S/A-S sampling streams differ")
        _atomic_json(output / "status.json", {
            "status": "completed_training_ready_for_validation", "completed_roles": completed,
            "completed_utc": parent.now(), "matched_effective_source_stream_verified": True,
            "discarded_partial_cs_epoch": value["interrupted_epoch"], "test_accessed": False,
        })
    except BaseException as error:
        _atomic_json(output / "status.json", {
            "status": "failed_preserve_before_recovery", "stage": role + "_training",
            "error": f"{type(error).__name__}: {error}", "child_pid": child.pid if child else None,
            "child_creation_ticks": child_ticks, "completed_roles": completed, "test_accessed": False,
            "automatic_retry": False,
        })
        raise


def monitor(path):
    _, value, paths, settings, study_paths, runs = load(path)
    result = parent.monitor(paths["parent_study"])
    result.pop("attention", None)
    output = paths["output_dir"]
    queue = parent.read(output / "status.json") if (output / "status.json").is_file() else {}
    result["queue"] = {k: queue[k] for k in ("status", "stage", "completed_roles", "error", "completed_utc") if k in queue}
    result["recovery"] = str(output)
    for role, cfg in runs.items():
        log = output / f"{role}.stdout.log"
        result[role].pop("log_tail", None)
        result[role].pop("log_modified_utc", None)
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 2048))
                result[role]["log_tail"] = stream.read().decode("utf-8", errors="replace").splitlines()[-3:]
            result[role]["log_age_seconds"] = round(time.time() - log.stat().st_mtime, 1)
        checkpoint = cfg.output_dir / "last.pt"
        if checkpoint.is_file():
            result[role]["checkpoint_modified_utc"] = datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat()
    owner = parent.read(output / "owner.json") if (output / "owner.json").is_file() else {}
    result["owner_identity_alive"] = bool(owner) and process_snapshot(owner["pid"], owner["creation_ticks"])["owned_active"]
    result["child_identity_alive"] = bool(queue.get("child_pid")) and process_snapshot(
        queue["child_pid"], queue["child_creation_ticks"])["owned_active"]
    if owner and not queue:
        result["attention"] = "Recovery owner exists without a queue state; inspect the launch error."
    if queue.get("status") == "running":
        if not result["owner_identity_alive"] or not result["child_identity_alive"]:
            result["attention"] = "Process termination detected; preserve state, no automatic duplicate/retry."
        role = queue.get("stage", "").split("_")[0]
        if result.get(role, {}).get("log_age_seconds", 0) > 1800:
            result["attention"] = "No recent logged progress; investigate even if the process is active."
    return result


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
