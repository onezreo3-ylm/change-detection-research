"""One explicitly authorized epoch-boundary recovery of the paused A-S row."""
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
from pcdc.robust_baselines import _digest, _restore_rng, _rng_state, robust_source_hashes, run_two_view_epoch, tensor_state_sha256
from pcdc.runtime import setup_runtime, write_json
from scripts import resume_valais_p1_study as previous
from scripts import valais_p1_study as parent
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.train_stratified_robust import make_sampled_loader, sampling_identity, trainer_hashes, validate_resume

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    source = Path(path).resolve()
    value = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    keys = {"schema_version", "parent_study", "previous_recovery", "output_dir", "pause_receipt",
            "expected_saved_epoch", "resume_epoch", "minimum_unittests", "recovery_limit", "protocol_document"}
    if set(value) != keys or value["schema_version"] != 1 or value["recovery_limit"] != 1:
        raise ValueError("Unexpected A-S one-shot recovery schema")
    if not 0 < value["expected_saved_epoch"] < 100 or value["resume_epoch"] != value["expected_saved_epoch"] + 1:
        raise ValueError("Resume must begin at the first uncommitted epoch")
    paths = {key: (source.parent / value[key]).resolve() for key in
             ("parent_study", "previous_recovery", "output_dir", "pause_receipt", "protocol_document")}
    paths["output_dir"].relative_to(ROOT / "runs")
    _, settings, study_paths, runs = parent.load(paths["parent_study"])
    _, _, old_paths, _, _, _ = previous.load(paths["previous_recovery"])
    return source, value, paths, settings, study_paths, runs, old_paths


def stopped(study_paths, old_paths, receipt):
    states = previous.old_processes_stopped(study_paths)
    owner, queue = parent.read(old_paths["output_dir"] / "owner.json"), parent.read(old_paths["output_dir"] / "status.json")
    states.update(previous_owner=process_snapshot(owner["pid"], owner["creation_ticks"]),
                  previous_worker=process_snapshot(queue["child_pid"], queue["child_creation_ticks"]))
    if receipt["status"] != "paused_by_user" or not all(row["confirmed_exited"] for row in receipt["stop_results"]):
        raise ValueError("Missing confirmed human pause")
    if any(row["owned_active"] for row in states.values()):
        raise RuntimeError("An original owner/worker is still active")
    return states


def validate_prefix(last, best, cfg, history, actual, planned, epochs):
    if (last["epoch"] != epochs or last["robust_mode"] != "b1" or last["robust_config"] != dict(cfg.values)
            or set(last["file_sha256"]) != {"train", "val"} or any("test" in row for row in last["history"])
            or last["history"] != history):
        raise ValueError("Wrong A-S saved epoch, configuration, history or data scope")
    previous.check_prefix_history(history, actual, planned, epochs)
    selected = max(history, key=lambda row: row["val"]["metrics"]["f1"])
    if (best["epoch"] != selected["epoch"] or last["best_epoch"] != selected["epoch"]
            or last["best_val_f1"] != selected["val"]["metrics"]["f1"]
            or best["best_val_f1"] != last["best_val_f1"]
            or best["history"] != history[:best["epoch"]]
            or best["sampling_identity"] != last["sampling_identity"]):
        raise ValueError("Best checkpoint is not the first highest clean-val selection")
    if set(last["rng"]) != {"python", "numpy", "torch", "cuda", "loader"} or len(last["rng"]["cuda"]) != 1:
        raise ValueError("Incomplete saved RNG")
    tensors = list(last["model"].values()) + [v for state in last["optimizer"]["state"].values()
                                             for v in state.values() if isinstance(v, torch.Tensor)]
    if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in tensors):
        raise FloatingPointError("Non-finite saved state")


def checkpoint_prefix(value, paths, settings, study_paths, runs):
    folder, cfg = paths["output_dir"] / "saved_prefix/as", runs["as"]
    last, best = read_checkpoint(folder / "last.pt"), read_checkpoint(folder / "best.pt")
    _, evidence, _, identity = sampling_identity(study_paths["sampling_proposal"])
    torch.set_num_threads(cfg.base.values["runtime"]["num_threads"])
    for checkpoint in (last, best):
        validate_resume(checkpoint, cfg, identity, robust_source_hashes(), trainer_hashes(),
                        _digest(parent.read(evidence["captured_data_audit"])), settings["initial_model_tensor_sha256"])
    validate_prefix(last, best, cfg, parent.read(folder / "history.json"), parent.read(folder / "sampling_actual.json"),
                    parent.read(study_paths["output_dir"] / "sampling_plan.json"), value["expected_saved_epoch"])
    return last


def rng_digest(state):
    tensors = {"torch": state["torch"], "loader": state["loader"]}
    tensors.update({f"cuda/{i}": tensor for i, tensor in enumerate(state["cuda"])})
    return _digest({"python": state["python"], "numpy": state["numpy"],
                    "tensor_rng_sha256": tensor_state_sha256(tensors)})


def prepare(path):
    source, value, paths, settings, study_paths, runs, old_paths = load(path)
    previous.verify_seal(old_paths, study_paths)
    receipt = parent.read(paths["pause_receipt"])
    states = stopped(study_paths, old_paths, receipt)
    if receipt["committed_as_epoch"] != value["expected_saved_epoch"] or receipt["completed_cs_epochs"] != 100:
        raise ValueError("Pause receipt has a different committed boundary")
    parent.check_completion(runs["cs"], study_paths, settings["initial_model_tensor_sha256"])
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    folders = {"as": runs["as"].output_dir, "cs": runs["cs"].output_dir,
               "previous_queue": old_paths["output_dir"], "original_queue": study_paths["output_dir"]}
    names = {"as": ["best.pt", "last.pt", "history.json", "status.json", "sampling_actual.json",
                    "training_exposure_summary.json", "temperature_guard_state.json", "source_sha256.json",
                    "stratified_trainer_source_sha256.json", "sampling_identity.json"],
             "cs": ["status.json", "guard_completion.json", "sampling_actual.json", "history.json"],
             "previous_queue": ["owner.json", "status.json", "authorization.json", "as.stdout.log", "as.stderr.log",
                                "cs.exit.json", "queue.stdout.log", "queue.stderr.log"],
             "original_queue": ["owner.json", "status.json", "authorization.json", "sampling_plan.json"]}
    copied = {}
    for role, folder in folders.items():
        for name in names[role]:
            original, target = folder / name, output / "saved_prefix" / role / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = file_sha256(original)
            shutil.copy2(original, target)
            if file_sha256(target) != expected or file_sha256(original) != expected:
                raise ValueError("Paused artifact changed during preservation")
            copied[str(target)] = expected
    shutil.copy2(paths["pause_receipt"], output / "pause_receipt.json")
    for folder in (runs["as"].output_dir / "temperature_guard",):
        target = output / "saved_prefix/as" / folder.name
        shutil.copytree(folder, target)
        for original in folder.rglob("*"):
            if original.is_file():
                copy = target / original.relative_to(folder)
                if file_sha256(copy) != file_sha256(original):
                    raise ValueError("Guard copy changed")
                copied[str(copy)] = file_sha256(copy)
    last = checkpoint_prefix(value, paths, settings, study_paths, runs)
    guard = parent.read(output / "saved_prefix/as/temperature_guard_state.json")
    if guard["decision"] != "ok" or any(row["decision"] != "ok" for row in guard["observations"]):
        raise ValueError("Recorded temperature stop requires diagnosis")
    actual = parent.read(output / "saved_prefix/as/sampling_actual.json")
    if actual != parent.read(runs["cs"].output_dir / "sampling_actual.json")[:len(actual)]:
        raise ValueError("A-S committed source stream differs from C-S")
    stopped(study_paths, old_paths, receipt)
    write_json(output / "preflight.json", {
        "status": "paused_as_prefix_verified", "recorded_utc": parent.now(), "old_processes": states,
        "saved_epoch": last["epoch"], "resume_epoch": value["resume_epoch"],
        "best_epoch": last["best_epoch"], "best_val_f1": last["best_val_f1"],
        "committed_optimizer_steps": sum(row["train"]["optimizer_steps"] for row in last["history"]),
        "committed_pair_views": sum(row["train"]["pair_views"] for row in last["history"]),
        "matched_source_prefix_verified": True, "cs_completion_verified": True,
        "guard_observed_peak_c": max(row["temperature_c"] for row in guard["observations"] if row["temperature_c"] is not None),
        "discarded_uncommitted_updates": None, "test_accessed": False,
        "physical_compute_note": "Any uncommitted work and recovery smoke are additional physical compute, excluded from effective budgets.",
        "bound_backup_sha256": copied,
    })
    return {"status": "paused_as_prefix_verified", "saved_epoch": last["epoch"], "resume_epoch": value["resume_epoch"]}


def smoke(path):
    _, value, paths, settings, study_paths, runs, old_paths = load(path)
    stopped(study_paths, old_paths, parent.read(paths["pause_receipt"]))
    last = checkpoint_prefix(value, paths, settings, study_paths, runs)
    cfg = runs["as"]
    output = paths["output_dir"] / "resume_smoke"
    output.mkdir(exist_ok=False)
    guard = EvaluationTemperatureGuard(load_guard_config(study_paths["guard_config"]), output)
    device = setup_runtime(cfg.base)
    dataset = LEVIRDataset(cfg.base, "train")
    paired, sampler, full_loader, _ = make_sampled_loader(dataset, cfg, study_paths["sampling_proposal"])
    paired.set_epoch(value["resume_epoch"])
    sampler.set_epoch(value["resume_epoch"])
    plan = sampler.plan()
    if plan.summary != parent.read(study_paths["output_dir"] / "sampling_plan.json")[value["expected_saved_epoch"]]:
        raise ValueError("Resumed sampling differs")
    for index in plan.batches[0]:
        record = dataset.records[index]
        for key in ("image_a", "image_b", "mask"):
            if file_sha256(getattr(record, key)) != last["file_sha256"]["train"][record.name][key]:
                raise ValueError("Smoke source content changed")
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
            train = cfg.base.values["train"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
                                         betas=tuple(train["adam_betas"]), weight_decay=train["weight_decay"])
            optimizer.load_state_dict(last["optimizer"])
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            scaler.load_state_dict(last["scaler"])
            before = {name: int(module.num_batches_tracked) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            _restore_rng(last["rng"], loader)
            report = run_two_view_epoch(model, loader, ChangeDetectionLoss(**cfg.base.values["loss"]), device,
                threshold=0.5, consistency_kind="none", consistency_weight=0.0, amp=False,
                memory_format=cfg.base.values["runtime"]["memory_format"], optimizer=optimizer, scaler=scaler,
                grad_clip_norm=train["grad_clip_norm"], guard_callback=guard)
            if report["optimizer_steps"] != 1 or report["pair_views"] != 8 or report["amp_skipped_steps"]:
                raise ValueError("Wrong resumed update budget")
            if any(int(module.num_batches_tracked) != before[name] + 1
                   for name, module in model.named_modules() if name in before):
                raise ValueError("Wrong resumed BN budget")
            state = {f"{index}/{key}": tensor for index, item in optimizer.state_dict()["state"].items()
                     for key, tensor in item.items() if isinstance(tensor, torch.Tensor)}
            results.append({"model_sha256": tensor_state_sha256(model.state_dict()),
                            "optimizer_sha256": tensor_state_sha256(state), "loss": report["loss"],
                            "rng_sha256": rng_digest(_rng_state(loader))})
            del model, optimizer, loader
            torch.cuda.empty_cache()
        if results[0] != results[1]:
            raise ValueError("Resumed next update is not exactly reproducible")
        guard.finish({"status": "resumed_as_cuda_smoke_passed"})
        write_json(output / "report.json", {
            "status": "resumed_as_cuda_smoke_passed", "research_result": False,
            "resume_epoch": value["resume_epoch"], "repeated_next_update_exact": True,
            "source_ids": [dataset.records[i].name for i in plan.batches[0]],
            "source_batch_size": 4, "pair_views_per_repeat": 8, "repeats": 2,
            "test_accessed": False, "val_accessed": False, "new_state": results[0],
        })
        return {"status": "resumed_as_cuda_smoke_passed", "repeated_next_update_exact": True}
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        raise


def seal(path):
    source, value, paths, _, study_paths, _, old_paths = load(path)
    previous.verify_seal(old_paths, study_paths)
    stopped(study_paths, old_paths, parent.read(paths["pause_receipt"]))
    output = paths["output_dir"]
    log = (output / "unittest.log").read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", log)
    if (not count or int(count[1]) < value["minimum_unittests"] or not re.search(r"(?m)^OK\s*$", log)
            or "FAILED" in log or "skipped=" in log):
        raise ValueError("Full unittests did not pass")
    if parent.read(output / "resume_smoke/report.json")["status"] != "resumed_as_cuda_smoke_passed":
        raise ValueError("Missing passing real saved-state smoke")
    files = [source, paths["protocol_document"], paths["pause_receipt"], Path(__file__),
             ROOT / "tests/test_p1_as_recovery.py", output / "unittest.log", output / "preflight.json",
             output / "resume_smoke/report.json", output / "resume_smoke/temperature_guard_state.json"]
    files.extend(p for p in (output / "saved_prefix").rglob("*") if p.is_file())
    write_json(output / "authorization.json", {
        "status": "explicit_user_as_resume_authorized", "human_instruction": "恢复实验，并继续进行可行性验证，并开始执行训练任务",
        "authorized_utc": parent.now(), "saved_epoch": value["expected_saved_epoch"], "recovery_limit": 1,
        "next": ["resume_as_to_100", "frozen_validation", "train_only_residual_coupling_diagnostic"],
        "test_accessed": False, "bound_sha256": {str(p.resolve()): file_sha256(p) for p in files},
    })
    return {"status": "explicit_user_as_resume_authorized", "tests": int(count[1])}


def verify(paths, study_paths, old_paths):
    previous.verify_seal(old_paths, study_paths)
    auth = parent.read(paths["output_dir"] / "authorization.json")
    if auth["status"] != "explicit_user_as_resume_authorized" or auth["recovery_limit"] != 1:
        raise ValueError("Wrong A-S recovery authorization")
    for name, expected in auth["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen A-S recovery input changed: {name}")


def run(path):
    source, value, paths, settings, study_paths, runs, old_paths = load(path)
    verify(paths, study_paths, old_paths)
    stopped(study_paths, old_paths, parent.read(paths["pause_receipt"]))
    parent.check_completion(runs["cs"], study_paths, settings["initial_model_tensor_sha256"])
    output, cfg = paths["output_dir"], runs["as"]
    for name in ("last.pt", "best.pt", "history.json", "sampling_actual.json", "status.json"):
        if file_sha256(cfg.output_dir / name) != file_sha256(output / "saved_prefix/as" / name):
            raise ValueError("Live paused state changed after preservation")
    write_json(output / "owner.json", {"pid": os.getpid(), "creation_ticks": creation_time_ticks(os.getpid()),
               "started_utc": parent.now(), "config": str(source), "automatic_retry": False})
    child, ticks = None, None
    try:
        command = [str(study_paths["python"]), "-u", "-m", "scripts.train_stratified_robust",
                   "--config", str(cfg.source), "--sampling-proposal", str(study_paths["sampling_proposal"]),
                   "--guard-config", str(study_paths["guard_config"]),
                   "--initial-sha256", settings["initial_model_tensor_sha256"], "--resume"]
        with (output / "as.stdout.log").open("x", encoding="utf-8") as stdout, (output / "as.stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            ticks = creation_time_ticks(child.pid)
            _atomic_json(output / "status.json", {"status": "running", "stage": "as_training", "updated_utc": parent.now(),
                "owner_pid": os.getpid(), "child_pid": child.pid, "child_creation_ticks": ticks,
                "completed_roles": ["cs"], "saved_as_epoch": value["expected_saved_epoch"],
                "child_command": command, "test_accessed": False})
            code = child.wait()
        if code:
            raise RuntimeError(f"A-S exited {code}; no automatic retry")
        for cfg in runs.values():
            parent.check_completion(cfg, study_paths, settings["initial_model_tensor_sha256"])
        if parent.read(runs["cs"].output_dir / "sampling_actual.json") != parent.read(runs["as"].output_dir / "sampling_actual.json"):
            raise ValueError("Completed C-S/A-S source streams differ")
        write_json(output / "as.exit.json", {"exit_code": code, "completed_utc": parent.now(),
            "training_status_sha256": file_sha256(runs["as"].output_dir / "status.json")})
        _atomic_json(output / "status.json", {"status": "completed_training_ready_for_validation",
            "completed_roles": ["cs", "as"], "completed_utc": parent.now(), "test_accessed": False,
            "matched_effective_source_stream_verified": True,
            "next": "Explicit stratified-flavor validation adapter, original 45 conditions, fixed train residual diagnostic."})
    except BaseException as error:
        _atomic_json(output / "status.json", {"status": "failed_preserve_before_recovery", "stage": "as_training",
            "error": f"{type(error).__name__}: {error}", "child_pid": child.pid if child else None,
            "child_creation_ticks": ticks, "completed_roles": ["cs"], "test_accessed": False, "automatic_retry": False})
        raise


def monitor(path):
    _, _, paths, _, _, runs, _ = load(path)
    result = parent.monitor(paths["parent_study"])
    result.pop("attention", None)
    output = paths["output_dir"]
    queue = parent.read(output / "status.json") if (output / "status.json").is_file() else {}
    result["queue"] = {k: queue[k] for k in ("status", "stage", "completed_roles", "completed_utc", "error") if k in queue}
    result["recovery"] = str(output)
    for role, cfg in runs.items():
        result[role].pop("log_tail", None)
        result[role].pop("log_modified_utc", None)
        log = output / "as.stdout.log" if role == "as" else None
        if log and log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 2048))
                result[role]["log_tail"] = stream.read().decode("utf-8", errors="replace").splitlines()[-3:]
            result[role]["log_age_seconds"] = round(time.time() - log.stat().st_mtime, 1)
        checkpoint = cfg.output_dir / "last.pt"
        result[role]["checkpoint_modified_utc"] = datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat()
    owner = parent.read(output / "owner.json") if (output / "owner.json").is_file() else {}
    result["owner_identity_alive"] = bool(owner) and process_snapshot(owner["pid"], owner["creation_ticks"])["owned_active"]
    result["child_identity_alive"] = bool(queue.get("child_pid")) and process_snapshot(queue["child_pid"], queue["child_creation_ticks"])["owned_active"]
    if queue.get("status") == "running":
        if not result["owner_identity_alive"] or not result["child_identity_alive"]:
            result["attention"] = "Process ended; preserve state and do not automatically retry."
        elif result.get("as", {}).get("log_age_seconds", 0) > 1800:
            result["attention"] = "No logged progress for over 30 minutes; inspect before proceeding."
    elif owner and not queue:
        result["attention"] = "Owner recorded without a queue status; inspect launch logs."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "seal", "run", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.action](args.config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
