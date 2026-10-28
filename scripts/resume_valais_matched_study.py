"""Resume the user-paused B0-E epoch boundary and finish its sealed comparison.

Original model/trainer/evaluator files are kept intact. Prefix reconstruction is
metadata-only and must match saved exposure hashes and the saved loader RNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from pcdc.data import LEVIRDataset, audit_data, file_sha256
from pcdc.engine import move_model, read_checkpoint
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
import pcdc.robust_baselines as robust
from pcdc.runtime import setup_runtime, write_json
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config, read_temperature
from scripts.train_robust_guarded import guarded_train
from scripts.valais_matched_study import (
    ROOT, bound_check, check_training, compare_pair, evaluation_job, load_study,
    no_other_python, now, read_json, resolve, run_child, verify_score,
)


class SourceNames(Dataset):
    """Metadata only; it cannot open images, labels, or held-out data."""
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return self.names[index]


def update_order(digest, names):
    for name in names:
        raw = name.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)


def names_from_manifest(manifest):
    if manifest["split"] != "train":
        raise ValueError("Source-order reconstruction is train-only")
    entries, tile = manifest["entries"], manifest["tile_size"]
    names = [item["name"] for item in entries]
    if names != sorted(set(names)) or manifest["source_pairs"] != len(names):
        raise ValueError("Training manifest is not unique and sorted")
    ids = []
    for item in entries:
        width, height = item["size"]
        if width % tile or height % tile:
            raise ValueError("Manifest requires dropping image borders")
        ids.extend(f"train/{item['name']}@x{x}_y{y}" for y in range(0, height, tile)
                   for x in range(0, width, tile))
    if len(ids) != manifest["tiles"]:
        raise ValueError("Training manifest tile count differs")
    return ids


def replay_prefix(checkpoint, reference_order):
    """Rebuild shuffle only; actual hashes/RNG prove the completed prefix."""
    names = names_from_manifest(checkpoint["manifests"]["train"])
    settings = checkpoint["config"]
    generator = torch.Generator().manual_seed(settings["experiment"]["seed"])
    loader = DataLoader(SourceNames(names), batch_size=settings["train"]["batch_size"],
                        shuffle=True, num_workers=0, pin_memory=False, generator=generator)
    boundary, total = checkpoint["epoch"], settings["train"]["epochs"]
    history = checkpoint["history"]
    if [row["epoch"] for row in history] != list(range(1, boundary + 1)) or not 0 < boundary < total:
        raise ValueError("Checkpoint is not a complete, unfinished epoch boundary")
    order, prefix, epochs, next_names = hashlib.sha256(), None, [], None
    for epoch in range(1, total + 1):
        stream, epoch_order, samples, batches = hashlib.sha256(), hashlib.sha256(), 0, 0
        for batch in loader:
            if epoch == boundary + 1 and next_names is None:
                next_names = list(batch)
            robust._update_stream_hash(stream, list(batch), ["clean"] * len(batch))
            update_order(order, batch)
            update_order(epoch_order, batch)
            samples += len(batch)
            batches += 1
        if epoch <= boundary:
            saved = history[epoch - 1]["train"]
            if (saved["exposure_stream_sha256"] != stream.hexdigest()
                    or saved["source_samples"] != samples or saved["pair_views"] != samples * 2
                    or saved["optimizer_steps"] != batches or saved["amp_skipped_steps"] != 0
                    or saved["exposure_counts"] != {"clean|l0|clean|none|none": samples}):
                raise ValueError(f"Saved B0-E source/exposure order differs at epoch {epoch}")
        epochs.append({"epoch": epoch, "source_order_sha256": epoch_order.hexdigest(),
                       "clean_stream_sha256": stream.hexdigest(), "source_pairs": samples,
                       "batches": batches})
        if epoch == boundary:
            if not torch.equal(generator.get_state(), checkpoint["rng"]["loader"]):
                raise ValueError("Reconstructed prefix does not recover the saved loader RNG")
            prefix = order.copy()
    if (order.hexdigest() != reference_order["source_order_sha256"]
            or sum(row["source_pairs"] for row in epochs) != reference_order["source_pairs"]
            or sum(row["batches"] for row in epochs) != reference_order["batches"]):
        raise ValueError("Reconstructed full source order differs from completed B1")
    return prefix, {"status": "saved_prefix_reconstruction_verified", "saved_epoch": boundary,
                    "planned_epochs": total, "prefix_source_order_sha256": prefix.hexdigest(),
                    "saved_loader_rng_matches": True,
                    "expected_full_source_order_sha256": order.hexdigest(),
                    "first_resumed_batch_ids": next_names, "epochs": epochs,
                    "metadata_only": True, "raw_images_opened": False, "test_accessed": False,
                    "worker_note": "Replay uses zero workers and no pinning; saved exposure and epoch-boundary RNG must match the original four-worker run."}


class OrderRecorder:
    """Append real resumed batch IDs to the proven prefix; verify each epoch."""
    def __init__(self, prefix, plan, progress_path=None):
        self.digest = prefix.copy()
        self.plan = plan
        self.epoch = plan["saved_epoch"] + 1
        rows = plan["epochs"][:plan["saved_epoch"]]
        self.samples = sum(row["source_pairs"] for row in rows)
        self.batches = sum(row["batches"] for row in rows)
        self.epoch_samples = self.epoch_batches = 0
        self.epoch_digest = hashlib.sha256()
        self.progress_path = progress_path

    def record(self, names, instances):
        if self.epoch > self.plan["planned_epochs"] or len(names) != len(instances) or any(item != "clean" for item in instances):
            raise ValueError("Unexpected resumed B0-E batch or instance")
        row = self.plan["epochs"][self.epoch - 1]
        self.samples += len(names)
        self.batches += 1
        self.epoch_samples += len(names)
        self.epoch_batches += 1
        update_order(self.digest, names)
        update_order(self.epoch_digest, names)
        if self.epoch_samples > row["source_pairs"] or self.epoch_batches > row["batches"]:
            raise ValueError("Resumed epoch exceeds its frozen source/batch budget")
        if self.epoch_batches == row["batches"]:
            if (self.epoch_samples != row["source_pairs"]
                    or self.epoch_digest.hexdigest() != row["source_order_sha256"]):
                raise ValueError(f"Actual resumed source order differs at epoch {self.epoch}")
            if self.progress_path:
                _atomic_json(self.progress_path, {"status": "source_batches_verified_not_checkpoint_attestation",
                             "last_train_epoch_observed": self.epoch,
                             "source_order_sha256": self.digest.hexdigest(),
                             "source_pairs": self.samples, "batches": self.batches, "test_accessed": False})
            self.epoch += 1
            self.epoch_samples = self.epoch_batches = 0
            self.epoch_digest = hashlib.sha256()

    def finish(self):
        if (self.epoch != self.plan["planned_epochs"] + 1 or self.epoch_batches
                or self.digest.hexdigest() != self.plan["expected_full_source_order_sha256"]):
            raise ValueError("Resumed source order is incomplete or differs from B1")
        return {"source_order_sha256": self.digest.hexdigest(), "source_pairs": self.samples,
                "batches": self.batches}


def load_context(path):
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    expected = {"schema_version", "study_config", "pause_record", "pause_record_sha256",
                "output_dir", "protocol_document", "launcher", "task_name", "saved_epoch",
                "minimum_unittests"}
    if (set(value) != expected or value["schema_version"] != 1 or value["saved_epoch"] != 31
            or not re.fullmatch(r"[0-9a-f]{64}", value["pause_record_sha256"])):
        raise ValueError("Unexpected user-pause recovery configuration")
    paths = {key: resolve(source, value[key]) for key in
             ("study_config", "pause_record", "output_dir", "protocol_document", "launcher")}
    paths["output_dir"].relative_to(ROOT / "runs")
    _, study, study_paths, runs = load_study(paths["study_config"])
    if file_sha256(paths["pause_record"]) != value["pause_record_sha256"]:
        raise ValueError("Pause preservation record changed")
    pause = read_json(paths["pause_record"])
    if pause["status"] != "user_pause_safely_preserved_and_verified" or pause["b0e_saved_epoch"] != value["saved_epoch"]:
        raise ValueError("Not the reviewed B0-E pause boundary")
    auth = read_json(study_paths["output_dir"] / "authorization.json")
    bound_check(auth["bound_sha256"])
    return source, value, paths, study, study_paths, runs, pause


def paused_checkpoint(context, *, worker=False):
    _, value, _, _, study_paths, runs, pause = context
    bindings = dict(pause["critical_artifact_sha256"])
    if worker:
        bindings.pop(str(study_paths["output_dir"] / "status.json"))
    bound_check(bindings)
    cfg = runs["b0e"]
    status = read_json(cfg.output_dir / "status.json")
    checkpoint = read_checkpoint(cfg.output_dir / "last.pt")
    if (status["status"] != "paused_user_requested" or status["manual_resume_required"] is not True
            or checkpoint["epoch"] != value["saved_epoch"] or checkpoint["robust_mode"] != "b0e"
            or checkpoint["kind"] != "research" or checkpoint["training_flavor"] != "matched_robust_baseline_v1"
            or checkpoint["training_signature"] != cfg.fingerprint
            or checkpoint["source_sha256"] != robust.robust_source_hashes()
            or checkpoint["history"] != read_json(cfg.output_dir / "history.json")):
        raise ValueError("Paused checkpoint/config/source/history differs")
    for item in pause["checkpoint_backups"]:
        if file_sha256(Path(item["backup"])) != item["sha256"]:
            raise ValueError("Pause checkpoint backup changed")
    check_training(runs["b1"], read_json(runs["b1"].output_dir / "source_order.json"))
    verify_score(runs["b1"].output_dir.with_name(runs["b1"].output_dir.name + "_val"))
    return cfg, checkpoint


def preflight(path):
    context = load_context(path)
    _, _, paths, _, _, runs, _ = context
    cfg, checkpoint = paused_checkpoint(context)
    no_other_python()
    _, plan = replay_prefix(checkpoint, read_json(runs["b1"].output_dir / "source_order.json"))
    paths["output_dir"].mkdir(exist_ok=False)
    write_json(paths["output_dir"] / "prefix_replay.json", plan)
    write_json(paths["output_dir"] / "preflight.json", {"status": "paused_matched_resume_preflight_passed",
               "starting_checkpoint_sha256": file_sha256(cfg.output_dir / "last.pt"),
               "resume_epoch": checkpoint["epoch"] + 1, "test_accessed": False,
               "new_model_inference": False, "protocol_changed": False})
    return {"status": "paused_matched_resume_preflight_passed", "resume_epoch": checkpoint["epoch"] + 1}


def smoke(path):
    context = load_context(path)
    _, _, paths, _, study_paths, runs, _ = context
    cfg, checkpoint = paused_checkpoint(context)
    no_other_python()
    output = paths["output_dir"] / "smoke"
    output.mkdir(exist_ok=False)
    guard = EvaluationTemperatureGuard(load_guard_config(study_paths["guard_config"]), output)
    try:
        guard(force=True)
        device = setup_runtime(cfg.base)
        dataset = LEVIRDataset(cfg.base, "train")
        plan = read_json(paths["output_dir"] / "prefix_replay.json")
        names = [f"train/{dataset.records[i].name}@x{x}_y{y}" for i, x, y in dataset.tiles]
        indices = [names.index(name) for name in plan["first_resumed_batch_ids"]]
        paired = robust.TwoViewTrainingDataset(dataset, cfg.protocol, cfg.secondary)
        paired.set_epoch(checkpoint["epoch"] + 1)
        loader = DataLoader(Subset(paired, indices), batch_size=len(indices), shuffle=False, num_workers=0)
        model = move_model(build_model(cfg.base, initialize_encoder=False), device, cfg.base.values["runtime"]["memory_format"])
        model.load_state_dict(checkpoint["model"], strict=True)
        settings = cfg.base.values["train"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                     betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        scaler.load_state_dict(checkpoint["scaler"])
        # Loader is a fixed smoke subset. The real loader's checkpoint RNG is
        # independently verified by replay and restored by the original trainer.
        before = {name: int(module.num_batches_tracked) for name, module in model.named_modules()
                  if isinstance(module, torch.nn.BatchNorm2d)}
        result = robust.run_two_view_epoch(model, loader, ChangeDetectionLoss(**cfg.base.values["loss"]), device,
                     threshold=cfg.base.values["evaluation"]["threshold"], consistency_kind="none",
                     consistency_weight=0.0, amp=False, memory_format=cfg.base.values["runtime"]["memory_format"],
                     optimizer=optimizer, scaler=scaler, grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard)
        bn_ok = all(int(module.num_batches_tracked) == before[name] + 1 for name, module in model.named_modules() if name in before)
        if result["optimizer_steps"] != 1 or result["amp_skipped_steps"] != 0 or result["pair_views"] != 8 or not bn_ok:
            raise ValueError("Resume engineering smoke update/exposure/BN differs")
        for key in result["loss"]:
            if not math.isfinite(result["loss"][key]):
                raise ValueError("Non-finite resume engineering smoke")
        paused_checkpoint(context)
        guard.finish({"status": "resume_real_cuda_fp32_smoke_passed"})
        record = {"status": "resume_real_cuda_fp32_smoke_passed", "research_result": False,
                  "original_checkpoint_unchanged": True, "input_checkpoint_sha256": file_sha256(cfg.output_dir / "last.pt"),
                  "source_ids": plan["first_resumed_batch_ids"], "source_batch_size": 4, "pair_views": 8,
                  "optimizer_state_loaded": True, "scheduler_state_loaded": True, "one_bn_update": bn_ok,
                  "successful_updates": 1, "amp_skips": 0, "test_accessed": False}
        write_json(output / "report.json", record)
        return record
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        raise


def authorize(path):
    context = load_context(path)
    source, value, paths, _, study_paths, runs, pause = context
    cfg, checkpoint = paused_checkpoint(context)
    if Path(sys.executable).resolve() != study_paths["python"]:
        raise ValueError("Recovery interpreter differs")
    no_other_python()
    output = paths["output_dir"]
    if (output / "authorization.json").exists():
        raise FileExistsError("Recovery authorization already exists")
    log = (output / "unittest.log").read_text(encoding="utf-8-sig")
    counts = re.findall(r"Ran (\d+) tests in", log)
    if not counts or int(counts[-1]) < value["minimum_unittests"] or not log.rstrip().endswith("OK"):
        raise ValueError("Full unittest gate did not pass without skipped CUDA checks")
    sm = read_json(output / "smoke/report.json")
    if sm["status"] != "resume_real_cuda_fp32_smoke_passed" or sm["original_checkpoint_unchanged"] is not True:
        raise ValueError("Real CUDA resumed-state smoke did not pass")
    if ((output / "launcher_smoke.stdout.log").read_text(encoding="utf-8-sig").strip() != "launcher_transport_smoke_passed"
            or "UserWarning: launcher_stderr_transport_smoke" not in (output / "launcher_smoke.stderr.log").read_text(encoding="utf-8-sig")):
        raise ValueError("PowerShell stderr transport smoke did not pass")
    replay = read_json(output / "prefix_replay.json")
    _, rebuilt = replay_prefix(checkpoint, read_json(runs["b1"].output_dir / "source_order.json"))
    if replay != rebuilt:
        raise ValueError("Prepared replay changed")
    original = read_json(study_paths["output_dir"] / "authorization.json")
    bindings = dict(original["bound_sha256"])
    for name, expected in pause["critical_artifact_sha256"].items():
        candidate = Path(name)
        if candidate.is_relative_to(runs["b1"].output_dir) or candidate.is_relative_to(runs["b1"].output_dir.with_name(runs["b1"].output_dir.name + "_val")):
            bindings[name] = expected
    files = [source, Path(__file__), paths["protocol_document"], paths["launcher"], paths["pause_record"],
             ROOT / "tests/test_resume_valais_matched_study.py", output / "prefix_replay.json", output / "preflight.json",
             output / "unittest.log", output / "smoke/report.json", output / "smoke/temperature_guard_state.json",
             output / "launcher_smoke.stdout.log", output / "launcher_smoke.stderr.log"]
    files.extend(sorted((output / "prestart_launcher_repair_01").glob("*")))
    files.extend(Path(item["backup"]) for item in pause["checkpoint_backups"])
    bindings.update({str(p.resolve()): file_sha256(p) for p in files})
    temperature = read_temperature(load_guard_config(study_paths["guard_config"]).values)
    if temperature >= load_guard_config(study_paths["guard_config"]).values["confirmed_pause_c"]:
        raise ValueError("GPU too hot to resume")
    write_json(output / "authorization.json", {"status": "user_authorized_paused_matched_resume",
               "authorized_utc": now(), "human_instruction": "The user said 可以继续 after the safely preserved manual pause.",
               "starting_epoch": checkpoint["epoch"], "next_epoch": checkpoint["epoch"] + 1,
               "original_authorization_sha256": file_sha256(study_paths["output_dir"] / "authorization.json"),
               "pause_record_sha256": file_sha256(paths["pause_record"]), "bound_sha256": bindings,
               "launcher_transport_smoke_passed": True,
               "automatic_retry": False, "automatic_resume": False, "test_accessed": False})
    return {"status": "user_authorized_paused_matched_resume", "next_epoch": checkpoint["epoch"] + 1}


def verify_authorization(context):
    auth = read_json(context[2]["output_dir"] / "authorization.json")
    if auth["status"] != "user_authorized_paused_matched_resume":
        raise ValueError("Missing recovery authorization")
    bound_check(auth["bound_sha256"])
    return auth


def train_stage(path):
    context = load_context(path)
    _, _, paths, _, study_paths, runs, _ = context
    verify_authorization(context)
    cfg, checkpoint = paused_checkpoint(context, worker=True)
    prefix, plan = replay_prefix(checkpoint, read_json(runs["b1"].output_dir / "source_order.json"))
    if plan != read_json(paths["output_dir"] / "prefix_replay.json"):
        raise ValueError("Recovery prefix differs from authorization")
    tracker = OrderRecorder(prefix, plan, paths["output_dir"] / "source_order_progress.json")
    original_update = robust._update_stream_hash
    def tracked_update(digest, names, instances):
        original_update(digest, names, instances)
        tracker.record(names, instances)
    robust._update_stream_hash = tracked_update
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        result = guarded_train(cfg.source, study_paths["guard_config"], resume=True)
        order = {**tracker.finish(), "initial_tensor_sha256": result["initial_model_tensor_sha256"],
                 "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "wall_seconds": time.monotonic() - started,
                 "timing_and_peak_scope": "resumed_epochs_only_original_prefix_not_measured",
                 "instrumentation_source_sha256": file_sha256(Path(__file__)),
                 "saved_prefix_epochs": checkpoint["epoch"], "prefix_reconstruction_sha256": file_sha256(paths["output_dir"] / "prefix_replay.json"),
                 "test_accessed": False}
        check_training(cfg, order)
        first = read_json(runs["b1"].output_dir / "source_order.json")
        for key in ("source_order_sha256", "initial_tensor_sha256", "source_pairs", "batches"):
            if order[key] != first[key]:
                raise ValueError(f"Actual resumed matching failed: {key}")
        write_json(cfg.output_dir / "source_order.json", order)
        print(json.dumps({"status": "resumed_matched_training_complete", "mode": "b0e", "test_accessed": False}), flush=True)
    finally:
        robust._update_stream_hash = original_update


def run(path):
    context = load_context(path)
    source, _, paths, study, study_paths, runs, _ = context
    auth = verify_authorization(context)
    paused_checkpoint(context)
    if Path(sys.executable).resolve() != study_paths["python"]:
        raise ValueError("Recovery interpreter differs")
    no_other_python()
    output, original_output = paths["output_dir"], study_paths["output_dir"]
    write_json(output / "execution_started.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
               "started_utc": now(), "command": sys.argv, "authorization_sha256": file_sha256(output / "authorization.json"),
               "starting_epoch": 31, "test_accessed": False})
    logs = output / "logs"
    logs.mkdir(exist_ok=False)
    stage = "b0e_training_resume"
    def state(status, **extra):
        record = {"status": status, "stage": stage, "updated_utc": now(), "test_accessed": False,
                  "automatic_resume": False, "resume_session": str(output), **extra}
        _atomic_json(output / "status.json", record)
        _atomic_json(original_output / "status.json", record)
    try:
        state("running", resume_epoch=32)
        run_child([str(study_paths["python"]), "-u", "-m", "scripts.resume_valais_matched_study", "train-stage",
                   "--config", str(source)], logs, stage)
        cfg = runs["b0e"]
        check_training(cfg, read_json(cfg.output_dir / "source_order.json"))
        if read_json(cfg.output_dir / "data_audit.json") != read_json(original_output / "live_data_audit.json"):
            raise ValueError("Original training data audit changed")
        if audit_data(cfg.base, full=True) != read_json(original_output / "live_data_audit.json"):
            raise ValueError("Live train/val data changed")
        verify_authorization(context)
        stage = "b0e_validation"
        state("running")
        job, evaluation = evaluation_job(study_paths, cfg)
        command = [str(study_paths["python"]), "-u", "-m", "scripts.evaluate_candidate_guarded",
                   "--config", str(job), "--guard-config", str(study_paths["guard_config"])]
        run_child(command + ["--preflight-only"], logs, "b0e_evaluation_preflight")
        run_child(command, logs, stage)
        verify_score(evaluation)
        verify_authorization(context)
        comparison = compare_pair(study_paths, runs, study["gates"])
        stage = "matched_pair_scored"
        state("completed", decision=comparison["gate"]["decision"])
        completion = {"status": "matched_pair_completed", "test_accessed": False,
                      "status_sha256": file_sha256(original_output / "status.json"),
                      "authorization_sha256": auth["original_authorization_sha256"],
                      "resume_authorization_sha256": file_sha256(output / "authorization.json"),
                      "pause_record_sha256": file_sha256(paths["pause_record"]),
                      "comparison_sha256": file_sha256(original_output / "comparison.json"),
                      "evaluation_summaries": {mode: file_sha256(cfg.output_dir.with_name(cfg.output_dir.name + "_val") / "summary.json")
                                               for mode, cfg in runs.items()}}
        write_json(original_output / "completion.json", completion)
        write_json(output / "completion.json", completion)
    except BaseException as error:
        state("failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "smoke", "authorize", "run", "train-stage"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    action = train_stage if args.action == "train-stage" else globals()[args.action]
    result = action(args.config)
    if result:
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
