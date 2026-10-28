"""Reviewed P1 sampling controls; reuse the unchanged two-view training math."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader

from pcdc.config import protocol_signature
from pcdc.data import audit_data, build_datasets, file_sha256, make_loader, seed_worker
from pcdc.engine import move_model, read_checkpoint, run_epoch
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import (
    TwoViewTrainingDataset, _digest, _exposure_summary, _restore_rng, _rng_state,
    _snapshot, load_robust_config, robust_source_hashes, run_two_view_epoch, tensor_state_sha256,
)
from pcdc.runtime import save_checkpoint, setup_runtime, write_json
from scripts.evaluate_guarded import _atomic_json, load_guard_config
from scripts.stratified_sampling import ObservedLoader, StratifiedBatchSampler, load_sampling
from scripts.train_robust_guarded import TrainingTemperatureGuard, _guard_output, _validate_target

ROOT = Path(__file__).resolve().parents[1]
FLAVOR = "stratified_two_view_baseline_v1"


def trainer_hashes():
    names = ("scripts/stratified_sampling.py", "scripts/train_stratified_robust.py")
    return {name: file_sha256(ROOT / name) for name in names}


def sampling_identity(proposal):
    proposal = Path(proposal).resolve()
    value, paths, positive = load_sampling(proposal)
    identity = {
        "proposal_sha256": file_sha256(proposal),
        "train_label_report_sha256": file_sha256(paths["train_label_report"]),
        "captured_data_audit_sha256": file_sha256(paths["captured_data_audit"]),
        "flavor": FLAVOR,
    }
    return value, paths, positive, identity


def signature(config, identity):
    return _digest({"parent_robust_training_signature": config.fingerprint, "sampling": identity})


def make_sampled_loader(base, config, proposal):
    value, paths, positive, identity = sampling_identity(proposal)
    if len(base.records) != len(base.tiles) or any(x or y for _, x, y in base.tiles):
        raise ValueError("P1 requires one unchanged 256 tile per source")
    paired = TwoViewTrainingDataset(base, config.protocol, config.secondary)
    sampler = StratifiedBatchSampler([record.name for record in base.records], positive, value["sampling"])
    generator = torch.Generator().manual_seed(config.base.values["experiment"]["seed"])
    loader = DataLoader(paired, batch_sampler=sampler,
                        num_workers=config.base.values["train"]["num_workers"],
                        pin_memory=config.base.values["runtime"]["device"].startswith("cuda"),
                        worker_init_fn=seed_worker, generator=generator)
    return paired, sampler, ObservedLoader(loader, sampler), identity


def validate_resume(checkpoint, config, identity, sources, new_sources, audit_sha, initial_sha):
    if (checkpoint.get("training_flavor") != FLAVOR
            or checkpoint.get("training_signature") != signature(config, identity)
            or checkpoint.get("sampling_identity") != identity
            or checkpoint.get("source_sha256") != sources
            or checkpoint.get("stratified_trainer_source_sha256") != new_sources
            or checkpoint.get("audit_sha256") != audit_sha
            or checkpoint.get("initial_model_tensor_sha256") != initial_sha
            or checkpoint.get("protocol_signature") != protocol_signature(config.base.values)):
        raise ValueError("Resume would change frozen sampling, source, data or initialization")
    history = checkpoint.get("history", [])
    if [row["epoch"] for row in history] != list(range(1, checkpoint["epoch"] + 1)):
        raise ValueError("Resume history has a missing or duplicate epoch")


def train(config_path, proposal_path, guard_path, initial_sha, *, resume=False):
    config = load_robust_config(config_path)
    if (config.base.run_test_after_training or config.base.values["train"]["amp"] is not False
            or config.base.values["train"]["epochs"] != 100 or config.base.values["train"]["batch_size"] != 4
            or config.mode not in ("b0e", "b1")):
        raise ValueError("P1 requires frozen FP32 C-S/A-S development settings")
    config.output_dir.relative_to(ROOT / "runs")
    value, paths, positive, identity = sampling_identity(proposal_path)
    output = config.output_dir
    if not resume and output.exists():
        raise FileExistsError(output)
    checkpoint = read_checkpoint(output / "last.pt") if resume else None
    guard_config = load_guard_config(guard_path)
    _validate_target(config, guard_config)
    guard = TrainingTemperatureGuard(guard_config, _guard_output(config, resume))
    sources, new_sources = robust_source_hashes(), trainer_hashes()
    device = setup_runtime(config.base)
    # A full development audit verifies labels and the complete frozen train/val files.
    audit = audit_data(config.base, full=True)
    captured = json.loads(paths["captured_data_audit"].read_text(encoding="utf-8"))
    if audit != captured or set(audit["splits"]) != {"train", "val"}:
        raise ValueError("Development data differ from the captured original audit")
    audit_sha = _digest(audit)
    if checkpoint:
        validate_resume(checkpoint, config, identity, sources, new_sources, audit_sha, initial_sha)
    else:
        output.mkdir(parents=True, exist_ok=False)
        _snapshot(config, output, sources, device, audit)
        shutil.copyfile(proposal_path, output / "sampling_proposal.yaml")
        write_json(output / "sampling_identity.json", identity)
        write_json(output / "stratified_trainer_source_sha256.json", new_sources)
        for name, expected in new_sources.items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if file_sha256(target) != expected:
                raise ValueError("New training source changed while snapshotting")
    common = {"kind": "research", "mode": config.mode, "study_role": "C-S" if config.mode == "b0e" else "A-S",
              "selection": "clean_val_f1", "test_evaluated": False, "sampling_identity": identity}
    started = time.monotonic()
    try:
        _atomic_json(output / "status.json", {"status": "running", "training_started": True, **common})
        guard(force=True)
        datasets = build_datasets(config.base, ("train", "val"))
        paired, sampler, loader, _ = make_sampled_loader(datasets["train"], config, proposal_path)
        val_loader = make_loader(datasets["val"], config.base, training=False)
        model = move_model(build_model(config.base, initialize_encoder=not resume), device,
                           config.base.values["runtime"]["memory_format"])
        actual_initial = checkpoint["initial_model_tensor_sha256"] if checkpoint else tensor_state_sha256(model.state_dict())
        if actual_initial != initial_sha:
            raise ValueError("Initial model tensors do not match the archived seed42 baseline")
        settings = config.base.values["train"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                     betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"],
                                                             eta_min=settings["min_learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        loss_fn = ChangeDetectionLoss(**config.base.values["loss"])
        history, best_f1, best_epoch, start = [], -1.0, 0, 1
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            best_f1, best_epoch, start = checkpoint["best_val_f1"], checkpoint["best_epoch"], checkpoint["epoch"] + 1
            _restore_rng(checkpoint["rng"], loader)
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start, settings["epochs"] + 1):
            paired.set_epoch(epoch)
            sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]
            training = run_two_view_epoch(
                model, loader, loss_fn, device, threshold=config.base.values["evaluation"]["threshold"],
                consistency_kind="none", consistency_weight=0.0, amp=False,
                memory_format=config.base.values["runtime"]["memory_format"], optimizer=optimizer, scaler=scaler,
                grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard,
                progress=f"{common['study_role']}/epoch={epoch}/train")
            if (loader.observed is None or training["source_samples"] != 11025
                    or training["optimizer_steps"] != 2757 or training["amp_skipped_steps"] != 0):
                raise ValueError("Actual sampling/update budget differs")
            training["sampling"] = loader.observed
            guard()
            validation = run_epoch(model, val_loader, loss_fn, device,
                                   threshold=config.base.values["evaluation"]["threshold"], amp=False,
                                   memory_format=config.base.values["runtime"]["memory_format"],
                                   progress=f"{common['study_role']}/epoch={epoch}/val")
            guard()
            improved = validation["metrics"]["f1"] > best_f1
            if improved:
                best_f1, best_epoch = validation["metrics"]["f1"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": lr, "train": training, "val": validation})
            payload = {
                "format_version": 1, "training_flavor": FLAVOR, "robust_mode": config.mode, "kind": "research",
                "epoch": epoch, "best_epoch": best_epoch, "best_val_f1": best_f1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config.base.values,
                "robust_config": dict(config.values), "protocol_signature": protocol_signature(config.base.values),
                "training_signature": signature(config, identity), "parent_robust_training_signature": config.fingerprint,
                "corruption_protocol_sha256": config.protocol.fingerprint,
                "severity_review_sha256": file_sha256(config.paths["severity_review"]),
                "initial_model_tensor_sha256": actual_initial, "audit_sha256": audit_sha,
                "manifests": {split: {k: v for k, v in report.items() if k != "file_sha256"}
                              for split, report in audit["splits"].items()},
                "file_sha256": {split: report["file_sha256"] for split, report in audit["splits"].items()},
                "source_sha256": sources, "stratified_trainer_source_sha256": new_sources,
                "sampling_identity": identity, "rng": _rng_state(loader), "history": history,
            }
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            _atomic_json(output / "history.json", history)
            _atomic_json(output / "training_exposure_summary.json", _exposure_summary(history))
            _atomic_json(output / "sampling_actual.json", [row["train"]["sampling"] for row in history])
            _atomic_json(output / "status.json", {
                "status": "running", "training_started": True, "epoch": epoch, "epochs": 100,
                "best_epoch": best_epoch, "best_val_f1": best_f1, "initial_model_tensor_sha256": actual_initial, **common,
            })
            print(f"{common['study_role']}/epoch={epoch} val_f1={validation['metrics']['f1']:.6f} best={best_f1:.6f}", flush=True)
        result = {
            "status": "completed", "training_started": True, "epochs": 100,
            "best_epoch": best_epoch, "best_val_f1": best_f1, "initial_model_tensor_sha256": actual_initial,
            "optimizer_steps": sum(row["train"]["optimizer_steps"] for row in history),
            "amp_skipped_steps": 0, "pair_views": sum(row["train"]["pair_views"] for row in history),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "session_wall_seconds": time.monotonic() - started, **common,
        }
        guard_record = guard.finish_training(result)
        guard_record["state_path"] = str((guard.output / "temperature_guard_state.json").relative_to(output))
        result.update(temperature_guard=guard_record, guarded_completion_required=True)
        _atomic_json(output / "status.json", result)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "training_status_sha256": file_sha256(output / "status.json"),
            "guard_state_path": guard_record["state_path"],
            "guard_state_sha256": file_sha256(guard.output / "temperature_guard_state.json"),
            "trainer_source_sha256": new_sources, "sampling_identity": identity,
        })
        return result
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        _atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": f"{type(error).__name__}: {error}", "resume_available": (output / "last.pt").is_file(),
            "training_started": True, **common,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sampling-proposal", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train(args.config, args.sampling_proposal, args.guard_config, args.initial_sha256, resume=args.resume)
    print(json.dumps({k: result[k] for k in ("status", "epochs", "best_val_f1")}), flush=True)


if __name__ == "__main__":
    main()
