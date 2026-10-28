"""Approved Phase 5 actor-detector and shared-adapter training.

Both stages are development-only, deterministic, epoch-resumable, and accept a
between-batch temperature callback.  The actor detector uses only the frozen
356-source actor partition.  Adapter optimization freezes that detector and
updates only the approved shared stage-2 residual adapter.
"""

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .config import protocol_signature
from .data import LEVIRDataset, audit_data, build_datasets, file_sha256, make_loader
from .engine import move_model, read_checkpoint, run_epoch
from .losses import ChangeDetectionLoss
from .metrics import BinaryMetrics
from .model import build_model
from .phase5_models import AdapterCorrectedDetector, Stage2ResidualAdapter
from .phase5_protocol import (
    Phase56Config,
    load_frozen_manifests,
    verify_partition_files,
)
from .robust_baselines import (
    TwoViewTrainingDataset,
    _exposure_summary,
    _restore_rng,
    _rng_state,
    _update_stream_hash,
    run_two_view_epoch,
    tensor_state_sha256,
)
from .runtime import environment_record, save_checkpoint, setup_runtime, write_json


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _nested_path(config: Phase56Config, section: str, key: str) -> Path:
    value = config.values[section][key]
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else config.source.parent / candidate).resolve()


class SourceSubsetDataset(Dataset):
    """A source-pair subset that preserves original record/tile identities."""

    def __init__(self, base: LEVIRDataset, names: list[str], *, partition: str):
        if base.split != "train" or partition not in ("actor", "gate"):
            raise ValueError("Source subsets are restricted to train actor/gate partitions")
        if len(names) != len(set(names)):
            raise ValueError("Source subset names must be unique")
        available = {record.name for record in base.records}
        if set(names) - available:
            raise ValueError("Source subset contains names outside the training split")
        self.base, self.partition = base, partition
        self.settings, self.split, self.root = base.settings, base.split, base.root
        self.tile_size, self.mean, self.std = base.tile_size, base.mean, base.std
        self.pair_transform = base.pair_transform
        self.records = base.records
        selected = set(names)
        self._base_tile_indices = [index for index, tile in enumerate(base.tiles)
                                   if base.records[tile[0]].name in selected]
        self.tiles = [base.tiles[index] for index in self._base_tile_indices]
        if {base.records[index].name for index, _, _ in self.tiles} != selected:
            raise ValueError("Source subset lost one or more complete source grids")

    def __len__(self) -> int:
        return len(self.tiles)

    def read_pair(self, *args, **kwargs):
        return self.base.read_pair(*args, **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self._base_tile_indices[index]]

    def manifest(self) -> dict[str, Any]:
        names = sorted({self.records[index].name for index, _, _ in self.tiles})
        payload = {"split": "train", "partition": self.partition, "names": names,
                   "source_pairs": len(names), "tiles": len(self.tiles)}
        return {**payload, "inventory_sha256": _digest(payload)}


def phase5_training_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/engine.py", "pcdc/losses.py", "pcdc/metrics.py", "pcdc/model.py",
        "pcdc/perturbations.py", "pcdc/phase5_models.py", "pcdc/phase5_protocol.py",
        "pcdc/phase5_training.py", "pcdc/robust_baselines.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def _source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in phase5_training_source_files().items()}


def _snapshot_common(config: Phase56Config, output: Path, *, stage: str,
                     device: torch.device, audit: dict[str, Any],
                     source_hashes: dict[str, str]) -> None:
    write_json(output / "environment.json", environment_record(config.base, device))
    write_json(output / "data_audit.json", audit)
    write_json(output / "source_sha256.json", source_hashes)
    (output / "phase5_6_config.yaml").write_bytes(config.raw)
    shutil.copyfile(config.paths["approval"], output / "approval.yaml")
    shutil.copyfile(config.paths["actor_gate_manifest"], output / "actor_gate_manifest.json")
    shutil.copyfile(config.paths["gate_action_manifest"], output / "gate_action_manifest.json")
    shutil.copyfile(config.paths["frozen_protocol_document"], output / "frozen_protocol_document.md")
    write_json(output / "run_binding.json", {
        "stage": stage, "phase56_fingerprint": config.fingerprint,
        "phase56_config_sha256": config.config_sha256,
        "approval_sha256": file_sha256(config.paths["approval"]),
        "actor_gate_manifest_sha256": file_sha256(config.paths["actor_gate_manifest"]),
        "gate_action_manifest_sha256": file_sha256(config.paths["gate_action_manifest"]),
        "test_evaluated": False,
    })
    for name, path in phase5_training_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != source_hashes[name]:
            raise ValueError(f"Training source changed while snapshotting: {name}")


def _audit_and_datasets(config: Phase56Config) -> tuple[
        dict[str, Any], dict[str, LEVIRDataset], SourceSubsetDataset, dict[str, Any]]:
    partition, _ = load_frozen_manifests(config)
    audit = audit_data(config.base, full=True)
    if set(audit["splits"]) != {"train", "val"}:
        raise ValueError("Phase 5 audit must contain train/val only")
    datasets = build_datasets(config.base, ("train", "val"))
    verify_partition_files(config, datasets["train"], partition)
    actor = SourceSubsetDataset(datasets["train"], partition["actor_names"], partition="actor")
    expected = config.values["actor_detector"]["expected_source_tiles"]
    if len(actor) != expected or actor.manifest()["source_pairs"] != 356:
        raise ValueError("Actor dataset differs from the frozen 356-source partition")
    return audit, datasets, actor, partition


def _checkpoint_common(config: Phase56Config, audit: dict[str, Any],
                       source_hashes: dict[str, str]) -> dict[str, Any]:
    manifests = {split: {key: value for key, value in report.items() if key != "file_sha256"}
                 for split, report in audit["splits"].items()}
    contents = {split: report["file_sha256"] for split, report in audit["splits"].items()}
    return {
        "kind": "research", "phase56_fingerprint": config.fingerprint,
        "phase56_config": copy.deepcopy(dict(config.values)),
        "phase56_config_sha256": config.config_sha256,
        "actor_gate_manifest_sha256": file_sha256(config.paths["actor_gate_manifest"]),
        "gate_action_manifest_sha256": file_sha256(config.paths["gate_action_manifest"]),
        "protocol_signature": protocol_signature(config.base.values),
        "corruption_protocol_sha256": config.corruption.fingerprint,
        "audit_sha256": _digest(audit), "manifests": manifests,
        "file_sha256": contents, "source_sha256": source_hashes,
        "test_evaluated": False,
    }


def actor_preflight(config: Phase56Config) -> dict[str, Any]:
    partition, gate = load_frozen_manifests(config)
    return {
        "status": "phase5_actor_preflight_passed",
        "kind": "metadata_only_no_training", "phase56_fingerprint": config.fingerprint,
        "actor_sources": len(partition["actor_names"]),
        "gate_sources": len(partition["gate_names"]),
        "reserved_gate_instances": gate["instance_count"],
        "expected_source_tiles": config.values["actor_detector"]["expected_source_tiles"],
        "dataset_opened": False, "model_initialized": False,
        "training_started": False, "sensor_queried": False, "test_evaluated": False,
    }


def train_actor_detector(config: Phase56Config, *, resume: bool = False,
                         guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    settings = config.values["actor_detector"]
    output = _nested_path(config, "actor_detector", "output_dir")
    if not resume and output.exists():
        raise FileExistsError(f"Actor run already exists: {output}")
    checkpoint = read_checkpoint(output / "last.pt") if resume else None
    sources = _source_hashes()
    if checkpoint:
        if (checkpoint.get("training_flavor") != "phase5_actor_detector_v1"
                or checkpoint.get("phase56_fingerprint") != config.fingerprint
                or checkpoint.get("source_sha256") != sources):
            raise ValueError("Actor resume checkpoint differs from the frozen implementation")
    device = setup_runtime(config.base)
    audit, datasets, actor, partition = _audit_and_datasets(config)
    audit_sha = _digest(audit)
    if checkpoint and checkpoint.get("audit_sha256") != audit_sha:
        raise ValueError("Dataset changed since the actor checkpoint")
    if not checkpoint:
        output.mkdir(parents=True, exist_ok=False)
        _snapshot_common(config, output, stage="actor_detector", device=device,
                         audit=audit, source_hashes=sources)
        write_json(output / "actor_dataset_manifest.json", actor.manifest())
    status_common = {
        "kind": "research", "stage": "phase5_actor_detector",
        "selection": "clean_val_f1", "actor_sources": 356, "gate_sources": 89,
        "test_evaluated": False,
    }
    try:
        write_json(output / "status.json", {"status": "running", "training_started": True,
                   **status_common}, overwrite=True)
        if guard_callback:
            guard_callback()
        train_dataset = TwoViewTrainingDataset(actor, config.corruption, "perturbed")
        train_loader = make_loader(train_dataset, config.base, training=True)
        val_loader = make_loader(datasets["val"], config.base, training=False)
        if len(train_loader) != settings["expected_batches_per_epoch"]:
            raise ValueError("Actor optimizer-batch count differs from protocol")
        model = move_model(build_model(config.base, initialize_encoder=not resume), device,
                           config.base.values["runtime"]["memory_format"])
        initial_sha = (checkpoint["initial_model_tensor_sha256"] if checkpoint
                       else tensor_state_sha256(model.state_dict()))
        if initial_sha != settings["expected_initial_model_tensor_sha256"]:
            raise ValueError("Actor fresh initialization differs from frozen B1 seed-42 state")
        train = config.base.values["train"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
                                      betas=tuple(train["adam_betas"]),
                                      weight_decay=train["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=train["min_learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=train["amp"])
        loss_fn = ChangeDetectionLoss(**config.base.values["loss"])
        history: list[dict[str, Any]] = []
        best_val_f1, best_epoch, start = -1.0, 0, 1
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            best_val_f1, best_epoch = checkpoint["best_val_f1"], checkpoint["best_epoch"]
            start = checkpoint["epoch"] + 1
            _restore_rng(checkpoint["rng"], train_loader)
        for epoch in range(start, settings["epochs"] + 1):
            train_dataset.set_epoch(epoch)
            learning_rate = optimizer.param_groups[0]["lr"]
            training = run_two_view_epoch(
                model, train_loader, loss_fn, device,
                threshold=config.base.values["evaluation"]["threshold"],
                consistency_kind="none", consistency_weight=0.0, amp=train["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                optimizer=optimizer, scaler=scaler, grad_clip_norm=train["grad_clip_norm"],
                guard_callback=guard_callback, progress=f"phase5-actor/epoch={epoch}/train")
            if guard_callback:
                guard_callback()
            validation = run_epoch(
                model, val_loader, loss_fn, device,
                threshold=config.base.values["evaluation"]["threshold"], amp=train["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                progress=f"phase5-actor/epoch={epoch}/val")
            if guard_callback:
                guard_callback()
            improved = validation["metrics"]["f1"] > best_val_f1
            if improved:
                best_val_f1, best_epoch = validation["metrics"]["f1"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": learning_rate,
                            "train": training, "val": validation})
            payload = {
                "format_version": 1, "training_flavor": "phase5_actor_detector_v1",
                "epoch": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "initial_model_tensor_sha256": initial_sha, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": _rng_state(train_loader),
                "history": history, "actor_dataset_manifest": actor.manifest(),
                **_checkpoint_common(config, audit, sources),
            }
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "history.json", history, overwrite=True)
            write_json(output / "training_exposure_summary.json", _exposure_summary(history),
                       overwrite=True)
            write_json(output / "status.json", {
                "status": "running", "training_started": True, "epoch": epoch,
                "epochs": settings["epochs"], "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "initial_model_tensor_sha256": initial_sha, **status_common,
            }, overwrite=True)
            print(f"phase5-actor/epoch={epoch} val_f1={validation['metrics']['f1']:.6f} "
                  f"best={best_val_f1:.6f}", flush=True)
        result = {
            "status": "completed", "training_started": True, "epochs": settings["epochs"],
            "best_epoch": best_epoch, "best_val_f1": best_val_f1,
            "initial_model_tensor_sha256": initial_sha,
            "optimizer_steps": sum(row["train"]["optimizer_steps"] for row in history),
            "amp_skipped_steps": sum(row["train"]["amp_skipped_steps"] for row in history),
            "attempted_optimizer_steps": sum(
                row["train"]["optimizer_steps"] + row["train"]["amp_skipped_steps"]
                for row in history),
            "pair_views": sum(row["train"]["pair_views"] for row in history),
            **status_common,
        }
        if (result["attempted_optimizer_steps"] != settings["expected_attempted_optimizer_steps"]
                or result["pair_views"] != settings["expected_pair_views"]):
            raise ValueError("Completed actor exposure/update totals differ from protocol")
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        if output.exists():
            write_json(output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "training_started": True, "error": f"{type(error).__name__}: {error}",
                "resume_available": (output / "last.pt").is_file(), **status_common,
            }, overwrite=True)
        raise


def _validate_actor_completion(config: Phase56Config) -> tuple[dict[str, Any], dict[str, str]]:
    actor_output = _nested_path(config, "actor_detector", "output_dir")
    paths = {
        "best": _nested_path(config, "adapter", "actor_best_checkpoint"),
        "last": _nested_path(config, "adapter", "actor_last_checkpoint"),
        "guard": _nested_path(config, "adapter", "actor_guard_completion"),
        "status": actor_output / "status.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Adapter training requires completed guarded actor artifacts")
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    guard = json.loads(paths["guard"].read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or status.get("test_evaluated") is not False
            or guard.get("status") != "completed"):
        raise ValueError("Actor detector is not a completed guarded development run")
    best, last = read_checkpoint(paths["best"]), read_checkpoint(paths["last"])
    if (best.get("training_flavor") != "phase5_actor_detector_v1"
            or last.get("training_flavor") != "phase5_actor_detector_v1"
            or best.get("phase56_fingerprint") != config.fingerprint
            or last.get("phase56_fingerprint") != config.fingerprint
            or last.get("epoch") != config.values["actor_detector"]["epochs"]
            or best.get("epoch") != status.get("best_epoch")
            or best.get("best_val_f1") != status.get("best_val_f1")):
        raise ValueError("Actor checkpoints do not match the completed frozen run")
    return best, {key: file_sha256(path) for key, path in paths.items()}


def run_adapter_epoch(model: AdapterCorrectedDetector, loader: DataLoader,
                      loss_fn: ChangeDetectionLoss, device: torch.device, *, threshold: float,
                      amp: bool, memory_format: str, optimizer: torch.optim.Optimizer,
                      scaler: Any, grad_clip_norm: float | None,
                      guard_callback: Callable[[], None] | None = None,
                      progress: str | None = None) -> dict[str, Any]:
    model.train(True)
    clean_metrics, secondary_metrics, combined_metrics = (
        BinaryMetrics(threshold), BinaryMetrics(threshold), BinaryMetrics(threshold))
    sums = {key: 0.0 for key in ("total", "bce", "dice")}
    exposures: Counter[str] = Counter()
    stream_digest = hashlib.sha256()
    samples = optimizer_steps = amp_skipped_steps = 0
    started = time.monotonic()
    transfer_format = (torch.channels_last if memory_format == "channels_last"
                       else torch.contiguous_format)
    parameters = list(model.adapter.parameters())
    if not parameters or any(parameter.requires_grad is False for parameter in parameters):
        raise ValueError("Adapter optimizer must contain only trainable adapter parameters")
    if any(parameter.requires_grad for parameter in model.detector.parameters()):
        raise ValueError("Detector parameters must remain frozen during adapter training")
    for step, batch in enumerate(loader, start=1):
        if guard_callback:
            guard_callback()
        clean_a = batch["clean_a"].to(device=device, non_blocking=True,
                                      memory_format=transfer_format)
        clean_b = batch["clean_b"].to(device=device, non_blocking=True,
                                      memory_format=transfer_format)
        secondary_a = batch["secondary_a"].to(device=device, non_blocking=True,
                                              memory_format=transfer_format)
        secondary_b = batch["secondary_b"].to(device=device, non_blocking=True,
                                              memory_format=transfer_format)
        target = batch["mask"].to(device, non_blocking=True)
        count = target.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(torch.cat((clean_a, secondary_a)),
                           torch.cat((clean_b, secondary_b)), apply_adapter=True)["logits"]
            losses = loss_fn(logits, torch.cat((target, target)))
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise FloatingPointError("Non-finite Phase 5 adapter loss")
        scale_before = float(scaler.get_scale())
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        total_norm = nn.utils.clip_grad_norm_(parameters, grad_clip_norm or float("inf"),
                                             error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        if torch.isfinite(total_norm).item():
            optimizer_steps += 1
        elif amp:
            scale_after = float(scaler.get_scale())
            if not scale_after < scale_before:
                raise FloatingPointError("Non-finite adapter gradients were not handled")
            amp_skipped_steps += 1
        clean_logits, secondary_logits = logits[:count], logits[count:]
        samples += count
        for key, value in losses.items():
            sums[key] += value.detach().item() * count
        clean_metrics.update(clean_logits.detach(), target)
        secondary_metrics.update(secondary_logits.detach(), target)
        combined_metrics.update(logits.detach(), torch.cat((target, target)))
        for values in zip(list(batch["secondary_kind"]), batch["secondary_level"].tolist(),
                          list(batch["secondary_direction"]),
                          list(batch["secondary_target_date"]),
                          list(batch["secondary_spatial_mode"])):
            exposures["|".join((values[0], f"l{values[1]}", *values[2:]))] += 1
        _update_stream_hash(stream_digest, list(batch["sample_id"]),
                            list(batch["instance_json"]))
        if progress and (step == 1 or step % max(1, len(loader) // 10) == 0
                         or step == len(loader)):
            print(f"{progress} step={step}/{len(loader)} loss={sums['total']/samples:.6f}",
                  flush=True)
    if samples == 0:
        raise ValueError("Empty adapter training loader")
    return {
        "loss": {key: value / samples for key, value in sums.items()},
        "metrics": combined_metrics.compute(), "clean_metrics": clean_metrics.compute(),
        "secondary_metrics": secondary_metrics.compute(), "source_samples": samples,
        "pair_views": samples * 2, "optimizer_steps": optimizer_steps,
        "amp_skipped_steps": amp_skipped_steps,
        "exposure_counts": dict(sorted(exposures.items())),
        "exposure_stream_sha256": stream_digest.hexdigest(),
        "seconds": time.monotonic() - started,
    }


def adapter_preflight(config: Phase56Config) -> dict[str, Any]:
    load_frozen_manifests(config)
    best, bindings = _validate_actor_completion(config)
    return {
        "status": "phase5_adapter_preflight_passed", "kind": "metadata_only_no_training",
        "phase56_fingerprint": config.fingerprint,
        "actor_best_epoch": best["epoch"], "actor_artifact_sha256": bindings,
        "dataset_opened": False, "model_initialized": False,
        "training_started": False, "sensor_queried": False, "test_evaluated": False,
    }


def train_adapter(config: Phase56Config, *, resume: bool = False,
                  guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    settings = config.values["adapter"]
    output = _nested_path(config, "adapter", "output_dir")
    if not resume and output.exists():
        raise FileExistsError(f"Adapter run already exists: {output}")
    actor_best, actor_bindings = _validate_actor_completion(config)
    checkpoint = read_checkpoint(output / "last.pt") if resume else None
    sources = _source_hashes()
    if checkpoint:
        if (checkpoint.get("training_flavor") != "phase5_stage2_adapter_v1"
                or checkpoint.get("phase56_fingerprint") != config.fingerprint
                or checkpoint.get("source_sha256") != sources
                or checkpoint.get("actor_artifact_sha256") != actor_bindings):
            raise ValueError("Adapter resume checkpoint differs from frozen inputs")
    device = setup_runtime(config.base)
    audit, datasets, actor, partition = _audit_and_datasets(config)
    audit_sha = _digest(audit)
    if checkpoint and checkpoint.get("audit_sha256") != audit_sha:
        raise ValueError("Dataset changed since the adapter checkpoint")
    detector = move_model(build_model(config.base, initialize_encoder=False), device,
                          config.base.values["runtime"]["memory_format"])
    detector.load_state_dict(actor_best["model"], strict=True)
    model = AdapterCorrectedDetector(detector, Stage2ResidualAdapter())
    model = move_model(model, device, config.base.values["runtime"]["memory_format"])
    initial_adapter_sha = (checkpoint["initial_adapter_tensor_sha256"] if checkpoint
                           else tensor_state_sha256(model.adapter.state_dict()))
    if not torch.count_nonzero(model.adapter.conv3.weight).item() == 0:
        raise ValueError("Adapter final weight must start at exact zero")
    if not torch.count_nonzero(model.adapter.conv3.bias).item() == 0:
        raise ValueError("Adapter final bias must start at exact zero")
    if not checkpoint:
        output.mkdir(parents=True, exist_ok=False)
        _snapshot_common(config, output, stage="stage2_adapter", device=device,
                         audit=audit, source_hashes=sources)
        write_json(output / "actor_artifact_sha256.json", actor_bindings)
        write_json(output / "actor_dataset_manifest.json", actor.manifest())
    status_common = {
        "kind": "research", "stage": "phase5_stage2_adapter",
        "selection": "clean_corrected_val_f1", "detector_frozen": True,
        "actor_sources": 356, "gate_sources": 89, "test_evaluated": False,
    }
    try:
        write_json(output / "status.json", {"status": "running", "training_started": True,
                   **status_common}, overwrite=True)
        if guard_callback:
            guard_callback()
        train_dataset = TwoViewTrainingDataset(actor, config.corruption, "perturbed")
        train_loader = make_loader(train_dataset, config.base, training=True)
        val_loader = make_loader(datasets["val"], config.base, training=False)
        if len(train_loader) != settings["expected_batches_per_epoch"]:
            raise ValueError("Adapter optimizer-batch count differs from protocol")
        train = config.base.values["train"]
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=train["learning_rate"],
                                      betas=tuple(train["adam_betas"]),
                                      weight_decay=train["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=train["min_learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=train["amp"])
        loss_fn = ChangeDetectionLoss(**config.base.values["loss"])
        history: list[dict[str, Any]] = []
        best_val_f1, best_epoch, start = -1.0, 0, 1
        if checkpoint:
            model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            best_val_f1, best_epoch = checkpoint["best_val_f1"], checkpoint["best_epoch"]
            start = checkpoint["epoch"] + 1
            _restore_rng(checkpoint["rng"], train_loader)
        for epoch in range(start, settings["epochs"] + 1):
            train_dataset.set_epoch(epoch)
            learning_rate = optimizer.param_groups[0]["lr"]
            training = run_adapter_epoch(
                model, train_loader, loss_fn, device,
                threshold=config.base.values["evaluation"]["threshold"], amp=train["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                optimizer=optimizer, scaler=scaler, grad_clip_norm=train["grad_clip_norm"],
                guard_callback=guard_callback, progress=f"phase5-adapter/epoch={epoch}/train")
            if guard_callback:
                guard_callback()
            validation = run_epoch(
                model, val_loader, loss_fn, device,
                threshold=config.base.values["evaluation"]["threshold"], amp=train["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                progress=f"phase5-adapter/epoch={epoch}/val")
            if guard_callback:
                guard_callback()
            improved = validation["metrics"]["f1"] > best_val_f1
            if improved:
                best_val_f1, best_epoch = validation["metrics"]["f1"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": learning_rate,
                            "train": training, "val": validation})
            payload = {
                "format_version": 1, "training_flavor": "phase5_stage2_adapter_v1",
                "epoch": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "initial_adapter_tensor_sha256": initial_adapter_sha,
                "adapter": model.adapter.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": _rng_state(train_loader), "history": history,
                "actor_artifact_sha256": actor_bindings,
                "actor_checkpoint_epoch": actor_best["epoch"],
                "actor_dataset_manifest": actor.manifest(),
                **_checkpoint_common(config, audit, sources),
            }
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "history.json", history, overwrite=True)
            write_json(output / "training_exposure_summary.json", _exposure_summary(history),
                       overwrite=True)
            write_json(output / "status.json", {
                "status": "running", "training_started": True, "epoch": epoch,
                "epochs": settings["epochs"], "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "initial_adapter_tensor_sha256": initial_adapter_sha, **status_common,
            }, overwrite=True)
            print(f"phase5-adapter/epoch={epoch} val_f1={validation['metrics']['f1']:.6f} "
                  f"best={best_val_f1:.6f}", flush=True)
        result = {
            "status": "completed", "training_started": True, "epochs": settings["epochs"],
            "best_epoch": best_epoch, "best_val_f1": best_val_f1,
            "initial_adapter_tensor_sha256": initial_adapter_sha,
            "optimizer_steps": sum(row["train"]["optimizer_steps"] for row in history),
            "amp_skipped_steps": sum(row["train"]["amp_skipped_steps"] for row in history),
            "attempted_optimizer_steps": sum(
                row["train"]["optimizer_steps"] + row["train"]["amp_skipped_steps"]
                for row in history),
            "pair_views": sum(row["train"]["pair_views"] for row in history),
            **status_common,
        }
        if (result["attempted_optimizer_steps"] != settings["expected_attempted_optimizer_steps"]
                or result["pair_views"] != settings["expected_pair_views"]):
            raise ValueError("Completed adapter exposure/update totals differ from protocol")
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        if output.exists():
            write_json(output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "training_started": True, "error": f"{type(error).__name__}: {error}",
                "resume_available": (output / "last.pt").is_file(), **status_common,
            }, overwrite=True)
        raise
