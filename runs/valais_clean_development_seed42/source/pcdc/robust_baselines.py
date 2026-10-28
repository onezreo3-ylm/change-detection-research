"""Matched B0-E/B1/B2 train-only robustness baselines.

This module is deliberately separate from the archived Phase-1 CLI.  It keeps
the clean detector architecture and inference protocol unchanged, performs one
optimizer step per original source batch, and places two pair views in one
model forward so B0-E, B1 and B2 have identical BatchNorm update structure.
Held-out test access is forbidden.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from .config import Config, UniqueKeyLoader, protocol_signature
from .corruption_protocol import (
    CorruptionProtocol,
    apply_instance,
    canonical,
    load_protocol,
    sample_training_instance,
)
from .data import LEVIRDataset, audit_data, build_datasets, file_sha256, make_loader
from .engine import move_model, read_checkpoint, training_signature
from .losses import ChangeDetectionLoss
from .metrics import BinaryMetrics
from .model import build_model
from .runtime import environment_record, save_checkpoint, setup_runtime, write_json


_CONFIG_KEYS = {
    "schema_version", "name", "mode", "base_config", "protocol_config",
    "protocol_sha256", "severity_review", "output_dir", "expected_source_tiles",
    "expected_batches_per_epoch", "views", "corruption", "consistency", "selection",
}
_VIEW_KEYS = {
    "primary", "secondary", "views_per_source", "concat_order",
    "optimizer_steps_per_source_batch", "batchnorm_update",
}
_CORRUPTION_KEYS = {"stream", "view", "selection"}
_CONSISTENCY_KEYS = {"kind", "weight", "space", "gradient_flow"}
_MODES = ("b0e", "b1", "b2")
_BN_RULE = "single_forward_clean_then_secondary_then_temporal_concat"
_SELECTION_RULE = "uniform_family_level_direction_date_spatial_per_sample_epoch_view"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label}: missing={expected - actual}, unexpected={actual - expected}")


def _sha_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else source.parent / candidate).resolve()


def _lower_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class RobustConfig:
    values: Mapping[str, Any]
    source: Path
    raw: bytes
    base: Config
    protocol: CorruptionProtocol
    review: Mapping[str, Any]
    paths: Mapping[str, Path]

    @property
    def output_dir(self) -> Path:
        return self.paths["output_dir"]

    @property
    def mode(self) -> str:
        return self.values["mode"]

    @property
    def secondary(self) -> str:
        return self.values["views"]["secondary"]

    @property
    def fingerprint(self) -> str:
        settings = copy.deepcopy(dict(self.values))
        for key in ("base_config", "protocol_config", "severity_review", "output_dir"):
            settings.pop(key)
        return _digest({
            "settings": settings,
            "base_training_signature": training_signature(self.base),
            "protocol_fingerprint": self.protocol.fingerprint,
            "severity_review_sha256": _sha_text(self.paths["severity_review"]),
        })


def _load_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value, raw


def _validate_review(path: Path, protocol: CorruptionProtocol, kind: str) -> Mapping[str, Any]:
    review, _ = _load_yaml(path)
    required = {
        "schema_version", "status", "approved_for", "protocol_sha256",
        "manifest_sha256", "instances_sha256", "checkpoint_sha256",
        "visual_review", "reviewer", "notes",
    }
    _keys(review, required, "severity_review")
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise ValueError("Only severity review schema_version 1 is supported")
    if review["approved_for"] != "validation_only":
        raise ValueError("Severity review must retain its validation-only lifecycle boundary")
    if review["protocol_sha256"] != protocol.fingerprint:
        raise ValueError("Severity review is bound to a different corruption protocol")
    for key in ("protocol_sha256", "manifest_sha256", "instances_sha256", "checkpoint_sha256"):
        _lower_sha(review[key], f"severity_review.{key}")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("Severity review requires an identified reviewer")
    if not isinstance(review["notes"], str) or not review["notes"].strip():
        raise ValueError("Severity review requires nonempty notes")
    expected = (("approved", "passed") if kind == "research"
                else ("engineering_only", "not_applicable_synthetic"))
    if (review["status"], review["visual_review"]) != expected:
        raise ValueError(f"Expected severity review status {expected} for kind={kind}")
    return MappingProxyType(review)


def load_robust_config(path: str | Path) -> RobustConfig:
    source = Path(path).resolve()
    values, raw = _load_yaml(source)
    _keys(values, _CONFIG_KEYS, "robust config")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only robust baseline schema_version 1 is supported")
    for key in ("name", "mode", "selection"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    mode = values["mode"]
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    paths = {key: _path(source, values[key], key)
             for key in ("base_config", "protocol_config", "severity_review", "output_dir")}
    from .config import load_config
    base = load_config(paths["base_config"])
    protocol = load_protocol(paths["protocol_config"])
    if protocol.path("base_config") != paths["base_config"]:
        raise ValueError("Robust training must use the exact base config archived by the protocol")
    expected_protocol = _lower_sha(values["protocol_sha256"], "protocol_sha256")
    if protocol.fingerprint != expected_protocol:
        raise ValueError("Robust config is bound to a different protocol fingerprint")
    if base.run_test_after_training:
        raise ValueError("Robust development training must keep held-out test access disabled")
    if base.values["experiment"]["kind"] not in ("research", "smoke"):
        raise ValueError("Unsupported base experiment kind")
    review = _validate_review(paths["severity_review"], protocol,
                              base.values["experiment"]["kind"])
    if paths["output_dir"] in (base.output_dir, protocol.path("output_dir")):
        raise ValueError("Robust output must not reuse a base or protocol-preparation directory")
    try:
        paths["output_dir"].relative_to(base.data_root)
    except ValueError:
        pass
    else:
        raise ValueError("Robust output directory cannot be inside the dataset")
    for key in ("expected_source_tiles", "expected_batches_per_epoch"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be an explicit positive integer")
    _keys(values["views"], _VIEW_KEYS, "views")
    views = values["views"]
    expected_secondary = "clean" if mode == "b0e" else "perturbed"
    if (views["primary"] != "clean" or views["secondary"] != expected_secondary
            or views["views_per_source"] != 2
            or views["concat_order"] != ["clean", "secondary"]
            or views["optimizer_steps_per_source_batch"] != 1
            or views["batchnorm_update"] != _BN_RULE):
        raise ValueError("View batching differs from the frozen matched two-view protocol")
    _keys(values["corruption"], _CORRUPTION_KEYS, "corruption")
    corruption = values["corruption"]
    if (corruption["stream"] != "train" or corruption["view"] != 0
            or corruption["selection"] != _SELECTION_RULE):
        raise ValueError("Corruption sampling differs from the frozen train-only stream")
    _keys(values["consistency"], _CONSISTENCY_KEYS, "consistency")
    consistency = values["consistency"]
    if consistency["space"] != "probability" or consistency["gradient_flow"] != "symmetric":
        raise ValueError("Only symmetric probability-space consistency is supported")
    weight = consistency["weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("consistency.weight must be a finite number >= 0")
    expected_consistency = ("probability_mse", 1.0) if mode == "b2" else ("none", 0.0)
    if (consistency["kind"], float(weight)) != expected_consistency:
        raise ValueError(f"{mode} requires consistency {expected_consistency}")
    if values["selection"] != "clean_val_f1":
        raise ValueError("Robust checkpoints must be selected by clean validation F1")
    return RobustConfig(MappingProxyType(values), source, raw, base, protocol, review,
                        MappingProxyType(paths))


def robust_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/__init__.py", "pcdc/config.py", "pcdc/corruption_protocol.py",
        "pcdc/data.py", "pcdc/engine.py", "pcdc/losses.py", "pcdc/metrics.py",
        "pcdc/model.py", "pcdc/perturbations.py", "pcdc/robust_baselines.py",
        "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def robust_source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in robust_source_files().items()}


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = _canonical({"name": name, "dtype": str(tensor.dtype),
                             "shape": list(tensor.shape)}).encode("utf-8")
        data = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class TwoViewTrainingDataset(Dataset):
    """Read a tile once and return clean plus clean/perturbed secondary views."""

    def __init__(self, base: LEVIRDataset, protocol: CorruptionProtocol, secondary: str):
        if base.split != "train":
            raise ValueError("Two-view optimization is train-only")
        if secondary not in ("clean", "perturbed"):
            raise ValueError("secondary must be clean or perturbed")
        self.base, self.protocol, self.secondary = base, protocol, secondary
        self.split = "train"
        self.epoch = 1

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Training epoch must be an integer >= 1")
        self.epoch = epoch

    def _normalize(self, image: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        return (tensor - self.base.mean) / self.base.std

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, x, y = self.base.tiles[index]
        record = self.base.records[record_index]
        box = (x, y, x + self.base.tile_size, y + self.base.tile_size)
        clean = self.base.read_pair(record, box)
        sample_id = f"train/{record.name}@x{x}_y{y}"
        if self.secondary == "perturbed":
            instance = sample_training_instance(
                self.protocol, sample_id, epoch=self.epoch,
                view=0,
            )
            secondary = apply_instance(clean, instance)
            fields = {
                "secondary_kind": instance["kind"],
                "secondary_level": instance["level"],
                "secondary_direction": instance["direction"],
                "secondary_target_date": instance["target_date"],
                "secondary_spatial_mode": instance["spatial_mode"],
                "instance_json": canonical(instance),
            }
        else:
            secondary = copy.deepcopy(clean)
            fields = {
                "secondary_kind": "clean", "secondary_level": 0,
                "secondary_direction": "clean", "secondary_target_date": "none",
                "secondary_spatial_mode": "none", "instance_json": "clean",
            }
        if not np.array_equal(clean.mask, secondary.mask):
            raise ValueError("Two-view transform changed the semantic target")
        return {
            "clean_a": self._normalize(clean.image_a),
            "clean_b": self._normalize(clean.image_b),
            "secondary_a": self._normalize(secondary.image_a),
            "secondary_b": self._normalize(secondary.image_b),
            "mask": torch.from_numpy(np.ascontiguousarray(clean.mask)).float().unsqueeze(0),
            "sample_id": sample_id,
            **fields,
        }


def _update_stream_hash(digest: Any, sample_ids: list[str], instances: list[str]) -> None:
    for sample_id, instance in zip(sample_ids, instances):
        payload = (sample_id + "\0" + instance).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def run_two_view_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ChangeDetectionLoss,
    device: torch.device,
    *,
    threshold: float,
    consistency_kind: str,
    consistency_weight: float,
    amp: bool,
    memory_format: str,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    grad_clip_norm: float | None,
    guard_callback: Callable[[], None] | None = None,
    progress: str | None = None,
) -> dict[str, Any]:
    model.train(True)
    clean_metrics, secondary_metrics, combined_metrics = (
        BinaryMetrics(threshold), BinaryMetrics(threshold), BinaryMetrics(threshold))
    sums = {key: 0.0 for key in ("total", "supervised", "bce", "dice", "consistency")}
    exposures: Counter[str] = Counter()
    stream_digest = hashlib.sha256()
    samples = optimizer_steps = amp_skipped_steps = 0
    started = time.monotonic()
    transfer_format = (torch.channels_last if memory_format == "channels_last"
                       else torch.contiguous_format)
    for step, batch in enumerate(loader, start=1):
        if guard_callback is not None:
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
            logits = model(torch.cat((clean_a, secondary_a), dim=0),
                           torch.cat((clean_b, secondary_b), dim=0))["logits"]
            supervised = loss_fn(logits, torch.cat((target, target), dim=0))
        clean_logits, secondary_logits = logits[:count], logits[count:]
        if consistency_kind == "probability_mse":
            consistency = F.mse_loss(clean_logits.float().sigmoid(),
                                     secondary_logits.float().sigmoid())
        elif consistency_kind == "none":
            consistency = supervised["total"].new_zeros(())
        else:
            raise ValueError("Unsupported consistency loss")
        total = supervised["total"] + consistency_weight * consistency
        losses = {
            "total": total, "supervised": supervised["total"],
            "bce": supervised["bce"], "dice": supervised["dice"],
            "consistency": consistency,
        }
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise FloatingPointError("Non-finite robust baseline loss")
        scale_before = float(scaler.get_scale())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        total_norm = nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm or float("inf"), error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        if torch.isfinite(total_norm).item():
            optimizer_steps += 1
        elif amp:
            scale_after = float(scaler.get_scale())
            if not scale_after < scale_before:
                raise FloatingPointError("Non-finite AMP gradients were not handled by GradScaler")
            amp_skipped_steps += 1
            if progress:
                print(f"{progress} step={step}/{len(loader)} amp_overflow_skip=1 "
                      f"scale={scale_before:g}->{scale_after:g}", flush=True)
        samples += count
        for key, value in losses.items():
            sums[key] += value.detach().item() * count
        clean_metrics.update(clean_logits.detach(), target)
        secondary_metrics.update(secondary_logits.detach(), target)
        combined_metrics.update(logits.detach(), torch.cat((target, target), dim=0))
        kinds = list(batch["secondary_kind"])
        levels = batch["secondary_level"].tolist()
        directions = list(batch["secondary_direction"])
        dates = list(batch["secondary_target_date"])
        spatial = list(batch["secondary_spatial_mode"])
        for values in zip(kinds, levels, directions, dates, spatial):
            exposures["|".join((values[0], f"l{values[1]}", *values[2:]))] += 1
        _update_stream_hash(stream_digest, list(batch["sample_id"]), list(batch["instance_json"]))
        if progress and (step == 1 or step % max(1, len(loader) // 10) == 0 or step == len(loader)):
            print(f"{progress} step={step}/{len(loader)} loss={sums['total']/samples:.6f}",
                  flush=True)
    if samples == 0:
        raise ValueError("Empty two-view loader")
    return {
        "loss": {key: value / samples for key, value in sums.items()},
        "metrics": combined_metrics.compute(),
        "clean_metrics": clean_metrics.compute(),
        "secondary_metrics": secondary_metrics.compute(),
        "source_samples": samples,
        "pair_views": samples * 2,
        "optimizer_steps": optimizer_steps,
        "amp_skipped_steps": amp_skipped_steps,
        "exposure_counts": dict(sorted(exposures.items())),
        "exposure_stream_sha256": stream_digest.hexdigest(),
        "seconds": time.monotonic() - started,
    }


def _rng_state(loader: DataLoader) -> dict[str, Any]:
    numpy_rng = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], loader: DataLoader) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def _exposure_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    streams = []
    for epoch in history:
        counts.update(epoch["train"]["exposure_counts"])
        streams.append({"epoch": epoch["epoch"],
                        "sha256": epoch["train"]["exposure_stream_sha256"]})
    return {
        "epochs": len(history),
        "counts": dict(sorted(counts.items())),
        "per_epoch_stream_sha256": streams,
        "combined_stream_sha256": _digest(streams),
    }


def _snapshot(config: RobustConfig, output: Path, source_hash: dict[str, str],
              device: torch.device, audit: dict[str, Any]) -> None:
    write_json(output / "environment.json", environment_record(config.base, device))
    write_json(output / "data_audit.json", audit)
    write_json(output / "source_sha256.json", source_hash)
    (output / "robust_source.yaml").write_bytes(config.raw)
    resolved = copy.deepcopy(dict(config.values))
    for key, path in config.paths.items():
        resolved[key] = str(path)
    with (output / "robust_resolved.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False, allow_unicode=True)
    for key, name in (("base_config", "base_config.yaml"),
                      ("protocol_config", "protocol_config.yaml"),
                      ("severity_review", "severity_review.yaml")):
        shutil.copyfile(config.paths[key], output / name)
    for name, path in robust_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != source_hash[name]:
            raise ValueError(f"Source changed during robust snapshot: {name}")


def preflight(config: RobustConfig) -> dict[str, Any]:
    return {
        "status": "robust_preflight_passed",
        "kind": "engineering_preflight_not_training",
        "mode": config.mode,
        "base_kind": config.base.values["experiment"]["kind"],
        "protocol_sha256": config.protocol.fingerprint,
        "severity_review_status": config.review["status"],
        "robust_config_sha256": hashlib.sha256(config.raw).hexdigest(),
        "robust_training_signature": config.fingerprint,
        "test_access": False,
        "dataset_opened": False,
        "model_initialized": False,
        "training_started": False,
    }


def train_robust(config: RobustConfig, *, resume: bool = False,
                 guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    output = config.output_dir
    if not resume and output.exists():
        raise FileExistsError(f"Run already exists: {output}. Use --resume or a new output_dir.")
    checkpoint = read_checkpoint(output / "last.pt") if resume else None
    current_sources = robust_source_hashes()
    if checkpoint:
        if checkpoint.get("training_flavor") != "matched_robust_baseline_v1":
            raise ValueError("Not a matched robust baseline checkpoint")
        if checkpoint.get("training_signature") != config.fingerprint:
            raise ValueError("Resume must preserve the robust training protocol")
        if checkpoint.get("source_sha256") != current_sources:
            raise ValueError("Relevant source changed since checkpoint; resume refused")
        if checkpoint.get("protocol_signature") != protocol_signature(config.base.values):
            raise ValueError("Resume checkpoint changed the clean inference protocol")
    device = setup_runtime(config.base)
    audit = audit_data(config.base, full=True)
    audit_sha = _digest(audit)
    manifests = {split: {key: value for key, value in report.items() if key != "file_sha256"}
                 for split, report in audit["splits"].items()}
    content = {split: report["file_sha256"] for split, report in audit["splits"].items()}
    if set(manifests) != {"train", "val"}:
        raise ValueError("Robust development audit must contain train/val only")
    if checkpoint and checkpoint.get("audit_sha256") != audit_sha:
        raise ValueError("Dataset changed since checkpoint; resume refused")
    if not checkpoint:
        output.mkdir(parents=True, exist_ok=False)
        _snapshot(config, output, current_sources, device, audit)
    status_common = {
        "kind": config.base.values["experiment"]["kind"], "mode": config.mode,
        "selection": "clean_val_f1", "test_evaluated": False,
    }
    try:
        write_json(output / "status.json", {"status": "running", "training_started": True,
                   **status_common}, overwrite=True)
        if guard_callback is not None:
            guard_callback()
        datasets = build_datasets(config.base, ("train", "val"))
        train_dataset = TwoViewTrainingDataset(
            datasets["train"], config.protocol, config.secondary)
        if len(train_dataset) != config.values["expected_source_tiles"]:
            raise ValueError("Training tile count differs from the robust config")
        train_loader = make_loader(train_dataset, config.base, training=True)
        val_loader = make_loader(datasets["val"], config.base, training=False)
        if len(train_loader) != config.values["expected_batches_per_epoch"]:
            raise ValueError("Optimizer-batch count differs from the robust config")
        model = move_model(build_model(config.base, initialize_encoder=not resume), device,
                           config.base.values["runtime"]["memory_format"])
        initial_model_sha = (checkpoint["initial_model_tensor_sha256"] if checkpoint
                             else tensor_state_sha256(model.state_dict()))
        settings = config.base.values["train"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"],
            betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
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
                consistency_kind=config.values["consistency"]["kind"],
                consistency_weight=float(config.values["consistency"]["weight"]),
                amp=settings["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                optimizer=optimizer, scaler=scaler,
                grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard_callback,
                progress=f"{config.mode}/epoch={epoch}/train")
            if guard_callback is not None:
                guard_callback()
            from .engine import run_epoch
            validation = run_epoch(
                model, val_loader, loss_fn, device,
                threshold=config.base.values["evaluation"]["threshold"],
                amp=settings["amp"],
                memory_format=config.base.values["runtime"]["memory_format"],
                progress=f"{config.mode}/epoch={epoch}/val")
            if guard_callback is not None:
                guard_callback()
            improved = validation["metrics"]["f1"] > best_val_f1
            if improved:
                best_val_f1, best_epoch = validation["metrics"]["f1"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": learning_rate,
                            "train": training, "val": validation})
            payload = {
                "format_version": 1,
                "training_flavor": "matched_robust_baseline_v1",
                "robust_mode": config.mode,
                "kind": config.base.values["experiment"]["kind"],
                "epoch": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "config": config.base.values,
                "robust_config": dict(config.values),
                "protocol_signature": protocol_signature(config.base.values),
                "training_signature": config.fingerprint,
                "corruption_protocol_sha256": config.protocol.fingerprint,
                "severity_review_sha256": _sha_text(config.paths["severity_review"]),
                "initial_model_tensor_sha256": initial_model_sha,
                "audit_sha256": audit_sha, "manifests": manifests,
                "file_sha256": content, "source_sha256": current_sources,
                "rng": _rng_state(train_loader), "history": history,
            }
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "history.json", history, overwrite=True)
            write_json(output / "training_exposure_summary.json", _exposure_summary(history),
                       overwrite=True)
            write_json(output / "status.json", {
                "status": "running", "training_started": True,
                "epoch": epoch, "epochs": settings["epochs"],
                "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "initial_model_tensor_sha256": initial_model_sha,
                **status_common,
            }, overwrite=True)
            print(f"{config.mode}/epoch={epoch} val_f1={validation['metrics']['f1']:.6f} "
                  f"best={best_val_f1:.6f}", flush=True)
        result = {
            "status": "completed", "training_started": True,
            "epochs": settings["epochs"], "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "initial_model_tensor_sha256": initial_model_sha,
            "optimizer_steps": sum(item["train"]["optimizer_steps"] for item in history),
            "amp_skipped_steps": sum(item["train"]["amp_skipped_steps"] for item in history),
            "pair_views": sum(item["train"]["pair_views"] for item in history),
            **status_common,
        }
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        if output.exists():
            last = output / "last.pt"
            write_json(output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "training_started": True, "error": f"{type(error).__name__}: {error}",
                "resume_available": last.is_file(), **status_common,
            }, overwrite=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_robust_config(args.config)
    if args.preflight_only:
        if args.resume:
            raise ValueError("Preflight and resume are mutually exclusive")
        result = preflight(config)
    else:
        result = train_robust(config, resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
