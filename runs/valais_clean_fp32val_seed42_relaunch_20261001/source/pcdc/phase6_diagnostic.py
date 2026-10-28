"""Exhaustive train-gate utility diagnostic for the two approved actions."""

from __future__ import annotations

from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .corruption_protocol import KINDS, apply_instance, canonical
from .data import LEVIRDataset, file_sha256, make_loader
from .engine import move_model, read_checkpoint
from .metrics import BinaryMetrics
from .model import build_model
from .perturbations import make_injection_mask
from .phase5_models import AdapterCorrectedDetector, Stage2ResidualAdapter, per_sample_change_loss
from .phase5_protocol import Phase56Config, load_frozen_manifests, verify_partition_files
from .phase5_training import SourceSubsetDataset
from .runtime import environment_record, setup_runtime, write_json


COUNT_KEYS = ("tn", "fp", "fn", "tp")


def _nested_path(config: Phase56Config, section: str, key: str) -> Path:
    value = config.values[section][key]
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else config.source.parent / candidate).resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_id(sample_id: str) -> str:
    return sample_id.split("@", 1)[0]


def _condition_order() -> list[str]:
    return [f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)]


def _metrics_from_counts(counts: Iterable[int]) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(value) for value in counts)
    total = tn + fp + fn + tp
    if total <= 0:
        raise ValueError("Cannot summarize empty action counts")

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {"precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "iou": divide(tp, tp + fp + fn), "oa": (tp + tn) / total,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "pixels": total}


def per_sample_confusion(logits: Tensor, target: Tensor, threshold: float) -> Tensor:
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("Confusion expects matching batched logits and labels")
    if not torch.isfinite(logits).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Confusion inputs must be finite logits and binary labels")
    predicted = logits.float().sigmoid() >= threshold
    truth = target.to(torch.bool)
    dimensions = tuple(range(1, logits.ndim))
    return torch.stack((
        ((~truth) & (~predicted)).sum(dim=dimensions, dtype=torch.int64),
        ((~truth) & predicted).sum(dim=dimensions, dtype=torch.int64),
        (truth & (~predicted)).sum(dim=dimensions, dtype=torch.int64),
        (truth & predicted).sum(dim=dimensions, dtype=torch.int64),
    ), dim=1).cpu()


def per_sample_transitions(a0_logits: Tensor, a1_logits: Tensor,
                           target: Tensor, threshold: float) -> Tensor:
    if a0_logits.shape != a1_logits.shape or a0_logits.shape != target.shape:
        raise ValueError("Transition tensors must have matching shapes")
    a0 = a0_logits.float().sigmoid() >= threshold
    a1 = a1_logits.float().sigmoid() >= threshold
    truth = target.to(torch.bool)
    dimensions = tuple(range(1, target.ndim))
    # FP->TN, FN->TP, TP->FN, TN->FP
    return torch.stack((
        ((~truth) & a0 & (~a1)).sum(dim=dimensions, dtype=torch.int64),
        (truth & (~a0) & a1).sum(dim=dimensions, dtype=torch.int64),
        (truth & a0 & (~a1)).sum(dim=dimensions, dtype=torch.int64),
        ((~truth) & (~a0) & a1).sum(dim=dimensions, dtype=torch.int64),
    ), dim=1).cpu()


class GateInstanceDataset(Dataset):
    """One fixed perturbed instance for every item in a single condition."""

    def __init__(self, base: SourceSubsetDataset, instances: list[dict[str, Any]]):
        if base.partition != "gate" or not instances:
            raise ValueError("Phase 6 instances require the gate source subset")
        self.base, self.instances, self.split = base, instances, "train"
        self.tile_size, self.mean, self.std = base.tile_size, base.mean, base.std
        lookup = {}
        for record_index, x, y in base.tiles:
            name = base.records[record_index].name
            lookup[f"train/{name}@x{x}_y{y}"] = (record_index, x, y)
        ids = [instance.get("sample_id") for instance in instances]
        if len(ids) != len(set(ids)) or set(ids) != set(lookup):
            raise ValueError("A gate condition must cover every gate tile exactly once")
        self.lookup = lookup

    def __len__(self) -> int:
        return len(self.instances)

    def _normalize(self, image: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        return (tensor - self.mean) / self.std

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self.instances[index]
        record_index, x, y = self.lookup[instance["sample_id"]]
        record = self.base.records[record_index]
        box = (x, y, x + self.tile_size, y + self.tile_size)
        clean = self.base.read_pair(record, box)
        perturbed = apply_instance(clean, instance)
        if not np.array_equal(clean.mask, perturbed.mask):
            raise ValueError("Gate corruption changed the semantic label")
        alpha = make_injection_mask(clean.mask.shape, instance["injection_mask"])
        foreground = clean.mask.astype(bool)
        overlap = int((foreground & (alpha > 0)).sum())
        category = ("no_foreground" if not foreground.any()
                    else "none" if overlap == 0 else "overlap")
        return {
            "image_a": self._normalize(perturbed.image_a),
            "image_b": self._normalize(perturbed.image_b),
            "mask": torch.from_numpy(np.ascontiguousarray(clean.mask)).float().unsqueeze(0),
            "sample_id": instance["sample_id"], "condition_id": instance["condition_id"],
            "kind": instance["kind"], "level": instance["level"],
            "direction": instance["direction"], "target_date": instance["target_date"],
            "spatial_mode": instance["spatial_mode"],
            # Decimal text avoids uint64 overflow in PyTorch's default collator.
            "seed_decimal": str(instance["seed"]),
            "instance_json": canonical(instance), "overlap_category": category,
            "injection_foreground_overlap_pixels": overlap,
            "foreground_pixels": int(foreground.sum()),
        }


def phase6_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/engine.py", "pcdc/losses.py", "pcdc/metrics.py", "pcdc/model.py",
        "pcdc/perturbations.py", "pcdc/phase5_models.py", "pcdc/phase5_protocol.py",
        "pcdc/phase5_training.py", "pcdc/phase6_diagnostic.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def _artifact_paths(config: Phase56Config) -> dict[str, Path]:
    actor_output = _nested_path(config, "actor_detector", "output_dir")
    adapter_output = _nested_path(config, "adapter", "output_dir")
    return {
        "actor_best": _nested_path(config, "adapter", "actor_best_checkpoint"),
        "actor_status": actor_output / "status.json",
        "actor_guard": _nested_path(config, "adapter", "actor_guard_completion"),
        "adapter_best": _nested_path(config, "diagnostic", "adapter_best_checkpoint"),
        "adapter_last": _nested_path(config, "diagnostic", "adapter_last_checkpoint"),
        "adapter_status": adapter_output / "status.json",
        "adapter_guard": _nested_path(config, "diagnostic", "adapter_guard_completion"),
    }


def _artifact_hashes(config: Phase56Config) -> dict[str, str]:
    paths = _artifact_paths(config)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Phase 6 requires completed actor and adapter artifacts")
    for prefix in ("actor", "adapter"):
        status = json.loads(paths[f"{prefix}_status"].read_text(encoding="utf-8"))
        guard = json.loads(paths[f"{prefix}_guard"].read_text(encoding="utf-8"))
        if (status.get("status") != "completed" or status.get("test_evaluated") is not False
                or guard.get("status") != "completed"):
            raise ValueError(f"{prefix} is not a completed guarded development run")
    return {key: _sha(path) for key, path in paths.items()}


def preflight(config: Phase56Config) -> dict[str, Any]:
    partition, gate = load_frozen_manifests(config)
    artifacts = _artifact_hashes(config)
    return {
        "status": "phase6_action_utility_preflight_passed",
        "kind": "metadata_only_no_inference", "phase56_fingerprint": config.fingerprint,
        "gate_sources": len(partition["gate_names"]), "gate_tiles": gate["tile_count"],
        "perturbed_instances": gate["instance_count"], "artifact_sha256": artifacts,
        "dataset_opened": False, "model_initialized": False,
        "inference_started": False, "sensor_queried": False, "test_evaluated": False,
    }


def _load_models(config: Phase56Config, artifact_hashes: Mapping[str, str],
                 device: torch.device) -> tuple[AdapterCorrectedDetector, dict[str, Any], dict[str, Any]]:
    paths = _artifact_paths(config)
    if {key: _sha(path) for key, path in paths.items()} != dict(artifact_hashes):
        raise ValueError("Phase 5 artifacts changed before checkpoint loading")
    actor = read_checkpoint(paths["actor_best"])
    adapter_best = read_checkpoint(paths["adapter_best"])
    adapter_last = read_checkpoint(paths["adapter_last"])
    adapter_status = json.loads(paths["adapter_status"].read_text(encoding="utf-8"))
    if (actor.get("training_flavor") != "phase5_actor_detector_v1"
            or adapter_best.get("training_flavor") != "phase5_stage2_adapter_v1"
            or adapter_last.get("training_flavor") != "phase5_stage2_adapter_v1"
            or actor.get("phase56_fingerprint") != config.fingerprint
            or adapter_best.get("phase56_fingerprint") != config.fingerprint
            or adapter_last.get("phase56_fingerprint") != config.fingerprint
            or adapter_best.get("epoch") != adapter_status.get("best_epoch")
            or adapter_last.get("epoch") != config.values["adapter"]["epochs"]):
        raise ValueError("Phase 5 checkpoints differ from completed status/protocol")
    detector = move_model(build_model(config.base, initialize_encoder=False), device,
                          config.base.values["runtime"]["memory_format"])
    detector.load_state_dict(actor["model"], strict=True)
    model = AdapterCorrectedDetector(detector, Stage2ResidualAdapter())
    model.adapter.load_state_dict(adapter_best["adapter"], strict=True)
    model = move_model(model, device, config.base.values["runtime"]["memory_format"])
    model.eval()
    if any(parameter.requires_grad for parameter in model.detector.parameters()):
        raise ValueError("Phase 6 detector must be frozen")
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(False)
    return model, actor, adapter_best


def _loss_kwargs(config: Phase56Config) -> dict[str, float]:
    return {key: float(config.base.values["loss"][key])
            for key in ("bce_weight", "dice_weight", "dice_smooth")}


def _transfer(batch: Mapping[str, Any], device: torch.device, memory_format: str):
    fmt = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
    a = batch["image_a"].to(device=device, non_blocking=True, memory_format=fmt)
    b = batch["image_b"].to(device=device, non_blocking=True, memory_format=fmt)
    target = batch["mask"].to(device=device, non_blocking=True)
    return a, b, target


def _clean_actions(model: AdapterCorrectedDetector, loader: DataLoader,
                   config: Phase56Config, device: torch.device,
                   guard_callback: Callable[[], None] | None) -> dict[str, dict[str, Any]]:
    clean: dict[str, dict[str, Any]] = {}
    threshold = config.base.values["evaluation"]["threshold"]
    train = config.base.values["train"]
    for batch in loader:
        if guard_callback:
            guard_callback()
        a, b, target = _transfer(batch, device, config.base.values["runtime"]["memory_format"])
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=train["amp"]):
            a0_logits, a1_logits = model.action_logits(a, b)
        a0_loss = per_sample_change_loss(a0_logits, target, **_loss_kwargs(config))
        a1_loss = per_sample_change_loss(a1_logits, target, **_loss_kwargs(config))
        a0_counts = per_sample_confusion(a0_logits, target, threshold)
        a1_counts = per_sample_confusion(a1_logits, target, threshold)
        transitions = per_sample_transitions(a0_logits, a1_logits, target, threshold)
        for index, sample_id in enumerate(batch["sample_id"]):
            if sample_id in clean:
                raise ValueError("Duplicate clean gate tile")
            clean[sample_id] = {
                "sample_id": sample_id, "source_id": _source_id(sample_id),
                "loss_a0": float(a0_loss["total"][index]),
                "loss_a1": float(a1_loss["total"][index]),
                "bce_a0": float(a0_loss["bce"][index]),
                "bce_a1": float(a1_loss["bce"][index]),
                "dice_a0": float(a0_loss["dice"][index]),
                "dice_a1": float(a1_loss["dice"][index]),
                "u0": float(a0_loss["total"][index] - a1_loss["total"][index]),
                "a0_counts": {key: int(a0_counts[index, i])
                              for i, key in enumerate(COUNT_KEYS)},
                "a1_counts": {key: int(a1_counts[index, i])
                              for i, key in enumerate(COUNT_KEYS)},
                "transitions_a0_to_a1": {
                    key: int(transitions[index, i]) for i, key in enumerate(
                        ("fp_to_tn", "fn_to_tp", "tp_to_fn", "tn_to_fp"))},
            }
    if len(clean) != config.values["diagnostic"]["gate_tiles"]:
        raise ValueError("Clean action cache does not cover every gate tile")
    return clean


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _Strata:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def update(self, key: str, utility: float, a0: np.ndarray,
               a1: np.ndarray, oracle: np.ndarray) -> None:
        row = self.rows.setdefault(key, {
            "instances": 0, "positive_utility": 0, "negative_utility": 0,
            "zero_utility": 0, "utility_sum": 0.0,
            "a0_counts": np.zeros(4, dtype=np.int64),
            "a1_counts": np.zeros(4, dtype=np.int64),
            "oracle_counts": np.zeros(4, dtype=np.int64),
        })
        row["instances"] += 1
        row["positive_utility"] += utility > 0
        row["negative_utility"] += utility < 0
        row["zero_utility"] += utility == 0
        row["utility_sum"] += utility
        row["a0_counts"] += a0
        row["a1_counts"] += a1
        row["oracle_counts"] += oracle

    def finish(self) -> dict[str, Any]:
        result = {}
        for key, row in sorted(self.rows.items()):
            count = row["instances"]
            result[key] = {
                "instances": count,
                "positive_utility": row["positive_utility"],
                "negative_utility": row["negative_utility"],
                "zero_utility": row["zero_utility"],
                "positive_fraction": row["positive_utility"] / count,
                "negative_fraction": row["negative_utility"] / count,
                "mean_uT": row["utility_sum"] / count,
                "a0": _metrics_from_counts(row["a0_counts"]),
                "a1": _metrics_from_counts(row["a1_counts"]),
                "oracle": _metrics_from_counts(row["oracle_counts"]),
            }
        return result


def _action_summary(counts: np.ndarray, conditions: list[str]) -> dict[str, Any]:
    names = ("a0_no_correction", "a1_shared_stage2_adapter", "label_informed_choice")
    result = {}
    for action, name in enumerate(names):
        condition_metrics = {}
        for index, condition in enumerate(conditions):
            pooled = counts[:, index, action, :].sum(axis=0)
            condition_metrics[condition] = _metrics_from_counts(pooled)
        overall = counts[:, :, action, :].sum(axis=(0, 1))
        result[name] = {
            "corruption_mf1": float(np.mean([
                condition_metrics[condition]["f1"] for condition in conditions])),
            "pooled": _metrics_from_counts(overall),
            "conditions": condition_metrics,
        }
    return result


def _bootstrap(counts: np.ndarray, config: Phase56Config,
               better_constant_index: int) -> dict[str, Any]:
    settings = config.values["diagnostic"]
    source_count = counts.shape[0]
    rng = np.random.Generator(np.random.PCG64(settings["bootstrap_seed"]))
    replicates = settings["bootstrap_replicates"]
    deltas = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        batch = min(500, replicates - cursor)
        weights = rng.multinomial(source_count, np.full(source_count, 1 / source_count),
                                  size=batch)
        pooled = np.einsum("bs,scak->bcak", weights, counts, optimize=True)
        denominator = 2 * pooled[..., 3] + pooled[..., 1] + pooled[..., 2]
        f1 = np.divide(2 * pooled[..., 3], denominator,
                       out=np.zeros_like(denominator, dtype=np.float64),
                       where=denominator != 0)
        mf1 = f1.mean(axis=1)
        deltas[cursor:cursor + batch] = mf1[:, 2] - mf1[:, better_constant_index]
        cursor += batch
    alpha = (1 - settings["confidence_level"]) / 2
    quantiles = np.quantile(deltas, [alpha, 0.5, 1 - alpha])
    return {
        "unit": "original_source_pair", "replicates": replicates,
        "seed": settings["bootstrap_seed"], "method": "percentile",
        "oracle_mf1_minus_better_constant": quantiles.tolist(),
    }


def _transition_summary(transitions: np.ndarray, a0_counts: np.ndarray) -> dict[str, Any]:
    totals = transitions.sum(axis=(0, 1))
    baseline = a0_counts.sum(axis=(0, 1))
    names = ("fp_to_tn", "fn_to_tp", "tp_to_fn", "tn_to_fp")
    denominator_indices = (1, 2, 3, 0)
    return {name: {
        "count": int(totals[index]),
        "denominator": int(baseline[denominator_indices[index]]),
        "rate": (float(totals[index] / baseline[denominator_indices[index]])
                 if baseline[denominator_indices[index]] else 0.0),
    } for index, name in enumerate(names)}


def _snapshot(config: Phase56Config, output: Path, artifact_hashes: Mapping[str, str],
              source_hashes: Mapping[str, str], device: torch.device) -> None:
    (output / "phase5_6_config.yaml").write_bytes(config.raw)
    for key, name in (("approval", "approval.yaml"),
                      ("frozen_protocol_document", "frozen_protocol_document.md"),
                      ("actor_gate_manifest", "actor_gate_manifest.json"),
                      ("gate_action_manifest", "gate_action_manifest.json")):
        shutil.copyfile(config.paths[key], output / name)
    write_json(output / "artifact_sha256.json", dict(artifact_hashes))
    write_json(output / "source_sha256.json", dict(source_hashes))
    write_json(output / "environment.json", environment_record(config.base, device))
    for name, path in phase6_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != source_hashes[name]:
            raise ValueError(f"Phase 6 source changed while snapshotting: {name}")


def evaluate(config: Phase56Config,
             guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    partition, gate_manifest = load_frozen_manifests(config)
    output = _nested_path(config, "diagnostic", "output_dir")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 6 diagnostic: {output}")
    artifact_hashes = _artifact_hashes(config)
    source_hashes = {name: file_sha256(path) for name, path in phase6_source_files().items()}
    device = setup_runtime(config.base)
    train_dataset = LEVIRDataset(config.base, "train")
    verify_partition_files(config, train_dataset, partition)
    gate_dataset = SourceSubsetDataset(train_dataset, partition["gate_names"], partition="gate")
    if len(gate_dataset) != config.values["diagnostic"]["gate_tiles"]:
        raise ValueError("Gate dataset differs from the frozen 89-source partition")
    model, actor_checkpoint, adapter_checkpoint = _load_models(config, artifact_hashes, device)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "status.json", {
            "status": "evaluating_clean_actions", "inference_started": True,
            "completed_conditions": 0, "test_evaluated": False})
        _snapshot(config, output, artifact_hashes, source_hashes, device)
        if guard_callback:
            guard_callback()
        clean = _clean_actions(model, make_loader(gate_dataset, config.base, training=False),
                               config, device, guard_callback)
        _write_jsonl(output / "clean_actions.jsonl",
                     (clean[sample_id] for sample_id in sorted(clean)))
        conditions = _condition_order()
        source_names = sorted(partition["gate_names"])
        source_index = {f"train/{name}": index for index, name in enumerate(source_names)}
        condition_index = {condition: index for index, condition in enumerate(conditions)}
        counts = np.zeros((len(source_names), len(conditions), 3, 4), dtype=np.int64)
        transitions = np.zeros((len(source_names), len(conditions), 4), dtype=np.int64)
        strata = {name: _Strata() for name in (
            "family", "level", "source", "target_date", "spatial_mode",
            "direction", "injection_overlap")}
        sign_by_family = {kind: {"positive": 0, "negative": 0, "zero": 0}
                          for kind in KINDS}
        utility_total = {"positive": 0, "negative": 0, "zero": 0, "sum": 0.0}
        instances_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for instance in gate_manifest["instances"]:
            instances_by_condition[instance["condition_id"]].append(instance)
        raw_path = output / "action_utility.jsonl"
        temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
        threshold = config.base.values["evaluation"]["threshold"]
        training = config.base.values["train"]
        started = time.monotonic()
        with temporary.open("x", encoding="utf-8") as raw:
            for completed, condition in enumerate(conditions, start=1):
                cell = GateInstanceDataset(gate_dataset, instances_by_condition[condition])
                loader = make_loader(cell, config.base, training=False)
                for batch in loader:
                    if guard_callback:
                        guard_callback()
                    image_a, image_b, target = _transfer(
                        batch, device, config.base.values["runtime"]["memory_format"])
                    with torch.no_grad(), torch.autocast(
                            device_type=device.type, enabled=training["amp"]):
                        a0_logits, a1_logits = model.action_logits(image_a, image_b)
                    a0_loss = per_sample_change_loss(a0_logits, target, **_loss_kwargs(config))
                    a1_loss = per_sample_change_loss(a1_logits, target, **_loss_kwargs(config))
                    a0_counts = per_sample_confusion(a0_logits, target, threshold).numpy()
                    a1_counts = per_sample_confusion(a1_logits, target, threshold).numpy()
                    batch_transitions = per_sample_transitions(
                        a0_logits, a1_logits, target, threshold).numpy()
                    for index, sample_id in enumerate(batch["sample_id"]):
                        source = _source_id(sample_id)
                        s, c = source_index[source], condition_index[condition]
                        u_t = float(a0_loss["total"][index] - a1_loss["total"][index])
                        u_0 = clean[sample_id]["u0"]
                        advantage = u_t - u_0
                        oracle_index = 1 if u_t > 0 else 0
                        oracle_counts = a1_counts[index] if oracle_index else a0_counts[index]
                        counts[s, c, 0] += a0_counts[index]
                        counts[s, c, 1] += a1_counts[index]
                        counts[s, c, 2] += oracle_counts
                        transitions[s, c] += batch_transitions[index]
                        sign = "positive" if u_t > 0 else "negative" if u_t < 0 else "zero"
                        utility_total[sign] += 1
                        utility_total["sum"] += u_t
                        kind = batch["kind"][index]
                        sign_by_family[kind][sign] += 1
                        groups = {
                            "family": kind, "level": f"l{int(batch['level'][index])}",
                            "source": source, "target_date": batch["target_date"][index],
                            "spatial_mode": batch["spatial_mode"][index],
                            "direction": f"{kind}/{batch['direction'][index]}",
                            "injection_overlap": batch["overlap_category"][index],
                        }
                        for group, key in groups.items():
                            strata[group].update(key, u_t, a0_counts[index], a1_counts[index],
                                                 oracle_counts)
                        record = {
                            "sample_id": sample_id, "source_id": source,
                            "condition_id": condition, "kind": kind,
                            "level": int(batch["level"][index]),
                            "direction": batch["direction"][index],
                            "target_date": batch["target_date"][index],
                            "spatial_mode": batch["spatial_mode"][index],
                            "seed": int(batch["seed_decimal"][index]),
                            "injection_overlap": batch["overlap_category"][index],
                            "injection_foreground_overlap_pixels": int(
                                batch["injection_foreground_overlap_pixels"][index]),
                            "foreground_pixels": int(batch["foreground_pixels"][index]),
                            "loss_a0_Tx": float(a0_loss["total"][index]),
                            "loss_a1_Tx": float(a1_loss["total"][index]),
                            "uT": u_t, "u0": u_0, "a": advantage,
                            "label_informed_action": "a1" if oracle_index else "a0",
                            "a0_counts": {key: int(a0_counts[index, i])
                                          for i, key in enumerate(COUNT_KEYS)},
                            "a1_counts": {key: int(a1_counts[index, i])
                                          for i, key in enumerate(COUNT_KEYS)},
                            "transitions_a0_to_a1": {
                                key: int(batch_transitions[index, i])
                                for i, key in enumerate((
                                    "fp_to_tn", "fn_to_tp", "tp_to_fn", "tn_to_fp"))},
                            "instance_json": batch["instance_json"][index],
                        }
                        raw.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                             separators=(",", ":")) + "\n")
                raw.flush()
                os.fsync(raw.fileno())
                write_json(output / "status.json", {
                    "status": "evaluating_perturbed_actions", "inference_started": True,
                    "completed_conditions": completed, "condition_count": len(conditions),
                    "test_evaluated": False,
                }, overwrite=True)
                print(f"phase6/{condition}: completed ({completed}/{len(conditions)})", flush=True)
        os.replace(temporary, raw_path)
        action_summary = _action_summary(counts, conditions)
        constant_names = ("a0_no_correction", "a1_shared_stage2_adapter")
        better_constant_index = int(
            action_summary[constant_names[1]]["corruption_mf1"]
            > action_summary[constant_names[0]]["corruption_mf1"])
        better_name = constant_names[better_constant_index]
        oracle_name = "label_informed_choice"
        oracle_margin = (action_summary[oracle_name]["corruption_mf1"]
                         - action_summary[better_name]["corruption_mf1"])
        bootstrap = _bootstrap(counts, config, better_constant_index)
        total = gate_manifest["instance_count"]
        positive_fraction = utility_total["positive"] / total
        negative_fraction = utility_total["negative"] / total
        positive_families = sorted(kind for kind, row in sign_by_family.items()
                                   if row["positive"] > 0)
        negative_families = sorted(kind for kind, row in sign_by_family.items()
                                   if row["negative"] > 0)
        diagnostic = config.values["diagnostic"]
        sign_pass = (
            positive_fraction >= diagnostic["min_positive_utility_fraction"]
            and negative_fraction >= diagnostic["min_negative_utility_fraction"]
            and len(positive_families) >= diagnostic["min_families_each_sign"]
            and len(negative_families) >= diagnostic["min_families_each_sign"])
        margin_pass = (
            oracle_margin >= diagnostic["min_oracle_mf1_margin"]
            and bootstrap["oracle_mf1_minus_better_constant"][0] > 0)
        transition_summary = _transition_summary(transitions, counts[:, :, 0, :])
        gate_decision = {
            "status": "completed", "sign_distribution_pass": sign_pass,
            "actionable_margin_pass": margin_pass,
            "decision": ("pass_to_separately_reviewed_phase7_protocol"
                         if sign_pass and margin_pass else "stop_mechanism_route"),
            "positive_utility_fraction": positive_fraction,
            "negative_utility_fraction": negative_fraction,
            "positive_families": positive_families,
            "negative_families": negative_families,
            "better_constant_action": better_name,
            "oracle_mf1_margin": oracle_margin,
            "oracle_margin_bootstrap": bootstrap["oracle_mf1_minus_better_constant"],
            "thresholds": {
                key: diagnostic[key] for key in (
                    "min_positive_utility_fraction", "min_negative_utility_fraction",
                    "min_families_each_sign", "min_oracle_mf1_margin",
                    "require_oracle_margin_ci_lower_positive")},
            "scope": "No learned gate, Phase 7-10, or held-out test is authorized here.",
        }
        summary = {
            "status": "completed", "kind": "research",
            "stage": "phase6_actual_action_utility_diagnostic", "split": "train_gate_only",
            "phase56_fingerprint": config.fingerprint,
            "actor_checkpoint_epoch": actor_checkpoint["epoch"],
            "adapter_checkpoint_epoch": adapter_checkpoint["epoch"],
            "gate_sources": len(source_names), "gate_tiles": gate_manifest["tile_count"],
            "condition_count": len(conditions), "perturbed_instances": total,
            "utility": {**utility_total, "positive_fraction": positive_fraction,
                        "negative_fraction": negative_fraction,
                        "mean_uT": utility_total["sum"] / total,
                        "by_family_sign": sign_by_family},
            "actions": action_summary, "transitions_a0_to_a1": transition_summary,
            "bootstrap": bootstrap, "gate_decision": gate_decision,
            "seconds": time.monotonic() - started,
            "artifact_sha256": artifact_hashes,
            "raw_action_utility_sha256": _sha(raw_path),
            "clean_actions_sha256": _sha(output / "clean_actions.jsonl"),
            "test_evaluated": False, "training_started": False,
            "guarded_completion_required": True,
        }
        write_json(output / "strata.json", {
            name: aggregate.finish() for name, aggregate in strata.items()})
        write_json(output / "gate_decision.json", gate_decision)
        if ({name: file_sha256(path) for name, path in phase6_source_files().items()}
                != source_hashes or _artifact_hashes(config) != artifact_hashes):
            raise ValueError("Phase 6 source or bound model artifacts changed during evaluation")
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        if output.exists():
            write_json(output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": f"{type(error).__name__}: {error}",
                "inference_started": True, "test_evaluated": False,
                "training_started": False,
            }, overwrite=True)
        raise
