"""Guarded, resumable train-gate descriptor cache for Phase 7."""

from __future__ import annotations

from collections import defaultdict
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

from .data import LEVIRDataset, file_sha256, make_loader
from .phase5_protocol import load_frozen_manifests, verify_partition_files
from .phase5_training import SourceSubsetDataset
from .phase6_diagnostic import (
    GateInstanceDataset,
    _artifact_hashes as phase6_artifact_hashes,
    _condition_order,
    _load_models as load_phase6_models,
)
from .phase7_models import phase7_descriptor
from .phase7_protocol import Phase7Config, load_selector_split
from .runtime import environment_record, setup_runtime, write_json


def _sha(path: Path) -> str:
    return file_sha256(path)


def _source_id(sample_id: str) -> str:
    return sample_id.split("@", 1)[0]


def _source_name(sample_id: str) -> str:
    return _source_id(sample_id).split("/", 1)[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _discard_uncommitted(paths: Iterable[Path], output: Path) -> None:
    """Remove only files for a shard absent from the durable resume status."""
    root = output.resolve()
    for path in paths:
        candidate = path.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Refusing to clean a Phase 7 cache path outside its output")
        for target in (candidate, candidate.with_name(candidate.name + ".tmp")):
            if target.is_file():
                target.unlink()


def _load_npy(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid Phase 7 descriptor shard: {path}")
    return value


def _transfer(batch: Mapping[str, Any], device: torch.device,
              memory_format: str) -> tuple[Tensor, Tensor]:
    fmt = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
    image_a = batch["image_a"].to(device=device, non_blocking=True, memory_format=fmt)
    image_b = batch["image_b"].to(device=device, non_blocking=True, memory_format=fmt)
    return image_a, image_b


def phase7_cache_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/engine.py", "pcdc/model.py", "pcdc/perturbations.py",
        "pcdc/phase5_models.py", "pcdc/phase5_protocol.py",
        "pcdc/phase5_training.py", "pcdc/phase6_diagnostic.py",
        "pcdc/phase7_protocol.py", "pcdc/phase7_models.py",
        "pcdc/phase7_cache.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def cache_preflight(config: Phase7Config) -> dict[str, Any]:
    split = load_selector_split(config)
    phase56_partition, gate = load_frozen_manifests(config.phase56)
    artifacts = phase6_artifact_hashes(config.phase56)
    if (set(split["fit_names"]) | set(split["calibration_names"])
            != set(phase56_partition["gate_names"])):
        raise ValueError("Phase 7 split no longer covers the frozen 89 gate sources")
    return {
        "status": "phase7_descriptor_cache_preflight_passed",
        "kind": "metadata_only_no_data_model_or_device",
        "phase7_fingerprint": config.fingerprint,
        "fit_sources": len(split["fit_names"]),
        "calibration_sources": len(split["calibration_names"]),
        "gate_tiles": gate["tile_count"],
        "perturbed_instances": gate["instance_count"],
        "artifact_sha256": artifacts,
        "dataset_opened": False, "model_initialized": False,
        "inference_started": False, "validation_evaluated": False,
        "sensor_queried": False, "test_evaluated": False,
    }


def _snapshot(config: Phase7Config, output: Path, device: torch.device,
              artifact_hashes: Mapping[str, str], source_hashes: Mapping[str, str]) -> None:
    (output / "phase7_config.yaml").write_bytes(config.raw)
    for key, name in (("approval", "approval.yaml"),
                      ("frozen_protocol_document", "frozen_protocol_document.md"),
                      ("selector_split_manifest", "selector_split_manifest.json")):
        shutil.copyfile(config.paths[key], output / name)
    write_json(output / "artifact_sha256.json", dict(artifact_hashes))
    write_json(output / "source_sha256.json", dict(source_hashes))
    write_json(output / "environment.json", environment_record(config.base, device))
    for name, path in phase7_cache_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if _sha(target) != source_hashes[name]:
            raise ValueError(f"Phase 7 cache source changed while copying: {name}")


def _initial_status(config: Phase7Config, artifact_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "status": "extracting", "kind": "research_support_cache",
        "stage": "phase7_train_gate_descriptor_cache",
        "phase7_fingerprint": config.fingerprint,
        "artifact_sha256": dict(artifact_hashes),
        "clean": None, "condition_shards": {}, "completed_conditions": 0,
        "condition_count": 45, "inference_started": True,
        "training_started": False, "validation_evaluated": False,
        "test_evaluated": False,
    }


def _verify_resume_status(config: Phase7Config, output: Path,
                          artifact_hashes: Mapping[str, str]) -> dict[str, Any]:
    status_path = output / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError("Phase 7 cache resume requires status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") == "completed":
        raise ValueError("Completed Phase 7 cache must not be resumed")
    if (status.get("phase7_fingerprint") != config.fingerprint
            or status.get("artifact_sha256") != dict(artifact_hashes)
            or status.get("validation_evaluated") is not False
            or status.get("test_evaluated") is not False):
        raise ValueError("Phase 7 cache resume identity differs")
    clean = status.get("clean")
    if clean is not None:
        for key in ("descriptor", "records"):
            path = output / clean[f"{key}_path"]
            if not path.is_file() or _sha(path) != clean[f"{key}_sha256"]:
                raise ValueError("Completed clean descriptor shard changed")
    for condition, row in status.get("condition_shards", {}).items():
        for key in ("descriptor", "records"):
            path = output / row[f"{key}_path"]
            if not path.is_file() or _sha(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Completed descriptor shard changed: {condition}")
    return status


def _extract_batch(model, image_a: Tensor, image_b: Tensor, *, amp: bool) -> np.ndarray:
    with torch.no_grad(), torch.autocast(device_type=image_a.device.type, enabled=amp):
        features_a, features_b = model._encode(image_a, image_b)
        descriptor = phase7_descriptor(features_a, features_b, model.adapter)
    return descriptor.cpu().numpy().astype(np.float32, copy=False)


def _extract_clean(config: Phase7Config, model, loader, split_by_name: Mapping[str, str],
                   output: Path, guard_callback: Callable[[], None] | None) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    phase6_clean = {row["sample_id"]: row for row in _read_jsonl(
        config.upstream_paths["phase6_clean_actions"])}
    for batch in loader:
        if guard_callback:
            guard_callback()
        image_a, image_b = _transfer(
            batch, next(model.parameters()).device,
            config.base.values["runtime"]["memory_format"])
        arrays.append(_extract_batch(
            model, image_a, image_b, amp=config.base.values["train"]["amp"]))
        for sample_id in batch["sample_id"]:
            if sample_id not in phase6_clean:
                raise ValueError(f"Missing Phase 6 clean target: {sample_id}")
            source_name = _source_name(sample_id)
            rows.append({**phase6_clean[sample_id],
                         "phase7_partition": split_by_name[source_name]})
    descriptor = np.concatenate(arrays, axis=0)
    if (descriptor.shape != (1424, 1024) or len(rows) != 1424
            or len({row["sample_id"] for row in rows}) != 1424):
        raise ValueError("Clean descriptor cache coverage differs from Phase 6")
    descriptor_path = output / "clean_descriptors.npy"
    records_path = output / "clean_records.jsonl"
    _save_npy(descriptor_path, descriptor)
    _write_jsonl(records_path, rows)
    return {
        "rows": len(rows), "dimensions": 1024,
        "descriptor_path": descriptor_path.name,
        "descriptor_sha256": _sha(descriptor_path),
        "records_path": records_path.name, "records_sha256": _sha(records_path),
    }


def _extract_condition(config: Phase7Config, model, base: SourceSubsetDataset,
                       instances: list[dict[str, Any]], condition: str,
                       targets: Mapping[tuple[str, str], dict[str, Any]],
                       split_by_name: Mapping[str, str], output: Path,
                       guard_callback: Callable[[], None] | None) -> dict[str, Any]:
    dataset = GateInstanceDataset(base, instances)
    loader = make_loader(dataset, config.base, training=False)
    arrays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if guard_callback:
            guard_callback()
        image_a, image_b = _transfer(
            batch, next(model.parameters()).device,
            config.base.values["runtime"]["memory_format"])
        arrays.append(_extract_batch(
            model, image_a, image_b, amp=config.base.values["train"]["amp"]))
        for sample_id in batch["sample_id"]:
            key = (sample_id, condition)
            if key not in targets:
                raise ValueError(f"Missing Phase 6 perturbed target: {key}")
            row = targets[key]
            if row.get("condition_id") != condition:
                raise ValueError("Phase 6 condition target mismatch")
            rows.append({**row, "phase7_partition": split_by_name[_source_name(sample_id)]})
    descriptor = np.concatenate(arrays, axis=0)
    fit = sum(row["phase7_partition"] == "gate_fit" for row in rows)
    calibration = sum(row["phase7_partition"] == "gate_calibration" for row in rows)
    if (descriptor.shape != (1424, 1024) or len(rows) != 1424
            or fit != 1136 or calibration != 288
            or len({row["sample_id"] for row in rows}) != 1424):
        raise ValueError(f"Descriptor coverage differs for {condition}")
    directory = output / "conditions"
    directory.mkdir(exist_ok=True)
    descriptor_path = directory / f"{condition}.npy"
    records_path = directory / f"{condition}.jsonl"
    _save_npy(descriptor_path, descriptor)
    _write_jsonl(records_path, rows)
    return {
        "rows": len(rows), "fit_rows": fit, "calibration_rows": calibration,
        "dimensions": 1024,
        "descriptor_path": str(descriptor_path.relative_to(output)),
        "descriptor_sha256": _sha(descriptor_path),
        "records_path": str(records_path.relative_to(output)),
        "records_sha256": _sha(records_path),
    }


def _weighted_stats(clean: np.ndarray, perturbed: Iterable[np.ndarray],
                    floor: float) -> tuple[np.ndarray, np.ndarray]:
    clean64 = np.asarray(clean, dtype=np.float64)
    clean_mean = clean64.mean(axis=0)
    clean_second = np.square(clean64).mean(axis=0)
    total = 0
    sums = np.zeros(clean.shape[1], dtype=np.float64)
    squares = np.zeros(clean.shape[1], dtype=np.float64)
    for array in perturbed:
        value = np.asarray(array, dtype=np.float64)
        total += value.shape[0]
        sums += value.sum(axis=0)
        squares += np.square(value).sum(axis=0)
    if total <= 0:
        raise ValueError("Cannot normalize an empty perturbed cache")
    pert_mean, pert_second = sums / total, squares / total
    mean = 0.5 * (clean_mean + pert_mean)
    variance = np.maximum(0.5 * (clean_second + pert_second) - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), floor)
    return mean, std


def _scalar_stats(clean: np.ndarray, perturbed: np.ndarray,
                  floor: float) -> dict[str, float]:
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    mean = 0.5 * (float(clean.mean()) + float(perturbed.mean()))
    second = 0.5 * (float(np.square(clean).mean()) + float(np.square(perturbed).mean()))
    std = max(math_sqrt(max(second - mean * mean, 0.0)), floor)
    return {"mean": mean, "std": std}


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def _normalization(config: Phase7Config, output: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    clean_info = status["clean"]
    clean_desc = _load_npy(output / clean_info["descriptor_path"], (1424, 1024))
    clean_rows = _read_jsonl(output / clean_info["records_path"])
    clean_fit_indices = np.array([index for index, row in enumerate(clean_rows)
                                  if row["phase7_partition"] == "gate_fit"], dtype=np.int64)
    if clean_fit_indices.size != 1136:
        raise ValueError("Clean fit descriptor count differs")
    fit_clean = np.asarray(clean_desc[clean_fit_indices], dtype=np.float32)
    fit_clean_u0 = np.asarray([clean_rows[index]["u0"] for index in clean_fit_indices],
                              dtype=np.float64)
    pert_arrays: list[np.ndarray] = []
    u_t: list[float] = []
    u_0: list[float] = []
    advantage: list[float] = []
    for condition in _condition_order():
        shard = status["condition_shards"][condition]
        array = _load_npy(output / shard["descriptor_path"], (1424, 1024))
        rows = _read_jsonl(output / shard["records_path"])
        indices = np.array([index for index, row in enumerate(rows)
                            if row["phase7_partition"] == "gate_fit"], dtype=np.int64)
        if indices.size != 1136:
            raise ValueError(f"Fit descriptor count differs: {condition}")
        pert_arrays.append(np.asarray(array[indices], dtype=np.float32))
        u_t.extend(float(rows[index]["uT"]) for index in indices)
        u_0.extend(float(rows[index]["u0"]) for index in indices)
        advantage.extend(float(rows[index]["a"]) for index in indices)
    descriptor_mean, descriptor_std = _weighted_stats(
        fit_clean, pert_arrays, config.values["descriptor"]["std_floor"])
    u_t_array = np.asarray(u_t, dtype=np.float64)
    u_0_array = np.asarray(u_0, dtype=np.float64)
    advantage_array = np.asarray(advantage, dtype=np.float64)
    if u_t_array.size != 51120:
        raise ValueError("Fit utility count differs from approval")
    target_floor = config.values["targets"]["target_std_floor"]
    primary = _scalar_stats(fit_clean_u0, u_t_array, target_floor)
    clean_target = _scalar_stats(fit_clean_u0, u_0_array, target_floor)
    difference = _scalar_stats(np.zeros_like(fit_clean_u0), advantage_array, target_floor)
    positive_fraction = 0.5 * (float((fit_clean_u0 > 0).mean())
                               + float((u_t_array > 0).mean()))
    nonpositive_fraction = 1.0 - positive_fraction
    if not 0 < positive_fraction < 1:
        raise ValueError("C1 fit classes are degenerate")
    class_weights = {
        "positive": 1.0 / (2.0 * positive_fraction),
        "nonpositive": 1.0 / (2.0 * nonpositive_fraction),
    }
    result = {
        "status": "frozen", "source": "gate_fit_only",
        "weighting": "equal_clean_perturbed", "dimensions": 1024,
        "descriptor": {
            "mean": descriptor_mean.tolist(), "std": descriptor_std.tolist(),
            "std_floor": config.values["descriptor"]["std_floor"],
        },
        "targets": {"u_obs": primary, "u_clean": clean_target, "a_obs": difference,
                    "std_floor": target_floor},
        "c1": {"positive_fraction": positive_fraction,
               "nonpositive_fraction": nonpositive_fraction,
               "class_weights": class_weights},
        "fit_clean_records": 1136, "fit_perturbed_records": 51120,
        "test_evaluated": False,
    }
    path = output / "normalization.json"
    write_json(path, result)
    return {**result, "sha256": _sha(path)}


def build_descriptor_cache(config: Phase7Config, *, resume: bool = False,
                           guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    split = load_selector_split(config)
    phase56_partition, gate_manifest = load_frozen_manifests(config.phase56)
    artifact_hashes = phase6_artifact_hashes(config.phase56)
    source_hashes = {name: _sha(path) for name, path in phase7_cache_source_files().items()}
    output = config.paths["descriptor_cache_dir"]
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("Phase 7 cache resume requires an output directory")
        status = _verify_resume_status(config, output, artifact_hashes)
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite Phase 7 descriptor cache: {output}")
        status = _initial_status(config, artifact_hashes)
    device = setup_runtime(config.base)
    train_dataset = LEVIRDataset(config.base, "train")
    verify_partition_files(config.phase56, train_dataset, phase56_partition)
    gate_dataset = SourceSubsetDataset(
        train_dataset, phase56_partition["gate_names"], partition="gate")
    if len(gate_dataset) != 1424:
        raise ValueError("Phase 7 gate dataset differs from the frozen 89 sources")
    model, _, _ = load_phase6_models(config.phase56, artifact_hashes, device)
    output.mkdir(parents=True, exist_ok=resume)
    if not resume:
        write_json(output / "status.json", status)
        _snapshot(config, output, device, artifact_hashes, source_hashes)
    split_by_name = {name: "gate_fit" for name in split["fit_names"]}
    split_by_name.update({name: "gate_calibration" for name in split["calibration_names"]})
    instances_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in gate_manifest["instances"]:
        instances_by_condition[instance["condition_id"]].append(instance)
    target_rows = _read_jsonl(config.upstream_paths["phase6_action_utility"])
    targets = {(row["sample_id"], row["condition_id"]): row for row in target_rows}
    if len(targets) != 64080:
        raise ValueError("Phase 6 action utility target coverage differs")
    started = time.monotonic()
    try:
        if status.get("clean") is None:
            _discard_uncommitted((output / "clean_descriptors.npy",
                                  output / "clean_records.jsonl"), output)
            if guard_callback:
                guard_callback()
            status["clean"] = _extract_clean(
                config, model, make_loader(gate_dataset, config.base, training=False),
                split_by_name, output, guard_callback)
            write_json(output / "status.json", status, overwrite=True)
        for condition in _condition_order():
            if condition in status["condition_shards"]:
                continue
            _discard_uncommitted((output / "conditions" / f"{condition}.npy",
                                  output / "conditions" / f"{condition}.jsonl"), output)
            shard = _extract_condition(
                config, model, gate_dataset, instances_by_condition[condition], condition,
                targets, split_by_name, output, guard_callback)
            status["condition_shards"][condition] = shard
            status["completed_conditions"] = len(status["condition_shards"])
            write_json(output / "status.json", status, overwrite=True)
            print(f"phase7/cache/{condition}: completed "
                  f"({status['completed_conditions']}/45)", flush=True)
        if set(status["condition_shards"]) != set(_condition_order()):
            raise ValueError("Phase 7 cache did not complete all 45 conditions")
        normalization = _normalization(config, output, status)
        if ({name: _sha(path) for name, path in phase7_cache_source_files().items()}
                != source_hashes or phase6_artifact_hashes(config.phase56) != artifact_hashes):
            raise ValueError("Phase 7 cache sources or model artifacts changed during extraction")
        result = {
            **status, "status": "completed", "seconds": time.monotonic() - started,
            "normalization_sha256": normalization["sha256"],
            "clean_rows": 1424, "perturbed_rows": 64080,
            "fit_clean_rows": 1136, "calibration_clean_rows": 288,
            "fit_perturbed_rows": 51120, "calibration_perturbed_rows": 12960,
            "descriptor_dimensions": 1024, "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "source_sha256": source_hashes, "guarded_completion_required": True,
        }
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        status.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}", validation_evaluated=False,
            test_evaluated=False)
        write_json(output / "status.json", status, overwrite=True)
        raise


def load_descriptor_cache(config: Phase7Config, *, require_guard: bool = True) -> dict[str, Any]:
    output = config.paths["descriptor_cache_dir"]
    status_path = output / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError("Phase 7 descriptor cache status is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("status") != "completed"
            or status.get("phase7_fingerprint") != config.fingerprint
            or status.get("validation_evaluated") is not False
            or status.get("test_evaluated") is not False
            or status.get("completed_conditions") != 45):
        raise ValueError("Phase 7 descriptor cache is not a completed train-only cache")
    if require_guard:
        guard_path = output / "guard_completion.json"
        if not guard_path.is_file() or json.loads(guard_path.read_text(
                encoding="utf-8")).get("status") != "completed":
            raise ValueError("Phase 7 descriptor cache lacks guarded completion")
    if _sha(output / "normalization.json") != status["normalization_sha256"]:
        raise ValueError("Phase 7 normalization changed")
    clean = status["clean"]
    for key in ("descriptor", "records"):
        path = output / clean[f"{key}_path"]
        if _sha(path) != clean[f"{key}_sha256"]:
            raise ValueError("Phase 7 clean cache changed")
    for condition in _condition_order():
        shard = status["condition_shards"][condition]
        for key in ("descriptor", "records"):
            path = output / shard[f"{key}_path"]
            if _sha(path) != shard[f"{key}_sha256"]:
                raise ValueError(f"Phase 7 condition cache changed: {condition}")
    return {"output": output, "status": status,
            "normalization": json.loads((output / "normalization.json").read_text(
                encoding="utf-8"))}
