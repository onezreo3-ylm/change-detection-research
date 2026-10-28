"""Frozen Phase 5-6 protocol, train-source partition, and gate instances.

This module is deliberately separate from the completed Phase 4 code.  It may
open only the official training split while preparing the actor/gate manifests;
model training may additionally use clean validation for checkpoint selection.
Held-out test access is never permitted here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

from .config import Config, UniqueKeyLoader, load_config, protocol_signature
from .corruption_protocol import (
    KINDS,
    CorruptionProtocol,
    canonical,
    keyed_seed,
    load_protocol,
    make_instance,
)
from .data import LEVIRDataset, file_sha256
from .robust_baselines import RobustConfig, load_robust_config
from .runtime import write_json


_CONFIG_KEYS = {
    "schema_version", "name", "status", "base_config", "base_config_sha256",
    "corruption_protocol", "corruption_protocol_file_sha256",
    "corruption_protocol_sha256", "severity_review", "severity_review_sha256",
    "frozen_protocol_document", "frozen_protocol_document_sha256",
    "reference_b1_config", "reference_b1_config_sha256", "protocol_output_dir",
    "actor_gate_manifest", "gate_action_manifest", "approval", "split",
    "actor_detector", "adapter", "diagnostic", "boundaries",
}
_PATH_KEYS = {
    "base_config", "corruption_protocol", "severity_review",
    "frozen_protocol_document", "reference_b1_config", "protocol_output_dir",
    "actor_gate_manifest", "gate_action_manifest", "approval",
}
_SPLIT_KEYS = {
    "source_split", "actor_sources", "gate_sources", "group_size", "ordering",
    "gate_selection", "salt", "label_decode",
}
_ACTOR_KEYS = {
    "output_dir", "method", "initialization", "expected_initial_model_tensor_sha256",
    "expected_source_tiles", "expected_batches_per_epoch", "views_per_source",
    "expected_pair_views", "expected_attempted_optimizer_steps", "epochs",
    "consistency_kind", "consistency_weight", "selection",
}
_ADAPTER_KEYS = {
    "output_dir", "actor_best_checkpoint", "actor_last_checkpoint",
    "actor_guard_completion", "detector_state", "shared_across_dates",
    "stage_number", "channels", "kernel_sizes", "padding", "convolution_bias",
    "normalization", "normalization_groups", "hidden_activation", "final_activation",
    "final_zero_init", "expected_source_tiles", "expected_batches_per_epoch",
    "views_per_source", "expected_pair_views", "expected_attempted_optimizer_steps",
    "epochs", "objective", "consistency_kind", "selection",
}
_DIAGNOSTIC_KEYS = {
    "output_dir", "adapter_best_checkpoint", "adapter_last_checkpoint",
    "adapter_guard_completion", "manifest_salt", "manifest_view", "gate_tiles",
    "conditions", "perturbed_instances", "actions", "clean_actions_cached_once",
    "loss", "bootstrap_unit", "bootstrap_replicates", "bootstrap_seed",
    "confidence_level", "min_positive_utility_fraction",
    "min_negative_utility_fraction", "min_families_each_sign",
    "min_oracle_mf1_margin", "require_oracle_margin_ci_lower_positive",
}
_BOUNDARY_KEYS = {
    "actor_and_adapter_training_sources", "gate_diagnostic_sources",
    "validation_use", "corruption_validation_selection", "held_out_test_access",
    "learned_gate", "local_actions", "phase7_to_10",
}
_APPROVAL_KEYS = {
    "schema_version", "status", "approved_for", "reviewer", "reviewed_at",
    "approval_text", "phase56_config_sha256", "frozen_protocol_document_sha256",
    "actor_gate_manifest_sha256", "gate_action_manifest_sha256", "test_access",
    "notes",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label}: missing={expected - actual}, unexpected={actual - expected}")


def _lower_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return float(value)


def _path(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else source.parent / candidate).resolve()


def _load_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value, raw


@dataclass(frozen=True)
class Phase56Config:
    values: Mapping[str, Any]
    source: Path
    raw: bytes
    base: Config
    corruption: CorruptionProtocol
    reference_b1: RobustConfig
    paths: Mapping[str, Path]
    approval: Mapping[str, Any] | None

    @property
    def config_sha256(self) -> str:
        return _sha_bytes(self.raw)

    @property
    def fingerprint(self) -> str:
        approval = self.approval
        return _digest({
            "phase56_config_sha256": self.config_sha256,
            "base_protocol_signature": protocol_signature(self.base.values),
            "corruption_protocol_sha256": self.corruption.fingerprint,
            "document_sha256": self.values["frozen_protocol_document_sha256"],
            "actor_gate_manifest_sha256": (
                approval["actor_gate_manifest_sha256"] if approval else None),
            "gate_action_manifest_sha256": (
                approval["gate_action_manifest_sha256"] if approval else None),
        })

    def path(self, key: str) -> Path:
        if key in self.paths:
            return self.paths[key]
        for section in ("actor_detector", "adapter", "diagnostic"):
            if key in self.values[section] and key.endswith(("_dir", "_checkpoint", "_completion")):
                return _path(self.source, self.values[section][key], f"{section}.{key}")
        raise KeyError(key)


def _validate_exact_protocol(values: dict[str, Any], base: Config,
                             corruption: CorruptionProtocol,
                             reference_b1: RobustConfig) -> None:
    _keys(values["split"], _SPLIT_KEYS, "split")
    _keys(values["actor_detector"], _ACTOR_KEYS, "actor_detector")
    _keys(values["adapter"], _ADAPTER_KEYS, "adapter")
    _keys(values["diagnostic"], _DIAGNOSTIC_KEYS, "diagnostic")
    _keys(values["boundaries"], _BOUNDARY_KEYS, "boundaries")

    split = values["split"]
    expected_split = {
        "source_split": "train", "actor_sources": 356, "gate_sources": 89,
        "group_size": 5, "ordering": "foreground_pixel_count_then_canonical_filename",
        "gate_selection": "minimum_sha256_per_group",
        "salt": "phase5_actor_gate_v1_seed42",
        "label_decode": "author_uint8_floor255",
    }
    if split != expected_split:
        raise ValueError("Phase 5 source split differs from the user-approved protocol")
    if base.run_test_after_training or base.values["data"]["label_decode"] != split["label_decode"]:
        raise ValueError("Phase 5 requires the development-only author-floor255 base config")
    if base.values["runtime"]["device"] not in ("cuda", "cuda:0"):
        raise ValueError("Phase 5 research runs require the approved real CUDA device")
    if base.values["experiment"]["seed"] != 42 or base.values["train"]["epochs"] != 100:
        raise ValueError("Phase 5 base seed/epoch budget differs from the approved protocol")
    if reference_b1.mode != "b1" or reference_b1.base.values != base.values:
        raise ValueError("Phase 5 actor detector must match the frozen B1 base protocol")
    if reference_b1.protocol.fingerprint != corruption.fingerprint:
        raise ValueError("Phase 5 corruption protocol differs from frozen B1")

    actor = values["actor_detector"]
    if {key: actor[key] for key in _ACTOR_KEYS - {"output_dir"}} != {
        "method": "b1_clean_plus_perturbed_supervision",
        "initialization": "fresh_random_seed42",
        "expected_initial_model_tensor_sha256": (
            "eb4fa2550e47d732c5b21acb6899ba5539856bee35b532da58006f56108001a8"),
        "expected_source_tiles": 5696, "expected_batches_per_epoch": 1424,
        "views_per_source": 2, "expected_pair_views": 1139200,
        "expected_attempted_optimizer_steps": 142400, "epochs": 100,
        "consistency_kind": "none", "consistency_weight": 0.0,
        "selection": "clean_val_f1",
    }:
        raise ValueError("Actor detector settings differ from the approved protocol")

    adapter = values["adapter"]
    if {key: adapter[key] for key in _ADAPTER_KEYS - {
            "output_dir", "actor_best_checkpoint", "actor_last_checkpoint",
            "actor_guard_completion"}} != {
        "detector_state": "frozen_eval", "shared_across_dates": True,
        "stage_number": 2, "channels": [128, 32, 32, 128],
        "kernel_sizes": [1, 3, 1], "padding": [0, 1, 0],
        "convolution_bias": [False, False, True], "normalization": "group_norm",
        "normalization_groups": 8, "hidden_activation": "relu",
        "final_activation": "none", "final_zero_init": True,
        "expected_source_tiles": 5696, "expected_batches_per_epoch": 1424,
        "views_per_source": 2, "expected_pair_views": 1139200,
        "expected_attempted_optimizer_steps": 142400, "epochs": 100,
        "objective": "equal_mean_supervised_clean_and_perturbed",
        "consistency_kind": "none", "selection": "clean_corrected_val_f1",
    }:
        raise ValueError("Adapter settings differ from the approved protocol")

    diagnostic = values["diagnostic"]
    if {key: diagnostic[key] for key in _DIAGNOSTIC_KEYS - {
            "output_dir", "adapter_best_checkpoint", "adapter_last_checkpoint",
            "adapter_guard_completion"}} != {
        "manifest_salt": "phase6_gate_action_v1_seed42", "manifest_view": 1,
        "gate_tiles": 1424, "conditions": 45, "perturbed_instances": 64080,
        "actions": ["a0_no_correction", "a1_shared_stage2_adapter"],
        "clean_actions_cached_once": True,
        "loss": "existing_per_sample_bce_plus_soft_dice",
        "bootstrap_unit": "original_source_pair", "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260915, "confidence_level": 0.95,
        "min_positive_utility_fraction": 0.05,
        "min_negative_utility_fraction": 0.05,
        "min_families_each_sign": 3, "min_oracle_mf1_margin": 0.002,
        "require_oracle_margin_ci_lower_positive": True,
    }:
        raise ValueError("Phase 6 diagnostic settings differ from the approved protocol")
    boundaries = values["boundaries"]
    if boundaries != {
        "actor_and_adapter_training_sources": "actor_only",
        "gate_diagnostic_sources": "gate_only",
        "validation_use": "clean_checkpoint_selection_only",
        "corruption_validation_selection": False, "held_out_test_access": False,
        "learned_gate": False, "local_actions": False, "phase7_to_10": False,
    }:
        raise ValueError("Phase 5-6 lifecycle boundaries differ from approval")


def _load_approval(config: Phase56Config) -> Mapping[str, Any]:
    path = config.paths["approval"]
    approval, _ = _load_yaml(path)
    _keys(approval, _APPROVAL_KEYS, "phase5_6 approval")
    if approval["schema_version"] != 1 or approval["status"] != "approved":
        raise ValueError("Phase 5-6 approval must be schema 1 and approved")
    if approval["approved_for"] != "phase5_6_action_diagnostic_only":
        raise ValueError("Approval scope must remain Phase 5-6 diagnostic only")
    for key in ("reviewer", "reviewed_at", "approval_text", "notes"):
        if not isinstance(approval[key], str) or not approval[key].strip():
            raise ValueError(f"approval.{key} must be nonempty")
    if approval["approval_text"] != "批准按草案执行":
        raise ValueError("Approval text differs from the user's explicit response")
    if approval["test_access"] is not False:
        raise ValueError("Phase 5-6 approval must keep test access false")
    bindings = {
        "phase56_config_sha256": config.config_sha256,
        "frozen_protocol_document_sha256": config.values["frozen_protocol_document_sha256"],
        "actor_gate_manifest_sha256": _sha_file(config.paths["actor_gate_manifest"]),
        "gate_action_manifest_sha256": _sha_file(config.paths["gate_action_manifest"]),
    }
    for key, actual in bindings.items():
        if _lower_sha(approval[key], f"approval.{key}") != actual:
            raise ValueError(f"Approval binding differs: {key}")
    return MappingProxyType(approval)


def load_phase56_config(path: str | Path, *, require_approval: bool = True) -> Phase56Config:
    source = Path(path).resolve()
    values, raw = _load_yaml(source)
    _keys(values, _CONFIG_KEYS, "phase5_6 config")
    if values["schema_version"] != 1 or values["status"] != "approved_frozen":
        raise ValueError("Phase 5-6 config must be approved_frozen schema version 1")
    if not isinstance(values["name"], str) or not values["name"].strip():
        raise ValueError("Phase 5-6 name must be nonempty")
    paths = {key: _path(source, values[key], key) for key in _PATH_KEYS}
    for key, sha_key in (
        ("base_config", "base_config_sha256"),
        ("corruption_protocol", "corruption_protocol_file_sha256"),
        ("severity_review", "severity_review_sha256"),
        ("frozen_protocol_document", "frozen_protocol_document_sha256"),
        ("reference_b1_config", "reference_b1_config_sha256"),
    ):
        expected = _lower_sha(values[sha_key], sha_key)
        if _sha_file(paths[key]) != expected:
            raise ValueError(f"Phase 5-6 bound file changed: {key}")
    base = load_config(paths["base_config"])
    corruption = load_protocol(paths["corruption_protocol"])
    if corruption.fingerprint != _lower_sha(values["corruption_protocol_sha256"],
                                             "corruption_protocol_sha256"):
        raise ValueError("Phase 5-6 corruption fingerprint differs")
    reference_b1 = load_robust_config(paths["reference_b1_config"])
    provisional = Phase56Config(MappingProxyType(values), source, raw, base, corruption,
                                reference_b1, MappingProxyType(paths), None)
    _validate_exact_protocol(values, base, corruption, reference_b1)
    if not require_approval:
        return provisional
    approval = _load_approval(provisional)
    return Phase56Config(provisional.values, source, raw, base, corruption, reference_b1,
                         provisional.paths, approval)


def _source_selection_hash(salt: str, name: str) -> str:
    return hashlib.sha256((salt + "\0" + name).encode("utf-8")).hexdigest()


def _cdf_distance(a: list[float], b: list[float]) -> float:
    support = sorted(set(a + b))
    if not support:
        raise ValueError("Cannot compare empty partitions")
    return max(abs(sum(value <= point for value in a) / len(a)
                   - sum(value <= point for value in b) / len(b)) for point in support)


def _partition_sources(config: Phase56Config, dataset: LEVIRDataset) -> dict[str, Any]:
    if dataset.split != "train" or config.base.run_test_after_training:
        raise ValueError("Actor/gate preparation is train-only")
    settings = config.values["split"]
    rows = []
    for record in dataset.records:
        pair = dataset.read_pair(record)
        foreground = int(pair.mask.sum(dtype=np.int64))
        pixels = int(pair.mask.size)
        rows.append({
            "name": record.name,
            "foreground_pixels": foreground,
            "total_pixels": pixels,
            "foreground_fraction": foreground / pixels,
            "selection_sha256": _source_selection_hash(settings["salt"], record.name),
            "size": list(record.size),
            "tile_count": (record.size[0] // dataset.tile_size)
                          * (record.size[1] // dataset.tile_size),
            "file_sha256": {
                "image_a": file_sha256(record.image_a),
                "image_b": file_sha256(record.image_b),
                "mask": file_sha256(record.mask),
            },
        })
    rows.sort(key=lambda row: (row["foreground_pixels"], row["name"]))
    expected = settings["actor_sources"] + settings["gate_sources"]
    if len(rows) != expected or len(rows) % settings["group_size"]:
        raise ValueError("Training source count cannot satisfy the frozen 356/89 grouping")
    for group_index, start in enumerate(range(0, len(rows), settings["group_size"])):
        group = rows[start:start + settings["group_size"]]
        gate_name = min(group, key=lambda row: (row["selection_sha256"], row["name"]))["name"]
        for row in group:
            row["group_index"] = group_index
            row["partition"] = "gate" if row["name"] == gate_name else "actor"
    actor = [row for row in rows if row["partition"] == "actor"]
    gate = [row for row in rows if row["partition"] == "gate"]
    if len(actor) != settings["actor_sources"] or len(gate) != settings["gate_sources"]:
        raise ValueError("Frozen actor/gate partition counts are not satisfied")

    def summary(partition: list[dict[str, Any]]) -> dict[str, Any]:
        foreground = sum(row["foreground_pixels"] for row in partition)
        pixels = sum(row["total_pixels"] for row in partition)
        fractions = np.array([row["foreground_fraction"] for row in partition])
        return {
            "sources": len(partition), "tiles": sum(row["tile_count"] for row in partition),
            "foreground_pixels": foreground, "total_pixels": pixels,
            "foreground_fraction": foreground / pixels,
            "median_source_fraction": float(np.median(fractions)),
            "zero_foreground_sources": sum(row["foreground_pixels"] == 0 for row in partition),
        }

    return {
        "schema_version": 1, "status": "frozen", "kind": "train_source_partition",
        "name": config.values["name"], "split": "train", "test_evaluated": False,
        "algorithm": {
            "ordering": settings["ordering"], "group_size": settings["group_size"],
            "gate_selection": settings["gate_selection"], "salt": settings["salt"],
            "hash_payload": "UTF8(salt + NUL + canonical_filename)",
        },
        "base_protocol_signature": protocol_signature(config.base.values),
        "corruption_protocol_sha256": config.corruption.fingerprint,
        "phase56_config_sha256": config.config_sha256,
        "sources": rows,
        "actor_names": sorted(row["name"] for row in actor),
        "gate_names": sorted(row["name"] for row in gate),
        "summary": {"actor": summary(actor), "gate": summary(gate),
                    "foreground_fraction_cdf_distance": _cdf_distance(
                        [row["foreground_fraction"] for row in actor],
                        [row["foreground_fraction"] for row in gate])},
    }


def _gate_sample_ids(dataset: LEVIRDataset, gate_names: set[str]) -> list[str]:
    ids = []
    for record_index, x, y in dataset.tiles:
        name = dataset.records[record_index].name
        if name in gate_names:
            ids.append(f"train/{name}@x{x}_y{y}")
    if len(ids) != len(gate_names) * 16 or len(ids) != len(set(ids)):
        raise ValueError("Gate tile IDs do not match 89 full 4x4 source grids")
    return ids


def gate_instances(config: Phase56Config, ids: list[str]) -> Iterator[dict[str, Any]]:
    diagnostic = config.values["diagnostic"]
    if (not ids or len(ids) != len(set(ids))
            or any(not sample_id.startswith("train/") for sample_id in ids)):
        raise ValueError("Phase 6 requires unique train/gate sample IDs")
    for kind in KINDS:
        directions = sorted(config.corruption.values["corruptions"][kind])
        strata = list(itertools.product(directions, ("A", "B"), ("global", "local")))
        order = sorted(ids, key=lambda sample_id: (
            keyed_seed(diagnostic["manifest_salt"], kind, sample_id), sample_id))
        assignment = {sample_id: strata[index % len(strata)]
                      for index, sample_id in enumerate(order)}
        for level in range(1, 6):
            for sample_id in ids:
                direction, target_date, spatial = assignment[sample_id]
                yield make_instance(
                    config.corruption, sample_id, kind, level, stream="train",
                    direction=direction, target_date=target_date, spatial=spatial,
                    epoch=0, view=diagnostic["manifest_view"])


def _make_gate_manifest(config: Phase56Config, dataset: LEVIRDataset,
                        partition: Mapping[str, Any]) -> dict[str, Any]:
    ids = _gate_sample_ids(dataset, set(partition["gate_names"]))
    instances = list(gate_instances(config, ids))
    diagnostic = config.values["diagnostic"]
    if len(ids) != diagnostic["gate_tiles"] or len(instances) != diagnostic["perturbed_instances"]:
        raise ValueError("Phase 6 gate manifest count differs from the frozen protocol")
    condition_counts = {f"{kind}_l{level}": 0 for kind in KINDS for level in range(1, 6)}
    stratum_counts: dict[str, int] = {}
    for instance in instances:
        condition_counts[instance["condition_id"]] += 1
        key = "|".join((instance["condition_id"], instance["direction"],
                        instance["target_date"], instance["spatial_mode"]))
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    if (len(condition_counts) != diagnostic["conditions"]
            or set(condition_counts.values()) != {diagnostic["gate_tiles"]}):
        raise ValueError("Gate manifest is not exactly balanced over 45 cells")
    return {
        "schema_version": 1, "status": "frozen", "kind": "phase6_gate_action_manifest",
        "name": config.values["name"], "split": "train_gate_only",
        "test_evaluated": False, "phase56_config_sha256": config.config_sha256,
        "actor_gate_partition_sha256": _digest(partition),
        "corruption_protocol_sha256": config.corruption.fingerprint,
        "manifest_salt": diagnostic["manifest_salt"],
        "manifest_view": diagnostic["manifest_view"],
        "source_pair_count": len(partition["gate_names"]), "tile_count": len(ids),
        "condition_count": len(condition_counts), "instance_count": len(instances),
        "sample_ids": ids, "condition_counts": condition_counts,
        "stratum_counts": dict(sorted(stratum_counts.items())), "instances": instances,
    }


def phase5_protocol_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/perturbations.py", "pcdc/phase5_protocol.py",
        "pcdc/robust_baselines.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def prepare_phase56_protocol(config: Phase56Config) -> dict[str, Any]:
    """Create immutable train-only split/action manifests; never load val/test."""
    if config.approval is not None:
        raise ValueError("Preparation expects config loaded without approval")
    output = config.paths["protocol_output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 5-6 protocol: {output}")
    if config.paths["actor_gate_manifest"].parent != output or config.paths[
            "gate_action_manifest"].parent != output:
        raise ValueError("Both Phase 5-6 manifests must live in the protocol output directory")
    dataset = LEVIRDataset(config.base, "train")
    partition = _partition_sources(config, dataset)
    gate_manifest = _make_gate_manifest(config, dataset, partition)
    sources = phase5_protocol_source_files()
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "phase5_6_config_source.yaml").write_bytes(config.raw)
        shutil.copyfile(config.paths["frozen_protocol_document"],
                        output / "frozen_protocol_document.md")
        write_json(config.paths["actor_gate_manifest"], partition)
        write_json(config.paths["gate_action_manifest"], gate_manifest)
        for name, path in sources.items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_sha256(target) != source_hashes[name]:
                raise ValueError(f"Phase 5 protocol source changed while snapshotting: {name}")
        write_json(output / "source_sha256.json", source_hashes)
        result = {
            "status": "prepared_awaiting_bound_approval_sidecar",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase56_config_sha256": config.config_sha256,
            "frozen_protocol_document_sha256": _sha_file(
                config.paths["frozen_protocol_document"]),
            "actor_gate_manifest_sha256": _sha_file(config.paths["actor_gate_manifest"]),
            "gate_action_manifest_sha256": _sha_file(config.paths["gate_action_manifest"]),
            "source_sha256": source_hashes,
            "actor_sources": len(partition["actor_names"]),
            "gate_sources": len(partition["gate_names"]),
            "gate_tiles": gate_manifest["tile_count"],
            "gate_instances": gate_manifest["instance_count"],
            "dataset_splits_opened": ["train"], "model_initialized": False,
            "training_started": False, "test_evaluated": False,
        }
        write_json(output / "status.json", result)
        return result
    except BaseException as error:
        if output.exists():
            write_json(output / "status.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
                "dataset_splits_opened": ["train"], "model_initialized": False,
                "training_started": False, "test_evaluated": False,
            }, overwrite=True)
        raise


def load_frozen_manifests(config: Phase56Config) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.approval is None:
        raise ValueError("Frozen manifests require the bound user approval")
    partition = json.loads(config.paths["actor_gate_manifest"].read_text(encoding="utf-8"))
    gate = json.loads(config.paths["gate_action_manifest"].read_text(encoding="utf-8"))
    if (partition.get("status") != "frozen" or partition.get("split") != "train"
            or partition.get("test_evaluated") is not False
            or partition.get("phase56_config_sha256") != config.config_sha256
            or len(partition.get("actor_names", [])) != 356
            or len(partition.get("gate_names", [])) != 89
            or set(partition["actor_names"]) & set(partition["gate_names"])):
        raise ValueError("Invalid frozen actor/gate manifest")
    expected_names = set(partition["actor_names"]) | set(partition["gate_names"])
    source_rows = partition.get("sources", [])
    if (len(source_rows) != 445 or {row.get("name") for row in source_rows} != expected_names
            or any(row.get("partition") not in ("actor", "gate") for row in source_rows)):
        raise ValueError("Actor/gate source records are incomplete")
    diagnostic = config.values["diagnostic"]
    instances = gate.get("instances", [])
    if (gate.get("status") != "frozen" or gate.get("split") != "train_gate_only"
            or gate.get("test_evaluated") is not False
            or gate.get("phase56_config_sha256") != config.config_sha256
            or gate.get("source_pair_count") != 89
            or gate.get("tile_count") != diagnostic["gate_tiles"]
            or gate.get("condition_count") != diagnostic["conditions"]
            or gate.get("instance_count") != diagnostic["perturbed_instances"]
            or len(instances) != diagnostic["perturbed_instances"]):
        raise ValueError("Invalid frozen Phase 6 action manifest")
    sample_ids = gate.get("sample_ids", [])
    if (len(sample_ids) != diagnostic["gate_tiles"] or len(sample_ids) != len(set(sample_ids))
            or any(not sample_id.startswith("train/") for sample_id in sample_ids)):
        raise ValueError("Invalid gate sample ID coverage")
    expected_conditions = {f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)}
    if (set(gate.get("condition_counts", {})) != expected_conditions
            or set(gate["condition_counts"].values()) != {diagnostic["gate_tiles"]}):
        raise ValueError("Invalid Phase 6 condition coverage")
    seen = set()
    for instance in instances:
        key = (instance.get("sample_id"), instance.get("condition_id"))
        if (key in seen or instance.get("sample_id") not in sample_ids
                or instance.get("condition_id") not in expected_conditions
                or instance.get("stream") != "train"
                or instance.get("epoch") != 0
                or instance.get("view") != diagnostic["manifest_view"]):
            raise ValueError("Invalid or duplicate Phase 6 instance")
        seen.add(key)
    if len(seen) != diagnostic["perturbed_instances"]:
        raise ValueError("Incomplete Phase 6 sample-condition Cartesian product")
    return partition, gate


def verify_partition_files(config: Phase56Config, dataset: LEVIRDataset,
                           partition: Mapping[str, Any]) -> None:
    """Bind every current training A/B/label file to the approved manifest."""
    if dataset.split != "train" or config.base.run_test_after_training:
        raise ValueError("Partition verification is train-only")
    rows = {row["name"]: row for row in partition["sources"]}
    if set(rows) != {record.name for record in dataset.records}:
        raise ValueError("Current train source names differ from the frozen partition")
    for record in dataset.records:
        expected = rows[record.name]
        actual = {key: file_sha256(getattr(record, key))
                  for key in ("image_a", "image_b", "mask")}
        if actual != expected["file_sha256"]:
            raise ValueError(f"Current train source content changed: {record.name}")


def preflight(config: Phase56Config) -> dict[str, Any]:
    partition, gate = load_frozen_manifests(config)
    return {
        "status": "phase5_6_preflight_passed",
        "kind": "metadata_only_no_data_or_model",
        "phase56_fingerprint": config.fingerprint,
        "approval_status": config.approval["status"] if config.approval else None,
        "actor_sources": len(partition["actor_names"]),
        "gate_sources": len(partition["gate_names"]),
        "gate_instances": gate["instance_count"],
        "dataset_opened": False, "model_initialized": False,
        "training_started": False, "sensor_queried": False,
        "test_evaluated": False,
    }
