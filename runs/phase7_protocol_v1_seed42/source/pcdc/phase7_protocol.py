"""Approved Phase 7 selector protocol, bindings, and source split.

Preparation is metadata-only: it derives the 71/18 split from the already frozen
Phase 5-6 manifest and never opens dataset imagery, validation, or test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .config import Config, UniqueKeyLoader, load_config
from .corruption_protocol import canonical
from .data import file_sha256
from .phase5_protocol import Phase56Config, load_phase56_config
from .runtime import write_json


_TOP_KEYS = {
    "schema_version", "name", "status", "base_config", "base_config_sha256",
    "phase56_config", "phase56_config_sha256", "frozen_protocol_document",
    "frozen_protocol_document_sha256", "reviewed_draft_sha256", "approval",
    "protocol_output_dir", "selector_split_manifest", "descriptor_cache_dir",
    "preval_seal", "validation_output_dir", "upstream", "split", "descriptor",
    "selector", "targets", "training", "policies", "evaluation", "boundaries",
}
_TOP_PATH_KEYS = {
    "base_config", "phase56_config", "frozen_protocol_document", "approval",
    "protocol_output_dir", "selector_split_manifest", "descriptor_cache_dir",
    "preval_seal", "validation_output_dir",
}
_UPSTREAM_PATH_HASH = {
    "actor_gate_manifest": "actor_gate_manifest_sha256",
    "gate_action_manifest": "gate_action_manifest_sha256",
    "actor_best_checkpoint": "actor_best_sha256",
    "actor_status": "actor_status_sha256",
    "actor_guard_completion": "actor_guard_sha256",
    "adapter_best_checkpoint": "adapter_best_sha256",
    "adapter_status": "adapter_status_sha256",
    "adapter_guard_completion": "adapter_guard_sha256",
    "phase6_action_utility": "phase6_action_utility_sha256",
    "phase6_clean_actions": "phase6_clean_actions_sha256",
    "phase6_status": "phase6_status_sha256",
    "phase6_gate_decision": "phase6_gate_decision_sha256",
    "validation_manifest": "validation_manifest_sha256",
}
_APPROVAL_KEYS = {
    "schema_version", "status", "approved_for", "reviewer", "reviewed_at",
    "approval_text", "reviewed_draft_sha256", "phase7_config_sha256",
    "frozen_protocol_document_sha256", "selector_split_manifest_sha256",
    "test_access", "notes",
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _digest(value: Any) -> str:
    return _sha_bytes(canonical(value).encode("utf-8"))


def _lower_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


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


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label}: missing={expected - actual}, unexpected={actual - expected}")


@dataclass(frozen=True)
class Phase7Config:
    values: Mapping[str, Any]
    source: Path
    raw: bytes
    base: Config
    phase56: Phase56Config
    paths: Mapping[str, Path]
    upstream_paths: Mapping[str, Path]
    approval: Mapping[str, Any] | None

    @property
    def config_sha256(self) -> str:
        return _sha_bytes(self.raw)

    @property
    def fingerprint(self) -> str:
        split_hash = (_sha_file(self.paths["selector_split_manifest"])
                      if self.paths["selector_split_manifest"].is_file() else None)
        return _digest({
            "phase7_config_sha256": self.config_sha256,
            "protocol_document_sha256": self.values["frozen_protocol_document_sha256"],
            "reviewed_draft_sha256": self.values["reviewed_draft_sha256"],
            "phase56_fingerprint": self.phase56.fingerprint,
            "selector_split_manifest_sha256": split_hash,
        })

    def training_output(self, variant: str) -> Path:
        value = self.values["training"]["output_dirs"][variant]
        return _path(self.source, value, f"training.output_dirs.{variant}")


def _validate_exact_protocol(values: dict[str, Any], base: Config,
                             phase56: Phase56Config) -> None:
    split = values["split"]
    if split != {
        "source_split": "train_gate_only", "gate_sources": 89,
        "fit_sources": 71, "calibration_sources": 18, "group_size": 5,
        "final_group_size": 4,
        "ordering": "foreground_pixel_count_then_canonical_filename",
        "calibration_selection": "minimum_sha256_per_group",
        "salt": "phase7_gate_fit_cal_v1_seed42",
        "separator": "ascii_backslash_zero", "fit_tiles": 1136,
        "calibration_tiles": 288, "fit_perturbed_instances": 51120,
        "calibration_perturbed_instances": 12960,
    }:
        raise ValueError("Phase 7 split differs from the approved protocol")
    if values["descriptor"] != {
        "stage_number": 2, "channels": 128,
        "tensors": ["symmetric_feature_mean", "absolute_feature_difference",
                    "symmetric_residual_mean", "absolute_residual_difference"],
        "statistics": ["spatial_mean", "spatial_population_std"],
        "dimensions": 1024, "dtype": "float32", "swap_invariant": True,
        "include_pixels": False, "include_action_logits": False,
        "include_metadata": False, "normalization_source": "gate_fit_only",
        "normalization_weighting": "equal_clean_perturbed", "std_floor": 1e-6,
    }:
        raise ValueError("Phase 7 descriptor differs from the approved protocol")
    if values["selector"] != {
        "variants": ["c1", "g1", "g2", "g3"], "input_dimensions": 1024,
        "hidden_dimensions": [128, 64], "output_dimensions": 2,
        "normalization": "layer_norm", "activation": "gelu", "dropout": 0.0,
        "linear_initialization": "xavier_uniform", "linear_bias_initialization": "zero",
        "layer_norm_scale_initialization": "one",
        "layer_norm_bias_initialization": "zero", "parameter_count": 139970,
        "seed": 42,
    }:
        raise ValueError("Phase 7 selector differs from the approved protocol")
    if values["targets"] != {
        "clean_perturbed_weighting": [0.5, 0.5], "g1": ["u_obs"],
        "g2": ["u_obs", "u_clean"], "g3": ["u_obs", "a_obs"],
        "c1": "benefit_sign_classification", "zero_utility_action": "a0",
        "regression_loss": "smooth_l1", "smooth_l1_beta": 1.0,
        "component_weighting": "equal_mean",
        "target_standardization": "weighted_gate_fit_mean_population_std",
        "target_std_floor": 1e-6,
        "c1_class_weighting": "inverse_weighted_frequency_expected_weight_one",
    }:
        raise ValueError("Phase 7 targets differ from the approved protocol")
    training = values["training"]
    expected_training = {
        "device": "cuda", "dtype": "float32", "amp": False,
        "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001,
        "adam_betas": [0.9, 0.999], "scheduler": "cosine",
        "min_learning_rate": 1e-6, "batch_perturbed_records": 256,
        "drop_last": False, "epochs": 100, "attempted_updates_per_epoch": 200,
        "attempted_updates_per_variant": 20000, "grad_clip_norm": 1.0,
        "checkpoint_every_epoch": True, "resume_boundary": "complete_epoch",
        "checkpoint_selection": {
            "regression": "calibration_primary_smooth_l1",
            "classification": "calibration_unweighted_binary_log_loss",
            "tie_break": "earliest_epoch",
        },
    }
    _exact_keys(training, set(expected_training) | {"output_dirs"}, "training")
    if {key: training[key] for key in expected_training} != expected_training:
        raise ValueError("Phase 7 training differs from the approved protocol")
    if (set(training["output_dirs"]) != {"c1", "g1", "g2", "g3"}
            or any(not isinstance(value, str) or not value.strip()
                   for value in training["output_dirs"].values())):
        raise ValueError("Phase 7 training output directories are invalid")
    if values["policies"] != {
        "constants": ["a0_no_correction", "a1_shared_stage2_adapter"],
        "diagnostic_reference": "label_informed_lower_task_loss_actual_action",
        "unconstrained_threshold": 0.0,
        "paired_harm_score": "min_predicted_u_obs_and_u_clean",
        "calibration_quantiles": 1001, "clean_f1_tolerance": 0.0005,
        "clean_recall_tolerance": 0.001,
        "threshold_objective": "calibration_corruption_mf1",
        "threshold_ties": ["higher_clean_recall", "higher_clean_f1",
                           "lower_negative_utility_mass", "larger_threshold"],
    }:
        raise ValueError("Phase 7 policies differ from the approved protocol")
    if values["evaluation"] != {
        "split": "val", "source_pairs": 64, "tiles": 1024, "conditions": 45,
        "perturbed_instances": 46080, "threshold": 0.5,
        "bootstrap_unit": "original_source_pair", "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260922, "confidence_level": 0.95,
        "min_selector_mf1_gain": 0.0005,
        "require_selector_gain_ci_lower_positive": True,
        "clean_f1_tolerance": 0.0005, "clean_recall_tolerance": 0.001,
        "corruption_recall_tolerance": 0.001,
        "min_paired_target_mf1_gain": 0.0005,
        "require_paired_target_gain_ci_lower_positive": True,
        "min_g3_over_g2_mf1_gain": 0.0005,
        "require_g3_over_g2_ci_lower_positive": True,
    }:
        raise ValueError("Phase 7 evaluation differs from the approved protocol")
    if values["boundaries"] != {
        "selector_fit_sources": "train_gate_fit_only",
        "selector_calibration_sources": "train_gate_calibration_only",
        "validation_use": "one_shot_after_preval_seal",
        "held_out_test_access": False, "actor_adapter_frozen": True,
        "local_actions": False, "joint_training": False, "phase8_to_10": False,
    }:
        raise ValueError("Phase 7 boundaries differ from the approved protocol")
    if (base.values != phase56.base.values or base.run_test_after_training
            or base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or base.values["experiment"]["seed"] != 42):
        raise ValueError("Phase 7 base runtime differs from approved Phase 5-6")


def _validate_upstream(values: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    expected_keys = set(_UPSTREAM_PATH_HASH) | set(_UPSTREAM_PATH_HASH.values())
    _exact_keys(values, expected_keys, "upstream")
    for path_key, hash_key in _UPSTREAM_PATH_HASH.items():
        expected = _lower_sha(values[hash_key], f"upstream.{hash_key}")
        path = paths[path_key]
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"Phase 7 upstream artifact changed: {path_key}")
    for status_key, guard_key in (("actor_status", "actor_guard_completion"),
                                  ("adapter_status", "adapter_guard_completion")):
        status = json.loads(paths[status_key].read_text(encoding="utf-8"))
        guard = json.loads(paths[guard_key].read_text(encoding="utf-8"))
        if (status.get("status") != "completed" or status.get("test_evaluated") is not False
                or guard.get("status") != "completed"):
            raise ValueError(f"Incomplete guarded upstream stage: {status_key}")
    phase6 = json.loads(paths["phase6_status"].read_text(encoding="utf-8"))
    decision = json.loads(paths["phase6_gate_decision"].read_text(encoding="utf-8"))
    if (phase6.get("status") != "completed" or phase6.get("test_evaluated") is not False
            or decision.get("decision") != "pass_to_separately_reviewed_phase7_protocol"
            or decision.get("sign_distribution_pass") is not True
            or decision.get("actionable_margin_pass") is not True):
        raise ValueError("Phase 6 did not validly authorize a Phase 7 review")


def _load_approval(config: Phase7Config) -> Mapping[str, Any]:
    approval, _ = _load_yaml(config.paths["approval"])
    _exact_keys(approval, _APPROVAL_KEYS, "phase7 approval")
    if (approval["schema_version"] != 1 or approval["status"] != "approved"
            or approval["approved_for"] != "phase7_whole_tile_selector_v1_only"
            or approval["approval_text"] != "批准，请继续"
            or approval["test_access"] is not False):
        raise ValueError("Phase 7 approval scope or explicit approval text differs")
    for key in ("reviewer", "reviewed_at", "notes"):
        if not isinstance(approval[key], str) or not approval[key].strip():
            raise ValueError(f"approval.{key} must be nonempty")
    bindings = {
        "reviewed_draft_sha256": config.values["reviewed_draft_sha256"],
        "phase7_config_sha256": config.config_sha256,
        "frozen_protocol_document_sha256": config.values[
            "frozen_protocol_document_sha256"],
        "selector_split_manifest_sha256": _sha_file(
            config.paths["selector_split_manifest"]),
    }
    for key, actual in bindings.items():
        if _lower_sha(approval[key], f"approval.{key}") != actual:
            raise ValueError(f"Phase 7 approval binding differs: {key}")
    return MappingProxyType(approval)


def load_phase7_config(path: str | Path, *, require_approval: bool = True) -> Phase7Config:
    source = Path(path).resolve()
    values, raw = _load_yaml(source)
    _exact_keys(values, _TOP_KEYS, "phase7 config")
    if (values["schema_version"] != 1 or values["status"] != "approved_frozen"
            or not isinstance(values["name"], str) or not values["name"].strip()):
        raise ValueError("Phase 7 config must be approved_frozen schema version 1")
    paths = {key: _path(source, values[key], key) for key in _TOP_PATH_KEYS}
    upstream_values = values["upstream"]
    upstream_paths = {key: _path(source, upstream_values[key], f"upstream.{key}")
                      for key in _UPSTREAM_PATH_HASH}
    for key, hash_key in (("base_config", "base_config_sha256"),
                          ("phase56_config", "phase56_config_sha256"),
                          ("frozen_protocol_document", "frozen_protocol_document_sha256")):
        expected = _lower_sha(values[hash_key], hash_key)
        if not paths[key].is_file() or _sha_file(paths[key]) != expected:
            raise ValueError(f"Phase 7 bound file changed: {key}")
    _lower_sha(values["reviewed_draft_sha256"], "reviewed_draft_sha256")
    base = load_config(paths["base_config"])
    phase56 = load_phase56_config(paths["phase56_config"], require_approval=True)
    if phase56.config_sha256 != values["phase56_config_sha256"]:
        raise ValueError("Phase 7 Phase 5-6 binding differs")
    _validate_exact_protocol(values, base, phase56)
    _validate_upstream(upstream_values, upstream_paths)
    provisional = Phase7Config(MappingProxyType(values), source, raw, base, phase56,
                               MappingProxyType(paths), MappingProxyType(upstream_paths), None)
    if not require_approval:
        return provisional
    approval = _load_approval(provisional)
    return Phase7Config(provisional.values, source, raw, base, phase56,
                        provisional.paths, provisional.upstream_paths, approval)


def _selection_hash(salt: str, name: str) -> str:
    return hashlib.sha256((salt + "\\0" + name).encode("utf-8")).hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    foreground = sum(int(row["foreground_pixels"]) for row in rows)
    pixels = sum(int(row["total_pixels"]) for row in rows)
    return {
        "sources": len(rows), "tiles": sum(int(row["tile_count"]) for row in rows),
        "foreground_pixels": foreground, "total_pixels": pixels,
        "foreground_fraction": foreground / pixels,
        "zero_foreground_sources": sum(int(row["foreground_pixels"]) == 0 for row in rows),
    }


def make_selector_split(config: Phase7Config) -> dict[str, Any]:
    partition = json.loads(config.upstream_paths["actor_gate_manifest"].read_text(
        encoding="utf-8"))
    gate_names = set(partition.get("gate_names", []))
    rows = [dict(row) for row in partition.get("sources", [])
            if row.get("name") in gate_names]
    rows.sort(key=lambda row: (int(row["foreground_pixels"]), row["name"]))
    settings = config.values["split"]
    if len(rows) != settings["gate_sources"] or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Frozen gate source rows are incomplete")
    groups = [rows[index:index + settings["group_size"]]
              for index in range(0, len(rows), settings["group_size"])]
    if (len(groups) != settings["calibration_sources"]
            or len(groups[-1]) != settings["final_group_size"]
            or any(len(group) != settings["group_size"] for group in groups[:-1])):
        raise ValueError("Gate sources cannot satisfy the approved Phase 7 grouping")
    calibration_names: set[str] = set()
    for group_index, group in enumerate(groups):
        for row in group:
            row["phase7_group_index"] = group_index
            row["phase7_selection_sha256"] = _selection_hash(settings["salt"], row["name"])
        selected = min(group, key=lambda row: (
            row["phase7_selection_sha256"], row["name"]))
        calibration_names.add(selected["name"])
    for row in rows:
        row["phase7_partition"] = (
            "gate_calibration" if row["name"] in calibration_names else "gate_fit")
    fit = [row for row in rows if row["phase7_partition"] == "gate_fit"]
    calibration = [row for row in rows if row["phase7_partition"] == "gate_calibration"]
    if (len(fit) != settings["fit_sources"]
            or len(calibration) != settings["calibration_sources"]
            or sum(row["tile_count"] for row in fit) != settings["fit_tiles"]
            or sum(row["tile_count"] for row in calibration) != settings["calibration_tiles"]):
        raise ValueError("Phase 7 selector split counts differ from approval")
    expected_calibration = {
        "train_5.png", "train_9.png", "train_33.png", "train_34.png", "train_52.png",
        "train_65.png", "train_70.png", "train_103.png", "train_114.png",
        "train_122.png", "train_152.png", "train_201.png", "train_212.png",
        "train_283.png", "train_306.png", "train_347.png", "train_375.png",
        "train_443.png",
    }
    if calibration_names != expected_calibration:
        raise ValueError("Derived calibration names differ from the approved audit")
    return {
        "schema_version": 1, "status": "frozen",
        "kind": "phase7_selector_source_partition", "split": "train_gate_only",
        "test_evaluated": False, "phase7_config_sha256": config.config_sha256,
        "reviewed_draft_sha256": config.values["reviewed_draft_sha256"],
        "frozen_protocol_document_sha256": config.values[
            "frozen_protocol_document_sha256"],
        "upstream_actor_gate_manifest_sha256": config.values["upstream"][
            "actor_gate_manifest_sha256"],
        "algorithm": {
            "ordering": settings["ordering"], "group_size": settings["group_size"],
            "final_group_size": settings["final_group_size"],
            "calibration_selection": settings["calibration_selection"],
            "salt": settings["salt"],
            "hash_payload": "UTF8(salt + ASCII_BACKSLASH_ZERO + canonical_filename)",
        },
        "sources": rows,
        "fit_names": sorted(row["name"] for row in fit),
        "calibration_names": sorted(row["name"] for row in calibration),
        "summary": {"gate_fit": _summary(fit),
                    "gate_calibration": _summary(calibration)},
    }


def phase7_protocol_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/phase5_protocol.py", "pcdc/phase7_protocol.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


def prepare_phase7_protocol(config: Phase7Config) -> dict[str, Any]:
    if config.approval is not None:
        raise ValueError("Phase 7 preparation expects config loaded without approval")
    output = config.paths["protocol_output_dir"]
    manifest_path = config.paths["selector_split_manifest"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 protocol: {output}")
    if manifest_path.parent != output:
        raise ValueError("Phase 7 selector split manifest must live in protocol output")
    manifest = make_selector_split(config)
    source_files = phase7_protocol_source_files()
    source_hashes = {name: file_sha256(path) for name, path in source_files.items()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "phase7_config_source.yaml").write_bytes(config.raw)
        shutil.copyfile(config.paths["frozen_protocol_document"],
                        output / "frozen_protocol_document.md")
        write_json(manifest_path, manifest)
        for name, path in source_files.items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_sha256(target) != source_hashes[name]:
                raise ValueError(f"Phase 7 protocol source changed while copying: {name}")
        write_json(output / "source_sha256.json", source_hashes)
        result = {
            "status": "prepared_awaiting_bound_approval_sidecar",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase7_config_sha256": config.config_sha256,
            "reviewed_draft_sha256": config.values["reviewed_draft_sha256"],
            "frozen_protocol_document_sha256": _sha_file(
                config.paths["frozen_protocol_document"]),
            "selector_split_manifest_sha256": _sha_file(manifest_path),
            "source_sha256": source_hashes,
            "fit_sources": len(manifest["fit_names"]),
            "calibration_sources": len(manifest["calibration_names"]),
            "fit_tiles": manifest["summary"]["gate_fit"]["tiles"],
            "calibration_tiles": manifest["summary"]["gate_calibration"]["tiles"],
            "dataset_splits_opened": [], "model_initialized": False,
            "training_started": False, "validation_evaluated": False,
            "test_evaluated": False,
        }
        write_json(output / "status.json", result)
        return result
    except BaseException as error:
        if output.exists():
            write_json(output / "status.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
                "dataset_splits_opened": [], "model_initialized": False,
                "training_started": False, "validation_evaluated": False,
                "test_evaluated": False,
            }, overwrite=True)
        raise


def load_selector_split(config: Phase7Config) -> dict[str, Any]:
    if config.approval is None:
        raise ValueError("Frozen Phase 7 split requires bound approval")
    manifest = json.loads(config.paths["selector_split_manifest"].read_text(encoding="utf-8"))
    expected = make_selector_split(config)
    if canonical(manifest) != canonical(expected):
        raise ValueError("Phase 7 selector split differs from deterministic approval")
    return manifest


def preflight(config: Phase7Config) -> dict[str, Any]:
    manifest = load_selector_split(config)
    _validate_upstream(config.values["upstream"], config.upstream_paths)
    return {
        "status": "phase7_metadata_preflight_passed",
        "kind": "metadata_only_no_data_model_or_device",
        "phase7_fingerprint": config.fingerprint,
        "approval_status": config.approval["status"],
        "fit_sources": len(manifest["fit_names"]),
        "calibration_sources": len(manifest["calibration_names"]),
        "dataset_opened": False, "model_initialized": False,
        "training_started": False, "validation_evaluated": False,
        "sensor_queried": False, "test_evaluated": False,
    }
