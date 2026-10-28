"""Read-only validation benchmark binding, review gate and raw-tile replay.

The preparation's files are immutable. A separate acceptance record permits
scoring; neither preflight nor clean-only compatibility checks approve severity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import features
import torch
from torch.utils.data import Dataset
import yaml

from .config import UniqueKeyLoader, protocol_signature
from .corruption_protocol import (CorruptionProtocol, apply_instance, canonical,
                                  load_protocol, sample_ids, validation_instances)
from .data import LEVIRDataset, file_sha256
from .engine import check_checkpoint, read_checkpoint, source_files
from .perturbations import make_injection_mask


_JOB_KEYS = {"schema_version", "name", "protocol_config", "manifest", "manifest_sha256",
             "protocol_sha256", "instances_sha256", "checkpoint_sha256",
             "preparation_environment_sha256", "acceptance", "output_dir"}
_REVIEW_KEYS = {"schema_version", "status", "approved_for", "protocol_sha256",
                "manifest_sha256", "instances_sha256", "checkpoint_sha256",
                "visual_review", "reviewer", "notes"}
_BINDING_KEYS = ("protocol_sha256", "manifest_sha256", "instances_sha256", "checkpoint_sha256")


class ReviewRequired(ValueError):
    """Candidate artifacts exist but real corruption scoring is not approved."""


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Invalid {label} schema; expected {sorted(expected)}")


def _sha(value: Any, label: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader)


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


@dataclass(frozen=True)
class EvaluationConfig:
    values: dict[str, Any]
    source: Path

    def path(self, key: str) -> Path:
        path = Path(self.values[key])
        return (path if path.is_absolute() else self.source.parent / path).resolve()


@dataclass(frozen=True)
class FrozenValidation:
    job: EvaluationConfig
    protocol: CorruptionProtocol
    manifest: dict[str, Any]
    manifest_path: Path
    instances: list[dict[str, Any]]
    fingerprints: dict[str, Any]
    approval: dict[str, Any] | None


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    source = Path(path).resolve()
    values = _yaml(source)
    _keys(values, _JOB_KEYS, "evaluation config")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only evaluation schema_version 1 is supported")
    for key in ("name", "protocol_config", "manifest", "acceptance", "output_dir"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in (*_BINDING_KEYS, "preparation_environment_sha256"):
        _sha(values[key], key)
    return EvaluationConfig(values, source)


def _approval(job: EvaluationConfig, *, required: bool) -> dict[str, Any] | None:
    path = job.path("acceptance")
    if not path.is_file():
        if required:
            raise ReviewRequired("Severity acceptance is missing; preflight/clean-only remain available")
        return None
    review = _yaml(path)
    _keys(review, _REVIEW_KEYS, "severity acceptance")
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise ValueError("Only acceptance schema_version 1 is supported")
    if review["approved_for"] != "validation_only":
        raise ValueError("This gate never authorizes training or held-out test")
    for key in _BINDING_KEYS:
        if review[key] != job.values[key]:
            raise ValueError(f"Acceptance is bound to a different {key}")
    if review["status"] not in ("pending", "approved", "engineering_only"):
        raise ValueError("Unknown acceptance status")
    if review["visual_review"] not in ("not_completed", "passed", "not_applicable_synthetic"):
        raise ValueError("Unknown visual review status")
    if review["reviewer"] is not None and not isinstance(review["reviewer"], str):
        raise ValueError("Reviewer must be a string or null")
    if not isinstance(review["notes"], str):
        raise ValueError("Acceptance notes must be text")
    if required:
        if review["status"] == "pending":
            raise ReviewRequired("Visual severity review is pending; real corruption scoring is blocked")
        if not review["reviewer"] or not review["reviewer"].strip() or not review["notes"].strip():
            raise ReviewRequired("Approved records require an identified reviewer and notes")
        expected = "passed" if review["status"] == "approved" else "not_applicable_synthetic"
        if review["visual_review"] != expected:
            raise ReviewRequired("Acceptance status does not match the visual-review evidence")
    return review


def _inventory_ids(inventory: dict[str, Any]) -> list[str]:
    _keys(inventory, {"split", "tile_size", "entries", "source_pairs", "tiles", "inventory_sha256"},
          "validation inventory")
    if inventory["split"] != "val":
        raise ValueError("Only a frozen validation split may be scored")
    tile = inventory["tile_size"]
    if type(tile) is not int or tile < 32:
        raise ValueError("Invalid frozen tile size")
    entries = inventory["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Validation inventory is empty")
    ids, names = [], set()
    for entry in entries:
        _keys(entry, {"name", "size", "bytes"}, "inventory entry")
        name, size, sizes = entry["name"], entry["size"], entry["bytes"]
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or name in names):
            raise ValueError("Invalid or duplicate source filename in inventory")
        names.add(name)
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value < tile or value % tile for value in size)):
            raise ValueError("Invalid source dimensions or dropped-border tiling")
        if (not isinstance(sizes, list) or len(sizes) != 3
                or any(type(value) is not int or value < 1 for value in sizes)):
            raise ValueError("Invalid source file byte counts")
        width, height = size
        ids.extend(f"val/{name}@x{x}_y{y}" for y in range(0, height, tile) for x in range(0, width, tile))
    payload = {key: inventory[key] for key in ("split", "tile_size", "entries")}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if (inventory["inventory_sha256"] != expected or inventory["source_pairs"] != len(entries)
            or inventory["tiles"] != len(ids)):
        raise ValueError("Frozen inventory digest/counts disagree")
    return ids


def load_bundle(job: EvaluationConfig, require_review: bool = True) -> FrozenValidation:
    if type(require_review) is not bool:
        raise ValueError("require_review must be boolean")
    # Missing/pending approval fails before data, checkpoint or CUDA access.
    review = _approval(job, required=require_review)
    review_hash = file_sha256(job.path("acceptance")) if review else None
    # Check the parsed record against the bytes whose hash will bind the run.
    if review is not None and _yaml(job.path("acceptance")) != review:
        raise ValueError("Acceptance changed while it was being read")
    job_hash = file_sha256(job.source)
    if _yaml(job.source) != job.values:
        raise ValueError("Evaluation config changed after it was loaded")
    manifest_path = job.path("manifest")
    if file_sha256(manifest_path) != job.values["manifest_sha256"]:
        raise ValueError("Frozen manifest SHA256 differs from the evaluation job")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("split") != "val":
        raise ValueError("Only schema v1 validation manifests are supported")
    if manifest.get("status") != "candidate_pending_review":
        raise ValueError("Do not edit the frozen candidate status; use a separate acceptance record")
    protocol_config_hash = file_sha256(job.path("protocol_config"))
    protocol = load_protocol(job.path("protocol_config"))
    if file_sha256(protocol.source) != protocol_config_hash:
        raise ValueError("Resolved protocol YAML changed while it was being read")
    if review and review["status"] == "engineering_only" and protocol.base.values["experiment"]["kind"] != "smoke":
        raise ReviewRequired("Engineering-only approval cannot authorize research data or checkpoints")
    expected_protocol = job.values["protocol_sha256"]
    if protocol.fingerprint != expected_protocol or manifest["protocol_sha256"] != expected_protocol:
        raise ValueError("Frozen protocol fingerprint differs")
    if manifest["base_protocol_signature"] != protocol_signature(protocol.base.values):
        raise ValueError("Base inference/preprocessing/metric protocol changed")
    if file_sha256(protocol.base.source) != manifest["archived_base_config_sha256"]:
        raise ValueError("Archived base YAML content changed")
    if file_sha256(manifest_path.parent / "candidate_source.yaml") != manifest["protocol_file_sha256"]:
        raise ValueError("Original candidate YAML snapshot changed")
    checkpoint_sha = job.values["checkpoint_sha256"]
    if (manifest["reference_checkpoint_sha256"] != checkpoint_sha
            or protocol.values["reference_checkpoint_sha256"] != checkpoint_sha):
        raise ValueError("Reference checkpoint identity differs")
    prepared_sources = manifest["source_sha256"]
    if not isinstance(prepared_sources, dict) or not prepared_sources:
        raise ValueError("Missing prepared source identity")
    if _json(manifest_path.parent / "source_sha256.json") != prepared_sources:
        raise ValueError("Prepared source hash index changed")
    current_sources = source_files()
    for name, checksum in prepared_sources.items():
        _sha(checksum, f"source {name}")
        if name not in current_sources:
            raise ValueError(f"Frozen source file is unavailable: {name}")
        # New evaluation modules are allowed; all preparation-time files stay fixed.
        if file_sha256(current_sources[name]) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
        if file_sha256(manifest_path.parent / "source" / name) != checksum:
            raise ValueError(f"Prepared source snapshot changed: {name}")
    env_path = manifest_path.parent / "environment.json"
    if file_sha256(env_path) != job.values["preparation_environment_sha256"]:
        raise ValueError("Preparation environment record changed")
    environment = _json(env_path)
    installed = {"numpy": np.__version__, "pillow": PIL.__version__,
                 "jpeg_codec": features.version_codec("jpg"),
                 "libjpeg_turbo": features.version_feature("libjpeg_turbo")}
    if any(environment.get(key) != value for key, value in installed.items()):
        raise ValueError("Corruption runtime/codec versions differ from preparation")
    ids = _inventory_ids(manifest["inventory"])
    filename = manifest["instance_file"]
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Instance file must be an adjacent filename")
    instance_path = manifest_path.parent / filename
    if (file_sha256(instance_path) != job.values["instances_sha256"]
            or manifest["instance_file_sha256"] != job.values["instances_sha256"]):
        raise ValueError("Frozen instance JSONL content changed")
    instances, counts, strata = [], Counter(), Counter()
    with instance_path.open(encoding="utf-8") as stream:
        for position, (line, expected) in enumerate(itertools.zip_longest(
                stream, validation_instances(protocol, ids)), start=1):
            if line is None or expected is None or not line.strip():
                raise ValueError(f"Missing/extra/blank instance at line {position}")
            actual = json.loads(line)
            if canonical(actual) != canonical(expected):
                raise ValueError(f"Frozen replay differs at line {position}; no regenerated replacement is allowed")
            instances.append(actual)
            counts[actual["condition_id"]] += 1
            strata["|".join((actual["condition_id"], actual["direction"], actual["target_date"],
                              actual["spatial_mode"]))] += 1
    if (dict(counts) != manifest["condition_counts"] or dict(strata) != manifest["stratum_counts"]
            or len(instances) != manifest["instance_count"] or len(counts) != 45):
        raise ValueError("Condition/stratum/instance counts differ from frozen manifest")
    fingerprints = {key: job.values[key] for key in (*_BINDING_KEYS, "preparation_environment_sha256")}
    if (file_sha256(job.source) != job_hash
            or (review is not None and file_sha256(job.path("acceptance")) != review_hash)):
        raise ValueError("Job or acceptance changed during frozen bundle validation")
    fingerprints.update(job_config_sha256=job_hash,
                        protocol_config_sha256=protocol_config_hash,
                        archived_base_config_sha256=manifest["archived_base_config_sha256"],
                        preparation_source_sha256=prepared_sources,
                        acceptance_sha256=review_hash)
    return FrozenValidation(job, protocol, manifest, manifest_path, instances, fingerprints, review)


def build_validation_dataset(bundle: FrozenValidation) -> tuple[LEVIRDataset, dict[str, Any]]:
    config = bundle.protocol.base
    if config.run_test_after_training:
        raise ValueError("Frozen evaluation must retain development-only test isolation")
    path = bundle.protocol.path("reference_checkpoint")
    if file_sha256(path) != bundle.job.values["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint content changed")
    checkpoint = read_checkpoint(path)
    check_checkpoint(config, checkpoint)
    if "val" not in checkpoint["manifests"] or "val" not in checkpoint["file_sha256"]:
        raise ValueError("Checkpoint has no audited validation split")
    dataset = LEVIRDataset(config, "val")
    inventory = dataset.manifest()
    if inventory != bundle.manifest["inventory"] or inventory != checkpoint["manifests"]["val"]:
        raise ValueError("Validation inventory differs from preparation/checkpoint")
    expected_content = checkpoint["file_sha256"]["val"]
    if expected_content != bundle.manifest["file_sha256"]:
        raise ValueError("Prepared validation content does not match checkpoint audit")
    for record in dataset.records:
        expected = expected_content[record.name]
        actual = {key: file_sha256(getattr(record, key)) for key in ("image_a", "image_b", "mask")}
        if actual != expected:
            raise ValueError(f"Validation source content changed: {record.name}")
    return dataset, checkpoint


class FrozenInstanceDataset(Dataset):
    """One immutable row per tile; metadata never enters the detection model."""

    split = "val"

    def __init__(self, dataset: LEVIRDataset, instances: list[dict[str, Any]]):
        if dataset.split != "val" or dataset.pair_transform is not None:
            raise ValueError("Frozen replay requires an untransformed validation dataset")
        self.dataset = dataset
        self.instances = list(instances)
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids(dataset))}
        if not instances:
            raise ValueError("Frozen instance dataset is empty")
        self.indices = []
        seen = set()
        for instance in self.instances:
            sample_id = instance["sample_id"]
            if sample_id not in lookup:
                raise ValueError(f"Unknown or out-of-bounds validation tile: {sample_id}")
            key = (instance["condition_id"], sample_id)
            if key in seen:
                raise ValueError("Duplicate condition/tile instance")
            seen.add(key)
            self.indices.append(lookup[sample_id])

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self.instances[index]
        record_index, x, y = self.dataset.tiles[self.indices[index]]
        record = self.dataset.records[record_index]
        tile = self.dataset.tile_size
        pair = self.dataset.read_pair(record, (x, y, x + tile, y + tile))
        changed = apply_instance(pair, instance)

        def normalize(image: np.ndarray) -> torch.Tensor:
            value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
            return (value - self.dataset.mean) / self.dataset.std

        # Large uint64 seeds and heterogeneous mask dictionaries stay inside a
        # JSON string, never in default_collate's signed integer/nested-map path.
        result = {"image_a": normalize(changed.image_a), "image_b": normalize(changed.image_b),
                  "mask": torch.from_numpy(np.ascontiguousarray(changed.mask)).float().unsqueeze(0),
                  "sample_id": instance["sample_id"], "instance_json": canonical(instance),
                  "injection_support": torch.from_numpy(make_injection_mask(
                      changed.mask.shape, instance["injection_mask"]) > 0).unsqueeze(0)}
        result.update({key: instance[key] for key in
                       ("condition_id", "kind", "level", "direction", "target_date", "spatial_mode")})
        return result
