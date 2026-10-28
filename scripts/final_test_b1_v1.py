"""One-shot, sealed B1 seed-42 held-out test. No real test access in preflight/smoke."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from pcdc.config import Config, UniqueKeyLoader, protocol_signature, validate_config
from pcdc.corruption_protocol import KINDS, apply_instance, canonical, keyed_seed, sample_ids
from pcdc.data import LEVIRDataset, RawPair, file_sha256, make_loader
from pcdc.engine import check_checkpoint, move_model, read_checkpoint
from pcdc.evaluate_candidate import (_load_metadata, _validate_checkpoints,
                                     load_candidate_evaluation_config)
from pcdc.evaluate_corruptions import (COUNT_KEYS, ConfusionAccumulator, counts_report,
                                       per_sample_confusions, summarize_conditions)
from pcdc.model import build_model
from pcdc.perturbations import make_injection_mask
from pcdc.robust_baselines import load_robust_config
from pcdc.runtime import environment_record, setup_runtime, write_json
from pcdc.smoke import create_fixture
from scripts.evaluate_guarded import (EvaluationTemperatureGuard, GuardError,
                                      _atomic_json, _sha, _utc_now,
                                      load_guard_config, read_temperature)


ROOT = Path(__file__).resolve().parents[1]
_PATHS = (
    "protocol_document", "r1_report", "r1_completion_audit",
    "candidate_evaluation_config", "training_config", "best_checkpoint",
    "last_checkpoint", "training_status", "training_guard_completion",
    "base_config", "corruption_config", "test_acceptance", "guard_config",
)
_JOB_KEYS = {"schema_version", "name", "status", "approval", "pretest_seal",
             "output_dir", "expected_test_pairs", "expected_test_tiles",
             "expected_conditions", "test_seed", "bootstrap_seed",
             "bootstrap_replicates", *_PATHS, *(name + "_sha256" for name in _PATHS)}
_SOURCES = (
    "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
    "pcdc/engine.py", "pcdc/evaluate_candidate.py", "pcdc/evaluate_corruptions.py",
    "pcdc/frozen_validation.py", "pcdc/model.py", "pcdc/perturbations.py",
    "pcdc/robust_baselines.py", "pcdc/runtime.py", "pcdc/smoke.py", "scripts/evaluate_guarded.py",
    "scripts/temperature_guard.py", "scripts/final_test_b1_v1.py",
)
_APPROVAL_KEYS = {"schema_version", "status", "approved_for", "reviewer", "notes",
                  "test_accessed", "protocol_document_sha256", "job_sha256",
                  "test_acceptance_sha256", "checkpoint_sha256",
                  "guard_config_sha256", "source_sha256"}
_SEAL_KEYS = {"schema_version", "status", "test_accessed", "test_evaluated",
              "protocol_document_sha256", "job_sha256", "approval_sha256",
              "checkpoint_sha256", "guard_config_sha256", "source_sha256",
              "unit_test_evidence", "unit_test_evidence_sha256", "smoke_evidence",
              "smoke_evidence_sha256", "preflight_evidence",
              "preflight_evidence_sha256"}
_ID = re.compile(r"^test/([^/\\]+)@x(0|[1-9][0-9]*)_y(0|[1-9][0-9]*)$")


def _sha(value: Any) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Expected lowercase SHA256")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return result


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} keys differ: {set(value) if isinstance(value, dict) else type(value)}")


@dataclass(frozen=True)
class Job:
    source: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        value = Path(self.values[key])
        return (value if value.is_absolute() else self.source.parent / value).resolve()

    @property
    def output(self) -> Path:
        return self.path("output_dir")


def load_job(path: str | Path) -> Job:
    source = Path(path).resolve()
    values = _yaml(source)
    _keys(values, _JOB_KEYS, "final test job")
    if (values["schema_version"] != 1 or values["name"] != "levir_final_test_b1_seed42_v1"
            or values["status"] != "frozen_pending_pretest_seal"):
        raise ValueError("Unknown final-test schema/name/status")
    for key in (*_PATHS, "approval", "pretest_seal", "output_dir"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"Invalid path: {key}")
    for key in _PATHS:
        _sha(values[key + "_sha256"])
        resolved = Job(source, values).path(key)
        if file_sha256(resolved) != values[key + "_sha256"]:
            raise ValueError(f"Frozen input hash differs: {key}")
    expected = {"expected_test_pairs": 128, "expected_test_tiles": 2048,
                "expected_conditions": 45, "test_seed": 310944,
                "bootstrap_seed": 20260926, "bootstrap_replicates": 5000}
    if any(type(values[key]) is not int or values[key] != number
           for key, number in expected.items()):
        raise ValueError("Final-test counts/seeds/replicates differ from protocol")
    job = Job(source, values)
    acceptance = _yaml(job.path("test_acceptance"))
    if (acceptance.get("status") != "approved"
            or acceptance.get("approved_for") != "final_test_only"
            or acceptance.get("test_seed") != 310944
            or acceptance.get("condition_count") != 45
            or acceptance.get("corruption_config_sha256") != values["corruption_config_sha256"]
            or not str(acceptance.get("reviewer", "")).strip()
            or not str(acceptance.get("notes", "")).strip()):
        raise ValueError("Independent test-only acceptance is incomplete")
    visual = job.path("test_acceptance").parent / acceptance["original_visual_acceptance"]
    if (file_sha256(visual) != _sha(acceptance["original_visual_acceptance_sha256"])
            or _yaml(visual).get("approved_for") != "validation_only"):
        raise ValueError("Original train-visual evidence/val-only boundary differs")
    return job


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(ROOT / name) for name in _SOURCES}


def _metadata(job: Job) -> tuple[Config, dict[str, Any]]:
    report = _json(job.path("r1_completion_audit"))
    if (report.get("status") != "completed_with_peak_gpu_memory_reporting_gap"
            or report.get("analysis_verification", {}).get("selection_decision") != "retain_b1"
            or report.get("held_out_test_closed") is not True
            or report.get("test_evaluated") is not False):
        raise ValueError("R1 evidence does not authorize this final B1 candidate")
    candidate = load_candidate_evaluation_config(job.path("candidate_evaluation_config"))
    metadata = _load_metadata(candidate, require_review=True)
    best, last = read_checkpoint(job.path("best_checkpoint")), read_checkpoint(job.path("last_checkpoint"))
    _validate_checkpoints(candidate, metadata, best, last)
    robust = load_robust_config(job.path("training_config"))
    base = robust.base
    if (robust.mode != "b1" or base.values["experiment"]["seed"] != 42
            or robust.paths["base_config"] != job.path("base_config")
            or robust.paths["protocol_config"] != job.path("corruption_config")
            or candidate.path("candidate_best_checkpoint") != job.path("best_checkpoint")
            or candidate.path("candidate_last_checkpoint") != job.path("last_checkpoint")
            or candidate.path("candidate_status") != job.path("training_status")
            or candidate.path("candidate_guard_completion") != job.path("training_guard_completion")
            or best["epoch"] != 68 or best["best_val_f1"] != 0.9060728685624768
            or best.get("robust_mode") != "b1"
            or set(best.get("manifests", {})) != {"train", "val"}):
        raise ValueError("Final checkpoint is not the pre-existing B1 seed-42 incumbent")
    if metadata.candidate_status.get("test_evaluated") is not False:
        raise ValueError("Training metadata unexpectedly reports test access")
    return base, best


def _test_config(base: Config, checkpoint: Mapping[str, Any]) -> Config:
    if base.run_test_after_training:
        raise ValueError("Archived development config unexpectedly allows test")
    values = copy.deepcopy(base.values)
    values["evaluation"]["run_test_after_training"] = True
    validate_config(values)
    config = Config(values, base.source)
    check_checkpoint(config, dict(checkpoint))
    if protocol_signature(values) != protocol_signature(base.values):
        raise ValueError("Test lifecycle flag changed inference semantics")
    return config


def _approval(job: Job) -> dict[str, Any]:
    approval = _yaml(job.path("approval"))
    _keys(approval, _APPROVAL_KEYS, "final-test approval")
    if (approval["schema_version"] != 1 or approval["status"] != "approved_before_test"
            or approval["approved_for"] != "single_b1_seed42_final_test_only"
            or approval["test_accessed"] is not False
            or not str(approval["reviewer"]).strip() or not str(approval["notes"]).strip()):
        raise ValueError("Final-test approval is not valid")
    expected = {"protocol_document_sha256": job.values["protocol_document_sha256"],
                "job_sha256": file_sha256(job.source),
                "test_acceptance_sha256": job.values["test_acceptance_sha256"],
                "checkpoint_sha256": job.values["best_checkpoint_sha256"],
                "guard_config_sha256": job.values["guard_config_sha256"]}
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Final-test approval binds another input")
    if approval["source_sha256"] != source_hashes():
        raise ValueError("Frozen final-test source differs from approval")
    return approval


def _seal(job: Job, approval: Mapping[str, Any]) -> dict[str, Any]:
    seal = _json(job.path("pretest_seal"))
    _keys(seal, _SEAL_KEYS, "pretest seal")
    if (seal["schema_version"] != 1 or seal["status"] != "sealed_before_test"
            or seal["test_accessed"] is not False or seal["test_evaluated"] is not False):
        raise ValueError("Seal was not created before held-out test access")
    expected = {"protocol_document_sha256": job.values["protocol_document_sha256"],
                "job_sha256": file_sha256(job.source),
                "approval_sha256": file_sha256(job.path("approval")),
                "checkpoint_sha256": job.values["best_checkpoint_sha256"],
                "guard_config_sha256": job.values["guard_config_sha256"],
                "source_sha256": approval["source_sha256"]}
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError("Pretest seal hash binding differs")
    for key, status in (("unit_test_evidence", "passed"),
                        ("smoke_evidence", "synthetic_cuda_smoke_passed"),
                        ("preflight_evidence", "final_test_preflight_passed")):
        path = Path(seal[key])
        path = (path if path.is_absolute() else job.path("pretest_seal").parent / path).resolve()
        if file_sha256(path) != _sha(seal[key + "_sha256"]):
            raise ValueError(f"Pretest evidence hash differs: {key}")
        evidence = _json(path)
        if (evidence.get("status") != status or evidence.get("test_accessed") is not False
                or evidence.get("job_sha256") != file_sha256(job.source)
                or evidence.get("source_sha256") != approval["source_sha256"]
                or evidence.get("checkpoint_sha256") != job.values["best_checkpoint_sha256"]):
            raise ValueError(f"Pretest evidence is not a passing closed-test record: {key}")
        if key == "unit_test_evidence":
            log = Path(evidence["log"])
            log = (log if log.is_absolute() else path.parent / log).resolve()
            if (type(evidence.get("tests_run")) is not int or evidence["tests_run"] < 195
                    or file_sha256(log) != _sha(evidence["log_sha256"])
                    or file_sha256(ROOT / "tests/test_final_test_b1_v1.py")
                    != _sha(evidence["final_test_unit_source_sha256"])):
                raise ValueError("Complete unit-test evidence is missing")
            lines = log.read_text(encoding="utf-8-sig").strip().splitlines()
            if (not lines or lines[-1] != "OK"
                    or not any(line.startswith(f"Ran {evidence['tests_run']} tests in ")
                               for line in lines[-4:])):
                raise ValueError("Complete unit-test log did not end in the declared pass")
        if key == "smoke_evidence" and (
                evidence.get("research_result") is not False
                or evidence.get("dataset") != "temporary_synthetic_fixture_only"
                or evidence.get("synthetic_data_audit_passed") is not True
                or evidence.get("hardware", {}).get("device") != "cuda:0"):
            raise ValueError("CUDA synthetic-only smoke evidence differs")
        if key == "preflight_evidence" and (
                evidence.get("dataset_opened") is not False
                or evidence.get("hardware", {}).get("device") != "cuda:0"):
            raise ValueError("Closed-test CUDA/guard preflight evidence differs")
    return seal


def hardware_preflight(job: Job, config: Config) -> dict[str, Any]:
    guard = load_guard_config(job.path("guard_config"))
    if (guard.config_sha256 != job.values["guard_config_sha256"]
            or config.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard.values["gpu_index"] != 0):
        raise ValueError("Real CUDA device/guard identity differs")
    guard.verify()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Final test requires the sole real CUDA GPU")
    temperature = read_temperature(guard.values)
    if temperature >= guard.values["confirmed_pause_c"]:
        raise GuardError("Preflight GPU temperature is above the confirmed-pause threshold")
    return {"device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0),
            "gpu_uuid": guard.values["gpu_uuid"], "temperature_c": temperature,
            "guard_config_sha256": guard.config_sha256,
            "guard_source_sha256": dict(guard.source_sha256)}


def preflight(job: Job, *, approval: bool, hardware: bool) -> dict[str, Any]:
    base, best = _metadata(job)
    config = _test_config(base, best)
    if approval:
        _approval(job)
    hardware_result = hardware_preflight(job, config) if hardware else None
    return {"status": "final_test_preflight_passed" if hardware else "metadata_preflight_passed",
            "job_sha256": file_sha256(job.source),
            "checkpoint_sha256": job.values["best_checkpoint_sha256"],
            "source_sha256": source_hashes(), "hardware": hardware_result,
            "dataset_opened": False, "model_inference_run": False,
            "test_accessed": False, "test_evaluated": False}


def test_instances(protocol: Any, ids: list[str], seed: int) -> list[dict[str, Any]]:
    """New test stream: exactly the predeclared ID-only, 45-cell recipe."""
    if (seed != 310944 or not ids or len(ids) != len(set(ids))
            or any(_ID.fullmatch(value) is None for value in ids)):
        raise ValueError("Expected unique test IDs and reserved test seed")
    result: list[dict[str, Any]] = []
    for kind in KINDS:
        directions = sorted(protocol.values["corruptions"][kind])
        strata = list(itertools.product(directions, ("A", "B"), ("global", "local")))
        order = sorted(ids, key=lambda value: (keyed_seed(seed, kind, value), value))
        assigned = {value: strata[index % len(strata)] for index, value in enumerate(order)}
        for level in range(1, 6):
            for sample_id in ids:
                direction, date, spatial = assigned[sample_id]
                instance_seed = keyed_seed(protocol.values["name"], seed, "test",
                                           sample_id, kind, direction, date, spatial, 0, 0)
                if spatial == "global":
                    mask = {"mode": "global"}
                else:
                    rng = np.random.Generator(np.random.PCG64(keyed_seed(instance_seed, "mask")))
                    ranges = protocol.values["local_mask"]
                    mask = {"mode": "ellipse", "feather": ranges["feather"],
                            **{key: float(rng.uniform(*ranges[key + "_range"]))
                               for key in ("center_x", "center_y", "radius_x", "radius_y")}}
                result.append({"sample_id": sample_id, "condition_id": f"{kind}_l{level}",
                               "stream": "test", "epoch": 0, "view": 0, "kind": kind,
                               "level": level, "direction": direction, "target_date": date,
                               "spatial_mode": spatial, "seed": instance_seed,
                               "parameters": copy.deepcopy(protocol.values["corruptions"][kind]
                                                           [direction][level - 1]),
                               "injection_mask": mask})
        assigned_counts = Counter(assigned.values())
        if max(assigned_counts.values()) - min(assigned_counts.values()) > 1:
            raise ValueError("Test nuisance strata are not balanced to within one tile")
    return result


class TestInstanceDataset(Dataset):
    split = "test"

    def __init__(self, dataset: LEVIRDataset, instances: list[dict[str, Any]]):
        if dataset.split != "test" or dataset.pair_transform is not None or not instances:
            raise ValueError("Final-test replay requires nonempty untransformed test data")
        self.dataset = dataset
        self.instances = instances
        lookup = {value: index for index, value in enumerate(sample_ids(dataset))}
        self.indices = []
        seen: set[str] = set()
        for instance in instances:
            sample_id = instance["sample_id"]
            if (_ID.fullmatch(sample_id) is None or sample_id not in lookup
                    or sample_id in seen or instance["stream"] != "test"):
                raise ValueError("Duplicate, missing, or non-test instance ID")
            seen.add(sample_id)
            self.indices.append(lookup[sample_id])

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self.instances[index]
        record_index, x, y = self.dataset.tiles[self.indices[index]]
        record = self.dataset.records[record_index]
        tile = self.dataset.tile_size
        changed = apply_instance(self.dataset.read_pair(record, (x, y, x + tile, y + tile)), instance)

        def normalize(image: np.ndarray) -> torch.Tensor:
            value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
            return (value - self.dataset.mean) / self.dataset.std

        result = {"image_a": normalize(changed.image_a),
                  "image_b": normalize(changed.image_b),
                  "mask": torch.from_numpy(np.ascontiguousarray(changed.mask)).float().unsqueeze(0),
                  "sample_id": instance["sample_id"], "instance_json": canonical(instance),
                  "injection_support": torch.from_numpy(make_injection_mask(
                      changed.mask.shape, instance["injection_mask"]) > 0).unsqueeze(0)}
        result.update({key: instance[key] for key in
                       ("condition_id", "kind", "level", "direction", "target_date", "spatial_mode")})
        return result


@torch.no_grad()
def score_condition(model: torch.nn.Module, loader: DataLoader, config: Config,
                    device: torch.device, *, expected_ids: list[str],
                    condition_id: str,
                    guard: Callable[[], None] | None = None) -> dict[str, Any]:
    """Pixel-pooled final-test scorer; refuses val IDs and mixed/partial cells."""
    if condition_id != "clean" and not re.fullmatch(r"[a-z_]+_l[1-5]", condition_id):
        raise ValueError("Invalid declared condition")
    model.eval()
    started = time.monotonic()
    regional = condition_id != "clean"
    total = ConfusionAccumulator(regional=regional)
    sources: dict[str, ConfusionAccumulator] = {}
    strata: dict[str, ConfusionAccumulator] = {}
    seen: set[str] = set()
    memory_format = (torch.channels_last if config.values["runtime"]["memory_format"] == "channels_last"
                     else torch.contiguous_format)
    for batch in loader:
        if guard is not None:
            guard()
        n = int(batch["mask"].shape[0])
        ids = batch["sample_id"]
        if not isinstance(ids, (list, tuple)) or len(ids) != n:
            raise ValueError("Invalid sample IDs in test batch")
        for index, sample_id in enumerate(ids):
            if _ID.fullmatch(sample_id) is None or sample_id in seen:
                raise ValueError("Malformed/duplicate final-test sample ID")
            seen.add(sample_id)
            if regional:
                if (batch["condition_id"][index] != condition_id
                        or batch["kind"][index] != condition_id.rsplit("_l", 1)[0]
                        or int(batch["level"][index]) != int(condition_id.rsplit("_l", 1)[1])
                        or batch["target_date"][index] not in ("A", "B")
                        or batch["spatial_mode"][index] not in ("global", "local")):
                    raise ValueError("Mixed or malformed final-test condition metadata")
        a = batch["image_a"].to(device=device, non_blocking=True, memory_format=memory_format)
        b = batch["image_b"].to(device=device, non_blocking=True, memory_format=memory_format)
        target = batch["mask"].to(device=device, non_blocking=True)
        support = batch["injection_support"].to(device=device, non_blocking=True) if regional else None
        with torch.autocast(device_type=device.type, enabled=config.values["train"]["amp"]):
            logits = model(a, b)["logits"]
        all_counts, inside = per_sample_confusions(
            logits, target, threshold=config.values["evaluation"]["threshold"],
            injection_support=support)
        for index, sample_id in enumerate(ids):
            counts = all_counts[index].tolist()
            region = inside[index].tolist() if inside is not None else None
            total.add(counts, region)
            source_id = sample_id.rsplit("@x", 1)[0]
            sources.setdefault(source_id, ConfusionAccumulator(regional=regional)).add(counts, region)
            if regional:
                label = "|".join((batch["direction"][index], batch["target_date"][index],
                                  batch["spatial_mode"][index]))
                strata.setdefault(label, ConfusionAccumulator(regional=True)).add(counts, region)
    if seen != set(expected_ids) or len(seen) != len(expected_ids):
        raise ValueError("Final-test condition omitted or added a tile")
    if guard is not None:
        guard()
    return {"condition_id": condition_id, "kind": "clean" if not regional else condition_id.rsplit("_l", 1)[0],
            "level": 0 if not regional else int(condition_id.rsplit("_l", 1)[1]),
            **total.report(),
            "joint_strata": {key: value.report() for key, value in sorted(strata.items())},
            "by_source": {key: value.report() for key, value in sorted(sources.items())},
            "seconds": time.monotonic() - started}


def audit_test_data(config: Config, checkpoint: Mapping[str, Any],
                    expected_pairs: int, expected_tiles: int) -> tuple[LEVIRDataset, dict[str, Any]]:
    """The first real test access, called only after the pretest seal and guard."""
    if not config.run_test_after_training or set(checkpoint["file_sha256"]) != {"train", "val"}:
        raise ValueError("Final test must have a separately audited development checkpoint")
    dataset = LEVIRDataset(config, "test")
    inventory = dataset.manifest()
    if (inventory["source_pairs"] != expected_pairs or inventory["tiles"] != expected_tiles
            or inventory["tile_size"] != 256):
        raise ValueError("Test pair/tile count differs from the frozen suite")
    for split in ("train", "val"):
        other = (config.data_root / config.values["data"]["splits"][split]).resolve()
        if dataset.root == other or other in dataset.root.parents or dataset.root in other.parents:
            raise ValueError("Test path aliases a development split")
    development_pairs: set[tuple[str, str]] = set()
    decoded_development_pairs: set[tuple[str, str]] = set()
    for split in ("train", "val"):
        development = LEVIRDataset(config, split)
        if development.manifest() != checkpoint["manifests"][split]:
            raise ValueError(f"Archived {split} inventory changed before test audit")
        for record in development.records:
            hashes = {key: file_sha256(getattr(record, key))
                      for key in ("image_a", "image_b", "mask")}
            if hashes != checkpoint["file_sha256"][split][record.name]:
                raise ValueError(f"Archived {split} source content changed")
            development_pairs.add(tuple(sorted((hashes["image_a"], hashes["image_b"]))))
            pair = development.read_pair(record)
            decoded_development_pairs.add(tuple(sorted((
                hashlib.sha256(np.ascontiguousarray(pair.image_a).tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(pair.image_b).tobytes()).hexdigest()))))
    content: dict[str, dict[str, str]] = {}
    side = config.values["data"]["expected_image_size"]
    if type(side) is not int or side < dataset.tile_size:
        raise ValueError("Final-test source dimensions must be explicit")
    for record in dataset.records:
        # Full decode checks mask decoding and RGB shape; content hashes bind
        # the original encoded files used by the ensuing tile inference.
        pair = dataset.read_pair(record)
        if (pair.image_a.shape != (side, side, 3)
                or pair.image_b.shape != (side, side, 3)
                or pair.mask.shape != (side, side)
                or not np.isin(pair.mask, (0, 1)).all()):
            raise ValueError(f"Invalid decoded held-out source: {record.name}")
        hashes = {key: file_sha256(getattr(record, key))
                  for key in ("image_a", "image_b", "mask")}
        if tuple(sorted((hashes["image_a"], hashes["image_b"]))) in development_pairs:
            raise ValueError("Exact encoded test image pair duplicates train/val")
        if tuple(sorted((hashlib.sha256(np.ascontiguousarray(pair.image_a).tobytes()).hexdigest(),
                         hashlib.sha256(np.ascontiguousarray(pair.image_b).tobytes()).hexdigest()))) in decoded_development_pairs:
            raise ValueError("Exact decoded test image pair duplicates train/val")
        content[record.name] = hashes
    return dataset, {"schema_version": 1, "split": "test", "inventory": inventory,
                     "file_sha256": content, "label_decode": config.values["data"]["label_decode"],
                     "limitation": "Exact encoded/decoded pair separation does not establish geographic independence."}


def validate_results(clean: Mapping[str, Any], conditions: Mapping[str, Any],
                     expected_ids: list[str], instances: list[dict[str, Any]],
                     tile_size: int) -> None:
    condition_ids = [f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)]
    if set(conditions) != set(condition_ids) or len(instances) != len(condition_ids) * len(expected_ids):
        raise ValueError("Final-test condition/instance coverage differs")
    sources = {sample_id.rsplit("@x", 1)[0] for sample_id in expected_ids}
    expected_pixels = len(expected_ids) * tile_size * tile_size
    if (clean["samples"] != len(expected_ids) or clean["pixels"] != expected_pixels
            or set(clean["by_source"]) != sources):
        raise ValueError("Clean final-test result is incomplete")
    expected_strata = Counter()
    for instance in instances:
        expected_strata[(instance["condition_id"], "|".join((instance["direction"],
                         instance["target_date"], instance["spatial_mode"]))) ] += 1
    for condition_id in condition_ids:
        result = conditions[condition_id]
        if (result["condition_id"] != condition_id or result["samples"] != len(expected_ids)
                or result["pixels"] != expected_pixels or set(result["by_source"]) != sources):
            raise ValueError(f"Incomplete final-test condition: {condition_id}")
        actual = {key: value["samples"] for key, value in result["joint_strata"].items()}
        expected = {key: count for (cell, key), count in expected_strata.items()
                    if cell == condition_id}
        if actual != expected:
            raise ValueError(f"Final-test stratum coverage differs: {condition_id}")
        for source in sources:
            if result["by_source"][source]["samples"] != clean["by_source"][source]["samples"]:
                raise ValueError(f"Final-test source tiles differ: {condition_id}/{source}")
    summarize_conditions(dict(conditions), condition_ids)


def _f1(matrix: np.ndarray) -> np.ndarray:
    # .../4 axis is TN,FP,FN,TP. Empty F1 denominators are fixed at zero.
    denominator = 2 * matrix[..., 3] + matrix[..., 1] + matrix[..., 2]
    return np.divide(2 * matrix[..., 3], denominator,
                     out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)


def bootstrap_sources(clean: Mapping[str, Any], conditions: Mapping[str, Any],
                      *, seed: int, replicates: int) -> dict[str, Any]:
    if seed != 20260926 or replicates != 5000:
        raise ValueError("Bootstrap settings are frozen")
    names = sorted(clean["by_source"])
    cells = [f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)]
    if len(names) != 128 or any(set(conditions[cell]["by_source"]) != set(names) for cell in cells):
        raise ValueError("Bootstrap source coverage differs")
    clean_counts = np.asarray([[clean["by_source"][name]["counts"][key] for key in COUNT_KEYS]
                               for name in names], dtype=np.float64)
    condition_counts = np.asarray([[[conditions[cell]["by_source"][name]["counts"][key]
                                     for key in COUNT_KEYS] for name in names]
                                   for cell in cells], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(names), size=(replicates, len(names)))
    frequencies = np.zeros((replicates, len(names)), dtype=np.float64)
    rows = np.repeat(np.arange(replicates), len(names))
    np.add.at(frequencies, (rows, indices.ravel()), 1)
    clean_f1 = _f1(frequencies @ clean_counts)
    condition_pooled = (frequencies @ condition_counts.transpose(1, 0, 2).reshape(len(names), -1))
    condition_pooled = condition_pooled.reshape(replicates, len(cells), 4)
    corruption_mf1 = _f1(condition_pooled).mean(axis=1)
    gap = clean_f1 - corruption_mf1
    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]
    return {"unit": "original_source_pair", "sources": len(names), "replicates": replicates,
            "seed": seed, "method": "same_source_draw_for_clean_and_all_45_cells",
            "percentile_95": {"clean_f1": interval(clean_f1),
                              "corruption_mf1": interval(corruption_mf1),
                              "clean_minus_corruption_mf1": interval(gap)},
            "limitation": "Conditional on one checkpoint; not an interval over training seeds."}


def summarize_test(clean: Mapping[str, Any], conditions: Mapping[str, Any],
                   bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    cells = [f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)]
    primary = summarize_conditions(dict(conditions), cells)
    pooled_counts = [sum(int(conditions[cell]["counts"][key]) for cell in cells)
                     for key in COUNT_KEYS]
    pooled = counts_report(pooled_counts)
    per_level = {str(level): sum(conditions[f"{kind}_l{level}"]["metrics"]["f1"]
                                  for kind in KINDS) / len(KINDS) for level in range(1, 6)}
    return {"clean": {key: clean[key] for key in ("counts", "metrics", "samples", "pixels")},
            **primary, "clean_minus_corruption_mf1": clean["metrics"]["f1"] - primary["corruption_mf1"],
            "per_level_mf1": per_level, "pooled_corruption": pooled,
            "net_fp_vs_repeated_clean": pooled["counts"]["fp"] - 45 * clean["counts"]["fp"],
            "net_fn_vs_repeated_clean": pooled["counts"]["fn"] - 45 * clean["counts"]["fn"],
            "source_pair_bootstrap": dict(bootstrap)}


def synthetic_cuda_smoke(job: Job) -> dict[str, Any]:
    """Exercise checkpoint/scorer/replay on temporary synthetic files only."""
    base, checkpoint = _metadata(job)
    config = _test_config(base, checkpoint)
    hardware = hardware_preflight(job, config)
    device = setup_runtime(config)
    if device.type != "cuda":
        raise RuntimeError("Synthetic final-test smoke must use real CUDA")
    model = move_model(build_model(config, initialize_encoder=False), device,
                       config.values["runtime"]["memory_format"])
    model.load_state_dict(checkpoint["model"], strict=True)
    protocol = load_robust_config(job.path("training_config")).protocol
    with tempfile.TemporaryDirectory(prefix="final-test-smoke-") as temporary:
        fixture_values = copy.deepcopy(config.values)
        fixture_values["experiment"].update(kind="smoke", name="synthetic_final_test_only",
                                              output_dir=str(Path(temporary) / "unused_run"))
        fixture_values["data"].update(root=str(Path(temporary) / "data"),
                                       expected_pairs={"train": 1, "val": 1, "test": 2},
                                       expected_image_size=256)
        validate_config(fixture_values)
        fixture_config = Config(fixture_values, Path(temporary) / "fixture.yaml")
        create_fixture(fixture_config)
        smoke_checkpoint = {"manifests": {}, "file_sha256": {}}
        for split in ("train", "val"):
            development = LEVIRDataset(fixture_config, split)
            smoke_checkpoint["manifests"][split] = development.manifest()
            smoke_checkpoint["file_sha256"][split] = {
                record.name: {key: file_sha256(getattr(record, key))
                              for key in ("image_a", "image_b", "mask")}
                for record in development.records}
        _, smoke_audit = audit_test_data(fixture_config, smoke_checkpoint, 2, 2)
        if smoke_audit["inventory"]["tiles"] != 2:
            raise ValueError("Synthetic final-test data audit failed")
        dataset = LEVIRDataset(fixture_config, "test")
        ids = sample_ids(dataset)
        clean = score_condition(model, make_loader(dataset, fixture_config, training=False), config,
                                device, expected_ids=ids, condition_id="clean")
        generated = test_instances(protocol, ids, job.values["test_seed"])
        first = [row for row in generated if row["condition_id"] == "brightness_l1"]
        corrupt = score_condition(model, make_loader(TestInstanceDataset(dataset, first),
                                                     fixture_config, training=False), config, device,
                                  expected_ids=ids, condition_id="brightness_l1")
        if (clean["samples"] != 2 or corrupt["samples"] != 2
                or clean["pixels"] != 2 * 256 * 256
                or corrupt["pixels"] != 2 * 256 * 256
                or not math.isfinite(clean["metrics"]["f1"])
                or not math.isfinite(corrupt["metrics"]["f1"])):
            raise ValueError("Synthetic CUDA final-test scorer smoke failed")
    return {"status": "synthetic_cuda_smoke_passed", "research_result": False,
            "dataset": "temporary_synthetic_fixture_only", "sample_count": 2,
            "generated_conditions": 45, "synthetic_data_audit_passed": True,
            "checkpoint_sha256": job.values["best_checkpoint_sha256"],
            "job_sha256": file_sha256(job.source), "source_sha256": source_hashes(),
            "hardware": hardware, "test_accessed": False, "test_evaluated": False}


class FinalTestTemperatureGuard(EvaluationTemperatureGuard):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.state["test_suite"] = "single_b1_seed42_final_test_v1"


def run(job: Job) -> dict[str, Any]:
    """No real held-out path is examined until all freeze/guard checks pass."""
    approval = _approval(job)
    seal = _seal(job, approval)
    seal_sha256 = file_sha256(job.path("pretest_seal"))
    base, checkpoint = _metadata(job)
    config = _test_config(base, checkpoint)
    if job.output == config.data_root or config.data_root in job.output.parents:
        raise ValueError("Final-test output must be outside the research dataset")
    hardware = hardware_preflight(job, config)
    if job.output.exists():
        raise FileExistsError(f"One-shot final-test output already exists: {job.output}")
    guard_config = load_guard_config(job.path("guard_config"))
    job.output.mkdir(parents=True, exist_ok=False)
    guard = FinalTestTemperatureGuard(guard_config, job.output)
    test_accessed = False
    test_evaluated = False
    started = time.monotonic()
    try:
        # A durable temperature observation precedes even the first test path
        # enumeration. No model inference occurs before the full test audit.
        guard(force=True)
        device = setup_runtime(config)
        if device.type != "cuda":
            raise RuntimeError("Final test cannot fall back from real CUDA")
        test_accessed = True
        guard.state["test_accessed"] = True
        guard._persist()
        dataset, audit = audit_test_data(config, checkpoint,
                                         job.values["expected_test_pairs"],
                                         job.values["expected_test_tiles"])
        write_json(job.output / "test_data_audit.json", audit)
        ids = sample_ids(dataset)
        protocol = load_robust_config(job.path("training_config")).protocol
        if (protocol.values["name"] != "levir_tile_corruptions_v1"
                or protocol.values["seeds"] != {"train": 310942, "val": 310943, "test": 310944}):
            raise ValueError("Test corruption recipe/seeds differ from frozen Phase 2")
        instances = test_instances(protocol, ids, job.values["test_seed"])
        if len(instances) != job.values["expected_conditions"] * len(ids):
            raise ValueError("Test corruption instance count differs")
        instance_path = job.output / "test_instances.jsonl"
        with instance_path.open("x", encoding="utf-8") as stream:
            for instance in instances:
                stream.write(canonical(instance) + "\n")
        manifest = {"schema_version": 1, "split": "test", "status": "generated_after_pretest_seal",
                    "stream_seed": job.values["test_seed"], "condition_count": 45,
                    "instances": len(instances), "source_pairs": 128, "tiles": len(ids),
                    "instance_file_sha256": file_sha256(instance_path),
                    "test_data_audit_sha256": file_sha256(job.output / "test_data_audit.json"),
                    "condition_counts": dict(Counter(row["condition_id"] for row in instances)),
                    "stratum_counts": dict(Counter("|".join((row["condition_id"],
                        row["direction"], row["target_date"], row["spatial_mode"]))
                        for row in instances))}
        write_json(job.output / "test_manifest.json", manifest)
        if (source_hashes() != approval["source_sha256"]
                or file_sha256(job.path("pretest_seal")) != seal_sha256):
            raise ValueError("Final-test source changed before inference")
        torch.cuda.reset_peak_memory_stats(device)
        model = move_model(build_model(config, initialize_encoder=False), device,
                           config.values["runtime"]["memory_format"])
        model.load_state_dict(checkpoint["model"], strict=True)
        guard()
        test_evaluated = True
        guard.state["test_evaluated"] = True
        guard._persist()
        clean = score_condition(model, make_loader(dataset, config, training=False), config,
                                device, expected_ids=ids, condition_id="clean", guard=guard)
        write_json(job.output / "clean.json", clean)
        conditions: dict[str, Any] = {}
        (job.output / "conditions").mkdir()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for instance in instances:
            grouped[instance["condition_id"]].append(instance)
        for kind in KINDS:
            for level in range(1, 6):
                cell = f"{kind}_l{level}"
                loader = make_loader(TestInstanceDataset(dataset, grouped[cell]), config,
                                     training=False)
                result = score_condition(model, loader, config, device,
                                         expected_ids=ids, condition_id=cell, guard=guard)
                write_json(job.output / "conditions" / f"{cell}.json", result)
                conditions[cell] = result
                _atomic_json(job.output / "progress.json", {
                    "status": "running", "completed_conditions": len(conditions),
                    "expected_conditions": 45, "test_accessed": True, "test_evaluated": True})
                print(f"Final B1 test {cell}: {len(conditions)}/45", flush=True)
        validate_results(clean, conditions, ids, instances, config.values["data"]["tile_size"])
        write_json(job.output / "conditions.json", conditions)
        if (_json(job.output / "clean.json") != clean
                or _json(job.output / "conditions.json") != conditions
                or any(_json(job.output / "conditions" / f"{cell}.json") != value
                       for cell, value in conditions.items())):
            raise ValueError("Final-test condition files differ from scored records")
        bootstrap = bootstrap_sources(clean, conditions,
                                      seed=job.values["bootstrap_seed"],
                                      replicates=job.values["bootstrap_replicates"])
        descriptors = summarize_test(clean, conditions, bootstrap)
        if (any({key: file_sha256(getattr(record, key))
                 for key in ("image_a", "image_b", "mask")}
                != audit["file_sha256"][record.name] for record in dataset.records)
                or file_sha256(instance_path) != manifest["instance_file_sha256"]
                or file_sha256(job.output / "test_data_audit.json") != manifest["test_data_audit_sha256"]):
            raise ValueError("Held-out test or generated instance/audit content changed during scoring")
        if source_hashes() != approval["source_sha256"]:
            raise ValueError("Final-test source changed during inference")
        if (file_sha256(job.path("pretest_seal")) != seal_sha256
                or file_sha256(job.path("approval")) != seal["approval_sha256"]
                or file_sha256(job.path("best_checkpoint")) != job.values["best_checkpoint_sha256"]
                or file_sha256(job.path("guard_config")) != job.values["guard_config_sha256"]):
            raise ValueError("Frozen final-test input changed during scoring")
        load_job(job.source)
        result = {"status": "completed", "mode": "final_heldout_test_single_b1_seed42_v1",
                  "method": "b1_ordinary_supervised_augmentation", "checkpoint_epoch": 68,
                  "checkpoint_sha256": job.values["best_checkpoint_sha256"],
                  "selection": "incumbent_seed42_clean_validation_f1_only",
                  "protocol_document_sha256": job.values["protocol_document_sha256"],
                  "job_sha256": file_sha256(job.source),
                  "approval_sha256": file_sha256(job.path("approval")),
                  "pretest_seal_sha256": seal_sha256,
                  "test_data_audit_sha256": file_sha256(job.output / "test_data_audit.json"),
                  "test_manifest_sha256": file_sha256(job.output / "test_manifest.json"),
                  "test_instances_sha256": file_sha256(instance_path),
                  "clean_result_sha256": file_sha256(job.output / "clean.json"),
                  "conditions_result_sha256": file_sha256(job.output / "conditions.json"),
                  "condition_file_sha256": {cell: file_sha256(job.output / "conditions" / f"{cell}.json")
                                            for cell in conditions},
                  "source_sha256": approval["source_sha256"],
                  "hardware_preflight": hardware, "environment": environment_record(config, device),
                  **descriptors, "seconds": time.monotonic() - started,
                  "peak_inference_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                  "test_accessed": True, "test_evaluated": True, "model_training_run": False,
                  "no_test_based_selection": True}
        if not guard.attached or not guard.state["observations"]:
            raise GuardError("Final-test temperature guard did not attach")
        result["temperature_guard"] = guard.finish(result)
        result["guarded_completion_required"] = True
        _atomic_json(job.output / "summary.json", result)
        _atomic_json(job.output / "status.json", result)
        _atomic_json(job.output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "summary_sha256": _sha((job.output / "summary.json").read_bytes()),
            "guard_state_sha256": _sha((job.output / "temperature_guard_state.json").read_bytes()),
            "wrapper_source_sha256": file_sha256(Path(__file__).resolve()),
            "test_accessed": True, "test_evaluated": True})
        return result
    except BaseException as error:
        guard.abort(error)
        _atomic_json(job.output / "status.json", {
            "status": "failed_incomplete_no_automatic_retry", "guarded_result_valid": False,
            "error": f"{type(error).__name__}: {error}",
            "test_accessed": test_accessed, "test_evaluated": test_evaluated})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metadata-preflight-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--synthetic-smoke-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    job = load_job(args.config)
    if args.metadata_preflight_only:
        result = preflight(job, approval=False, hardware=False)
    elif args.preflight_only:
        result = preflight(job, approval=True, hardware=True)
    elif args.synthetic_smoke_only:
        result = synthetic_cuda_smoke(job)
    else:
        if args.record is not None:
            raise ValueError("--record is for pretest evidence only")
        result = run(job)
    if args.record is not None:
        write_json(args.record, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
