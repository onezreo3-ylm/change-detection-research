"""R1E-only, separately frozen evaluation of approved replication seeds.

This leaves the original R1 evaluator and pre-validation seal untouched.  The
only compatibility correction is to check a replica checkpoint against its own
hash-bound training protocol, then prove that protocol differs from seed 42 only
in the R1-declared training identity and train-stream seed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Callable

from PIL import features
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import file_sha256, make_loader
from pcdc.engine import move_model, read_checkpoint, source_files, source_hashes
from pcdc.evaluate_candidate import (
    CandidateMetadata, _condition_diagnostics, _load_metadata, _snapshot,
    _validate_checkpoints, compare_results, load_candidate_evaluation_config,
)
from pcdc.evaluate_corruptions import (
    COUNT_KEYS, METRIC_KEYS, _bind_hash, _bind_validation_files,
    _verify_input_hashes, run_condition, summarize_conditions,
)
from pcdc.frozen_validation import FrozenInstanceDataset, build_validation_dataset
from pcdc.model import build_model
from pcdc.robust_baselines import RobustConfig, load_robust_config, robust_source_files
from pcdc.runtime import environment_record, setup_runtime, write_json


RUN_KEYS = ("b1_43", "b2_43", "b1_44", "b2_44")
ROOT = Path(__file__).resolve().parents[1]
APPROVAL_KEYS = {
    "schema_version", "status", "protocol_document", "protocol_document_sha256",
    "original_r1_seal", "original_r1_seal_sha256", "original_r1_approval",
    "original_r1_approval_sha256", "new_source_sha256", "runs", "test_evaluated",
}
RUN_KEYS_SCHEMA = {
    "candidate_config", "candidate_config_sha256", "training_config",
    "training_config_sha256", "reference_training_config",
    "reference_training_config_sha256", "training_seed", "train_stream_seed",
}


def _path(source: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("R1E path must be nonempty text")
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def _verified_path(source: Path, value: str, expected: str) -> Path:
    if not isinstance(expected, str) or len(expected) != 64 or any(
            c not in "0123456789abcdef" for c in expected):
        raise ValueError("R1E binding must be a lowercase SHA256")
    path = _path(source, value)
    if file_sha256(path) != expected:
        raise ValueError(f"R1E frozen input hash differs: {path}")
    return path


def _without(value: dict[str, Any], *paths: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for path in paths:
        current = result
        keys = path.split(".")
        for key in keys[:-1]:
            current = current[key]
        current.pop(keys[-1])
    return result


def check_declared_replication_delta(training: RobustConfig,
                                     reference: RobustConfig, *, seed: int,
                                     train_stream_seed: int) -> None:
    """Fail closed on any training/inference difference beyond R1's seed mapping."""
    if (type(seed) is not int or seed not in (43, 44)
            or train_stream_seed != {43: 310945, 44: 310946}[seed]
            or training.mode != reference.mode
            or training.base.values["experiment"]["seed"] != seed
            or reference.base.values["experiment"]["seed"] != 42
            or training.protocol.values["seeds"] != {
                "train": train_stream_seed, "val": 310943, "test": 310944}
            or reference.protocol.values["seeds"] != {
                "train": 310942, "val": 310943, "test": 310944}
            or training.base.run_test_after_training
            or reference.base.run_test_after_training):
        raise ValueError("R1E replication seed or split declaration differs")
    if _without(dict(training.base.values), "experiment.name", "experiment.seed",
                "experiment.output_dir") != _without(
                dict(reference.base.values), "experiment.name", "experiment.seed",
                "experiment.output_dir"):
        raise ValueError("R1E base training/inference settings differ beyond the seed")
    if (training.protocol.path("reference_checkpoint")
            != reference.protocol.path("reference_checkpoint")
            or training.protocol.path("reference_audit")
            != reference.protocol.path("reference_audit")
            or _without(dict(training.protocol.values), "base_config", "output_dir",
                        "reference_checkpoint", "reference_audit", "seeds.train")
            != _without(dict(reference.protocol.values), "base_config", "output_dir",
                        "reference_checkpoint", "reference_audit", "seeds.train")):
        raise ValueError("R1E corruption operator or reserved validation settings differ")
    if _without(dict(training.values), "name", "base_config", "protocol_config",
                "protocol_sha256", "severity_review", "output_dir") != _without(
                dict(reference.values), "name", "base_config", "protocol_config",
                "protocol_sha256", "severity_review", "output_dir"):
        raise ValueError("R1E robust baseline objective or exposure settings differ")


@dataclass(frozen=True)
class R1EContext:
    approval_path: Path
    approval_sha256: str
    approval: dict[str, Any]
    key: str
    candidate: Any
    training: RobustConfig
    seal: dict[str, Any]
    protocol_path: Path
    seal_path: Path


def load_context(approval_path: str | Path, key: str) -> R1EContext:
    source = Path(approval_path).resolve()
    raw = source.read_bytes()
    values = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    if (not isinstance(values, dict) or set(values) != APPROVAL_KEYS
            or values["schema_version"] != 1 or values["status"] != "approved"
            or values["test_evaluated"] is not False
            or not isinstance(values["runs"], dict)
            or set(values["runs"]) != set(RUN_KEYS)
            or key not in RUN_KEYS):
        raise ValueError("Invalid or unapproved R1E freeze")
    protocol_path = _verified_path(source, values["protocol_document"],
                                   values["protocol_document_sha256"])
    seal_path = _verified_path(source, values["original_r1_seal"],
                               values["original_r1_seal_sha256"])
    _verified_path(source, values["original_r1_approval"],
                   values["original_r1_approval_sha256"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("status") != "sealed_before_validation"
            or seal.get("validation_evaluated") is not False
            or seal.get("test_evaluated") is not False):
        raise ValueError("Original R1 seal is not the pre-validation seal")
    for name, expected in seal["evaluation_source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Original frozen evaluator source changed: {name}")
    new_sources = values["new_source_sha256"]
    if (not isinstance(new_sources, dict)
            or set(new_sources) != {
                "scripts/r1e_evaluation.py", "scripts/evaluate_r1e_guarded.py"}):
        raise ValueError("R1E source freeze is incomplete")
    for name, expected in new_sources.items():
        _verified_path(source, str(ROOT / name), expected)
    row = values["runs"][key]
    if not isinstance(row, dict) or set(row) != RUN_KEYS_SCHEMA:
        raise ValueError("Incomplete R1E run binding")
    candidate_path = _verified_path(source, row["candidate_config"],
                                    row["candidate_config_sha256"])
    training_path = _verified_path(source, row["training_config"],
                                   row["training_config_sha256"])
    reference_path = _verified_path(source, row["reference_training_config"],
                                    row["reference_training_config_sha256"])
    candidate = load_candidate_evaluation_config(candidate_path)
    training = load_robust_config(training_path)
    reference = load_robust_config(reference_path)
    mode, seed = key.split("_")
    seed = int(seed)
    if (row["training_seed"] != seed or candidate.values["expected_mode"] != mode
            or candidate.values["expected_kind"] != "research"
            or training.mode != mode
            or row["training_config_sha256"] != seal["runs"][key]["config_sha256"]):
        raise ValueError("R1E method/seed/training config binding differs")
    check_declared_replication_delta(training, reference, seed=seed,
                                     train_stream_seed=row["train_stream_seed"])
    old = seal["evaluation_configs"][key]
    old_path = (ROOT / old["path"]).resolve()
    _verified_path(source, str(old_path), old["sha256"])
    old_candidate = load_candidate_evaluation_config(old_path)
    if (_without(dict(candidate.values), "name", "output_dir")
            != _without(dict(old_candidate.values), "name", "output_dir")
            or candidate.output_dir == old_candidate.output_dir
            or candidate.output_dir.exists()):
        raise ValueError("R1E candidate evaluation changed more than its output identity")
    for field, artifact in (("candidate_best_checkpoint_sha256", "best.pt"),
                            ("candidate_last_checkpoint_sha256", "last.pt"),
                            ("candidate_status_sha256", "status.json"),
                            ("candidate_guard_completion_sha256", "guard_completion.json"),
                            ("candidate_exposure_sha256", "training_exposure_summary.json")):
        if candidate.values[field] != seal["runs"][key]["artifact_sha256"][artifact]:
            raise ValueError(f"R1E sealed training artifact differs: {artifact}")
    if file_sha256(source) != hashlib.sha256(raw).hexdigest():
        raise ValueError("R1E approval changed during load")
    return R1EContext(source, file_sha256(source), values, key, candidate,
                      training, seal, protocol_path, seal_path)


def checked_metadata_and_checkpoints(context: R1EContext) -> tuple[Any, Any, Any, Any]:
    job = context.candidate
    metadata = _load_metadata(job, require_review=True)
    best = read_checkpoint(job.path("candidate_best_checkpoint"))
    last = read_checkpoint(job.path("candidate_last_checkpoint"))
    # The original strict checker is reused unchanged, but its checkpoint
    # training-protocol view is the approved seed-specific training protocol.
    view = CandidateMetadata(
        SimpleNamespace(protocol=context.training.protocol,
                        manifest=metadata.bundle.manifest),
        metadata.baseline_summary, metadata.baseline_conditions,
        metadata.candidate_status, metadata.guard_completion, metadata.exposure,
        metadata.input_hashes)
    expected_clean = _validate_checkpoints(job, view, best, last)
    approved_sources = {name: context.seal["approved_source_sha256"][name]
                        for name in robust_source_files()}
    for checkpoint in (best, last):
        if (checkpoint.get("config") != context.training.base.values
                or checkpoint.get("robust_config") != dict(context.training.values)
                or checkpoint.get("training_signature") != context.training.fingerprint
                or checkpoint.get("severity_review_sha256")
                != file_sha256(context.training.paths["severity_review"])
                or checkpoint.get("source_sha256") != approved_sources):
            raise ValueError("R1E checkpoint differs from its exact approved training run")
    _bind_hash(metadata.input_hashes, context.approval_path, context.approval_sha256)
    _bind_hash(metadata.input_hashes, context.protocol_path,
               context.approval["protocol_document_sha256"])
    _bind_hash(metadata.input_hashes, context.seal_path,
               context.approval["original_r1_seal_sha256"])
    _verify_input_hashes(metadata.input_hashes)
    return metadata, best, last, expected_clean


def preflight(context: R1EContext) -> dict[str, Any]:
    metadata, best, last, _ = checked_metadata_and_checkpoints(context)
    del best, last
    return {
        "status": "r1e_preflight_passed", "run": context.key,
        "approval_sha256": context.approval_sha256,
        "benchmark_fingerprints": metadata.bundle.fingerprints,
        "dataset_opened": False, "checkpoint_loaded": True,
        "model_inference_run": False, "sensor_queried": False,
        "test_evaluated": False,
    }


def evaluate(context: R1EContext,
             guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    """Run the unchanged validation conditions with the corrected provenance gate."""
    job = context.candidate
    metadata, best, last, expected_clean = checked_metadata_and_checkpoints(context)
    output = job.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite R1E validation: {output}")
    dataset, _ = build_validation_dataset(metadata.bundle)
    config = metadata.bundle.protocol.base
    if dataset.split != "val" or config.run_test_after_training:
        raise ValueError("R1E may only evaluate development validation")
    _bind_validation_files(metadata.input_hashes, metadata.bundle, dataset, best)
    _verify_input_hashes(metadata.input_hashes)
    code_hashes = source_hashes()
    if any(code_hashes[name] != context.seal["evaluation_source_sha256"][name]
           for name in code_hashes):
        raise ValueError("Original frozen evaluation source changed")
    device = setup_runtime(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {
        "status": "evaluating_clean", "mode": job.values["expected_mode"],
        "test_evaluated": False, "training_started": False})
    try:
        _snapshot(job, output, code_hashes, source_files())
        for source, destination in (
                (context.approval_path, "r1e_approval.yaml"),
                (context.protocol_path, "r1e_protocol.md"),
                (context.seal_path, "original_r1_prevalidation_seal.json"),
                (Path(__file__).resolve(), "source/scripts/r1e_evaluation.py"),
                (ROOT / "scripts/evaluate_r1e_guarded.py",
                 "source/scripts/evaluate_r1e_guarded.py")):
            target = output / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_sha256(target) != file_sha256(source):
                raise ValueError("R1E source snapshot changed")
        write_json(output / "provenance.json", {
            "batch": "R1E", "run": context.key,
            "r1e_approval_sha256": context.approval_sha256,
            "original_r1_prevalidation_seal_sha256": (
                context.approval["original_r1_seal_sha256"]),
            "candidate_evaluation_config_sha256": job.fingerprint,
            "training_config_sha256": context.approval["runs"][context.key][
                "training_config_sha256"],
            "input_file_sha256": metadata.input_hashes,
            "benchmark_fingerprints": metadata.bundle.fingerprints,
            "candidate_mode": job.values["expected_mode"],
            "candidate_training_signature": best["training_signature"],
            "candidate_initial_model_tensor_sha256": best[
                "initial_model_tensor_sha256"],
            "candidate_source_sha256": best["source_sha256"],
            "evaluation_source_sha256": code_hashes,
            "new_source_sha256": context.approval["new_source_sha256"],
            "selection": "clean_validation_f1_only", "test_evaluated": False,
        })
        environment = environment_record(config, device)
        environment.update(jpeg_codec=features.version_codec("jpg"),
                           libjpeg_turbo=features.version_feature("libjpeg_turbo"))
        write_json(output / "environment.json", environment)
        model = move_model(build_model(config, initialize_encoder=False), device,
                           config.values["runtime"]["memory_format"])
        model.load_state_dict(best["model"], strict=True)
        del last
        _verify_input_hashes(metadata.input_hashes)
        clean = run_condition(model, make_loader(dataset, config, training=False),
                              config, device, guard_callback=guard_callback)
        clean["matches_candidate_checkpoint"] = (
            clean["samples"] == expected_clean["samples"]
            and all(clean["counts"][key] == expected_clean["metrics"][key]
                    for key in COUNT_KEYS)
            and all(clean["metrics"][key] == expected_clean["metrics"][key]
                    for key in METRIC_KEYS))
        clean["checkpoint_metrics"] = expected_clean["metrics"]
        write_json(output / "clean_metrics.json", clean)
        if not clean["matches_candidate_checkpoint"]:
            raise ValueError("R1E clean validation differs from selected checkpoint")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for instance in metadata.bundle.instances:
            grouped.setdefault(instance["condition_id"], []).append(instance)
        expected_ids = list(metadata.bundle.manifest["condition_counts"])
        if set(grouped) != set(expected_ids):
            raise ValueError("R1E validation instance coverage is incomplete")
        results: dict[str, dict[str, Any]] = {}
        (output / "conditions").mkdir()
        for condition_id in expected_ids:
            cell = FrozenInstanceDataset(dataset, grouped[condition_id])
            result = run_condition(model, make_loader(cell, config, training=False),
                                   config, device, guard_callback=guard_callback)
            if (result["condition_id"] != condition_id
                    or result["samples"]
                    != metadata.bundle.manifest["condition_counts"][condition_id]
                    or result["pixels"] != result["samples"] * dataset.tile_size ** 2):
                raise ValueError(f"R1E incomplete condition: {condition_id}")
            expected_strata = {
                key[len(condition_id) + 1:]: value
                for key, value in metadata.bundle.manifest["stratum_counts"].items()
                if key.startswith(condition_id + "|")}
            actual_strata = {key: value["samples"]
                             for key, value in result["joint_strata"].items()}
            if actual_strata != expected_strata:
                raise ValueError(f"R1E stratum coverage differs: {condition_id}")
            result["clean_reference_f1"] = clean["metrics"]["f1"]
            result["clean_minus_f1"] = clean["metrics"]["f1"] - result[
                "metrics"]["f1"]
            write_json(output / "conditions" / f"{condition_id}.json", result)
            results[condition_id] = result
            write_json(output / "status.json", {
                "status": "evaluating_corruptions", "mode": job.values["expected_mode"],
                "completed_conditions": len(results),
                "condition_count": len(expected_ids), "test_evaluated": False,
                "training_started": False}, overwrite=True)
            print(f"R1E/{context.key}/{condition_id}: "
                  f"f1={result['metrics']['f1']:.6f} "
                  f"({len(results)}/{len(expected_ids)})", flush=True)
        write_json(output / "conditions.json", results)
        aggregate = summarize_conditions(results, expected_ids)
        diagnostics = _condition_diagnostics(results, expected_ids)
        baseline_diagnostics = _condition_diagnostics(
            metadata.baseline_conditions, expected_ids)
        per_condition, paired = compare_results(
            results, metadata.baseline_conditions, expected_ids)
        write_json(output / "comparison_by_condition.json", per_condition)
        write_json(output / "paired_by_source.json", paired)
        family_delta = {
            kind: aggregate["family_mf1"][kind]
            - metadata.baseline_summary["family_mf1"][kind]
            for kind in aggregate["family_mf1"]}
        summary = {
            "status": "completed", "kind": job.values["expected_kind"],
            "mode": "frozen_candidate_corruption_validation",
            "candidate_mode": job.values["expected_mode"], "split": "val",
            "batch": "R1E", "r1e_approval_sha256": context.approval_sha256,
            "candidate_checkpoint_epoch": best["epoch"],
            "selection": "clean_validation_f1_only",
            "clean_f1": clean["metrics"]["f1"],
            "baseline_clean_f1": metadata.baseline_summary["clean_f1"],
            "clean_f1_delta_candidate_minus_baseline": (
                clean["metrics"]["f1"] - metadata.baseline_summary["clean_f1"]),
            **aggregate,
            "baseline_corruption_mf1": metadata.baseline_summary["corruption_mf1"],
            "corruption_mf1_delta_candidate_minus_baseline": (
                aggregate["corruption_mf1"]
                - metadata.baseline_summary["corruption_mf1"]),
            "clean_minus_corruption_mf1": (
                clean["metrics"]["f1"] - aggregate["corruption_mf1"]),
            "baseline_clean_minus_corruption_mf1": (
                metadata.baseline_summary["clean_minus_corruption_mf1"]),
            "degradation_delta_candidate_minus_baseline": (
                (clean["metrics"]["f1"] - aggregate["corruption_mf1"])
                - metadata.baseline_summary["clean_minus_corruption_mf1"]),
            "family_mf1_delta_candidate_minus_baseline": family_delta,
            "candidate_diagnostics": diagnostics,
            "baseline_diagnostics": baseline_diagnostics,
            "test_evaluated": False, "training_started": False,
            "guarded_completion_required": True,
        }
        if (source_hashes() != code_hashes
                or any(file_sha256(ROOT / name) != expected
                       for name, expected in context.approval[
                           "new_source_sha256"].items())):
            raise ValueError("R1E evaluation source changed during scoring")
        _verify_input_hashes(metadata.input_hashes)
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt)
            else "failed", "mode": job.values["expected_mode"],
            "error": f"{type(error).__name__}: {error}",
            "test_evaluated": False, "training_started": False,
        }, overwrite=True)
        raise
