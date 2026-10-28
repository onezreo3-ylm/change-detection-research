"""Score one completed robust checkpoint on the immutable validation benchmark.

The candidate configuration binds both the completed training artifacts and the
already-scored B0 artifacts by SHA256.  Preflight never opens image data or
loads a checkpoint.  Full evaluation remains validation-only and selects the
candidate's clean-validation best checkpoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from PIL import features
import torch
import yaml

from .data import file_sha256, make_loader
from .engine import check_checkpoint, move_model, read_checkpoint, source_files, source_hashes
from .evaluate_corruptions import (
    COUNT_KEYS,
    METRIC_KEYS,
    _bind_hash,
    _bind_validation_files,
    _bound_input_hashes,
    _expected_clean,
    _verify_input_hashes,
    counts_report,
    run_condition,
    summarize_conditions,
)
from .frozen_validation import (
    FrozenInstanceDataset,
    FrozenValidation,
    build_validation_dataset,
    load_bundle,
    load_evaluation_config,
)
from .model import build_model
from .robust_baselines import _exposure_summary
from .runtime import environment_record, setup_runtime, write_json


_SHA_KEYS = (
    "baseline_summary_sha256",
    "baseline_conditions_sha256",
    "candidate_best_checkpoint_sha256",
    "candidate_last_checkpoint_sha256",
    "candidate_status_sha256",
    "candidate_guard_completion_sha256",
    "candidate_exposure_sha256",
)
_PATH_KEYS = (
    "benchmark_config",
    "baseline_summary",
    "baseline_conditions",
    "candidate_best_checkpoint",
    "candidate_last_checkpoint",
    "candidate_status",
    "candidate_guard_completion",
    "candidate_exposure",
    "output_dir",
)
_JOB_KEYS = {"schema_version", "name", "expected_kind", "expected_mode", *_PATH_KEYS, *_SHA_KEYS}
_MODES = {"b0e", "b1", "b2"}
_KINDS = {"research", "smoke"}


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {path}: {value}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value, raw


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    values: Mapping[str, Any]
    source: Path
    raw: bytes

    def path(self, key: str) -> Path:
        value = Path(self.values[key])
        return (value if value.is_absolute() else self.source.parent / value).resolve()

    @property
    def output_dir(self) -> Path:
        return self.path("output_dir")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


@dataclass(frozen=True)
class CandidateMetadata:
    bundle: FrozenValidation
    baseline_summary: dict[str, Any]
    baseline_conditions: dict[str, Any]
    candidate_status: dict[str, Any]
    guard_completion: dict[str, Any]
    exposure: dict[str, Any]
    input_hashes: dict[str, str]


def load_candidate_evaluation_config(path: str | Path) -> CandidateEvaluationConfig:
    source = Path(path).resolve()
    values, raw = _yaml(source)
    _keys(values, _JOB_KEYS, "candidate evaluation config")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only candidate evaluation schema_version 1 is supported")
    if not isinstance(values["name"], str) or not values["name"].strip():
        raise ValueError("name must be nonempty text")
    if values["expected_mode"] not in _MODES:
        raise ValueError(f"expected_mode must be one of {sorted(_MODES)}")
    if values["expected_kind"] not in _KINDS:
        raise ValueError(f"expected_kind must be one of {sorted(_KINDS)}")
    for key in _PATH_KEYS:
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    for key in _SHA_KEYS:
        _sha(values[key], key)
    return CandidateEvaluationConfig(copy.deepcopy(values), source, raw)


def _verified_json(job: CandidateEvaluationConfig, path_key: str, sha_key: str) -> dict[str, Any]:
    path = job.path(path_key)
    if file_sha256(path) != job.values[sha_key]:
        raise ValueError(f"Frozen SHA256 differs for {path_key}")
    value = _json(path)
    if file_sha256(path) != job.values[sha_key]:
        raise ValueError(f"{path_key} changed while it was read")
    return value


def _condition_diagnostics(conditions: Mapping[str, dict[str, Any]],
                           expected_ids: list[str]) -> dict[str, Any]:
    totals = [0, 0, 0, 0]
    foreground_recall, background_fpr = [], []
    for condition_id in expected_ids:
        report = conditions[condition_id]
        counts = report["counts"]
        totals = [value + int(counts[key]) for value, key in zip(totals, COUNT_KEYS)]
        metrics = report["metrics"]
        foreground_recall.append(float(metrics["foreground_recall"]))
        background_fpr.append(float(metrics["background_fpr"]))
    pooled = counts_report(totals)
    return {
        "pooled_over_conditions": pooled,
        "macro_condition_foreground_recall": sum(foreground_recall) / len(foreground_recall),
        "macro_condition_background_fpr": sum(background_fpr) / len(background_fpr),
    }


def _validate_baseline(bundle: FrozenValidation, summary: dict[str, Any],
                       conditions: dict[str, Any]) -> None:
    if (summary.get("status") != "completed"
            or summary.get("mode") != "frozen_corruption_validation"
            or summary.get("split") != "val"
            or summary.get("corruption_scoring_run") is not True
            or summary.get("test_evaluated") is not False
            or summary.get("training_started") is not False
            or summary.get("guarded_completion_required") is not True
            or summary.get("temperature_guard", {}).get("status") != "completed"):
        raise ValueError("B0 summary is not a completed guarded validation evaluation")
    if summary.get("fingerprints") != bundle.fingerprints:
        raise ValueError("B0 summary is bound to a different frozen benchmark")
    expected_ids = list(bundle.manifest["condition_counts"])
    if set(conditions) != set(expected_ids):
        raise ValueError("B0 condition coverage differs from the frozen benchmark")
    for condition_id in expected_ids:
        report = conditions[condition_id]
        if (report.get("condition_id") != condition_id
                or report.get("samples") != bundle.manifest["condition_counts"][condition_id]
                or report.get("pixels") != report["samples"] * bundle.protocol.base.values["data"]["tile_size"] ** 2):
            raise ValueError(f"Invalid B0 condition result: {condition_id}")
        expected_strata = {key[len(condition_id) + 1:]: value
                           for key, value in bundle.manifest["stratum_counts"].items()
                           if key.startswith(condition_id + "|")}
        actual_strata = {key: value["samples"] for key, value in report["joint_strata"].items()}
        if actual_strata != expected_strata:
            raise ValueError(f"Invalid B0 stratum coverage: {condition_id}")
    recalculated = summarize_conditions(conditions, expected_ids)
    for key in ("condition_count", "corruption_mf1", "family_mf1", "aggregation_rule"):
        if summary.get(key) != recalculated[key]:
            raise ValueError(f"B0 aggregate differs from condition records: {key}")


def _validate_training_metadata(job: CandidateEvaluationConfig, status: dict[str, Any],
                                completion: dict[str, Any], exposure: dict[str, Any]) -> None:
    mode = job.values["expected_mode"]
    if (status.get("status") != "completed" or status.get("training_started") is not True
            or status.get("mode") != mode or status.get("kind") != job.values["expected_kind"]
            or status.get("selection") != "clean_val_f1"
            or status.get("test_evaluated") is not False
            or status.get("guarded_completion_required") is not True
            or status.get("temperature_guard", {}).get("status") != "completed"):
        raise ValueError("Candidate is not a completed guarded training run of the expected kind")
    if (completion.get("status") != "completed"
            or completion.get("training_status_sha256") != job.values["candidate_status_sha256"]):
        raise ValueError("Candidate guard completion does not commit the bound training status")
    integers = ("epochs", "best_epoch", "optimizer_steps", "amp_skipped_steps", "pair_views")
    if any(type(status.get(key)) is not int or status[key] < 0 for key in integers):
        raise ValueError("Candidate status has invalid integer counters")
    if (exposure.get("epochs") != status["epochs"]
            or not isinstance(exposure.get("counts"), dict)
            or not isinstance(exposure.get("per_epoch_stream_sha256"), list)
            or len(exposure["per_epoch_stream_sha256"]) != status["epochs"]
            or not exposure["counts"]):
        raise ValueError("Candidate exposure summary is incomplete")
    for row in exposure["per_epoch_stream_sha256"]:
        if set(row) != {"epoch", "sha256"}:
            raise ValueError("Invalid per-epoch exposure identity")
        _sha(row["sha256"], "exposure stream")
    _sha(exposure.get("combined_stream_sha256"), "combined exposure stream")


def _load_metadata(job: CandidateEvaluationConfig, *, require_review: bool) -> CandidateMetadata:
    benchmark_job = load_evaluation_config(job.path("benchmark_config"))
    bundle = load_bundle(benchmark_job, require_review=require_review)
    baseline_summary = _verified_json(job, "baseline_summary", "baseline_summary_sha256")
    baseline_conditions = _verified_json(job, "baseline_conditions", "baseline_conditions_sha256")
    status = _verified_json(job, "candidate_status", "candidate_status_sha256")
    completion = _verified_json(
        job, "candidate_guard_completion", "candidate_guard_completion_sha256")
    exposure = _verified_json(job, "candidate_exposure", "candidate_exposure_sha256")
    _validate_baseline(bundle, baseline_summary, baseline_conditions)
    _validate_training_metadata(job, status, completion, exposure)
    bindings = _bound_input_hashes(bundle)
    for path_key, sha_key in (
            ("baseline_summary", "baseline_summary_sha256"),
            ("baseline_conditions", "baseline_conditions_sha256"),
            ("candidate_best_checkpoint", "candidate_best_checkpoint_sha256"),
            ("candidate_last_checkpoint", "candidate_last_checkpoint_sha256"),
            ("candidate_status", "candidate_status_sha256"),
            ("candidate_guard_completion", "candidate_guard_completion_sha256"),
            ("candidate_exposure", "candidate_exposure_sha256")):
        _bind_hash(bindings, job.path(path_key), job.values[sha_key])
    _verify_input_hashes(bindings)
    return CandidateMetadata(bundle, baseline_summary, baseline_conditions, status,
                             completion, exposure, bindings)


def preflight(job: CandidateEvaluationConfig) -> dict[str, Any]:
    metadata = _load_metadata(job, require_review=True)
    return {
        "status": "candidate_preflight_passed",
        "kind": "engineering_preflight_not_scoring",
        "expected_mode": job.values["expected_mode"],
        "expected_kind": job.values["expected_kind"],
        "candidate_config_sha256": job.fingerprint,
        "candidate_best_checkpoint_sha256": job.values["candidate_best_checkpoint_sha256"],
        "benchmark_fingerprints": metadata.bundle.fingerprints,
        "approval_status": metadata.bundle.approval["status"],
        "dataset_opened": False,
        "checkpoint_loaded": False,
        "model_inference_run": False,
        "sensor_queried": False,
        "test_evaluated": False,
    }


def _validate_checkpoints(job: CandidateEvaluationConfig, metadata: CandidateMetadata,
                          best: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    config = metadata.bundle.protocol.base
    mode = job.values["expected_mode"]
    def semantic_config(value: Mapping[str, Any]) -> dict[str, Any]:
        """Compare every setting except paths already verified by content identity."""
        result = copy.deepcopy(dict(value))
        result["experiment"].pop("output_dir")
        result["data"].pop("root")
        result["model"].pop("weights_cache")
        return result

    for label, checkpoint in (("best", best), ("last", last)):
        check_checkpoint(config, checkpoint)
        if (checkpoint.get("training_flavor") != "matched_robust_baseline_v1"
                or checkpoint.get("robust_mode") != mode
                or checkpoint.get("kind") != job.values["expected_kind"]
                or not isinstance(checkpoint.get("config"), dict)
                or semantic_config(checkpoint["config"]) != semantic_config(config.values)
                or checkpoint.get("corruption_protocol_sha256") != metadata.bundle.protocol.fingerprint
                or checkpoint.get("file_sha256", {}).get("val") != metadata.bundle.manifest["file_sha256"]
                or checkpoint.get("manifests", {}).get("val") != metadata.bundle.manifest["inventory"]):
            raise ValueError(f"Candidate {label} checkpoint differs from the frozen protocol")
        if set(checkpoint.get("file_sha256", {})) != {"train", "val"}:
            raise ValueError("Candidate checkpoint must contain development splits only")
        if any("test" in row for row in checkpoint.get("history", [])):
            raise ValueError("Candidate history unexpectedly contains test results")
    status = metadata.candidate_status
    if (last.get("epoch") != status["epochs"]
            or len(last.get("history", [])) != status["epochs"]
            or last.get("best_epoch") != status["best_epoch"]
            or last.get("best_val_f1") != status["best_val_f1"]):
        raise ValueError("Last checkpoint does not match the completed training status")
    if (best.get("epoch") != best.get("best_epoch")
            or best.get("best_epoch") != status["best_epoch"]
            or best.get("best_val_f1") != status["best_val_f1"]
            or best.get("history") != last["history"][:best["epoch"]]
            or best.get("training_signature") != last.get("training_signature")
            or best.get("initial_model_tensor_sha256") != last.get("initial_model_tensor_sha256")):
        raise ValueError("Best checkpoint is not the clean-validation selection from this run")
    expected_clean = _expected_clean(best)
    if expected_clean["metrics"]["f1"] != status["best_val_f1"]:
        raise ValueError("Candidate clean validation record differs from status")
    expected_exposure = _exposure_summary(last["history"])
    if expected_exposure != metadata.exposure:
        raise ValueError("Exposure summary differs from the completed checkpoint history")
    robust = last.get("robust_config", {})
    source_tiles = robust.get("expected_source_tiles")
    batches = robust.get("expected_batches_per_epoch")
    if (type(source_tiles) is not int or type(batches) is not int
            or status["pair_views"] != 2 * source_tiles * status["epochs"]
            or sum(metadata.exposure["counts"].values()) != source_tiles * status["epochs"]
            or status["optimizer_steps"] + status["amp_skipped_steps"] != batches * status["epochs"]):
        raise ValueError("Completed exposure/update counters violate the matched protocol")
    return expected_clean


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = (*METRIC_KEYS, "foreground_recall", "background_fpr", "foreground_fn_rate")
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def compare_results(candidate: Mapping[str, dict[str, Any]],
                    baseline: Mapping[str, dict[str, Any]],
                    expected_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return condition deltas and original-image-pair grouped deltas."""
    if set(candidate) != set(expected_ids) or set(baseline) != set(expected_ids):
        raise ValueError("Candidate/B0 comparison condition coverage differs")
    per_condition: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for condition_id in expected_ids:
        cand, base = candidate[condition_id], baseline[condition_id]
        if set(cand["by_source"]) != set(base["by_source"]):
            raise ValueError(f"Candidate/B0 source coverage differs: {condition_id}")
        per_condition[condition_id] = {
            "candidate_counts": cand["counts"],
            "baseline_counts": base["counts"],
            "count_delta_candidate_minus_baseline": {
                key: int(cand["counts"][key]) - int(base["counts"][key]) for key in COUNT_KEYS},
            "candidate_metrics": cand["metrics"],
            "baseline_metrics": base["metrics"],
            "metric_delta_candidate_minus_baseline": _metric_delta(cand["metrics"], base["metrics"]),
        }
        paired[condition_id] = {}
        for source_id in sorted(cand["by_source"]):
            cand_source, base_source = cand["by_source"][source_id], base["by_source"][source_id]
            paired[condition_id][source_id] = {
                "candidate_counts": cand_source["counts"],
                "baseline_counts": base_source["counts"],
                "candidate_metrics": cand_source["metrics"],
                "baseline_metrics": base_source["metrics"],
                "metric_delta_candidate_minus_baseline": _metric_delta(
                    cand_source["metrics"], base_source["metrics"]),
            }
    return per_condition, paired


def _snapshot(job: CandidateEvaluationConfig, output: Path,
              code_hashes: dict[str, str], files: dict[str, Path]) -> None:
    (output / "candidate_evaluation_source.yaml").write_bytes(job.raw)
    resolved = copy.deepcopy(dict(job.values))
    for key in _PATH_KEYS:
        resolved[key] = str(job.path(key))
    with (output / "candidate_evaluation_resolved.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False, allow_unicode=True)
    for name, path in files.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != code_hashes[name]:
            raise ValueError(f"Evaluator source changed during snapshot: {name}")
    write_json(output / "source_sha256.json", code_hashes)


def evaluate(job: CandidateEvaluationConfig,
             guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    metadata = _load_metadata(job, require_review=True)
    output = job.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate validation evaluation: {output}")
    files = source_files()
    code_hashes = {name: file_sha256(path) for name, path in files.items()}
    dataset, _ = build_validation_dataset(metadata.bundle)
    config = metadata.bundle.protocol.base
    if dataset.split != "val" or config.run_test_after_training:
        raise ValueError("Candidate evaluation is restricted to development validation")
    best = read_checkpoint(job.path("candidate_best_checkpoint"))
    last = read_checkpoint(job.path("candidate_last_checkpoint"))
    expected_clean = _validate_checkpoints(job, metadata, best, last)
    _bind_validation_files(metadata.input_hashes, metadata.bundle, dataset, best)
    _verify_input_hashes(metadata.input_hashes)
    device = setup_runtime(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {
        "status": "evaluating_clean", "mode": job.values["expected_mode"],
        "test_evaluated": False, "training_started": False})
    try:
        _snapshot(job, output, code_hashes, files)
        write_json(output / "provenance.json", {
            "candidate_evaluation_config_sha256": job.fingerprint,
            "input_file_sha256": metadata.input_hashes,
            "benchmark_fingerprints": metadata.bundle.fingerprints,
            "approval": metadata.bundle.approval,
            "candidate_mode": job.values["expected_mode"],
            "candidate_training_signature": best["training_signature"],
            "candidate_initial_model_tensor_sha256": best["initial_model_tensor_sha256"],
            "candidate_source_sha256": best["source_sha256"],
            "evaluation_source_sha256": code_hashes,
            "selection": "clean_validation_f1_only",
            "test_evaluated": False,
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
        clean = run_condition(model, make_loader(dataset, config, training=False), config, device,
                              guard_callback=guard_callback)
        clean["matches_candidate_checkpoint"] = (
            clean["samples"] == expected_clean["samples"]
            and all(clean["counts"][key] == expected_clean["metrics"][key] for key in COUNT_KEYS)
            and all(clean["metrics"][key] == expected_clean["metrics"][key] for key in METRIC_KEYS))
        clean["checkpoint_metrics"] = expected_clean["metrics"]
        write_json(output / "clean_metrics.json", clean)
        if not clean["matches_candidate_checkpoint"]:
            raise ValueError("Candidate clean validation no longer matches its selected checkpoint")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for instance in metadata.bundle.instances:
            grouped.setdefault(instance["condition_id"], []).append(instance)
        expected_ids = list(metadata.bundle.manifest["condition_counts"])
        if set(grouped) != set(expected_ids):
            raise ValueError("Frozen candidate instances have incomplete condition coverage")
        results: dict[str, dict[str, Any]] = {}
        (output / "conditions").mkdir()
        for condition_id in expected_ids:
            cell = FrozenInstanceDataset(dataset, grouped[condition_id])
            result = run_condition(model, make_loader(cell, config, training=False), config, device,
                                   guard_callback=guard_callback)
            if (result["condition_id"] != condition_id
                    or result["samples"] != metadata.bundle.manifest["condition_counts"][condition_id]
                    or result["pixels"] != result["samples"] * dataset.tile_size ** 2):
                raise ValueError(f"Incomplete candidate condition: {condition_id}")
            expected_strata = {key[len(condition_id) + 1:]: value
                               for key, value in metadata.bundle.manifest["stratum_counts"].items()
                               if key.startswith(condition_id + "|")}
            actual_strata = {key: value["samples"] for key, value in result["joint_strata"].items()}
            if actual_strata != expected_strata:
                raise ValueError(f"Candidate stratum coverage differs: {condition_id}")
            result["clean_reference_f1"] = clean["metrics"]["f1"]
            result["clean_minus_f1"] = clean["metrics"]["f1"] - result["metrics"]["f1"]
            write_json(output / "conditions" / f"{condition_id}.json", result)
            results[condition_id] = result
            write_json(output / "status.json", {
                "status": "evaluating_corruptions", "mode": job.values["expected_mode"],
                "completed_conditions": len(results), "condition_count": len(expected_ids),
                "test_evaluated": False, "training_started": False}, overwrite=True)
            print(f"{job.values['expected_mode']}/{condition_id}: "
                  f"f1={result['metrics']['f1']:.6f} ({len(results)}/{len(expected_ids)})",
                  flush=True)
        write_json(output / "conditions.json", results)
        aggregate = summarize_conditions(results, expected_ids)
        diagnostics = _condition_diagnostics(results, expected_ids)
        baseline_diagnostics = _condition_diagnostics(metadata.baseline_conditions, expected_ids)
        per_condition, paired = compare_results(
            results, metadata.baseline_conditions, expected_ids)
        write_json(output / "comparison_by_condition.json", per_condition)
        write_json(output / "paired_by_source.json", paired)
        family_delta = {kind: aggregate["family_mf1"][kind]
                        - metadata.baseline_summary["family_mf1"][kind]
                        for kind in aggregate["family_mf1"]}
        summary = {
            "status": "completed",
            "kind": job.values["expected_kind"],
            "mode": "frozen_candidate_corruption_validation",
            "candidate_mode": job.values["expected_mode"],
            "split": "val",
            "candidate_checkpoint_epoch": best["epoch"],
            "selection": "clean_validation_f1_only",
            "clean_f1": clean["metrics"]["f1"],
            "baseline_clean_f1": metadata.baseline_summary["clean_f1"],
            "clean_f1_delta_candidate_minus_baseline": (
                clean["metrics"]["f1"] - metadata.baseline_summary["clean_f1"]),
            **aggregate,
            "baseline_corruption_mf1": metadata.baseline_summary["corruption_mf1"],
            "corruption_mf1_delta_candidate_minus_baseline": (
                aggregate["corruption_mf1"] - metadata.baseline_summary["corruption_mf1"]),
            "clean_minus_corruption_mf1": clean["metrics"]["f1"] - aggregate["corruption_mf1"],
            "baseline_clean_minus_corruption_mf1": (
                metadata.baseline_summary["clean_minus_corruption_mf1"]),
            "degradation_delta_candidate_minus_baseline": (
                (clean["metrics"]["f1"] - aggregate["corruption_mf1"])
                - metadata.baseline_summary["clean_minus_corruption_mf1"]),
            "family_mf1_delta_candidate_minus_baseline": family_delta,
            "candidate_diagnostics": diagnostics,
            "baseline_diagnostics": baseline_diagnostics,
            "test_evaluated": False,
            "training_started": False,
            "guarded_completion_required": True,
        }
        if source_hashes() != code_hashes:
            raise ValueError("Evaluation source changed during candidate scoring")
        _verify_input_hashes(metadata.input_hashes)
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "mode": job.values["expected_mode"],
            "error": f"{type(error).__name__}: {error}",
            "test_evaluated": False, "training_started": False,
        }, overwrite=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    job = load_candidate_evaluation_config(args.config)
    result = preflight(job) if args.preflight_only else evaluate(job)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
