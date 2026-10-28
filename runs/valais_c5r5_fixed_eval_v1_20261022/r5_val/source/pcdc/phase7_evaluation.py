"""Pre-validation seal and one-shot official-val evaluation for Phase 7."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import file_sha256, make_loader
from .frozen_validation import (
    FrozenInstanceDataset,
    build_validation_dataset,
    load_bundle,
    load_evaluation_config,
)
from .phase5_models import per_sample_change_loss
from .phase6_diagnostic import (
    COUNT_KEYS,
    _artifact_hashes as phase6_artifact_hashes,
    _condition_order,
    _load_models as load_phase6_models,
    _metrics_from_counts,
    per_sample_confusion,
    per_sample_transitions,
)
from .phase7_cache import load_descriptor_cache
from .phase7_models import build_phase7_selector, phase7_descriptor
from .phase7_protocol import Phase7Config
from .phase7_training import VARIANTS, _scores, _threshold_value
from .runtime import environment_record, setup_runtime, write_json


POLICIES = (
    "A0", "G0", "C1-U", "C1-H", "G1-U", "G1-H",
    "G2-U", "G2-H", "G3-U", "G3-H", "O",
)
LEARNED_POLICIES = POLICIES[2:10]
TRANSITION_KEYS = ("fp_to_tn", "fn_to_tp", "tp_to_fn", "tn_to_fp")


def _sha(path: Path) -> str:
    return file_sha256(path)


def _source_id(sample_id: str) -> str:
    return sample_id.split("@", 1)[0]


def _metric(counts: Iterable[int]) -> dict[str, Any]:
    result = _metrics_from_counts(counts)
    denominator = result["tn"] + result["fp"]
    result["background_fpr"] = result["fp"] / denominator if denominator else 0.0
    return result


def phase7_evaluation_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/corruption_protocol.py", "pcdc/data.py",
        "pcdc/engine.py", "pcdc/frozen_validation.py", "pcdc/model.py",
        "pcdc/perturbations.py", "pcdc/phase5_models.py",
        "pcdc/phase5_protocol.py", "pcdc/phase5_training.py",
        "pcdc/phase6_diagnostic.py", "pcdc/phase7_protocol.py",
        "pcdc/phase7_models.py", "pcdc/phase7_cache.py",
        "pcdc/phase7_training.py", "pcdc/phase7_evaluation.py",
        "pcdc/runtime.py", "scripts/evaluate_guarded.py",
        "scripts/temperature_guard.py", "scripts/train_robust_guarded.py",
        "scripts/train_phase7_guarded.py", "scripts/cache_phase7_guarded.py",
        "scripts/smoke_phase7.py", "scripts/seal_phase7.py",
        "scripts/evaluate_phase7_guarded.py",
    )
    return {name: root / name for name in names}


def _validation_job_path(config: Phase7Config) -> Path:
    return (config.source.parent / "evaluate_corruptions_val_v1.yaml").resolve()


def load_validation_metadata(config: Phase7Config):
    job_path = _validation_job_path(config)
    job = load_evaluation_config(job_path)
    if (job.path("manifest") != config.upstream_paths["validation_manifest"]
            or job.values["manifest_sha256"] != config.values["upstream"][
                "validation_manifest_sha256"]):
        raise ValueError("Phase 7 validation job does not bind the frozen manifest")
    bundle = load_bundle(job, require_review=True)
    if (bundle.protocol.base.values != config.base.values
            or bundle.manifest["inventory"]["source_pairs"] != 64
            or bundle.manifest["inventory"]["tiles"] != 1024
            or bundle.manifest["instance_count"] != 46080
            or list(bundle.manifest["condition_counts"]) != _condition_order()):
        raise ValueError("Phase 7 validation metadata differs from the approved protocol")
    binding = {
        "evaluation_job_path": str(job_path),
        "evaluation_job_sha256": _sha(job_path),
        "acceptance_sha256": _sha(job.path("acceptance")),
        "manifest_sha256": _sha(bundle.manifest_path),
        "instances_sha256": _sha(bundle.manifest_path.parent / bundle.manifest["instance_file"]),
        "corruption_protocol_sha256": _sha(bundle.protocol.source),
        "preparation_environment_sha256": _sha(
            bundle.manifest_path.parent / "environment.json"),
    }
    return bundle, binding


def _selector_artifacts(config: Phase7Config) -> dict[str, Any]:
    result = {}
    for variant in VARIANTS:
        output = config.training_output(variant)
        paths = {name: output / filename for name, filename in (
            ("status", "status.json"), ("guard_completion", "guard_completion.json"),
            ("best_checkpoint", "best.pt"), ("last_checkpoint", "last.pt"),
            ("policy_calibration", "policy_calibration.json"),
            ("normalization", "normalization.json"),
            ("run_binding", "run_binding.json"), ("source_sha256", "source_sha256.json"),
        )}
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Incomplete Phase 7 selector artifacts: {variant}")
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        guard = json.loads(paths["guard_completion"].read_text(encoding="utf-8"))
        calibration = json.loads(paths["policy_calibration"].read_text(encoding="utf-8"))
        if (status.get("status") != "completed" or status.get("variant") != variant
                or status.get("epochs") != 100 or status.get("attempted_updates") != 20000
                or status.get("successful_updates") != 20000
                or status.get("validation_evaluated") is not False
                or status.get("test_evaluated") is not False
                or guard.get("status") != "completed" or guard.get("variant") != variant
                or guard.get("training_status_sha256") != _sha(paths["status"])
                or calibration.get("status") != "calibrated"
                or calibration.get("variant") != variant
                or calibration.get("test_evaluated") is not False):
            raise ValueError(f"Phase 7 selector is not a complete guarded run: {variant}")
        hashes = {name: _sha(path) for name, path in paths.items()}
        if (status.get("best_checkpoint_sha256") != hashes["best_checkpoint"]
                or status.get("last_checkpoint_sha256") != hashes["last_checkpoint"]
                or status.get("policy_calibration_sha256") != hashes["policy_calibration"]):
            raise ValueError(f"Phase 7 selector artifact hash differs: {variant}")
        result[variant] = {
            "output_dir": str(output), "best_epoch": status["best_epoch"],
            "best_calibration_primary_metric": status["best_calibration_primary_metric"],
            "hashes": hashes,
        }
    return result


def create_preval_seal(config: Phase7Config) -> dict[str, Any]:
    output = config.paths["preval_seal"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 pre-val seal: {output}")
    if config.paths["validation_output_dir"].exists():
        raise ValueError("Validation output exists before the Phase 7 pre-val seal")
    cache = load_descriptor_cache(config)
    selectors = _selector_artifacts(config)
    _, validation_binding = load_validation_metadata(config)
    source_files = phase7_evaluation_source_files()
    if any(not path.is_file() for path in source_files.values()):
        missing = [name for name, path in source_files.items() if not path.is_file()]
        raise FileNotFoundError(f"Phase 7 evaluation source is missing: {missing}")
    source_hashes = {name: _sha(path) for name, path in source_files.items()}
    seal = {
        "schema_version": 1, "status": "sealed_before_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase7_fingerprint": config.fingerprint,
        "phase7_config_sha256": config.config_sha256,
        "approval_sha256": _sha(config.paths["approval"]),
        "frozen_protocol_document_sha256": _sha(config.paths["frozen_protocol_document"]),
        "selector_split_manifest_sha256": _sha(config.paths["selector_split_manifest"]),
        "descriptor_cache": {
            "status_sha256": _sha(cache["output"] / "status.json"),
            "guard_completion_sha256": _sha(cache["output"] / "guard_completion.json"),
            "normalization_sha256": _sha(cache["output"] / "normalization.json"),
        },
        "selectors": selectors, "validation_binding": validation_binding,
        "evaluation_variants": list(POLICIES), "source_sha256": source_hashes,
        "validation_evaluated": False, "test_evaluated": False,
        "scope": "One official Phase 7 validation run only; held-out test and Phase 8-10 remain closed.",
    }
    write_json(output, seal)
    return {**seal, "preval_seal_sha256": _sha(output)}


def load_preval_seal(config: Phase7Config) -> dict[str, Any]:
    path = config.paths["preval_seal"]
    if not path.is_file():
        raise FileNotFoundError("Phase 7 official validation requires the pre-val seal")
    seal = json.loads(path.read_text(encoding="utf-8"))
    cache = load_descriptor_cache(config)
    selectors = _selector_artifacts(config)
    _, validation_binding = load_validation_metadata(config)
    expected_sources = {name: _sha(source) for name, source in
                        phase7_evaluation_source_files().items()}
    if (seal.get("schema_version") != 1
            or seal.get("status") != "sealed_before_validation"
            or seal.get("phase7_fingerprint") != config.fingerprint
            or seal.get("phase7_config_sha256") != config.config_sha256
            or seal.get("approval_sha256") != _sha(config.paths["approval"])
            or seal.get("frozen_protocol_document_sha256") != _sha(
                config.paths["frozen_protocol_document"])
            or seal.get("selector_split_manifest_sha256") != _sha(
                config.paths["selector_split_manifest"])
            or seal.get("evaluation_variants") != list(POLICIES)
            or seal.get("validation_binding") != validation_binding
            or seal.get("source_sha256") != expected_sources
            or seal.get("selectors") != selectors
            or seal.get("validation_evaluated") is not False
            or seal.get("test_evaluated") is not False):
        raise ValueError("Phase 7 pre-val seal differs from current frozen artifacts")
    expected_cache = {
        "status_sha256": _sha(cache["output"] / "status.json"),
        "guard_completion_sha256": _sha(cache["output"] / "guard_completion.json"),
        "normalization_sha256": _sha(cache["output"] / "normalization.json"),
    }
    if seal.get("descriptor_cache") != expected_cache:
        raise ValueError("Phase 7 descriptor cache changed after the pre-val seal")
    return seal


def preflight(config: Phase7Config) -> dict[str, Any]:
    seal = load_preval_seal(config)
    return {
        "status": "phase7_official_val_preflight_passed",
        "kind": "metadata_only_no_data_model_or_device",
        "phase7_fingerprint": config.fingerprint,
        "preval_seal_sha256": _sha(config.paths["preval_seal"]),
        "evaluation_variants": seal["evaluation_variants"],
        "dataset_opened": False, "model_initialized": False,
        "inference_started": False, "sensor_queried": False,
        "validation_evaluated": False, "test_evaluated": False,
    }


def _load_selectors(config: Phase7Config, device: torch.device,
                    normalization: Mapping[str, Any]) -> tuple[dict[str, nn.Module], dict[str, float]]:
    models, thresholds = {}, {}
    for variant in VARIANTS:
        output = config.training_output(variant)
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(output / "best.pt", map_location=device)
        calibration = json.loads((output / "policy_calibration.json").read_text(
            encoding="utf-8"))
        if (checkpoint.get("training_flavor") != "phase7_selector_v1"
                or checkpoint.get("variant") != variant
                or checkpoint.get("phase7_fingerprint") != config.fingerprint
                or checkpoint.get("epoch") != status["best_epoch"]):
            raise ValueError(f"Invalid sealed Phase 7 selector checkpoint: {variant}")
        model = build_phase7_selector(config.values["selector"]["seed"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError(f"Non-finite Phase 7 selector parameter: {variant}")
        models[variant] = model
        thresholds[variant] = _threshold_value(
            calibration["harm_constrained"]["threshold"])
    descriptor = normalization["descriptor"]
    if (len(descriptor["mean"]) != 1024 or len(descriptor["std"]) != 1024
            or not np.isfinite(descriptor["mean"]).all()
            or not np.isfinite(descriptor["std"]).all()
            or np.min(descriptor["std"]) < 1e-6):
        raise ValueError("Invalid sealed Phase 7 descriptor normalization")
    return models, thresholds


def _timed_sync(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


def _batch_inference(detector, selectors: Mapping[str, nn.Module],
                     normalization: Mapping[str, Any], image_a: Tensor, image_b: Tensor,
                     *, amp: bool, timings: dict[str, float]):
    device = image_a.device
    begin = _timed_sync(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        features_a, features_b = detector._encode(image_a, image_b)
    split = _timed_sync(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        descriptor = phase7_descriptor(features_a, features_b, detector.adapter)
    after_descriptor = _timed_sync(device)
    mean = torch.as_tensor(normalization["descriptor"]["mean"], dtype=torch.float32,
                           device=device)
    std = torch.as_tensor(normalization["descriptor"]["std"], dtype=torch.float32,
                          device=device)
    normalized = (descriptor.float() - mean) / std
    with torch.no_grad():
        outputs = {variant: model(normalized) for variant, model in selectors.items()}
    after_selector = _timed_sync(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        size = tuple(image_a.shape[-2:])
        a0 = detector._decode(features_a, features_b, size, apply_adapter=False)
        a1 = detector._decode(features_a, features_b, size, apply_adapter=True)
    end = _timed_sync(device)
    timings["encoder_seconds"] += split - begin
    timings["descriptor_adapter_seconds"] += after_descriptor - split
    timings["four_selectors_seconds"] += after_selector - after_descriptor
    timings["both_decoders_seconds"] += end - after_selector
    timings["model_compute_seconds"] += end - begin
    timings["batches"] += 1
    return outputs, a0, a1


def _decision_matrix(outputs: Mapping[str, Tensor], thresholds: Mapping[str, float],
                     normalization: Mapping[str, Any], utility: np.ndarray
                     ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    samples = len(utility)
    decisions = np.zeros((samples, len(POLICIES)), dtype=bool)
    decisions[:, 1] = True
    score_rows: dict[str, dict[str, np.ndarray]] = {}
    raw_rows: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        raw = outputs[variant].detach().cpu().numpy().astype(np.float64, copy=False)
        primary, constrained = _scores(variant, raw, normalization)
        name = variant.upper()
        decisions[:, POLICIES.index(f"{name}-U")] = primary > 0.0
        decisions[:, POLICIES.index(f"{name}-H")] = constrained > thresholds[variant]
        score_rows[variant] = {"primary": primary, "constrained": constrained}
        raw_rows[variant] = raw
    decisions[:, -1] = utility > 0.0
    return decisions, score_rows, raw_rows


def _transition_summary(transitions: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    totals = np.asarray(transitions, dtype=np.int64).sum(axis=0)
    base = np.asarray(baseline, dtype=np.int64).sum(axis=0)
    denominator_indices = (1, 2, 3, 0)
    return {name: {
        "count": int(totals[index]),
        "denominator": int(base[denominator_indices[index]]),
        "rate": (float(totals[index] / base[denominator_indices[index]])
                 if base[denominator_indices[index]] else 0.0),
    } for index, name in enumerate(TRANSITION_KEYS)}


class _StrataAccumulator:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, dict[str, Any]]] = {
            name: {} for name in (
                "family", "level", "source", "target_date", "spatial_mode",
                "direction", "injection_overlap")}

    def update(self, keys: Mapping[str, str], selected: np.ndarray,
               chosen_counts: np.ndarray) -> None:
        for group, key in keys.items():
            row = self.rows[group].setdefault(key, {
                "instances": 0, "selected": np.zeros(len(POLICIES), dtype=np.int64),
                "counts": np.zeros((len(POLICIES), 4), dtype=np.int64),
            })
            row["instances"] += 1
            row["selected"] += selected.astype(np.int64)
            row["counts"] += chosen_counts

    def finish(self) -> dict[str, Any]:
        result = {}
        for group, rows in self.rows.items():
            result[group] = {}
            for key, row in sorted(rows.items()):
                result[group][key] = {
                    "instances": row["instances"],
                    "policies": {policy: {
                        "a1_selected": int(row["selected"][index]),
                        "a1_selection_rate": float(
                            row["selected"][index] / row["instances"]),
                        "metrics": _metric(row["counts"][index]),
                    } for index, policy in enumerate(POLICIES)},
                }
        return result


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _selector_diagnostic(variant: str, raw: np.ndarray, score: np.ndarray,
                         utility: np.ndarray, normalization: Mapping[str, Any]) -> dict[str, Any]:
    label = utility > 0
    predicted = score > 0
    positive_accuracy = float(predicted[label].mean()) if label.any() else 0.0
    negative_accuracy = float((~predicted[~label]).mean()) if (~label).any() else 0.0
    order = np.argsort(score, kind="mergesort")
    curve = []
    for bin_index, indices in enumerate(np.array_split(order, 10)):
        if not len(indices):
            continue
        row = {
            "bin": bin_index, "count": int(len(indices)),
            "score_mean": float(score[indices].mean()),
            "utility_mean": float(utility[indices].mean()),
            "positive_fraction": float(label[indices].mean()),
        }
        if variant == "c1":
            row["predicted_probability_mean"] = float(
                (1.0 / (1.0 + np.exp(-np.clip(score[indices], -50, 50)))).mean())
        curve.append(row)
    result = {
        "samples": len(utility), "sign_accuracy": float((predicted == label).mean()),
        "balanced_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
        "auroc": _auc(score, label), "calibration_curve_equal_count_bins": curve,
    }
    if variant == "c1":
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
        result.update(regression_mae=None, standardized_smooth_l1=None,
                      binary_log_loss=float(-np.mean(
                          label * np.log(np.maximum(probability, 1e-12))
                          + (~label) * np.log(np.maximum(1 - probability, 1e-12)))))
    else:
        stats = normalization["targets"]["u_obs"]
        error = (score - utility) / float(stats["std"])
        absolute = np.abs(error)
        smooth = np.where(absolute < 1.0, 0.5 * np.square(error), absolute - 0.5)
        result.update(regression_mae=float(np.abs(score - utility).mean()),
                      standardized_smooth_l1=float(smooth.mean()), binary_log_loss=None)
        if variant in ("g2", "g3"):
            name = "u_clean" if variant == "g2" else "a_obs"
            aux_stats = normalization["targets"][name]
            aux = raw[:, 1] * aux_stats["std"] + aux_stats["mean"]
            result["auxiliary_prediction"] = {
                "target": name, "prediction_mean": float(aux.mean()),
                "prediction_std": float(aux.std()),
            }
    return result


def _interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = (1.0 - confidence) / 2.0
    return np.quantile(values, [alpha, 0.5, 1.0 - alpha]).tolist()


def _bootstrap(config: Phase7Config, corruption_counts: np.ndarray,
               clean_counts: np.ndarray, regret_sum: np.ndarray,
               regret_count: np.ndarray, better_index: int) -> dict[str, Any]:
    settings = config.values["evaluation"]
    replicates = settings["bootstrap_replicates"]
    sources = corruption_counts.shape[0]
    rng = np.random.Generator(np.random.PCG64(settings["bootstrap_seed"]))
    mf1 = np.empty((replicates, len(POLICIES)), dtype=np.float64)
    clean_f1 = np.empty_like(mf1)
    clean_recall = np.empty_like(mf1)
    regret = np.empty_like(mf1)
    cursor = 0
    while cursor < replicates:
        batch = min(250, replicates - cursor)
        weights = rng.multinomial(sources, np.full(sources, 1 / sources), size=batch)
        pooled = np.einsum("bs,scpk->bcpk", weights, corruption_counts, optimize=True)
        denominator = 2 * pooled[..., 3] + pooled[..., 1] + pooled[..., 2]
        f1 = np.divide(2 * pooled[..., 3], denominator,
                       out=np.zeros_like(denominator, dtype=np.float64), where=denominator != 0)
        mf1[cursor:cursor + batch] = f1.mean(axis=1)
        clean = np.einsum("bs,spk->bpk", weights, clean_counts, optimize=True)
        clean_denominator = 2 * clean[..., 3] + clean[..., 1] + clean[..., 2]
        clean_f1[cursor:cursor + batch] = np.divide(
            2 * clean[..., 3], clean_denominator,
            out=np.zeros_like(clean_denominator, dtype=np.float64),
            where=clean_denominator != 0)
        recall_denominator = clean[..., 3] + clean[..., 2]
        clean_recall[cursor:cursor + batch] = np.divide(
            clean[..., 3], recall_denominator,
            out=np.zeros_like(recall_denominator, dtype=np.float64),
            where=recall_denominator != 0)
        regret_numerator = np.einsum("bs,sp->bp", weights, regret_sum, optimize=True)
        regret_denominator = np.einsum("bs,sp->bp", weights, regret_count, optimize=True)
        regret[cursor:cursor + batch] = regret_numerator / regret_denominator
        cursor += batch
    confidence = settings["confidence_level"]
    learned = {policy: _interval(mf1[:, index] - mf1[:, better_index], confidence)
               for index, policy in enumerate(POLICIES) if policy in LEARNED_POLICIES}
    pairwise = {}
    for left in LEARNED_POLICIES:
        for right in LEARNED_POLICIES:
            if left == right:
                continue
            pairwise[f"{left}_minus_{right}"] = _interval(
                mf1[:, POLICIES.index(left)] - mf1[:, POLICIES.index(right)], confidence)
    return {
        "unit": "original_source_pair", "replicates": replicates,
        "seed": settings["bootstrap_seed"], "method": "percentile",
        "interval_order": ["lower", "median", "upper"],
        "learned_mf1_minus_better_constant": learned,
        "clean_f1_minus_A0": {policy: _interval(
            clean_f1[:, index] - clean_f1[:, 0], confidence)
            for index, policy in enumerate(POLICIES)},
        "clean_recall_minus_A0": {policy: _interval(
            clean_recall[:, index] - clean_recall[:, 0], confidence)
            for index, policy in enumerate(POLICIES)},
        "O_mf1_minus_better_constant": _interval(
            mf1[:, -1] - mf1[:, better_index], confidence),
        "mean_regret": {policy: _interval(regret[:, index], confidence)
                        for index, policy in enumerate(POLICIES)},
        "pairwise_learned_mf1": pairwise,
    }


def _decision(config: Phase7Config, actions: Mapping[str, Any],
              bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.values["evaluation"]
    better = max(("A0", "G0"), key=lambda name: actions[name]["corruption_mf1"])
    passing = []
    gates = {}
    for policy in LEARNED_POLICIES:
        delta = actions[policy]["corruption_mf1"] - actions[better]["corruption_mf1"]
        ci = bootstrap["learned_mf1_minus_better_constant"][policy]
        checks = {
            "mf1_gain": delta >= settings["min_selector_mf1_gain"],
            "mf1_gain_ci_lower_positive": ci[0] > 0,
            "clean_f1": actions[policy]["clean"]["f1"]
            >= actions["A0"]["clean"]["f1"] - settings["clean_f1_tolerance"],
            "clean_recall": actions[policy]["clean"]["recall"]
            >= actions["A0"]["clean"]["recall"] - settings["clean_recall_tolerance"],
            "corruption_recall": actions[policy]["pooled"]["recall"]
            >= actions[better]["pooled"]["recall"] - settings["corruption_recall_tolerance"],
            "transitions_and_regret_finite": actions[policy][
                "transitions_a0_to_policy"]["fully_reported_and_finite"],
        }
        gates[policy] = {"passes": all(checks.values()), "checks": checks,
                         "mf1_gain_over_better_constant": delta,
                         "mf1_gain_bootstrap": ci}
        if all(checks.values()):
            passing.append(policy)

    def best(names: tuple[str, ...]) -> str | None:
        eligible = [name for name in names if name in passing]
        return max(eligible, key=lambda name: actions[name]["corruption_mf1"]) if eligible else None

    ordinary = best(("C1-U", "C1-H", "G1-U", "G1-H"))
    paired = best(("G2-U", "G2-H", "G3-U", "G3-H"))
    paired_rule = {"ordinary": ordinary, "paired": paired, "supported": False}
    if ordinary and paired:
        delta = actions[paired]["corruption_mf1"] - actions[ordinary]["corruption_mf1"]
        ci = bootstrap["pairwise_learned_mf1"][f"{paired}_minus_{ordinary}"]
        paired_rule.update(delta=delta, bootstrap=ci, supported=(
            delta >= settings["min_paired_target_mf1_gain"] and ci[0] > 0))

    equivalence_rows = {}
    g3_supported = False
    for suffix in ("U", "H"):
        g3, g2 = f"G3-{suffix}", f"G2-{suffix}"
        delta = actions[g3]["corruption_mf1"] - actions[g2]["corruption_mf1"]
        ci = bootstrap["pairwise_learned_mf1"][f"{g3}_minus_{g2}"]
        supported = (g3 in passing and delta >= settings["min_g3_over_g2_mf1_gain"]
                     and ci[0] > 0)
        g3_supported |= supported
        equivalence_rows[suffix] = {"delta": delta, "bootstrap": ci,
                                    "g3_advantage_supported": supported}
    return {
        "status": "completed", "better_constant": better,
        "selector_feasibility": {"passing_policies": passing, "policies": gates,
            "decision": ("pass_to_separately_reviewed_phase8_protocol"
                         if passing else "stop_whole_tile_selector_route")},
        "paired_auxiliary_rule": paired_rule,
        "g2_g3_equivalence_rule": {
            "matched": equivalence_rows, "g3_advantage_supported": g3_supported,
            "conclusion": ("g3_parameterization_advantage_supported" if g3_supported
                           else "g2_g3_empirically_equivalent_parameterizations")},
        "automatic_next_phase_authorized": False, "test_evaluated": False,
    }


def _snapshot(config: Phase7Config, output: Path, device: torch.device,
              seal: Mapping[str, Any], source_hashes: Mapping[str, str]) -> None:
    (output / "phase7_config.yaml").write_bytes(config.raw)
    for key, name in (("approval", "approval.yaml"),
                      ("frozen_protocol_document", "frozen_protocol_document.md"),
                      ("selector_split_manifest", "selector_split_manifest.json"),
                      ("preval_seal", "preval_seal.json")):
        shutil.copyfile(config.paths[key], output / name)
    write_json(output / "source_sha256.json", dict(source_hashes))
    write_json(output / "environment.json", environment_record(config.base, device))
    write_json(output / "run_binding.json", {
        "phase7_fingerprint": config.fingerprint,
        "phase7_config_sha256": config.config_sha256,
        "preval_seal_sha256": _sha(config.paths["preval_seal"]),
        "validation_binding": seal["validation_binding"],
        "test_evaluated": False,
    })
    for name, path in phase7_evaluation_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if _sha(target) != source_hashes[name]:
            raise ValueError(f"Phase 7 evaluation source changed while copying: {name}")


def evaluate(config: Phase7Config,
             guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    seal = load_preval_seal(config)
    output = config.paths["validation_output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 official val output: {output}")
    source_hashes = {name: _sha(path) for name, path in
                     phase7_evaluation_source_files().items()}
    device = setup_runtime(config.base)
    bundle, validation_binding = load_validation_metadata(config)
    if validation_binding != seal["validation_binding"]:
        raise ValueError("Validation binding changed after pre-val seal")
    cache = load_descriptor_cache(config)
    normalization = cache["normalization"]
    detector, actor_checkpoint, adapter_checkpoint = load_phase6_models(
        config.phase56, phase6_artifact_hashes(config.phase56), device)
    selectors, thresholds = _load_selectors(config, device, normalization)
    output.mkdir(parents=True, exist_ok=False)
    _snapshot(config, output, device, seal, source_hashes)
    write_json(output / "status.json", {
        "status": "opening_official_validation_after_seal", "inference_started": False,
        "phase7_fingerprint": config.fingerprint, "completed_conditions": 0,
        "validation_evaluated": False, "test_evaluated": False,
    })
    started = time.monotonic()
    try:
        dataset, _ = build_validation_dataset(bundle)
        if len(dataset) != 1024:
            raise ValueError("Official validation tile count differs from Phase 7 approval")
        conditions = _condition_order()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for instance in bundle.instances:
            grouped[instance["condition_id"]].append(instance)
        if any(len(grouped[name]) != 1024 for name in conditions):
            raise ValueError("Official validation condition coverage differs")
        source_names = [f"val/{entry['name']}" for entry in bundle.manifest[
            "inventory"]["entries"]]
        source_index = {name: index for index, name in enumerate(source_names)}
        condition_index = {name: index for index, name in enumerate(conditions)}
        p_count = len(POLICIES)
        corruption_counts = np.zeros((64, 45, p_count, 4), dtype=np.int64)
        clean_counts = np.zeros((64, p_count, 4), dtype=np.int64)
        corruption_transitions = np.zeros((64, 45, p_count, 4), dtype=np.int64)
        clean_transitions = np.zeros((64, p_count, 4), dtype=np.int64)
        regret_sum = np.zeros((64, p_count), dtype=np.float64)
        regret_count = np.zeros((64, p_count), dtype=np.int64)
        all_utility: list[np.ndarray] = []
        all_decisions: list[np.ndarray] = []
        all_raw = {variant: [] for variant in VARIANTS}
        all_scores = {variant: [] for variant in VARIANTS}
        clean_utility_rows: list[np.ndarray] = []
        clean_decision_rows: list[np.ndarray] = []
        clean_raw = {variant: [] for variant in VARIANTS}
        clean_scores = {variant: [] for variant in VARIANTS}
        strata = _StrataAccumulator()
        timings = {key: 0.0 for key in (
            "encoder_seconds", "descriptor_adapter_seconds", "four_selectors_seconds",
            "both_decoders_seconds", "model_compute_seconds", "batches")}
        amp = bool(config.base.values["train"]["amp"])
        memory_format = config.base.values["runtime"]["memory_format"]
        fmt = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
        threshold = config.base.values["evaluation"]["threshold"]
        loss_kwargs = {key: float(config.base.values["loss"][key])
                       for key in ("bce_weight", "dice_weight", "dice_smooth")}
        torch.cuda.reset_peak_memory_stats(device)

        def transfer(batch):
            return (batch["image_a"].to(device=device, non_blocking=True, memory_format=fmt),
                    batch["image_b"].to(device=device, non_blocking=True, memory_format=fmt),
                    batch["mask"].to(device=device, non_blocking=True))

        clean_path = output / "clean_decisions.jsonl"
        clean_temp = clean_path.with_suffix(clean_path.suffix + ".tmp")
        with clean_temp.open("x", encoding="utf-8") as stream:
            for batch in make_loader(dataset, config.base, training=False):
                if guard_callback:
                    guard_callback()
                image_a, image_b, target = transfer(batch)
                outputs, a0_logits, a1_logits = _batch_inference(
                    detector, selectors, normalization, image_a, image_b,
                    amp=amp, timings=timings)
                a0_loss = per_sample_change_loss(a0_logits, target, **loss_kwargs)["total"]
                a1_loss = per_sample_change_loss(a1_logits, target, **loss_kwargs)["total"]
                utility = (a0_loss - a1_loss).detach().cpu().numpy().astype(np.float64)
                a0 = per_sample_confusion(a0_logits, target, threshold).numpy()
                a1 = per_sample_confusion(a1_logits, target, threshold).numpy()
                transition = per_sample_transitions(a0_logits, a1_logits, target, threshold).numpy()
                decisions, scores, raw = _decision_matrix(
                    outputs, thresholds, normalization, utility)
                chosen = np.where(decisions[:, :, None], a1[:, None, :], a0[:, None, :])
                chosen_transition = decisions[:, :, None].astype(np.int64) * transition[:, None, :]
                clean_utility_rows.append(utility)
                clean_decision_rows.append(decisions)
                for variant in VARIANTS:
                    clean_raw[variant].append(raw[variant])
                    clean_scores[variant].append(scores[variant]["primary"])
                for row_index, sample_id in enumerate(batch["sample_id"]):
                    source = _source_id(sample_id)
                    s = source_index[source]
                    clean_counts[s] += chosen[row_index]
                    clean_transitions[s] += chosen_transition[row_index]
                    record = {
                        "sample_id": sample_id, "source_id": source, "observation": "clean",
                        "u_obs": float(utility[row_index]),
                        "a0_counts": {key: int(a0[row_index, i]) for i, key in enumerate(COUNT_KEYS)},
                        "a1_counts": {key: int(a1[row_index, i]) for i, key in enumerate(COUNT_KEYS)},
                        "transitions_a0_to_a1": {key: int(transition[row_index, i])
                            for i, key in enumerate(TRANSITION_KEYS)},
                        "selector_scores": {variant: {
                            "primary": float(scores[variant]["primary"][row_index]),
                            "constrained": float(scores[variant]["constrained"][row_index]),
                        } for variant in VARIANTS},
                        "decisions": {policy: ("a1" if decisions[row_index, i] else "a0")
                                      for i, policy in enumerate(POLICIES)},
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                            separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(clean_temp, clean_path)
        clean_action_summary = {policy: _metric(clean_counts[:, index, :].sum(axis=0))
                                for index, policy in enumerate(POLICIES)}
        expected_actor = json.loads(config.upstream_paths["actor_status"].read_text(
            encoding="utf-8"))["best_val_f1"]
        expected_adapter = json.loads(config.upstream_paths["adapter_status"].read_text(
            encoding="utf-8"))["best_val_f1"]
        if (abs(clean_action_summary["A0"]["f1"] - expected_actor) > 1e-12
                or abs(clean_action_summary["G0"]["f1"] - expected_adapter) > 1e-12):
            raise ValueError("Phase 7 fixed actions do not reproduce bound clean-val checkpoints")
        write_json(output / "status.json", {
            "status": "evaluating_official_validation_corruptions",
            "inference_started": True, "clean_completed": True,
            "completed_conditions": 0, "condition_count": 45,
            "validation_evaluated": True, "test_evaluated": False,
        }, overwrite=True)

        raw_path = output / "corruption_decisions.jsonl"
        raw_temp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        with raw_temp.open("x", encoding="utf-8") as stream:
            for completed, condition in enumerate(conditions, start=1):
                loader = make_loader(FrozenInstanceDataset(dataset, grouped[condition]),
                                     config.base, training=False)
                for batch in loader:
                    if guard_callback:
                        guard_callback()
                    image_a, image_b, target = transfer(batch)
                    outputs, a0_logits, a1_logits = _batch_inference(
                        detector, selectors, normalization, image_a, image_b,
                        amp=amp, timings=timings)
                    a0_loss = per_sample_change_loss(a0_logits, target, **loss_kwargs)["total"]
                    a1_loss = per_sample_change_loss(a1_logits, target, **loss_kwargs)["total"]
                    utility = (a0_loss - a1_loss).detach().cpu().numpy().astype(np.float64)
                    a0 = per_sample_confusion(a0_logits, target, threshold).numpy()
                    a1 = per_sample_confusion(a1_logits, target, threshold).numpy()
                    transition = per_sample_transitions(
                        a0_logits, a1_logits, target, threshold).numpy()
                    decisions, scores, raw = _decision_matrix(
                        outputs, thresholds, normalization, utility)
                    chosen = np.where(
                        decisions[:, :, None], a1[:, None, :], a0[:, None, :])
                    chosen_transition = (decisions[:, :, None].astype(np.int64)
                                         * transition[:, None, :])
                    regret = np.maximum(utility[:, None], 0.0) - decisions * utility[:, None]
                    if not np.isfinite(regret).all():
                        raise ValueError("Non-finite Phase 7 decision regret")
                    all_utility.append(utility)
                    all_decisions.append(decisions)
                    for variant in VARIANTS:
                        all_raw[variant].append(raw[variant])
                        all_scores[variant].append(scores[variant]["primary"])
                    support = batch["injection_support"].to(torch.bool)
                    truth = batch["mask"].to(torch.bool)
                    foreground = truth.flatten(1).any(dim=1)
                    overlap = (support & truth).flatten(1).any(dim=1)
                    overlap_category = [
                        "no_foreground" if not bool(foreground[i])
                        else "overlap" if bool(overlap[i]) else "none"
                        for i in range(len(batch["sample_id"]))]
                    c = condition_index[condition]
                    for row_index, sample_id in enumerate(batch["sample_id"]):
                        source = _source_id(sample_id)
                        s = source_index[source]
                        corruption_counts[s, c] += chosen[row_index]
                        corruption_transitions[s, c] += chosen_transition[row_index]
                        regret_sum[s] += regret[row_index]
                        regret_count[s] += 1
                        kind = batch["kind"][row_index]
                        keys = {
                            "family": kind, "level": f"l{int(batch['level'][row_index])}",
                            "source": source, "target_date": batch["target_date"][row_index],
                            "spatial_mode": batch["spatial_mode"][row_index],
                            "direction": f"{kind}/{batch['direction'][row_index]}",
                            "injection_overlap": overlap_category[row_index],
                        }
                        strata.update(keys, decisions[row_index], chosen[row_index])
                        record = {
                            "sample_id": sample_id, "source_id": source,
                            "condition_id": condition, "kind": kind,
                            "level": int(batch["level"][row_index]),
                            "direction": batch["direction"][row_index],
                            "target_date": batch["target_date"][row_index],
                            "spatial_mode": batch["spatial_mode"][row_index],
                            "injection_overlap": overlap_category[row_index],
                            "u_obs": float(utility[row_index]),
                            "a0_counts": {key: int(a0[row_index, i])
                                          for i, key in enumerate(COUNT_KEYS)},
                            "a1_counts": {key: int(a1[row_index, i])
                                          for i, key in enumerate(COUNT_KEYS)},
                            "transitions_a0_to_a1": {key: int(transition[row_index, i])
                                for i, key in enumerate(TRANSITION_KEYS)},
                            "selector_scores": {variant: {
                                "primary": float(scores[variant]["primary"][row_index]),
                                "constrained": float(scores[variant]["constrained"][row_index]),
                            } for variant in VARIANTS},
                            "decisions": {policy: (
                                "a1" if decisions[row_index, i] else "a0")
                                for i, policy in enumerate(POLICIES)},
                        }
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                                separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                write_json(output / "status.json", {
                    "status": "evaluating_official_validation_corruptions",
                    "inference_started": True, "clean_completed": True,
                    "completed_conditions": completed, "condition_count": 45,
                    "validation_evaluated": True, "test_evaluated": False,
                }, overwrite=True)
                print(f"phase7/val/{condition}: completed ({completed}/45)", flush=True)
        os.replace(raw_temp, raw_path)

        utility = np.concatenate(all_utility)
        decisions = np.concatenate(all_decisions)
        clean_decisions = np.concatenate(clean_decision_rows)
        if (utility.shape != (46080,) or decisions.shape != (46080, p_count)
                or clean_decisions.shape != (1024, p_count)):
            raise ValueError("Phase 7 official-val decision coverage differs")
        actions = {}
        for p, policy in enumerate(POLICIES):
            condition_metrics = {condition: _metric(
                corruption_counts[:, c, p, :].sum(axis=0))
                for c, condition in enumerate(conditions)}
            pooled = _metric(corruption_counts[:, :, p, :].sum(axis=(0, 1)))
            selected = decisions[:, p]
            selected_count = int(selected.sum())
            positive_total = float(np.maximum(utility, 0).sum())
            captured = float(np.maximum(utility[selected], 0).sum())
            negative_mass = float(np.maximum(-utility[selected], 0).sum())
            regret = np.maximum(utility, 0.0) - selected * utility
            corruption_transition = _transition_summary(
                corruption_transitions[:, :, p, :].reshape(-1, 4),
                corruption_counts[:, :, 0, :].reshape(-1, 4))
            clean_transition = _transition_summary(
                clean_transitions[:, p, :], clean_counts[:, 0, :])
            finite = bool(np.isfinite(regret).all())
            actions[policy] = {
                "corruption_mf1": float(np.mean([
                    condition_metrics[name]["f1"] for name in conditions])),
                "pooled": pooled, "conditions": condition_metrics,
                "clean": clean_action_summary[policy],
                "selection": {
                    "a1_selected": selected_count,
                    "a1_selection_rate": float(selected.mean()),
                    "clean_a1_selected": int(clean_decisions[:, p].sum()),
                    "clean_a1_selection_rate": float(clean_decisions[:, p].mean()),
                    "selected_positive_utility_count": int((selected & (utility > 0)).sum()),
                    "selected_negative_utility_count": int((selected & (utility < 0)).sum()),
                    "selected_positive_fraction_of_selected": (
                        float((selected & (utility > 0)).sum() / selected_count)
                        if selected_count else 0.0),
                    "selected_negative_fraction_of_selected": (
                        float((selected & (utility < 0)).sum() / selected_count)
                        if selected_count else 0.0),
                    "captured_positive_utility_mass": captured,
                    "captured_positive_utility_mass_fraction": (
                        captured / positive_total if positive_total else 0.0),
                    "selected_negative_utility_mass": negative_mass,
                },
                "regret": {"mean": float(regret.mean()),
                           "median": float(np.median(regret)),
                           "p95": float(np.quantile(regret, 0.95)),
                           "p99": float(np.quantile(regret, 0.99)),
                           "max": float(regret.max())},
                "transitions_a0_to_policy": {
                    "corruption": corruption_transition, "clean": clean_transition,
                    "fully_reported_and_finite": finite and all(
                        math.isfinite(row["rate"])
                        for summary in (corruption_transition, clean_transition)
                        for row in summary.values()),
                },
            }
        diagnostics = {}
        clean_utility = np.concatenate(clean_utility_rows)
        for variant in VARIANTS:
            raw = np.concatenate(all_raw[variant])
            score = np.concatenate(all_scores[variant])
            diagnostics[variant] = {
                "corruption": _selector_diagnostic(
                    variant, raw, score, utility, normalization),
                "clean": _selector_diagnostic(
                    variant, np.concatenate(clean_raw[variant]),
                    np.concatenate(clean_scores[variant]), clean_utility, normalization),
            }
        better_index = 1 if actions["G0"]["corruption_mf1"] > actions["A0"][
            "corruption_mf1"] else 0
        bootstrap = _bootstrap(config, corruption_counts, clean_counts,
                               regret_sum, regret_count, better_index)
        decision = _decision(config, actions, bootstrap)
        strata_result = strata.finish()
        cache_bytes = sum(path.stat().st_size for path in cache["output"].rglob("*")
                          if path.is_file())
        total_observations = 46080 + 1024
        compute = timings["model_compute_seconds"]
        compute_report = {
            **timings, "wall_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "selector_parameters_each": 139970,
            "descriptor_cache_bytes": cache_bytes,
            "observations_including_clean": total_observations,
            "amortized_model_compute_ms_per_tile": 1000.0 * compute / total_observations,
            "selected_decoder_counts": {policy: {
                "corruption": {"a0": int((~decisions[:, index]).sum()),
                               "a1": int(decisions[:, index].sum())},
                "clean": {"a0": int((~clean_decisions[:, index]).sum()),
                          "a1": int(clean_decisions[:, index].sum())},
                "total": {"a0": int((~decisions[:, index]).sum()
                                     + (~clean_decisions[:, index]).sum()),
                          "a1": int(decisions[:, index].sum()
                                    + clean_decisions[:, index].sum())},
            } for index, policy in enumerate(POLICIES)},
            "measurement_note": (
                "The matched evaluator computes both frozen actions once per tile to score all "
                "policies and O; selected counts are whole-tile decisions. No compute-saving "
                "claim is made. Encoder, adapter descriptor, selector, and decoder costs are included."),
        }
        write_json(output / "actions.json", actions)
        write_json(output / "strata.json", strata_result)
        write_json(output / "selector_diagnostics.json", diagnostics)
        write_json(output / "bootstrap.json", bootstrap)
        write_json(output / "decision.json", decision)
        write_json(output / "compute.json", compute_report)
        result = {
            "status": "completed", "kind": "research",
            "stage": "phase7_whole_tile_action_selector", "split": "val",
            "phase7_fingerprint": config.fingerprint,
            "preval_seal_sha256": _sha(config.paths["preval_seal"]),
            "actor_checkpoint_epoch": actor_checkpoint["epoch"],
            "adapter_checkpoint_epoch": adapter_checkpoint["epoch"],
            "source_pairs": 64, "clean_tiles": 1024, "condition_count": 45,
            "perturbed_instances": 46080, "policies": list(POLICIES),
            "policy_thresholds": {variant: {
                "unconstrained": {"kind": "finite", "value": 0.0},
                "harm_constrained": json.loads((config.training_output(variant)
                    / "policy_calibration.json").read_text(encoding="utf-8"))[
                        "harm_constrained"]["threshold"],
            } for variant in VARIANTS},
            "actions": actions, "bootstrap": bootstrap, "decision": decision,
            "selector_diagnostics": diagnostics, "compute": compute_report,
            "raw_corruption_decisions_sha256": _sha(raw_path),
            "raw_clean_decisions_sha256": _sha(clean_path),
            "source_sha256": source_hashes, "training_started": False,
            "validation_evaluated": True, "test_evaluated": False,
            "guarded_completion_required": True,
        }
        if ({name: _sha(path) for name, path in phase7_evaluation_source_files().items()}
                != source_hashes or load_preval_seal(config) != seal):
            raise ValueError("Phase 7 source or pre-val seal changed during evaluation")
        write_json(output / "summary.json", result)
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": f"{type(error).__name__}: {error}",
            "inference_started": True, "validation_evaluated": True,
            "test_evaluated": False,
        }, overwrite=True)
        raise
