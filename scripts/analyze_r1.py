"""Freeze-bound, source-paired analysis of the six R1 validation results.

The input manifest binds already completed guarded validation artifacts by hash.
The analysis rules are fixed here before the seed-43/44 corruption evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import yaml


SEEDS = (42, 43, 44)
MODES = ("b1", "b2")
COUNT_KEYS = ("tn", "fp", "fn", "tp")
ARTIFACTS = (
    "summary.json", "status.json", "conditions.json", "clean_metrics.json",
    "guard_completion.json", "temperature_guard_state.json",
)
METRICS = (
    "clean_f1", "corruption_mf1", "clean_minus_corruption_mf1",
    "pooled_recall", "background_fpr", "net_fp", "net_fn",
)
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_REPLICATES = 20_000
EXPECTED_SOURCES = 64
EXPECTED_CONDITIONS = 45


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, expected_sha: str) -> dict[str, Any]:
    if _sha(path) != expected_sha:
        raise ValueError(f"Frozen hash differs: {path}")
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict) or _sha(path) != expected_sha:
        raise ValueError(f"Invalid or changing JSON artifact: {path}")
    return result


def _counts(value: dict[str, Any]) -> np.ndarray:
    if set(value) != set(COUNT_KEYS):
        raise ValueError("Confusion count keys differ")
    result = np.array([value[key] for key in COUNT_KEYS], dtype=np.int64)
    if any(type(value[key]) is not int or value[key] < 0 for key in COUNT_KEYS):
        raise ValueError("Confusion counts must be nonnegative integers")
    return result


def _f1(counts: np.ndarray) -> np.ndarray:
    fp, fn, tp = counts[..., 1], counts[..., 2], counts[..., 3]
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float),
                     where=denominator != 0)


def _same_number(actual: float, expected: float, label: str) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 1e-12:
        raise ValueError(f"Aggregate differs from bound records: {label}")


def _load_run(manifest_dir: Path, item: dict[str, Any], seed: int,
              mode: str) -> dict[str, Any]:
    if set(item) != {"directory", "sha256"} or set(item["sha256"]) != set(ARTIFACTS):
        raise ValueError(f"Incomplete artifact binding for {mode}/{seed}")
    directory = (manifest_dir / item["directory"]).resolve()
    artifacts = {}
    for name in ARTIFACTS:
        expected = item["sha256"][name]
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"Invalid SHA256 for {mode}/{seed}/{name}")
        artifacts[name] = _read_json(directory / name, expected)
    summary, status = artifacts["summary.json"], artifacts["status.json"]
    conditions, clean = artifacts["conditions.json"], artifacts["clean_metrics.json"]
    guard = artifacts["guard_completion.json"]
    guard_state = artifacts["temperature_guard_state.json"]
    if (summary.get("status") != "completed" or status.get("status") != "completed"
            or summary.get("candidate_mode") != mode or status.get("candidate_mode") != mode
            or summary.get("split") != "val" or status.get("split") != "val"
            or summary.get("test_evaluated") is not False
            or status.get("test_evaluated") is not False
            or summary.get("training_started") is not False
            or status.get("training_started") is not False
            or summary.get("temperature_guard", {}).get("status") != "completed"
            or status.get("temperature_guard", {}).get("status") != "completed"
            or guard.get("status") != "completed"
            or guard_state.get("status") != "completed"
            or guard_state.get("decision") != "ok"
            or guard_state.get("test_evaluated") is not False
            or guard.get("summary_sha256") != item["sha256"]["summary.json"]
            or guard.get("guard_state_sha256")
            != item["sha256"]["temperature_guard_state.json"]):
        raise ValueError(f"Validation or guard completion invalid: {mode}/{seed}")
    if len(conditions) != EXPECTED_CONDITIONS or len(clean.get("counts", {})) != 4:
        raise ValueError(f"Incomplete validation coverage: {mode}/{seed}")
    clean_counts = _counts(clean["counts"])
    _same_number(float(_f1(clean_counts)), float(summary["clean_f1"]), "clean F1")
    if clean.get("matches_candidate_checkpoint") is not True:
        raise ValueError(f"Clean checkpoint replay differs: {mode}/{seed}")
    condition_ids = sorted(conditions)
    source_ids = sorted(next(iter(conditions.values()))["by_source"])
    if len(source_ids) != EXPECTED_SOURCES:
        raise ValueError(f"Expected {EXPECTED_SOURCES} validation source pairs")
    by_source = np.empty((EXPECTED_SOURCES, EXPECTED_CONDITIONS, 4), dtype=np.int64)
    pooled = np.zeros(4, dtype=np.int64)
    condition_f1 = []
    for c, condition_id in enumerate(condition_ids):
        row = conditions[condition_id]
        if (row.get("condition_id") != condition_id
                or sorted(row.get("by_source", {})) != source_ids
                or row.get("samples") != 1024):
            raise ValueError(f"Condition/source identity differs: {mode}/{seed}/{condition_id}")
        for s, source_id in enumerate(source_ids):
            source = row["by_source"][source_id]
            if source.get("samples") != 16:
                raise ValueError(f"Source tile coverage differs: {source_id}")
            by_source[s, c] = _counts(source["counts"])
        condition_counts = _counts(row["counts"])
        if not np.array_equal(by_source[:, c].sum(axis=0), condition_counts):
            raise ValueError(f"Source counts do not pool: {mode}/{seed}/{condition_id}")
        pooled += condition_counts
        f1 = float(_f1(condition_counts))
        _same_number(f1, float(row["metrics"]["f1"]), "condition F1")
        condition_f1.append(f1)
    _same_number(mean(condition_f1), float(summary["corruption_mf1"]), "corruption mF1")
    diagnostics = summary["candidate_diagnostics"]["pooled_over_conditions"]
    if not np.array_equal(pooled, _counts(diagnostics["counts"])):
        raise ValueError(f"Pooled corruption counts differ: {mode}/{seed}")
    metrics = {
        "clean_f1": float(summary["clean_f1"]),
        "corruption_mf1": float(summary["corruption_mf1"]),
        "clean_minus_corruption_mf1": float(summary["clean_minus_corruption_mf1"]),
        "pooled_recall": float(diagnostics["metrics"]["foreground_recall"]),
        "background_fpr": float(diagnostics["metrics"]["background_fpr"]),
        "net_fp": int(pooled[1] - EXPECTED_CONDITIONS * clean_counts[1]),
        "net_fn": int(pooled[2] - EXPECTED_CONDITIONS * clean_counts[2]),
    }
    _same_number(metrics["clean_f1"] - metrics["corruption_mf1"],
                 metrics["clean_minus_corruption_mf1"], "clean/corruption gap")
    return {"metrics": metrics, "condition_ids": condition_ids,
            "source_ids": source_ids, "condition_f1": condition_f1,
            "by_source": by_source, "artifact_sha256": item["sha256"]}


def paired_bootstrap(counts: np.ndarray, *, replicates: int = BOOTSTRAP_REPLICATES,
                     seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Resample one common source draw for every seed, method and condition."""
    if counts.ndim != 5 or counts.shape[1:3] != (3, 2) or counts.shape[-1] != 4:
        raise ValueError("Expected source × three seeds × two methods × conditions × counts")
    sources, _, _, conditions, _ = counts.shape
    if sources < 1 or conditions < 1 or replicates < 1:
        raise ValueError("Empty bootstrap input")
    flat = counts.reshape(sources, -1)
    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        weights = rng.multinomial(sources, [1 / sources] * sources, size=stop - start)
        pooled = (weights @ flat).reshape(stop - start, 3, 2, conditions, 4)
        mf1 = _f1(pooled).mean(axis=-1)
        deltas[start:stop] = (mf1[:, :, 1] - mf1[:, :, 0]).mean(axis=1)
    return [float(value) for value in np.quantile(deltas, [0.025, 0.5, 0.975])]


def selection_gate(by_seed: dict[int, dict[str, dict[str, float]]],
                   interval: list[float]) -> dict[str, Any]:
    means = {mode: {key: mean(by_seed[seed][mode][key] for seed in SEEDS)
                    for key in METRICS} for mode in MODES}
    ranges = {mode: max(by_seed[seed][mode]["corruption_mf1"] for seed in SEEDS)
              - min(by_seed[seed][mode]["corruption_mf1"] for seed in SEEDS)
              for mode in MODES}
    wins = sum(by_seed[seed]["b2"]["corruption_mf1"]
               > by_seed[seed]["b1"]["corruption_mf1"] for seed in SEEDS)
    checks = {
        "mean_mf1_delta_at_least_0_0005": means["b2"]["corruption_mf1"]
        - means["b1"]["corruption_mf1"] >= 0.0005,
        "bootstrap_lower_strictly_positive": interval[0] > 0,
        "clean_f1_within_0_0005": means["b2"]["clean_f1"]
        >= means["b1"]["clean_f1"] - 0.0005,
        "recall_within_0_001": means["b2"]["pooled_recall"]
        >= means["b1"]["pooled_recall"] - 0.001,
        "b2_wins_at_least_two_seeds": wins >= 2,
    }
    unstable = any(value > 0.005 for value in ranges.values())
    decision = ("stop_for_instability_review" if unstable else
                "select_b2" if all(checks.values()) else "retain_b1")
    return {"decision": decision, "checks": checks,
            "cross_seed_mf1_ranges": ranges, "b2_seed_wins": wins,
            "means": means, "bootstrap_interval": interval}


def analyze(manifest_path: Path) -> dict[str, Any]:
    raw = manifest_path.read_bytes()
    manifest = yaml.safe_load(raw)
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "runs"}
            or manifest["schema_version"] != 1
            or set(manifest["runs"]) != {str(seed) for seed in SEEDS}):
        raise ValueError("R1 analysis manifest differs from the frozen six-run plan")
    runs: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        rows = manifest["runs"][str(seed)]
        if set(rows) != set(MODES):
            raise ValueError(f"Missing method for seed {seed}")
        runs[seed] = {mode: _load_run(manifest_path.parent, rows[mode], seed, mode)
                      for mode in MODES}
    reference_conditions = runs[42]["b1"]["condition_ids"]
    reference_sources = runs[42]["b1"]["source_ids"]
    for seed in SEEDS:
        for mode in MODES:
            row = runs[seed][mode]
            if (row["condition_ids"] != reference_conditions
                    or row["source_ids"] != reference_sources):
                raise ValueError("Runs do not share the same condition and source pairs")
    by_seed = {seed: {mode: runs[seed][mode]["metrics"] for mode in MODES}
               for seed in SEEDS}
    counts = np.stack([np.stack([runs[seed][mode]["by_source"] for mode in MODES],
                               axis=1) for seed in SEEDS], axis=1)
    interval = paired_bootstrap(counts)
    gate = selection_gate(by_seed, interval)
    summary = {mode: {key: {"mean": mean(by_seed[seed][mode][key] for seed in SEEDS),
                            "sample_std": stdev(by_seed[seed][mode][key] for seed in SEEDS)}
                      for key in METRICS} for mode in MODES}
    deltas = {seed: {key: by_seed[seed]["b2"][key] - by_seed[seed]["b1"][key]
                     for key in METRICS} for seed in SEEDS}
    delta_summary = {key: {"mean": mean(deltas[seed][key] for seed in SEEDS),
                           "sample_std": stdev(deltas[seed][key] for seed in SEEDS)}
                     for key in METRICS}
    condition_wins = {}
    for seed in SEEDS:
        b1, b2 = runs[seed]["b1"]["condition_f1"], runs[seed]["b2"]["condition_f1"]
        condition_wins[seed] = {"b1": sum(a > b for a, b in zip(b1, b2)),
                                "b2": sum(b > a for a, b in zip(b1, b2)),
                                "ties": sum(a == b for a, b in zip(b1, b2))}
    return {"schema_version": 1, "status": "completed", "split": "val",
            "test_evaluated": False, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "seeds": list(SEEDS), "modes": list(MODES),
            "condition_count": len(reference_conditions),
            "source_pair_count": len(reference_sources),
            "aggregation_rule": "Pool pixels within each condition, then average 45 condition F1 values equally",
            "per_seed": by_seed, "method_summary": summary,
            "b2_minus_b1_per_seed": deltas, "b2_minus_b1_summary": delta_summary,
            "condition_wins": condition_wins,
            "paired_source_bootstrap": {"seed": BOOTSTRAP_SEED,
                                        "replicates": BOOTSTRAP_REPLICATES,
                                        "method": "multinomial resampling of common original validation source pairs; percentile 95%",
                                        "b2_minus_b1_mean_mf1_interval": interval},
            "selection_gate": gate,
            "input_sha256": {seed: {mode: runs[seed][mode]["artifact_sha256"]
                                    for mode in MODES} for seed in SEEDS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite R1 analysis: {args.output}")
    result = analyze(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "selection_gate": result["selection_gate"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
