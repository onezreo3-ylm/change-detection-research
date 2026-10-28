"""Audit saved validation subgroup counts without loading a model or raw data.

This independent sidecar leaves the active study's sealed legacy analysis intact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

COUNT_KEYS = ("tn", "fp", "fn", "tp")
AXES = ("spatial_mode", "direction", "target_date")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_counts(rows):
    """Consume an iterable once, accumulating every confusion-count field."""
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for row in rows:
        values = row["counts"]
        if set(values) != set(COUNT_KEYS):
            raise ValueError("Unexpected confusion-count fields")
        for key in COUNT_KEYS:
            value = values[key]
            if type(value) is not int or value < 0:
                raise ValueError("Confusion counts must be nonnegative integers")
            counts[key] += value
    return counts


def metrics(counts):
    tn, fp, fn, tp = (counts[key] for key in COUNT_KEYS)
    def divide(a, b):
        return a / b if b else 0.0
    return {"f1": divide(2 * tp, 2 * tp + fp + fn),
            "recall": divide(tp, tp + fn), "precision": divide(tp, tp + fp),
            "background_fpr": divide(fp, tn + fp)}


def require_metrics(actual, expected):
    for key, value in actual.items():
        if not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"Saved aggregate metric differs: {key}")


def validate_sources(record):
    sources = record["by_source"]
    if len(sources) != record["samples"] or pooled_counts(sources.values()) != record["counts"]:
        raise ValueError("Saved source coverage/counts differ from condition totals")
    for source in sources.values():
        counts = source["counts"]
        if (sum(counts.values()) != source["pixels"]
                or counts["tp"] + counts["fn"] != source["foreground_pixels"]
                or counts["tn"] + counts["fp"] != source["background_pixels"]):
            raise ValueError("Saved source pixel/label totals differ")
    if sum(record["counts"].values()) != record["pixels"]:
        raise ValueError("Saved total pixels differ")


def condition_subgroups(condition, clean, assignments):
    validate_sources(clean)
    validate_sources(condition)
    sources = set(clean["by_source"])
    if set(assignments) != sources or set(condition["by_source"]) != sources:
        raise ValueError("Subgroup assignment/source coverage differs")
    for source in sources:
        if (condition["by_source"][source]["foreground_pixels"]
                != clean["by_source"][source]["foreground_pixels"]):
            raise ValueError("Foreground labels changed between clean and corruption")
    rows = []
    for axis in AXES:
        partition = []
        for value in sorted({item[axis] for item in assignments.values()}):
            keys = sorted(key for key, item in assignments.items() if item[axis] == value)
            perturbed = pooled_counts(condition["by_source"][key] for key in keys)
            reference = pooled_counts(clean["by_source"][key] for key in keys)
            partition.append({"condition": condition["condition_id"], "axis": axis,
                              "value": value, "samples": len(keys), "counts": perturbed,
                              "same_sources_clean_counts": reference,
                              "perturbed": metrics(perturbed),
                              "same_sources_clean": metrics(reference),
                              "net_fp": perturbed["fp"] - reference["fp"],
                              "net_fn": perturbed["fn"] - reference["fn"]})
        if (pooled_counts(partition) != condition["counts"]
                or pooled_counts({"counts": row["same_sources_clean_counts"]} for row in partition)
                != clean["counts"] or sum(row["samples"] for row in partition) != len(sources)):
            raise ValueError("Subgroup partition does not recover all saved counts")
        rows.extend(partition)
    return rows


def analyze(directory, instances, sealed_analysis):
    directory, instances, sealed_analysis = map(Path, (directory, instances, sealed_analysis))
    summary = read_json(directory / "summary.json")
    guard = read_json(directory / "guard_completion.json")
    state = read_json(directory / "temperature_guard_state.json")
    if (summary.get("status") != "completed" or summary.get("split") != "val"
            or summary.get("test_evaluated") is not False
            or summary.get("clean_matches_reference") is not True
            or guard.get("status") != "completed" or state.get("status") != "completed"
            or state.get("decision") != "ok"
            or guard["summary_sha256"] != sha256(directory / "summary.json")
            or guard["guard_state_sha256"] != sha256(directory / "temperature_guard_state.json")
            or read_json(directory / "status.json") != summary
            or summary["fingerprints"]["instances_sha256"] != sha256(instances)):
        raise ValueError("Evaluation is not an intact completed guarded validation result")
    legacy = read_json(sealed_analysis)
    if not legacy.get("source_hashes"):
        raise ValueError("Missing sealed analysis source hashes")
    for path, expected_hash in legacy["source_hashes"].items():
        if sha256(path) != expected_hash:
            raise ValueError(f"Sealed analysis source changed: {path}")
    expected = {f"{family}_l{level}" for family in summary["family_mf1"] for level in range(1, 6)}
    files = {path.stem: path for path in (directory / "conditions").glob("*.json")}
    if len(expected) != 45 or set(files) != expected or summary["condition_count"] != 45:
        raise ValueError("Incomplete 45-condition coverage")
    assignments = defaultdict(dict)
    with instances.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source, separator, _ = item["sample_id"].partition("@")
            name = item["condition_id"]
            if (item["stream"] != "val" or not separator or name not in expected
                    or source in assignments[name]):
                raise ValueError("Invalid or duplicate validation instance")
            assignments[name][source] = {axis: item[axis] for axis in AXES}
    if set(assignments) != expected:
        raise ValueError("Instance condition coverage differs")
    clean = read_json(directory / "clean_metrics.json")
    validate_sources(clean)
    require_metrics(metrics(clean["counts"]), clean["metrics"])
    require_metrics(metrics(clean["counts"]), legacy["clean"]["metrics"])
    rows, totals, empty_totals, condition_f1 = [], [], [], []
    for name, path in sorted(files.items()):
        condition = read_json(path)
        if (condition["condition_id"] != name or condition["pixels"] != clean["pixels"]
                or name != f"{condition['kind']}_l{condition['level']}"):
            raise ValueError("Condition identity/pixel count differs")
        require_metrics(metrics(condition["counts"]), condition["metrics"])
        rows.extend(condition_subgroups(condition, clean, assignments[name]))
        totals.append({"counts": condition["counts"]})
        empty_totals.append({"counts": pooled_counts(source for source in condition["by_source"].values()
                                                    if source["foreground_pixels"] == 0)})
        condition_f1.append(metrics(condition["counts"])["f1"])
    mf1 = sum(condition_f1) / 45
    for expected_mf1 in (summary["corruption_mf1"], legacy["corruption_mf1"]):
        if not math.isclose(mf1, expected_mf1, rel_tol=0, abs_tol=1e-15):
            raise ValueError("Aggregate mF1 differs")
    pooled = metrics(pooled_counts(totals))
    require_metrics(pooled, legacy["pooled_corruption"])
    def row_key(row):
        return row["condition"], row["axis"], row["value"]
    old_rows = {row_key(row): row for row in legacy["subgroups"]}
    if len(old_rows) != len(legacy["subgroups"]) or set(old_rows) != {row_key(row) for row in rows}:
        raise ValueError("Legacy subgroup identities differ")
    discrepancies = []
    for row in rows:
        old = old_rows[row_key(row)]
        if old["samples"] != row["samples"]:
            raise ValueError("Legacy subgroup sample coverage differs")
        fields = [key for key in ("perturbed", "same_sources_clean", "net_fp", "net_fn")
                  if old[key] != row[key]]
        if fields:
            discrepancies.append({"condition": row["condition"], "axis": row["axis"],
                                  "value": row["value"], "changed_fields": fields})
    source_hashes = {**legacy["source_hashes"], str(sealed_analysis.resolve()): sha256(sealed_analysis)}
    return {"status": "saved_validation_subgroups_corrected", "new_model_inference": False,
            "training_started": False, "test_accessed": False, "condition_count": 45,
            "subgroup_count": len(rows), "partition_checks_passed": True,
            "aggregate_metrics_unchanged": True, "legacy_subgroups_valid": not discrepancies,
            "legacy_root_cause": "An iterator was consumed separately for tn/fp/fn/tp; only tn survived.",
            "changed_subgroup_count": len(discrepancies), "discrepancies": discrepancies,
            "unchanged_aggregates": {"clean": metrics(clean["counts"]), "mf1": mf1,
                                     "pooled_corruption": pooled,
                                     "empty_tile_corruption_fpr": metrics(pooled_counts(empty_totals))["background_fpr"]},
            "subgroups": rows, "source_hashes": source_hashes,
            "limitation": "Saved-count correction only; one validation city, no spatial independence claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    values = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    keys = {"schema_version", "evaluation_dir", "instances", "sealed_analysis", "output_dir"}
    if set(values) != keys or values["schema_version"] != 1:
        raise ValueError("Unexpected saved-count audit configuration")
    paths = {key: (config.parent / values[key]).resolve() for key in keys - {"schema_version"}}
    output = paths["output_dir"]
    if output.exists():
        raise FileExistsError("Saved-count audit output already exists; preserve previous evidence")
    report = analyze(paths["evaluation_dir"], paths["instances"], paths["sealed_analysis"])
    report["source_hashes"].update({str(config): sha256(config), str(Path(__file__).resolve()): sha256(__file__)})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    flat_rows = [{**{key: row[key] for key in ("condition", "axis", "value", "samples", "net_fp", "net_fn")},
                  **{f"corruption_{key}": value for key, value in row["perturbed"].items()},
                  **{f"same_sources_clean_{key}": value for key, value in row["same_sources_clean"].items()}}
                 for row in report["subgroups"]]
    with (output / "subgroups.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({key: report[key] for key in ("status", "condition_count", "subgroup_count",
                                                 "changed_subgroup_count", "aggregate_metrics_unchanged",
                                                 "new_model_inference", "test_accessed")}))


if __name__ == "__main__":
    main()
