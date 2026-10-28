"""Analyze saved Valais validation counts without inference or raw test access."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from pcdc.data import file_sha256
from pcdc.runtime import write_json


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pooled_counts(rows):
    return {key: sum(row["counts"][key] for row in rows) for key in ("tn", "fp", "fn", "tp")}


def metrics(counts):
    tn, fp, fn, tp = [counts[key] for key in ("tn", "fp", "fn", "tp")]
    def div(a, b): return a / b if b else 0.0
    return {"f1": div(2 * tp, 2 * tp + fp + fn), "recall": div(tp, tp + fn),
            "precision": div(tp, tp + fp), "background_fpr": div(fp, tn + fp)}


def source_strata(sources):
    out = {}
    for group in ("empty", "positive"):
        rows = [row for row in sources.values()
                if (row["foreground_pixels"] == 0) == (group == "empty")]
        counts = pooled_counts(rows)
        out[group] = {"samples": len(rows), "counts": counts, "metrics": metrics(counts),
                      "tiles_with_prediction": sum(r["counts"]["tp"] + r["counts"]["fp"] > 0 for r in rows),
                      "positive_tiles_without_tp": sum(r["foreground_pixels"] > 0 and r["counts"]["tp"] == 0 for r in rows)}
    return out


def analyze(directory: Path, instances: Path) -> dict:
    summary = read_json(directory / "summary.json")
    guard = read_json(directory / "guard_completion.json")
    state = read_json(directory / "temperature_guard_state.json")
    clean = read_json(directory / "clean_metrics.json")
    if (summary.get("status") != "completed" or summary.get("split") != "val"
            or summary.get("test_evaluated") is not False
            or summary.get("clean_matches_reference") is not True
            or guard.get("status") != "completed" or state.get("status") != "completed"
            or guard["summary_sha256"] != file_sha256(directory / "summary.json")
            or guard["guard_state_sha256"] != file_sha256(directory / "temperature_guard_state.json")
            or read_json(directory / "status.json") != summary
            or file_sha256(instances) != summary["fingerprints"]["instances_sha256"]):
        raise ValueError("Baseline is not an intact completed guarded validation result")
    conditions = {p.stem: read_json(p) for p in sorted((directory / "conditions").glob("*.json"))}
    expected = {f"{family}_l{level}" for family in summary["family_mf1"] for level in range(1, 6)}
    if set(conditions) != expected or len(expected) != 45:
        raise ValueError("Incomplete or duplicate condition coverage")
    for name, row in conditions.items():
        if (row["condition_id"] != name or row["samples"] != clean["samples"]
                or row["pixels"] != clean["pixels"]
                or set(row["by_source"]) != set(clean["by_source"])
                or pooled_counts(row["by_source"].values()) != row["counts"]
                or metrics(row["counts"])["f1"] != row["metrics"]["f1"]):
            raise ValueError(f"Condition/source counts differ: {name}")
    mf1 = sum(row["metrics"]["f1"] for row in conditions.values()) / 45
    if not math.isclose(mf1, summary["corruption_mf1"], abs_tol=1e-15, rel_tol=0):
        raise ValueError("Saved mF1 differs from all condition counts")
    assignments = defaultdict(dict)
    with instances.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["stream"] != "val": raise ValueError("Non-validation instance")
            source = item["sample_id"].split("@", 1)[0]
            name = item["condition_id"]
            if source in assignments[name]: raise ValueError("Duplicate source condition")
            assignments[name][source] = item
    rows, subgroup_rows = [], []
    for name, row in conditions.items():
        if set(assignments[name]) != set(clean["by_source"]):
            raise ValueError("Instance assignment differs from source coverage")
        rows.append({"condition": name, "family": row["kind"], "level": row["level"],
                     **metrics(row["counts"]), "fp": row["counts"]["fp"], "fn": row["counts"]["fn"],
                     "net_fp": row["counts"]["fp"] - clean["counts"]["fp"],
                     "net_fn": row["counts"]["fn"] - clean["counts"]["fn"],
                     "source_strata": source_strata(row["by_source"])})
        for axis in ("spatial_mode", "direction", "target_date"):
            for value in sorted({i[axis] for i in assignments[name].values()}):
                keys = [key for key, i in assignments[name].items() if i[axis] == value]
                pert = pooled_counts(row["by_source"][key] for key in keys)
                base = pooled_counts(clean["by_source"][key] for key in keys)
                subgroup_rows.append({"condition": name, "axis": axis, "value": value,
                                      "samples": len(keys), "perturbed": metrics(pert),
                                      "same_sources_clean": metrics(base),
                                      "net_fp": pert["fp"] - base["fp"],
                                      "net_fn": pert["fn"] - base["fn"]})
    grouped = {}
    for axis in ("family", "level"):
        grouped[axis] = []
        for value in sorted({r[axis] for r in rows}):
            selected = [r for r in rows if r[axis] == value]
            grouped[axis].append({axis: value, "condition_count": len(selected),
                                  "mf1": sum(r["f1"] for r in selected) / len(selected),
                                  "mean_net_fp": sum(r["net_fp"] for r in selected) / len(selected),
                                  "mean_net_fn": sum(r["net_fn"] for r in selected) / len(selected)})
    all_counts = pooled_counts(conditions.values())
    return {"status": "saved_validation_analysis_complete", "new_inference": False,
            "raw_test_data_accessed": False, "condition_count": 45,
            "clean": {"counts": clean["counts"], "metrics": metrics(clean["counts"]),
                      "source_strata": source_strata(clean["by_source"])},
            "corruption_mf1": mf1, "pooled_corruption": metrics(all_counts),
            "net_fp_vs_clean45": all_counts["fp"] - 45 * clean["counts"]["fp"],
            "net_fn_vs_clean45": all_counts["fn"] - 45 * clean["counts"]["fn"],
            "grouped": grouped, "conditions": rows, "subgroups": subgroup_rows,
            "source_hashes": {str(p.resolve()): file_sha256(p) for p in
                              [directory / "summary.json", directory / "guard_completion.json",
                               directory / "temperature_guard_state.json", directory / "clean_metrics.json",
                               instances, *sorted((directory / "conditions").glob("*.json"))]},
            "uncertainty": "Descriptive single-city validation; no independent spatial IDs, no tile bootstrap."}


def save_analysis(report: dict, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "analysis.json", report)
    for name, rows in (("family", report["grouped"]["family"]),
                       ("level", report["grouped"]["level"]),
                       ("conditions", [{k: v for k, v in r.items() if k != "source_strata"}
                                       for r in report["conditions"]])):
        with (output / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.directory.resolve(), args.instances.resolve())
    save_analysis(report, args.output.resolve())
    print(json.dumps({k: report[k] for k in ("status", "condition_count", "corruption_mf1", "new_inference")}))


if __name__ == "__main__": main()
