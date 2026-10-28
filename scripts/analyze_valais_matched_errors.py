"""Compare intact saved B0/B1/B0-E validation counts; no model or raw-image reads."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import yaml

from scripts.analyze_valais_saved_subgroups import (
    AXES, COUNT_KEYS, condition_subgroups, metrics, pooled_counts,
    read_json, require_metrics, sha256, validate_sources,
)

MODELS = ("b0", "b1", "b0e")
PAIRS = (("b1", "b0"), ("b1", "b0e"), ("b0e", "b0"))


def describe_sources(sources):
    rows = list(sources.values())
    counts = pooled_counts(rows)
    positive = [row for row in rows if row["foreground_pixels"] > 0]
    empty = [row for row in rows if row["foreground_pixels"] == 0]
    return {"samples": len(rows), "counts": counts, "metrics": metrics(counts),
            "foreground_pixels": counts["tp"] + counts["fn"],
            "positive_tiles": len(positive), "empty_tiles": len(empty),
            "positive_tiles_without_tp": sum(row["counts"]["tp"] == 0 for row in positive),
            "positive_tiles_without_predictions": sum(row["counts"]["tp"] + row["counts"]["fp"] == 0
                                                        for row in positive),
            "empty_tiles_with_predictions": sum(row["counts"]["fp"] > 0 for row in empty)}


def require_same_sources(candidate, reference):
    if set(candidate) != set(reference):
        raise ValueError("Compared source coverage differs")
    for name, row in candidate.items():
        old = reference[name]
        if row["foreground_pixels"] != old["foreground_pixels"] or row["pixels"] != old["pixels"]:
            raise ValueError("Compared source label/pixel totals differ")


def paired_sources(candidate, reference):
    require_same_sources(candidate, reference)
    current, old = pooled_counts(candidate.values()), pooled_counts(reference.values())
    rows = [(candidate[name], reference[name]) for name in sorted(candidate)]
    return {"counts_delta": {key: current[key] - old[key] for key in COUNT_KEYS},
            "metrics_delta": {key: metrics(current)[key] - metrics(old)[key] for key in metrics(current)},
            "sources_with_more_fn": sum(a["counts"]["fn"] > b["counts"]["fn"] for a, b in rows),
            "sources_with_fewer_fn": sum(a["counts"]["fn"] < b["counts"]["fn"] for a, b in rows),
            "sources_with_more_fp": sum(a["counts"]["fp"] > b["counts"]["fp"] for a, b in rows),
            "sources_with_fewer_fp": sum(a["counts"]["fp"] < b["counts"]["fp"] for a, b in rows),
            "previous_zero_tp_positive_tiles_now_with_tp": sum(
                a["foreground_pixels"] > 0 and a["counts"]["tp"] > 0 and b["counts"]["tp"] == 0 for a, b in rows),
            "previous_detected_positive_tiles_now_zero_tp": sum(
                a["foreground_pixels"] > 0 and a["counts"]["tp"] == 0 and b["counts"]["tp"] > 0 for a, b in rows)}


def size_groups(sources, boundaries):
    if (not boundaries or any(type(value) is not int or value <= 0 for value in boundaries)
            or boundaries != sorted(set(boundaries))):
        raise ValueError("Foreground-area boundaries must be unique ascending positive integers")
    groups = {"empty": {}, **{f"{lo + 1}..{hi}": {} for lo, hi in zip([0] + boundaries[:-1], boundaries)},
              f">{boundaries[-1]}": {}}
    for name, row in sources.items():
        pixels = row["foreground_pixels"]
        if pixels == 0:
            label = "empty"
        else:
            lower = 0
            label = f">{boundaries[-1]}"
            for upper in boundaries:
                if pixels <= upper:
                    label = f"{lower + 1}..{upper}"
                    break
                lower = upper
        groups[label][name] = row
    result = [{"foreground_area": label, **describe_sources(rows)} for label, rows in groups.items()]
    if pooled_counts(result) != pooled_counts(sources.values()) or sum(row["samples"] for row in result) != len(sources):
        raise ValueError("Foreground-area partition does not recover the saved counts")
    return result


def aggregate_conditions(rows, clean_counts):
    counts = pooled_counts(rows)
    empty = pooled_counts({"counts": row["strata"]["empty"]["counts"]} for row in rows)
    positive = pooled_counts({"counts": row["strata"]["positive"]["counts"]} for row in rows)
    if {key: empty[key] + positive[key] for key in COUNT_KEYS} != counts:
        raise ValueError("Empty/positive partition does not recover condition counts")
    repeat = len(rows)
    return {"condition_count": repeat, "mean_condition_f1": sum(row["metrics"]["f1"] for row in rows) / repeat,
            "counts": counts, "pooled_metrics": metrics(counts),
            "empty_counts": empty, "empty_metrics": metrics(empty),
            "positive_counts": positive, "positive_metrics": metrics(positive),
            "net_fp_vs_repeated_clean": counts["fp"] - repeat * clean_counts["fp"],
            "net_fn_vs_repeated_clean": counts["fn"] - repeat * clean_counts["fn"],
            "positive_tile_condition_appearances": sum(row["strata"]["positive"]["samples"] for row in rows),
            "positive_tile_condition_appearances_without_tp": sum(
                row["strata"]["positive"]["positive_tiles_without_tp"] for row in rows),
            "empty_tile_condition_appearances_with_predictions": sum(
                row["strata"]["empty"]["empty_tiles_with_predictions"] for row in rows)}


def load_assignments(instances, expected, sources):
    assignments = defaultdict(dict)
    with Path(instances).open(encoding="utf-8-sig") as stream:
        for line in stream:
            item = json.loads(line)
            source, separator, _ = item["sample_id"].partition("@")
            condition = item["condition_id"]
            if (item["stream"] != "val" or not separator or source not in sources or condition not in expected
                    or source in assignments[condition]
                    or condition != f"{item['kind']}_l{item['level']}"):
                raise ValueError("Invalid/duplicate validation instance or source/condition")
            assignments[condition][source] = {axis: item[axis] for axis in AXES}
    if set(assignments) != expected or any(set(rows) != sources for rows in assignments.values()):
        raise ValueError("Validation instance coverage differs")
    return assignments


def analyze(directories, instances, audit_path, boundaries):
    audit = read_json(audit_path)
    if (audit.get("status") != "completed_matched_study_independently_verified"
            or audit.get("test_accessed") is not False or set(directories) != set(MODELS)):
        raise ValueError("An intact completed matched study audit is required")
    hashes = {}

    def read_saved(path):
        path = Path(path).resolve()
        name, digest = str(path), sha256(path)
        if audit["source_hashes"].get(name) != digest:
            raise ValueError(f"Saved input differs from the completion audit: {path}")
        hashes[name] = digest
        return read_json(path)

    clean = {mode: read_saved(Path(directories[mode]) / "clean_metrics.json") for mode in MODELS}
    summaries = {mode: read_saved(Path(directories[mode]) / "summary.json") for mode in MODELS}
    for record in clean.values():
        validate_sources(record)
        require_metrics(metrics(record["counts"]), record["metrics"])
    for mode in MODELS[1:]:
        require_same_sources(clean[mode]["by_source"], clean["b0"]["by_source"])
    expected = {f"{family}_l{level}" for family in summaries["b0"]["family_mf1"] for level in range(1, 6)}
    if len(expected) != 45:
        raise ValueError("Expected nine-family, five-level benchmark")
    instances = Path(instances).resolve()
    if sha256(instances) != audit["source_hashes"].get(str(instances)):
        raise ValueError("Validation instances differ from the completion audit")
    hashes[str(instances)] = sha256(instances)
    assignments = load_assignments(instances, expected, set(clean["b0"]["by_source"]))
    models, conditions, subgroups, clean_sources = {}, {}, [], []
    for mode in MODELS:
        directory = Path(directories[mode])
        combined = read_saved(directory / "conditions.json")
        files = {path.stem: path for path in (directory / "conditions").glob("*.json")}
        if set(files) != expected or set(combined) != expected or summaries[mode]["condition_count"] != 45:
            raise ValueError("Saved 45-condition coverage differs")
        model_rows, conditions[mode] = [], {}
        for name in sorted(expected):
            condition = read_saved(files[name])
            if condition != combined[name] or condition["condition_id"] != name:
                raise ValueError("Individual and combined condition reports differ")
            require_metrics(metrics(condition["counts"]), condition["metrics"])
            partition = condition_subgroups(condition, clean[mode], assignments[name])
            subgroups.extend({"model": mode, **row} for row in partition)
            sources = condition["by_source"]
            strata = {label: describe_sources({key: row for key, row in sources.items()
                                              if (row["foreground_pixels"] > 0) == positive})
                      for label, positive in (("empty", False), ("positive", True))}
            row = {"condition": name, "family": condition["kind"], "level": condition["level"],
                   "counts": condition["counts"], "metrics": metrics(condition["counts"]), "strata": strata,
                   "net_fp_vs_clean": condition["counts"]["fp"] - clean[mode]["counts"]["fp"],
                   "net_fn_vs_clean": condition["counts"]["fn"] - clean[mode]["counts"]["fn"]}
            model_rows.append(row)
            conditions[mode][name] = condition
        aggregate = aggregate_conditions(model_rows, clean[mode]["counts"])
        if not math.isclose(aggregate["mean_condition_f1"], summaries[mode]["corruption_mf1"], rel_tol=0, abs_tol=1e-15):
            raise ValueError("Condition-derived mF1 differs from completed summary")
        family_rows = [{"family": family, **aggregate_conditions([row for row in model_rows if row["family"] == family],
                                                               clean[mode]["counts"])}
                       for family in summaries["b0"]["family_mf1"]]
        for row in family_rows:
            if not math.isclose(row["mean_condition_f1"], summaries[mode]["family_mf1"][row["family"]], rel_tol=0, abs_tol=1e-15):
                raise ValueError("Condition-derived family mF1 differs")
        score = {"mf1": aggregate["mean_condition_f1"], "clean_f1": metrics(clean[mode]["counts"])["f1"],
                 "clean_recall": metrics(clean[mode]["counts"])["recall"],
                 "corruption_recall": aggregate["pooled_metrics"]["recall"],
                 "empty_tile_fpr": aggregate["empty_metrics"]["background_fpr"]}
        for key, value in score.items():
            if not math.isclose(value, audit["scores"][mode][key], rel_tol=0, abs_tol=1e-15):
                raise ValueError("Count-derived score differs from independent completion audit")
        models[mode] = {"clean": describe_sources(clean[mode]["by_source"]),
                        "clean_size_groups": size_groups(clean[mode]["by_source"], boundaries),
                        "clean_strata": {label: describe_sources({key: row for key, row in clean[mode]["by_source"].items()
                                                                if (row["foreground_pixels"] > 0) == positive})
                                          for label, positive in (("empty", False), ("positive", True))},
                        "corruption": aggregate, "conditions": model_rows, "families": family_rows,
                        "levels": [{"level": level, **aggregate_conditions([row for row in model_rows if row["level"] == level],
                                                                            clean[mode]["counts"])} for level in range(1, 6)]}
        for name, row in sorted(clean[mode]["by_source"].items()):
            clean_sources.append({"model": mode, "source": name, "foreground_pixels": row["foreground_pixels"],
                                  **row["counts"], **metrics(row["counts"])})
    comparisons = []
    for candidate, reference in PAIRS:
        identity = {"candidate": candidate, "reference": reference}
        comparisons.append({**identity, "scope": "clean", **paired_sources(clean[candidate]["by_source"], clean[reference]["by_source"])})
        for name in sorted(expected):
            comparisons.append({**identity, "scope": name,
                                **paired_sources(conditions[candidate][name]["by_source"], conditions[reference][name]["by_source"])})
    subgroup_totals = []
    for mode in MODELS:
        for axis in AXES:
            selected = [row for row in subgroups if row["model"] == mode and row["axis"] == axis]
            for value in sorted({row["value"] for row in selected}):
                rows = [row for row in selected if row["value"] == value]
                counts = pooled_counts(rows)
                reference = pooled_counts({"counts": row["same_sources_clean_counts"]} for row in rows)
                subgroup_totals.append({"model": mode, "axis": axis, "value": value,
                                        "condition_contributions": len(rows), "sample_appearances": sum(row["samples"] for row in rows),
                                        "counts": counts, "same_sources_clean_counts": reference,
                                        "perturbed": metrics(counts), "same_sources_clean": metrics(reference),
                                        "net_fp": counts["fp"] - reference["fp"], "net_fn": counts["fn"] - reference["fn"]})
            if pooled_counts(selected) != models[mode]["corruption"]["counts"]:
                raise ValueError("Aggregate subgroup axis does not recover all condition counts")
    for name, expected_hash in hashes.items():
        if sha256(name) != expected_hash:
            raise ValueError(f"Saved count input changed during error analysis: {name}")
    hashes[str(Path(audit_path).resolve())] = sha256(audit_path)
    return {"status": "completed_matched_saved_count_error_review", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "new_model_inference": False, "raw_images_read": False, "training_started": False, "test_accessed": False,
            "models": models, "subgroups": subgroups, "subgroup_totals": subgroup_totals,
            "comparisons": comparisons, "clean_by_source": clean_sources, "source_hashes": hashes,
            "gate": audit["gate"], "condition_count_per_model": 45, "subgroup_count": len(subgroups),
            "partition_checks_passed": True, "foreground_area_boundaries": boundaries,
            "limitations": ["Counts give net pixel changes, not joint pixel error transitions or correction-action benefits.",
                            "TP=0 on a positive tile does not imply absence of all positive predictions.",
                            "Foreground area is labelled pixel area per tile, not building count or physical size.",
                            "Repeated appearances of the same tile are not independent observations.",
                            "One seed, Sierre validation only, no spatial bootstrap or significance claim."]}


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "_"))
        elif not isinstance(value, list):
            result[name] = value
    return result


def write_csv(path, rows):
    rows = [flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    values = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    keys = {"schema_version", "completion_audit", "completion_audit_sha256", "evaluations", "instances", "output_dir", "foreground_area_boundaries"}
    if set(values) != keys or values["schema_version"] != 1 or set(values["evaluations"]) != set(MODELS):
        raise ValueError("Unexpected matched error review configuration")
    def path(value):
        return (config.parent / value).resolve()
    output = path(values["output_dir"])
    if output.exists():
        raise FileExistsError("Error review already exists; preserve previous evidence")
    audit_path = path(values["completion_audit"])
    if sha256(audit_path) != values["completion_audit_sha256"]:
        raise ValueError("Bound completion audit changed")
    report = analyze({mode: path(directory) for mode, directory in values["evaluations"].items()},
                     path(values["instances"]), audit_path, values["foreground_area_boundaries"])
    report["source_hashes"].update({str(config): sha256(config), str(Path(__file__).resolve()): sha256(__file__),
                                  str(Path(__file__).with_name("analyze_valais_saved_subgroups.py")): sha256(Path(__file__).with_name("analyze_valais_saved_subgroups.py"))})
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("subgroups", report["subgroups"]), ("subgroup_totals", report["subgroup_totals"]),
                       ("comparisons", report["comparisons"]), ("clean_by_source", report["clean_by_source"])):
        write_csv(output / (name + ".csv"), rows)
    for name in ("conditions", "families", "levels", "clean_size_groups"):
        write_csv(output / (name + ".csv"), [{"model": mode, **row} for mode, model in report["models"].items() for row in model[name]])
    outputs = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "completion.json").write_text(json.dumps({"status": report["status"], "test_accessed": False,
                                                          "output_sha256": outputs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "condition_count_per_model", "subgroup_count", "partition_checks_passed", "new_model_inference", "test_accessed")}))


if __name__ == "__main__":
    main()
