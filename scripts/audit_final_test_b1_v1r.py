"""Read-only, sealed forensic certification of existing V1 test score records.

This program never opens raw held-out data, loads a checkpoint, or runs inference.
The original V1 output is immutable and remains formally incomplete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_test_b1_v1r_audit.yaml"
PROTOCOL_DIR = ROOT / "runs/final_test_b1_v1r_protocol"
SOURCE = ROOT / "scripts/audit_final_test_b1_v1r.py"
TEST_SOURCE = ROOT / "tests/test_audit_final_test_b1_v1r.py"
KINDS = ("brightness", "contrast", "gamma", "color_bias", "haze", "shadow",
         "noise", "blur", "jpeg")
CELLS = tuple(f"{kind}_l{level}" for kind in KINDS for level in range(1, 6))
COUNT_KEYS = ("tn", "fp", "fn", "tp")
SHA = re.compile(r"[0-9a-f]{64}\Z")
TILE_ID = re.compile(r"test/([^/\\]+)@x(0|[1-9][0-9]*)_y(0|[1-9][0-9]*)\Z")
STATIC_INPUTS = ("original_protocol_document", "original_config", "original_approval",
                 "original_pretest_seal", "original_evaluator", "original_guard_config",
                 "original_corruption_config", "original_checkpoint", "failure_audit",
                 "original_stdout", "original_stderr")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Nonfinite JSON number: {value}")))


def path_for(config: dict[str, Any], key: str) -> Path:
    value = config[key]
    require(isinstance(value, str) and bool(value), f"Invalid path: {key}")
    return (CONFIG.parent / value).resolve()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Invalid recovery config")
    require(config["schema_version"] == 1 and
            config["name"] == "levir_final_test_b1_seed42_v1r_audit_only" and
            config["status"] == "draft_before_audit_seal", "Unknown recovery configuration")
    for key, value in (("expected_sources", 128), ("expected_tiles", 2048),
                       ("expected_conditions", 45), ("expected_instances", 92160),
                       ("test_seed", 310944), ("bootstrap_seed", 20260926),
                       ("bootstrap_replicates", 5000)):
        require(type(config[key]) is int and config[key] == value, f"Frozen setting differs: {key}")
    require(sha(path_for(config, "protocol_document")) ==
            config["protocol_document_sha256"], "Recovery protocol hash changed")
    return config


def static_and_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = path_for(config, "original_output")
    require(output.is_dir() and not output.is_symlink(), "Original V1 output missing/symlinked")
    require(not (output / "guard_completion.json").exists(),
            "V1 guard-completion marker unexpectedly exists")
    conditions = output / "conditions"
    require({path.name for path in conditions.glob("*.json")} ==
            {f"{cell}.json" for cell in CELLS}, "Condition file set differs")
    result = {f"static/{key}": path_for(config, key) for key in STATIC_INPUTS}
    for path in output.rglob("*"):
        require(not path.is_symlink(), f"Symlink in original output: {path}")
        if path.is_file():
            result[f"v1/{path.relative_to(output).as_posix()}"] = path
    require(len(result) == len(STATIC_INPUTS) + 59, "Original V1 artifact count differs")
    for key, path in result.items():
        require(path.is_file() and not path.is_symlink(), f"Missing/symlinked input: {key}")
    return dict(sorted(result.items()))


def input_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha(path) for key, path in static_and_output_paths(config).items()}


def write_json_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"Refuse to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    result = {"schema_version": 1, "status": "bytes_snapshotted_before_score_interpretation",
              "scores_interpreted": False, "new_test_inference": False,
              "input_sha256": input_hashes(config)}
    write_json_new(path_for(config, "snapshot"), result)
    return result


def approval_and_seal(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    approval = yaml.safe_load(path_for(config, "approval").read_text(encoding="utf-8"))
    require(approval["schema_version"] == 1 and
            approval["status"] == "approved_audit_only_before_score_interpretation" and
            approval["scores_interpreted"] is False and approval["new_test_inference"] is False and
            bool(approval["reviewer"]) and bool(approval["notes"]), "Recovery approval invalid")
    for key, path in (("protocol_document_sha256", path_for(config, "protocol_document")),
                      ("config_sha256", CONFIG), ("source_sha256", SOURCE),
                      ("test_source_sha256", TEST_SOURCE),
                      ("input_snapshot_sha256", path_for(config, "snapshot"))):
        require(approval[key] == sha(path), f"Approval binding differs: {key}")
    seal = json_file(path_for(config, "seal"))
    require(seal["schema_version"] == 1 and seal["status"] == "sealed_before_score_interpretation"
            and seal["scores_interpreted"] is False and seal["new_test_inference"] is False,
            "Recovery seal invalid")
    for key, path in (("approval_sha256", path_for(config, "approval")),
                      ("protocol_document_sha256", path_for(config, "protocol_document")),
                      ("config_sha256", CONFIG), ("source_sha256", SOURCE),
                      ("test_source_sha256", TEST_SOURCE),
                      ("input_snapshot_sha256", path_for(config, "snapshot")),
                      ("unit_test_evidence_sha256", PROTOCOL_DIR / "unittest_evidence.json"),
                      ("smoke_evidence_sha256", PROTOCOL_DIR / "synthetic_smoke.json")):
        require(seal[key] == sha(path), f"Recovery seal binding differs: {key}")
    return approval, seal


def create_seal(config: dict[str, Any]) -> dict[str, Any]:
    approval = yaml.safe_load(path_for(config, "approval").read_text(encoding="utf-8"))
    require(approval["status"] == "approved_audit_only_before_score_interpretation" and
            approval["scores_interpreted"] is False and approval["new_test_inference"] is False,
            "Audit-only approval missing")
    snapshot_record = json_file(path_for(config, "snapshot"))
    require(snapshot_record["input_sha256"] == input_hashes(config),
            "Input changed since byte snapshot")
    for key, path in (("protocol_document_sha256", path_for(config, "protocol_document")),
                      ("config_sha256", CONFIG), ("source_sha256", SOURCE),
                      ("test_source_sha256", TEST_SOURCE),
                      ("input_snapshot_sha256", path_for(config, "snapshot"))):
        require(approval[key] == sha(path), f"Approval differs: {key}")
    tests = json_file(PROTOCOL_DIR / "unittest_evidence.json")
    smoke = json_file(PROTOCOL_DIR / "synthetic_smoke.json")
    require(tests["status"] == "passed" and smoke["status"] == "passed" and
            tests["synthetic_only"] is True and smoke["synthetic_only"] is True,
            "Synthetic preseal checks incomplete")
    result = {"schema_version": 1, "status": "sealed_before_score_interpretation",
              "scores_interpreted": False, "new_test_inference": False,
              "approval_sha256": sha(path_for(config, "approval")),
              "protocol_document_sha256": sha(path_for(config, "protocol_document")),
              "config_sha256": sha(CONFIG), "source_sha256": sha(SOURCE),
              "test_source_sha256": sha(TEST_SOURCE),
              "input_snapshot_sha256": sha(path_for(config, "snapshot")),
              "unit_test_evidence_sha256": sha(PROTOCOL_DIR / "unittest_evidence.json"),
              "smoke_evidence_sha256": sha(PROTOCOL_DIR / "synthetic_smoke.json")}
    write_json_new(path_for(config, "seal"), result)
    return result


def compare_number(actual: Any, expected: float | int, label: str) -> None:
    require(type(actual) in (float, int) and math.isfinite(actual) and
            abs(actual - expected) <= 1e-12, f"Numerical mismatch: {label}")


def keyed_seed(*parts: Any) -> int:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:8], "big")


def replay_instances(config: dict[str, Any], audit: dict[str, Any],
                     manifest: dict[str, Any], instance_path: Path) -> tuple[list[str], Counter]:
    inventory = audit["inventory"]
    entries = inventory["entries"]
    require(audit["schema_version"] == 1 and audit["split"] == "test" and
            inventory["split"] == "test" and inventory["source_pairs"] == 128 and
            inventory["tiles"] == 2048 and inventory["tile_size"] == 256 and
            len(entries) == 128, "Test audit inventory differs")
    inventory_payload = {"split": "test", "tile_size": 256, "entries": entries}
    inventory_digest = hashlib.sha256(json.dumps(inventory_payload, sort_keys=True).encode()).hexdigest()
    require(inventory["inventory_sha256"] == inventory_digest,
            "Original test inventory fingerprint differs")
    names = [entry["name"] for entry in entries]
    require(names == sorted(names) and len(set(names)) == 128, "Source names differ")
    require(set(audit["file_sha256"]) == set(names), "Audit source hashes incomplete")
    ids: list[str] = []
    for entry in entries:
        require(entry["size"] == [1024, 1024] and len(entry["bytes"]) == 3 and
                all(type(value) is int and value > 0 for value in entry["bytes"]),
                "Source inventory geometry/bytes differ")
        require(all(set(audit["file_sha256"][entry["name"]]) == {"image_a", "image_b", "mask"}
                    and SHA.fullmatch(value) for value in audit["file_sha256"][entry["name"]].values()),
                "Malformed source file hashes")
        ids.extend(f"test/{entry['name']}@x{x}_y{y}" for y in range(0, 1024, 256)
                   for x in range(0, 1024, 256))
    require(len(ids) == 2048 and len(set(ids)) == 2048 and
            all(TILE_ID.fullmatch(value) for value in ids), "Test tile IDs differ")
    protocol = yaml.safe_load(path_for(config, "original_corruption_config").read_text(encoding="utf-8"))
    require(protocol["name"] == "levir_tile_corruptions_v1" and
            protocol["seeds"] == {"train": 310942, "val": 310943, "test": 310944},
            "Original corruption recipe changed")
    counts: Counter = Counter()
    strata_counts: Counter = Counter()
    with instance_path.open("r", encoding="utf-8") as stream:
        for kind in KINDS:
            directions = sorted(protocol["corruptions"][kind])
            strata = list(itertools.product(directions, ("A", "B"), ("global", "local")))
            order = sorted(ids, key=lambda value: (keyed_seed(310944, kind, value), value))
            assigned = {value: strata[index % len(strata)] for index, value in enumerate(order)}
            assigned_counts = Counter(assigned.values())
            require(max(assigned_counts.values()) - min(assigned_counts.values()) <= 1,
                    "Unbalanced test nuisance strata")
            for level in range(1, 6):
                cell = f"{kind}_l{level}"
                for sample_id in ids:
                    direction, date, spatial = assigned[sample_id]
                    seed = keyed_seed(protocol["name"], 310944, "test", sample_id,
                                      kind, direction, date, spatial, 0, 0)
                    if spatial == "global":
                        mask = {"mode": "global"}
                    else:
                        rng = np.random.Generator(np.random.PCG64(keyed_seed(seed, "mask")))
                        ranges = protocol["local_mask"]
                        mask = {"mode": "ellipse", "feather": ranges["feather"],
                                **{key: float(rng.uniform(*ranges[key + "_range"]))
                                   for key in ("center_x", "center_y", "radius_x", "radius_y")}}
                    expected = {"sample_id": sample_id, "condition_id": cell, "stream": "test",
                                "epoch": 0, "view": 0, "kind": kind, "level": level,
                                "direction": direction, "target_date": date,
                                "spatial_mode": spatial, "seed": seed,
                                "parameters": protocol["corruptions"][kind][direction][level - 1],
                                "injection_mask": mask}
                    line = stream.readline()
                    require(bool(line), "Instance JSONL ended early")
                    require(json.loads(line) == expected, "Reserved-seed instance replay differs")
                    counts[cell] += 1
                    strata_counts["|".join((cell, direction, date, spatial))] += 1
        require(stream.readline() == "", "Instance JSONL has extra rows")
    require(manifest == {"schema_version": 1, "split": "test",
                         "status": "generated_after_pretest_seal", "stream_seed": 310944,
                         "condition_count": 45, "instances": 92160, "source_pairs": 128,
                         "tiles": 2048, "instance_file_sha256": sha(instance_path),
                         "test_data_audit_sha256": sha(instance_path.parent / "test_data_audit.json"),
                         "condition_counts": dict(counts), "stratum_counts": dict(strata_counts)},
            "Test manifest does not match deterministic replay")
    return ids, strata_counts


def report_metrics(counts: dict[str, int]) -> dict[str, float] | None:
    tn, fp, fn, tp = (counts[key] for key in COUNT_KEYS)
    total = tn + fp + fn + tp
    if not total:
        return None
    div = lambda top, bottom: top / bottom if bottom else 0.0
    return {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "iou": div(tp, tp + fp + fn),
            "oa": (tn + tp) / total, "foreground_recall": div(tp, tp + fn),
            "background_fpr": div(fp, tn + fp), "foreground_fn_rate": div(fn, tp + fn)}


def check_counts_report(value: dict[str, Any], label: str,
                        *, samples: int | None = None, regional: bool = False) -> dict[str, int]:
    counts = value["counts"]
    require(set(counts) == set(COUNT_KEYS) and
            all(type(counts[key]) is int and counts[key] >= 0 for key in COUNT_KEYS),
            f"Invalid integer confusion counts: {label}")
    total = sum(counts.values())
    require(value["pixels"] == total and value["foreground_pixels"] == counts["tp"] + counts["fn"]
            and value["background_pixels"] == counts["tn"] + counts["fp"],
            f"Pixel totals differ: {label}")
    metrics = report_metrics(counts)
    require((metrics is None) == (value["metrics"] is None), f"Null metrics differ: {label}")
    if metrics is not None:
        require(set(value["metrics"]) == set(metrics), f"Metric keys differ: {label}")
        for key, expected in metrics.items():
            compare_number(value["metrics"][key], expected, f"{label}/{key}")
    if samples is not None:
        require(value["samples"] == samples and total == samples * 256 * 256,
                f"Sample/pixel coverage differs: {label}")
    if regional:
        regions = value["regions"]
        require(set(regions) == {"injection_support", "outside_injection"},
                f"Regional records missing: {label}")
        inside = check_counts_report(regions["injection_support"], label + "/inside")
        outside = check_counts_report(regions["outside_injection"], label + "/outside")
        require(all(inside[key] + outside[key] == counts[key] for key in COUNT_KEYS),
                f"Regional confusion reconciliation differs: {label}")
    elif "regions" in value:
        require(value["regions"] is None, f"Clean report unexpectedly regional: {label}")
    return counts


def add_counts(values: Any) -> dict[str, int]:
    return {key: sum(value[key] for value in values) for key in COUNT_KEYS}


def check_condition(value: dict[str, Any], cell: str, sources: list[str],
                    expected_strata: Counter) -> None:
    regional = cell != "clean"
    kind, level = ("clean", 0) if not regional else (cell.rsplit("_l", 1)[0], int(cell[-1]))
    require(value["condition_id"] == cell and value["kind"] == kind and
            value["level"] == level and type(value["seconds"]) in (int, float) and
            math.isfinite(value["seconds"]) and value["seconds"] > 0,
            f"Condition metadata differs: {cell}")
    overall = check_counts_report(value, cell, samples=2048, regional=regional)
    by_source = value["by_source"]
    require(set(by_source) == set(sources), f"Source coverage differs: {cell}")
    for source in sources:
        check_counts_report(by_source[source], f"{cell}/{source}", samples=16,
                            regional=regional)
    require(add_counts(row["counts"] for row in by_source.values()) == overall,
            f"Source counts do not sum: {cell}")
    if regional:
        strata = value["joint_strata"]
        expected = {key.split("|", 1)[1]: count for key, count in expected_strata.items()
                    if key.startswith(cell + "|")}
        require(set(strata) == set(expected) and
                {key: row["samples"] for key, row in strata.items()} == expected,
                f"Joint strata coverage differs: {cell}")
        for name, row in strata.items():
            check_counts_report(row, f"{cell}/{name}", samples=expected[name], regional=True)
        require(add_counts(row["counts"] for row in strata.values()) == overall,
                f"Joint strata counts do not sum: {cell}")
        for region in ("injection_support", "outside_injection"):
            require(add_counts(row["regions"][region]["counts"] for row in by_source.values()) ==
                    value["regions"][region]["counts"], f"Source region totals differ: {cell}")
            require(add_counts(row["regions"][region]["counts"] for row in strata.values()) ==
                    value["regions"][region]["counts"], f"Stratum region totals differ: {cell}")
    else:
        require(value["joint_strata"] == {}, "Clean record has joint strata")


def bootstrap(clean: dict[str, Any], conditions: dict[str, Any], sources: list[str]) -> dict[str, Any]:
    clean_array = np.asarray([[clean["by_source"][name]["counts"][key] for key in COUNT_KEYS]
                              for name in sources], dtype=np.float64)
    condition_array = np.asarray([[[conditions[cell]["by_source"][name]["counts"][key]
                                    for key in COUNT_KEYS] for name in sources]
                                  for cell in CELLS], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(20260926))
    draws = rng.integers(0, len(sources), size=(5000, len(sources)))
    weights = np.zeros((5000, len(sources)), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(5000), len(sources)), draws.ravel()), 1)
    a = weights @ clean_array
    b = (weights @ condition_array.transpose(1, 0, 2).reshape(len(sources), -1)).reshape(5000, 45, 4)
    def f1(matrix: np.ndarray) -> np.ndarray:
        denominator = 2 * matrix[..., 3] + matrix[..., 1] + matrix[..., 2]
        return np.divide(2 * matrix[..., 3], denominator,
                         out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)
    clean_f1 = f1(a)
    corruption_mf1 = f1(b).mean(axis=1)
    results = {"clean_f1": clean_f1, "corruption_mf1": corruption_mf1,
               "clean_minus_corruption_mf1": clean_f1 - corruption_mf1}
    return {key: [float(value) for value in np.quantile(array, [0.025, 0.975])]
            for key, array in results.items()}


def check_summary(summary: dict[str, Any], clean: dict[str, Any],
                  conditions: dict[str, Any], sources: list[str]) -> None:
    require(summary["status"] == "completed" and summary["method"] ==
            "b1_ordinary_supervised_augmentation" and summary["checkpoint_epoch"] == 68 and
            summary["selection"] == "incumbent_seed42_clean_validation_f1_only" and
            summary["test_accessed"] is True and summary["test_evaluated"] is True and
            summary["no_test_based_selection"] is True and summary["model_training_run"] is False,
            "Original summary status or fixed selection differs")
    require(summary["clean"] == {key: clean[key] for key in
                                  ("counts", "metrics", "samples", "pixels")},
            "Summary clean descriptor differs")
    f1s = [report_metrics(conditions[cell]["counts"])["f1"] for cell in CELLS]
    mf1 = sum(f1s) / 45
    compare_number(summary["corruption_mf1"], mf1, "corruption_mf1")
    compare_number(summary["clean_minus_corruption_mf1"],
                   clean["metrics"]["f1"] - mf1, "clean_minus_corruption_mf1")
    require(summary["condition_count"] == 45 and summary["aggregation_rule"] ==
            "Pool pixels within each condition, then average condition F1s equally.",
            "Condition aggregation rule differs")
    require(set(summary["family_mf1"]) == set(KINDS) and
            set(summary["per_level_mf1"]) == set(str(i) for i in range(1, 6)),
            "Family/level aggregation keys differ")
    for kind in KINDS:
        compare_number(summary["family_mf1"][kind],
                       sum(report_metrics(conditions[f"{kind}_l{level}"]["counts"])["f1"]
                           for level in range(1, 6)) / 5, f"family_mf1/{kind}")
    for level in range(1, 6):
        compare_number(summary["per_level_mf1"][str(level)],
                       sum(report_metrics(conditions[f"{kind}_l{level}"]["counts"])["f1"]
                           for kind in KINDS) / 9, f"per_level_mf1/{level}")
    pooled_counts = add_counts(conditions[cell]["counts"] for cell in CELLS)
    require(summary["pooled_corruption"]["counts"] == pooled_counts,
            "Pooled corruption counts differ")
    pooled = summary["pooled_corruption"]
    require(pooled["pixels"] == sum(pooled_counts.values()) and
            pooled["foreground_pixels"] == pooled_counts["tp"] + pooled_counts["fn"] and
            pooled["background_pixels"] == pooled_counts["tn"] + pooled_counts["fp"],
            "Pooled corruption pixel totals differ")
    for key, expected in report_metrics(pooled_counts).items():
        compare_number(pooled["metrics"][key], expected, f"pooled/{key}")
    require(summary["net_fp_vs_repeated_clean"] == pooled_counts["fp"] - 45 * clean["counts"]["fp"]
            and summary["net_fn_vs_repeated_clean"] ==
            pooled_counts["fn"] - 45 * clean["counts"]["fn"], "Net FP/FN differs")
    boot = summary["source_pair_bootstrap"]
    require(boot["unit"] == "original_source_pair" and boot["sources"] == 128 and
            boot["replicates"] == 5000 and boot["seed"] == 20260926 and boot["method"] ==
            "same_source_draw_for_clean_and_all_45_cells" and
            boot["limitation"] ==
            "Conditional on one checkpoint; not an interval over training seeds.",
            "Bootstrap method differs")
    intervals = bootstrap(clean, conditions, sources)
    require(set(boot["percentile_95"]) == set(intervals), "Bootstrap interval keys differ")
    for key, pair in intervals.items():
        require(len(boot["percentile_95"][key]) == 2, f"Bootstrap interval length differs: {key}")
        for index, value in enumerate(pair):
            compare_number(boot["percentile_95"][key][index], value,
                           f"bootstrap/{key}/{index}")


def check_guard(config: dict[str, Any], summary: dict[str, Any],
                guard: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    seal = json_file(path_for(config, "original_pretest_seal"))
    approval = yaml.safe_load(path_for(config, "original_approval").read_text(encoding="utf-8"))
    require(guard["status"] == "failed_guard" and guard["evaluation_status"] == "completed" and
            guard["error"] == "ValueError: Expected lowercase SHA256" and
            guard["test_accessed"] is True and guard["test_evaluated"] is True and
            guard["training_started"] is False and guard["automatic_resume"] is False and
            guard["execution"] == "foreground_in_process_between_batches" and
            guard["guard_pid"] == 9552 and guard["callback_count"] >= len(guard["observations"]),
            "Original guard final state differs")
    require(guard["guard_config_sha256"] == sha(path_for(config, "original_guard_config")) ==
            seal["guard_config_sha256"] == approval["guard_config_sha256"] ==
            provenance["config_sha256"] == summary["temperature_guard"]["config_sha256"],
            "Guard configuration binding differs")
    require(guard["guard_source_sha256"] == provenance["source_sha256"] ==
            summary["temperature_guard"]["source_sha256"] and
            summary["temperature_guard"]["status"] == "completed" and
            summary["temperature_guard"]["completion_file"] == "guard_completion.json",
            "Guard source/completion record differs")
    for name, digest in provenance["source_sha256"].items():
        require(SHA.fullmatch(digest) is not None and sha(ROOT / name) == digest,
                f"Guard source changed: {name}")
        require(sha(path_for(config, "original_output") / "temperature_guard/source" / name) == digest,
                f"Attached guard source snapshot differs: {name}")
    require(sha(path_for(config, "original_output") / "temperature_guard/config.yaml") ==
            provenance["config_sha256"], "Attached guard configuration differs")
    require(guard["settings"]["gpu_index"] == 0 and
            guard["settings"]["gpu_uuid"] == summary["hardware_preflight"]["gpu_uuid"] and
            summary["hardware_preflight"]["guard_config_sha256"] ==
            provenance["config_sha256"] and
            summary["hardware_preflight"]["guard_source_sha256"] ==
            provenance["source_sha256"], "Guard GPU identity differs")
    observations = guard["observations"]
    require(len(observations) == 143 and guard["decision"] == "ok" and
            guard["sensor_error"] is None and guard["sensor_failures"] == 0 and
            guard["hot_samples"] == 0, "Guard observations/decision differ")
    previous = -math.inf
    peak = 0
    for item in observations:
        timestamp = datetime.fromisoformat(item["sample_utc"])
        require(timestamp.tzinfo is not None and item["decision"] == "ok" and
                item["sensor_error"] is None and type(item["temperature_c"]) is int and
                0 <= item["temperature_c"] < guard["settings"]["confirmed_pause_c"] and
                type(item["monotonic_seconds"]) in (float, int) and
                math.isfinite(item["monotonic_seconds"]) and
                item["monotonic_seconds"] >= previous, "Guard sensor sample differs")
        previous = item["monotonic_seconds"]
        peak = max(peak, item["temperature_c"])
    require(guard["temperature_c"] == observations[-1]["temperature_c"] and
            peak == 64 and guard["settings"]["confirmed_pause_c"] == 85 and
            guard["settings"]["immediate_pause_c"] == 86,
            "Guard final/maximum temperature differs")
    return {"observations": len(observations), "callbacks": guard["callback_count"],
            "max_temperature_c": peak, "final_temperature_c": guard["temperature_c"],
            "original_guard_status": "failed_guard", "embedded_guard_status": "completed"}


def certify(config: dict[str, Any]) -> dict[str, Any]:
    _, seal = approval_and_seal(config)
    snap = json_file(path_for(config, "snapshot"))
    require(snap["status"] == "bytes_snapshotted_before_score_interpretation" and
            snap["scores_interpreted"] is False and snap["new_test_inference"] is False and
            snap["input_sha256"] == input_hashes(config), "Original inputs changed since seal")
    output = path_for(config, "original_output")
    failure = json_file(path_for(config, "failure_audit"))
    status = json_file(output / "status.json")
    progress = json_file(output / "progress.json")
    stderr = path_for(config, "original_stderr").read_text(encoding="utf-8")
    require(failure["status"] == "invalid_incomplete_no_automatic_retry" and
            failure["failure_stage"] == "durable_guard_completion_commit_after_summary_and_status_write"
            and failure["completed_condition_files"] == 45 and
            failure["valid_final_test_result"] is False and
            failure["automatic_rerun_authorized"] is False and
            status == {"status": "failed_incomplete_no_automatic_retry", "guarded_result_valid": False,
                       "error": "ValueError: Expected lowercase SHA256", "test_accessed": True,
                       "test_evaluated": True} and
            progress["completed_conditions"] == progress["expected_conditions"] == 45,
            "Original failure/progress state differs")
    require('File "E:\\new research\\detection\\scripts\\final_test_b1_v1.py", line 808, in run' in stderr
            and '"summary_sha256": _sha((job.output / "summary.json").read_bytes())' in stderr
            and "ValueError: Expected lowercase SHA256" in stderr and
            stderr.count("Traceback (most recent call last):") == 1,
            "Post-scoring bookkeeping traceback differs")
    failure_hashes = failure["artifacts_sha256"]
    for name, digest in failure_hashes.items():
        target = (path_for(config, "original_stderr") if name == "evaluation_stderr.log"
                  else output / name)
        require(sha(target) == digest, f"Failure audit hash differs: {name}")
    require(failure["pretest_seal_sha256"] == sha(path_for(config, "original_pretest_seal")) and
            failure["frozen_evaluator_source_sha256"] ==
            sha(path_for(config, "original_evaluator")), "V1 failure binding differs")
    original_config = yaml.safe_load(path_for(config, "original_config").read_text(encoding="utf-8"))
    original_approval = yaml.safe_load(path_for(config, "original_approval").read_text(encoding="utf-8"))
    original_seal = json_file(path_for(config, "original_pretest_seal"))
    require(original_seal["status"] == "sealed_before_test" and
            original_seal["test_accessed"] is False and original_seal["test_evaluated"] is False and
            original_seal["source_sha256"] == original_approval["source_sha256"],
            "Original pretest seal differs")
    for key, path_key in (("protocol_document_sha256", "original_protocol_document"),
                          ("job_sha256", "original_config"),
                          ("approval_sha256", "original_approval"),
                          ("checkpoint_sha256", "original_checkpoint"),
                          ("guard_config_sha256", "original_guard_config")):
        require(original_seal[key] == sha(path_for(config, path_key)),
                f"Original seal binding differs: {key}")
    require(original_config["best_checkpoint_sha256"] ==
            original_seal["checkpoint_sha256"] and
            original_config["corruption_config_sha256"] ==
            sha(path_for(config, "original_corruption_config")) and
            original_approval["job_sha256"] == original_seal["job_sha256"],
            "Original config/approval binding differs")
    for name, digest in original_approval["source_sha256"].items():
        require(SHA.fullmatch(digest) is not None and sha(ROOT / name) == digest,
                f"Original source changed: {name}")
    audit = json_file(output / "test_data_audit.json")
    manifest = json_file(output / "test_manifest.json")
    ids, strata = replay_instances(config, audit, manifest, output / "test_instances.jsonl")
    sources = sorted({sample_id.rsplit("@x", 1)[0] for sample_id in ids})
    require(len(sources) == 128, "Source-pair grouping differs")
    clean = json_file(output / "clean.json")
    check_condition(clean, "clean", sources, strata)
    combined = json_file(output / "conditions.json")
    require(set(combined) == set(CELLS), "Combined condition set differs")
    summary = json_file(output / "summary.json")
    require(summary["protocol_document_sha256"] ==
            sha(path_for(config, "original_protocol_document")) and
            summary["job_sha256"] == sha(path_for(config, "original_config")) and
            summary["approval_sha256"] == sha(path_for(config, "original_approval")) and
            summary["pretest_seal_sha256"] == sha(path_for(config, "original_pretest_seal")) and
            summary["checkpoint_sha256"] == sha(path_for(config, "original_checkpoint")) and
            summary["test_data_audit_sha256"] == sha(output / "test_data_audit.json") and
            summary["test_manifest_sha256"] == sha(output / "test_manifest.json") and
            summary["test_instances_sha256"] == sha(output / "test_instances.jsonl") and
            summary["clean_result_sha256"] == sha(output / "clean.json") and
            summary["conditions_result_sha256"] == sha(output / "conditions.json") and
            summary["source_sha256"] == original_approval["source_sha256"] and
            set(summary["condition_file_sha256"]) == set(CELLS),
            "Original summary provenance differs")
    for cell in CELLS:
        path = output / "conditions" / f"{cell}.json"
        value = json_file(path)
        require(value == combined[cell] and summary["condition_file_sha256"][cell] == sha(path),
                f"Individual condition commitment differs: {cell}")
        check_condition(value, cell, sources, strata)
    check_summary(summary, clean, combined, sources)
    guard = json_file(output / "temperature_guard_state.json")
    provenance = json_file(output / "temperature_guard/provenance.json")
    guard_facts = check_guard(config, summary, guard, provenance)
    require(summary["guarded_completion_required"] is True and
            not (output / "guard_completion.json").exists() and
            summary["hardware_preflight"]["device"] == "cuda:0" and
            str(summary["environment"]["device"]).startswith("cuda") and
            summary["peak_inference_gpu_memory_bytes"] > 0 and
            summary["seconds"] > 0, "Original incomplete completion/compute differs")
    require(snap["input_sha256"] == input_hashes(config), "Original inputs changed during audit")
    result = {"schema_version": 1, "status": "certified_recovered_scores",
              "recovery_type": "audit_only_post_scoring_bookkeeping_failure",
              "original_v1_valid": False, "new_test_inference": False,
              "recovered_scores_certified": True, "original_guard_completion_exists": False,
              "original_status": "failed_incomplete_no_automatic_retry",
              "original_guard_status": "failed_guard", "test_based_selection": False,
              "checkpoint_sha256": original_seal["checkpoint_sha256"],
              "source_pair_count": len(sources), "tile_count": len(ids),
              "condition_count": len(CELLS), "instance_count": sum(manifest["condition_counts"].values()),
              "guard_audit": guard_facts, "original_summary_sha256": sha(output / "summary.json"),
              "input_snapshot_sha256": sha(path_for(config, "snapshot")),
              "preaudit_seal_sha256": sha(path_for(config, "seal")),
              "input_sha256": snap["input_sha256"],
              "limitations": ["V1 has no durable guard_completion.json; this is independent recovery, not V1 completion.",
                              "Single checkpoint and one held-out split; bootstrap is conditional on original source pairs.",
                              "No raw test files were reopened and no new test inference was run by V1R."]}
    write_json_new(path_for(config, "output_dir") / "recovery_certificate.json", result)
    return result


def synthetic_smoke() -> dict[str, Any]:
    synthetic = {"tn": 5, "fp": 2, "fn": 1, "tp": 3}
    metrics = report_metrics(synthetic)
    require(metrics is not None, "Synthetic metrics absent")
    compare_number(metrics["f1"], 6 / 9, "synthetic_f1")
    row = {"counts": synthetic, "pixels": 11, "foreground_pixels": 4,
           "background_pixels": 7, "metrics": metrics, "samples": 11,
           "regions": None}
    check_counts_report(row, "synthetic")
    require(keyed_seed("fixture", 7) == keyed_seed("fixture", 7), "Synthetic seed nondeterministic")
    return {"status": "passed", "synthetic_only": True,
            "raw_test_access": False, "model_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "seal", "run", "synthetic-smoke"))
    args = parser.parse_args()
    if args.action == "synthetic-smoke":
        result = synthetic_smoke()
        write_json_new(PROTOCOL_DIR / "synthetic_smoke.json", result)
    else:
        config = load_config()
        result = {"snapshot": snapshot, "seal": create_seal, "run": certify}[args.action](config)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
