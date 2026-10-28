"""Independently verify the completed, resumed Valais study without inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.analyze_valais_saved_subgroups import read_json, sha256
from scripts.valais_matched_study import (
    check_training, descriptive_gates, load_study, score_metrics,
)


def validate_exposure(history, exposure):
    counts = Counter()
    streams = []
    for row in history:
        train = row["train"]
        if (train["source_samples"], train["pair_views"], train["optimizer_steps"],
                train["amp_skipped_steps"]) != (11025, 22050, 2757, 0):
            raise ValueError("Per-epoch matched training budget differs")
        if sum(train["exposure_counts"].values()) != 11025:
            raise ValueError("Per-epoch exposure counts differ")
        counts.update(train["exposure_counts"])
        streams.append({"epoch": row["epoch"], "sha256": train["exposure_stream_sha256"]})
    encoded = json.dumps(streams, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (exposure["epochs"] != len(history) or exposure["counts"] != dict(counts)
            or exposure["per_epoch_stream_sha256"] != streams
            or exposure["combined_stream_sha256"] != hashlib.sha256(encoded).hexdigest()):
        raise ValueError("Exposure summary differs from the saved history")


def audit(study_config, resume):
    config, values, paths, runs = load_study(study_config)
    study, resume = paths["output_dir"], Path(resume).resolve()
    comparison = read_json(study / "comparison.json")
    completion = read_json(study / "completion.json")
    if (completion != read_json(resume / "completion.json")
            or completion["status"] != "matched_pair_completed"
            or completion["test_accessed"] is not False
            or read_json(study / "status.json") != read_json(resume / "status.json")
            or read_json(study / "status.json")["status"] != "completed"):
        raise ValueError("Study completion/status mirrors differ or are incomplete")
    hashes = {}

    def fingerprint(path):
        path = Path(path).resolve()
        if any(part.lower() in ("test", "martigny") for part in path.parts):
            raise ValueError(f"Out-of-scope hash target: {path}")
        name = str(path)
        if name not in hashes:
            hashes[name] = sha256(path)
        return hashes[name]

    def bindings(records):
        for name, expected in records.items():
            if fingerprint(name) != expected:
                raise ValueError(f"Sealed input changed: {name}")
        return len(records)

    for name, key in (("status.json", "status_sha256"), ("comparison.json", "comparison_sha256"),
                      ("authorization.json", "authorization_sha256")):
        if fingerprint(study / name) != completion[key]:
            raise ValueError(f"Completion hash differs: {name}")
    original = read_json(study / "authorization.json")
    recovery = read_json(resume / "authorization.json")
    if (fingerprint(resume / "authorization.json") != completion["resume_authorization_sha256"]
            or recovery["original_authorization_sha256"] != completion["authorization_sha256"]
            or recovery["pause_record_sha256"] != completion["pause_record_sha256"]):
        raise ValueError("Resume/pause authorization hashes differ")
    checks = {"original_seal_files": bindings(original["bound_sha256"]),
              "resume_seal_files": bindings(recovery["bound_sha256"])}
    training, orders = {}, {}
    for mode, cfg in runs.items():
        directory = cfg.output_dir
        order = read_json(directory / "source_order.json")
        orders[mode] = order
        status = check_training(cfg, order)
        history = read_json(directory / "history.json")
        validate_exposure(history, read_json(directory / "training_exposure_summary.json"))
        if read_json(directory / "data_audit.json") != read_json(study / "live_data_audit.json"):
            raise ValueError("Captured training audit differs from the sealed B0 data")
        best = max(history, key=lambda row: row["val"]["metrics"]["f1"])
        if (status["best_epoch"] != best["epoch"]
                or status["best_val_f1"] != best["val"]["metrics"]["f1"]
                or order["initial_tensor_sha256"] != status["initial_model_tensor_sha256"]
                or fingerprint(directory / "source_order.json") != comparison["source_order_sha256"][mode]):
            raise ValueError("Best checkpoint selection/initialization/order metadata differs")
        guard = read_json(directory / "guard_completion.json")
        state = read_json(directory / guard["guard_state_path"])
        temperatures = [row["temperature_c"] for row in state["observations"]
                        if row.get("temperature_c") is not None]
        training[mode] = {"completed_epochs": 100, "best_epoch": status["best_epoch"],
                         "successful_updates": status["optimizer_steps"], "pair_views": status["pair_views"],
                         "amp_skips": status["amp_skipped_steps"], "source_pairs": order["source_pairs"],
                         "source_order_digest": order["source_order_sha256"],
                         "initial_tensor_digest": order["initial_tensor_sha256"],
                         "guard_state_path": guard["guard_state_path"], "guard_status": state["status"],
                         "guard_decision": state["decision"], "max_observed_temperature_c": max(temperatures),
                         "sensor_failures": state["sensor_failures"],
                         "timing_scope": order.get("timing_and_peak_scope", "entire_uninterrupted_run")}
        for name in ("history.json", "source_order.json", "status.json", "guard_completion.json",
                     "training_exposure_summary.json", "data_audit.json", guard["guard_state_path"]):
            fingerprint(directory / name)
    for key in ("source_order_sha256", "initial_tensor_sha256", "source_pairs", "batches"):
        if orders["b1"][key] != orders["b0e"][key]:
            raise ValueError(f"Actual matched training metadata differs: {key}")
    replay = read_json(resume / "prefix_replay.json")
    if (replay["saved_loader_rng_matches"] is not True
            or replay["expected_full_source_order_sha256"] != orders["b0e"]["source_order_sha256"]
            or fingerprint(resume / "prefix_replay.json") != orders["b0e"]["prefix_reconstruction_sha256"]):
        raise ValueError("Resume prefix/order reconstruction differs")
    directories = {"b0": paths["baseline_evaluation"],
                   **{mode: cfg.output_dir.with_name(cfg.output_dir.name + "_val") for mode, cfg in runs.items()}}
    scores, evaluations = {}, {}
    for mode, directory in directories.items():
        summary = read_json(directory / "summary.json")
        guard = read_json(directory / "guard_completion.json")
        state = read_json(directory / "temperature_guard_state.json")
        if (summary["status"] != "completed" or summary["split"] != "val"
                or summary["test_evaluated"] is not False or read_json(directory / "status.json") != summary
                or state["status"] != "completed" or state["decision"] != "ok"
                or guard["status"] != "completed"
                or guard["summary_sha256"] != fingerprint(directory / "summary.json")
                or guard["guard_state_sha256"] != fingerprint(directory / "temperature_guard_state.json")
                or comparison["source_summary_sha256"][mode] != fingerprint(directory / "summary.json")):
            raise ValueError("Guarded validation result/hash differs")
        if mode != "b0" and completion["evaluation_summaries"][mode] != fingerprint(directory / "summary.json"):
            raise ValueError("Completion evaluation summary hash differs")
        provenance = read_json(directory / "provenance.json")
        count = bindings(provenance["input_file_sha256"])
        scores[mode] = score_metrics(directory)
        if scores[mode] != comparison["scores"][mode]:
            raise ValueError("Saved counts disagree with completed comparison")
        for name in ("summary.json", "status.json", "clean_metrics.json", "conditions.json", "provenance.json",
                     "guard_completion.json", "temperature_guard_state.json"):
            fingerprint(directory / name)
        for path in sorted((directory / "conditions").glob("*.json")):
            fingerprint(path)
        evaluations[mode] = {"condition_count": summary["condition_count"], "input_hashes_verified": count,
                             "test_evaluated": False, "guard_status": state["status"], "guard_decision": state["decision"]}
    if (values["gates"] != comparison["thresholds"]
            or descriptive_gates(scores, values["gates"]) != comparison["gate"]):
        raise ValueError("Frozen screening gate differs from recalculation")
    for path in (study / "completion.json", study / "RESULT.md", resume / "completion.json", config, Path(__file__)):
        fingerprint(path)
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError(f"Input changed during completion audit: {name}")
    return {"status": "completed_matched_study_independently_verified",
            "audited_utc": datetime.now(timezone.utc).isoformat(), "new_model_inference": False,
            "training_started": False, "test_accessed": False, "validation_raw_files_hashed_only": True,
            "training_data_scope": "Captured full train/val audits and sealed checkpoint/protocol verified; no new raw train scan.",
            "completion_mirrors_match": True, "scores_recalculated_from_saved_counts": True,
            "gate_recalculated_matches": True, "initialization_and_actual_order_match": True,
            "checks": checks, "training": training, "evaluations": evaluations, "scores": scores,
            "gate": comparison["gate"], "source_hashes": hashes,
            "limitation": "One seed and one validation city; checkpoint/provenance hashes verified without new inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Completion audit already exists; preserve previous evidence")
    report = audit(args.study_config, args.resume_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit_sha256": sha256(target),
                      "unique_files_verified": len(report["source_hashes"]),
                      "gates_passed": report["gate"]["passed"], "training": report["training"]}))


if __name__ == "__main__":
    main()
