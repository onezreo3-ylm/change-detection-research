"""Explicit stratified-checkpoint adapter for immutable P1 val and train diagnostics."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from PIL import features
import torch
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import file_sha256, make_loader
from pcdc.engine import check_checkpoint, move_model, read_checkpoint, source_files
from pcdc import evaluate_candidate as candidate
from pcdc.evaluate_corruptions import (COUNT_KEYS, METRIC_KEYS, _bind_validation_files,
    _expected_clean, _verify_input_hashes, run_condition, summarize_conditions)
from pcdc.frozen_validation import FrozenInstanceDataset, build_validation_dataset
from pcdc.model import build_model
from pcdc.robust_baselines import _digest, _exposure_summary, robust_source_hashes
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts import valais_p1_study as parent
from scripts import resume_valais_p1_as as recovery
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.p1_process_state import process_snapshot
from scripts.resume_valais_p1_study import check_prefix_history
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.stratified_sampling import StratifiedBatchSampler
from scripts.train_stratified_robust import FLAVOR, sampling_identity, trainer_hashes, validate_resume

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("cs", "as")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    source = Path(path).resolve()
    value = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    keys = {"schema_version", "parent_study", "latest_recovery", "previous_as_recovery", "output_dir",
            "b0e_evaluation", "p0_manifest", "p0_review", "protocol_document", "minimum_unittests",
            "validation_max_seconds_per_model", "residual_max_seconds", "gates"}
    if set(value) != keys or value["schema_version"] != 1:
        raise ValueError("Unexpected P1 posttraining schema")
    paths = {key: (source.parent / value[key]).resolve() for key in
             ("parent_study", "latest_recovery", "previous_as_recovery", "output_dir", "b0e_evaluation",
              "p0_manifest", "p0_review", "protocol_document")}
    paths["output_dir"].relative_to(ROOT / "runs")
    _, settings, study_paths, runs = parent.load(paths["parent_study"])
    if (value["validation_max_seconds_per_model"] != 21600 or value["residual_max_seconds"] != 21600
            or value["minimum_unittests"] < 352):
        raise ValueError("Changed finite evaluation budget")
    expected_gates = {
        "cs_vs_b0e": {"minimum_clean_f1_gain": .01, "minimum_clean_recall_gain": .02,
            "maximum_mf1_drop": .005, "maximum_corruption_recall_drop": .01,
            "maximum_clean_empty_fpr_increase": .0002, "maximum_corruption_empty_fpr_increase": .0002},
        "as_vs_cs": {"minimum_mf1_gain": .005, "maximum_clean_f1_drop": .005,
            "maximum_clean_recall_drop": .01, "maximum_corruption_recall_drop": .01,
            "maximum_clean_empty_fpr_increase": .0002, "maximum_corruption_empty_fpr_increase": .0002}}
    if value["gates"] != expected_gates:
        raise ValueError("P1 gates changed after training")
    return source, value, paths, settings, study_paths, runs


def adapter_sources():
    files = dict(source_files())
    for name in ("scripts/valais_p1_posttraining.py", "scripts/valais_p1_residual.py",
                 "scripts/stratified_sampling.py", "scripts/train_stratified_robust.py",
                 "scripts/evaluate_coupling_p0.py", "scripts/coupling_diagnostic.py", "scripts/audit_coupling_p0.py",
                 "scripts/evaluate_guarded.py", "scripts/temperature_guard.py", "scripts/valais_p1_study.py"):
        files[name] = ROOT / name
    return files


def validate_history(last, best, actual, planned, *, epochs=100):
    if last["epoch"] != epochs or len(planned) != epochs:
        raise ValueError("Incomplete final checkpoint or sampling plan")
    history = last["history"]
    check_prefix_history(history, actual, planned, epochs)
    selected = max(history, key=lambda row: row["val"]["metrics"]["f1"])
    if (best["epoch"] != selected["epoch"] or last["best_epoch"] != selected["epoch"]
            or best["best_epoch"] != selected["epoch"]
            or best["best_val_f1"] != selected["val"]["metrics"]["f1"]
            or last["best_val_f1"] != best["best_val_f1"]
            or best["history"] != history[:best["epoch"]]):
        raise ValueError("Not the first highest clean-val checkpoint")
    for cp in (best, last):
        if (set(cp["file_sha256"]) != {"train", "val"} or any("test" in row for row in cp["history"])
                or cp.get("training_flavor") != FLAVOR):
            raise ValueError("Wrong checkpoint flavor or development data scope")
        tensors = list(cp["model"].values()) + [v for state in cp["optimizer"]["state"].values()
                                                for v in state.values() if isinstance(v, torch.Tensor)]
        if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in tensors):
            raise FloatingPointError("Non-finite completed checkpoint")


def validate_row(role, cfg, settings, study_paths, *, bundle=None, planned=None):
    status = parent.check_completion(cfg, study_paths, settings["initial_model_tensor_sha256"])
    if (status["kind"] != "research" or status["mode"] != cfg.mode
            or status["selection"] != "clean_val_f1" or status["study_role"] != {"cs": "C-S", "as": "A-S"}[role]):
        raise ValueError("Completed status belongs to another training role")
    _, evidence, _, identity = sampling_identity(study_paths["sampling_proposal"])
    audit = read(evidence["captured_data_audit"])
    planned = planned if planned is not None else read(study_paths["output_dir"] / "sampling_plan.json")
    actual = read(cfg.output_dir / "sampling_actual.json")
    best, last = read_checkpoint(cfg.output_dir / "best.pt"), read_checkpoint(cfg.output_dir / "last.pt")
    torch.set_num_threads(cfg.base.values["runtime"]["num_threads"])
    validate_history(last, best, actual, planned)
    for cp in (best, last):
        validate_resume(cp, cfg, identity, robust_source_hashes(), trainer_hashes(),
                        _digest(audit), settings["initial_model_tensor_sha256"])
        check_checkpoint(cfg.base, cp)
        if (cp["robust_mode"] != cfg.mode or cp["kind"] != "research"
                or cp["config"] != cfg.base.values or cp["robust_config"] != dict(cfg.values)
                or cp["file_sha256"] != {split: row["file_sha256"] for split, row in audit["splits"].items()}
                or cp["manifests"] != {split: {k: v for k, v in row.items() if k != "file_sha256"}
                                       for split, row in audit["splits"].items()}
                or cp["corruption_protocol_sha256"] != cfg.protocol.fingerprint
                or cp["parent_robust_training_signature"] != cfg.fingerprint
                or cp["severity_review_sha256"] != file_sha256(cfg.paths["severity_review"])):
            raise ValueError("Stratified checkpoint changed source, configuration or data")
        if bundle is not None:
            check_checkpoint(bundle.protocol.base, cp)
            if (cp["corruption_protocol_sha256"] != bundle.protocol.fingerprint
                    or cp["file_sha256"]["val"] != bundle.manifest["file_sha256"]
                    or cp["manifests"]["val"] != bundle.manifest["inventory"]):
                raise ValueError("Stratified checkpoint differs from frozen validation benchmark")
    if (best["epoch"] != status["best_epoch"] or last["best_val_f1"] != status["best_val_f1"]
            or last["history"] != read(cfg.output_dir / "history.json")
            or _exposure_summary(last["history"]) != read(cfg.output_dir / "training_exposure_summary.json")):
        raise ValueError("Training status/history/exposure do not commit these checkpoints")
    files = [cfg.source, *cfg.paths.values()]
    files = [p for p in files if p.is_file()]
    for name in ("best.pt", "last.pt", "status.json", "guard_completion.json", "history.json",
                 "sampling_actual.json", "sampling_identity.json", "training_exposure_summary.json",
                 "source_sha256.json", "stratified_trainer_source_sha256.json"):
        files.append(cfg.output_dir / name)
    completion = read(cfg.output_dir / "guard_completion.json")
    files.append(cfg.output_dir / completion["guard_state_path"])
    return best, {"role": role, "status": "completed_training_audited", "epochs": 100,
        "optimizer_steps": status["optimizer_steps"], "pair_views": status["pair_views"],
        "best_epoch": best["epoch"], "clean_metrics": _expected_clean(best)["metrics"],
        "sampling_identity": identity, "bound_sha256": {str(p.resolve()): file_sha256(p) for p in files}}


def make_job(output, role, cfg, study_paths):
    baseline = study_paths["baseline_evaluation"]
    values = {"schema_version": 1, "name": cfg.values["name"] + "_stratified_frozen_val",
        "expected_kind": "research", "expected_mode": cfg.mode,
        "benchmark_config": str(study_paths["benchmark_config"]),
        "baseline_summary": str(baseline / "summary.json"), "baseline_conditions": str(baseline / "conditions.json"),
        "output_dir": str(output / (role + "_val"))}
    for key, name in (("candidate_best_checkpoint", "best.pt"), ("candidate_last_checkpoint", "last.pt"),
                      ("candidate_status", "status.json"), ("candidate_guard_completion", "guard_completion.json"),
                      ("candidate_exposure", "training_exposure_summary.json")):
        values[key] = str(cfg.output_dir / name)
    for key in ("baseline_summary", "baseline_conditions", "candidate_best_checkpoint", "candidate_last_checkpoint",
                "candidate_status", "candidate_guard_completion", "candidate_exposure"):
        values[key + "_sha256"] = file_sha256(Path(values[key]))
    path = output / (role + "_evaluation.yaml")
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, sort_keys=False)
    return path


def prepare(path):
    source, value, paths, settings, study_paths, runs = load(path)
    for key in ("latest_recovery", "previous_as_recovery"):
        _, _, rpaths, _, s, _, old = recovery.load(paths[key])
        recovery.verify(rpaths, s, old)
        owner, queue = read(rpaths["output_dir"] / "owner.json"), read(rpaths["output_dir"] / "status.json")
        if queue.get("child_pid"):
            worker_pid, worker_ticks = queue["child_pid"], queue["child_creation_ticks"]
        else:
            proof = read(rpaths["output_dir"] / "train_start_verified.json")
            worker = proof.get("process_states", proof)["worker"]
            worker_pid, worker_ticks = worker["pid"], worker["creation_ticks"]
        if (process_snapshot(owner["pid"], owner["creation_ticks"])["owned_active"]
                or process_snapshot(worker_pid, worker_ticks)["owned_active"]):
            raise RuntimeError("A training owner or worker is still active")
    _, evidence, positives, _ = sampling_identity(study_paths["sampling_proposal"])
    audit = read(evidence["captured_data_audit"])
    proposal, _, _, _ = sampling_identity(study_paths["sampling_proposal"])
    sampler = StratifiedBatchSampler(sorted(audit["splits"]["train"]["file_sha256"]), positives, proposal["sampling"])
    regenerated = []
    for epoch in range(1, 101):
        sampler.set_epoch(epoch)
        regenerated.append(sampler.plan().summary)
    if regenerated != read(study_paths["output_dir"] / "sampling_plan.json"):
        raise ValueError("Frozen plans differ from actual train identities and reviewed sampler")
    evidence_rows = {}
    for role in ROLES:
        _, evidence_rows[role] = validate_row(role, runs[role], settings, study_paths, planned=regenerated)
    if read(runs["cs"].output_dir / "sampling_actual.json") != read(runs["as"].output_dir / "sampling_actual.json"):
        raise ValueError("C-S/A-S actual source streams differ")
    from scripts import evaluate_coupling_p0 as p0
    manifest = read(paths["p0_manifest"])
    p0.validate_manifest(manifest)
    p0.validate_review(manifest, paths["p0_manifest"], paths["p0_review"])
    if manifest["settings"]["max_seconds"] != value["residual_max_seconds"]:
        raise ValueError("Residual budget differs from original P0")
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "regenerated_sampling_plan.json", regenerated)
    write_json(output / "training_completion_audit.json", {"status": "passed_completed_training_audit",
        "rows": evidence_rows, "matched_actual_source_stream_verified": True,
        "sampling_regenerated_from_frozen_train_identities": True, "test_accessed": False,
        "physical_compute_note": "Committed budgets match; discarded partial C-S95/A-S97 updates and engineering smoke are additional physical work."})
    for role in ROLES:
        job = candidate.load_candidate_evaluation_config(make_job(output, role, runs[role], study_paths))
        metadata = candidate._load_metadata(job, require_review=True)
        old_provenance = read(paths["b0e_evaluation"] / "provenance.json")
        if (old_provenance["benchmark_fingerprints"] != metadata.bundle.fingerprints
                or old_provenance["candidate_mode"] != "b0e"):
            raise ValueError("Original B0-E reference uses another frozen benchmark")
    audit_validation(paths["b0e_evaluation"], audit_destination=output / "b0e_saved_result_audit.json")
    return {"status": "posttraining_prepared", "matched_committed_updates_per_model": 275700}


def verify_seal(paths):
    auth = read(paths["output_dir"] / "authorization.json")
    for name, expected in auth["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen posttraining input changed: {name}")
    return auth


@torch.inference_mode()
def evaluate_row(path, role):
    _, value, paths, settings, study_paths, runs = load(path)
    verify_seal(paths)
    job = candidate.load_candidate_evaluation_config(paths["output_dir"] / (role + "_evaluation.yaml"))
    metadata = candidate._load_metadata(job, require_review=True)
    best, _ = validate_row(role, runs[role], settings, study_paths, bundle=metadata.bundle)
    expected_clean = _expected_clean(best)
    output = job.output_dir
    if output.exists():
        raise FileExistsError(output)
    dataset, _ = build_validation_dataset(metadata.bundle)
    if dataset.split != "val" or metadata.bundle.protocol.base.run_test_after_training:
        raise ValueError("Val-only adapter received another scope")
    _bind_validation_files(metadata.input_hashes, metadata.bundle, dataset, best)
    _verify_input_hashes(metadata.input_hashes)
    config = metadata.bundle.protocol.base
    device = setup_runtime(config)
    files = adapter_sources()
    hashes = {name: file_sha256(p) for name, p in files.items()}
    output.mkdir(exist_ok=False)
    candidate._snapshot(job, output, hashes, files)
    write_json(output / "provenance.json", {"training_flavor": FLAVOR, "candidate_role": role,
        "training_signature": best["training_signature"], "sampling_identity": best["sampling_identity"],
        "benchmark_fingerprints": metadata.bundle.fingerprints, "input_file_sha256": metadata.input_hashes,
        "adapter_source_sha256": hashes, "checkpoint_fields_modified": False, "test_evaluated": False})
    environment = environment_record(config, device)
    environment.update(jpeg_codec=features.version_codec("jpg"), libjpeg_turbo=features.version_feature("libjpeg_turbo"))
    write_json(output / "environment.json", environment)
    guard = EvaluationTemperatureGuard(load_guard_config(study_paths["guard_config"]), output)
    started = time.monotonic()
    def guarded():
        if time.monotonic() - started > value["validation_max_seconds_per_model"]:
            raise RuntimeError("Fixed validation budget exhausted; no automatic continuation")
        guard()
    try:
        model = move_model(build_model(config, initialize_encoder=False), device, config.values["runtime"]["memory_format"])
        model.load_state_dict(best["model"], strict=True)
        model.requires_grad_(False)
        _atomic_json(output / "status.json", {"status": "evaluating_clean", "role": role, "test_evaluated": False})
        clean = run_condition(model, make_loader(dataset, config, training=False), config, device, guard_callback=guarded)
        clean["matches_candidate_checkpoint"] = (clean["samples"] == expected_clean["samples"]
            and all(clean["counts"][k] == expected_clean["metrics"][k] for k in COUNT_KEYS)
            and all(clean["metrics"][k] == expected_clean["metrics"][k] for k in METRIC_KEYS))
        write_json(output / "clean_metrics.json", clean)
        if not clean["matches_candidate_checkpoint"]:
            raise ValueError("Clean val does not exactly reproduce selected checkpoint counts/metrics")
        grouped = {}
        for instance in metadata.bundle.instances:
            grouped.setdefault(instance["condition_id"], []).append(instance)
        ids = list(metadata.bundle.manifest["condition_counts"])
        if len(ids) != 45 or set(grouped) != set(ids):
            raise ValueError("Original 45 condition coverage changed")
        results = {}
        (output / "conditions").mkdir()
        for condition_id in ids:
            cell = FrozenInstanceDataset(dataset, grouped[condition_id])
            row = run_condition(model, make_loader(cell, config, training=False), config, device, guard_callback=guarded)
            strata = {k[len(condition_id)+1:]: n for k, n in metadata.bundle.manifest["stratum_counts"].items()
                      if k.startswith(condition_id + "|")}
            if (row["condition_id"] != condition_id or row["samples"] != metadata.bundle.manifest["condition_counts"][condition_id]
                    or row["pixels"] != row["samples"] * dataset.tile_size ** 2
                    or {k: v["samples"] for k, v in row["joint_strata"].items()} != strata):
                raise ValueError("Incomplete condition/source/strata coverage")
            row.update(clean_reference_f1=clean["metrics"]["f1"], clean_minus_f1=clean["metrics"]["f1"]-row["metrics"]["f1"])
            write_json(output / "conditions" / (condition_id + ".json"), row)
            results[condition_id] = row
            _atomic_json(output / "status.json", {"status": "evaluating_corruptions", "role": role,
                "completed_conditions": len(results), "condition_count": 45, "test_evaluated": False})
            print(f"{role}/{condition_id}: f1={row['metrics']['f1']:.6f} ({len(results)}/45)", flush=True)
        write_json(output / "conditions.json", results)
        by_condition, paired = candidate.compare_results(results, metadata.baseline_conditions, ids)
        write_json(output / "comparison_by_condition.json", by_condition)
        write_json(output / "paired_by_source.json", paired)
        summary = {"status": "completed", "kind": "research", "mode": "stratified_frozen_validation_v1",
            "candidate_mode": runs[role].mode, "candidate_role": role, "training_flavor": FLAVOR, "split": "val",
            "candidate_checkpoint_epoch": best["epoch"], "selection": "clean_validation_f1_only",
            "clean_f1": clean["metrics"]["f1"], **summarize_conditions(results, ids),
            "candidate_diagnostics": candidate._condition_diagnostics(results, ids),
            "test_evaluated": False, "training_started": False, "guarded_completion_required": True}
        _verify_input_hashes(metadata.input_hashes)
        verify_seal(paths)
        if hashes != {name: file_sha256(p) for name, p in adapter_sources().items()}:
            raise ValueError("Adapter source changed during scoring")
        summary["temperature_guard"] = guard.finish(summary)
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "status.json", summary)
        _atomic_json(output / "guard_completion.json", {"status": "completed", "summary_sha256": file_sha256(output / "summary.json"),
            "guard_state_sha256": file_sha256(output / "temperature_guard_state.json"), "adapter_source_sha256": hashes})
        audit_validation(output)
        return summary
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        _atomic_json(output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "test_evaluated": False})
        raise


def measure_reports(clean, conditions):
    """Reconstruct aggregate counts from per-source records, without model inference."""
    def audit_one(row):
        c = row["counts"]
        if any(type(c[k]) is not int or c[k] < 0 for k in COUNT_KEYS):
            raise ValueError("Invalid saved counts")
        pooled = {k: sum(s["counts"][k] for s in row["by_source"].values()) for k in COUNT_KEYS}
        if pooled != c or sum(c.values()) != row["pixels"]:
            raise ValueError("Per-source counts do not conserve overall pixels")
        tn, fp, fn, tp = (c[k] for k in COUNT_KEYS)
        divide = lambda a, b: a / b if b else 0.0
        calculated = {"precision": divide(tp, tp+fp), "recall": divide(tp, tp+fn),
            "f1": divide(2*tp, 2*tp+fp+fn), "iou": divide(tp, tp+fp+fn), "oa": divide(tp+tn, tp+tn+fp+fn)}
        if any(not math.isclose(row["metrics"][k], n, abs_tol=1e-12, rel_tol=0) for k, n in calculated.items()):
            raise ValueError("Saved metrics differ from pooled counts")
        for source_id, source in row["by_source"].items():
            if not source_id.startswith("val/") or source["foreground_pixels"] != source["counts"]["tp"]+source["counts"]["fn"]:
                raise ValueError("Wrong val source identity or foreground denominator")
        return calculated
    clean_metrics = audit_one(clean)
    for row in conditions.values():
        audit_one(row)
        if (set(row["by_source"]) != set(clean["by_source"])
                or any(source["foreground_pixels"] != clean["by_source"][sid]["foreground_pixels"]
                       for sid, source in row["by_source"].items())):
            raise ValueError("Condition source coverage or target foreground differs from clean")
        for groups in [row.get("joint_strata", {}), *row.get("marginals", {}).values()]:
            if groups and {k: sum(g["counts"][k] for g in groups.values()) for k in COUNT_KEYS} != row["counts"]:
                raise ValueError("Stratum/marginal counts do not conserve condition pixels")
    def empty_fpr(reports):
        counts = {k: sum(s["counts"][k] for r in reports for s in r["by_source"].values()
                         if s["foreground_pixels"] == 0) for k in COUNT_KEYS}
        n = counts["tn"] + counts["fp"]
        return counts["fp"] / n if n else 0.0
    tp = sum(r["counts"]["tp"] for r in conditions.values())
    fn = sum(r["counts"]["fn"] for r in conditions.values())
    return {"clean_f1": clean_metrics["f1"], "clean_recall": clean_metrics["recall"],
        "mf1": sum(r["metrics"]["f1"] for r in conditions.values())/len(conditions),
        "corruption_recall": tp/(tp+fn) if tp+fn else 0.0,
        "clean_empty_fpr": empty_fpr([clean]), "corruption_empty_fpr": empty_fpr(conditions.values())}


def audit_validation(directory, *, audit_destination=None):
    directory = Path(directory)
    summary, completion = read(directory / "summary.json"), read(directory / "guard_completion.json")
    if (summary["status"] != "completed" or read(directory / "status.json")["status"] != "completed"
            or summary["test_evaluated"] is not False
            or summary["condition_count"] != 45 or completion["status"] != "completed"
            or completion["summary_sha256"] != file_sha256(directory / "summary.json")
            or completion["guard_state_sha256"] != file_sha256(directory / "temperature_guard_state.json")
            or read(directory / "temperature_guard_state.json")["status"] != "completed"):
        raise ValueError("Incomplete val/guard commitment")
    conditions = read(directory / "conditions.json")
    if len(conditions) != 45 or {p.stem for p in (directory / "conditions").glob("*.json")} != set(conditions):
        raise ValueError("Condition file coverage incomplete")
    for key, row in conditions.items():
        if read(directory / "conditions" / (key + ".json")) != row or row["condition_id"] != key:
            raise ValueError("Condition files disagree")
    measurement = measure_reports(read(directory / "clean_metrics.json"), conditions)
    if abs(measurement["mf1"] - summary["corruption_mf1"]) > 1e-12:
        raise ValueError("Saved macro F1 differs from independent aggregation")
    result = {"status": "passed_saved_counts_and_guard_audit", "metrics": measurement,
              "summary_sha256": file_sha256(directory / "summary.json"), "model_inference_performed": False}
    audit_path = Path(audit_destination) if audit_destination is not None else directory / "saved_result_audit.json"
    if not audit_path.exists():
        write_json(audit_path, result)
    return measurement


def gate_comparison(scores, gates):
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in scores.values() for v in row.values()):
        raise ValueError("Nonfinite or out-of-range comparison metrics")
    delta = {"cs_vs_b0e": {k: scores["cs"][k]-scores["b0e"][k] for k in scores["cs"]},
             "as_vs_cs": {k: scores["as"][k]-scores["cs"][k] for k in scores["as"]}}
    checks = {}
    for name, d in delta.items():
        g = gates[name]
        checks[name] = {
            "clean_f1": d["clean_f1"] >= g.get("minimum_clean_f1_gain", -g.get("maximum_clean_f1_drop", 0))-1e-12,
            "clean_recall": d["clean_recall"] >= g.get("minimum_clean_recall_gain", -g.get("maximum_clean_recall_drop", 0))-1e-12,
            "mf1": d["mf1"] >= g.get("minimum_mf1_gain", -g.get("maximum_mf1_drop", 0))-1e-12,
            "corruption_recall": d["corruption_recall"] >= -g["maximum_corruption_recall_drop"]-1e-12,
            "clean_empty_fpr": d["clean_empty_fpr"] <= g["maximum_clean_empty_fpr_increase"]+1e-12,
            "corruption_empty_fpr": d["corruption_empty_fpr"] <= g["maximum_corruption_empty_fpr_increase"]+1e-12}
    return {"deltas": delta, "checks": checks, "passed": {name: all(v.values()) for name, v in checks.items()},
            "single_seed_development_only": True, "p2_training_authorized": False}


def seal(path):
    source, value, paths, _, study_paths, _ = load(path)
    output = paths["output_dir"]
    log = (output / "unittest.log").read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", log)
    if (not count or int(count[1]) < value["minimum_unittests"] or not re.search(r"(?m)^OK\s*$", log)
            or "FAILED" in log or "skipped=" in log):
        raise ValueError("Complete tests did not pass")
    smoke_report = read(output / "cuda_smoke/report.json")
    if smoke_report["status"] != "passed_train_only_p1_cuda_smoke":
        raise ValueError("Missing passing real CUDA smoke")
    files = [source, paths["protocol_document"], paths["parent_study"], paths["p0_manifest"], paths["p0_review"],
        output / "training_completion_audit.json", output / "regenerated_sampling_plan.json", output / "unittest.log",
        output / "cuda_smoke/report.json", ROOT / "tests/test_p1_posttraining.py", *adapter_sources().values()]
    bindings = {}
    for row in read(output / "training_completion_audit.json")["rows"].values():
        bindings.update(row["bound_sha256"])
    for role in ROLES:
        job_path = output / (role + "_evaluation.yaml")
        files.append(job_path)
        job = candidate.load_candidate_evaluation_config(job_path)
        metadata = candidate._load_metadata(job, require_review=True)
        bindings.update(metadata.input_hashes)
    files.extend(p for p in paths["b0e_evaluation"].rglob("*.json") if p.parent == paths["b0e_evaluation"]
                 or p.parent == paths["b0e_evaluation"] / "conditions")
    files.extend(study_paths[key] for key in ("sampling_proposal", "guard_config", "protocol_document"))
    bindings.update({str(p.resolve()): file_sha256(p) for p in files})
    write_json(output / "authorization.json", {"status": "authorized_fixed_p1_posttraining_evaluation",
        "authorized_utc": now(), "human_instruction": "恢复并继续", "test_accessed": False,
        "training_authorized": False, "bound_sha256": bindings})
    return {"status": "fixed_posttraining_evaluation_sealed", "tests": int(count[1]), "files": len(bindings)}


def run(path):
    source, value, paths, _, study_paths, _ = load(path)
    verify_seal(paths)
    output = paths["output_dir"]
    write_json(output / "owner.json", {"pid": os.getpid(), "creation_ticks": creation_time_ticks(os.getpid()),
        "started_utc": now(), "config": str(source), "automatic_retry": False})
    role, child = "cs", None
    try:
        for role in (*ROLES, "residual"):
            verify_seal(paths)
            command = [str(study_paths["python"]), "-u", "-m", "scripts.valais_p1_posttraining",
                       "--config", str(source), "action", "--stage", role]
            with (output / (role + ".stdout.log")).open("x", encoding="utf-8") as stdout, \
                    (output / (role + ".stderr.log")).open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                _atomic_json(output / "status.json", {"status": "running", "stage": role,
                    "child_pid": child.pid, "child_creation_ticks": creation_time_ticks(child.pid),
                    "updated_utc": now(), "test_accessed": False})
                code = child.wait()
            if code:
                raise RuntimeError(f"{role} exited {code}; preserve failure, no automatic retry")
        from scripts.valais_p1_residual import audit_saved
        audit_saved(paths)
        scores = {role: audit_validation(output / (role + "_val")) for role in ROLES}
        scores["b0e"] = audit_validation(paths["b0e_evaluation"], audit_destination=output / "b0e_saved_result_audit.json")
        comparison = {"status": "completed_p1_development_comparison", "scores": scores,
            "gates": gate_comparison(scores, value["gates"]), "test_accessed": False,
            "residual_gate": read(output / "residual/report.json")["gate"],
            "physical_compute_note": read(output / "training_completion_audit.json")["physical_compute_note"]}
        write_json(output / "comparison.json", comparison)
        _atomic_json(output / "status.json", {"status": "completed_p1_evaluation_ready_for_review", "completed_utc": now(),
            "test_accessed": False, "p2_training_authorized": False})
    except BaseException as error:
        _atomic_json(output / "status.json", {"status": "failed", "stage": role,
            "error": f"{type(error).__name__}: {error}", "test_accessed": False, "automatic_retry": False})
        raise


def monitor(path):
    _, _, paths, _, study_paths, _ = load(path)
    output = paths["output_dir"]
    result = {"checked_utc": now(), "test_accessed": False}
    queue = read(output / "status.json") if (output / "status.json").exists() else {}
    result["queue"] = queue
    owner = read(output / "owner.json") if (output / "owner.json").exists() else {}
    result["owner_alive"] = bool(owner) and process_snapshot(owner["pid"], owner["creation_ticks"])["owned_active"]
    result["worker_alive"] = bool(queue.get("child_pid")) and process_snapshot(queue["child_pid"], queue["child_creation_ticks"])["owned_active"]
    for stage in (*ROLES, "residual"):
        folder = output / (stage + "_val") if stage in ROLES else output / "residual"
        for name in ("status.json", "progress.json"):
            if (folder / name).is_file():
                row = read(folder / name)
                result[stage] = {k: row[k] for k in ("status", "completed_conditions", "condition_count", "model", "sample", "records", "error") if k in row}
                break
        log = output / (stage + ".stdout.log")
        if log.exists():
            result.setdefault(stage, {})["log_age_seconds"] = round(time.time()-log.stat().st_mtime, 1)
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size-1200))
                result[stage]["log_tail"] = stream.read().decode("utf-8", errors="replace").splitlines()[-2:]
    if queue.get("status") == "running" and (not result["owner_alive"] or not result["worker_alive"]):
        result["attention"] = "Posttraining process ended without completion; preserve before any recovery."
    guard = load_guard_config(study_paths["guard_config"])
    telemetry = subprocess.run([guard.values["nvidia_smi"], "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().split(",")
    if telemetry[0].strip() != guard.values["gpu_uuid"]:
        raise ValueError("Monitored GPU identity changed")
    result["gpu"] = dict(zip(("temperature_c", "utilization_percent", "memory_used_mib"), map(int, telemetry[1:])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "seal", "run", "action", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(*ROLES, "residual"))
    args = parser.parse_args()
    if args.command in ("smoke", "action"):
        from scripts import valais_p1_residual as residual
        result = residual.run(args.config, smoke=True) if args.command == "smoke" else (
            residual.run(args.config) if args.stage == "residual" else evaluate_row(args.config, args.stage))
    else:
        result = globals()[args.command](args.config)
    if result is not None:
        print(json.dumps({k: result[k] for k in ("status", "tests", "files", "matched_committed_updates_per_model") if k in result}
            if args.command != "monitor" else result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
