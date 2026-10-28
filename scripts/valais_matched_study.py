"""One-shot, sequential FP32 Valais B1/B0-E study using existing trainers."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import LEVIRDataset, audit_data, file_sha256
from pcdc.engine import check_checkpoint, move_model, read_checkpoint, run_epoch, source_hashes
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import (
    TwoViewTrainingDataset, load_robust_config, preflight, robust_source_files,
    run_two_view_epoch, tensor_state_sha256,
)
from pcdc.runtime import setup_runtime, write_json
from scripts.analyze_valais_baseline import analyze, metrics, read_json, save_analysis
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config, read_temperature

ROOT = Path(__file__).resolve().parents[1]


def now(): return datetime.now(timezone.utc).isoformat()


def resolve(source, value):
    p = Path(value)
    return (p if p.is_absolute() else source.parent / p).resolve()


def load_study(path):
    source = Path(path).resolve()
    v = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    keys = {"schema_version", "name", "output_dir", "python", "task_name", "launcher",
            "protocol_document", "baseline_evaluation", "benchmark_config", "label_balance_report",
            "visual_review", "visual_review_sha256", "guard_config", "minimum_unittests", "order",
            "runs", "diagnostic", "gates"}
    if not isinstance(v, dict) or set(v) != keys or v["schema_version"] != 1:
        raise ValueError("Unexpected study schema")
    if v["order"] != ["b1", "b0e"] or set(v["runs"]) != {"b1", "b0e"}:
        raise ValueError("Only the frozen B1 then B0-E pair is supported")
    paths = {key: resolve(source, v[key]) for key in keys if key in
             {"output_dir", "python", "launcher", "protocol_document", "baseline_evaluation",
              "benchmark_config", "label_balance_report", "visual_review", "guard_config"}}
    paths["output_dir"].relative_to(ROOT / "runs")
    runs = {mode: load_robust_config(resolve(source, v["runs"][mode])) for mode in v["order"]}
    a, b = runs["b1"], runs["b0e"]
    if (a.base.values != b.base.values or a.paths["base_config"] != b.paths["base_config"]
            or a.paths["protocol_config"] != b.paths["protocol_config"]
            or a.paths["severity_review"] != b.paths["severity_review"]
            or a.protocol.fingerprint != b.protocol.fingerprint):
        raise ValueError("Matched configurations do not share the same frozen base/protocol/review")
    for mode, run in runs.items():
        base = run.base.values
        if (run.mode != mode or base["experiment"]["kind"] != "research"
                or base["runtime"]["device"] != "cuda" or base["train"]["amp"] is not False
                or base["experiment"]["seed"] != 42 or base["train"]["epochs"] != 100
                or base["train"]["batch_size"] != 4 or base["train"]["eval_batch_size"] != 32
                or run.base.run_test_after_training or base["model"]["encoder_weights"] is not None
                or base["evaluation"]["threshold"] != 0.5
                or run.values["expected_source_tiles"] != 11025
                or run.values["expected_batches_per_epoch"] != 2757):
            raise ValueError("The Valais frozen FP32 seed-42 protocol differs")
        run.output_dir.relative_to(ROOT / "runs")
    if a.output_dir == b.output_dir: raise ValueError("Matched output directories alias")
    if file_sha256(paths["visual_review"]) != v["visual_review_sha256"]:
        raise ValueError("Accepted train severity visual evidence changed")
    expected_gates = {"minimum_mf1_gain_over_b0e", "minimum_mf1_gain_over_b0",
                      "maximum_clean_f1_drop", "maximum_clean_recall_drop",
                      "maximum_corruption_recall_drop", "maximum_empty_tile_fpr_increase",
                      "independent_seeds_required_before_mechanism"}
    if set(v["gates"]) != expected_gates or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                                               for x in v["gates"].values()):
        raise ValueError("Invalid preregistered gates")
    return source, v, paths, runs


def select_sources(balance, names, settings):
    if set(settings) != {"selection_seed", "small_positive_tiles", "large_positive_tiles", "empty_tiles", "smoke_source_batch_size"}:
        raise ValueError("Unexpected train diagnostic selection schema")
    positives = {row["name"]: row["positive_pixels"] for row in balance["train"]["positive_manifest"]}
    if not set(positives) < set(names): raise ValueError("Train positive manifest differs")
    median = balance["train"]["nonempty_positive_pixels_p50"]
    groups = {"small_positive": [n for n, count in positives.items() if count <= median],
              "large_positive": [n for n, count in positives.items() if count > median],
              "empty": list(set(names) - set(positives))}
    result = []
    for group, candidates in groups.items():
        def key(name): return hashlib.sha256(f"{settings['selection_seed']}\0{group}\0{name}".encode()).hexdigest(), name
        ordered = sorted(candidates, key=key)
        count = settings[group + "_tiles"]
        if type(count) is not int or count < 1 or count > len(ordered):
            raise ValueError("Invalid fixed train QA sample count")
        result.extend({"name": name, "stratum": group, "positive_pixels": positives.get(name, 0)}
                      for name in ordered[:count])
    return result


def _check_train_files(record, checkpoint):
    expected = checkpoint["file_sha256"]["train"][record.name]
    for key in ("image_a", "image_b", "mask"):
        if file_sha256(getattr(record, key)) != expected[key]: raise ValueError("Frozen train file changed")


def train_diagnostic(paths, runs, settings):
    cfg = runs["b1"].base
    reference = runs["b1"].protocol.path("reference_checkpoint")
    if file_sha256(reference) != runs["b1"].protocol.values["reference_checkpoint_sha256"]:
        raise ValueError("Selected clean baseline changed")
    checkpoint = read_checkpoint(reference)
    check_checkpoint(cfg, checkpoint)
    if checkpoint["source_sha256"] != source_hashes(): raise ValueError("Baseline implementation changed")
    balance = read_json(paths["label_balance_report"])
    if balance.get("test_accessed") is not False: raise ValueError("Unexpected label analysis scope")
    dataset = LEVIRDataset(cfg, "train")
    selected = select_sources(balance, [r.name for r in dataset.records], settings)
    lookup = {r.name: i for i, r in enumerate(dataset.records)}
    output = paths["output_dir"] / "train_diagnostic"
    output.mkdir(exist_ok=False)
    guard = EvaluationTemperatureGuard(load_guard_config(paths["guard_config"]), output)
    try:
        guard(force=True)
        device = setup_runtime(cfg)
        model = move_model(build_model(cfg, initialize_encoder=False), device, "channels_last")
        model.load_state_dict(checkpoint["model"], strict=True); model.eval()
        for item in selected:
            index = lookup[item["name"]]; record = dataset.records[index]
            _check_train_files(record, checkpoint)
            pair, sample = dataset.read_pair(record), dataset[index]
            if int(pair.mask.sum()) != item["positive_pixels"]: raise ValueError("Train label manifest mismatch")
            guard()
            with torch.no_grad():
                logits = model(sample["image_a"].unsqueeze(0).to(device=device, memory_format=torch.channels_last),
                               sample["image_b"].unsqueeze(0).to(device=device, memory_format=torch.channels_last))["logits"]
                if not torch.isfinite(logits).all().item(): raise FloatingPointError("Diagnostic logits are non-finite")
                probability = logits.sigmoid()[0, 0].cpu().numpy()
            prediction = probability >= cfg.values["evaluation"]["threshold"]
            label = pair.mask.astype(bool)
            counts = {"tp": int((prediction & label).sum()), "fp": int((prediction & ~label).sum()),
                      "fn": int((~prediction & label).sum()), "tn": int((~prediction & ~label).sum())}
            error = pair.image_b.copy()
            error[prediction & label] = [0, 230, 90]
            error[prediction & ~label] = [255, 40, 40]
            error[~prediction & label] = [40, 110, 255]
            sheet = Image.new("RGB", (1024, 290), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 4), f"Sion train/{record.name} | {item['stratum']} | TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}", fill="black")
            for col, (title, pixels) in enumerate(zip(["A 2017", "B 2023", "Label", "TP green / FP red / FN blue"],
                                                    [pair.image_a, pair.image_b, np.repeat((pair.mask*255)[..., None], 3, axis=2), error])):
                draw.text((256*col + 4, 18), title, fill="black")
                sheet.paste(Image.fromarray(pixels), (256*col, 34))
            png = output / (Path(record.name).stem + ".png")
            sheet.save(png)
            np.save(output / (Path(record.name).stem + ".probability.npy"), probability, allow_pickle=False)
            _check_train_files(record, checkpoint)
            item.update(counts=counts, metrics=metrics(counts), image=str(png.resolve()),
                        image_sha256=file_sha256(png), image_bytes=png.stat().st_size,
                        raw_pixels_per_panel=256, source_sha256=checkpoint["file_sha256"]["train"][record.name])
        report = {"status": "fixed_train_diagnostic_complete", "test_accessed": False,
                  "val_images_accessed": False, "checkpoint_sha256": file_sha256(reference),
                  "selection_uses_train_labels": True, "selection_uses_model_scores": False,
                  "small_large_boundary_train_positive_median": balance["train"]["nonempty_positive_pixels_p50"],
                  "samples": selected, "total_image_bytes": sum(x["image_bytes"] for x in selected),
                  "limitation": "Eight fixed train examples, single-tile inference; no dataset-wide label-quality estimate."}
        write_json(output / "report.json", report)
        guard.finish({"status": report["status"]})
        write_json(output / "completion.json", {"status": "guarded_train_diagnostic_complete",
                   "report_sha256": file_sha256(output / "report.json"),
                   "guard_state_sha256": file_sha256(output / "temperature_guard_state.json"), "test_accessed": False})
        return report
    except BaseException as error:
        if guard.attached: guard.abort(error)
        raise


def prepare(path):
    source, v, paths, runs = load_study(path)
    output = paths["output_dir"]
    if output.exists() or any(r.output_dir.exists() for r in runs.values()): raise FileExistsError("Study/run output already exists")
    output.mkdir(parents=True, exist_ok=False)
    instances = runs["b1"].protocol.path("output_dir") / "val_instances.jsonl"
    save_analysis(analyze(paths["baseline_evaluation"], instances), output / "baseline_analysis")
    qa = train_diagnostic(paths, runs, v["diagnostic"])
    write_json(output / "preflight.json", {"status": "matched_preparation_complete",
               "robust": {mode: preflight(r) for mode, r in runs.items()},
               "diagnostic_report_sha256": file_sha256(output / "train_diagnostic/report.json"),
               "test_accessed": False, "training_started": False})
    return {"status": "matched_preparation_complete", "qa_images": len(qa["samples"]), "image_bytes": qa["total_image_bytes"]}


def smoke(path):
    source, v, paths, runs = load_study(path)
    qa = read_json(paths["output_dir"] / "train_diagnostic/report.json")
    chosen = [x["name"] for x in qa["samples"] if x["stratum"] == "small_positive"][:1]
    chosen += [x["name"] for x in qa["samples"] if x["stratum"] == "large_positive"][:1]
    chosen += [x["name"] for x in qa["samples"] if x["stratum"] == "empty"][:2]
    count = v["diagnostic"]["smoke_source_batch_size"]
    if len(chosen) != count or count != runs["b1"].base.values["train"]["batch_size"]:
        raise ValueError("Smoke must use the exact source batch size and both label strata")
    output = paths["output_dir"] / "smoke"; output.mkdir(exist_ok=False)
    reports = {}
    for mode, cfg in runs.items():
        stage = output / mode; stage.mkdir(exist_ok=False)
        guard = EvaluationTemperatureGuard(load_guard_config(paths["guard_config"]), stage)
        try:
            guard(force=True); device = setup_runtime(cfg.base)
            dataset = LEVIRDataset(cfg.base, "train")
            indices = [next(i for i, r in enumerate(dataset.records) if r.name == name) for name in chosen]
            paired = TwoViewTrainingDataset(dataset, cfg.protocol, cfg.secondary); paired.set_epoch(1)
            loader = DataLoader(Subset(paired, indices), batch_size=count, shuffle=False, num_workers=0)
            model = move_model(build_model(cfg.base), device, "channels_last")
            initial = tensor_state_sha256(model.state_dict())
            before = {name: int(m.num_batches_tracked) for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
            torch.cuda.reset_peak_memory_stats(device)
            t = cfg.base.values["train"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], betas=tuple(t["adam_betas"]), weight_decay=t["weight_decay"])
            result = run_two_view_epoch(model, loader, ChangeDetectionLoss(**cfg.base.values["loss"]), device,
                        threshold=cfg.base.values["evaluation"]["threshold"], consistency_kind="none", consistency_weight=0.0,
                        amp=False, memory_format="channels_last", optimizer=optimizer,
                        scaler=torch.cuda.amp.GradScaler(enabled=False), grad_clip_norm=t["grad_clip_norm"], guard_callback=guard)
            bn_ok = all(int(m.num_batches_tracked) - before[name] == 1 for name, m in model.named_modules() if name in before)
            if result["optimizer_steps"] != 1 or result["amp_skipped_steps"] != 0 or result["pair_views"] != 2*count or not bn_ok:
                raise ValueError("Matched FP32 smoke update/BN/exposure differs")
            vd = LEVIRDataset(cfg.base, "val")
            vl = DataLoader(Subset(vd, range(count)), batch_size=count, num_workers=0)
            guard(force=True)
            vr = run_epoch(model, vl, ChangeDetectionLoss(**cfg.base.values["loss"]), device, threshold=0.5,
                           amp=False, memory_format="channels_last")
            if not all(math.isfinite(x) for x in vr["loss"].values()): raise FloatingPointError("FP32 validation smoke failed")
            reports[mode] = {"status": "real_fp32_two_view_smoke_passed", "research_result": False,
                             "source_ids": chosen, "source_batch_size": count, "pair_views": 2*count,
                             "initial_tensor_sha256": initial, "one_bn_update": bn_ok,
                             "successful_updates": result["optimizer_steps"], "amp_skips": result["amp_skipped_steps"],
                             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device), "test_accessed": False}
            guard.finish({"status": reports[mode]["status"]}); write_json(stage / "report.json", reports[mode])
            del model, optimizer, paired, loader
            torch.cuda.empty_cache()
        except BaseException as error:
            if guard.attached: guard.abort(error)
            raise
    if reports["b1"]["initial_tensor_sha256"] != reports["b0e"]["initial_tensor_sha256"]:
        raise ValueError("Matched initialization smoke failed")
    write_json(output / "completion.json", {"status": "matched_real_cuda_smoke_passed", "training_amp": False,
               "validation_amp": False, "research_result": False, "test_accessed": False, "runs": reports})
    return {"status": "matched_real_cuda_smoke_passed", "runs": list(reports), "test_accessed": False}


def bound_check(bindings):
    for name, sha in bindings.items():
        if file_sha256(Path(name)) != sha: raise ValueError(f"Sealed input/source changed: {name}")


def no_other_python():
    # A graphics process reported under WDDM is not a competing research run.
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
               f"-and $_.ProcessId -ne {os.getpid()} "
               "-and $_.CommandLine -match '-m\\s+(pcdc|scripts\\.)' } "
               "| Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile",
                             "-NonInteractive", "-Command", command], capture_output=True, text=True,
                            check=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.stdout.strip() not in ("", "null"):
        raise RuntimeError("Another Python research task is active; refusing competing CUDA work")


def authorize(path):
    source, v, paths, runs = load_study(path)
    out = paths["output_dir"]; target = out / "authorization.json"
    if target.exists() or any(r.output_dir.exists() for r in runs.values()): raise FileExistsError("Already authorized/started")
    if Path(sys.executable).resolve() != paths["python"]: raise ValueError("Configured interpreter differs")
    no_other_python()
    log = (out / "unittest.log").read_text(encoding="utf-8-sig")
    counts = re.findall(r"Ran (\d+) tests in", log)
    if not counts or int(counts[-1]) < v["minimum_unittests"] or not log.rstrip().endswith("OK"):
        raise ValueError("Full unittest gate did not pass")
    sm = read_json(out / "smoke/completion.json")
    if sm["status"] != "matched_real_cuda_smoke_passed" or sm["test_accessed"] is not False:
        raise ValueError("Matched real CUDA smoke gate did not pass")
    review = read_json(out / "diagnostic_review.json")
    if review["status"] != "reviewed_no_protocol_change" or review["report_sha256"] != file_sha256(out / "train_diagnostic/report.json"):
        raise ValueError("Fixed original-resolution train diagnostic has not been reviewed")
    audit = audit_data(runs["b1"].base, full=True)
    if audit != read_json(runs["b1"].protocol.path("reference_audit")): raise ValueError("Live train/val data differs from B0")
    write_json(out / "live_data_audit.json", audit)
    guard = load_guard_config(paths["guard_config"])
    temperature = read_temperature(guard.values)
    if temperature >= guard.values["confirmed_pause_c"]: raise ValueError("GPU too warm to authorize")
    files = [source, paths["python"], paths["launcher"], paths["protocol_document"], paths["benchmark_config"],
             paths["label_balance_report"], paths["visual_review"], paths["guard_config"],
             Path(__file__), ROOT / "scripts/analyze_valais_baseline.py", ROOT / "scripts/train_robust_guarded.py",
             ROOT / "scripts/evaluate_candidate_guarded.py", out / "unittest.log", out / "smoke/completion.json",
             out / "preflight.json", out / "diagnostic_review.json", out / "train_diagnostic/report.json",
             out / "train_diagnostic/completion.json", out / "live_data_audit.json", out / "baseline_analysis/analysis.json",
             *robust_source_files().values(), *guard.source_files.values()]
    for cfg in runs.values(): files.extend([cfg.source, *cfg.paths.values()])
    files = [p for p in files if p.is_file()]
    files.extend(runs["b1"].protocol.path("output_dir") / n for n in ["val_manifest.json", "val_instances.jsonl"])
    files.append(runs["b1"].protocol.path("reference_checkpoint"))
    files.extend(Path(item["image"]) for item in read_json(out / "train_diagnostic/report.json")["samples"])
    files.extend(out / "smoke" / mode / name for mode in runs
                 for name in ["report.json", "temperature_guard_state.json"])
    files.extend(paths["baseline_evaluation"] / n for n in ["summary.json", "conditions.json", "clean_metrics.json", "guard_completion.json", "temperature_guard_state.json"])
    report = {"status": "matched_development_study_authorized", "authorized_utc": now(),
              "human_instruction": "Start execution of the proposed Valais B1/B0-E matched study in this chat.",
              "order": v["order"], "task_name": v["task_name"], "automatic_resume": False,
              "automatic_retry": False, "test_accessed": False, "training_amp": False, "validation_amp": False,
              "gpu_uuid": guard.values["gpu_uuid"], "gpu_temperature_c": temperature,
              "gates": v["gates"], "bound_sha256": {str(p.resolve()): file_sha256(p) for p in files}}
    write_json(target, report)
    return {"status": report["status"], "authorization_sha256": file_sha256(target), "test_accessed": False}


def check_training(run, source_order):
    directory = run.output_dir
    s, g = read_json(directory / "status.json"), read_json(directory / "guard_completion.json")
    h = read_json(directory / "history.json")
    state = read_json(directory / g["guard_state_path"])
    if (s["status"] != "completed" or s["test_evaluated"] is not False or s["mode"] != run.mode
            or s["epochs"] != 100 or s["optimizer_steps"] != 275700 or s["amp_skipped_steps"] != 0
            or s["pair_views"] != 2205000 or len(h) != 100
            or [x["epoch"] for x in h] != list(range(1,101))
            or g["training_status_sha256"] != file_sha256(directory / "status.json")
            or g["guard_state_sha256"] != file_sha256(directory / g["guard_state_path"])
            or state["status"] != "completed" or state["decision"] != "ok"
            or source_order["source_pairs"] != 1102500
            or source_order["batches"] != 275700):
        raise ValueError("Training did not complete with the frozen matched budget and guard")
    return s


def train_stage(path, mode):
    source, v, paths, runs = load_study(path)
    authorization = read_json(paths["output_dir"] / "authorization.json")
    bound_check(authorization["bound_sha256"])
    if mode not in runs: raise ValueError("Unexpected training stage")
    cfg = runs[mode]
    if cfg.output_dir.exists(): raise FileExistsError(cfg.output_dir)
    import pcdc.robust_baselines as robust
    from scripts.train_robust_guarded import guarded_train
    original = robust._update_stream_hash
    digest, count, batches = hashlib.sha256(), 0, 0
    def record_order(stream_digest, sample_ids, instances):
        nonlocal count, batches
        original(stream_digest, sample_ids, instances)
        for name in sample_ids:
            data = name.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big")); digest.update(data)
        count += len(sample_ids); batches += 1
    # Instrument only actual source order; the archived trainer/loss/BN math stays identical.
    robust._update_stream_hash = record_order
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        result = guarded_train(cfg.source, paths["guard_config"])
        record = {"source_order_sha256": digest.hexdigest(), "source_pairs": count, "batches": batches,
                  "initial_tensor_sha256": result["initial_model_tensor_sha256"],
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "wall_seconds": time.monotonic() - started,
                  "instrumentation_source_sha256": file_sha256(Path(__file__)), "test_accessed": False}
        check_training(cfg, record)
        write_json(cfg.output_dir / "source_order.json", record)
        print(json.dumps({"status": "matched_training_stage_complete", "mode": mode, "test_accessed": False}))
    finally:
        robust._update_stream_hash = original


def evaluation_job(paths, cfg):
    baseline, directory = paths["baseline_evaluation"], cfg.output_dir
    values = {"schema_version": 1, "name": cfg.values["name"] + "_frozen_val",
              "expected_kind": "research", "expected_mode": cfg.mode,
              "benchmark_config": str(paths["benchmark_config"]),
              "baseline_summary": str(baseline / "summary.json"),
              "baseline_conditions": str(baseline / "conditions.json"),
              "candidate_best_checkpoint": str(directory / "best.pt"),
              "candidate_last_checkpoint": str(directory / "last.pt"),
              "candidate_status": str(directory / "status.json"),
              "candidate_guard_completion": str(directory / "guard_completion.json"),
              "candidate_exposure": str(directory / "training_exposure_summary.json"),
              "output_dir": str(directory.with_name(directory.name + "_val"))}
    for key in ["baseline_summary", "baseline_conditions", "candidate_best_checkpoint", "candidate_last_checkpoint",
                "candidate_status", "candidate_guard_completion", "candidate_exposure"]:
        values[key + "_sha256"] = file_sha256(Path(values[key]))
    job = paths["output_dir"] / (cfg.mode + "_evaluation.yaml")
    with job.open("x", encoding="utf-8") as stream: yaml.safe_dump(values, stream, sort_keys=False)
    return job, Path(values["output_dir"])


def verify_score(directory):
    summary = read_json(directory / "summary.json")
    guard = read_json(directory / "guard_completion.json")
    state = read_json(directory / "temperature_guard_state.json")
    if (summary["status"] != "completed" or summary["test_evaluated"] is not False
            or summary["condition_count"] != 45 or guard["status"] != "completed"
            or state["status"] != "completed" or guard["summary_sha256"] != file_sha256(directory / "summary.json")
            or guard["guard_state_sha256"] != file_sha256(directory / "temperature_guard_state.json")):
        raise ValueError("Candidate scoring did not complete with 45 conditions and matching guard hashes")
    return summary


def score_metrics(directory):
    summary = verify_score(directory)
    clean = read_json(directory / "clean_metrics.json")
    all_counts = {key: 0 for key in ("tn", "fp", "fn", "tp")}
    empty_counts = dict(all_counts)
    for path in sorted((directory / "conditions").glob("*.json")):
        row = read_json(path)
        for key in all_counts: all_counts[key] += row["counts"][key]
        for source in row["by_source"].values():
            if source["foreground_pixels"] == 0:
                for key in empty_counts: empty_counts[key] += source["counts"][key]
    return {"mf1": summary["corruption_mf1"], "clean_f1": clean["metrics"]["f1"],
            "clean_recall": clean["metrics"]["recall"], "corruption_recall": metrics(all_counts)["recall"],
            "empty_tile_fpr": metrics(empty_counts)["background_fpr"]}


def descriptive_gates(scores, thresholds):
    b1 = scores["b1"]
    checks, differences = {}, {}
    for reference in ("b0", "b0e"):
        differences[reference] = {key: b1[key] - scores[reference][key] for key in b1}
        delta = differences[reference]
        checks[reference] = {
            "mf1_gain": delta["mf1"] >= thresholds["minimum_mf1_gain_over_" + reference] - 1e-12,
            "clean_f1_protected": delta["clean_f1"] >= -thresholds["maximum_clean_f1_drop"] - 1e-12,
            "clean_recall_protected": delta["clean_recall"] >= -thresholds["maximum_clean_recall_drop"] - 1e-12,
            "corruption_recall_protected": delta["corruption_recall"] >= -thresholds["maximum_corruption_recall_drop"] - 1e-12,
            "empty_fpr_protected": delta["empty_tile_fpr"] <= thresholds["maximum_empty_tile_fpr_increase"] + 1e-12}
    if any(not math.isfinite(value) for s in scores.values() for value in s.values()):
        raise ValueError("Non-finite matched comparison score")
    passed = all(value for group in checks.values() for value in group.values())
    return {"passed": passed, "checks": checks, "b1_minus_reference": differences,
            "decision": "propose_independent_seed_confirmation" if passed else "stop_augmentation_claim_and_review_clean_detection",
            "mechanism_training_authorized": False, "statistical_significance_claim": False}


def compare_pair(paths, runs, thresholds):
    directories = {"b0": paths["baseline_evaluation"], **{mode: cfg.output_dir.with_name(cfg.output_dir.name + "_val") for mode, cfg in runs.items()}}
    scores = {mode: score_metrics(directory) for mode, directory in directories.items()}
    gate = descriptive_gates(scores, thresholds)
    report = {"status": "single_seed_matched_descriptive_comparison", "scores": scores, "gate": gate,
              "test_accessed": False, "thresholds": thresholds, "new_model_inference": False,
              "source_summary_sha256": {mode: file_sha256(directory / "summary.json") for mode, directory in directories.items()},
              "source_order_sha256": {mode: file_sha256(cfg.output_dir / "source_order.json") for mode, cfg in runs.items()},
              "limitation": "One seed and one validation city without reliable spatial source IDs; no bootstrap or significance claim."}
    write_json(paths["output_dir"] / "comparison.json", report)
    lines = ["# Valais B1 / B0-E 单seed匹配结果", "", "| 模型 | Clean F1 | Corruption mF1 | Clean Recall | Corruption Recall | Empty FPR |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ("b0", "b1", "b0e"):
        row = scores[mode]
        lines.append(f"| {mode} | {row['clean_f1']:.6f} | {row['mf1']:.6f} | {row['clean_recall']:.6f} | {row['corruption_recall']:.6f} | {row['empty_tile_fpr']:.6f} |")
    lines.extend(["", "预声明描述性筛选门：" + ("通过" if gate["passed"] else "未通过"), "", "决策：`" + gate["decision"] + "`。",
                  "", "仅seed42、Sierre单城，无空间独立置信区间。本批不授权新机制或test访问。完整门和哈希见 comparison.json。"])
    (paths["output_dir"] / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_child(command, log_dir, name):
    with (log_dir / (name + ".stdout.log")).open("x", encoding="utf-8") as stdout, \
         (log_dir / (name + ".stderr.log")).open("x", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        write_json(log_dir / (name + ".process.json"), {"pid": child.pid, "command": command,
                   "started_utc": now(), "parent_pid": os.getpid()})
        code = child.wait()
    if code != 0: raise RuntimeError(f"{name} exited with code {code}; no automatic retry/resume")


def run_queue(path):
    source, v, paths, runs = load_study(path)
    out = paths["output_dir"]; auth = read_json(out / "authorization.json")
    if auth["status"] != "matched_development_study_authorized": raise ValueError("Missing study authorization")
    bound_check(auth["bound_sha256"])
    if (out / "execution_started.json").exists(): raise FileExistsError("One-shot queue already started")
    if Path(sys.executable).resolve() != paths["python"]: raise ValueError("Interpreter differs")
    no_other_python()
    write_json(out / "execution_started.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
               "started_utc": now(), "command": sys.argv, "python": sys.executable,
               "authorization_sha256": file_sha256(out / "authorization.json"), "test_accessed": False})
    logs = out / "logs"; logs.mkdir(exist_ok=False)
    current = "starting"
    try:
        for mode in v["order"]:
            current = mode + "_training"
            bound_check(auth["bound_sha256"])
            _atomic_json(out / "status.json", {"status": "running", "stage": current, "updated_utc": now(), "test_accessed": False})
            run_child([str(paths["python"]), "-u", "-m", "scripts.valais_matched_study", "train-stage", "--config", str(source), "--mode", mode], logs, current)
            cfg = runs[mode]
            order = read_json(cfg.output_dir / "source_order.json")
            check_training(cfg, order)
            if read_json(cfg.output_dir / "data_audit.json") != read_json(out / "live_data_audit.json"):
                raise ValueError("Training input audit differs from the sealed B0 data")
            if audit_data(cfg.base, full=True) != read_json(out / "live_data_audit.json"):
                raise ValueError("Live train/val data changed during training")
            if mode == "b0e":
                first = read_json(runs["b1"].output_dir / "source_order.json")
                for key in ("source_order_sha256", "initial_tensor_sha256", "source_pairs", "batches"):
                    if order[key] != first[key]: raise ValueError(f"B1/B0-E actual matching failed: {key}")
            current = mode + "_validation"
            _atomic_json(out / "status.json", {"status": "running", "stage": current, "updated_utc": now(), "test_accessed": False})
            job, evaluation = evaluation_job(paths, cfg)
            command = [str(paths["python"]), "-u", "-m", "scripts.evaluate_candidate_guarded", "--config", str(job), "--guard-config", str(paths["guard_config"])]
            run_child(command + ["--preflight-only"], logs, mode + "_evaluation_preflight")
            run_child(command, logs, current)
            verify_score(evaluation)
        bound_check(auth["bound_sha256"])
        comparison = compare_pair(paths, runs, v["gates"])
        _atomic_json(out / "status.json", {"status": "completed", "stage": "matched_pair_scored",
                     "completed_utc": now(), "test_accessed": False, "automatic_resume": False,
                     "decision": comparison["gate"]["decision"]})
        write_json(out / "completion.json", {"status": "matched_pair_completed", "test_accessed": False,
                   "status_sha256": file_sha256(out / "status.json"), "authorization_sha256": file_sha256(out / "authorization.json"),
                   "comparison_sha256": file_sha256(out / "comparison.json"),
                   "evaluation_summaries": {mode: file_sha256(cfg.output_dir.with_name(cfg.output_dir.name + "_val") / "summary.json") for mode, cfg in runs.items()}})
    except BaseException as error:
        _atomic_json(out / "status.json", {"status": "failed", "stage": current, "updated_utc": now(),
                     "error": f"{type(error).__name__}: {error}", "test_accessed": False, "automatic_resume": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke", "authorize", "run", "train-stage"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["b1", "b0e"])
    args = parser.parse_args()
    if args.action == "train-stage": train_stage(args.config, args.mode)
    elif args.action == "run": run_queue(args.config)
    else: print(json.dumps(globals()[args.action](args.config)))


if __name__ == "__main__": main()
