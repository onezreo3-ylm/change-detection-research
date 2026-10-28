"""Original P0 inputs and audit math, explicitly applied to the frozen 40ep C5/R5 models."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from pcdc.data import file_sha256
from pcdc.engine import move_model
from pcdc.model import build_model
from pcdc.runtime import setup_runtime, write_json
from pcdc.robust_baselines import tensor_state_sha256
from scripts import audit_coupling_p0 as independent
from scripts import evaluate_coupling_p0 as p0
from scripts.coupling_diagnostic import four_cell_diagnostic, metrics_from_counts, pool_counts, pool_transitions
from scripts import c5r5_fixed_evaluation as parent
from scripts.evaluate_guarded import _atomic_json


def summarize(rows, roles=("c5", "r5")):
    aggregates, gates = {}, {}
    for role in roles:
        selected_model = [row for row in rows if row["model"] == role]
        buckets = {"all": selected_model}
        buckets.update({"family:"+family: [r for r in selected_model if r["family"] == family] for family in sorted(p0.FAMILIES)})
        buckets.update({"stratum:"+s: [r for r in selected_model if r["stratum"] == s] for s in sorted({r["stratum"] for r in selected_model})})
        buckets.update({f"condition:{family}:{level}": [r for r in selected_model if r["family"] == family and r["level"] == level]
                        for family in sorted(p0.FAMILIES) for level in (1,3,5)})
        output = {}
        for key, selected in buckets.items():
            groups = {}
            for group in ("D", "O"):
                counts = pool_counts(r["diagnostic"]["groups"][group]["counts"] for r in selected)
                groups[group] = {"counts": counts, "metrics": metrics_from_counts(counts),
                    "transitions": pool_transitions(r["diagnostic"]["groups"][group]["transitions"] for r in selected)}
            effects = {}
            for effect in ("m", "a", "b", "i"):
                effects[effect] = {}
                for area in ("all", "foreground", "background"):
                    stats = {name: sum(r["diagnostic"]["logit_effects"][effect][area][name] for r in selected)
                             for name in ("pixels", "sum", "abs_sum", "square_sum")}
                    n = stats["pixels"]
                    stats.update(mean=stats["sum"]/n if n else None, mean_abs=stats["abs_sum"]/n if n else None,
                                 rms=math.sqrt(stats["square_sum"]/n) if n else None)
                    effects[effect][area] = stats
            output[key] = {"records": len(selected), "groups": groups, "logit_effects": effects,
                "D_minus_O": {k: groups["D"]["metrics"][k]-groups["O"]["metrics"][k] for k in ("f1", "recall", "fpr")}}
        unique = {r["source_id"]: r["diagnostic"]["clean"]["counts"] for r in selected_model}
        counts = pool_counts(unique.values())
        output["clean_unique_sources"] = {"sources": len(unique), "counts": counts, "metrics": metrics_from_counts(counts)}
        aggregates[role] = output
        passing = []
        for family in sorted(p0.FAMILIES):
            row = output["family:"+family]
            harms = sum(row["groups"][g]["transitions"][k] for g in ("D", "O") for k in ("tp_to_fn", "tn_to_fp"))
            if abs(row["D_minus_O"]["f1"]) >= .005 and harms > 0:
                passing.append(family)
        gates[role] = {"qualifying_families": passing, "passed": len(passing) >= 2}
    return {"aggregates": aggregates, "gate": {"models": gates, "passed": all(r["passed"] for r in gates.values()),
        "interpretation": "Fixed enriched train screening, not significance, causality or P2 authorization."}}


def audit_records(manifest, rows, report):
    """Rebuild D/O from cells with the original independent conservation checks."""
    samples = {row["id"]: row for row in manifest["samples"]}
    expected = {(role, sid, family, level) for role in parent.ROLES for sid in samples for family in p0.FAMILIES for level in (1,3,5)}
    seen, clean = set(), {}
    for row in rows:
        key = (row["model"], row["source_id"], row["family"], row["level"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected residual record")
        seen.add(key)
        independent.check_record(row, samples[row["source_id"]])
        c = row["diagnostic"]["clean"]["counts"]
        if clean.setdefault(key[:2], c) != c:
            raise ValueError("Clean prediction changed across conditions")
    if seen != expected or len(rows) != 1536:
        raise ValueError("Incomplete residual Cartesian coverage")
    for role in parent.ROLES:
        for label, saved in report["aggregates"][role].items():
            if label == "clean_unique_sources":
                counts = independent.sum_fields((c for (r, _), c in clean.items() if r == role), independent.COUNTS)
                if saved["sources"] != 64 or saved["counts"] != counts:
                    raise ValueError("Unique clean source counts changed")
                independent.equal_metrics(saved["metrics"], independent.metrics(counts))
                continue
            selected = [r for r in rows if r["model"] == role and
                (label == "all" or label == "family:"+r["family"] or label == "stratum:"+r["stratum"]
                 or label == f"condition:{r['family']}:{r['level']}")]
            if saved["records"] != len(selected):
                raise ValueError("Residual subgroup coverage differs")
            for group, cells in (("D", ("uu", "vv")), ("O", ("uv", "vu"))):
                counts = independent.sum_fields((r["diagnostic"]["cells"][cell]["counts"] for r in selected for cell in cells), independent.COUNTS)
                transitions = independent.sum_fields((r["diagnostic"]["cells"][cell]["transitions"] for r in selected for cell in cells), independent.TRANSITIONS)
                if saved["groups"][group]["counts"] != counts or saved["groups"][group]["transitions"] != transitions:
                    raise ValueError("Residual aggregate differs from original cells")
                independent.equal_metrics(saved["groups"][group]["metrics"], independent.metrics(counts))
    recomputed = summarize(rows)
    if recomputed != {k: report[k] for k in ("aggregates", "gate")}:
        raise ValueError("Residual effects or gate were altered")
    return {"status": "passed_saved_residual_result_audit", "records": len(rows), "train_sources": 64,
            "new_model_inference": False, "test_accessed": False, "val_accessed": False}


def validate_execution(report, observations):
    if (report["status"] != "completed_c5r5_train_residual_diagnostic" or report["smoke_only"] is not False
            or report["records"] != 1536 or report["main_predictions"] != 6272 or report["predictions"] != 6276
            or report["training"] is not False or report["gradients_enabled"] is not False
            or report["amp"] is not False or report["test_accessed"] is not False or report["val_accessed"] is not False
            or not 0 <= report["elapsed_seconds"] <= report["budget_seconds"] == 21600
            or not observations or any(type(t) is not int or not 0 <= t < 85 for t in observations)
            or max(observations) != report["maximum_observed_temperature_c"]):
        raise ValueError("Incomplete residual execution, scope or guard coverage")
    checks = report["self_checks"]
    if (len(checks) != 2 or {c["model"] for c in checks} != set(parent.ROLES)
            or any(c["repeat_bitwise_equal"] is not True or c["temporal_swap_allclose"] is not True
                   or not math.isfinite(c["max_swap_abs_error"]) or c["max_swap_abs_error"] < 0 for c in checks)):
        raise ValueError("Residual repeat/symmetry checks incomplete")
    if set(report["models"]) != set(parent.ROLES) or any(not r["model_state_unchanged"] for r in report["models"].values()):
        raise ValueError("Residual model state changed")


def audit_saved(paths):
    directory = paths["output_dir"] / "residual"
    report = parent.read(directory / "report.json")
    completion = parent.read(directory / "guard_completion.json")
    temperatures = parent.read(directory / "guard_observations.json")["temperatures_c"]
    validate_execution(report, temperatures)
    if (completion["status"] != "completed" or completion["report_sha256"] != file_sha256(directory / "report.json")
            or completion["guard_observations_sha256"] != file_sha256(directory / "guard_observations.json")):
        raise ValueError("Residual completion hashes changed")
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    result = audit_records(parent.read(paths["p0_manifest"]), rows, report)
    result.update(report_sha256=file_sha256(directory / "report.json"), records_sha256=file_sha256(directory / "records.jsonl"),
                  completed_guard_verified=True)
    target = directory / "final_saved_result_audit.json"
    if not target.exists():
        write_json(target, result)
    return result


@torch.inference_mode()
def run(path, *, smoke=False):
    _, value, paths, settings, study_paths, runs = parent.load(path)
    if not smoke:
        parent.verify_seal(paths)
    manifest = parent.read(paths["p0_manifest"])
    p0.validate_manifest(manifest)
    p0.validate_review(manifest, paths["p0_manifest"], paths["p0_review"])
    output = paths["output_dir"] / ("cuda_smoke" if smoke else "residual")
    output.mkdir(exist_ok=False)
    guard = p0.Guard(manifest["settings"])
    def guarded():
        if smoke and time.monotonic()-guard.started >= value["smoke_max_seconds"]:
            raise RuntimeError("Fixed engineering smoke exceeded 600 seconds")
        guard.check()
    rows, checks = [], []
    report = {"status": "incomplete", "train_only": True, "val_accessed": False, "test_accessed": False,
        "training": False, "gradients_enabled": False, "amp": False, "smoke_only": smoke,
        "manifest_sha256": file_sha256(paths["p0_manifest"]), "review_sha256": file_sha256(paths["p0_review"]), "models": {}}
    main_predictions = 0
    try:
        guarded()
        for role in parent.ROLES:
            best, _ = parent.validate_row(role, runs[role], settings, study_paths)
            report["models"][role] = {"checkpoint": str(runs[role].output_dir / "best.pt"),
                "checkpoint_sha256": file_sha256(runs[role].output_dir / "best.pt"),
                "training_signature": best["training_signature"], "training_identity": best["identity"]}
            config = runs[role].base
            if config.data_root != Path(manifest["data_root"]).resolve() or config.run_test_after_training:
                raise ValueError("Residual data scope changed")
            device = setup_runtime(config)
            model = move_model(build_model(config, initialize_encoder=False), device, config.values["runtime"]["memory_format"])
            model.load_state_dict(best["model"], strict=True)
            del best
            model.requires_grad_(False)
            model.eval()
            before_state = tensor_state_sha256(model.state_dict())
            mean = torch.tensor(config.values["data"]["mean"], dtype=torch.float32).view(3,1,1)
            std = torch.tensor(config.values["data"]["std"], dtype=torch.float32).view(3,1,1)
            def tensor(raw):
                rgb = torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1))).float()/255
                return ((rgb-mean)/std).unsqueeze(0).to(device, memory_format=torch.channels_last)
            def infer(a, b):
                guarded()
                with torch.autocast(device_type="cuda", enabled=False):
                    logits = model(a,b)["logits"]
                torch.cuda.synchronize(device)
                guard.predictions += len(a)
                guarded()
                if logits.dtype != torch.float32 or not torch.isfinite(logits).all().item():
                    raise FloatingPointError("Nonfinite or non-FP32 residual inference")
                return logits.cpu()
            selected = manifest["samples"][:1] if smoke else manifest["samples"]
            for sample_index, sample in enumerate(selected):
                raw_a, raw_b, target = p0.read_sample(sample, config)
                a,b = tensor(raw_a),tensor(raw_b)
                clean = infer(a,b)
                main_predictions += 1
                if sample_index == 0:
                    repeat, swap = infer(a,b), infer(b,a)
                    if not torch.equal(clean,repeat) or not torch.allclose(clean,swap,atol=1e-5,rtol=1e-5):
                        raise ValueError("Residual repeat or temporal symmetry self check failed")
                    checks.append({"model":role,"repeat_bitwise_equal":True,"temporal_swap_allclose":True,
                                   "max_swap_abs_error":(clean-swap).abs().max().item()})
                for condition in sample["conditions"]:
                    views,marginal = p0.render_views(raw_a,raw_b,condition)
                    inputs = {key:tensor(raw) for key,raw in views.items()}
                    logits = {cell:infer(inputs[x],inputs[y]) for cell,(x,y) in p0.CELLS.items()}
                    main_predictions += 4
                    if marginal["fully_degenerate"] and any(not torch.equal(logits["uu"],x) for x in logits.values()):
                        raise ValueError("Identical u/v four-cell predictions differ")
                    row = {"model":role,"source_id":sample["id"],"stratum":sample["stratum"],
                        "foreground_pixels":sample["foreground_pixels"],"family":condition["family"],"level":condition["level"],
                        "marginal_check":marginal,"diagnostic":four_cell_diagnostic(clean,logits,target)}
                    independent.check_record(row,sample)
                    if not smoke:
                        rows.append(row)
                        with (output/"records.jsonl").open("a",encoding="utf-8") as stream:
                            stream.write(json.dumps(row,allow_nan=False)+"\n")
                if not smoke:
                    _atomic_json(output/"progress.json",{"model":role,"sample":sample_index+1,"records":len(rows),"predictions":guard.predictions})
                    print(f"residual/{role} sample={sample_index+1}/64 records={len(rows)}",flush=True)
            if tensor_state_sha256(model.state_dict()) != before_state:
                raise ValueError("Frozen model parameters or buffers changed during inference")
            report["models"][role].update(model_state_sha256=before_state, model_state_unchanged=True)
            del model
            torch.cuda.empty_cache()
        if smoke:
            report.update(status="passed_train_only_c5r5_cuda_smoke",research_result=False)
        else:
            if len(rows)!=1536 or main_predictions!=6272 or guard.predictions!=6276:
                raise ValueError("Residual inference coverage incomplete")
            guarded()
            report.update(summarize(rows),status="completed_c5r5_train_residual_diagnostic")
            audit = audit_records(manifest,rows,report)
            p0.validate_manifest(manifest)
            p0.validate_review(manifest, paths["p0_manifest"], paths["p0_review"])
            parent.verify_seal(paths)
            guarded()
            write_json(output/"saved_result_audit.json",audit)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(predictions=guard.predictions,main_predictions=main_predictions,records=len(rows),self_checks=checks,
            elapsed_seconds=time.monotonic()-guard.started,budget_seconds=value["smoke_max_seconds"] if smoke else value["residual_max_seconds"],
            maximum_observed_temperature_c=max(guard.temperatures) if guard.temperatures else None)
        write_json(output/"guard_observations.json",{"temperatures_c":guard.temperatures,"automatic_retry":False})
        write_json(output/"report.json",report)
        if report["status"].startswith("completed"):
            write_json(output/"guard_completion.json",{"status":"completed","report_sha256":file_sha256(output/"report.json"),
                "guard_observations_sha256":file_sha256(output/"guard_observations.json")})
    if not smoke:
        audit_saved(paths)
    else:
        parent.validate_smoke(report,guard.temperatures)
    return report
