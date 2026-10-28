"""Finite saved-result analysis, train-only BN intervention and one R5 engineering step.

Never edits an existing checkpoint, trains a research candidate, or reads held-out data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.data import file_sha256
from pcdc.engine import check_checkpoint, move_model, read_checkpoint, source_files
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts import evaluate_coupling_p0 as p0
from scripts.analyze_valais_matched_errors import describe_sources, paired_sources, size_groups
from scripts.analyze_valais_saved_subgroups import validate_sources
from scripts.coupling_diagnostic import metrics_from_counts, pool_counts
from scripts.evaluate_guarded import _atomic_json
from scripts.prepare_coupling_p0 import train_path
from scripts.stratified_sampling import load_sampling

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("b0e", "cs", "as")
ARMS = ("original", "uniform_clean", "stratified_clean")
COUNT_KEYS = ("tn", "fp", "fn", "tp")
BN_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    path = Path(path).resolve()
    v = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    expected = {"schema_version": 1, "selection_seed": 2026102001,
        "calibration_sources_per_arm": 256, "source_batch_size": 4, "probe_sources": 64,
        "bn_arms": list(ARMS), "bn_views_per_source": 2, "bn_max_seconds": 3600,
        "pilot_max_seconds": 600, "pilot_steps": 1,
        "pilot_condition": {"family": "gamma", "level": 3},
        "pilot_view_weights": [.5, .125, .125, .125, .125]}
    if any(v.get(k) != value for k, value in expected.items()) or set(v["models"]) != set(ROLES):
        raise ValueError("Changed finite screen scope")
    paths = {k: (path.parent / v[k]).resolve() for k in ("output_dir", "protocol_document",
        "posttraining_dir", "p0_manifest", "p0_review", "p0_saved_audit", "b0e_evaluation", "data_audit",
        "label_report", "sampling_proposal")}
    paths["output_dir"].relative_to(ROOT / "runs")
    models = {role: {k: (path.parent / value).resolve() if k != "checkpoint_sha256" else value
                       for k, value in row.items()} for role, row in v["models"].items()}
    return path, v, paths, models


def calibration_selection(names, foreground, probe_names, seed, count=256):
    """Hash-ranked train names; labels fix composition, never predictions."""
    if len(names) != len(set(names)) or not set(probe_names) <= set(names) or count % 4:
        raise ValueError("Invalid/disjoint selection inventory")
    if any(Path(n).name != n or "/" in n or "\\" in n or not n.endswith(".tif") for n in names):
        raise ValueError("Train names must be bare TIFF identities")
    rank = lambda n: (digest(["bn-screen-selection-v1", seed, n]), n)
    ordered = sorted(set(names) - set(probe_names), key=rank)
    positive = [n for n in ordered if foreground.get(n, 0) > 0]
    empty = [n for n in ordered if foreground.get(n, 0) == 0]
    if len(ordered) < count or len(positive) < count//4 or len(empty) < 3*count//4:
        raise ValueError("Insufficient real train sources; no substitution")
    stratified = [n for i in range(count//4) for n in [positive[i], *empty[3*i:3*i+3]]]
    return {"uniform_clean": ordered[:count], "stratified_clean": stratified}


def validate_selection(manifest):
    probe = manifest["probe"]
    ids = {s["name"] for s in probe}
    if len(ids) != 64 or len(probe) != 64 or sum(s["foreground_pixels"] > 0 for s in probe) != 16:
        raise ValueError("Changed original enriched probe")
    if set(manifest["calibration"])!=set(ARMS[1:]):
        raise ValueError("Both fixed calibration arms are required")
    for name, samples in manifest["calibration"].items():
        if name not in ARMS[1:] or len(samples) != 256 or len({s["name"] for s in samples}) != 256:
            raise ValueError("Changed calibration budget/uniqueness")
        if ids & {s["name"] for s in samples}:
            raise ValueError("Calibration/probe overlap")
        if name == "stratified_clean" and any(
                sum(s["foreground_pixels"] > 0 for s in samples[i:i+4]) != 1 for i in range(0,256,4)):
            raise ValueError("Changed 1:3 calibration composition")
    for sample in probe + [s for rows in manifest["calibration"].values() for s in rows]:
        for key, folder in (("image_a", "A"), ("image_b", "B"), ("mask", "label")):
            if Path(sample["files"][key]["path"]).resolve() != train_path(manifest["data_root"], folder, sample["name"]):
                raise ValueError("Non-train or aliased image path")


def prepare(path):
    source, v, paths, models = load(path)
    final = read(paths["posttraining_dir"] / "final_completion_audit.json")
    if final["status"] != "passed_final_p1_completion_audit" or final["test_accessed"]:
        raise ValueError("Completed P1 evidence required")
    for name, sha in final["bound_completed_artifact_sha256"].items():
        p0.checked_hash(name, sha)
    old = read(paths["posttraining_dir"] / "authorization.json")
    for key in ("p0_manifest", "p0_review", "sampling_proposal"):
        p0.checked_hash(paths[key], old["bound_sha256"][str(paths[key])])
    _, evidence, _ = load_sampling(paths["sampling_proposal"])
    if evidence["captured_data_audit"] != paths["data_audit"] or evidence["train_label_report"] != paths["label_report"]:
        raise ValueError("Changed original train evidence paths")
    saved_p0=read(paths["p0_saved_audit"])
    if saved_p0["status"]!="passed_saved_result_audit" or saved_p0["test_accessed"]:
        raise ValueError("Original P0 saved audit required")
    for name in ("manifest", "report", "records"):
        item=saved_p0["inputs"][name]
        p0.checked_hash(item["path"],item["sha256"])
    original = read(paths["p0_manifest"])
    p0.validate_manifest(original)
    p0.validate_review(original, paths["p0_manifest"], paths["p0_review"])
    audit, labels = read(paths["data_audit"]), read(paths["label_report"])
    train = audit["splits"]["train"]
    names = [r["name"] for r in train["entries"]]
    foreground = {r["name"]: r["positive_pixels"] for r in labels["train"]["positive_manifest"]}
    if len(names) != 11025 or len(foreground) != 1046 or labels["test_accessed"]:
        raise ValueError("Changed training inventory")
    selected = calibration_selection(names, foreground, {s["name"] for s in original["samples"]}, v["selection_seed"])
    def sample(name):
        return {"name": name, "id": f"train/{name}@x0_y0", "foreground_pixels": foreground.get(name,0),
            "files": {k: {"path": str(train_path(original["data_root"], f, name)), "sha256": train["file_sha256"][name][k]}
                      for k,f in (("image_a","A"),("image_b","B"),("mask","label"))}}
    manifest = {"schema_version": 1, "data_root": original["data_root"], "settings": original["settings"],
        "probe": original["samples"], "calibration": {k: [sample(n) for n in rows] for k,rows in selected.items()},
        "selection_seed": v["selection_seed"], "selection_depends_on_predictions": False,
        "pilot_sources": [next(s["name"] for s in original["samples"] if s["foreground_pixels"]>0),
                          *[s["name"] for s in original["samples"] if s["foreground_pixels"]==0][:3]]}
    validate_selection(manifest)
    bindings = {str(source): file_sha256(source), str(paths["protocol_document"]): file_sha256(paths["protocol_document"])}
    dependencies = list(source_files().values()) + [ROOT / "scripts/valais_feasibility_screen.py",
        ROOT / "tests/test_feasibility_screen.py", ROOT / "scripts/evaluate_coupling_p0.py",
        ROOT / "scripts/coupling_diagnostic.py", ROOT / "scripts/analyze_valais_matched_errors.py",
        ROOT / "scripts/analyze_valais_saved_subgroups.py", ROOT / "scripts/prepare_coupling_p0.py",
        ROOT / "scripts/evaluate_guarded.py", ROOT / "scripts/stratified_sampling.py"]
    dependencies += [p for k,p in paths.items() if k not in ("output_dir", "posttraining_dir", "b0e_evaluation")]
    dependencies += [paths["posttraining_dir"] / n for n in ("final_completion_audit.json", "training_completion_audit.json",
        "authorization.json", "comparison.json", "residual/records.jsonl", "residual/report.json")]
    dependencies += [Path(saved_p0["inputs"][n]["path"]) for n in ("report","records")]
    for role, row in models.items():
        p0.checked_hash(row["checkpoint"], row["checkpoint_sha256"])
        dependencies += [row[k] for k in ("checkpoint", "base_config", "history")]
        dependencies += [row["evaluation"] / n for n in ("clean_metrics.json", "summary.json", "guard_completion.json")]
    dependencies += [Path(s["files"][k]["path"]) for samples in [manifest["probe"], *manifest["calibration"].values()]
                      for s in samples for k in s["files"]]
    for samples in [manifest["probe"], *manifest["calibration"].values()]:
        for s in samples:
            for item in s["files"].values():
                p0.checked_hash(item["path"], item["sha256"])
    for dep in set(dependencies):
        bindings[str(dep.resolve())] = file_sha256(dep)
    output = paths["output_dir"]
    output.mkdir(exist_ok=False)
    write_json(output / "manifest.json", manifest)
    bindings[str(output / "manifest.json")] = file_sha256(output / "manifest.json")
    write_json(output / "authorization.json", {"status":"authorized_finite_feasibility_screen",
        "human_instruction":"好，开始执行", "bound_sha256":bindings,
        "research_training_authorized":False,"engineering_optimizer_steps_authorized":1,
        "test_accessed":False,"new_validation_inference_authorized":False})
    return {"status":"prepared", "sealed_inputs":len(bindings),
        "calibration_positive_counts":{k:sum(s["foreground_pixels"]>0 for s in rows) for k,rows in manifest["calibration"].items()}}


def verify(paths):
    auth = read(paths["output_dir"] / "authorization.json")
    for name, sha in auth["bound_sha256"].items():
        p0.checked_hash(name, sha)
    manifest = read(paths["output_dir"] / "manifest.json")
    validate_selection(manifest)
    return manifest


def analyze(path):
    _, _, paths, models = load(path)
    manifest = verify(paths)
    histories = {r:read(m["history"]) for r,m in models.items()}
    clean = {r:read(m["evaluation"] / "clean_metrics.json") for r,m in models.items()}
    labels = read(paths["label_report"])["train"]
    result = {"status":"completed_saved_result_analysis", "new_model_inference":False,
        "raw_image_reads":False, "test_accessed":False, "saved_validation_results_read":True,
        "area_boundaries":[64,256,1024,4096],"models":{},"comparisons":{}}
    for role in ROLES:
        validate_sources(clean[role])
        h = histories[role]
        if len(h)!=100 or [r["epoch"] for r in h]!=list(range(1,101)):
            raise ValueError("Incomplete completed history")
        selected = max(h,key=lambda row:row["val"]["metrics"]["f1"])
        if clean[role]["counts"]!={k:selected["val"]["metrics"][k] for k in COUNT_KEYS}:
            raise ValueError("Clean saved counts differ from selected checkpoint history")
        uniform = role=="b0e"
        sampling = []
        for row in h:
            train = row["train"]
            if train["optimizer_steps"]!=2757 or train["pair_views"]!=22050 or train["amp_skipped_steps"]:
                raise ValueError("Changed committed historical budget")
            draws = train["clean_metrics"]["tp"]+train["clean_metrics"]["fn"]
            sampling.append({"epoch":row["epoch"],"positive_pixel_draws":draws,
                "positive_sources":1046 if uniform else train["sampling"]["positive_draws"],
                "empty_sources":9979 if uniform else train["sampling"]["empty_draws"]})
        result["models"][role] = {"selected_epoch":selected["epoch"],
            "clean":describe_sources(clean[role]["by_source"]),
            "clean_size_groups":size_groups(clean[role]["by_source"],[64,256,1024,4096]),
            "selected_online_train_clean_metrics":selected["train"]["clean_metrics"],
            "final_val_metrics":h[-1]["val"]["metrics"],
            "history":[{"epoch":row["epoch"],"train_loss":row["train"]["loss"]["total"],
                "online_train_clean_f1":row["train"]["clean_metrics"]["f1"],
                "val_f1":row["val"]["metrics"]["f1"],"val_recall":row["val"]["metrics"]["recall"]} for row in h],
            "exposure": {"positive_source_draws":sum(x["positive_sources"] for x in sampling),
                "empty_source_draws":sum(x["empty_sources"] for x in sampling),
                "positive_pixel_draws":sum(x["positive_pixel_draws"] for x in sampling),
                "per_epoch":sampling}}
    for current,reference in (("cs","b0e"),("as","cs"),("as","b0e")):
        a,b=clean[current]["by_source"],clean[reference]["by_source"]
        partitions={"all":set(a)}
        partitions.update({label:{sid for sid,row in a.items() if area_label(row["foreground_pixels"])==label}
                           for label in ("empty","1-64","65-256","257-1024","1025-4096",">4096")})
        result["comparisons"][f"{current}_minus_{reference}"] = {label:paired_sources(
            {sid:a[sid] for sid in ids},{sid:b[sid] for sid in ids}) for label,ids in partitions.items()}
    q=math.prod((9979-i)/(11025-i) for i in range(4))
    result["sampling_summary"] = {"natural_positive_source_fraction":1046/11025,
        "stratified_positive_source_fraction":2757/11025,"positive_exposure_multiplier":2757/1046,
        "empty_exposure_multiplier":8268/9979,"uniform_all_empty_full_batch_probability":q,
        "stratified_all_empty_batch_probability":0,
        "natural_positive_pixels_per_epoch":labels["positive_pixels"],
        "interpretation":"Composition, repetition, batch context, BN and optimization changed jointly; not isolated causes."}
    result["bn_calibration_manifest"]={k:{"sources":len(rows),"positive_sources":sum(s["foreground_pixels"]>0 for s in rows),
        "foreground_pixels":sum(s["foreground_pixels"] for s in rows)} for k,rows in manifest["calibration"].items()}
    rows=[json.loads(line) for line in (paths["posttraining_dir"] / "residual/records.jsonl").read_text(encoding="utf-8").splitlines()]
    result["residual_effects"]={}
    for role in ("cs","as"):
        by={}
        for family in sorted(p0.FAMILIES):
            selected=[r for r in rows if r["model"]==role and r["family"]==family and r["foreground_pixels"]>0
                      and not r["marginal_check"]["fully_degenerate"]]
            energies={k:sum(r["diagnostic"]["logit_effects"][k]["foreground"]["square_sum"] for r in selected) for k in ("m","a","b","i")}
            by[family]={"nondegenerate_positive_records":len(selected),"foreground_effect_square_sums":energies,
                "interaction_to_main_rms_ratio":math.sqrt(energies["i"]/(energies["a"]+energies["b"])) if energies["a"]+energies["b"] else None,
                "interaction_to_mean_logit_rms_ratio":math.sqrt(energies["i"]/energies["m"]) if energies["m"] else None,
                "D_counts":pool_counts(r["diagnostic"]["groups"]["D"]["counts"] for r in selected),
                "O_counts":pool_counts(r["diagnostic"]["groups"]["O"]["counts"] for r in selected)}
        result["residual_effects"][role]=by
    result["limitations"]=["Online train metrics use changing weights and different source mixtures, not a matched frozen-model train/val gap.",
        "Area is total foreground per tile, not object size.","Aggregate i/a/b association does not identify the cause of task failures.",
        "Single seed; source names are not independent geographic units."]
    write_json(paths["output_dir"] / "saved_analysis.json",result)
    return {"status":result["status"],"models":{r:{"best_epoch":v["selected_epoch"],"clean_f1":v["clean"]["metrics"]["f1"]} for r,v in result["models"].items()}}


def area_label(n):
    return "empty" if n==0 else ("1-64" if n<=64 else "65-256" if n<=256 else
        "257-1024" if n<=1024 else "1025-4096" if n<=4096 else ">4096")


@contextmanager
def recalibrating_bn(model):
    """Reset only BN buffers; enable only BN training; leave all learned weights alone."""
    modules=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    if not modules:
        raise ValueError("Expected real BatchNorm modules")
    saved=[(m,m.momentum) for m in modules]
    if any(not m.track_running_stats for m in modules):
        raise ValueError("BN must track running statistics")
    model.eval()
    for m,_ in saved:
        m.reset_running_stats()
        m.momentum=None
        m.train(True)
    try:
        with torch.no_grad():
            yield
    finally:
        model.eval()
        for m,momentum in saved:
            m.momentum=momentum


def unchanged_except_bn(original, current):
    if set(original)!=set(current):
        raise ValueError("Model state keys changed")
    changed=[]
    for k,t in original.items():
        if not torch.equal(t.cpu(),current[k].detach().cpu()):
            if not k.endswith(BN_SUFFIXES):
                raise ValueError(f"Learned weight or non-BN buffer changed: {k}")
            changed.append(k)
    return changed


def original_clean(paths, role):
    if role in ("cs","as"):
        rows=[json.loads(line) for line in (paths["posttraining_dir"] / "residual/records.jsonl").read_text(encoding="utf-8").splitlines()]
    else:
        records=Path(read(paths["p0_saved_audit"])["inputs"]["records"]["path"])
        rows=[json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    result={}
    for row in rows:
        if row["model"]==role:
            counts=row["diagnostic"]["clean"]["counts"]
            if row["source_id"] in result and result[row["source_id"]]!=counts:
                raise ValueError("Inconsistent original clean records")
            result[row["source_id"]]=counts
    return result


def verified_checkpoint(role, row, paths):
    p0.checked_hash(row["checkpoint"],row["checkpoint_sha256"])
    cfg=load_config(row["base_config"])
    cp=read_checkpoint(row["checkpoint"])
    check_checkpoint(cfg,cp)
    expected_flavor="matched_robust_baseline_v1" if role=="b0e" else "stratified_two_view_baseline_v1"
    if (cp.get("training_flavor")!=expected_flavor or cp.get("robust_mode")!={"b0e":"b0e","cs":"b0e","as":"b1"}[role]
            or cp.get("kind")!="research" or set(cp["file_sha256"])!={"train","val"}
            or cfg.run_test_after_training or cfg.values["runtime"]["device"]!="cuda" or cfg.values["train"]["amp"]):
        raise ValueError("Wrong checkpoint identity or numerical/development scope")
    audit=read(paths["data_audit"])
    if cp["file_sha256"]!={k:r["file_sha256"] for k,r in audit["splits"].items()}:
        raise ValueError("Checkpoint data identity differs from original audit")
    for field in ("source_sha256","stratified_trainer_source_sha256"):
        for name,sha in cp.get(field,{}).items():
            p0.checked_hash(ROOT / name,sha)
    h=read(row["history"])
    selected=max(h,key=lambda r:r["val"]["metrics"]["f1"])
    if cp["epoch"]!=selected["epoch"] or cp["history"]!=h[:cp["epoch"]]:
        raise ValueError("Checkpoint is not first highest original clean selection")
    return cfg,cp


def tensorize(raw,cfg,device):
    data=cfg.values["data"]
    t=torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1))).float()/255
    t=(t-torch.tensor(data["mean"]).view(3,1,1))/torch.tensor(data["std"]).view(3,1,1)
    return t.unsqueeze(0).to(device,memory_format=torch.channels_last)


def report_counts(rows):
    counts=pool_counts(r["counts"] for r in rows)
    empty=pool_counts(r["counts"] for r in rows if r["foreground_pixels"]==0)
    return {"sources":len(rows),"counts":counts,"metrics":metrics_from_counts(counts),
        "empty_fpr":empty["fp"]/(empty["fp"]+empty["tn"]) if empty["fp"]+empty["tn"] else 0,
        "by_area":{label:{"counts":pool_counts(r["counts"] for r in rows if area_label(r["foreground_pixels"])==label)}
            for label in ("empty","1-64","65-256","257-1024","1025-4096",">4096")}}


def probe_measurements(clean_logits, logits, target):
    if clean_logits.shape!=target.shape or logits.shape!=target.shape or not target.numel():
        raise ValueError("Probe logit/target shape mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(clean_logits).all() or not ((target==0)|(target==1)).all():
        raise ValueError("Probe needs finite logits/binary target")
    clean,pred,truth=clean_logits.sigmoid()>=.5,logits.sigmoid()>=.5,target.bool()
    counts=dict(zip(COUNT_KEYS,[int(x.sum()) for x in (~truth&~pred,~truth&pred,truth&~pred,truth&pred)]))
    transitions=dict(zip(("tp_to_fn","fn_to_tp","tn_to_fp","fp_to_tn"),[int(x.sum()) for x in
        (truth&clean&~pred,truth&~clean&pred,~truth&~clean&pred,~truth&clean&~pred)]))
    return counts,transitions


def bn_run(path,smoke=False):
    _,v,paths,models=load(path)
    manifest=verify(paths)
    output=paths["output_dir"] / ("cuda_smoke" if smoke else "bn_diagnostic")
    output.mkdir(exist_ok=False)
    settings=dict(manifest["settings"],max_seconds=v["bn_max_seconds"])
    guard=p0.Guard(settings)
    rows=[]
    report={"status":"incomplete","engineering_only":smoke,"models":{},"parameter_training":False,
        "batchnorm_recalibration":True,"new_val_inference":False,"test_accessed":False,"train_only":True}
    try:
        guard.check()
        for role in ROLES:
            cfg,cp=verified_checkpoint(role,models[role],paths)
            if cfg.data_root!=Path(manifest["data_root"]).resolve():
                raise ValueError("Diagnostic/checkpoint training data roots differ")
            device=setup_runtime(cfg)
            if role=="b0e": report["environment"]=environment_record(cfg,device)
            model=move_model(build_model(cfg,initialize_encoder=False),device,"channels_last").float()
            original={k:t.clone() for k,t in cp["model"].items()}
            del cp
            model.requires_grad_(False)
            old=original_clean(paths,role)
            model_result={}
            baselines={}
            for arm in ARMS:
                model.load_state_dict(original,strict=True)
                model.eval()
                calibration=manifest["calibration"].get(arm,[])
                calibration=calibration[:4] if smoke else calibration
                steps=0
                if calibration:
                    with recalibrating_bn(model):
                        for i in range(0,len(calibration),4):
                            batch=[p0.read_sample(s,cfg) for s in calibration[i:i+4]]
                            a=torch.cat([tensorize(x[0],cfg,device) for x in batch])
                            b=torch.cat([tensorize(x[1],cfg,device) for x in batch])
                            guard.check()
                            with torch.autocast(device_type="cuda",enabled=False):
                                z=model(torch.cat((a,a)),torch.cat((b,b)))["logits"]
                            torch.cuda.synchronize(device)
                            guard.check()
                            if z.dtype!=torch.float32 or not torch.isfinite(z).all():
                                raise FloatingPointError("Invalid calibration output")
                            guard.predictions+=8
                            steps+=1
                    if any(int(m.num_batches_tracked)!=steps for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)):
                        raise ValueError("Unexpected calibration BN update count")
                changed=unchanged_except_bn(original,model.state_dict())
                calibrated_hash=tensor_state_sha256(model.state_dict())
                arm_rows=[]
                for i,sample in enumerate(manifest["probe"][:1] if smoke else manifest["probe"]):
                    raw_a,raw_b,target=p0.read_sample(sample,cfg)
                    a,b=tensorize(raw_a,cfg,device),tensorize(raw_b,cfg,device)
                    def infer(x,y):
                        guard.check()
                        with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=False):
                            z=model(x,y)["logits"]
                        torch.cuda.synchronize(device)
                        guard.predictions+=len(x)
                        guard.check()
                        if z.dtype!=torch.float32 or not torch.isfinite(z).all():
                            raise FloatingPointError("Nonfinite/non-FP32 probe")
                        return z.cpu()
                    z=infer(a,b)
                    if i==0:
                        repeat,swap=infer(a,b),infer(b,a)
                        if not torch.equal(z,repeat) or not torch.allclose(z,swap,atol=1e-5,rtol=1e-5):
                            raise ValueError("Probe determinism/symmetry failed")
                    counts,_=probe_measurements(z,z,target)
                    if arm=="original":
                        if counts!=old[sample["id"]]:
                            raise ValueError("Original probe counts failed exact saved-result reproduction")
                        baselines[sample["id"]]=z.clone()
                    _,transitions=probe_measurements(baselines[sample["id"]],z,target)
                    row={"model":role,"arm":arm,"source_id":sample["id"],"foreground_pixels":sample["foreground_pixels"],
                        "counts":counts,"transitions":transitions,
                        "logits":{"mean":z.double().mean().item(),"rms":z.double().square().mean().sqrt().item(),
                                  "std":z.double().std(unbiased=False).item()}}
                    rows.append(row)
                    arm_rows.append(row)
                    with (output / "records.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(row,allow_nan=False)+"\n")
                if tensor_state_sha256(model.state_dict())!=calibrated_hash:
                    raise ValueError("Probe eval changed model state")
                model_result[arm]={"calibration_sources":len(calibration),"calibration_steps":steps,
                    "only_bn_buffers_changed":True,"changed_buffer_keys":changed,"state_sha256":calibrated_hash,
                    "probe":report_counts(arm_rows)}
                _atomic_json(output / "progress.json",{"model":role,"arm":arm,"predictions":guard.predictions,"records":len(rows)})
            model.load_state_dict(original,strict=True)
            if tensor_state_sha256(model.state_dict())!=tensor_state_sha256(original):
                raise ValueError("Model restore failed")
            model_result["original_restored"]=True
            report["models"][role]=model_result
            del model,original,baselines
            torch.cuda.empty_cache()
        expected=9 if smoke else 576
        if len(rows)!=expected or guard.predictions!=(75 if smoke else 3666):
            raise ValueError("BN inference/calibration budget mismatch")
        report["status"]="engineering_cuda_smoke_passed" if smoke else "completed_train_only_bn_diagnostic"
    except BaseException as error:
        report["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(records=len(rows),pair_views=guard.predictions,elapsed_seconds=time.monotonic()-guard.started,
            maximum_observed_temperature_c=max(guard.temperatures) if guard.temperatures else None)
        write_json(output / "report.json",report)
        write_json(output / "guard_observations.json",{"temperatures_c":guard.temperatures,"automatic_retry":False})
    return {k:report[k] for k in ("status","records","pair_views","elapsed_seconds")}


def r5_loss(logits,target,loss_fn):
    count=len(target)
    if len(logits)!=5*count:
        raise ValueError("R5 requires exactly five equally sized views")
    losses=[loss_fn(logits[j*count:(j+1)*count],target)["total"] for j in range(5)]
    return .5*losses[0]+sum(losses[1:])*.125


def pilot(path):
    _,v,paths,models=load(path)
    manifest=verify(paths)
    diagnostic=read(paths["output_dir"] / "bn_diagnostic/report.json")
    if diagnostic["status"]!="completed_train_only_bn_diagnostic":
        raise ValueError("Complete identity/numerical diagnostic first")
    output=paths["output_dir"] / "r5_engineering_pilot"
    output.mkdir(exist_ok=False)
    guard=p0.Guard(dict(manifest["settings"],max_seconds=v["pilot_max_seconds"]))
    report={"status":"incomplete","engineering_only":True,"research_score_produced":False,
        "test_accessed":False,"new_val_inference":False,"optimizer_steps":0,"automatic_retry":False,
        "source_batch_size":4,"views_per_source":5,"amp":False,"batchnorm":"single_joint_forward"}
    try:
        guard.check()
        cfg=load_config(models["b0e"]["base_config"])
        device=setup_runtime(cfg)
        model=move_model(build_model(cfg,initialize_encoder=False),device,"channels_last").float().train()
        before=tensor_state_sha256(model.state_dict())
        if before!=v["initial_model_tensor_sha256"]:
            raise ValueError("R5 pilot changed original seed42 initialization")
        report["initial_model_tensor_sha256"]=before
        samples={s["name"]:s for s in manifest["probe"]}
        view_a,view_b,targets=[[] for _ in range(5)],[[] for _ in range(5)],[]
        for name in manifest["pilot_sources"]:
            sample=samples[name]
            a,b,target=p0.read_sample(sample,cfg)
            condition=next(c for c in sample["conditions"] if {"family":c["family"],"level":c["level"]}==v["pilot_condition"])
            views,marginal=p0.render_views(a,b,condition)
            combinations=[(a,b)]+[(views[x],views[y]) for x,y in p0.CELLS.values()]
            for j,(x,y) in enumerate(combinations):
                view_a[j].append(tensorize(x,cfg,device))
                view_b[j].append(tensorize(y,cfg,device))
            targets.append(target)
        a,b=torch.cat([torch.cat(xs) for xs in view_a]),torch.cat([torch.cat(xs) for xs in view_b])
        target=torch.cat(targets).to(device)
        opt=cfg.values["train"]
        optimizer=torch.optim.AdamW(model.parameters(),lr=opt["learning_rate"],betas=opt["adam_betas"],weight_decay=opt["weight_decay"])
        loss_fn=ChangeDetectionLoss(**cfg.values["loss"])
        torch.cuda.reset_peak_memory_stats(device)
        free,total=torch.cuda.mem_get_info(device)
        report.update(environment=environment_record(cfg,device),free_cuda_bytes_before_step=free,total_cuda_bytes=total,
            sources=manifest["pilot_sources"],view_weights=v["pilot_view_weights"])
        guard.check()
        started=time.monotonic()
        with torch.autocast(device_type="cuda",enabled=False):
            logits=model(a,b)["logits"]
            loss=r5_loss(logits,target,loss_fn)
        guard.predictions+=20
        if logits.dtype!=torch.float32 or not torch.isfinite(loss):
            raise FloatingPointError("Invalid R5 FP32 forward/loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),opt["grad_clip_norm"],error_if_nonfinite=True)
        guard.check()
        optimizer.step()
        report["optimizer_steps"]=1
        torch.cuda.synchronize(device)
        guard.check()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite updated pilot parameters")
        if any(int(m.num_batches_tracked)!=1 for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)):
            raise ValueError("R5 must use exactly one joint BN update")
        if tensor_state_sha256(model.state_dict())==before:
            raise ValueError("Engineering update did not change state")
        report.update(status="passed_one_step_r5_engineering_pilot",loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
            single_step_seconds=time.monotonic()-started,all_bn_updates_exactly_one=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="r5_single_batch_memory_infeasible",error=str(error),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-guard.started,pair_views=guard.predictions,
            maximum_observed_temperature_c=max(guard.temperatures) if guard.temperatures else None)
        write_json(output / "report.json",report)
        write_json(output / "guard_observations.json",{"temperatures_c":guard.temperatures,"automatic_retry":False})
    return {k:report[k] for k in ("status","optimizer_steps","elapsed_seconds")}


def audit(path):
    _,_,paths,_=load(path)
    manifest=verify(paths)
    directory=paths["output_dir"] / "bn_diagnostic"
    report=read(directory / "report.json")
    rows=[json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    expected={(r,a,s["id"]) for r in ROLES for a in ARMS for s in manifest["probe"]}
    keys=[(r["model"],r["arm"],r["source_id"]) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected) or report["records"]!=576 or report["pair_views"]!=3666:
        raise ValueError("Incomplete or duplicate BN diagnostic coverage")
    for role in ROLES:
        original={r["source_id"]:r for r in rows if r["model"]==role and r["arm"]=="original"}
        for arm in ARMS:
            chosen=[r for r in rows if r["model"]==role and r["arm"]==arm]
            if report_counts(chosen)!=report["models"][role][arm]["probe"]:
                raise ValueError("Saved probe pooling mismatch")
            for row in chosen:
                c,old,t=row["counts"],original[row["source_id"]]["counts"],row["transitions"]
                if (sum(c.values())!=65536 or c["tp"]+c["fn"]!=row["foreground_pixels"]
                        or c["fn"]-old["fn"]!=t["tp_to_fn"]-t["fn_to_tp"]
                        or c["fp"]-old["fp"]!=t["tn_to_fp"]-t["fp_to_tn"]):
                    raise ValueError("Pixel count/transition conservation failed")
    temps=read(directory / "guard_observations.json")["temperatures_c"]
    if (report["status"]!="completed_train_only_bn_diagnostic" or not temps or max(temps)>=85
            or report["elapsed_seconds"]>=3600 or not all(report["models"][r]["original_restored"] for r in ROLES)):
        raise ValueError("BN numerical/budget/restore completion failed")
    p=read(paths["output_dir"] / "r5_engineering_pilot/report.json")
    if p["status"] not in ("passed_one_step_r5_engineering_pilot","r5_single_batch_memory_infeasible"):
        raise ValueError("Pilot has no bounded engineering outcome")
    if p["elapsed_seconds"]>=600 or p["test_accessed"] or p["new_val_inference"] or p["optimizer_steps"]>1:
        raise ValueError("Pilot exceeded its frozen scope/budget")
    if p["status"]=="passed_one_step_r5_engineering_pilot" and (
            p["optimizer_steps"]!=1 or not p["all_bn_updates_exactly_one"]
            or not math.isfinite(p["loss"]) or not math.isfinite(p["gradient_norm_before_clip"])
            or p["maximum_observed_temperature_c"]>=85):
        raise ValueError("Pilot did not complete one finite FP32 update")
    result={"status":"passed_finite_screen_saved_audit","probe_records":576,"probe_predictions":594,
        "calibration_pair_views":3072,"calibration_forwards":384,"new_val_inference":False,"test_accessed":False,
        "old_checkpoints_and_bound_inputs_unchanged":True,"pilot_status":p["status"],"research_training_admitted":False,
        "result_sha256":{str(f):file_sha256(f) for f in (directory / "report.json",directory / "records.jsonl",
            directory / "guard_observations.json", paths["output_dir"] / "saved_analysis.json",
            paths["output_dir"] / "r5_engineering_pilot/report.json")}}
    write_json(paths["output_dir"] / "final_audit.json",result)
    return {k:result[k] for k in ("status","probe_records","pilot_status","research_training_admitted")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","analyze","smoke","bn","pilot","audit"))
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    if args.action in ("smoke","bn"):
        value=bn_run(args.config,smoke=args.action=="smoke")
    else:
        value=globals()[args.action](args.config)
    print(json.dumps(value,ensure_ascii=False))


if __name__=="__main__":
    main()
