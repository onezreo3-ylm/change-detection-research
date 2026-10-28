"""Explicit risk-checkpoint adapter to the unchanged frozen val evaluator.

No relabeling or monkeypatching of the archived matched evaluator. Training,
test access and parameter/threshold search are absent from this entry point.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from pcdc.config import protocol_signature
from pcdc.data import LEVIRDataset, audit_data, file_sha256, make_loader
from pcdc.engine import move_model, read_checkpoint, source_files as core_sources
from pcdc.evaluate_corruptions import (
    COUNT_KEYS, METRIC_KEYS, _bound_input_hashes, _bind_validation_files,
    _verify_input_hashes, counts_report, run_condition,
)
from pcdc.frozen_validation import FrozenInstanceDataset, load_bundle, load_evaluation_config
from pcdc.model import build_model
from pcdc.robust_baselines import _exposure_summary
from pcdc.runtime import environment_record, setup_runtime
from scripts.levir_tail_risk_control import FLAVOR, load_study, verify_bound
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    path = Path(path).resolve()
    v = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (v["schema_version"] != 1 or (path.parent / v["workspace_root"]).resolve() != ROOT
            or v["condition_count"] != 45 or v["source_count"] != 64
            or v["held_out_test_open"] or v["automatic_retry"]
            or v["bootstrap"]["replicates"] != 20000 or v["bootstrap"]["seed"] != 20261024):
        raise ValueError("Frozen evaluation scope changed")
    return path, v, ROOT / v["output_dir"]


def check_risk_identity(cp, authority, ref, study):
    if (cp.get("format_version") != 1 or cp.get("training_flavor") != FLAVOR or cp.get("kind") != "research"
            or cp.get("training_signature") != authority["risk_training_signature"]
            or cp.get("config") != ref.base.values or cp.get("risk_config") != study
            or cp.get("reference_robust_config") != dict(ref.values)
            or cp.get("protocol_signature") != protocol_signature(ref.base.values)
            or cp.get("initial_model_tensor_sha256") != study["candidate"]["initialization_sha256"]
            or cp.get("corruption_protocol_sha256") != ref.protocol.fingerprint
            or set(cp.get("file_sha256", {})) != {"train", "val"}):
        raise ValueError("Not the frozen actual risk checkpoint; identity/scope mismatch")


def check_finished_history(best, last, status, reference_history):
    history = last.get("history", [])
    if (last.get("epoch") != 100 or len(history) != 100 or len(reference_history) != 100 or status.get("epoch") != 100
            or status.get("status") != "completed_training_requires_explicit_validation"
            or status.get("test_evaluated") is not False):
        raise ValueError("100 completed development-only epochs required")
    selected = max(history, key=lambda row: row["val"]["metrics"]["f1"])
    if (best.get("epoch") != selected["epoch"] or best.get("best_epoch") != selected["epoch"]
            or last.get("best_epoch") != selected["epoch"] or status.get("best_epoch") != selected["epoch"]
            or best.get("best_val_f1") != selected["val"]["metrics"]["f1"]
            or status.get("best_val_f1") != best["best_val_f1"] or last.get("best_val_f1") != best["best_val_f1"]
            or best.get("history") != history[:best["epoch"]]):
        raise ValueError("Risk best checkpoint is not the original clean-val selection")
    for epoch, (row, reference) in enumerate(zip(history, reference_history), 1):
        if row["epoch"] != epoch or any("test" in key for key in row):
            raise ValueError("Bad epoch sequence or held-out leakage")
        for field in ("source_samples", "pair_views", "exposure_stream_sha256", "exposure_counts"):
            if row["train"][field] != reference["train"][field]:
                raise ValueError(f"Original B1 exposure differs at epoch {epoch}: {field}")
        if (row["learning_rate"] != reference["learning_rate"]
                or row["train"]["optimizer_steps"] + row["train"]["amp_skipped_steps"] != 1780):
            raise ValueError("Original schedule/attempts changed")
    successes = sum(row["train"]["optimizer_steps"] for row in history)
    skips = sum(row["train"]["amp_skipped_steps"] for row in history)
    views = sum(row["train"]["pair_views"] for row in history)
    if (successes != status["optimizer_steps"] or skips != status["amp_skipped_steps"]
            or successes + skips != 178000 or views != 1424000 or status["pair_views"] != views):
        raise ValueError("Final exposure/update accounting mismatch")
    return {"epochs": 100, "attempts": successes+skips, "successful_updates": successes,
            "amp_skips": skips, "pair_views": views, "best_epoch": selected["epoch"],
            "best_clean_f1": best["best_val_f1"], "all_original_epoch_streams_exact": True,
            "expected_clean": selected["val"]}


def prepare(path):
    path, v, output = load(path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        raise FileExistsError("One preparation only")
    _, study, ref, training = load_study(ROOT / v["training_study"])
    authority = read(training / "authorization.json")
    verify_bound(authority["bound_sha256"])
    owner = read(training / "training_session/owner_state.json")
    worker = read(training / "training_session/worker_result.json")
    if owner["status"] != "completed" or owner["worker_exit_code"] != 0 or worker["status"] != "completed":
        raise ValueError("Bound training did not complete normally")
    guards = {role: read(training / f"training_session/{role}_state.json") for role in ("guard", "watchdog")}
    if any(g["status"] != "worker_exited" or g.get("error") for g in guards.values()):
        raise ValueError("Independent training supervision did not end normally")
    run = training / "candidate_seed42"
    best, last = read_checkpoint(run / "best.pt"), read_checkpoint(run / "last.pt")
    for cp in (best, last):
        check_risk_identity(cp, authority, ref, study)
        for name, expected in cp["source_sha256"].items():
            if file_sha256(ROOT / name) != expected:
                raise ValueError("Frozen training source differs: " + name)
        if cp["authority_sha256"] != file_sha256(training / "authorization.json"):
            raise ValueError("Checkpoint authority changed")
    reference_history = read(ROOT / study["reference_run"] / "history.json")
    verified = check_finished_history(best, last, read(run / "status.json"), reference_history)
    if _exposure_summary(last["history"]) != read(run / "training_exposure_summary.json"):
        raise ValueError("Final exposure summary mismatch")
    audit = audit_data(ref.base, full=True)
    if audit != read(training / "data_audit.json") or audit != read(ROOT / study["reference_run"] / "data_audit.json"):
        raise ValueError("Data content changed since original B1 / risk training")
    bundle = load_bundle(load_evaluation_config(ROOT / v["benchmark_config"]), require_review=True)
    if (bundle.protocol.fingerprint != ref.protocol.fingerprint
            or best["file_sha256"]["val"] != bundle.manifest["file_sha256"]
            or best["manifests"]["val"] != bundle.manifest["inventory"]):
        raise ValueError("Risk data and frozen benchmark differ")
    bindings = _bound_input_hashes(bundle)
    dataset = LEVIRDataset(ref.base, "val")
    if dataset.manifest() != bundle.manifest["inventory"]:
        raise ValueError("Current val inventory differs")
    _bind_validation_files(bindings, bundle, dataset, best)
    sources = dict(core_sources())
    for name in ("scripts/levir_tail_risk_evaluation.py", "scripts/levir_tail_risk_eval_owner.py",
                 "scripts/levir_tail_risk_supervision.py", "scripts/windows_shared_json.py",
                 "scripts/temperature_guard.py", "tests/test_levir_tail_risk_evaluation.py"):
        sources[name] = ROOT / name
    extra = [path, ROOT / v["runtime_config"], ROOT / v["training_study"], training / "authorization.json",
             run / "best.pt", run / "last.pt", run / "history.json", run / "status.json",
             ROOT / study["reference_run"] / "best.pt"]
    if file_sha256(extra[-1]) != v["reference_checkpoint_sha256"]:
        raise ValueError("Original B1 reference checkpoint changed")
    reference = ROOT / v["reference_validation"]
    for name in ("conditions.json", "clean_metrics.json", "summary.json", "provenance.json"):
        extra.append(reference / name)
    baseline = read(reference / "conditions.json")
    if set(baseline) != set(bundle.manifest["condition_counts"]) or read(reference/"summary.json")["candidate_mode"] != "b1":
        raise ValueError("Wrong mean reference or condition coverage")
    for file in [*sources.values(), *extra]:
        bindings[str(file)] = file_sha256(file)
    _verify_input_hashes(bindings)
    verified.update(status="completed_training_identity_verified", training_flavor=FLAVOR,
        best_checkpoint_sha256=file_sha256(run/"best.pt"), last_checkpoint_sha256=file_sha256(run/"last.pt"),
        original_training_wall_seconds=owner["elapsed_seconds"], training_maximum_observed_temperature_c=guards["guard"]["maximum_observed_temperature_c"],
        discarded_prefix_updates=201, discarded_prefix_seconds=47.906, test_evaluated=False)
    atomic(output/"training_completion_verification.json", verified)
    bindings[str(output/"training_completion_verification.json")] = file_sha256(output/"training_completion_verification.json")
    for name, src in sources.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    proof = {"status": "prepared", "created_utc": now(), "bound_sha256": bindings,
             "benchmark_fingerprints": bundle.fingerprints, "training": verified,
             "original_mean_reference": str(reference), "test_open": False}
    atomic(output/"prepared.json", proof)
    return verified


def f1(counts):
    counts = np.asarray(counts)
    denominator = 2*counts[...,3]+counts[...,1]+counts[...,2]
    return np.divide(2*counts[...,3],denominator,out=np.zeros_like(denominator,dtype=np.float64),where=denominator!=0)


def grouped_counts(candidate, baseline, ids, source_count=None):
    if set(candidate) != set(ids) or set(baseline) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError("Condition coverage mismatch")
    sources = sorted(candidate[ids[0]]["by_source"])
    if source_count is not None and len(sources) != source_count:
        raise ValueError("Wrong original source count")
    result = []
    for conditions in (baseline,candidate):
        model = []
        for cid in ids:
            row = conditions[cid]
            if sorted(row["by_source"]) != sources:
                raise ValueError("Source coverage mismatch")
            counts = np.array([[row["by_source"][s]["counts"][k] for k in COUNT_KEYS] for s in sources],dtype=np.int64)
            pooled = np.array([row["counts"][k] for k in COUNT_KEYS],dtype=np.int64)
            if (counts<0).any() or not np.array_equal(counts.sum(0),pooled):
                raise ValueError("Source counts do not sum to pooled counts")
            model.append(counts)
        result.append(model)
    array = np.asarray(result)
    if not np.array_equal(array[0,...,0]+array[0,...,1],array[1,...,0]+array[1,...,1]) or not np.array_equal(array[0,...,2]+array[0,...,3],array[1,...,2]+array[1,...,3]):
        raise ValueError("Paired original labels/pixel supports differ")
    for model in array:
        if not np.all(model[...,0]+model[...,1]==(model[0,:,0]+model[0,:,1])[None,:]) or not np.all(model[...,2]+model[...,3]==(model[0,:,2]+model[0,:,3])[None,:]):
            raise ValueError("Original label supports changed between conditions")
    return array,sources


def bootstrap_tail(counts, *, replicates, seed, k=5, chunk=512):
    counts = np.asarray(counts,dtype=np.int64)
    if counts.ndim != 4 or counts.shape[0]!=2 or counts.shape[-1]!=4 or not 1<=k<=counts.shape[1] or replicates<1:
        raise ValueError("Expected two models x conditions x sources x four counts")
    rng = np.random.default_rng(seed)
    n=counts.shape[2];values=[]
    flat=counts.transpose(2,0,1,3).reshape(n,-1)
    for start in range(0,replicates,chunk):
        b=min(chunk,replicates-start)
        indices=rng.integers(0,n,size=(b,n))
        weights=np.zeros((b,n),dtype=np.int64)
        np.add.at(weights,(np.repeat(np.arange(b),n),indices.ravel()),1)
        pooled=(weights@flat).reshape(b,2,counts.shape[1],4)
        scores=f1(pooled)
        tails=np.sort(scores,axis=2)[:,:,:k].mean(2)
        values.append(tails[:,1]-tails[:,0])
    return np.concatenate(values)


def model_scores(conditions, ids, clean):
    vectors=np.array([[conditions[c]["counts"][k] for k in COUNT_KEYS] for c in ids],dtype=np.int64)
    scores=f1(vectors);pooled=vectors.sum(0)
    pooled_metrics=counts_report(pooled.tolist())["metrics"]
    worst=sorted(range(len(ids)),key=lambda i:(scores[i],ids[i]))
    return {"clean_f1":clean["metrics"]["f1"], "mean_f1":float(scores.mean()),
            "worst_five_f1":float(scores[worst[:5]].mean()), "worst_single_f1":float(scores.min()),
            "pooled_recall":pooled_metrics["recall"],
            "pooled_background_fpr":pooled_metrics["background_fpr"],
            "worst_five_conditions":[ids[i] for i in worst[:5]]}


def gate_results(candidate, baseline, rules, lower):
    deltas={k:candidate[k]-baseline[k] for k in candidate if k!='worst_five_conditions'}
    checks={"clean_f1":deltas["clean_f1"]>=rules["clean_f1_min_delta"],
        "mean_f1":deltas["mean_f1"]>=rules["mean_corruption_f1_min_delta"],
        "pooled_recall":deltas["pooled_recall"]>=rules["pooled_corruption_recall_min_delta"],
        "pooled_background_fpr":deltas["pooled_background_fpr"]<=rules["pooled_background_fpr_max_delta"],
        "worst_five_f1":deltas["worst_five_f1"]>=rules["worst_five_mean_f1_min_delta"],
        "worst_single_f1":deltas["worst_single_f1"]>=rules["worst_single_condition_f1_min_delta"],
        "bootstrap_tail_lower_positive":lower>0}
    return {"delta_candidate_minus_mean_b1":deltas,"gates":checks,"all_passed":all(checks.values())}


def analyze(path):
    _,v,out=load(path);_,study,_,_=load_study(ROOT/v["training_study"])
    candidate=read(out/"conditions.json");baseline=read(ROOT/v["reference_validation"]/"conditions.json")
    ids=list(baseline)
    counts,sources=grouped_counts(candidate,baseline,ids,v["source_count"])
    clean=read(out/"clean_metrics.json");base_clean=read(ROOT/v["reference_validation"]/"clean_metrics.json")
    c,b=model_scores(candidate,ids,clean),model_scores(baseline,ids,base_clean)
    samples=bootstrap_tail(counts,replicates=v["bootstrap"]["replicates"],seed=v["bootstrap"]["seed"],chunk=v["bootstrap"]["chunk_size"])
    np.save(out/"bootstrap_tail_deltas.npy",samples,allow_pickle=False)
    low,high=np.quantile(samples,[.025,.975],method="linear")
    result={"status":"completed", "candidate":c,"ordinary_mean_b1":b,
        **gate_results(c,b,study["evaluation"],float(low)),
        "source_pair_bootstrap":{"replicates":len(samples),"seed":v["bootstrap"]["seed"],
            "source_count":len(sources),"source_ids":sources,"lower95":float(low),"upper95":float(high),
            "shared_source_draws_across_all_conditions_and_models":True,"tail_resorted_per_model_per_replicate":True,
            "percentile":"linear two-sided 95%"}, "test_evaluated":False,
        "new_method_claim":False,"more_training_authorized_by_single_seed":False}
    result["decision"]="single_seed_pass_only_replication_discussion" if result["all_passed"] else "reject_this_fixed_candidate_no_more_q_or_lambda_search"
    atomic(out/"analysis.json",result)
    return result


class ProgressLoader:
    def __init__(self, loader, session, cid, completed):
        self.loader,self.session,self.cid,self.completed=loader,session,cid,completed
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for step,batch in enumerate(self.loader,1):
            yield batch
            atomic(self.session/"progress.json",{"stage":"fixed_validation_only","condition_id":self.cid,
                "completed_conditions":self.completed,"step":step,"steps_in_stage":len(self),
                "updated_utc":now(),"numeric_alert":False,"optimizer_updates":0})


def evaluate(path,session,guard):
    path,v,out=load(path);proof=read(out/"prepared.json")
    _verify_input_hashes(proof["bound_sha256"])
    _,study,ref,training=load_study(ROOT/v["training_study"])
    bundle=load_bundle(load_evaluation_config(ROOT/v["benchmark_config"]),require_review=True)
    best=read_checkpoint(training/"candidate_seed42/best.pt")
    check_risk_identity(best,read(training/"authorization.json"),ref,study)
    data=LEVIRDataset(ref.base,"val")
    if data.manifest()!=bundle.manifest["inventory"]:raise ValueError("Validation inventory changed")
    guard();device=setup_runtime(ref.base)
    model=move_model(build_model(ref.base,initialize_encoder=False),device,ref.base.values["runtime"]["memory_format"])
    model.load_state_dict(best["model"],strict=True)
    atomic(out/"environment.json",environment_record(ref.base,device))
    clean=run_condition(model,ProgressLoader(make_loader(data,ref.base,training=False),session,"clean",0),ref.base,device,guard_callback=guard)
    expected=proof["training"]["expected_clean"]
    if (clean["samples"]!=expected["samples"] or any(clean["counts"][k]!=expected["metrics"][k] for k in COUNT_KEYS)
            or any(clean["metrics"][k]!=expected["metrics"][k] for k in METRIC_KEYS)):
        raise ValueError("Risk clean replay differs from its selected checkpoint")
    clean["matches_risk_checkpoint_exactly"]=True;atomic(out/"clean_metrics.json",clean)
    grouped=defaultdict(list)
    for instance in bundle.instances:grouped[instance["condition_id"]].append(instance)
    ids=list(bundle.manifest["condition_counts"]);results={}
    (out/"conditions").mkdir(exist_ok=False)
    for cid in ids:
        cell=FrozenInstanceDataset(data,grouped[cid])
        row=run_condition(model,ProgressLoader(make_loader(cell,ref.base,training=False),session,cid,len(results)),ref.base,device,guard_callback=guard)
        expected_strata={key[len(cid)+1:]:value for key,value in bundle.manifest["stratum_counts"].items() if key.startswith(cid+'|')}
        if (row["condition_id"]!=cid or row["samples"]!=1024 or len(row["by_source"])!=64
                or row["pixels"]!=1024*256**2 or {key:x["samples"] for key,x in row["joint_strata"].items()}!=expected_strata):
            raise ValueError("Frozen condition/source/stratum support differs")
        atomic(out/"conditions"/(cid+".json"),row);results[cid]=row
        atomic(out/"status.json",{"status":"evaluating_original45","completed_conditions":len(results),"condition_count":45,"training_flavor":FLAVOR,"test_evaluated":False,"updated_utc":now()})
        print(f"risk_eval/{cid} f1={row['metrics']['f1']:.9f} completed={len(results)}/45",flush=True)
    atomic(out/"conditions.json",results)
    del model,best;torch.cuda.empty_cache();guard()
    result=analyze(path)
    atomic(out/"status.json",{"status":"completed_fixed_validation_and_gates","completed_conditions":45,"all_gates_passed":result["all_passed"],"decision":result["decision"],"test_evaluated":False,"training_flavor":FLAVOR,"optimizer_updates":0,"updated_utc":now()})
    return result


def seal(path,test_log):
    import re
    _,v,out=load(path);proof=read(out/"prepared.json");_verify_input_hashes(proof["bound_sha256"])
    log=Path(test_log).resolve();text=log.read_text(encoding="utf-8");match=re.search(r"Ran (\d+) tests? in ",text)
    if not match or int(match[1])<460 or not text.rstrip().endswith('OK') or read(out/"smoke.json")["status"]!='passed':
        raise ValueError("Required full tests and smoke have not passed")
    bound=dict(proof["bound_sha256"])
    for p in (log,out/"prepared.json",out/"smoke.json"):bound[str(p)]=file_sha256(p)
    if (out/"authorization.json").exists():raise FileExistsError("Already sealed")
    atomic(out/"authorization.json",{"status":"sealed_explicit_risk_validation","bound_sha256":bound,"created_utc":now(),"full_tests":int(match[1]),"optimizer_updates":0,"test_open":False})
    return {"status":"sealed_explicit_risk_validation","full_tests":int(match[1])}


def smoke(path):
    _,v,out=load(path);_,_,ref,_=load_study(ROOT/v['training_study'])
    from pcdc.evaluate_corruptions import per_sample_confusions
    torch.manual_seed(42);torch.set_num_threads(2)
    model=build_model(ref.base,initialize_encoder=False).eval()
    with torch.no_grad():
        logits=model(torch.randn(2,3,32,32),torch.randn(2,3,32,32))['logits']
        target=torch.randint(0,2,logits.shape).float()
        counts,_=per_sample_confusions(logits,target,threshold=.5)
    if logits.shape!=(2,1,32,32) or counts.sum().item()!=2048:raise ValueError('CPU model/count smoke failed')
    result={'status':'passed','kind':'synthetic_original_architecture_CPU_FP32_engineering_only',
            'optimizer_updates':0,'research_result':False,'gpu_compute':False,'pixels':2048,
            'actual_research_evaluation_still_uses_original_FP16':True}
    atomic(out/'smoke.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','analyze','seal','smoke']);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--test-log',type=Path)
    args=parser.parse_args();r=seal(args.config,args.test_log) if args.action=='seal' else globals()[args.action](args.config)
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
