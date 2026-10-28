"""Matched clean window-size controls; archived model/epoch/loss stay unchanged."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from pcdc.config import Config, protocol_signature
from pcdc.data import LEVIRDataset, audit_data, file_sha256, make_loader, seed_worker
from pcdc.engine import move_model, read_checkpoint, run_epoch
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import (_restore_rng, _rng_state, load_robust_config,
    robust_source_files, robust_source_hashes, run_two_view_epoch, tensor_state_sha256)
from pcdc.runtime import environment_record, save_checkpoint, setup_runtime
from scripts.levir_tail_risk_control import LimitedLoader, SingleBatch, ObservedLoader, state_digest
from scripts.levir_tail_risk_supervision import now
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT = Path(__file__).resolve().parents[1]
FLAVOR = "ordinary_matched_native_window_v1"
ARMS = ("native512", "grouped256")


def validate_study(v):
    expected = {"schema_version": 1, "kind": "ordinary_matched_native_window_training_control",
        "seed": 42, "epochs": 100, "group_window": 512, "loss_unit": 256,
        "groups_per_epoch": 1780, "attempts_per_arm": 178000, "views_per_group": 2,
        "secondary": "identical_clean", "arms": list(ARMS), "threshold": .5,
        "selection": "original_clean_validation_f1_only",
        "validation": "original_clean_val256_batch32_each_epoch",
        "training_order": "same_seed42_group_shuffle_for_both_arms",
        "new_architecture": False, "new_normalization": False,
        "geometric_image_transforms": False, "test_or_martigny_access": False,
        "pilot_steps_per_run": 32, "pilot_runs": ["native512", "grouped256", "native512"]}
    if any(v.get(k) != x for k, x in expected.items()):
        raise ValueError("Frozen native-window protocol changed")
    return v


def load_study(path):
    path = Path(path).resolve()
    if path != ROOT / "configs/levir_native_window_training_20261025.yaml":
        raise ValueError("Only the explicitly frozen training study is admitted")
    v = validate_study(yaml.safe_load(path.read_text(encoding="utf-8")))
    ref = load_robust_config(ROOT / v["reference_config"])
    if ref.mode != "b0e" or ref.base.run_test_after_training:
        raise ValueError("Original clean two-view development reference required")
    if ref.base.values["runtime"]["device"] != "cuda" or not ref.base.values["train"]["amp"]:
        raise ValueError("Original CUDA FP16 is required")
    return path, v, ref, ROOT / v["output_dir"]


def quadrants(x, unit):
    if x.ndim != 4 or x.shape[-2:] != (2*unit, 2*unit):
        raise ValueError("Expected one exact 2x2 spatial group, without padding/resampling")
    n, c = x.shape[:2]
    return x.reshape(n,c,2,unit,2,unit).permute(0,2,4,1,3,5).reshape(4*n,c,unit,unit)


class PerTileLoss(nn.Module):
    def __init__(self, original, unit=256):
        super().__init__(); self.original, self.unit = original, unit

    def forward(self, logits, target):
        if logits.shape != target.shape:
            raise ValueError("Logits and labels must share exact pixel support")
        if logits.shape[-2:] == (2*self.unit, 2*self.unit):
            logits, target = quadrants(logits,self.unit), quadrants(target,self.unit)
        elif logits.shape[-2:] != (self.unit,self.unit):
            raise ValueError("Loss unit changed")
        return self.original(logits,target)


class WindowGroups(Dataset):
    def __init__(self, base, arm):
        if base.split != "train" or arm not in ARMS or base.tile_size != 512:
            raise ValueError("Training requires original native512 groups and a frozen arm")
        self.base, self.arm, self.split = base, arm, "train"

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        s=self.base[index]; ri,x,y=self.base.tiles[index]; name=self.base.records[ri].name
        tensors=[s[k].unsqueeze(0) for k in ("image_a","image_b","mask")]
        ids=[s["sample_id"]]
        if self.arm=="grouped256":
            tensors=[quadrants(t,256) for t in tensors]
            ids=[f"train/{name}@x{x+dx}_y{y+dy}" for dy in (0,256) for dx in (0,256)]
        a,b,mask=tensors; n=len(ids)
        return {"clean_a":a,"clean_b":b,"secondary_a":a.clone(),"secondary_b":b.clone(),
            "mask":mask,"sample_id":ids,"group_id":s["sample_id"],
            "secondary_kind":["clean"]*n,"secondary_level":torch.zeros(n,dtype=torch.int64),
            "secondary_direction":["clean"]*n,"secondary_target_date":["none"]*n,
            "secondary_spatial_mode":["none"]*n,"instance_json":["clean"]*n}


def one_group(items):
    if len(items)!=1:
        raise ValueError("Exactly one native512 group per optimizer attempt")
    return items[0]


def group_hash_update(h, group):
    data=group.encode("utf-8");h.update(len(data).to_bytes(8,"big"));h.update(data)


class GroupObserved(ObservedLoader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.group_digest=hashlib.sha256();self.groups=0

    def __iter__(self):
        for batch in super().__iter__():
            group_hash_update(self.group_digest,batch["group_id"]);self.groups+=1
            yield batch


def group_config(base):
    values=copy.deepcopy(base.values);values["data"]["tile_size"]=512
    values["train"]["batch_size"]=1
    return Config(values,base.source)


def construct(ref,arm,*,validation=False):
    device=setup_runtime(ref.base)
    base=LEVIRDataset(group_config(ref.base),"train");dataset=WindowGroups(base,arm)
    gen=torch.Generator().manual_seed(ref.base.values["experiment"]["seed"])
    loader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=ref.base.values["train"]["num_workers"],
        pin_memory=True,drop_last=False,collate_fn=one_group,worker_init_fn=seed_worker,generator=gen)
    val_loader=make_loader(LEVIRDataset(ref.base,"val"),ref.base,training=False) if validation else None
    model=move_model(build_model(ref.base),device,ref.base.values["runtime"]["memory_format"])
    s=ref.base.values["train"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=s["learning_rate"],betas=tuple(s["adam_betas"]),weight_decay=s["weight_decay"])
    scaler=torch.cuda.amp.GradScaler(enabled=True)
    return device,model,optimizer,scaler,dataset,loader,val_loader


def source_files():
    files=robust_source_files()
    for name in ("scripts/native_window_training.py","scripts/native_window_supervision.py",
                 "scripts/levir_tail_risk_control.py","scripts/levir_tail_risk_supervision.py",
                 "scripts/temperature_guard.py","scripts/windows_shared_json.py",
                 "tests/test_native_window_training.py"):
        files[name]=ROOT/name
    return files


def prepare(path):
    path,v,ref,out=load_study(path)
    if (out/"prepared.json").exists():raise FileExistsError("Preparation already sealed")
    audit=audit_data(ref.base,full=True,splits=("train","val"))
    original=ROOT/v["reference_run"]
    if audit!=read(original/"data_audit.json"):
        raise ValueError("Original decoded train/val identity differs")
    old_sources=read(original/"source_sha256.json")
    if robust_source_hashes()!=old_sources:
        raise ValueError("Original model/training core changed")
    torch.manual_seed(v["seed"]);torch.set_num_threads(4)
    model=build_model(ref.base);initial=tensor_state_sha256(model.state_dict());del model
    if initial!=v["initialization_sha256"]:raise ValueError("Original initialization differs")
    base=LEVIRDataset(group_config(ref.base),"train")
    if len(base)!=v["groups_per_epoch"]:raise ValueError("Unexpected group count")
    group_ids=[f"train/{base.records[ri].name}@x{x}_y{y}" for ri,x,y in base.tiles]
    order_loader=DataLoader(group_ids,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(v["seed"]))
    orders=[]
    for epoch in range(v["epochs"]):
        h=hashlib.sha256()
        for batch in order_loader:group_hash_update(h,batch[0])
        orders.append(h.hexdigest())
    out.mkdir(exist_ok=True,parents=True);atomic(out/"data_audit.json",audit)
    files=source_files()
    for p in (path,ref.source,ref.paths["base_config"],ROOT/v["protocol"],
              ROOT/"configs/levir_native_window_runtime_20261025.yaml",ROOT/"configs/smoke_native_window_20261025.yaml",
              out/"data_audit.json"):
        files[str(p)]=p
    bound={str(p.resolve()):file_sha256(p) for p in files.values()}
    for name,p in source_files().items():
        dst=out/"source_snapshot"/name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,dst)
    proof={"status":"passed_data_core_initialization_and_order_identity","created_utc":now(),
        "bound_sha256":bound,"initial_model_tensor_sha256":initial,"epoch_group_order_sha256":orders,
        "group_count":len(group_ids),"raw_splits_audited":["train","val"],"held_out_opened":False,
        "new_research_training_started":False,"source_scene_groups_claimed":False}
    atomic(out/"prepared.json",proof);return {k:v for k,v in proof.items() if k not in ("bound_sha256","epoch_group_order_sha256")}


def verify(bound):
    for p,digest in bound.items():
        if file_sha256(Path(p))!=digest:raise ValueError("Sealed source changed: "+p)


def epoch_train(ref,model,loader,device,optimizer,scaler,guard):
    s=ref.base.values["train"]
    return run_two_view_epoch(model,loader,PerTileLoss(ChangeDetectionLoss(**ref.base.values["loss"])),device,
        threshold=.5,consistency_kind="none",consistency_weight=0.,amp=True,memory_format="channels_last",
        optimizer=optimizer,scaler=scaler,grad_clip_norm=s["grad_clip_norm"],guard_callback=guard)


def state(model,optimizer,scaler,loader):
    return {"format_version":1,"model":copy.deepcopy(model.state_dict()),"optimizer":copy.deepcopy(optimizer.state_dict()),
        "scaler":copy.deepcopy(scaler.state_dict()),"rng":_rng_state(loader)}


def pilot(path,session,runtime,guard):
    _,v,ref,out=load_study(path);proof=read(out/"prepared.json");verify(proof["bound_sha256"])
    rows=[];saved=None;next_saved=None;next_sha=None
    for ordinal,arm in enumerate(v["pilot_runs"]):
        guard();device,model,opt,scaler,ds,loader,_=construct(ref,arm)
        if tensor_state_sha256(model.state_dict())!=v["initialization_sha256"]:raise ValueError("Initial state mismatch")
        torch.cuda.reset_peak_memory_stats(device)
        limited=LimitedLoader(loader,v["pilot_steps_per_run"])
        observed=GroupObserved(limited,session,scaler,"pilot_"+arm,1,guard,runtime)
        report=epoch_train(ref,model,observed,device,opt,scaler,guard);torch.cuda.synchronize(device)
        next_batch=next(limited.iterator);boundary=state(model,opt,scaler,loader);boundary_sha=state_digest(boundary)
        intervals=[b-a for a,b in zip(observed.completion_seconds,observed.completion_seconds[1:])]
        row={"arm":arm,"ordinal":ordinal,"initialization_exact":True,"report":report,
            "boundary_state_sha256":boundary_sha,"group_prefix_sha256":observed.group_digest.hexdigest(),
            "mean_seconds_per_step_after_first":sum(intervals)/len(intervals),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device)}
        if ordinal==0:
            save_checkpoint(session/"boundary.pt",boundary);saved=read_checkpoint(session/"boundary.pt")
            next_saved=next_batch
            observed_next=ObservedLoader(SingleBatch(next_batch),session,scaler,"pilot_next_update",1,guard,runtime)
            row["next_update"]=epoch_train(ref,model,observed_next,device,opt,scaler,guard)
            next_sha=state_digest(state(model,opt,scaler,loader))
        elif ordinal==2:
            if boundary_sha!=rows[0]["boundary_state_sha256"] or state_digest(next_batch)!=state_digest(next_saved):
                raise ValueError("Independent real512 repeat/next input failed")
            model.load_state_dict(saved["model"],strict=True);opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
            _restore_rng(saved["rng"],loader)
            observed_next=ObservedLoader(SingleBatch(next_batch),session,scaler,"pilot_next_replay",1,guard,runtime)
            row["next_update_replay"]=epoch_train(ref,model,observed_next,device,opt,scaler,guard)
            if state_digest(state(model,opt,scaler,loader))!=next_sha:raise ValueError("Exact next update restore failed")
        rows.append(row)
        del boundary,model,opt,scaler,ds,loader,limited,observed,next_batch
        gc.collect();torch.cuda.empty_cache()
    if len({r["group_prefix_sha256"] for r in rows})!=1:raise ValueError("Pilot group streams differ")
    result={"status":"passed_engineering_only","rows":rows,"repeat_full_state_exact":True,
        "next_update_full_state_exact":True,"paired_group_prefix_exact":True,
        "physical_training_attempts":98,"physical_pair_predictions":388,"research_optimizer_updates":0,
        "pilot_only_not_an_effectiveness_result":True,"raw_validation_scored":False,"completed_utc":now()}
    atomic(session/"pilot_verified.json",result);return result


def budget_estimate(rows,remaining,study_remaining,limit):
    times=[r["mean_seconds_per_step_after_first"] for r in rows[:2]]
    expected=[(s*178000+6000)*1.25 for s in times]
    return {"predicted_arm_seconds":dict(zip(ARMS,expected)),"predicted_total_seconds":sum(expected),
        "passed":all(s<=limit for s in expected) and sum(expected)<=min(remaining,study_remaining),
        "allowance":"(post-first mean step seconds*178000+6000)*1.25 per arm"}


def train(path,session,runtime,guard,arm):
    _,v,ref,out=load_study(path);authority=read(out/"training_authorization.json");verify(authority["bound_sha256"])
    proof=read(out/"prepared.json");run=out/arm;run.mkdir(exist_ok=False)
    device,model,opt,scaler,ds,loader,val_loader=construct(ref,arm,validation=True)
    if tensor_state_sha256(model.state_dict())!=v["initialization_sha256"] or len(loader)!=1780:
        raise ValueError("Frozen initialization or step count differs")
    settings=ref.base.values["train"];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=settings["min_learning_rate"])
    audit=read(out/"data_audit.json");history=[];best=-1.;best_epoch=0
    atomic(run/"environment.json",environment_record(ref.base,device))
    atomic(run/"status.json",{"status":"running","training_started":True,"arm":arm,"epoch":0,"epochs":100,
        "optimizer_steps":0,"amp_skipped_steps":0,"updated_utc":now(),"test_evaluated":False})
    for epoch in range(1,101):
        observed=GroupObserved(loader,session,scaler,arm+"_train",epoch,guard,runtime,(epoch-1)*1780)
        lr=opt.param_groups[0]["lr"];training=epoch_train(ref,model,observed,device,opt,scaler,guard)
        stream=observed.group_digest.hexdigest()
        if observed.groups!=1780 or stream!=proof["epoch_group_order_sha256"][epoch-1]:
            raise ValueError("Frozen paired group order mismatch")
        if training["optimizer_steps"]+training["amp_skipped_steps"]!=1780:
            raise ValueError("Attempt accounting mismatch")
        training.update(group_order_sha256=stream,source_groups=1780,tile256_equivalent_pair_views=14240)
        validation_loader=ObservedLoader(val_loader,session,None,arm+"_clean_validation",epoch,guard,runtime,epoch*1780)
        validation=run_epoch(model,validation_loader,ChangeDetectionLoss(**ref.base.values["loss"]),device,
            threshold=.5,amp=True,memory_format="channels_last")
        guard();improved=validation["metrics"]["f1"]>best
        if improved:best,best_epoch=validation["metrics"]["f1"],epoch
        scheduler.step();history.append({"epoch":epoch,"learning_rate":lr,"train":training,"val":validation})
        effective=copy.deepcopy(ref.base.values)
        effective["data"]["tile_size"]=512 if arm=="native512" else 256
        effective["train"]["batch_size"]=1 if arm=="native512" else 4
        payload={"format_version":1,"kind":"research","training_flavor":FLAVOR,"arm":arm,"epoch":epoch,
            "best_epoch":best_epoch,"best_val_f1":best,"model":model.state_dict(),"optimizer":opt.state_dict(),
            "scaler":scaler.state_dict(),"scheduler":scheduler.state_dict(),"config":effective,
            "inference_config":ref.base.values,"native_window_study":v,"loss_unit":256,
            "protocol_signature":protocol_signature(ref.base.values),"initial_model_tensor_sha256":v["initialization_sha256"],
            "manifests":{s:{k:x for k,x in d.items() if k!="file_sha256"} for s,d in audit["splits"].items()},
            "file_sha256":{s:d["file_sha256"] for s,d in audit["splits"].items()},
            "authority_sha256":file_sha256(out/"training_authorization.json"),
            "source_sha256":{k:file_sha256(p) for k,p in source_files().items()},"rng":_rng_state(loader),"history":history}
        if improved:save_checkpoint(run/"best.pt",payload)
        save_checkpoint(run/"last.pt",payload);atomic(run/"history.json",history)
        status={"status":"running","arm":arm,"training_started":True,"epoch":epoch,"epochs":100,
            "best_epoch":best_epoch,"best_val_f1":best,"updated_utc":now(),"training_flavor":FLAVOR,
            "optimizer_steps":sum(r["train"]["optimizer_steps"] for r in history),
            "amp_skipped_steps":sum(r["train"]["amp_skipped_steps"] for r in history),
            "attempts":epoch*1780,"tile256_equivalent_pair_views":epoch*14240,
            "actual_pair_views":sum(r["train"]["pair_views"] for r in history),"test_evaluated":False}
        atomic(run/"status.json",status)
        print(f"native/{arm} epoch={epoch}/100 val256_f1={validation['metrics']['f1']:.8f} best={best:.8f} attempts={epoch*1780}",flush=True)
    status.update(status="completed",completed_utc=now());atomic(run/"status.json",status);return status


def smoke(path):
    _,_,ref,_=load_study(path);cfg=yaml.safe_load((ROOT/"configs/smoke_native_window_20261025.yaml").read_text(encoding="utf-8"))
    if cfg["device"]!="cpu" or cfg["amp"] is not False or cfg["kind"]!="synthetic_engineering_only":raise ValueError("Explicit CPU smoke only")
    torch.set_num_threads(4);rows={}
    for arm in ARMS:
        torch.manual_seed(cfg["seed"]);model=build_model(ref.base).cpu();before=tensor_state_sha256(model.state_dict())
        n=cfg["group_window"];a=torch.rand(1,3,n,n);b=torch.rand_like(a);mask=(torch.rand(1,1,n,n)>.8).float()
        if arm=="grouped256":a,b,mask=[quadrants(x,cfg["loss_unit"]) for x in (a,b,mask)]
        count=len(a);batch={"clean_a":a,"clean_b":b,"secondary_a":a.clone(),"secondary_b":b.clone(),"mask":mask,
            "sample_id":[str(i) for i in range(count)],"secondary_kind":["clean"]*count,
            "secondary_level":torch.zeros(count,dtype=torch.int64),"secondary_direction":["clean"]*count,
            "secondary_target_date":["none"]*count,"secondary_spatial_mode":["none"]*count,"instance_json":["clean"]*count}
        opt=torch.optim.AdamW(model.parameters(),lr=ref.base.values["train"]["learning_rate"])
        result=run_two_view_epoch(model,SingleBatch(batch),PerTileLoss(ChangeDetectionLoss(**ref.base.values["loss"]),cfg["loss_unit"]),torch.device("cpu"),
            threshold=.5,consistency_kind="none",consistency_weight=0.,amp=False,memory_format="channels_last",
            optimizer=opt,scaler=torch.cuda.amp.GradScaler(enabled=False),grad_clip_norm=1.)
        assert result["optimizer_steps"]==1 and before!=tensor_state_sha256(model.state_dict())
        rows[arm]={"optimizer_steps":1,"finite_loss":True,"parameters_updated":True,"pair_views":result["pair_views"]}
    result={"status":"passed","synthetic_only":True,"device":"cpu","amp":False,"original_fpn_width":64,"arms":rows}
    out=ROOT/cfg["output_dir"];out.mkdir(exist_ok=False,parents=True);atomic(out/"verification.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","smoke"])
    p.add_argument("--study",type=Path,required=True);a=p.parse_args()
    print(json.dumps(globals()[a.action](a.study),ensure_ascii=False))


if __name__=="__main__":main()
