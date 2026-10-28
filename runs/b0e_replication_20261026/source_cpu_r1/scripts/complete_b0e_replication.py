"""Fill missing B0-E seed controls using the unchanged archived trainer."""
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
from torch.utils.data import DataLoader
import yaml

from pcdc.corruption_protocol import canonical, sample_training_instance
from pcdc.data import LEVIRDataset, audit_data, make_loader
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import (TwoViewTrainingDataset, _rng_state, _restore_rng,
    _update_stream_hash, load_robust_config, robust_source_files, robust_source_hashes,
    run_two_view_epoch, tensor_state_sha256, train_robust)
from pcdc.runtime import setup_runtime
from scripts.levir_tail_risk_control import LimitedLoader, SingleBatch, state_digest
from scripts.r1e_evaluation import check_declared_replication_delta
from scripts.levir_tail_risk_supervision import now, sha
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'configs/b0e_replication_20261026.yaml'


def load_study(path=STUDY):
    path = Path(path).resolve()
    if path != STUDY:
        raise ValueError('Only the frozen missing-control study is admitted')
    v = yaml.safe_load(path.read_text(encoding='utf-8'))
    expected = dict(schema_version=1, kind='missing_clean_exposure_matched_replications',
        seeds=[43,44], epochs=100, attempts_per_run=178000, pair_views_per_run=1424000,
        pilot_steps=32, pilot_order=[43,44,43], evaluation_reserve_seconds=3600,
        single_training_limit_seconds=86400, pilot_wall_limit_seconds=900,
        test_or_martigny_access=False, automatic_retry=False)
    if any(v.get(k) != x for k,x in expected.items()):
        raise ValueError('Frozen study scope/budget changed')
    refs = {s:load_robust_config(ROOT/v['runs'][str(s)]['training_config']) for s in v['seeds']}
    reference = load_robust_config(ROOT/v['reference_training'])
    for seed,ref in refs.items():
        check_declared_replication_delta(ref,reference,seed=seed,train_stream_seed=310902+seed)
        if ref.mode != 'b0e' or ref.base.run_test_after_training:
            raise ValueError('Clean two-view development training required')
        original = load_robust_config(ROOT/f'configs/r1_b1_seed{seed}.yaml')
        # Within seed the only scientific change is the secondary clean view.
        a,b = copy.deepcopy(dict(ref.values)),copy.deepcopy(dict(original.values))
        for item in (a,b):
            for key in ('name','mode','output_dir'):
                item.pop(key)
            item['views'].pop('secondary')
        if a != b or ref.base.values != original.base.values:
            raise ValueError('Within-seed scientific settings differ beyond B0-E view')
    return v,refs,ROOT/v['output_dir']


def source_files():
    files = dict(robust_source_files())
    for name in ('scripts/complete_b0e_replication.py','scripts/b0e_replication_supervision.py',
                 'scripts/levir_tail_risk_control.py','scripts/levir_tail_risk_supervision.py',
                 'scripts/r1e_evaluation.py','scripts/temperature_guard.py','scripts/windows_shared_json.py'):
        files[name] = ROOT/name
    return files


def verify(bound):
    for name,expected in bound.items():
        if sha(name) != expected:
            raise ValueError('Sealed input changed: '+name)


def charge_cpu(identifier,seconds,basis):
    v,_,_ = load_study()
    path=ROOT/v['shared_ledger']; ledger=read(path)
    if any(e.get('status')=='running' for e in ledger['entries']):
        raise RuntimeError('Cannot revise ledger during an active reservation')
    if any(e.get('id')==identifier for e in ledger['entries']):
        raise ValueError('Duplicate charge identity')
    if seconds < 0 or not torch.isfinite(torch.tensor(seconds)):
        raise ValueError('Invalid charge')
    ledger['entries'].append(dict(id=identifier,kind='b0e_replication_engineering',
        status='completed',charged_seconds=float(seconds),basis=basis,research_training=False))
    ledger['total_charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
    atomic(path,ledger)


def build_plan(ref,base,history):
    """Reconstruct all old B1 streams and paired clean streams without pixels."""
    if len(history)!=100 or len(base)!=7120:
        raise ValueError('Expected complete original training history/data')
    g=torch.Generator().manual_seed(ref.base.values['experiment']['seed'])
    loader=DataLoader(range(len(base)),batch_size=4,shuffle=True,num_workers=0,generator=g)
    rows=[];prefix=hashlib.sha256();first=[]
    for epoch in range(1,101):
        old,clean=hashlib.sha256(),hashlib.sha256()
        for step,indices in enumerate(loader,1):
            ids=[];instances=[]
            for index in indices.tolist():
                ri,x,y=base.tiles[index]
                sid=f'train/{base.records[ri].name}@x{x}_y{y}'
                ids.append(sid)
                instances.append(canonical(sample_training_instance(ref.protocol,sid,epoch=epoch,view=0)))
            _update_stream_hash(old,ids,instances)
            _update_stream_hash(clean,ids,['clean']*len(ids))
            if epoch==1 and step<=32:
                _update_stream_hash(prefix,ids,['clean']*len(ids));first.extend(ids)
        h=history[epoch-1]
        if h['epoch']!=epoch or old.hexdigest()!=h['train']['exposure_stream_sha256']:
            raise ValueError(f'Original B1 source/instance order mismatch at epoch{epoch}')
        rows.append(dict(epoch=epoch,clean_exposure_sha256=clean.hexdigest(),
                         original_b1_exposure_sha256=old.hexdigest()))
    return dict(epochs=rows,first128_sample_ids=first,
                pilot_clean_stream_sha256=prefix.hexdigest())


def prepare():
    v,refs,out=load_study()
    if out.exists():
        raise FileExistsError('Preserve existing preparation')
    out.mkdir(parents=True)
    audit=audit_data(refs[43].base,full=True)
    if set(audit['splits'])!={'train','val'}:
        raise ValueError('Development-only audit required')
    plans={};bound={str(p):sha(p) for p in source_files().values()}
    config_paths=[STUDY,ROOT/v['protocol'],ROOT/'configs/b0e_replication_runtime_20261026.yaml',
                  ROOT/'configs/b0e_replication_smoke_20261026.yaml']
    for seed,ref in refs.items():
        old=ROOT/v['runs'][str(seed)]['reference_b1']
        if robust_source_hashes()!=read(old/'source_sha256.json') or audit!=read(old/'data_audit.json'):
            raise ValueError('Old B1 data/core identity mismatch')
        if read(old/'status.json')['initial_model_tensor_sha256']!=v['runs'][str(seed)]['initialization_sha256']:
            raise ValueError('Original initialization metadata differs')
        history=read(old/'history.json')
        plans[str(seed)]=build_plan(ref,LEVIRDataset(ref.base,'train'),history)
        config_paths.extend([ref.source,*[ref.paths[k] for k in ('base_config','protocol_config','severity_review')],
                             *[old/n for n in ('history.json','status.json','data_audit.json','source_sha256.json')]])
    for p in config_paths:
        bound[str(p)]=sha(p)
    atomic(out/'data_audit.json',audit)
    atomic(out/'stream_plan.json',plans)
    for name,p in source_files().items():
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in (out/'data_audit.json',out/'stream_plan.json'):
        bound[str(p)]=sha(p)
    result=dict(status='prepared_identity_and_200_epoch_streams_verified',created_utc=now(),
                bound_sha256=bound,raw_test_accessed=False,original_core_sha256=robust_source_hashes())
    atomic(out/'prepared.json',result)
    return {k:x for k,x in result.items() if k!='bound_sha256'}


def synthetic_batch(n,size):
    a,b=torch.rand(n,3,size,size),torch.rand(n,3,size,size)
    return dict(clean_a=a,clean_b=b,secondary_a=a.clone(),secondary_b=b.clone(),
        mask=(torch.rand(n,1,size,size)>.8).float(),sample_id=[str(i) for i in range(n)],
        secondary_kind=['clean']*n,secondary_level=torch.zeros(n,dtype=torch.int64),
        secondary_direction=['clean']*n,secondary_target_date=['none']*n,
        secondary_spatial_mode=['none']*n,instance_json=['clean']*n)


def epoch_run(ref,model,loader,device,opt,scaler,guard,amp=True):
    if scaler is None and not amp:
        # The archived runner reads the scaler even for an explicit FP32 run.
        # A disabled scaler preserves CPU FP32; it does not enable CUDA/AMP.
        scaler=torch.cuda.amp.GradScaler(enabled=False)
    return run_two_view_epoch(model,loader,ChangeDetectionLoss(**ref.base.values['loss']),device,
        threshold=.5,consistency_kind='none',consistency_weight=0.,amp=amp,
        memory_format='channels_last' if amp else 'contiguous',optimizer=opt,scaler=scaler,
        grad_clip_norm=ref.base.values['train']['grad_clip_norm'],guard_callback=guard)


def smoke():
    _,refs,out=load_study();cfg=yaml.safe_load((ROOT/'configs/b0e_replication_smoke_20261026.yaml').read_text())
    if (cfg['kind'],cfg['device'],cfg['amp'])!=('synthetic_engineering_only','cpu',False):
        raise ValueError('Explicit CPU synthetic smoke only')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);model=build_model(refs[43].base).cpu()
    before=tensor_state_sha256(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.001)
    report=epoch_run(refs[43],model,SingleBatch(synthetic_batch(cfg['source_batch_size'],cfg['image_size'])),
                     torch.device('cpu'),opt,None,lambda:None,amp=False)
    if report['optimizer_steps']!=1 or before==tensor_state_sha256(model.state_dict()):
        raise ValueError('Smoke did not perform one real model update')
    result=dict(status='passed_synthetic_engineering_only',device='cpu',amp=False,
                optimizer_steps=1,research_result=False)
    atomic(out/'smoke.json',result);return result


def budget_estimate(rows,remaining,reserve=3600):
    if len(rows)!=3 or [r['seed'] for r in rows]!=[43,44,43]:
        raise ValueError('Complete frozen pilot order required')
    # Use the slower repeated43 timing; do not select the faster one.
    times={43:max(r['seconds_per_step_after_first'] for r in rows if r['seed']==43),
           44:rows[1]['seconds_per_step_after_first']}
    if any(t<=0 or not torch.isfinite(torch.tensor(t)) for t in times.values()):
        raise ValueError('Invalid measured timing')
    predicted={str(s):(t*178000+6000)*1.25 for s,t in times.items()}
    return dict(predicted_training_seconds=predicted,evaluation_reserve_seconds=reserve,
        available_shared_seconds=remaining,
        passed=all(x<=86400 for x in predicted.values()) and sum(predicted.values())+reserve<=remaining)


def pilot(session,runtime,guard):
    v,refs,out=load_study();verify(read(out/'engineering_authorization.json')['bound_sha256'])
    plan=read(out/'stream_plan.json');rows=[];boundary43=None
    for ordinal,seed in enumerate(v['pilot_order']):
        ref=refs[seed];device=setup_runtime(ref.base)
        ds=TwoViewTrainingDataset(LEVIRDataset(ref.base,'train'),ref.protocol,'clean');ds.set_epoch(1)
        loader=make_loader(ds,ref.base,training=True)
        model=move_model(build_model(ref.base),device,'channels_last')
        if tensor_state_sha256(model.state_dict())!=v['runs'][str(seed)]['initialization_sha256']:
            raise ValueError('Pilot initialization differs from original B1')
        s=ref.base.values['train'];opt=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],betas=tuple(s['adam_betas']),weight_decay=s['weight_decay'])
        scaler=torch.cuda.amp.GradScaler(enabled=True);limited=LimitedLoader(loader,32)
        completion=[]
        class Timed:
            def __len__(self):return len(limited)
            def __iter__(self):
                for batch in limited:
                    yield batch
                    completion.append(time.monotonic())
        report=epoch_run(ref,model,Timed(),device,opt,scaler,guard)
        if report['exposure_stream_sha256']!=plan[str(seed)]['pilot_clean_stream_sha256']:
            raise ValueError('Actual pilot stream differs from paired frozen plan')
        state=lambda:dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=_rng_state(loader))
        boundary=state_digest(state())
        if ordinal==0:boundary43=boundary
        if ordinal==2 and boundary!=boundary43:
            raise ValueError('Independent43 full-state repeat differs')
        row=dict(seed=seed,seconds_per_step_after_first=(completion[-1]-completion[0])/31,
                 boundary_state_sha256=boundary,attempts=32)
        if ordinal==2:
            batch=next(limited.iterator);saved=copy.deepcopy(state())
            epoch_run(ref,model,SingleBatch(batch),device,opt,scaler,guard);expected=state_digest(state())
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);_restore_rng(saved['rng'],loader)
            epoch_run(ref,model,SingleBatch(batch),device,opt,scaler,guard)
            if state_digest(state())!=expected:
                raise ValueError('Exact next-update restore differs')
            row['next_update_state_sha256']=expected
        rows.append(row)
        limited.iterator._shutdown_workers()
        del model,opt,scaler,loader,ds,limited
        gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed_engineering_only',rows=rows,repeat_full_state_exact=True,
                next_update_full_state_exact=True,physical_training_attempts=98,physical_pair_views=784,
                research_optimizer_updates=0,raw_validation_scored=False)
    atomic(session/'pilot_verified.json',result);return result


def check_history(history,plan,status,initial):
    if len(history)!=100 or status.get('status')!='completed' or status.get('test_evaluated') is not False:
        raise ValueError('Completed development-only100epoch run required')
    if status['initial_model_tensor_sha256']!=initial:
        raise ValueError('Actual training initialization mismatch')
    for i,row in enumerate(history):
        t=row['train']
        if (row['epoch']!=i+1 or t['source_samples']!=7120 or t['pair_views']!=14240 or
            t['optimizer_steps']+t['amp_skipped_steps']!=1780 or
            t['exposure_stream_sha256']!=plan['epochs'][i]['clean_exposure_sha256']):
            raise ValueError('Actual complete clean exposure/order/update accounting mismatch')
    best=max(range(100),key=lambda i:history[i]['val']['metrics']['f1'])
    if status['best_epoch']!=best+1 or status['best_val_f1']!=history[best]['val']['metrics']['f1']:
        raise ValueError('First highest cleanF1 checkpoint selection mismatch')
    return dict(status='passed_saved_training_accounting',best_epoch=best+1,
        best_val_f1=status['best_val_f1'],optimizer_steps=sum(r['train']['optimizer_steps'] for r in history),
        amp_skipped_steps=sum(r['train']['amp_skipped_steps'] for r in history),
        attempts=178000,pair_views=1424000)


def train(session,runtime,guard):
    v,refs,out=load_study();verify(read(out/'training_authorization.json')['bound_sha256'])
    plan=read(out/'stream_plan.json');results={}
    for seed,ref in refs.items():
        started=time.monotonic();checks=0;last_write=-10.
        def check():
            nonlocal checks,last_write
            guard();checks+=1;elapsed=time.monotonic()-started
            if elapsed>v['single_training_limit_seconds']:
                raise RuntimeError('Per-run original24hour cap reached')
            if elapsed-last_write>=2:
                atomic(session/'progress.json',dict(stage='original_b0e_training_batch_checks',seed=seed,
                    guard_checkpoints=checks,run_elapsed_seconds=elapsed,updated_utc=now(),
                    note='Batch callback progress; committed optimizer counts are in run status/history.'))
                last_write=elapsed
        check()
        # The archived function, numerical operations, output flavor and source
        # fingerprints are unchanged. The callback only records/protects progress.
        result=train_robust(ref,guard_callback=check)
        verified=check_history(read(ref.output_dir/'history.json'),plan[str(seed)],result,
                               v['runs'][str(seed)]['initialization_sha256'])
        if read(ref.output_dir/'data_audit.json')!=read(out/'data_audit.json'):
            raise ValueError('Actual training data identity changed')
        verified.update(seed=seed,run_wall_seconds=time.monotonic()-started,
                        checkpoint_sha256={n:sha(ref.output_dir/n) for n in ('best.pt','last.pt')})
        atomic(ref.output_dir/'execution_completion.json',verified);results[str(seed)]=verified
        print(f'b0e replication seed{seed} complete; saved order/accounting verified',flush=True)
        gc.collect();torch.cuda.empty_cache()
    result=dict(status='completed',runs=results,research_optimizer_updates=sum(r['optimizer_steps'] for r in results.values()))
    atomic(out/'training_completed.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','smoke','charge'])
    parser.add_argument('--seconds',type=float);parser.add_argument('--id');parser.add_argument('--basis')
    parser.add_argument('--engineering-revision',choices=['initial','cpu_r1'],default='initial')
    a=parser.parse_args();started=time.monotonic()
    if a.engineering_revision!='initial' and a.action!='smoke':
        raise ValueError('CPU smoke revision only; no training retry')
    if a.action=='charge':charge_cpu(a.id,a.seconds,a.basis);result={'status':'charged'}
    else:
        try:result=globals()[a.action]()
        finally:charge_cpu('b0e_replication_'+a.action+'_'+a.engineering_revision+'_20261026',time.monotonic()-started+1,
                          'Full CPU/engineering process wall plus1second startup/write allowance, including any failed attempt.')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
