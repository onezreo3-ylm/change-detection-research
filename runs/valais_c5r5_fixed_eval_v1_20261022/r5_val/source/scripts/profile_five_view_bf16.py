"""Finite preparation, real CUDA admission and serial supervision for C5/R5."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import yaml

from pcdc.config import load_config, protocol_signature
from pcdc.data import audit_data, file_sha256
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.metrics import BinaryMetrics
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256, _rng_state, _restore_rng
from pcdc.runtime import environment_record, save_checkpoint, setup_runtime, write_json
from scripts import five_view_baselines as f
from scripts import five_view_bf16 as bf16
from scripts.c5r5_40ep_identity import load_study
from scripts import evaluate_coupling_p0 as p0
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks


def estimate_budget(timings,v,pilot_seconds=0):
    mean=statistics.mean(timings);debit=v['_prior_debit']+pilot_seconds
    worker=mean*110280*1.25+40*60
    return {'steady_mean_seconds_per_step':mean,'engineering_seconds_debited':debit,
        'projected_training_and_pilot_seconds':worker+debit,'projected_worker_seconds_including_margin':worker,
        'within_frozen_budget':worker+debit<=86400,'maximum_role_seconds':86400}


def replay_update(model,optimizer,batch,loader,loss_fn,device,cfg,output):
    """Exactly compare the next real CUDA update after a weights-only safe reload."""
    state={'format_version':1,'kind':'smoke','engineering_only':True,
        'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
        'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':_rng_state(loader)}
    save_checkpoint(output/'replay_before.pt',state)
    losses,norm,_,_=bf16.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
    first={'losses':losses,'norm':norm,'model':tensor_state_sha256(model.state_dict()),
        'optimizer':f.nested_fingerprint(optimizer.state_dict()),'rng':f.nested_fingerprint(_rng_state(loader))}
    loaded=torch.load(output/'replay_before.pt',map_location='cpu',weights_only=True)
    # Reuse the same allocated model; restore tensors and the complete optimizer/RNG state.
    # No new CUDA kernel shape, stream, device, batch or stochastic approximation is introduced.
    model.load_state_dict(loaded['model'],strict=True)
    optimizer.load_state_dict(loaded['optimizer']);_restore_rng(loaded['rng'],loader)
    second_losses,second_norm,_,_=bf16.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
    second={'losses':second_losses,'norm':second_norm,'model':tensor_state_sha256(model.state_dict()),
        'optimizer':f.nested_fingerprint(optimizer.state_dict()),'rng':f.nested_fingerprint(_rng_state(loader))}
    if first!=second: raise ValueError('Real next CUDA update failed exact model/optimizer/loss/RNG replay')
    return {'exact_model_optimizer_loss_rng_replay':True,'physical_updates':2,'result':first,
        'safe_saved_state_sha256':file_sha256(output/'replay_before.pt')}


def pilot(path):
    source=Path(path).resolve();settings=yaml.safe_load(source.read_bytes())
    expected={'schema_version':1,'parent_config':'valais_c5r5_40ep_seed42_v1_20261020.yaml',
        'output_dir':'../runs/valais_c5r5_bf16_pilot_v1_20261020','maximum_seconds_per_role':600,
        'train_autocast_dtype':'bfloat16','loss_dtype':'float32','parameter_optimizer_dtype':'float32',
        'validation_dtype':'float32','grad_scaler':False,'engineering_only':True,'research_training_authorized':False}
    if settings!=expected:raise ValueError('Unreviewed BF16 engineering scope')
    _,v,paths,_=load_study(source.parent/settings['parent_config'])
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Actual device has no BF16 support')
    proof=f.read(paths['output_dir']/'authorization.json')
    for name,sha in proof['bound_sha256'].items():
        if '/prepared_v2/' in name.replace(chr(92),'/') or Path(name).suffix=='.pt':continue
        p0.checked_hash(name,sha)
    prior=f.read(f.ROOT/'runs/valais_c5r5_speed_pilot_v1_20261020/report.json')
    if prior['status']!='completed_bounded_speed_pilot':raise ValueError('Previous performance work has no cost receipt')
    v=dict(v,pilot_max_seconds_per_role=600)
    output=(source.parent/settings['output_dir']).resolve();output.mkdir(exist_ok=False)
    metadata=f.read(paths['output_dir']/'metadata.json')
    report={'status':'incomplete','engineering_only':True,'score_selection_performed':False,
        'test_accessed':False,'val_model_inference':False,'roles':{},'physical_optimizer_steps':0}
    try:
        for role in f.ROLES:
            row=output/role;row.mkdir()
            v['_prior_debit']=prior['roles'][role]['total_engineering_debit_seconds']
            cfg=load_config(paths['output_dir']/role/'base_config.yaml')
            guard=EvaluationTemperatureGuard(load_guard_config(paths['guard_config']),row)
            guard.state.update(training_started=True,execution='bf16_five_view_engineering_only')
            started=time.monotonic();physical_steps=0
            try:
                guard(force=True);device=setup_runtime(cfg)
                model=move_model(build_model(cfg,initialize_encoder=False),device,'channels_last').float()
                initial=tensor_state_sha256(model.state_dict())
                if initial!=v['initial_model_tensor_sha256']: raise ValueError('Pilot initialization changed')
                optimizer=f.optimizer_for(model,cfg);loss_fn=ChangeDetectionLoss(**cfg.values['loss'])
                dataset,sampler,loader=f.make_train_loader(metadata,v,metadata['corruptions'],role)
                iterator=iter(loader)
                timings=[];source_batches=[];inst_h,pixel_h,used_h=(hashlib.sha256() for _ in range(3))
                metrics=BinaryMetrics(.5);combined=BinaryMetrics(.5)
                torch.cuda.reset_peak_memory_stats(0)
                first_batch=None;checks={}
                for step in range(1,v['pilot_steps_per_role']+1):
                    if time.monotonic()-started>=v['pilot_max_seconds_per_role']: raise RuntimeError('Pilot budget exhausted')
                    tick=time.monotonic();batch=next(iterator);guard()
                    if step==1:
                        model.eval();a,b,target=f.transfer(batch,device)
                        before=tensor_state_sha256(model.state_dict())
                        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                            z=model(a[:1],b[:1])['logits'];repeat=model(a[:1],b[:1])['logits'];swap=model(b[:1],a[:1])['logits']
                        if not torch.equal(z,repeat) or not torch.allclose(z,swap,atol=1e-5,rtol=1e-5):
                            raise ValueError('Real initial train repeat/swap failed')
                        if before!=tensor_state_sha256(model.state_dict()): raise ValueError('Eval self-check changed state')
                        checks={'repeat_bitwise_equal':True,'temporal_swap_allclose':True,'model_state_unchanged':True}
                        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
                            bz=model(a[:1],b[:1])['logits'];br=model(a[:1],b[:1])['logits'];bs=model(b[:1],a[:1])['logits']
                        if bz.dtype!=torch.bfloat16 or not torch.equal(bz,br) or not torch.allclose(bz,bs,atol=1e-3,rtol=1e-3):
                            raise ValueError('BF16 repeat/swap engineering self-check failed')
                        if before!=tensor_state_sha256(model.state_dict()):raise ValueError('BF16 eval mutated state')
                        checks.update(bf16_repeat_bitwise=True,bf16_swap_allclose=True,fp32_validation_forward_finite=bool(torch.isfinite(z).all()))
                        del bz,br,bs,z,repeat,swap,a,b,target
                        model.train()
                        tick=time.monotonic()
                    before_bn=f.bn_counts(model)
                    losses,norm,logits,target=bf16.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
                    physical_steps+=1
                    n=len(target);metrics.update(logits[:n],target);combined.update(logits,torch.cat([target]*5))
                    torch.cuda.synchronize(device)
                    elapsed=time.monotonic()-tick
                    if any(f.bn_counts(model)[k]-x!=1 for k,x in before_bn.items()): raise ValueError('Not one joint BN update')
                    timings.append(elapsed);source_batches.append(list(batch['sample_id']))
                    for sid,inst,pixels,used in zip(batch['sample_id'],batch['instance_json'],batch['candidate_pixels'],batch['used_pixels']):
                        f.stream_update(inst_h,sid,inst);f.stream_update(pixel_h,sid,pixels);f.stream_update(used_h,sid,used)
                    guard()
                    if step%8==0: print(f'{role} pilot {step}/32 step_seconds={elapsed:.4f}',flush=True)
                next_batch=next(iterator);guard(force=True)
                replay=replay_update(model,optimizer,next_batch,loader,loss_fn,device,cfg,row)
                physical_steps+=2
                tail_indices=sampler.batches()[-1]
                tail=torch.utils.data.default_collate([dataset[i] for i in tail_indices])
                if len(tail['sample_id'])!=1: raise ValueError('Expected the exact real single-source tail')
                bf16.update(model,optimizer,loss_fn,tail,device,cfg.values['train']['grad_clip_norm']);physical_steps+=1
                guard(force=True)
                if any(t.dtype!=torch.float32 for t in model.parameters()):raise ValueError('Parameters ceased to be FP32')
                if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
                    raise FloatingPointError('Nonfinite engineering model state')
                if time.monotonic()-started>=v['pilot_max_seconds_per_role']: raise RuntimeError('Pilot budget exhausted')
                estimates=estimate_budget(timings[v['pilot_warmup_steps']:],v,time.monotonic()-started)
                result={'status':'passed_bf16_engineering_pilot','training_autocast_dtype':'bfloat16','loss_dtype':'float32','parameter_optimizer_dtype':'float32','validation_dtype':'float32','grad_scaler':None,'initial_model_tensor_sha256':initial,
                    'physical_optimizer_steps':physical_steps,'optimizer_steps_retained_for_research':0,
                    'source_batch_size':4,'view_order':list(f.VIEWS),'view_weights':v['view_weights'],
                    'self_checks':checks,'replay':replay,'single_source_tail_passed':True,
                    'source_prefix_sha256':f.digest(source_batches),'instance_prefix_sha256':inst_h.hexdigest(),
                    'candidate_pixel_prefix_sha256':pixel_h.hexdigest(),'used_pixel_prefix_sha256':used_h.hexdigest(),
                    'timings_seconds':timings,'budget_projection':estimates,'elapsed_seconds':time.monotonic()-started,
                    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0),
                    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(0),'environment':environment_record(cfg,device)}
                guard.finish(result)
                result['elapsed_seconds']=time.monotonic()-started
                if result['elapsed_seconds']>=600: raise RuntimeError('Pilot exceeded its complete budget')
                result['budget_projection']=estimate_budget(timings[v['pilot_warmup_steps']:],v,result['elapsed_seconds'])
                result['maximum_observed_temperature_c']=max(x['temperature_c'] for x in guard.state['observations'] if x['temperature_c'] is not None)
                write_json(row/'report.json',result);report['roles'][role]=result
                report['physical_optimizer_steps']+=physical_steps
                del iterator,loader,model,optimizer,logits,target,batch,next_batch,tail
                torch.cuda.empty_cache()
            except BaseException as error:
                guard.abort(error)
                write_json(row/'failure.json',{'status':'failed_preserve','error':f'{type(error).__name__}: {error}',
                    'physical_optimizer_steps':physical_steps,'automatic_retry':False,'elapsed_seconds':time.monotonic()-started})
                raise
        a,b=report['roles']['c5'],report['roles']['r5']
        fields=('source_prefix_sha256','instance_prefix_sha256','candidate_pixel_prefix_sha256')
        if any(a[k]!=b[k] for k in fields) or a['used_pixel_prefix_sha256']==b['used_pixel_prefix_sha256']:
            raise ValueError('Actual matched candidate streams or distinct role inputs failed')
        report.update(status='completed_bf16_engineering_proposal_not_authorized',matched_candidate_streams_verified=True,
            computationally_admitted=all(r['budget_projection']['within_frozen_budget'] for r in report['roles'].values()))
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        write_json(output/'report.json',report)
    return {'status':report['status'],'computationally_admitted':report['computationally_admitted'],
        'physical_optimizer_steps':report['physical_optimizer_steps'],
        'projected_hours':{r:x['budget_projection']['projected_training_and_pilot_seconds']/3600 for r,x in report['roles'].items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(pilot(args.config)))
