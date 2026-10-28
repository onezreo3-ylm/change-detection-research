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
from scripts import evaluate_coupling_p0 as p0
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks


def prepared_metadata(paths): return f.read(paths['output_dir']/'metadata.json')


def prepare(path):
    source,v,paths,base=f.load_study(path)
    output=paths['output_dir']
    if output.exists(): raise FileExistsError(output)
    for key in ('data_audit','label_report'): p0.checked_hash(paths[key],v[key+'_sha256'])
    prior=f.read(paths['prior_screen_closeout'])
    if prior['status']!='completed_authorized_finite_feasibility_screen' or prior['total_actual_engineering_optimizer_steps']!=1:
        raise ValueError('Prior finite screen has not closed correctly')
    p0.checked_hash(prior['report'],prior['report_sha256'])
    prior_seal=f.read(paths['prior_screen_closeout'].parent/'authorization.json')
    p0.checked_hash(paths['base_config'],prior_seal['bound_sha256'][str(paths['base_config'])])
    manifest=f.read(paths['p0_manifest'])
    p0.validate_manifest(manifest);p0.validate_review(manifest,paths['p0_manifest'],paths['p0_review'])
    expected=f.read(paths['data_audit'])
    audit=audit_data(base,full=True,splits=('train','val'))
    if audit!=expected: raise ValueError('Current train/val content differs from archived data audit')
    labels=f.read(paths['label_report'])
    positives={r['name']:r['positive_pixels'] for r in labels['train']['positive_manifest']}
    if labels['test_accessed'] or len(positives)!=1046 or len(audit['splits']['train']['file_sha256'])!=11025:
        raise ValueError('Unexpected original train partition')
    corruptions=yaml.safe_load(paths['corruption_config'].read_text(encoding='utf-8'))['corruptions']
    original_cfg=Path(manifest['config_path'])
    original_settings=yaml.safe_load(original_cfg.read_text(encoding='utf-8'))
    original_corruptions=yaml.safe_load((original_cfg.parent/original_settings['corruption_config']).resolve().read_text(encoding='utf-8'))['corruptions']
    if corruptions!=original_corruptions: raise ValueError('Frozen operator parameter table changed')
    metadata={'audit':audit,'base_values':base.values,'base_config_source':str(base.source),'data_root':str(base.data_root),
        'positive_pixels':positives,'corruptions':corruptions,'test_accessed':False,
        'intentional_changes':['uniform_local_epoch_source_order','five_view_forward_and_weighted_loss',
            'four_global_photometric_families_levels_1_3_5_with_replacement_pairing']}
    names=sorted(audit['splits']['train']['file_sha256'])
    sampler=f.UniformEpochSampler(names,v['source_seed_template'])
    plan=[]
    for epoch in range(1,101):
        sampler.set_epoch(epoch);plan.append(sampler.summary(v,corruptions))
        if epoch%10==0: print(f'sealed-source/instance plan {epoch}/100',flush=True)
    output.mkdir(exist_ok=False)
    write_json(output/'metadata.json',metadata);write_json(output/'sampling_plan.json',plan)
    implementations={name:file_sha256(p) for name,p in f.implementation_files().items()}
    for role in f.ROLES:
        row=output/role;row.mkdir()
        values=copy.deepcopy(base.values)
        values['experiment'].update(name=v['name']+'_'+role,output_dir=str(row))
        values['data']['root']=str(base.data_root)
        values['model']['weights_cache']=str(base.path(base.values['model']['weights_cache']))
        with (row/'base_config.yaml').open('x',encoding='utf-8') as stream: yaml.safe_dump(values,stream,sort_keys=False)
        identity={'training_flavor':f.FLAVOR,'role':role,'study_config_sha256':file_sha256(source),
            'implementation_sha256':implementations,'base_protocol_signature':protocol_signature(values),
            'base_training_values':values,'data_audit_sha256':v['data_audit_sha256'],
            'data_files_sha256':f.digest({k:x['file_sha256'] for k,x in audit['splits'].items()}),
            'manifests_sha256':f.digest({k:{kk:xx for kk,xx in x.items() if kk!='file_sha256'} for k,x in audit['splits'].items()}),
            'sampling_plan_sha256':file_sha256(output/'sampling_plan.json'),
            'initial_model_tensor_sha256':v['initial_model_tensor_sha256'],'selection':'first_highest_clean_val_f1'}
        write_json(row/'identity.json',identity)
    deps=[source,*[p for k,p in paths.items() if k not in ('output_dir','python')],
        output/'metadata.json',output/'sampling_plan.json',
        *[p for role in f.ROLES for p in ((output/role/'identity.json'),(output/role/'base_config.yaml'))],
        *f.implementation_files().values(),f.ROOT/'tests/test_five_view_baselines.py']
    for split,row in audit['splits'].items():
        for name,hashes in row['file_sha256'].items():
            for key,folder in base.values['data']['folders'].items():
                target=(base.data_root/split/folder/name).resolve()
                deps.append(target)
    bound={str(p.resolve()):file_sha256(p) for p in set(deps)}
    write_json(output/'prepared_identity.json',{'status':'prepared_not_admitted','bound_sha256':bound,
        'research_training_started':False,'test_accessed':False,'maximum_training_seconds_per_role':86400})
    return {'status':'prepared_c5r5_ordinary_control','sealed_inputs':len(bound),'source_plan_epochs':100,
        'updates_per_role':275700,'pair_views_per_role':5512500}


def estimate_budget(step_seconds,settings,pilot_seconds=0.):
    if not step_seconds or any(not 0<x<1800 for x in step_seconds): raise ValueError('Invalid real step timings')
    if not isinstance(pilot_seconds,(int,float)) or not 0<=pilot_seconds<1800:
        raise ValueError('Invalid engineering time debit')
    mean=statistics.mean(step_seconds);median=statistics.median(step_seconds)
    estimate=mean*275700*settings['pilot_training_time_multiplier']+100*settings['pilot_extra_seconds_per_epoch']
    return {'steady_mean_seconds_per_step':mean,'steady_median_seconds_per_step':median,
        'projected_worker_seconds_including_margin':estimate,
        'engineering_seconds_debited':pilot_seconds,
        'projected_training_and_pilot_seconds':estimate+pilot_seconds,
        'maximum_worker_seconds':settings['maximum_training_seconds_per_role'],
        'within_frozen_budget':estimate+pilot_seconds<=settings['maximum_training_seconds_per_role'],
        'interpretation':'32-step engineering projection with fixed 25% margin; not a guarantee of 100-epoch stability.'}


def replay_update(model,optimizer,batch,loader,loss_fn,device,cfg,output):
    """Exactly compare the next real CUDA update after a weights-only safe reload."""
    state={'format_version':1,'kind':'smoke','engineering_only':True,
        'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
        'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':_rng_state(loader)}
    save_checkpoint(output/'replay_before.pt',state)
    losses,norm,_,_=f.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
    first={'losses':losses,'norm':norm,'model':tensor_state_sha256(model.state_dict()),
        'optimizer':f.nested_fingerprint(optimizer.state_dict()),'rng':f.nested_fingerprint(_rng_state(loader))}
    loaded=torch.load(output/'replay_before.pt',map_location='cpu',weights_only=True)
    # Reuse the same allocated model; restore tensors and the complete optimizer/RNG state.
    # No new CUDA kernel shape, stream, device, batch or stochastic approximation is introduced.
    model.load_state_dict(loaded['model'],strict=True)
    optimizer.load_state_dict(loaded['optimizer']);_restore_rng(loaded['rng'],loader)
    second_losses,second_norm,_,_=f.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
    second={'losses':second_losses,'norm':second_norm,'model':tensor_state_sha256(model.state_dict()),
        'optimizer':f.nested_fingerprint(optimizer.state_dict()),'rng':f.nested_fingerprint(_rng_state(loader))}
    if first!=second: raise ValueError('Real next CUDA update failed exact model/optimizer/loss/RNG replay')
    return {'exact_model_optimizer_loss_rng_replay':True,'physical_updates':2,'result':first,
        'safe_saved_state_sha256':file_sha256(output/'replay_before.pt')}


def pilot(path):
    _,v,paths,_=f.load_study(path)
    f.verify_prepared(paths)
    output=paths['output_dir']/'multi_batch_cuda_pilot';output.mkdir(exist_ok=False)
    metadata=prepared_metadata(paths)
    report={'status':'incomplete','engineering_only':True,'score_selection_performed':False,
        'test_accessed':False,'val_model_inference':False,'roles':{},'physical_optimizer_steps':0}
    try:
        for role in f.ROLES:
            row=output/role;row.mkdir()
            cfg=load_config(paths['output_dir']/role/'base_config.yaml')
            guard=EvaluationTemperatureGuard(load_guard_config(paths['guard_config']),row)
            guard.state.update(training_started=True,execution='multi_batch_five_view_engineering_pilot')
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
                        del z,repeat,swap,a,b,target
                        model.train()
                        tick=time.monotonic()
                    before_bn=f.bn_counts(model)
                    losses,norm,logits,target=f.update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
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
                f.update(model,optimizer,loss_fn,tail,device,cfg.values['train']['grad_clip_norm']);physical_steps+=1
                guard(force=True)
                if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
                    raise FloatingPointError('Nonfinite engineering model state')
                if time.monotonic()-started>=v['pilot_max_seconds_per_role']: raise RuntimeError('Pilot budget exhausted')
                estimates=estimate_budget(timings[v['pilot_warmup_steps']:],v,time.monotonic()-started)
                result={'status':'passed_multi_batch_cuda_pilot','initial_model_tensor_sha256':initial,
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
                if result['elapsed_seconds']>=1800: raise RuntimeError('Pilot exceeded its complete budget')
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
        report.update(status='completed_engineering_admission',matched_candidate_streams_verified=True,
            computationally_admitted=all(r['budget_projection']['within_frozen_budget'] for r in report['roles'].values()))
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        write_json(output/'report.json',report)
    return {'status':report['status'],'computationally_admitted':report['computationally_admitted'],
        'physical_optimizer_steps':report['physical_optimizer_steps'],
        'projected_hours':{r:x['budget_projection']['projected_training_and_pilot_seconds']/3600 for r,x in report['roles'].items()}}


def seal(path,unittest_log=None):
    source,v,paths,_=f.load_study(path)
    proof=f.verify_prepared(paths)
    output=paths['output_dir'];pilot=f.read(output/'multi_batch_cuda_pilot/report.json')
    if pilot['status']!='completed_engineering_admission' or pilot['physical_optimizer_steps']!=70:
        raise ValueError('Complete exact multi-batch pilot required')
    if not pilot['computationally_admitted']:
        write_json(output/'budget_stop.json',{'status':'not_launched_projected_budget_exceeded',
            'user_budget_seconds_per_role':86400,'actual_research_optimizer_steps':0,
            'projected_hours':{r:x['budget_projection']['projected_training_and_pilot_seconds']/3600 for r,x in pilot['roles'].items()},
            'test_accessed':False,'no_automatic_epoch_reduction_or_budget_extension':True})
        return {'status':'not_launched_projected_budget_exceeded'}
    if unittest_log is None: raise ValueError('Explicit passing complete unittest log required')
    log=Path(unittest_log).resolve();raw=log.read_bytes()
    text=raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8-sig')
    import re
    found=re.findall(r'Ran (\d+) tests in',text)
    if not found or int(found[-1])<v['minimum_unittests'] or not text.rstrip().endswith('OK'):
        raise ValueError('Full test suite did not pass')
    bound=dict(proof['bound_sha256'])
    for p in [output/'prepared_identity.json',log,*sorted((output/'multi_batch_cuda_pilot').rglob('*'))]:
        if p.is_file(): bound[str(p.resolve())]=file_sha256(p)
    write_json(output/'authorization.json',{'status':'authorized_matched_ordinary_c5r5_training',
        'human_instruction':'继续；总计最多48小时，两行各24小时','bound_sha256':bound,
        'maximum_training_seconds_per_role':86400,'research_updates_per_role':275700,
        'research_pair_views_per_role':5512500,'test_accessed':False,'new_method_training_authorized':False,
        'unittests':int(found[-1]),'pilot_steps_are_additional_physical_compute':70})
    return {'status':'sealed_and_admitted','unittests':int(found[-1]),'roles':list(f.ROLES)}


def check_completion(paths,role):
    row=paths['output_dir']/role
    status=f.read(row/'status.json');proof=f.read(row/'guard_completion.json')
    if (status['status']!='completed' or status['epoch']!=100 or status['optimizer_steps']!=275700
            or status['pair_views']!=5512500 or status['source_draws']!=1102500
            or status['cumulative_role_seconds_including_pilot']>86400 or status['test_evaluated'] or status['training_flavor']!=f.FLAVOR):
        raise ValueError('Incomplete/overbudget ordinary training')
    for key,name in (('training_status_sha256','status.json'),('guard_state_sha256','temperature_guard_state.json'),
        ('history_sha256','history.json'),('sampling_sha256','sampling_actual.json'),('best_sha256','best.pt'),('last_sha256','last.pt')):
        p0.checked_hash(row/name,proof[key])
    if f.read(row/'sampling_actual.json')['rows']!=f.read(paths['output_dir']/'sampling_plan.json'):
        raise ValueError('Completed actual source/instance streams differ')
    return status


def run(path):
    source,v,paths,_=f.load_study(path)
    f.verify_prepared(paths,sealed=True)
    output=paths['output_dir']
    write_json(output/'owner.json',{'pid':os.getpid(),'creation_ticks':creation_time_ticks(os.getpid()),
        'started_utc':f.now(),'automatic_retry':False,'command':[sys.executable,'-m','scripts.valais_c5r5_study','run','--config',str(source)]})
    child=None;completed=[];role='c5'
    try:
        for role in f.ROLES:
            command=[str(paths['python']),'-u','-m','scripts.five_view_baselines','--config',str(source),'--role',role]
            with (output/f'{role}.stdout.log').open('x',encoding='utf-8') as stdout,(output/f'{role}.stderr.log').open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=f.ROOT,stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                _atomic_json(output/'status.json',{'status':'running','stage':role+'_training','owner_pid':os.getpid(),
                    'child_pid':child.pid,'child_creation_ticks':creation_time_ticks(child.pid),'child_command':command,
                    'completed_roles':completed,'updated_utc':f.now(),'test_accessed':False})
                code=child.wait()
            if code: raise RuntimeError(f'{role} exited {code}; preserve before any explicit recovery')
            checked=check_completion(paths,role)
            write_json(output/f'{role}.exit.json',{'exit_code':0,'completed_utc':f.now(),'status_sha256':file_sha256(output/role/'status.json')})
            completed.append(role)
        histories=[f.read(output/r/'history.json')['rows'] for r in f.ROLES]
        for a,b in zip(*histories):
            if a['train']['candidate_pixel_stream_sha256']!=b['train']['candidate_pixel_stream_sha256']:
                raise ValueError('Actual full candidate pixel streams differ between ordinary controls')
        _atomic_json(output/'status.json',{'status':'completed_training_ready_for_explicit_evaluation',
            'completed_roles':completed,'completed_utc':f.now(),'matched_actual_source_instance_pixels':True,
            'test_accessed':False,'next':'Seal explicit five-view checkpoint adapter; original 45 val conditions then fixed train residual.'})
    except BaseException as error:
        _atomic_json(output/'status.json',{'status':'failed_preserve','stage':role+'_training','error':f'{type(error).__name__}: {error}',
            'owner_pid':os.getpid(),'child_pid':child.pid if child else None,'completed_roles':completed,
            'automatic_retry':False,'updated_utc':f.now(),'test_accessed':False})
        raise


def monitor(path):
    _,v,paths,_=f.load_study(path);output=paths['output_dir']
    result={'study':v['name'],'checked_utc':f.now(),'test_accessed':False}
    if (output/'budget_stop.json').exists(): result['budget_stop']=f.read(output/'budget_stop.json')
    queue=f.read(output/'status.json') if (output/'status.json').exists() else {}
    result['queue']={k:queue[k] for k in ('status','stage','completed_roles','updated_utc','completed_utc','error') if k in queue}
    if (output/'owner.json').exists():
        owner=f.read(output/'owner.json');result['owner']=process_snapshot(owner['pid'],owner['creation_ticks'])
    if queue.get('child_pid') and queue.get('child_creation_ticks'):
        result['worker']=process_snapshot(queue['child_pid'],queue['child_creation_ticks'])
    for role in f.ROLES:
        row=output/role;state={}
        for filename,label in (('status.json','committed'),('progress.json','progress')):
            if (row/filename).exists():
                value=f.read(row/filename)
                state[label]={k:value[k] for k in ('status','epoch','committed_epochs','step','successful_steps_in_epoch',
                    'best_epoch','best_val_f1','updated_utc','elapsed_seconds','error') if k in value}
                state[label+'_mtime_utc']=datetime.fromtimestamp((row/filename).stat().st_mtime,timezone.utc).isoformat()
        for name in ('best.pt','last.pt'):
            if (row/name).exists():state[name+'_mtime_utc']=datetime.fromtimestamp((row/name).stat().st_mtime,timezone.utc).isoformat()
        log=output/f'{role}.stdout.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-1024));state['log_tail']=stream.read().decode('utf-8',errors='replace').splitlines()[-3:]
            state['log_age_seconds']=time.time()-log.stat().st_mtime
            if queue.get('stage')==role+'_training' and state['log_age_seconds']>900:
                state['attention']='No actual log progress for >15 minutes; diagnose without restarting.'
        result[role]=state
    if queue.get('status')=='running' and not result.get('worker',{}).get('owned_active',False):
        result['attention']='Worker no longer active by creation identity/wait/exit status; preserve and inspect.'
    settings=load_guard_config(paths['guard_config']).values
    data=subprocess.run([settings['nvidia_smi'],'--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip().split(',')
    uuid,temp,util,mem=[x.strip() for x in data]
    if uuid!=settings['gpu_uuid']: raise ValueError('Monitor GPU identity changed')
    result['gpu']={'temperature_c':int(temp),'utilization_percent':int(util),'memory_used_mib':int(mem)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','pilot','seal','run','monitor'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--unittest-log',type=Path)
    args=parser.parse_args()
    value=seal(args.config,args.unittest_log) if args.action=='seal' else globals()[args.action](args.config)
    if value is not None: print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__': main()
