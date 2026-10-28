"""Prepare, smoke, seal and supervise the separately authorized 40-epoch study."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import yaml

from pcdc.config import load_config, protocol_signature
from pcdc.data import file_sha256
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256
from pcdc.runtime import setup_runtime, write_json, environment_record
from scripts import five_view_baselines as f
from scripts import evaluate_coupling_p0 as p0
from scripts.c5r5_40ep_identity import (FLAVOR, EPOCHS, UPDATES, PAIR_VIEWS,
    load_study, implementation_files, estimate_budget)
from scripts.evaluate_guarded import EvaluationTemperatureGuard, load_guard_config, _atomic_json
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.valais_c5r5_study import replay_update


def prepare(path):
    source, v, paths, _ = load_study(path)
    output, parent = paths['output_dir'], paths['parent_output_dir']
    if output.exists():
        raise FileExistsError(output)
    parent_proof = f.verify_prepared({'output_dir': parent})
    stop = f.read(parent / 'budget_stop.json')
    pilot = f.read(parent / 'multi_batch_cuda_pilot/report.json')
    if (stop['actual_research_optimizer_steps'] != 0 or (parent / 'authorization.json').exists()
            or pilot['status'] != 'completed_engineering_admission'
            or pilot['physical_optimizer_steps'] != 70 or pilot['computationally_admitted']):
        raise ValueError('100-epoch ancestry or unused research budget changed')
    metadata = f.read(parent / 'metadata.json')
    old_plan = f.read(parent / 'sampling_plan.json')
    sampler = f.UniformEpochSampler(sorted(metadata['audit']['splits']['train']['file_sha256']),
        v['source_seed_template'])
    plan = []
    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch)
        plan.append(sampler.summary(v, metadata['corruptions']))
        if epoch % 10 == 0:
            print(f'40ep source/instance plan {epoch}/40', flush=True)
    if plan != old_plan[:EPOCHS]:
        raise ValueError('40-epoch streams differ from original first 40 plans')
    output.mkdir(exist_ok=False)
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'sampling_plan.json', plan)
    implementations = {k: file_sha256(p) for k, p in implementation_files().items()}
    for role in f.ROLES:
        row = output / role
        row.mkdir()
        cfg = load_config(parent / role / 'base_config.yaml')
        values = copy.deepcopy(cfg.values)
        values['experiment'].update(name=v['name'] + '_' + role, output_dir=str(row))
        values['train']['epochs'] = EPOCHS
        with (row / 'base_config.yaml').open('x', encoding='utf-8') as stream:
            yaml.safe_dump(values, stream, sort_keys=False)
        identity = dict(f.read(parent / role / 'identity.json'), training_flavor=FLAVOR,
            study_config_sha256=file_sha256(source), implementation_sha256=implementations,
            base_protocol_signature=protocol_signature(values), base_training_values=values,
            sampling_plan_sha256=file_sha256(output / 'sampling_plan.json'),
            epochs=EPOCHS, cosine_tmax=EPOCHS,
            parent_identity_sha256=file_sha256(parent / role / 'identity.json'),
            historical_b0e_100ep_is_unmatched_context=True)
        write_json(row / 'identity.json', identity)
    bound = dict(parent_proof['bound_sha256'])
    dependencies = [source, paths['protocol_document'], parent / 'prepared_identity.json',
        parent / 'budget_stop.json', parent / 'admission_audit.json',
        output / 'metadata.json', output / 'sampling_plan.json',
        f.ROOT / 'tests/test_c5r5_40ep.py',
        f.ROOT / 'scripts/launch_valais_c5r5_40ep_20261020.ps1',
        *implementation_files().values(),
        *[p for r in f.ROLES for p in (output/r/'identity.json', output/r/'base_config.yaml')],
        *[p for p in (parent / 'multi_batch_cuda_pilot').rglob('*') if p.is_file()]]
    for p in dependencies:
        bound[str(p.resolve())] = file_sha256(p)
    write_json(output / 'prepared_identity.json', {'status': 'prepared_40ep_not_started',
        'bound_sha256': bound, 'human_instruction': '改为40轮独立对照',
        'parent_budget_stop_preserved': True, 'test_accessed': False})
    return {'status': 'prepared_40ep', 'sealed_inputs': len(bound),
        'epochs': EPOCHS, 'updates_per_role': UPDATES, 'pair_views_per_role': PAIR_VIEWS}


def smoke(path):
    _, v, paths, _ = load_study(path)
    f.verify_prepared(paths)
    output = paths['output_dir'] / 'cuda_smoke'
    output.mkdir(exist_ok=False)
    metadata = f.read(paths['output_dir'] / 'metadata.json')
    parent_pilot = f.read(paths['parent_output_dir'] / 'multi_batch_cuda_pilot/report.json')
    report = {'status': 'incomplete', 'engineering_only': True, 'roles': {},
        'score_selection_performed': False, 'test_accessed': False}
    try:
        for role in f.ROLES:
            started = time.monotonic()
            row = output / role
            row.mkdir()
            guard = EvaluationTemperatureGuard(load_guard_config(paths['guard_config']), row)
            guard.state.update(training_started=True, execution='independent_40ep_cuda_smoke')
            physical_steps = 0
            try:
                guard(force=True)
                cfg = load_config(paths['output_dir'] / role / 'base_config.yaml')
                device = setup_runtime(cfg)
                model = move_model(build_model(cfg, initialize_encoder=False), device, 'channels_last').float()
                if tensor_state_sha256(model.state_dict()) != v['initial_model_tensor_sha256']:
                    raise ValueError('40ep smoke original seed42 initialization changed')
                optimizer = f.optimizer_for(model, cfg)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40,
                    eta_min=cfg.values['train']['min_learning_rate'])
                loss_fn = ChangeDetectionLoss(**cfg.values['loss'])
                _, _, loader = f.make_train_loader(metadata, v, metadata['corruptions'], role)
                iterator = iter(loader)
                batch = next(iterator)
                model.train()
                before = f.bn_counts(model)
                f.update(model, optimizer, loss_fn, batch, device, cfg.values['train']['grad_clip_norm'])
                physical_steps += 1
                if any(f.bn_counts(model)[k] - x != 1 for k, x in before.items()):
                    raise ValueError('40ep smoke joint BN failed')
                scheduler.step()
                next_batch = next(iterator)
                replay = replay_update(model, optimizer, next_batch, loader, loss_fn, device, cfg, row)
                physical_steps += 2
                if any(f.bn_counts(model)[k] - x != 2 for k, x in before.items()):
                    raise ValueError('Replay did not restore exact BN state')
                guard(force=True)
                state = scheduler.state_dict()
                if state['T_max'] != 40 or state['last_epoch'] != 1:
                    raise ValueError('40ep schedule smoke failed')
                guard.finish({'status': 'passed_40ep_cuda_smoke'})
                elapsed = time.monotonic() - started
                if elapsed >= 1800:
                    raise RuntimeError('Finite smoke budget exhausted')
                debit = elapsed + parent_pilot['roles'][role]['elapsed_seconds']
                projection = estimate_budget(parent_pilot['roles'][role]['timings_seconds'][8:], v, debit)
                result = {'status': 'passed_40ep_cuda_smoke', 'physical_optimizer_steps': physical_steps,
                    'optimizer_steps_retained_for_research': 0, 'pair_forward_views': 60,
                    'initial_model_tensor_sha256': v['initial_model_tensor_sha256'],
                    'scheduler': state, 'replay': replay, 'joint_bn_verified': True,
                    'elapsed_seconds': elapsed, 'parent_pilot_seconds_debited': debit-elapsed,
                    'cumulative_engineering_debit_seconds': debit, 'budget_projection': projection,
                    'source_ids': list(batch['sample_id']),
                    'candidate_pixels': list(batch['candidate_pixels']),
                    'used_pixels': list(batch['used_pixels']), 'environment': environment_record(cfg, device)}
                write_json(row / 'report.json', result)
                report['roles'][role] = result
                del iterator, loader, batch, next_batch, model, optimizer, scheduler
                torch.cuda.empty_cache()
                print(f'{role} 40ep smoke passed; projected total {projection["projected_total_role_seconds"]/3600:.3f}h', flush=True)
            except BaseException as error:
                guard.abort(error)
                write_json(row / 'failure.json', {'error': f'{type(error).__name__}: {error}',
                    'physical_optimizer_steps': physical_steps, 'automatic_retry': False,
                    'elapsed_seconds': time.monotonic()-started})
                raise
        a, b = (report['roles'][r] for r in f.ROLES)
        if a['source_ids'] != b['source_ids'] or a['candidate_pixels'] != b['candidate_pixels'] or a['used_pixels'] == b['used_pixels']:
            raise ValueError('40ep actual candidate matching or distinct role input failed')
        report.update(status='passed_independent_40ep_smoke', physical_optimizer_steps=6,
            pair_forward_views=120, research_optimizer_steps=0,
            computationally_admitted=all(x['budget_projection']['within_frozen_budget'] for x in report['roles'].values()))
    finally:
        write_json(output / 'report.json', report)
    return {'status': report['status'], 'computationally_admitted': report['computationally_admitted'],
        'projected_hours': {r:x['budget_projection']['projected_total_role_seconds']/3600 for r,x in report['roles'].items()}}


def passing_test_count(path, minimum):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8-sig')
    found = re.findall(r'Ran (\d+) tests in', text)
    if not found or int(found[-1]) < minimum or not text.rstrip().endswith('OK'):
        raise ValueError('Passing complete test suite required')
    return int(found[-1])


def seal(path, unittest_log):
    _, v, paths, _ = load_study(path)
    proof = f.verify_prepared(paths)
    output = paths['output_dir']
    smoke_report = f.read(output / 'cuda_smoke/report.json')
    if (smoke_report['status'] != 'passed_independent_40ep_smoke'
            or not smoke_report['computationally_admitted'] or smoke_report['physical_optimizer_steps'] != 6):
        raise ValueError('40ep engineering or compute gate failed; no automatic retry')
    count = passing_test_count(unittest_log, v['minimum_unittests'])
    bound = dict(proof['bound_sha256'])
    for p in [Path(unittest_log).resolve(), output/'prepared_identity.json',
              *[p for p in (output/'cuda_smoke').rglob('*') if p.is_file()]]:
        bound[str(p.resolve())] = file_sha256(p)
    write_json(output/'authorization.json', {'status':'authorized_independent_40ep_c5r5',
        'human_instruction':'改为40轮独立对照；总计最多48小时，两行各24小时',
        'bound_sha256':bound, 'training_flavor':FLAVOR, 'epochs':EPOCHS, 'cosine_tmax':EPOCHS,
        'research_updates_per_role':UPDATES, 'research_pair_views_per_role':PAIR_VIEWS,
        'maximum_training_seconds_per_role':86400,
        'engineering_seconds_debited':{r:x['cumulative_engineering_debit_seconds'] for r,x in smoke_report['roles'].items()},
        'unittests':count, 'additional_physical_optimizer_steps':76,
        'additional_pair_forward_views':1496, 'test_accessed':False, 'new_method_training_authorized':False})
    return {'status':'sealed_and_admitted_40ep', 'unittests':count, 'roles':list(f.ROLES)}


def check_completion(paths, role):
    row = paths['output_dir']/role
    status, proof = f.read(row/'status.json'), f.read(row/'guard_completion.json')
    if (status['status']!='completed' or status['epoch']!=EPOCHS or status['optimizer_steps']!=UPDATES
            or status['pair_views']!=PAIR_VIEWS or status['source_draws']!=441000
            or status['cumulative_role_seconds_including_pilot']>86400 or status['test_evaluated']
            or status['training_flavor']!=FLAVOR):
        raise ValueError('Incomplete or overbudget 40ep training')
    for key, name in (('training_status_sha256','status.json'),('guard_state_sha256','temperature_guard_state.json'),
            ('history_sha256','history.json'),('sampling_sha256','sampling_actual.json'),
            ('best_sha256','best.pt'),('last_sha256','last.pt')):
        p0.checked_hash(row/name, proof[key])
    if f.read(row/'sampling_actual.json')['rows'] != f.read(paths['output_dir']/'sampling_plan.json'):
        raise ValueError('Incomplete actual first-40 source/instance stream')
    return status


def run(path):
    source,v,paths,_=load_study(path)
    f.verify_prepared(paths,sealed=True)
    output=paths['output_dir']
    if (output/'owner.json').exists() or (output/'status.json').exists():
        raise FileExistsError('One queue launch only; explicit recovery required')
    write_json(output/'owner.json',{'pid':os.getpid(),'creation_ticks':creation_time_ticks(os.getpid()),
        'started_utc':f.now(),'automatic_retry':False,'command':[sys.executable,'-m','scripts.valais_c5r5_40ep','run','--config',str(source)]})
    child=None;completed=[];role='c5'
    try:
        for role in f.ROLES:
            command=[str(paths['python']),'-u','-m','scripts.five_view_40ep','--config',str(source),'--role',role]
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
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=60)
        _atomic_json(output/'status.json',{'status':'failed_preserve','stage':role+'_training','error':f'{type(error).__name__}: {error}',
            'owner_pid':os.getpid(),'child_pid':child.pid if child else None,'completed_roles':completed,
            'automatic_retry':False,'updated_utc':f.now(),'test_accessed':False})
        raise


def monitor(path):
    _,v,paths,_=load_study(path);output=paths['output_dir']
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
    parser.add_argument('action',choices=('prepare','smoke','seal','run','monitor'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unittest-log',type=Path)
    args=parser.parse_args()
    result=seal(args.config,args.unittest_log) if args.action=='seal' else globals()[args.action](args.config)
    if result is not None: print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
