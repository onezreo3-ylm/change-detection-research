"""Independent CPU supervision of the completed B0-E fixed validation; no torch here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

import yaml

from scripts import levir_tail_risk_supervision as common
from scripts.temperature_guard import TemperaturePolicy, WindowsProcess
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT=Path(__file__).resolve().parents[1]
MODULE='scripts.fixed_development_evaluation_supervision'
RUNTIME=ROOT/'configs/b0e_replication_evaluation_runtime_20261026.yaml'


def load_runtime(path=RUNTIME):
    path=Path(path).resolve()
    if path!=RUNTIME:raise ValueError('Only the frozen development evaluation runtime is admitted')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    expected=dict(schema_version=1,kind='development_validation_external_supervision',gpu_index=0,
        gpu_uuid='GPU-8ada1895-7628-bb88-1dda-c57679117fa2',poll_seconds=2,confirmation_seconds=1,
        confirmed_pause_c=85,immediate_pause_c=86,confirmation_samples=3,max_sensor_failures=3,
        sensor_timeout_seconds=1,heartbeat_max_age_seconds=30,startup_limit_seconds=120,
        progress_stall_seconds=900,automatic_retry=False,python='E:/Download/anaconda3/envs/geosa-basa/python.exe')
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen evaluation protection changed')
    study=(path.parent/cfg['study_config']).resolve()
    if study!=ROOT/'configs/b0e_replication_evaluation_20261026.yaml':raise ValueError('Wrong validation study')
    v=yaml.safe_load(study.read_text(encoding='utf-8'))
    if (v.get('kind')!='completed_b0e_replication_fixed_validation' or v.get('seeds')!=[43,44] or
        v.get('whole_evaluation_limit_seconds')!=3600 or v.get('postprocessing_reserve_seconds')!=120 or
        v.get('test_or_martigny_access') is not False or v.get('automatic_retry') is not False):
        raise ValueError('Fixed validation isolation/budget changed')
    training=yaml.safe_load((ROOT/v['training_study']).read_text(encoding='utf-8'))
    for key in ('python','powershell','nvidia_smi'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    cfg.update(study_path=str(study),study_output=str(ROOT/v['output_dir']),
        ledger_path=str(ROOT/training['shared_ledger']),global_lease=str(ROOT/training['global_lease']))
    return path,cfg,v



def verify(bound):
    for name,expected in bound.items():
        if common.sha(name)!=expected:raise ValueError('Sealed input changed: '+name)


def session_path(cfg,mode):
    if mode!='evaluate':raise ValueError('Only the frozen development evaluation mode is admitted')
    return Path(cfg['study_output'])/'evaluation_session'



def seal(checks):
    _,cfg,_=load_runtime();out=Path(cfg['study_output']);prepared=read(out/'prepared.json')
    verify(prepared['bound_sha256'])
    checks=Path(checks).resolve();log=checks.read_text(encoding='utf-8')
    passed_suite_count(log)
    if read(out/'smoke.json')['status']!='passed_cpu_synthetic_engineering_only':raise ValueError('Explicit CPU smoke required')
    sync_costs(cfg)
    costs=read(out/'engineering_costs.json')
    if not {'prepare','smoke','full_tests'}.issubset({r['phase'] for r in costs['entries']}):
        raise ValueError('Preparation, smoke and actual full-suite costs must be recorded')
    bound=dict(prepared['bound_sha256'])
    for p in (out/'prepared.json',out/'smoke.json',checks):bound[str(p)]=common.sha(p)
    target=out/'authorization.json'
    if target.exists():raise FileExistsError('Preserve existing validation authorization')
    atomic(target,dict(status='authorized_fixed_development_validation_only',bound_sha256=bound,
        created_utc=common.now(),automatic_retry=False,test_accessed=False))
    return dict(status='sealed_fixed_development_validation',bound_inputs=len(bound))






def reserve(cfg,v,mode):
    session_path(cfg,mode);ledger=read(cfg['ledger_path']);identifier='b0e_fixed_validation_owner_20261026'
    if any(e.get('status')=='running' for e in ledger['entries']):raise RuntimeError('Active/unreconciled reservation')
    if any(e.get('id')==identifier for e in ledger['entries']):raise RuntimeError('No evaluation retry or budget reset')
    spent=sum(e['charged_seconds'] for e in ledger['entries'] if e.get('kind','').startswith('b0e_fixed_validation_'))
    reserve=v.get('postprocessing_reserve_seconds',120)
    limit=min(v['whole_evaluation_limit_seconds']-spent-reserve,
              ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds']-reserve)
    if limit<=0:raise RuntimeError('Original whole-evaluation/shared budget exhausted')
    ledger['entries'].append(dict(id=identifier,kind='b0e_fixed_validation_owner_wall',status='running',
        charged_seconds=0.,started_utc=common.now(),maximum_seconds=limit,research_training=False,
        optimizer_updates=0,basis='Complete external owner wall; engineering and discarded work retain the original shared charges.'))
    atomic(cfg['ledger_path'],ledger);return identifier,limit



def ledger_update(cfg,identifier,elapsed,status,exit_code=None):
    ledger=read(cfg['ledger_path']);rows=[e for e in ledger['entries'] if e.get('id')==identifier]
    if len(rows)!=1 or elapsed<rows[0]['charged_seconds']:raise ValueError('Unique monotonic charge required')
    rows[0].update(charged_seconds=elapsed,status=status,updated_utc=common.now())
    if exit_code is not None:rows[0]['worker_exit_code']=exit_code
    ledger['total_charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
    atomic(cfg['ledger_path'],ledger)



def external_check(cfg):
    common.external_worker_check(cfg)
    result=subprocess.run([cfg['powershell'],'-NoProfile','-NonInteractive','-Command',
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True,text=True,check=True,timeout=20,creationflags=common.HIDDEN)
    rows=json.loads(result.stdout or 'null') or []
    if isinstance(rows,dict):rows=[rows]
    for row in rows:
        command=row.get('CommandLine') or ''
        if row['ProcessId']!=os.getpid() and re.search(r'(native_window_supervision|tiling_probe_supervision|b0e_replication_supervision|fixed_development_evaluation_supervision)\s+worker\b',command):
            raise RuntimeError('Another possible GPU worker exists: '+str(row['ProcessId']))


def launch(mode):
    path,cfg,_=load_runtime();out=Path(cfg['study_output']);session=session_path(cfg,mode)
    if session.exists() or Path(cfg['global_lease']).exists():raise FileExistsError('Existing execution evidence/lease')
    verify(read(out/'authorization.json')['bound_sha256'])
    sync_costs(cfg);external_check(cfg);sample=common.gpu_state(cfg)
    if sample['temperature_c']>=cfg['confirmed_pause_c']:raise RuntimeError('GPU already hot; no launch')
    session.mkdir()
    command=[cfg['python'],'-X','utf8','-u','-m',MODULE,'owner','--mode',mode]
    with (session/'owner.stdout.log').open('x') as a,(session/'owner.stderr.log').open('x') as b:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=common.HIDDEN)
    return dict(status='hidden_cpu_owner_dispatched',mode=mode,owner_pid=proc.pid,
                owner_creation_ticks=common.creation_ticks(proc.pid),gpu=sample)


def owner(mode):
    path,cfg,v=load_runtime();session=session_path(cfg,mode);lease=Path(cfg['global_lease'])
    with lease.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created_utc=common.now()),f)
    child=None;guards=[];entry=None;started=time.monotonic()
    state=dict(status='running',mode=mode,updated_utc=common.now(),automatic_retry=False)
    try:
        external_check(cfg)
        verify(read(Path(cfg['study_output'])/'authorization.json')['bound_sha256'])
        own=common.identity(cfg,os.getpid());entry,limit=reserve(cfg,v,mode)
        atomic(session/'owner_state.json',dict(state,allowed_wall_seconds=limit))
        def spawn(role):
            command=[cfg['python'],'-X','utf8','-u','-m',MODULE,role,'--mode',mode]
            with (session/(role+'.stdout.log')).open('x') as a,(session/(role+'.stderr.log')).open('x') as b:
                return subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=common.HIDDEN)
        child=spawn('worker');identities=dict(owner=own,worker=common.identity(cfg,child.pid))
        for role in ('guard','watchdog'):
            proc=spawn(role);guards.append(proc);identities[role]=common.identity(cfg,proc.pid)
        atomic(session/'attachment.json',dict(identities,allowed_wall_seconds=limit,created_utc=common.now(),
            runtime_sha256=common.sha(path),adapter_sha256=common.sha(__file__),
            source_sha256=common.sha(common.__file__),mode=mode))
        while child.poll() is None:
            elapsed=time.monotonic()-started
            atomic(session/'owner_state.json',dict(state,elapsed_seconds=elapsed,allowed_wall_seconds=limit,updated_utc=common.now()))
            ledger_update(cfg,entry,elapsed,'running')
            if elapsed>=limit or any(p.poll() is not None for p in guards):raise RuntimeError('Wall cap or independent supervisor exit')
            if (session/'progress.json').exists():
                if common.age_seconds(read(session/'progress.json')['updated_utc'])>cfg['progress_stall_seconds']:
                    raise RuntimeError('Actual batch checkpoint progress stalled')
            elif elapsed>cfg['progress_stall_seconds']:raise RuntimeError('No actual worker progress')
            time.sleep(cfg['poll_seconds'])
        if child.returncode!=0 or read(session/'worker_result.json')['status']!='completed':
            raise RuntimeError('Worker failed: '+str(child.returncode))
        state.update(status='completed',worker_exit_code=0)
    except BaseException as error:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=15)
        state.update(status='failed_preserve',worker_exit_code=child.returncode if child else None,error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        for proc in guards:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
            if proc.returncode and state['status']=='completed':state.update(status='failed_preserve',error='Supervisor failed')
        elapsed=time.monotonic()-started
        state.update(elapsed_seconds=elapsed,updated_utc=common.now());atomic(session/'owner_state.json',state)
        if entry:ledger_update(cfg,entry,elapsed,state['status'],state.get('worker_exit_code'))
        lease.unlink()
    if state['status']!='completed':raise RuntimeError(state.get('error','Owner failed'))
    return state


def attached(cfg,session):
    proof=common.wait_attachment(session,cfg)
    if proof['runtime_sha256']!=common.sha(RUNTIME) or proof['adapter_sha256']!=common.sha(__file__):
        raise ValueError('Execution source identity changed')
    return proof


def guard_loop(mode,role):
    _,cfg,_=load_runtime();session=session_path(cfg,mode);proof=attached(cfg,session)
    if proof[role]['process_id']!=os.getpid() or proof[role]['creation_time_ticks']!=common.creation_ticks(os.getpid()):
        raise ValueError('Supervisor creation identity differs')
    handles={r:WindowsProcess(proof[r]) for r in ('worker','owner','watchdog' if role=='guard' else 'guard')}
    state=dict(status='monitoring',role=role,updated_utc=common.now());policy=TemperaturePolicy(cfg)
    started=time.monotonic();maximum=None
    try:
        atomic(session/(role+'_state.json'),state)
        peer='watchdog' if role=='guard' else 'guard'
        while not (session/(peer+'_state.json')).exists():
            if time.monotonic()-started>cfg['startup_limit_seconds'] or not all(h.is_alive() for h in handles.values()):
                raise RuntimeError('Independent peer heartbeat unavailable at startup')
            atomic(session/(role+'_state.json'),dict(state,updated_utc=common.now()))
            time.sleep(.1)
        while handles['worker'].is_alive():
            if not all(h.is_alive() for h in handles.values()):raise RuntimeError('Bound peer exited')
            if time.monotonic()-started>=proof['allowed_wall_seconds']:raise RuntimeError('Independent wall cap')
            for r in ('owner','watchdog' if role=='guard' else 'guard'):
                common.require_fresh(session/(r+'_state.json'),cfg['heartbeat_max_age_seconds'])
            if role=='guard':
                temperature=None;sensor_error=None
                try:temperature=common.gpu_state(cfg)['temperature_c']
                except Exception as error:sensor_error=str(error)
                decision=policy.observe(temperature)
                if temperature is not None:maximum=max(maximum if maximum is not None else temperature,temperature)
                state.update(temperature_c=temperature,maximum_observed_temperature_c=maximum,sensor_error=sensor_error,decision=decision)
                if decision in ('temperature_pause','sensor_pause'):raise RuntimeError('Temperature/sensor stop: '+decision)
            atomic(session/(role+'_state.json'),dict(state,updated_utc=common.now()))
            time.sleep(cfg['confirmation_seconds'] if role=='guard' and policy.hot_samples else cfg['poll_seconds'])
        state.update(status='worker_exited')
    except BaseException as error:
        if handles['worker'].is_alive():handles['worker'].stop();state.update(status='stopped_worker_preserve',error=f'{type(error).__name__}: {error}')
        else:state.update(status='worker_exited')
    finally:
        atomic(session/(role+'_state.json'),dict(state,updated_utc=common.now()))
        for h in handles.values():h.close()
    return state


def worker(mode):
    _,cfg,_=load_runtime();session=session_path(cfg,mode);handles=[]
    try:
        proof=attached(cfg,session)
        if proof['worker']['process_id']!=os.getpid() or proof['worker']['creation_time_ticks']!=common.creation_ticks(os.getpid()):
            raise ValueError('Evaluation worker creation identity differs')
        handles=[WindowsProcess(proof[r]) for r in ('owner','guard','watchdog')];started=time.monotonic()
        while not all((session/(r+'_state.json')).exists() for r in ('guard','watchdog')):
            if time.monotonic()-started>cfg['startup_limit_seconds'] or not all(h.is_alive() for h in handles):
                raise RuntimeError('Independent protection not ready before torch import')
            time.sleep(.2)
        def check():
            for role,handle in zip(('owner','guard','watchdog'),handles):
                if not handle.is_alive():raise RuntimeError('Bound evaluation supervisor exited: '+role)
                common.require_fresh(session/(role+'_state.json'),cfg['heartbeat_max_age_seconds'])
        check();common.gpu_state(cfg)
        atomic(session/'progress.json',dict(stage='evaluation_supervisors_ready_before_torch_import',updated_utc=common.now()))
        verify(read(Path(cfg['study_output'])/'authorization.json')['bound_sha256'])
        from scripts import evaluate_b0e_replications as runner
        result=runner.evaluate(session,check)
        atomic(session/'worker_result.json',dict(status='completed',result_status=result['status'],
            completed_utc=common.now(),optimizer_updates=0,successful_research_optimizer_updates=0,test_accessed=False))
        return dict(status='completed')
    except BaseException as error:
        atomic(session/'worker_result.json',dict(status='failed_preserve',error=f'{type(error).__name__}: {error}',
            updated_utc=common.now(),automatic_retry=False));raise
    finally:
        for handle in handles:handle.close()



def monitor(mode):
    _,cfg,v=load_runtime();session=session_path(cfg,mode);result=dict(mode=mode,checked_utc=common.now(),gpu=common.gpu_state(cfg))
    for n in ('owner_state','progress','guard_state','watchdog_state','worker_result'):
        if (session/(n+'.json')).exists():result[n]=read(session/(n+'.json'))
    if (session/'attachment.json').exists():
        proof=read(session/'attachment.json');result['processes']={r:common.snapshot_identity(cfg,proof[r]) for r in ('owner','worker','guard','watchdog')}
    for seed in v['seeds']:
        p=Path(cfg['study_output'])/f'seed{seed}'/'execution_status.json'
        if p.exists():result[f'seed{seed}']=read(p)
    l=read(cfg['ledger_path']);result['shared_charged_seconds']=l['total_charged_seconds'];result['shared_upper_bound_seconds']=l['batch_upper_bound_seconds']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seal','launch','owner','worker','guard','watchdog','monitor','record-cost','settle-costs'])
    p.add_argument('--mode',choices=['evaluate'],default='evaluate');p.add_argument('--checks',type=Path)
    p.add_argument('--phase');p.add_argument('--seconds',type=float);p.add_argument('--basis');a=p.parse_args()
    if a.action=='seal':
        started=time.monotonic();status='failed_preserve'
        try:result=seal(a.checks);status='completed'
        finally:record_cost('seal',time.monotonic()-started+1,'Actual CPU sealing wall plus1second import allowance; '+status)
    elif a.action=='record-cost':result=record_cost(a.phase,a.seconds,a.basis)
    elif a.action=='settle-costs':
        _,cfg,_=load_runtime();result=sync_costs(cfg)
    elif a.action in ('guard','watchdog'):result=guard_loop(a.mode,a.action)
    else:result=globals()[a.action](a.mode)
    print(json.dumps(result,ensure_ascii=True))




def record_cost(phase,seconds,basis):
    import math
    if phase not in ('prepare','smoke','full_tests','seal','certify','analysis') or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('Explicit valid engineering phase and measured wall required')
    _,cfg,_=load_runtime();out=Path(cfg['study_output'])
    if not out.is_dir():raise ValueError('Preparation directory does not exist')
    path=out/'engineering_costs.json';value=read(path) if path.exists() else {'entries':[]}
    identifier='b0e_fixed_validation_cpu_'+phase+'_20261026'
    if any(r['id']==identifier for r in value['entries']):raise ValueError('Preserve existing cost entry; no retry')
    value['entries'].append(dict(id=identifier,phase=phase,charged_seconds=float(seconds),basis=basis,recorded_utc=common.now()))
    atomic(path,value);return dict(status='engineering_cost_recorded',phase=phase,seconds=seconds)


def sync_costs(cfg):
    path=Path(cfg['study_output'])/'engineering_costs.json'
    value=read(path);ledger=read(cfg['ledger_path'])
    if any(r.get('status')=='running' for r in ledger['entries']):raise RuntimeError('Cannot revise an active shared reservation')
    for cost in value['entries']:
        existing=[r for r in ledger['entries'] if r.get('id')==cost['id']]
        if existing:
            if len(existing)!=1 or existing[0]['charged_seconds']!=cost['charged_seconds']:
                raise ValueError('Existing physical compute charge changed')
        else:
            ledger['entries'].append(dict(cost,kind='b0e_fixed_validation_engineering',status='completed',research_training=False,optimizer_updates=0))
    ledger['total_charged_seconds']=sum(r['charged_seconds'] for r in ledger['entries'])
    atomic(cfg['ledger_path'],ledger)
    spent=sum(r['charged_seconds'] for r in ledger['entries'] if r.get('kind','').startswith('b0e_fixed_validation_'))
    if spent>3600 or ledger['total_charged_seconds']>ledger.get('batch_upper_bound_seconds',float('inf')):
        raise RuntimeError('Physical evaluation/shared budget exceeded; all costs preserved')
    return dict(status='engineering_costs_settled',whole_evaluation_charged_seconds=sum(r['charged_seconds'] for r in ledger['entries'] if r.get('kind','').startswith('b0e_fixed_validation_')))


def passed_suite_count(log):
    match=re.search(r'Ran (\d+) tests? in [\d.]+s\s+OK\s*\Z',log)
    if (not match or int(match[1])<546 or 'test_b0e_fixed_evaluation.' not in log or
        'test_b0e_replication_analysis.' not in log):
        raise ValueError('New full project suite including both new modules must pass; old526test log is insufficient')
    return int(match[1])


if __name__=='__main__':main()
