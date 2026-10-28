"""Creation-bound CPU supervision of the missing B0-E controls; no torch here."""
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
MODULE='scripts.b0e_replication_supervision'
RUNTIME=ROOT/'configs/b0e_replication_runtime_20261026.yaml'


def load_runtime(path=RUNTIME):
    path=Path(path).resolve()
    if path!=RUNTIME:raise ValueError('Only the frozen runtime is admitted')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    expected=dict(schema_version=1,gpu_index=0,gpu_uuid='GPU-8ada1895-7628-bb88-1dda-c57679117fa2',
        poll_seconds=2,confirmation_seconds=1,confirmed_pause_c=85,immediate_pause_c=86,
        confirmation_samples=3,max_sensor_failures=3,sensor_timeout_seconds=1,
        heartbeat_max_age_seconds=30,startup_limit_seconds=120,progress_stall_seconds=900,
        automatic_retry=False,python='E:/Download/anaconda3/envs/geosa-basa/python.exe')
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen protection policy changed')
    study=(path.parent/cfg['study_config']).resolve()
    if study!=ROOT/'configs/b0e_replication_20261026.yaml':raise ValueError('Wrong study')
    v=yaml.safe_load(study.read_text(encoding='utf-8'))
    if v['seeds']!=[43,44] or v['test_or_martigny_access'] is not False or v['automatic_retry'] is not False:
        raise ValueError('Study isolation changed')
    for k in ('python','powershell','nvidia_smi'):
        if not Path(cfg[k]).is_file():raise FileNotFoundError(cfg[k])
    cfg.update(study_path=str(study),study_output=str(ROOT/v['output_dir']),
        ledger_path=str(ROOT/v['shared_ledger']),global_lease=str(ROOT/v['global_lease']))
    return path,cfg,v


def verify(bound):
    for name,expected in bound.items():
        if common.sha(name)!=expected:raise ValueError('Sealed input changed: '+name)


def session_path(cfg,mode):
    if mode not in ('pilot','train'):raise ValueError('Unplanned execution mode')
    return Path(cfg['study_output'])/(mode+'_session')


def seal(checks):
    _,cfg,_=load_runtime();out=Path(cfg['study_output']);prepared=read(out/'prepared.json')
    bound=dict(prepared['bound_sha256'])
    correction=out/'cpu_smoke_correction.json'
    if correction.exists():
        c=read(correction)
        if (c['original_prepared_sha256']!=common.sha(out/'prepared.json') or
            c['reason']!='explicit_disabled_scaler_for_cpu_fp32_smoke_only' or
            set(c['changed_source_sha256'])!={str(ROOT/'scripts/complete_b0e_replication.py'),str(Path(__file__).resolve())}):
            raise ValueError('Invalid bounded CPU engineering correction')
        for name,expected in c['changed_source_sha256'].items():
            relative=Path(name).relative_to(ROOT)
            if common.sha(out/'source'/relative)!=bound[name]:
                raise ValueError('Original implementation snapshot changed')
            bound[name]=expected
        bound[str(correction)]=common.sha(correction)
    verify(bound)
    checks=Path(checks).resolve();log=checks.read_text(encoding='utf-8')
    if not re.search(r'Ran \d+ tests? in [\d.]+s\s+OK\s*\Z',log):
        raise ValueError('Actual successful full unittest log required')
    if read(out/'smoke.json')['status']!='passed_synthetic_engineering_only':
        raise ValueError('Successful explicit synthetic smoke required')
    for p in (out/'prepared.json',out/'smoke.json',checks):bound[str(p)]=common.sha(p)
    target=out/'engineering_authorization.json'
    if target.exists():raise FileExistsError('Preserve existing one-shot engineering seal')
    atomic(target,dict(status='engineering_authorized_under_explicit_continuation',bound_sha256=bound,
        created_utc=common.now(),test_accessed=False,automatic_retry=False))
    return dict(status='engineering_sealed',bound_inputs=len(bound))


def authorize():
    _,cfg,v=load_runtime();out=Path(cfg['study_output']);session=session_path(cfg,'pilot')
    a=read(out/'engineering_authorization.json');verify(a['bound_sha256'])
    p=read(session/'pilot_verified.json');owner=read(session/'owner_state.json')
    if (p['status']!='passed_engineering_only' or not p['repeat_full_state_exact'] or
        not p['next_update_full_state_exact'] or owner['status']!='completed' or owner['worker_exit_code']!=0 or
        any(read(session/(r+'_state.json'))['status']!='worker_exited' for r in ('guard','watchdog'))):
        raise ValueError('Exact pilot and independent protected completion required')
    ledger=read(cfg['ledger_path'])
    if any(e.get('status')=='running' for e in ledger['entries']):raise RuntimeError('Active compute reservation')
    from scripts.complete_b0e_replication import budget_estimate
    budget=budget_estimate(p['rows'],ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds'],v['evaluation_reserve_seconds'])
    if not budget['passed']:
        atomic(out/'budget_rejection.json',budget);raise RuntimeError('Frozen whole-study budget gate failed')
    bound=dict(a['bound_sha256'])
    for name in ('pilot_verified.json','owner_state.json','guard_state.json','watchdog_state.json','attachment.json'):
        bound[str(session/name)]=common.sha(session/name)
    target=out/'training_authorization.json'
    if target.exists():raise FileExistsError('Preserve existing training authorization')
    atomic(target,dict(status='training_authorized_under_explicit_publication_evidence_request',
        user_request=v['user_request'],created_utc=common.now(),bound_sha256=bound,budget=budget,
        test_accessed=False,automatic_retry=False))
    return dict(status='training_authorized',budget=budget)


def reserve(cfg,v,mode):
    ledger=read(cfg['ledger_path']);identifier='b0e_replication_'+mode+'_20261026'
    if any(e.get('status')=='running' for e in ledger['entries']):raise RuntimeError('Active/unreconciled compute reservation')
    if any(e.get('id')==identifier for e in ledger['entries']):raise RuntimeError('No retry or reset')
    remaining=ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds']
    limit=min(v['pilot_wall_limit_seconds'] if mode=='pilot' else 2*v['single_training_limit_seconds'],
              remaining-(v['evaluation_reserve_seconds'] if mode=='train' else 0))
    if limit<=0:raise RuntimeError('Original shared budget exhausted')
    ledger['entries'].append(dict(id=identifier,kind='b0e_replication_'+mode+'_owner_wall',
        status='running',charged_seconds=0.,started_utc=common.now(),maximum_seconds=limit,
        research_training=mode=='train',basis='Complete owner wall including CPU/I/O/setup/guard cleanup; original shared ledger.'))
    atomic(cfg['ledger_path'],ledger);return identifier,limit


def ledger_update(cfg,identifier,elapsed,status,exit_code=None):
    ledger=read(cfg['ledger_path']);matches=[e for e in ledger['entries'] if e.get('id')==identifier]
    if len(matches)!=1 or elapsed<matches[0]['charged_seconds']:
        raise ValueError('Unique monotonic compute charge required')
    entry=matches[0];entry.update(charged_seconds=elapsed,status=status,updated_utc=common.now())
    if exit_code is not None:entry['worker_exit_code']=exit_code
    result=session_path(cfg,'train')/'worker_result.json'
    if entry.get('research_training') and status=='completed' and result.exists():
        entry['successful_research_optimizer_updates']=read(result)['successful_research_optimizer_updates']
        ledger['new_research_optimizer_updates']=sum(e.get('successful_research_optimizer_updates',0) for e in ledger['entries'])
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
        if row['ProcessId']!=os.getpid() and re.search(r'(native_window_supervision|tiling_probe_supervision|b0e_replication_supervision)\s+worker\b',command):
            raise RuntimeError('Another possible GPU worker exists: '+str(row['ProcessId']))


def launch(mode):
    path,cfg,_=load_runtime();out=Path(cfg['study_output']);session=session_path(cfg,mode)
    if session.exists() or Path(cfg['global_lease']).exists():raise FileExistsError('Existing execution evidence/lease')
    verify(read(out/('engineering_authorization.json' if mode=='pilot' else 'training_authorization.json'))['bound_sha256'])
    external_check(cfg);sample=common.gpu_state(cfg)
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
        verify(read(Path(cfg['study_output'])/('engineering_authorization.json' if mode=='pilot' else 'training_authorization.json'))['bound_sha256'])
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
            raise ValueError('Worker creation identity differs')
        handles=[WindowsProcess(proof[r]) for r in ('owner','guard','watchdog')];started=time.monotonic()
        while not all((session/(r+'_state.json')).exists() for r in ('guard','watchdog')):
            if time.monotonic()-started>cfg['startup_limit_seconds'] or not all(h.is_alive() for h in handles):
                raise RuntimeError('Independent protection not ready before torch import')
            time.sleep(.2)
        last=-10.;counter=0
        def check():
            nonlocal last,counter
            for r,h in zip(('owner','guard','watchdog'),handles):
                if not h.is_alive():raise RuntimeError('Bound supervisor exited: '+r)
                common.require_fresh(session/(r+'_state.json'),cfg['heartbeat_max_age_seconds'])
            counter+=1
            if mode=='pilot' and time.monotonic()-last>=2:
                atomic(session/'progress.json',dict(stage='train_only_engineering_pilot',guard_checkpoints=counter,updated_utc=common.now()))
                last=time.monotonic()
        check();common.gpu_state(cfg)
        atomic(session/'progress.json',dict(stage='supervisors_ready_before_torch_import',updated_utc=common.now()))
        verify(read(Path(cfg['study_output'])/('engineering_authorization.json' if mode=='pilot' else 'training_authorization.json'))['bound_sha256'])
        from scripts import complete_b0e_replication as runner
        result=runner.pilot(session,cfg,check) if mode=='pilot' else runner.train(session,cfg,check)
        atomic(session/'worker_result.json',dict(status='completed',result_status=result['status'],completed_utc=common.now(),
             successful_research_optimizer_updates=result.get('research_optimizer_updates',0),test_accessed=False))
        return dict(status='completed')
    except BaseException as error:
        atomic(session/'worker_result.json',dict(status='failed_preserve',error=f'{type(error).__name__}: {error}',updated_utc=common.now(),automatic_retry=False));raise
    finally:
        for h in handles:h.close()


def monitor(mode):
    _,cfg,v=load_runtime();session=session_path(cfg,mode);result=dict(mode=mode,checked_utc=common.now(),gpu=common.gpu_state(cfg))
    for n in ('owner_state','progress','guard_state','watchdog_state','worker_result'):
        if (session/(n+'.json')).exists():result[n]=read(session/(n+'.json'))
    if (session/'attachment.json').exists():
        proof=read(session/'attachment.json');result['processes']={r:common.snapshot_identity(cfg,proof[r]) for r in ('owner','worker','guard','watchdog')}
    for seed in v['seeds']:
        p=Path(cfg['study_output'])/f'seed{seed}'/'status.json'
        if p.exists():result[f'seed{seed}']=read(p)
    l=read(cfg['ledger_path']);result['shared_charged_seconds']=l['total_charged_seconds'];result['shared_upper_bound_seconds']=l['batch_upper_bound_seconds']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seal','authorize','launch','owner','worker','guard','watchdog','monitor'])
    p.add_argument('--mode',choices=['pilot','train'],default='pilot');p.add_argument('--checks',type=Path);a=p.parse_args()
    if a.action=='seal':result=seal(a.checks)
    elif a.action=='authorize':result=authorize()
    elif a.action in ('guard','watchdog'):result=guard_loop(a.mode,a.action)
    else:result=globals()[a.action](a.mode)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
