"""CPU-only owner for fixed validation, reusing the sealed independent guards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts import levir_tail_risk_supervision as safety
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT=Path(__file__).resolve().parents[1]


def verify(bound):
    for name,expected in bound.items():
        h=hashlib.sha256()
        with Path(name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Frozen evaluation input changed: '+name)


def launch(path):
    path,cfg=safety.load_runtime(path);out=Path(cfg['study_output']);session=out/'training_session'
    if session.exists() or (out/'gpu_execution.lock').exists():raise FileExistsError('Existing evaluation evidence; no duplicate run')
    authority=read(out/'authorization.json')
    if authority['status']!='sealed_explicit_risk_validation':raise ValueError('Explicit evaluation seal required')
    verify(authority['bound_sha256']);safety.external_worker_check(cfg)
    if safety.gpu_state(cfg)['temperature_c']>=85:raise RuntimeError('Initial temperature pause')
    session.mkdir(exist_ok=False)
    command=[cfg['python'],'-X','utf8','-u','-m','scripts.levir_tail_risk_eval_owner','owner','--runtime',str(path)]
    with (session/'owner.stdout.log').open('x',encoding='utf-8') as a,(session/'owner.stderr.log').open('x',encoding='utf-8') as b:
        child=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=safety.HIDDEN)
    return {'status':'fixed_validation_cpu_owner_dispatched','pid':child.pid,'creation_ticks':safety.creation_ticks(child.pid),'optimizer_updates':0}


def reserve(cfg,settings):
    ledger=read(cfg['ledger_path'])
    if any(e.get('status')=='running' for e in ledger['entries']):raise RuntimeError('Another compute session is active')
    candidate_spent=sum(e['charged_seconds'] for e in ledger['entries'] if e.get('kind') in ('tail_risk_train_whole_worker_process','tail_risk_fixed45_validation_process'))
    limit=min(settings['evaluation_wall_limit_seconds'],86400-candidate_spent,ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds'])
    if limit<=0:raise RuntimeError('Original candidate/shared budget exhausted')
    identity='tail_risk_fixed45_validation_20261025'
    if any(e.get('id')==identity for e in ledger['entries']):raise RuntimeError('No automatic evaluation rerun')
    ledger['entries'].append({'id':identity,'kind':'tail_risk_fixed45_validation_process','status':'running',
        'charged_seconds':0.,'started_utc':safety.now(),'maximum_seconds':limit,'research_training':False,
        'optimizer_updates':0,'candidate_training_seconds_already_debited':candidate_spent,
        'basis':'Conservative whole validation worker wall time including setup, CPU bootstrap and I/O; same original budgets.'})
    atomic(cfg['ledger_path'],ledger)
    return identity,limit


def owner(path):
    path,cfg=safety.load_runtime(path);out=Path(cfg['study_output']);session=out/'training_session'
    import yaml
    settings=yaml.safe_load(Path(cfg['study_path']).read_text(encoding='utf-8'))
    lock=out/'gpu_execution.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    child=None;guards=[];entry=None;started=time.monotonic()
    state={'status':'running','kind':'fixed_validation_only','optimizer_updates':0,'automatic_retry':False,'updated_utc':safety.now()}
    try:
        safety.external_worker_check(cfg)
        own=safety.identity(cfg,os.getpid());atomic(session/'owner_identity.json',own)
        state.update(owner_pid=own['process_id'],owner_creation_ticks=own['creation_time_ticks'])
        atomic(session/'owner_state.json',state)
        entry,limit=reserve(cfg,settings)
        command=[cfg['python'],'-X','utf8','-u','-m','scripts.levir_tail_risk_eval_owner','worker','--runtime',str(path)]
        with (session/'worker.stdout.log').open('x',encoding='utf-8') as a,(session/'worker.stderr.log').open('x',encoding='utf-8') as b:
            child=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=safety.HIDDEN)
        worker=safety.identity(cfg,child.pid);atomic(session/'worker_identity.json',worker)
        state.update(worker_pid=child.pid,worker_creation_ticks=worker['creation_time_ticks'],allowed_wall_seconds=limit)
        identities={}
        for role in ['guard','watchdog']:
            command=[cfg['python'],'-X','utf8','-u','-m','scripts.levir_tail_risk_eval_owner',role,'--runtime',str(path)]
            with (session/(role+'.stdout.log')).open('x',encoding='utf-8') as a,(session/(role+'.stderr.log')).open('x',encoding='utf-8') as b:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=safety.HIDDEN)
            guards.append(proc);identities[role]=safety.identity(cfg,proc.pid)
        atomic(session/'attachment.json',{'owner':own,'worker':worker,**identities,'allowed_wall_seconds':limit,
            'created_utc':safety.now(),'runtime_sha256':safety.sha(path),'source_sha256':safety.sha(safety.__file__),
            'evaluation_owner_source_sha256':safety.sha(__file__),'kind':'fixed_validation_only','optimizer_updates':0})
        state.update(guard_pid=guards[0].pid,watchdog_pid=guards[1].pid)
        while child.poll() is None:
            elapsed=time.monotonic()-started
            atomic(session/'owner_state.json',dict(state,updated_utc=safety.now(),elapsed_seconds=elapsed))
            safety.ledger_update(cfg,entry,elapsed,'running')
            if elapsed>=limit or any(p.poll() is not None for p in guards):raise RuntimeError('Evaluation cap or independent supervisor exit')
            if (session/'progress.json').exists():
                p=read(session/'progress.json')
                if p.get('numeric_alert') or safety.age_seconds(p['updated_utc'])>cfg['progress_stall_seconds']:raise RuntimeError('Evaluation numeric/progress guard')
            elif elapsed>cfg['progress_stall_seconds']:raise RuntimeError('No actual evaluation progress')
            time.sleep(cfg['poll_seconds'])
        if child.returncode!=0 or read(session/'worker_result.json')['status']!='completed':raise RuntimeError('Evaluation worker exit '+str(child.returncode))
        state.update(status='completed',worker_exit_code=0)
    except BaseException as error:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        state.update(status='failed_preserve',worker_exit_code=child.returncode if child else None,error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        elapsed=time.monotonic()-started;state.update(updated_utc=safety.now(),elapsed_seconds=elapsed)
        atomic(session/'owner_state.json',state)
        if entry:safety.ledger_update(cfg,entry,elapsed,state['status'],state.get('worker_exit_code'))
        for p in guards:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
        lock.unlink()
    if state['status']!='completed':raise RuntimeError(state.get('error','Evaluation failed'))
    return state


def worker(path):
    path,cfg=safety.load_runtime(path);session=Path(cfg['study_output'])/'training_session';handles=[]
    try:
        proof=safety.wait_attachment(session,cfg)
        if proof['runtime_sha256']!=safety.sha(path) or proof['evaluation_owner_source_sha256']!=safety.sha(__file__):raise ValueError('Evaluation code/runtime changed')
        if proof['worker']['process_id']!=os.getpid() or proof['worker']['creation_time_ticks']!=safety.creation_ticks(os.getpid()):raise ValueError('Evaluation worker identity differs')
        handles=[safety.WindowsProcess(proof[r]) for r in ['owner','guard','watchdog']]
        deadline=time.monotonic()+cfg['startup_limit_seconds']
        while not all((session/(r+'_state.json')).exists() for r in ['guard','watchdog']):
            if time.monotonic()>deadline or not all(p.is_alive() for p in handles):raise RuntimeError('Evaluation supervisors not ready')
            time.sleep(.2)
        def check():
            for role,handle in zip(['owner','guard','watchdog'],handles):
                if not handle.is_alive():raise RuntimeError('Bound evaluation supervisor exited: '+role)
                safety.require_fresh(session/(role+'_state.json'),cfg['heartbeat_max_age_seconds'])
        check();atomic(session/'progress.json',{'stage':'evaluation_supervisors_attached_before_cuda_import','step':0,'updated_utc':safety.now(),'numeric_alert':False})
        verify(read(Path(cfg['study_output'])/'authorization.json')['bound_sha256'])
        from scripts.levir_tail_risk_evaluation import evaluate
        result=evaluate(cfg['study_path'],session,check)
        atomic(session/'worker_result.json',{'status':'completed','kind':'fixed_validation_only','all_gates_passed':result['all_passed'],'optimizer_updates':0,'test_accessed':False,'completed_utc':safety.now()})
        return {'status':'completed'}
    except BaseException as error:
        atomic(session/'worker_result.json',{'status':'failed_preserve','error':f'{type(error).__name__}: {error}','updated_utc':safety.now(),'automatic_retry':False})
        raise
    finally:
        for p in handles:p.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['launch','owner','worker','guard','watchdog','monitor']);parser.add_argument('--runtime',type=Path,required=True);args=parser.parse_args()
    if args.action in ['guard','watchdog']:result=safety.guard_loop(args.runtime,'evaluation',args.action)
    elif args.action=='monitor':result=safety.monitor(args.runtime,'evaluation')
    else:result=globals()[args.action](args.runtime)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
