"""CPU-only supervision of the already-running C5; never starts C5 again."""
from __future__ import annotations
import argparse
import contextlib
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.p1_process_state import process_snapshot

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def validate_attachment(owner,worker,queue):
    if owner.get('owned_active'):
        raise ValueError('Original owner is still active; no second supervisor')
    if not worker.get('owned_active') or not worker.get('identity_matches'):
        raise ValueError('Existing worker lacks live creation identity')
    if (queue.get('status')!='running' or queue.get('stage')!='c5_training'
            or queue.get('completed_roles')!=[] or queue.get('child_pid')!=worker['pid']
            or queue.get('child_creation_ticks')!=worker['creation_ticks']):
        raise ValueError('Only the original un-restarted C5 may be attached')


def remaining_seconds(progress,debit,utc_now):
    elapsed=progress['elapsed_seconds']
    updated=datetime.fromisoformat(progress['updated_utc'])
    age=(utc_now-updated).total_seconds()
    if not 0<=debit<2400 or not 0<=elapsed<86400 or age < -1:
        raise ValueError('Invalid unchanged role budget or progress clock')
    remaining=86400-debit-elapsed-max(0,age)
    if remaining<=0:raise RuntimeError('Original role budget exhausted')
    return remaining


def open_existing(pid,expected_ticks):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;k.GetProcessTimes.restype=wintypes.BOOL
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.GetExitCodeProcess.restype=wintypes.BOOL
    k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];k.TerminateProcess.restype=wintypes.BOOL
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x1000|0x100000|1,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    times=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(handle,*(ctypes.byref(t) for t in times)):
        k.CloseHandle(handle);raise ctypes.WinError(ctypes.get_last_error())
    actual=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
    if actual!=expected_ticks:
        k.CloseHandle(handle);raise ValueError('Existing C5 PID was reused')
    return k,handle


def run(config,attachment):
    from scripts import valais_c5r5_runtime as frozen
    from scripts.run_valais_clean_guarded import creation_time_ticks
    from pcdc.data import file_sha256
    _,settings,paths,_=frozen.load_study(config)
    root=paths['output_dir'];attachment=Path(attachment).resolve()
    proof=read(attachment/'authorization.json')
    for name,expected in proof['bound_sha256'].items():
        if sha(name)!=expected:raise ValueError('Supervision input changed: '+name)
    if (attachment/'owner.json').exists():raise FileExistsError('One CPU attachment only')
    queue=read(root/'status.json');original=read(root/'owner.json')
    owner_state=process_snapshot(original['pid'],original['creation_ticks'])
    worker_state=process_snapshot(proof['existing_c5_pid'],proof['existing_c5_creation_ticks'])
    validate_attachment(owner_state,worker_state,queue)
    k,handle=open_existing(worker_state['pid'],worker_state['creation_ticks'])
    owner={'pid':os.getpid(),'creation_ticks':creation_time_ticks(os.getpid()),'started_utc':now(),
        'existing_c5_pid':worker_state['pid'],'existing_c5_creation_ticks':worker_state['creation_ticks'],
        'c5_restarted':False,'automatic_retry':False,'consoleless_host':True}
    with (attachment/'owner.json').open('x',encoding='utf-8') as stream:json.dump(owner,stream,indent=2)
    authority=read(root/'authorization.json');completed=[];child=None;stage='c5_training'
    state={'status':'running','stage':stage,'owner_pid':os.getpid(),'child_pid':worker_state['pid'],
        'child_creation_ticks':worker_state['creation_ticks'],'completed_roles':[],'c5_restarted':False,
        'updated_utc':now(),'test_accessed':False}
    try:
        progress=read(root/'c5/progress.json')
        deadline=time.monotonic()+remaining_seconds(progress,authority['engineering_seconds_debited']['c5'],datetime.now(timezone.utc))
        while True:
            atomic(attachment/'status.json',dict(state,updated_utc=now()))
            left=deadline-time.monotonic()
            if left<=0:
                k.TerminateProcess(handle,1)
                raise RuntimeError('Original C5 role time cap exhausted; retained completed epochs only')
            wait=k.WaitForSingleObject(handle,min(30000,max(1,int(left*1000))))
            if wait==0:break
            if wait!=258:raise ctypes.WinError(ctypes.get_last_error())
        exit_code=wintypes.DWORD()
        if not k.GetExitCodeProcess(handle,ctypes.byref(exit_code)):raise ctypes.WinError(ctypes.get_last_error())
        if exit_code.value!=0:raise RuntimeError('Existing C5 exited '+str(exit_code.value))
        frozen.check_completion(paths,'c5')
        atomic(attachment/'c5.exit.json',{'exit_code':exit_code.value,'observed_utc':now(),
            'pid':worker_state['pid'],'creation_ticks':worker_state['creation_ticks'],
            'status_sha256':file_sha256(root/'c5/status.json'),'c5_restarted':False})
        completed.append('c5')
        stage='r5_training'
        if any((root/'r5'/n).exists() for n in ('status.json','progress.json','last.pt')):
            raise FileExistsError('R5 already has execution evidence; refusing a duplicate')
        command=[str(paths['python']),'-u','-m','scripts.five_view_40ep_runtime','--config',str(Path(config).resolve()),'--role','r5']
        with (root/'r5.stdout.log').open('x',encoding='utf-8') as out,(root/'r5.stderr.log').open('x',encoding='utf-8') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(stage=stage,child_pid=child.pid,child_creation_ticks=creation_time_ticks(child.pid),
                completed_roles=list(completed),child_command=command,updated_utc=now())
            atomic(attachment/'status.json',state)
            try:code=child.wait(timeout=86400-authority['engineering_seconds_debited']['r5'])
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=60)
                raise RuntimeError('Original R5 role cap exhausted; no extension or retry')
        if code!=0:raise RuntimeError('R5 exited '+str(code))
        frozen.check_completion(paths,'r5')
        atomic(attachment/'r5.exit.json',{'exit_code':code,'observed_utc':now(),'pid':child.pid,
            'status_sha256':file_sha256(root/'r5/status.json')})
        completed.append('r5')
        a,b=(read(root/r/'history.json')['rows'] for r in ('c5','r5'))
        if len(a)!=40 or len(b)!=40:raise ValueError('Incomplete matched histories')
        for left,right in zip(a,b):
            if left['train']['candidate_pixel_stream_sha256']!=right['train']['candidate_pixel_stream_sha256']:
                raise ValueError('Full candidate pixel streams differ')
        atomic(attachment/'status.json',{'status':'completed_training_ready_for_explicit_evaluation',
            'completed_roles':completed,'completed_utc':now(),'matched_actual_source_instance_pixels':True,
            'original_owner_control_c_exit_preserved':True,'c5_restarted':False,'test_accessed':False})
    except BaseException as error:
        atomic(attachment/'status.json',dict(state,status='failed_preserve',stage=stage,completed_roles=completed,
            error=f'{type(error).__name__}: {error}',updated_utc=now(),automatic_retry=False))
        raise
    finally:k.CloseHandle(handle)


def monitor(config,attachment):
    from scripts import valais_c5r5_runtime as frozen
    _,_,paths,_=frozen.load_study(config);root=paths['output_dir'];attachment=Path(attachment).resolve()
    result=frozen.monitor(config)
    result['original_queue']=result.pop('queue');result['original_owner']=result.pop('owner',{})
    result['original_worker']=result.pop('worker',{})
    state=read(attachment/'status.json') if (attachment/'status.json').exists() else {}
    result['queue']={k:state[k] for k in ('status','stage','completed_roles','updated_utc','completed_utc','error') if k in state}
    if (attachment/'owner.json').exists():
        owner=read(attachment/'owner.json');result['owner']=process_snapshot(owner['pid'],owner['creation_ticks'])
    if state.get('child_pid') and state.get('child_creation_ticks'):
        result['worker']=process_snapshot(state['child_pid'],state['child_creation_ticks'])
    result.pop('attention',None)
    for role in ('c5','r5'):
        result[role].pop('attention',None)
        if state.get('stage')==role+'_training' and result[role].get('log_age_seconds',0)>900:
            result[role]['attention']='Active role lacks actual log progress for >15 minutes; diagnose without retry.'
    if state.get('status')=='running' and (not result.get('owner',{}).get('owned_active') or not result.get('worker',{}).get('owned_active')):
        result['attention']='Attached owner/worker creation identity or wait/exit state is not active; preserve and diagnose.'
    result['supervision_attachment']=str(attachment)
    result['c5_restarted']=False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','monitor'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--attachment',type=Path,required=True)
    args=p.parse_args()
    if args.action=='monitor':print(json.dumps(monitor(args.config,args.attachment),ensure_ascii=False));return
    with (args.attachment/'stdout.log').open('x',encoding='utf-8') as out,(args.attachment/'stderr.log').open('x',encoding='utf-8') as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:run(args.config,args.attachment)
            except BaseException:traceback.print_exc();raise


if __name__=='__main__':main()
