"""One-shot engineering, sealed protected launch; never fits a predictor."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.windows_shared_json import atomic_json as atomic
from scripts import temporal_action_probe_supervision as supervisor

PYTHON='E:/Download/anaconda3/envs/geosa-basa/python.exe'
OUT=ROOT/'runs/temporal_action_probe_engineering_20261027'
OUT.mkdir(exist_ok=True)
if (OUT/'admission_started.json').exists():raise FileExistsError('No automatic retry')
atomic(OUT/'admission_started.json',dict(status='engineering_only_before_seal',automatic_retry=False,new_fitting=False))

def step(name,module,args,phase=None,allowance=1.):
    path=OUT/(name+'.log');started=time.perf_counter()
    with path.open('x',encoding='utf-8') as log:
        p=subprocess.run([PYTHON,'-X','utf8','-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    seconds=time.perf_counter()-started
    atomic(OUT/(name+'_cost.json'),dict(actual_subprocess_wall_seconds=seconds,returncode=p.returncode))
    if phase:
        supervisor.record_cost(phase,seconds+allowance,
            'Measured '+name+' subprocess wall plus explicit'+str(allowance)+'s allowance; includes prior targeted checks/literature in prepare allowance, no removed old charges.')
    if p.returncode:raise RuntimeError('Stopped without retry: '+name)
    print(dict(step=name,returncode=p.returncode,seconds=seconds),flush=True)

step('prepare','scripts.run_temporal_action_probe',['prepare-metadata'],'prepare',40.)
step('smoke','scripts.run_temporal_action_probe',['smoke'],'smoke')
_,cfg,_=supervisor.load_runtime();lease=Path(cfg['global_lease'])
if any(e.get('status')=='running' for e in supervisor.read(cfg['ledger_path'])['entries']):
    raise RuntimeError('An active research reservation exists')
supervisor.external_check(cfg)
with lease.open('x',encoding='utf-8') as f:f.write('exclusive temporal engineering full suite')
try:step('full_project_tests','unittest',['discover','-s','tests','-v'],'full_tests',2.)
finally:lease.unlink()
supervisor.sync_costs(cfg)
step('seal','scripts.temporal_action_probe_supervision',['seal','--checks',str(OUT/'full_project_tests.log')])
supervisor.sync_costs(cfg)
step('launch','scripts.temporal_action_probe_supervision',['launch','--mode','probe'])
atomic(OUT/'admission_completed.json',dict(status='protected_train_only_probe_launched',new_fitting=False,automatic_retry=False))
