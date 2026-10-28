"""One-shot verification and charge under the user-retained original cap."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.windows_shared_json import atomic_json as atomic,read_json as read
from scripts import levir_tail_risk_supervision as common

OUT=Path(__file__).resolve().parent
PYTHON='E:/Download/anaconda3/envs/geosa-basa/python.exe'
ledger_path=ROOT/'runs/managed_continuation_20261024/compute_ledger.json'
ledger=read(ledger_path)
if ledger['batch_upper_bound_seconds']!=172800 or any(e.get('status')=='running' for e in ledger['entries']):
    raise RuntimeError('Original cap and inactive ledger required')
if (OUT/'verification.json').exists():raise FileExistsError('Preserve completed verification')
proof=read(ROOT/'runs/temporal_action_probe_20261027/probe_session/attachment.json')
if any(common.snapshot_identity({},proof[r])['active'] for r in ('owner','worker','guard','watchdog')):
    raise RuntimeError('Previous actual process remains active')
lease=ROOT/'runs/managed_continuation_20261024/gpu_study.lock'
entries=[]
def step(name,module,args):
    path=OUT/(name+'.log');started=time.perf_counter()
    with path.open('x',encoding='utf-8') as log:
        p=subprocess.run([PYTHON,'-X','utf8','-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-started
    entries.append(dict(name=name,wall_seconds=elapsed,returncode=p.returncode,log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    atomic(OUT/'verification_phases.json',dict(entries=entries))
    if p.returncode:raise RuntimeError('Stopped without retry: '+name)

with lease.open('x',encoding='utf-8') as f:f.write('exclusive engineering tests after user retained48h cap')
try:
    step('full_project_tests','unittest',['discover','-s','tests','-v'])
finally:lease.unlink()
step('CPU_smoke','scripts.screen_task_projection',[])
log=(OUT/'full_project_tests.log').read_text(encoding='utf-8')
match=re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*\Z',log)
if not match or int(match[1])!=641:raise ArithmeticError('Require the complete current641-test suite')
charged=sum(e['wall_seconds'] for e in entries)+6.235+180.
if charged>600:raise RuntimeError('Preparation subcap exceeded; preserve measured costs')
ledger=read(ledger_path);identifier='budget_retained_method_screen_preparation_20261027'
if any(e.get('status')=='running' or e.get('id')==identifier for e in ledger['entries']):
    raise RuntimeError('Active reservation or duplicate charge')
ledger['entries'].append(dict(id=identifier,kind='method_screen_preparation',status='completed',charged_seconds=charged,
    optimizer_updates=0,research_training=False,recorded_utc=datetime.now(timezone.utc).isoformat(),
    basis='Measured full-suite and CPU smoke subprocess walls, measured6.235s final primary search, explicit additional180s conservative allowance for previous primary requests/section reads, targeted tests, import/coordination. Not claimed exact aggregate wall. Original charges/cap retained.'))
ledger['total_charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
if ledger['total_charged_seconds']>172800:raise RuntimeError('Original48h cap exceeded')
atomic(ledger_path,ledger)
result=dict(status='passed641_tests_and_algebraic_CPU_smoke',phases=entries,charged_seconds=charged,
    compute_subcap_seconds=600,original_shared_cap_seconds=172800,
    remaining_seconds=172800-ledger['total_charged_seconds'],real_data_read=False,new_research_updates=0,
    new_method_admitted=False,publication_ready=False)
atomic(OUT/'verification.json',result)
print(result,flush=True)
