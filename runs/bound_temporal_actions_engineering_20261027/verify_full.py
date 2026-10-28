"""One-shot engineering verification, exclusive GPU lease, no research inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.windows_shared_json import atomic_json, read_json
from scripts import levir_tail_risk_supervision as common

OUT=Path(__file__).resolve().parent
PYTHON='E:/Download/anaconda3/envs/geosa-basa/python.exe'
ledger_path=ROOT/'runs/managed_continuation_20261024/compute_ledger.json'
lease=ROOT/'runs/managed_continuation_20261024/gpu_study.lock'

def main():
    result_path=OUT/'full_verification.json'
    if result_path.exists() or (OUT/'full_project_tests.log').exists():
        raise FileExistsError('One-shot verification evidence already exists')
    ledger=read_json(ledger_path)
    if any(e.get('status')=='running' for e in ledger['entries']):
        raise RuntimeError('Active research reservation')
    completion=read_json(ROOT/'runs/local_composition_probe_v2_20261027/completion.json')
    if completion['actual_all_roles_inactive'] is not True:
        raise RuntimeError('Previous protected study is not closed')
    proof=read_json(ROOT/'runs/local_composition_probe_v2_20261027/probe_session/attachment.json')
    if any(common.snapshot_identity({},proof[r])['active'] for r in ('owner','worker','guard','watchdog')):
        raise RuntimeError('Previous process remains active')
    started=time.perf_counter()
    with lease.open('x',encoding='utf-8') as f:
        json.dump({'purpose':'exclusive_full_engineering_tests','source':str(__file__)},f)
    try:
        with (OUT/'full_project_tests.log').open('x',encoding='utf-8') as log:
            proc=subprocess.run([PYTHON,'-X','utf8','-m','unittest','discover','-s','tests','-v'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-started
        costs=[read_json(OUT/name)['actual_subprocess_wall_seconds'] for name in
               ('targeted_tests_cost.json','original_architecture_smoke_cost.json')]
        charged=elapsed+sum(costs)+5.
        ledger=read_json(ledger_path)
        identifier='bound_temporal_actions_engineering_20261027'
        if any(e.get('status')=='running' or e.get('id')==identifier for e in ledger['entries']):
            raise RuntimeError('Ledger changed or duplicate charge')
        ledger['entries'].append(dict(id=identifier,kind='temporal_actions_engineering',status='completed',
            charged_seconds=charged,recorded_utc=datetime.now(timezone.utc).isoformat(),
            research_training=False,optimizer_updates=0,
            basis='Measured final targeted tests, original-architecture synthetic CPU smoke, full suite; 5s additional allowance. Retains all prior charges.'))
        ledger['total_charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
        atomic_json(ledger_path,ledger)
        result=dict(full_project_returncode=proc.returncode,full_project_seconds=elapsed,
            all_engineering_charge_seconds=charged,total_shared_charge_seconds=ledger['total_charged_seconds'],
            original_shared_cap_seconds=ledger['batch_upper_bound_seconds'],
            full_log_sha256=hashlib.sha256((OUT/'full_project_tests.log').read_bytes()).hexdigest(),
            bound_source_sha256=hashlib.sha256((ROOT/'pcdc/bound_temporal_actions.py').read_bytes()).hexdigest(),
            real_data_loaded=False,research_optimizer_updates=0)
        atomic_json(result_path,result)
        print(json.dumps(result),flush=True)
        if proc.returncode or ledger['total_charged_seconds']>ledger['batch_upper_bound_seconds']:
            raise RuntimeError('Verification or original budget gate failed')
    finally:
        lease.unlink()

if __name__=='__main__':main()
