"""One-shot closure of already authorized, frozen studies; no model code.

Every GPU launch goes through its existing independent protected entry point.
This script never retries a phase, changes a protocol, resets a budget, fits a
model, or opens raw datasets. It stops on any failure and preserves all files.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import levir_tail_risk_supervision as common

OUT=Path(__file__).resolve().parent
PYTHON='E:/Download/anaconda3/envs/geosa-basa/python.exe'


def event(value):
    value=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),**value)
    with (OUT/'serial_closure_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,separators=(',',':'))+'\n')
    print(json.dumps(value),flush=True)


def run_step(name,module,args):
    path=OUT/(name+'.log')
    with path.open('x',encoding='utf-8') as log:
        started=time.perf_counter()
        command=[PYTHON,'-X','utf8','-m',module,*args]
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,encoding='utf-8')
    elapsed=time.perf_counter()-started
    event(dict(stage=name,argv=command,returncode=result.returncode,subprocess_wall_seconds=elapsed,
               log=str(path),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if result.returncode: raise RuntimeError('Stopped without retry: '+name)
    return elapsed


def wait_normal_exit(session,*,label,maximum_seconds):
    started=time.monotonic();last=None
    while time.monotonic()-started<maximum_seconds:
        owner_path=session/'owner_state.json'
        if owner_path.exists():
            owner=common.read(owner_path)
            if owner.get('status','').startswith('failed'):
                raise RuntimeError('Protected owner failed; preserve: '+label)
            progress_path=session/'progress.json'
            if progress_path.exists():
                progress=common.read(progress_path)
                key=(progress.get('stage'),progress.get('seed'),progress.get('condition'),progress.get('condition_id'),progress.get('completed_conditions'))
                if key!=last and (progress.get('completed_conditions',0)%5==0 or progress.get('condition_id')):
                    event(dict(stage=label+'_progress',progress=progress));last=key
            if owner.get('status')=='completed' and (session/'attachment.json').exists():
                attachment=common.read(session/'attachment.json')
                active={role:common.snapshot_identity({},attachment[role])['active']
                        for role in ('owner','worker','guard','watchdog')}
                if not any(active.values()):
                    event(dict(stage=label+'_actual_all_roles_inactive',elapsed_wait_seconds=time.monotonic()-started))
                    return
        time.sleep(5)
    raise TimeoutError('Bounded completion wait ended without launching another phase: '+label)


def main():
    if (OUT/'serial_closure_started.json').exists(): raise FileExistsError('One-shot coordinator already exists')
    (OUT/'serial_closure_started.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        automatic_retry=False,new_model_fitting=False,shared_budget_extension=False),indent=2)+'\n',encoding='utf-8')
    wait_normal_exit(ROOT/'runs/b0e_replication_evaluation_20261026/evaluation_session',label='fixed_evaluation',maximum_seconds=3700)
    run_step('fixed_evaluation_certify','scripts.evaluate_b0e_replications',['certify'])
    seconds=run_step('paired_saved_analysis','scripts.analyze_b0e_replication',[
        'analyze','--manifest',str(ROOT/'runs/b0e_replication_evaluation_20261026/analysis_manifest.json')])
    run_step('paired_analysis_cost','scripts.fixed_development_evaluation_supervision',[
        'record-cost','--phase','analysis','--seconds',str(seconds+2),
        '--basis','Actual saved-analysis subprocess wall plus2seconds coordination allowance; no new inference.'])
    run_step('fixed_evaluation_final_settlement','scripts.fixed_development_evaluation_supervision',['settle-costs'])
    analysis=common.read(ROOT/'runs/b0e_replication_analysis_20261026/analysis.json')
    event(dict(stage='paired_analysis_completed',decision=analysis['primary']['decision'],
               mean_deltas=analysis['primary']['mean_deltas'],publication_ready=False))
    run_step('local_probe_engineering_settlement','scripts.local_composition_probe_supervision',['settle-costs'])
    run_step('local_probe_seal','scripts.local_composition_probe_supervision',[
        'seal','--checks',str(OUT/'full_project_tests.log')])
    run_step('local_probe_launch','scripts.local_composition_probe_supervision',['launch','--mode','probe'])
    wait_normal_exit(ROOT/'runs/local_composition_probe_20261027/probe_session',label='local_probe',maximum_seconds=3550)
    run_step('local_probe_final_proof','scripts.local_composition_probe_supervision',['finalize'])
    run_step('local_probe_final_settlement','scripts.local_composition_probe_supervision',['settle-costs'])
    path=ROOT/'runs/local_composition_probe_20261027/result.json'
    if path.exists():
        result=common.read(path)
        event(dict(stage='local_probe_completed',action_space_resource_gate=result['next_fitting_resource_gate'],
            group_oracle_margins={name:row['reference_oracle_gain_over_better_fixed'] for name,row in result['groups'].items()},
            automatic_head_fitting=False,equal_cost_postselection_control_required=True))
    else:
        event(dict(stage='local_probe_pilot_only',admission=common.read(ROOT/'runs/local_composition_probe_20261027/pilot_admission.json'),
                   automatic_head_fitting=False))
    event(dict(stage='one_shot_closure_completed',publication_ready=False))


if __name__=='__main__':
    try: main()
    except BaseException as error:
        event(dict(stage='stopped_preserve_without_retry',error=type(error).__name__+': '+str(error)))
        raise
