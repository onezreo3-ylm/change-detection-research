"""Preserve successful tests; repair cost-summary field mismatch without rerunning."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

out=Path(__file__).resolve().parent
path=out/'full_verification.json'
if path.exists():raise FileExistsError('Existing verification preserved')
log=out/'full_project_tests.log';text=log.read_text(encoding='utf-8')
match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*\Z',text)
if not match or int(match[1])!=626:raise ValueError('Original successful 626-test log required')
stat=log.stat();log_lifetime=stat.st_mtime-stat.st_ctime
targeted=read(out/'targeted_tests_cost.json')['elapsed_seconds']
smoke=read(out/'original_architecture_smoke_cost.json')['actual_subprocess_wall_seconds']
charged=max(float(match[2]),log_lifetime)+targeted+smoke+30.
ledger_path=ROOT/'runs/managed_continuation_20261024/compute_ledger.json'
ledger=read(ledger_path);identifier='bound_temporal_actions_engineering_20261027'
if any(e.get('status')=='running' or e.get('id')==identifier for e in ledger['entries']):
    raise RuntimeError('Active reservation or duplicate charge')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(status='passed_all626_and_original_architecture_CPU_smoke',full_project_count=626,
    full_suite_reported_seconds=float(match[2]),test_log_lifetime_seconds=log_lifetime,
    elapsed_perf_counter_lost_after_post_test_field_error=True,
    cost_basis='Maximum of successful suite time and Windows log lifetime, measured final targeted/smoke process walls, plus30s conservative allowance for initial7-test run, lost process tail and repair. Prior charges retained.',
    additional_allowance_seconds=30.,charged_seconds=charged,
    preserved_failed_summary_harness_sha256=sha(out/'verify_full.py'),
    post_test_error="KeyError: actual_subprocess_wall_seconds; targeted cost used elapsed_seconds",
    tests_or_model_rerun=False,
    bound_sha256={str(p):sha(p) for p in [log,ROOT/'pcdc/bound_temporal_actions.py',ROOT/'tests/test_bound_temporal_actions.py',
        ROOT/'scripts/smoke_bound_temporal_actions_cpu.py',ROOT/'configs/bound_temporal_actions_cpu_20261027.yaml',
        ROOT/'runs/bound_temporal_actions_cpu_20261027/result.json']},
    real_data_loaded=False,research_training=False,optimizer_updates=0)
ledger['entries'].append(dict(id=identifier,kind='temporal_actions_engineering',status='completed',
    charged_seconds=charged,recorded_utc=datetime.now(timezone.utc).isoformat(),
    basis=result['cost_basis'],research_training=False,optimizer_updates=0))
ledger['total_charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
if ledger['total_charged_seconds']>ledger['batch_upper_bound_seconds']:raise RuntimeError('Original cap exceeded')
atomic(ledger_path,ledger);atomic(path,result)
status_path=ROOT/'runs/managed_continuation_20261024/status.json';previous=status_path.read_bytes()
snapshot=out/'managed_status_before_completion_update.json'
if snapshot.exists():raise FileExistsError('Preserve previous status snapshot')
snapshot.write_bytes(previous);status=read(status_path)
status.update(updated_utc=datetime.now(timezone.utc).isoformat(),current_authorization_status='completed_negative_probe',
    current_method_candidate='Mutually exclusive temporal action placement; CPU engineering passed; real utility and novelty unproved',
    local_probe_v2_result='runs/local_composition_probe_v2_20261027/result.json',
    local_probe_v2_completion='runs/local_composition_probe_v2_20261027/completion.json',
    local_probe_v2_scientific_gate_passed=False,local_probe_fixed_four_region_candidate_closed=True,
    local_probe_v2_scientific_stop='runs/local_composition_resource_v2_engineering_20261027/scientific_stop.json',
    current_method_result='docs/LOCAL_COMPOSITION_REAL_PROBE_RESULT_2026-10-27.md',
    new_method_training_started=False,new_method_admitted=False,new_method_research_gpu_job_started=False,
    full_project_tests_count=626,full_project_tests_passed=True,full_project_tests_pending_gpu_idle=False,
    full_project_tests_log=str(log.relative_to(ROOT)),full_project_tests_sha256=sha(log),
    temporal_alternatives_production_wrapper_prepared=True,temporal_alternatives_real_FP16_probe_passed=False,
    temporal_bound_engineering_verification=str(path.relative_to(ROOT)),
    shared_budget=dict(charged_seconds=ledger['total_charged_seconds'],upper_bound_seconds=ledger['batch_upper_bound_seconds'],automatic_extension=False),
    historical_state_preserved=str(snapshot.relative_to(ROOT)),historical_state_sha256=hashlib.sha256(previous).hexdigest(),
    next_action='Finish primary closest-method alignment and freeze a distinct exploratory temporal-placement train-only feasibility protocol. No head fitting, raw test/Martigny or automatic budget extension.')
status.pop('bound_processes',None)
status['bound_processes_scope']='none_currently_active'
atomic(status_path,status)
print(dict(status=result['status'],charged_seconds=charged,shared_remaining_seconds=ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds']))
