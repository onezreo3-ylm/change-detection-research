"""One-shot closure of the existing protected job; no new inference or fitting."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import temporal_action_probe_supervision as sup
from scripts.windows_shared_json import atomic_json as atomic,read_json as read

OUT=Path(__file__).resolve().parent
_,cfg,v=sup.load_runtime();study=Path(cfg['study_output']);session=study/'probe_session'
marker=OUT/'closure_started.json'
if marker.exists():raise FileExistsError('One-shot closure already attempted')
atomic(marker,dict(started_utc=sup.common.now(),automatic_retry=False,new_fitting=False))
started=time.monotonic()
while time.monotonic()-started<3100:
    state=read(session/'owner_state.json')
    if state['status'].startswith('failed'):raise RuntimeError('Protected owner failed; preserve all evidence')
    if state['status']=='completed':
        proof=read(session/'attachment.json')
        if not any(sup.common.snapshot_identity(cfg,proof[r])['active'] for r in ('owner','worker','guard','watchdog')):break
    time.sleep(5)
else:raise TimeoutError('Closure wait ended; no retry or new phase')
cpu_started=time.perf_counter()
certificate=sup.finalize()
if certificate['status']!='completed_probe':raise RuntimeError('Pilot only; scientific utility remains unmeasured')
result=read(study/'result.json');passed=result['next_fitting_resource_gate']
lines=['# Temporal-placement feasibility result','',
    '2026-10-27. Fixed train-only exploratory upper-bound probe; no learned method or held-out evaluation.','',
    '| Group | Old16 oracle mF1 | New256 oracle mF1 | Best fixed mF1 | Gain vs best fixed (pp) | Extra vs old16 (pp) | Joint gates |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for group,row in result['groups'].items():
    best=max(row['constant_mf1'],key=row['constant_mf1'].get)
    lines.append(f"| {group} | {row['old16_oracle_mf1']:.9f} | {row['new256_oracle_mf1']:.9f} | {row['constant_mf1'][best]:.9f} ({best}) | {row['oracle_gain_over_best_fixed']*100:+.6f} | {row['temporal_oracle_increment']*100:+.6f} | {'PASS' if row['all_gates_passed'] else 'FAIL'} |")
lines.extend(['',
    'Fit and calibration each required +0.5pp over the strongest of seven constants and +0.2pp over the old16 oracle, plus original utility-sign/family gates. No rule changed after outcomes.',
    '',
    ('The action-space resource gates passed; this is only label-informed room. A distinct matched-learner, information/capacity and actual-cost protocol is still required before any training.' if passed else
     'The fixed temporal-placement candidate is CLOSED. Even label-informed selection did not satisfy the frozen joint utility gates. Do not fit its predictor, enlarge regions, change thresholds or reselect sources to rescue it.'),
    '',
    'The original a0/full-both counts and every old16 menu replayed exactly for all measured train cells. Pilot clean/brightness_l1 each actually checked all256 assignments on one original32 batch; main checked four fixed mixed assignments per batch and derived the remaining252. CUDA FP16 composition and final identity checks passed.',
    '',
    'All256 choices are diagnostic and use labels; this is not deployment accuracy, novelty certification or a universal impossibility result. Twelve fit and four calibration sources were previously used development data. Reference normalization/cache and generic routing/finite optimization remain known components. Decoder view counts do not establish a latency advantage.',
    '',
    f"Protected worker and independent owner/guard/watchdog exited normally; maximum recorded GPU temperature {certificate['maximum_observed_temperature_c']}C. No optimizer updates, raw test or Martigny access. Original cumulative48h ledger retained.",
    '',
    'Evidence: runs/temporal_action_probe_20261027/result.json, completion.json, probe_rows.jsonl, authorization.json. Protocol: docs/TEMPORAL_ACTION_FEASIBILITY_PROTOCOL_2026-10-27.md.',
    ''])
report=ROOT/'docs/TEMPORAL_ACTION_FEASIBILITY_RESULT_2026-10-27.md'
with report.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
stop=dict(status='resource_room_only_no_automatic_fitting' if passed else 'closed_failed_temporal_action_space',
    result_sha256=sup.common.sha(study/'result.json'),report_sha256=sup.common.sha(report),
    original_gates_changed=False,predictor_fitting_started=False,
    next_fitting_resource_gate=passed,publication_ready=False,created_utc=sup.common.now())
atomic(OUT/'scientific_decision.json',stop)
sup.record_cost('finalize',time.perf_counter()-cpu_started+2.,'Actual finalization and saved-result report wall plus2s allowance; no new inference.')
settled=sup.sync_costs(cfg)
path=ROOT/'runs/managed_continuation_20261024/status.json';old=path.read_bytes()
(OUT/'managed_status_before_temporal_closure.json').write_bytes(old)
status=read(path);ledger=read(cfg['ledger_path'])
status.update(updated_utc=sup.common.now(),current_authorization_status='completed',
    current_method_result=str(report.relative_to(ROOT)),
    temporal_action_probe_result_pending=False,temporal_action_probe_result='runs/temporal_action_probe_20261027/result.json',
    temporal_action_probe_completion='runs/temporal_action_probe_20261027/completion.json',
    temporal_action_probe_resource_gate=passed,temporal_action_probe_closed=not passed,
    new_method_admitted=False,new_method_training_started=False,new_method_research_gpu_job_started=False,
    bound_processes_scope='none_currently_active',bound_processes={},
    temporal_alternatives_real_FP16_probe_passed=True,
    shared_budget=dict(charged_seconds=ledger['total_charged_seconds'],upper_bound_seconds=ledger['batch_upper_bound_seconds'],automatic_extension=False),
    temporal_probe_cumulative_cost=settled,
    next_action='Reassess the scientific question from completed controls and failed/sufficient action-space evidence; no automatic predictor fitting, source/mask changes, held-out access or budget extension.',
    historical_state_preserved='runs/temporal_action_probe_engineering_20261027/managed_status_before_temporal_closure.json',
    historical_state_sha256=hashlib.sha256(old).hexdigest())
atomic(path,status)
print(dict(status=stop['status'],groups={g:{k:r[k] for k in ('oracle_gain_over_best_fixed','temporal_oracle_increment','gates')} for g,r in result['groups'].items()},
    cumulative_temporal_cost=settled,shared_remaining_seconds=ledger['batch_upper_bound_seconds']-ledger['total_charged_seconds']),flush=True)
