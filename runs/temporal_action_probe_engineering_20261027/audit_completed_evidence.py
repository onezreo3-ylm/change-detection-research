"""Saved-evidence reconciliation only; no torch, raw data or model inference."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.windows_shared_json import atomic_json as atomic,read_json as read

OUT=ROOT/'runs/publication_evidence_audit_20261027_r4'
if OUT.exists():raise FileExistsError('Preserve completed evidence audit')
OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={};checks={}
for relative in ('runs/local_composition_probe_v2_20261027','runs/temporal_action_probe_20261027'):
    study=ROOT/relative;auth=read(study/'authorization.json');completion=read(study/'completion.json')
    inputs=dict(auth['bound_sha256']);inputs.update(completion['input_sha256'])
    for name,expected in inputs.items():
        p=Path(name)
        # The frozen inputs are sources/checkpoints/saved records, never new raw test access.
        if sha(p)!=expected:raise ArithmeticError('Changed sealed evidence: '+name)
        manifest[name]=expected
    if not completion['actual_all_roles_inactive'] or completion['actual_worker_exit_code']!=0:
        raise ArithmeticError('Protected execution closure is missing')
    result=read(study/'result.json')
    if result['next_fitting_resource_gate'] is not False:raise ArithmeticError('Expected closed negative candidate')
    checks[relative]=dict(bound_files=len(inputs),normal_execution=True,scientific_gate=False,
        maximum_temperature_c=completion['maximum_observed_temperature_c'])
    for p in (study/'authorization.json',study/'completion.json',study/'result.json'):
        manifest[str(p)]=sha(p)
controls=ROOT/'runs/b0e_three_seed_controls_report_20261027/all_nine_cells.csv'
with controls.open(encoding='utf-8',newline='') as f:cells=list(csv.DictReader(f))
if {(r['seed'],r['mode']) for r in cells}!={(str(s),m) for s in (42,43,44) for m in ('b0e','b1','b2')}:
    raise ArithmeticError('Require all nine final controlled cells')
lookup={(r['seed'],r['mode']):r for r in cells}
deltas=[float(lookup[str(s),'b1']['corruption_mf1'])-float(lookup[str(s),'b0e']['corruption_mf1']) for s in (42,43,44)]
analysis_path=ROOT/'runs/b0e_replication_analysis_20261026/analysis.json';analysis=read(analysis_path)
if abs(statistics.mean(deltas)-.02280251288712587)>1e-14 or not all(d>0 for d in deltas):
    raise ArithmeticError('Canonical three-seed delta did not replay from saved cells')
for relative in ('runs/b0e_replication_evaluation_20261026/analysis_manifest.json',
    'runs/b0e_replication_evaluation_20261026/evaluation_session/owner_state.json',
    'runs/b0e_analysis_native_json_recovery_20261027/completion.json',
    'runs/bound_temporal_actions_engineering_20261027/full_project_tests.log',
    'runs/temporal_action_probe_engineering_20261027/full_project_tests.log',
    'docs/LOCAL_COMPOSITION_REAL_PROBE_RESULT_2026-10-27.md',
    'docs/TEMPORAL_ACTION_FEASIBILITY_RESULT_2026-10-27.md',
    'docs/NEXT_METHOD_STAGE_DECISION_2026-10-27.md',
    'runs/manuscript_working_draft_20261026/manuscript.md'):
    p=ROOT/relative;manifest[str(p)]=sha(p)
manifest[str(controls)]=sha(controls);manifest[str(analysis_path)]=sha(analysis_path)
ledger=read(ROOT/'runs/managed_continuation_20261024/compute_ledger.json')
if any(e.get('status')=='running' for e in ledger['entries']):raise RuntimeError('Unexpected active reservation')
if abs(sum(e['charged_seconds'] for e in ledger['entries'])-ledger['total_charged_seconds'])>1e-7:
    raise ArithmeticError('Compute ledger total mismatch')
atomic(OUT/'audit.json',dict(status='saved_evidence_reconciled',created_utc=datetime.now(timezone.utc).isoformat(),
    checks=checks,all_nine_control_cells=True,b1_minus_b0e_mf1=statistics.mean(deltas),
    raw_data_opened=False,model_inference=False,new_optimizer_updates=0,
    new_method_effective=False,novelty_verified=False,publication_ready=False,
    pending_budget_response=True,original_shared_charge_seconds=ledger['total_charged_seconds'],
    original_shared_cap_seconds=172800,verified_artifact_sha256=manifest))
print(dict(status='saved_evidence_reconciled',verified_unique_files=len(manifest),new_method_effective=False,publication_ready=False))
