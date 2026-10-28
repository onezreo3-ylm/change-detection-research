"""Explicit same-rule saved-result recovery; archived analysis source unchanged."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np
import yaml

from scripts import analyze_b0e_replication as frozen

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/b0e_analysis_serialization_recovery_20261027.yaml'


def normalize_numpy_booleans(value):
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,dict):return {k:normalize_numpy_booleans(v) for k,v in value.items()}
    if isinstance(value,list):return [normalize_numpy_booleans(v) for v in value]
    if isinstance(value,tuple):return tuple(normalize_numpy_booleans(v) for v in value)
    return value


def strict_bytes(value):
    return (json.dumps(normalize_numpy_booleans(value),indent=2,allow_nan=False)+'\n').encode('utf-8')


def require_invalid_partial(raw):
    try:json.loads(raw)
    except json.JSONDecodeError:pass
    else:raise ValueError('Preserve every completed valid JSON analysis')
    if not raw.endswith(b'"source_paired_lower_positive": '):
        raise ValueError('Only the recorded NumPy Boolean write interruption is admitted')


def require_same_prefix(partial,complete):
    if not complete.startswith(partial):raise ValueError('Serialized numeric prefix differs from the original failed computation')


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (v.get('schema_version')!=1 or v.get('kind')!='saved_analysis_numpy_boolean_serialization_recovery_only'
            or any(v.get(k) is not False for k in ('change_statistical_rules','new_inference','raw_data_access'))
            or ROOT/v['original_analysis_config']!=frozen.CONFIG):
        raise ValueError('Serialization recovery scope changed')
    return v,ROOT/v['recovery_dir']


def stage():
    v,out=load()
    if out.exists():raise FileExistsError('Preserve existing recovery staging')
    partial=ROOT/v['failed_analysis'];raw=partial.read_bytes();require_invalid_partial(raw)
    log=ROOT/v['failure_log']
    if 'Object of type bool_ is not JSON serializable' not in log.read_text(encoding='utf-8'):
        raise ValueError('The actual failure does not match this narrow repair')
    frozen.verify_rules(ROOT/frozen.load_config()['output_dir'])
    out.mkdir()
    (out/'failed_analysis.json.partial').write_bytes(raw)
    paths=[CONFIG,ROOT/v['protocol'],Path(__file__).resolve(),ROOT/'tests/test_b0e_analysis_serialization_recovery.py',
        Path(frozen.__file__).resolve(),Path(frozen.original.__file__).resolve(),frozen.CONFIG,
        ROOT/frozen.load_config()['output_dir']/'analysis_rules_seal.json',ROOT/v['analysis_manifest'],log]
    audit=dict(status='staged_serialization_recovery_before_reexecution',
        failed_partial_sha256=frozen.sha(partial),bound_sha256={str(p):frozen.sha(p) for p in paths},
        original_rules_unchanged=True,new_inference=False,raw_data_access=False)
    (out/'stage.json').write_bytes(strict_bytes(audit))
    return dict(status=audit['status'],partial_bytes=len(raw))


def reconstruct(manifest_path):
    v=frozen.load_config();old_out=ROOT/v['output_dir'];frozen.verify_rules(old_out)
    manifest=json.loads(Path(manifest_path).read_bytes())
    if manifest['config_sha256']!=frozen.sha(frozen.CONFIG):raise ValueError('Original analysis config changed')
    historical_path=old_out/'historical_manifest.json'
    if manifest.get('historical_manifest_sha256')!=frozen.sha(historical_path):raise ValueError('Historical grid changed')
    historical=json.loads(historical_path.read_bytes())
    for seed,runs in historical['runs'].items():
        for mode,item in runs.items():
            if manifest['runs'].get(seed,{}).get(mode)!=item:raise ValueError('Historical result substituted')
    if set(manifest.get('new_completion_proofs',{}))!={'43','44'}:raise ValueError('Both actual new completions required')
    for seed,item in manifest['new_completion_proofs'].items():
        frozen.check_new_completion(item['path'],item['sha256'],int(seed))
    records,rows,sources,conditions=frozen.load_records(manifest,complete=True)
    counts=np.stack([np.stack([rows[(s,m)]['by_source'] for m in ('b0e','b1')],axis=1)
                     for s in frozen.SEEDS],axis=1)
    if counts.shape!=(64,3,2,45,4):raise ValueError('Original paired axes changed')
    interval=frozen.original.paired_bootstrap(counts,replicates=v['bootstrap_replicates'],seed=v['bootstrap_seed'])
    gate=frozen.primary_gate(records,interval,v['rules'])
    summary={m:{k:dict(mean=mean(records[s][m][k] for s in frozen.SEEDS),
                      sample_sd=stdev(records[s][m][k] for s in frozen.SEEDS))
                for k in frozen.original.METRICS} for m in frozen.MODES}
    result=dict(status='completed_saved_development_analysis',by_seed=records,summary=summary,primary=gate,
        source_count=64,condition_count=45,seeds=list(frozen.SEEDS),bootstrap_replicates=20000,
        bootstrap_seed=20261026,interval_scope='Conditional on these three trained seeds and the measured source pairs; not a geographic or training-seed population interval.',
        new_inference=False,test_accessed=False,publication_ready=False,
        manifest_sha256=frozen.sha(manifest_path),analysis_config_sha256=frozen.sha(frozen.CONFIG),
        source_sha256={str(Path(frozen.__file__).resolve()):frozen.sha(frozen.__file__),
                       str(Path(frozen.original.__file__).resolve()):frozen.sha(frozen.original.__file__)})
    return result,rows,conditions


def recover():
    v,out=load();audit=json.loads((out/'stage.json').read_bytes())
    for name,h in audit['bound_sha256'].items():
        if frozen.sha(name)!=h:raise ValueError('Staged recovery input changed: '+name)
    partial=ROOT/v['failed_analysis'];saved=(out/'failed_analysis.json.partial').read_bytes()
    if partial.read_bytes()!=saved:raise ValueError('Original partial changed after staging')
    checks=json.loads((out/'engineering_acceptance.json').read_bytes())
    if checks.get('status')!='passed_actual_full_suite_and_synthetic_serialization_smoke':
        raise ValueError('Actual repair acceptance required')
    for name,h in checks['bound_sha256'].items():
        if frozen.sha(name)!=h:raise ValueError('Repair acceptance input changed')
    result,rows,conditions=reconstruct(ROOT/v['analysis_manifest'])
    result['serialization_recovery']=dict(execution='explicit_derived_same_rule_serialization_adapter',
        adapter_sha256=frozen.sha(__file__),stage_sha256=frozen.sha(out/'stage.json'),
        old_partial_preserved=str(out/'failed_analysis.json.partial'),old_partial_sha256=audit['failed_partial_sha256'],
        conversion='NumPy bool_ to same-valued Python bool only',statistical_rules_changed=False)
    complete=strict_bytes(result);require_same_prefix(saved,complete)
    csv_path=partial.parent/'all_conditions.csv'
    if csv_path.exists():raise FileExistsError('Preserve completed condition table')
    (out/'recovered_analysis.json').write_bytes(complete)
    # Only the known invalid partial is replaced; its complete bytes remain above.
    if partial.read_bytes()!=saved:raise ValueError('Concurrent analysis change')
    temporary=partial.with_name('analysis.serialization-recovery.tmp')
    with temporary.open('xb') as f:f.write(complete)
    temporary.replace(partial)
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','condition','b0e_f1','b1_f1','b2_f1','b1_minus_b0e'])
        for seed in frozen.SEEDS:
            for i,cid in enumerate(conditions):
                values=[rows[(seed,m)]['condition_f1'][i] for m in frozen.MODES]
                writer.writerow([seed,cid,*values,values[1]-values[0]])
    frozen.verify_rules(partial.parent)
    certificate=dict(status='completed_same_rule_serialization_recovery',numeric_prefix_bit_exact=True,
        failed_partial_sha256=audit['failed_partial_sha256'],complete_analysis_sha256=frozen.sha(partial),
        recovered_copy_sha256=frozen.sha(out/'recovered_analysis.json'),condition_table_sha256=frozen.sha(csv_path),
        source_sha256=frozen.sha(__file__),rules_unchanged=True,new_inference=False)
    (out/'completion.json').write_bytes(strict_bytes(certificate))
    return dict(status=certificate['status'],decision=result['primary']['decision'],mean_deltas=result['primary']['mean_deltas'])


def smoke():
    _,out=load()
    fixture={'primary':{'checks':{'source_paired_lower_positive':np.bool_(True)},'mean_deltas':{'f1':.0123456789}}}
    raw=strict_bytes(fixture);parsed=json.loads(raw)
    if parsed['primary']['checks']['source_paired_lower_positive'] is not True:raise ValueError('Boolean conversion failed')
    if parsed['primary']['mean_deltas']['f1']!=fixture['primary']['mean_deltas']['f1']:raise ValueError('Numeric value changed')
    (out/'smoke.json').write_bytes(strict_bytes(dict(status='passed_CPU_synthetic_serialization_only',new_inference=False)))
    return dict(status='passed_CPU_synthetic_serialization_only')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','smoke','recover']);a=p.parse_args()
    print(json.dumps(globals()[a.action](),allow_nan=False))


if __name__=='__main__':main()
