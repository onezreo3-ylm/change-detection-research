"""Preserve original Windows JSON text bytes; same frozen saved-result rules."""
import argparse
import csv
import io
import json
from pathlib import Path

import yaml

from scripts import recover_b0e_analysis_serialization as prior

ROOT=prior.ROOT
CONFIG=ROOT/'configs/b0e_analysis_native_json_recovery_20261027.yaml'


def native_json_bytes(value):
    stream=io.StringIO(newline='\r\n')
    json.dump(prior.normalize_numpy_booleans(value),stream,indent=2,allow_nan=False)
    stream.write('\n')
    return stream.getvalue().encode('utf-8')


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (v.get('schema_version')!=1 or v.get('kind')!='preserve_original_CRLF_saved_analysis_serialization'
            or v.get('original_text_newline')!='CRLF' or v.get('new_inference') is not False
            or v.get('change_values_or_statistical_rules') is not False):raise ValueError('Native JSON recovery scope changed')
    return v,ROOT/v['output_dir']


def stage():
    v,out=load()
    if out.exists():raise FileExistsError('Preserve native recovery staging')
    old= json.loads((ROOT/v['previous_stage']).read_bytes())
    for name,h in old['bound_sha256'].items():
        if prior.frozen.sha(name)!=h:raise ValueError('Original recovery or frozen input changed')
    pv,po=prior.load();partial=ROOT/pv['failed_analysis'];raw=partial.read_bytes()
    prior.require_invalid_partial(raw)
    if raw!=(po/'failed_analysis.json.partial').read_bytes():raise ValueError('Original partial changed')
    out.mkdir();(out/'failed_analysis.json.partial').write_bytes(raw)
    paths=[CONFIG,ROOT/v['protocol'],ROOT/v['previous_stage'],ROOT/v['previous_recovery_failure_log'],
        Path(__file__).resolve(),ROOT/'tests/test_b0e_native_json_recovery.py']
    audit=dict(status='staged_original_CRLF_recovery',bound_sha256={str(p):prior.frozen.sha(p) for p in paths},
        old_bound_sha256=old['bound_sha256'],old_partial_sha256=prior.frozen.sha(partial),
        same_statistical_rules=True,new_inference=False)
    (out/'stage.json').write_bytes(prior.strict_bytes(audit))
    return dict(status=audit['status'])


def recover():
    v,out=load();audit=json.loads((out/'stage.json').read_bytes())
    for mapping in (audit['bound_sha256'],audit['old_bound_sha256']):
        for name,h in mapping.items():
            if prior.frozen.sha(name)!=h:raise ValueError('Staged input changed: '+name)
    acceptance=json.loads((out/'engineering_acceptance.json').read_bytes())
    if acceptance['status']!='passed_actual_full_suite_and_native_writer_smoke':raise ValueError('Actual engineering acceptance required')
    for name,h in acceptance['bound_sha256'].items():
        if prior.frozen.sha(name)!=h:raise ValueError('Acceptance input changed')
    pv,_=prior.load();partial=ROOT/pv['failed_analysis'];raw=(out/'failed_analysis.json.partial').read_bytes()
    if partial.read_bytes()!=raw:raise ValueError('Preserve an already changed/completed analysis')
    result,rows,conditions=prior.reconstruct(ROOT/pv['analysis_manifest'])
    result['serialization_recovery']=dict(execution='explicit_same_rule_numpy_boolean_and_original_CRLF_adapter',
        adapter_sha256=prior.frozen.sha(__file__),prior_adapter_sha256=prior.frozen.sha(prior.__file__),
        stage_sha256=prior.frozen.sha(out/'stage.json'),old_partial_preserved=str(out/'failed_analysis.json.partial'),
        old_partial_sha256=audit['old_partial_sha256'],statistical_rules_changed=False,
        value_conversion='NumPy bool_ to same-valued Python bool only',original_text_newline='CRLF')
    complete=native_json_bytes(result);prior.require_same_prefix(raw,complete)
    candidate=out/'recovered_analysis.json'
    with candidate.open('xb') as f:f.write(complete)
    csv_path=partial.parent/'all_conditions.csv'
    if csv_path.exists():raise FileExistsError('Preserve complete conditions table')
    temporary=partial.with_name('analysis.native-json-recovery.tmp')
    if partial.read_bytes()!=raw:raise ValueError('Concurrent analysis change')
    with temporary.open('xb') as f:f.write(complete)
    temporary.replace(partial)
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','condition','b0e_f1','b1_f1','b2_f1','b1_minus_b0e'])
        for seed in prior.frozen.SEEDS:
            for i,cid in enumerate(conditions):
                values=[rows[(seed,m)]['condition_f1'][i] for m in prior.frozen.MODES]
                writer.writerow([seed,cid,*values,values[1]-values[0]])
    prior.frozen.verify_rules(partial.parent)
    certificate=dict(status='completed_same_rule_native_json_recovery',numeric_prefix_bit_exact=True,
        old_partial_sha256=audit['old_partial_sha256'],canonical_sha256=prior.frozen.sha(partial),
        recovered_copy_sha256=prior.frozen.sha(candidate),conditions_sha256=prior.frozen.sha(csv_path),
        no_new_inference=True,statistical_rules_changed=False)
    (out/'completion.json').write_bytes(prior.strict_bytes(certificate))
    return dict(status=certificate['status'],decision=result['primary']['decision'],mean_deltas=result['primary']['mean_deltas'])


def smoke():
    import numpy as np
    _,out=load();value=dict(x=.123456789,checks=[np.bool_(True),np.bool_(False)],text='line1\nline2')
    encoded=native_json_bytes(value)
    if b'\r\n' not in encoded or b'\r\r\n' in encoded:raise ValueError('Original CRLF text format failed')
    if json.loads(encoded)!=prior.normalize_numpy_booleans(value):raise ValueError('Values changed')
    result=dict(status='passed_CPU_synthetic_native_writer_only',new_inference=False)
    (out/'smoke.json').write_bytes(prior.strict_bytes(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','smoke','recover']);a=p.parse_args()
    print(json.dumps(globals()[a.action](),allow_nan=False))


if __name__=='__main__':main()
