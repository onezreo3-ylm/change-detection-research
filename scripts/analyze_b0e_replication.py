"""Saved-result analysis of the complete B0-E/B1/B2 seed grid; no torch/raw data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,stdev
from datetime import datetime,timezone

import numpy as np
import yaml

from scripts import analyze_r1 as original

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/b0e_replication_analysis_20261026.yaml'
SEEDS=(42,43,44)
MODES=('b0e','b1','b2')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def load_config():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (v.get('schema_version')!=1 or v.get('scope')!='saved_development_results_only' or
        v.get('seeds')!=list(SEEDS) or v.get('modes')!=list(MODES) or
        v.get('primary_contrast')!='b1_minus_b0e' or v.get('bootstrap_replicates')!=20000 or
        v.get('bootstrap_seed')!=20261026 or v.get('source_count')!=64 or
        v.get('condition_count')!=45 or v.get('test_or_martigny_access') is not False):
        raise ValueError('Frozen saved-result analysis scope changed')
    return v


def bind_run(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT/'runs'):
        raise ValueError('Only saved workspace results may be bound')
    return dict(directory=str(directory),sha256={name:sha(directory/name) for name in original.ARTIFACTS})


def primary_gate(records,interval,rules):
    """Predeclared mean effects and safeguards; never rescue with a smaller gap."""
    if set(records)!=set(SEEDS):raise ValueError('All three seeds required')
    interval=np.asarray(interval,dtype=float)
    if interval.shape!=(3,) or not np.isfinite(interval).all() or np.any(np.diff(interval)<0):
        raise ValueError('Ordered finite lower/median/upper interval required')
    deltas={s:{k:records[s]['b1'][k]-records[s]['b0e'][k] for k in
        ('corruption_mf1','clean_f1','pooled_recall','background_fpr')} for s in SEEDS}
    avg={k:mean(deltas[s][k] for s in SEEDS) for k in deltas[42]}
    if not all(np.isfinite(x) for row in deltas.values() for x in row.values()):
        raise ValueError('Nonfinite effect')
    wins=sum(deltas[s]['corruption_mf1']>0 for s in SEEDS)
    checks=dict(
        mean_mf1_gain=avg['corruption_mf1']>=rules['minimum_mean_mf1_delta'],
        positive_seed_count=wins>=rules['minimum_positive_seeds'],
        clean_f1_protection=avg['clean_f1']>=rules['minimum_mean_clean_f1_delta'],
        corruption_recall_protection=avg['pooled_recall']>=rules['minimum_mean_pooled_recall_delta'],
        background_fpr_protection=avg['background_fpr']<=rules['maximum_mean_background_fpr_delta'],
        source_paired_lower_positive=(interval[0]>0 or not rules['require_source_paired_lower_above_zero']))
    return dict(checks=checks,all_passed=all(checks.values()),mean_deltas=avg,
        by_seed_deltas=deltas,positive_seeds=wins,source_paired_interval=interval.tolist(),
        decision='supported_on_measured_development_protocol' if all(checks.values()) else 'joint_benefit_claim_not_supported')


def validate_source_support(row,reference=None):
    counts=row['by_source']
    if counts.shape!=(64,45,4) or np.any(counts<0) or np.any(counts.sum(axis=-1)!=1024**2):
        raise ValueError('Each original source/condition must contain exactly1024square pixels')
    foreground=counts[:,:,2]+counts[:,:,3]
    if np.any(foreground!=foreground[:,0,None]):
        raise ValueError('Non-geometric conditions changed source label support')
    support=foreground[:,0]
    if reference is not None and not np.array_equal(reference,support):
        raise ValueError('Methods/seeds do not share the same source label support')
    return support


def load_records(manifest,*,complete):
    runs=manifest['runs'];expected=[(s,m) for s in SEEDS for m in MODES]
    records={s:{} for s in SEEDS};sources=conditions=support=None;counts={}
    for seed,mode in expected:
        item=runs.get(str(seed),{}).get(mode)
        if item is None:
            if complete:raise ValueError(f'Missing required result {mode}/{seed}; no partial primary claim')
            continue
        if set(item)!={'directory','sha256'}:raise ValueError('Unexpected artifact binding')
        row=original._load_run(ROOT,item,seed,mode)
        if sources is None:sources,conditions=row['source_ids'],row['condition_ids']
        if row['source_ids']!=sources or row['condition_ids']!=conditions:
            raise ValueError('Methods/seeds do not share the exact source/condition axes')
        support=validate_source_support(row,support)
        clean=original._read_json(Path(item['directory'])/'clean_metrics.json',item['sha256']['clean_metrics.json'])
        cc=original._counts(clean['counts'])
        if int(cc.sum())!=64*1024**2 or int(cc[2]+cc[3])!=int(support.sum()):
            raise ValueError('Clean and corruption do not have the same complete label/pixel support')
        records[seed][mode]=row['metrics'];counts[(seed,mode)]=row
    return records,counts,sources,conditions


def historical_stage():
    v=load_config();out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Preserve existing historical staging')
    manifest_path=ROOT/v['historical_r1_manifest']
    old=yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    runs={str(s):{} for s in SEEDS}
    for seed in SEEDS:
        for mode in ('b1','b2'):
            item=old['runs'][str(seed)][mode]
            runs[str(seed)][mode]=dict(directory=str((manifest_path.parent/item['directory']).resolve()),sha256=item['sha256'])
    runs['42']['b0e']=bind_run(ROOT/v['historical_b0e42'])
    manifest=dict(schema_version=1,runs=runs,scope='historical_results_with_two_explicitly_missing_controls',
        config_sha256=sha(CONFIG),historical_r1_manifest_sha256=sha(manifest_path))
    records,_,sources,conditions=load_records(manifest,complete=False)
    out.mkdir(parents=True)
    write_new(out/'historical_manifest.json',manifest)
    result=dict(status='seven_historical_results_checked_two_controls_pending',
        historical_metrics=records,source_ids=sources,condition_ids=conditions,
        missing=[dict(seed=43,mode='b0e'),dict(seed=44,mode='b0e')],
        raw_data_read=False,new_inference=False,primary_three_seed_result=None,
        caveat='Historical snapshot and arithmetic check; no new independent effectiveness evidence.')
    write_new(out/'historical_verification.json',result)
    return dict(status=result['status'],sources=len(sources),conditions=len(conditions),primary_claim=False)


def check_new_completion(cert_path,expected_sha,expected_seed):
    path=Path(cert_path).resolve()
    allowed=ROOT/load_config()['new_evaluation_root']
    if not path.is_relative_to(allowed):raise ValueError('New completion proof must be in the explicit evaluation output')
    cert=original._read_json(path,expected_sha)
    if (expected_seed not in (43,44) or cert.get('seed')!=expected_seed or
        cert.get('status')!='completed_independent_external_supervision_audit' or cert.get('test_accessed') is not False):
        raise ValueError('New evaluation completion is not certified')
    if set(cert.get('roles',{}))!={'owner','worker','guard','watchdog'}:
        raise ValueError('All four actual supervised process records required')
    required={str((allowed/'evaluation_session'/(r+'_state.json' if r!='worker' else 'worker_result.json')).resolve())
              for r in cert['roles']}
    required.update(str((allowed/f'seed{expected_seed}'/n).resolve()) for n in
                    ('summary.json','conditions.json','clean_metrics.json'))
    if not required.issubset(cert.get('bound_sha256',{})):
        raise ValueError('Completion does not commit actual process and result evidence')
    for name,expected in cert['bound_sha256'].items():
        # This certificate commits derived evidence only; avoid raw-data traversal.
        p=Path(name).resolve()
        if not p.is_relative_to(ROOT/'runs') or p.suffix.lower() not in ('.json','.yaml','.md','.log'):
            raise ValueError('Completion proof includes non-derived data')
        if sha(p)!=expected:raise ValueError('New completion evidence changed: '+name)
    states={}
    for role,name in cert['roles'].items():
        expected=allowed/'evaluation_session'/(role+'_state.json' if role!='worker' else 'worker_result.json')
        if Path(name).resolve()!=expected.resolve():raise ValueError('Unexpected supervision record path')
        states[role]=json.loads(Path(name).read_text(encoding='utf-8'))
    if (states['owner'].get('status')!='completed' or states['owner'].get('worker_exit_code')!=0 or
        states['worker'].get('status')!='completed' or states['worker'].get('test_accessed') is not False or
        any(states[r].get('status')!='worker_exited' or 'error' in states[r] for r in ('guard','watchdog'))):
        raise ValueError('Actual protected execution did not complete normally')
    maximum=states['guard'].get('maximum_observed_temperature_c')
    if type(maximum) is not int or not 0<=maximum<86:
        raise ValueError('Missing/invalid actual thermal evidence')
    return cert


def rule_sources():
    return [CONFIG,Path(__file__).resolve(),Path(original.__file__).resolve(),
            ROOT/'tests/test_b0e_replication_analysis.py']


def freeze_rules():
    v=load_config();out=ROOT/v['output_dir']
    historical=out/'historical_manifest.json'
    if not historical.is_file():raise ValueError('Verified historical grid required before rule freeze')
    evaluation=ROOT/v['new_evaluation_root']
    if any((evaluation/f'seed{s}'/'conditions.json').exists() for s in (43,44)):
        raise ValueError('Cannot freeze analysis after new corruption results exist')
    value=dict(status='frozen_before_new_corruption_scores',created_utc=datetime.now(timezone.utc).isoformat(),
        bound_sha256={str(p):sha(p) for p in [*rule_sources(),historical]},
        full_suite_for_new_analysis='pending_until_active_CUDA_training_is_idle',
        smoke_kind='CPU_fixture_paired_bootstrap_and_seven_saved_historical_result_replay')
    write_new(out/'analysis_rules_seal.json',value)
    return dict(status=value['status'],bound_files=len(value['bound_sha256']))


def verify_rules(out):
    seal=json.loads((out/'analysis_rules_seal.json').read_text(encoding='utf-8'))
    expected={str(p) for p in [*rule_sources(),out/'historical_manifest.json']}
    if seal.get('status')!='frozen_before_new_corruption_scores' or set(seal.get('bound_sha256',{}))!=expected:
        raise ValueError('Complete pre-score rule/source freeze required')
    for name,expected_sha in seal['bound_sha256'].items():
        if sha(name)!=expected_sha:raise ValueError('Frozen analysis source or historical grid changed: '+name)


def analyze(manifest_path):
    v=load_config();out=ROOT/v['output_dir'];manifest_path=Path(manifest_path).resolve()
    verify_rules(out)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['config_sha256']!=sha(CONFIG):raise ValueError('Analysis rules changed after sealing')
    historical_path=out/'historical_manifest.json'
    if manifest.get('historical_manifest_sha256')!=sha(historical_path):
        raise ValueError('Complete analysis must preserve the original seven-result snapshot')
    historical=json.loads(historical_path.read_text(encoding='utf-8'))
    for seed,runs in historical['runs'].items():
        for mode,item in runs.items():
            if manifest['runs'].get(seed,{}).get(mode)!=item:
                raise ValueError('An already observed historical baseline was substituted')
    if set(manifest.get('new_completion_proofs',{}))!={'43','44'}:
        raise ValueError('Both independent new evaluation completions required')
    for seed,item in manifest['new_completion_proofs'].items():
        check_new_completion(item['path'],item['sha256'],int(seed))
    records,rows,sources,conditions=load_records(manifest,complete=True)
    # The original reviewed sampler uses one common source draw for every seed,
    # both methods and every condition. Methods are [B0-E,B1], giving B1-B0-E.
    counts=np.stack([np.stack([rows[(s,m)]['by_source'] for m in ('b0e','b1')],axis=1)
                     for s in SEEDS],axis=1)
    if counts.shape!=(64,3,2,45,4):raise ValueError('Unexpected paired bootstrap axes')
    interval=original.paired_bootstrap(counts,replicates=v['bootstrap_replicates'],seed=v['bootstrap_seed'])
    gate=primary_gate(records,interval,v['rules'])
    summary={m:{k:dict(mean=mean(records[s][m][k] for s in SEEDS),
                      sample_sd=stdev(records[s][m][k] for s in SEEDS))
                for k in original.METRICS} for m in MODES}
    result=dict(status='completed_saved_development_analysis',by_seed=records,summary=summary,primary=gate,
        source_count=64,condition_count=45,seeds=list(SEEDS),bootstrap_replicates=20000,
        bootstrap_seed=20261026,interval_scope='Conditional on these three trained seeds and the measured source pairs; not a geographic or training-seed population interval.',
        new_inference=False,test_accessed=False,publication_ready=False,
        manifest_sha256=sha(manifest_path),analysis_config_sha256=sha(CONFIG),
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(original.__file__).resolve()):sha(original.__file__)})
    write_new(out/'analysis.json',result)
    with (out/'all_conditions.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','condition','b0e_f1','b1_f1','b2_f1','b1_minus_b0e'])
        for seed in SEEDS:
            for i,cid in enumerate(conditions):
                values=[rows[(seed,m)]['condition_f1'][i] for m in MODES]
                writer.writerow([seed,cid,*values,values[1]-values[0]])
    return dict(status=result['status'],decision=gate['decision'],mean_deltas=gate['mean_deltas'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage-historical','freeze-rules','analyze'])
    p.add_argument('--manifest',type=Path);a=p.parse_args()
    if a.action=='stage-historical':result=historical_stage()
    elif a.action=='freeze-rules':result=freeze_rules()
    else:result=analyze(a.manifest)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
