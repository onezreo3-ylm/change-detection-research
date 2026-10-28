"""Build review tables from saved evidence only; missing controls stay missing."""
from pathlib import Path
from statistics import mean,stdev
import csv
import hashlib
import json

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'tables'


def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(value):return 'pending' if value is None else f'{value:.6f}'


def table(path,columns,rows):
    with path.with_suffix('.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    lines+=['| '+' | '.join(str(x) for x in row)+' |' for row in rows]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    if OUT.exists():raise FileExistsError('Preserve existing generated tables')
    historical=read('runs/b0e_replication_analysis_20261026/historical_verification.json')
    r1=read('runs/r1e_protocol_20260926/analysis.json')
    p7=read('runs/levir_phase7_val_seed42/summary.json')
    b7=read('runs/levir_phase7_val_seed42/bootstrap.json')
    d7=read('runs/levir_phase7_val_seed42/decision.json')
    p6=read('runs/levir_phase6_action_utility_seed42/summary.json')
    g6=read('runs/levir_phase6_action_utility_seed42/gate_decision.json')
    if historical['primary_three_seed_result'] is not None or len(historical['missing'])!=2:
        raise ValueError('This working draft must retain the two missing controls')
    if p7['status']!='completed' or p7['test_evaluated'] is not False:
        raise ValueError('Expected frozen development action results')
    metrics=('clean_f1','corruption_mf1','clean_minus_corruption_mf1','pooled_recall','background_fpr')
    OUT.mkdir();rows=[];raw=[]
    for seed in (42,43,44):
        for mode in ('b0e','b1','b2'):
            values=historical['historical_metrics'][str(seed)].get(mode)
            scalars=[values[k] if values else None for k in metrics]
            raw.append(dict(seed=seed,mode=mode,**dict(zip(metrics,scalars)),status='saved_verified_result' if values else 'pending_final_training_and_fixed_validation'))
            rows.append([seed,mode,*[fmt(x) for x in scalars]])
    table(OUT/'table1_seed_grid',['seed','mode',*metrics],rows)
    aggregates={};rows=[]
    for mode in ('b0e','b1','b2'):
        available=[historical['historical_metrics'][str(s)].get(mode) for s in (42,43,44)]
        count=sum(x is not None for x in available)
        if count!=3:
            aggregates[mode]=dict(n=count,mean=None,sample_sd=None)
            rows.append([mode,f'{count}/3','pending','pending','pending','pending'])
            continue
        aggregates[mode]={k:dict(mean=mean(x[k] for x in available),sample_sd=stdev(x[k] for x in available)) for k in metrics}
        # Independently reproduce every mean/SD using a second library and the
        # already published R1 values, without a new resampling analysis.
        for key in metrics:
            a=np.asarray([x[key] for x in available])
            assert abs(float(a.mean())-aggregates[mode][key]['mean'])<1e-12
            assert abs(float(a.std(ddof=1))-aggregates[mode][key]['sample_sd'])<1e-12
            assert abs(aggregates[mode][key]['mean']-r1['method_summary'][mode][key]['mean'])<1e-12
            assert abs(aggregates[mode][key]['sample_sd']-r1['method_summary'][mode][key]['sample_std'])<1e-12
        rows.append([mode,'3/3',*[f"{aggregates[mode][key]['mean']:.6f} +/- {aggregates[mode][key]['sample_sd']:.6f}" for key in ('clean_f1','corruption_mf1','pooled_recall','background_fpr')]])
    table(OUT/'table2_seed_summary',['mode','completed_seeds','clean_F1_mean_SD','corruption_mF1_mean_SD','pooled_corruption_recall_mean_SD','background_FPR_mean_SD'],rows)
    rows=[];g0=p7['actions']['G0']['corruption_mf1'];learned=[]
    for name,action in p7['actions'].items():
        delta=action['corruption_mf1']-g0
        interval=b7['learned_mf1_minus_better_constant'].get(name)
        if interval is not None:learned.append(name);assert delta<0
        rows.append([name,fmt(action['corruption_mf1']),fmt(delta),
            f'[{interval[0]:.6f}, {interval[2]:.6f}]' if interval else 'diagnostic_or_constant',
            fmt(action['clean']['f1']),fmt(action['clean']['recall']),fmt(action['pooled']['recall'])])
    assert len(learned)==8
    table(OUT/'table3_whole_tile_policies',['policy','corruption_mF1','delta_vs_G0','nominal95_source_interval','clean_F1','clean_recall','pooled_corruption_recall'],rows)
    signs=p6['utility'];assert signs['positive']+signs['negative']+signs['zero']==64080
    for sign in ('positive','negative','zero'):
        assert sum(f[sign] for f in signs['by_family_sign'].values())==signs[sign]
    table(OUT/'table4_train_action_signs',['utility_sign','count','fraction'],
        [[sign,signs[sign],f'{signs[sign]/64080:.10f}'] for sign in ('positive','negative','zero')])
    rows=[]
    for name,row in p6['transitions_a0_to_a1'].items():
        assert abs(row['count']/row['denominator']-row['rate'])<1e-15
        rows.append([name,row['count'],row['denominator'],f"{row['rate']:.10f}"])
    table(OUT/'table5_action_transitions',['transition','count','denominator','rate'],rows)
    selected=p6['actions']['label_informed_choice']['corruption_mf1']
    constant=p6['actions']['a1_shared_stage2_adapter']['corruption_mf1']
    assert abs(selected-constant-g6['oracle_mf1_margin'])<1e-12
    results=dict(status='working_tables_missing_two_b0e_controls',rows=raw,aggregates=aggregates,
        existing_b2_minus_b1_interval=r1['paired_source_bootstrap']['b2_minus_b1_mean_mf1_interval'],
        phase6_label_informed_margin=g6['oracle_mf1_margin'],phase6_legacy_field='oracle_mf1_margin; diagnostic, not an upper bound',
        phase7_decision=d7,source_unit='original image pair; geographic independence unproven',
        new_inference=False,new_bootstrap=False,publication_ready=False)
    (OUT/'table_data.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    inputs=['runs/b0e_replication_analysis_20261026/historical_verification.json','runs/r1e_protocol_20260926/analysis.json',
        'runs/levir_phase7_val_seed42/summary.json','runs/levir_phase7_val_seed42/bootstrap.json','runs/levir_phase7_val_seed42/decision.json',
        'runs/levir_phase6_action_utility_seed42/summary.json','runs/levir_phase6_action_utility_seed42/gate_decision.json']
    evidence=dict(status='generated_and_independently_checked_table_arithmetic',input_sha256={p:sha(ROOT/p) for p in inputs},
        builder_sha256=sha(__file__),output_sha256={p.name:sha(p) for p in OUT.iterdir()},
        missing_controls=['b0e43','b0e44'],missing_means_are_null=True,new_inference=False,new_resampling=False)
    (OUT/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=evidence['status'],tables=5,missing_controls=evidence['missing_controls'])))


if __name__=='__main__':main()
