"""Explicit fixed-val adapter for the new B0-E runs; archived evaluators unchanged."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import yaml

from pcdc.data import file_sha256 as _file_sha256,make_loader
from pcdc.engine import move_model,read_checkpoint
from pcdc.evaluate_candidate import (CandidateMetadata,_load_metadata,_validate_checkpoints,
    _condition_diagnostics,load_candidate_evaluation_config)
from pcdc.evaluate_corruptions import (COUNT_KEYS,METRIC_KEYS,_bind_validation_files,
    _verify_input_hashes,counts_report,run_condition,summarize_conditions)
from pcdc.frozen_validation import (FrozenInstanceDataset,build_validation_dataset,
    load_bundle,load_evaluation_config)
from pcdc.model import build_model
from pcdc.robust_baselines import robust_source_hashes,tensor_state_sha256
from pcdc.runtime import environment_record,setup_runtime
from scripts import complete_b0e_replication as training
from scripts import levir_tail_risk_supervision as common
from scripts.windows_shared_json import atomic_json as atomic,read_json as read

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/b0e_replication_evaluation_20261026.yaml'


def file_sha256(path):
    return _file_sha256(Path(path))


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=dict(schema_version=1,kind='completed_b0e_replication_fixed_validation',seeds=[43,44],
        whole_evaluation_limit_seconds=3600,threshold=.5,test_or_martigny_access=False,automatic_retry=False)
    if any(v.get(k)!=x for k,x in expected.items()):raise ValueError('Frozen fixed-validation scope changed')
    return v,ROOT/v['output_dir']


def verify_training_session(session,*,phase='training',require_inactive=True):
    """A status title or lock never substitutes for actual protected completion."""
    session=Path(session)
    states={r:read(session/(r+'_state.json' if r!='worker' else 'worker_result.json'))
            for r in ('owner','worker','guard','watchdog')}
    if (states['owner'].get('status')!='completed' or states['owner'].get('worker_exit_code')!=0 or
        states['worker'].get('status')!='completed' or states['worker'].get('test_accessed') is not False or
        any(states[r].get('status')!='worker_exited' or 'error' in states[r] for r in ('guard','watchdog'))):
        raise ValueError('Actual bound execution and external supervision must finish before certification')
    maximum=states['guard'].get('maximum_observed_temperature_c')
    if type(maximum) is not int or not 0<=maximum<86:raise ValueError('Actual completed thermal evidence required')
    proof=read(session/'attachment.json')
    if phase not in ('training','evaluation'):raise ValueError('Unknown protected execution phase')
    module='scripts.b0e_replication_supervision' if phase=='training' else 'scripts.fixed_development_evaluation_supervision'
    runtime=ROOT/('configs/b0e_replication_runtime_20261026.yaml' if phase=='training' else
                  'configs/b0e_replication_evaluation_runtime_20261026.yaml')
    adapter=ROOT/('scripts/b0e_replication_supervision.py' if phase=='training' else
                  'scripts/fixed_development_evaluation_supervision.py')
    mode='train' if phase=='training' else 'evaluate'
    if (proof.get('mode')!=mode or proof.get('runtime_sha256')!=file_sha256(runtime) or
        proof.get('adapter_sha256')!=file_sha256(adapter) or proof.get('source_sha256')!=file_sha256(common.__file__)):
        raise ValueError('Actual supervision implementation/runtime binding differs')
    pids=[]
    for role in states:
        identity=proof.get(role,{})
        if (type(identity.get('process_id')) is not int or identity['process_id']<=0 or
            type(identity.get('creation_time_ticks')) is not int or identity['creation_time_ticks']<=0 or
            Path(identity.get('executable','')).resolve()!=Path('E:/Download/anaconda3/envs/geosa-basa/python.exe').resolve() or
            not re.search(r'\s-m '+re.escape(module)+r'\s+'+role+r'\s+--mode '+mode+r'\b',identity.get('command_line',''))):
            raise ValueError('Actual process creation/command identity is incomplete')
        pids.append(identity['process_id'])
    if len(set(pids))!=4:raise ValueError('Independent protected roles must have distinct process identities')
    if require_inactive:
        for role in states:
            if common.snapshot_identity({},proof[role])['active']:
                raise RuntimeError('A bound process is still active; wait without reloading models')
    return states,proof


def check_checkpoint_identity(cp,ref,initial,audit):
    if (cp.get('training_flavor')!='matched_robust_baseline_v1' or cp.get('robust_mode')!='b0e' or
        cp.get('config')!=ref.base.values or cp.get('robust_config')!=dict(ref.values) or
        cp.get('training_signature')!=ref.fingerprint or
        cp.get('severity_review_sha256')!=file_sha256(ref.paths['severity_review']) or
        cp.get('source_sha256')!=robust_source_hashes() or
        cp.get('initial_model_tensor_sha256')!=initial):
        raise ValueError('Checkpoint differs from the exact clean-view training identity')
    for split in ('train','val'):
        expected={k:x for k,x in audit['splits'][split].items() if k!='file_sha256'}
        if (cp.get('file_sha256',{}).get(split)!=audit['splits'][split]['file_sha256'] or
            cp.get('manifests',{}).get(split)!=expected):
            raise ValueError('Checkpoint data identity differs from the prepared original data')


def strict_context(job_path,ref,initial,audit):
    job=load_candidate_evaluation_config(job_path);metadata=_load_metadata(job,require_review=True)
    best=read_checkpoint(job.path('candidate_best_checkpoint'));last=read_checkpoint(job.path('candidate_last_checkpoint'))
    view=CandidateMetadata(SimpleNamespace(protocol=ref.protocol,manifest=metadata.bundle.manifest),
        metadata.baseline_summary,metadata.baseline_conditions,metadata.candidate_status,
        metadata.guard_completion,metadata.exposure,metadata.input_hashes)
    expected=_validate_checkpoints(job,view,best,last)
    for cp in (best,last):check_checkpoint_identity(cp,ref,initial,audit)
    return job,metadata,best,last,expected


def prepare():
    v,out=load();tv,refs,train_out=training.load_study(ROOT/v['training_study'])
    if out.exists():raise FileExistsError('Preserve existing fixed-validation preparation')
    states,attachment=verify_training_session(train_out/'train_session')
    authority=read(train_out/'training_authorization.json');training.verify(authority['bound_sha256'])
    if read(train_out/'training_completed.json').get('status')!='completed':raise ValueError('Both saved training completions required')
    plans=read(train_out/'stream_plan.json');audit=read(train_out/'data_audit.json')
    out.mkdir(parents=True);jobs={};bindings={}
    for name,expected in authority['bound_sha256'].items():bindings[name]=expected
    for name in ('training_completed.json','training_authorization.json','data_audit.json','stream_plan.json'):
        p=train_out/name;bindings[str(p)]=file_sha256(p)
    for name in ('owner_state.json','worker_result.json','guard_state.json','watchdog_state.json','attachment.json'):
        p=train_out/'train_session'/name;bindings[str(p)]=file_sha256(p)
    template=load_candidate_evaluation_config(ROOT/v['candidate_template'])
    for seed,ref in refs.items():
        raw_status=read(ref.output_dir/'status.json')
        training.check_history(read(ref.output_dir/'history.json'),plans[str(seed)],raw_status,tv['runs'][str(seed)]['initialization_sha256'])
        if read(ref.output_dir/'data_audit.json')!=audit:raise ValueError('Actual training audit changed')
        side=out/f'training_seed{seed}';side.mkdir()
        status=dict(raw_status,guarded_completion_required=True,
            temperature_guard=dict(status='completed',execution='derived_from_actual_external_cpu_guard_and_watchdog'),
            original_training_status_sha256=file_sha256(ref.output_dir/'status.json'))
        atomic(side/'guarded_status.json',status)
        atomic(side/'external_training_completion.json',dict(status='completed',
            execution='explicit_external_supervision_certificate_not_original_in_process_guard',
            training_status_sha256=file_sha256(side/'guarded_status.json'),
            original_training_status_sha256=file_sha256(ref.output_dir/'status.json'),
            original_process_record_sha256={name:file_sha256(train_out/'train_session'/name)
                for name in ('owner_state.json','worker_result.json','guard_state.json','watchdog_state.json','attachment.json')}))
        value=copy.deepcopy(dict(template.values))
        # Resolve every inherited path before moving the YAML into a new folder.
        for key in ('benchmark_config','baseline_summary','baseline_conditions'):
            value[key]=str(template.path(key))
        value.update(name=f'b0e_seed{seed}_fixed_val_20261026',output_dir=str(out/f'seed{seed}'))
        targets=dict(candidate_best_checkpoint=ref.output_dir/'best.pt',candidate_last_checkpoint=ref.output_dir/'last.pt',
            candidate_status=side/'guarded_status.json',candidate_guard_completion=side/'external_training_completion.json',
            candidate_exposure=ref.output_dir/'training_exposure_summary.json')
        for key,p in targets.items():value[key]=str(p);value[key+'_sha256']=file_sha256(p)
        job_path=out/f'seed{seed}_evaluation.yaml'
        with job_path.open('x',encoding='utf-8') as f:yaml.safe_dump(value,f,sort_keys=False)
        job,metadata,best,last,expected=strict_context(job_path,ref,tv['runs'][str(seed)]['initialization_sha256'],audit)
        jobs[str(seed)]=dict(config=str(job_path),expected_clean=expected,checkpoint_sha256=value['candidate_best_checkpoint_sha256'])
        bindings.update(metadata.input_hashes)
        for p in (job_path,ref.output_dir/'execution_completion.json',ref.output_dir/'history.json',ref.output_dir/'data_audit.json',ref.source):
            bindings[str(p)]=file_sha256(p)
        del best,last
    for p in (CONFIG,ROOT/v['protocol'],ROOT/v['analysis_config'],Path(__file__).resolve(),
              ROOT/'scripts/fixed_development_evaluation_supervision.py',
              ROOT/'pcdc/evaluate_candidate.py',ROOT/'pcdc/evaluate_corruptions.py',ROOT/'pcdc/frozen_validation.py',
              ROOT/'scripts/analyze_b0e_replication.py',ROOT/'scripts/analyze_r1.py',
              ROOT/'tests/test_b0e_fixed_evaluation.py',
              ROOT/'configs/b0e_replication_evaluation_runtime_20261026.yaml',
              ROOT/'runs/b0e_replication_analysis_20261026/analysis_rules_seal.json'):
        bindings[str(p)]=file_sha256(p)
    result=dict(status='prepared_two_completed_replications_fixed_validation',jobs=jobs,
        bound_sha256=bindings,created_utc=common.now(),new_validation_inference=False,test_accessed=False)
    atomic(out/'prepared.json',result)
    return dict(status=result['status'],jobs=list(jobs),bound_inputs=len(bindings))


def check_replay(actual,expected):
    if (actual['samples']!=expected['samples'] or any(actual['counts'][k]!=expected['metrics'][k] for k in COUNT_KEYS) or
        any(actual['metrics'][k]!=expected['metrics'][k] for k in METRIC_KEYS)):
        raise ValueError('Exact selected clean checkpoint replay failed')


def check_condition(row,cid,clean,manifest):
    if (row.get('condition_id')!=cid or row.get('samples')!=1024 or row.get('pixels')!=1024*256**2 or
        len(row.get('by_source',{}))!=64 or set(row['by_source'])!=set(clean['by_source'])):
        raise ValueError('Complete condition/source support required')
    pooled={k:0 for k in COUNT_KEYS}
    for name,source in row['by_source'].items():
        counts=source['counts'];reference=clean['by_source'][name]['counts']
        if (source.get('samples')!=16 or any(type(counts[k]) is not int or counts[k]<0 for k in COUNT_KEYS) or
            sum(counts.values())!=1024**2 or counts['tp']+counts['fn']!=reference['tp']+reference['fn']):
            raise ValueError('Source pixel/label support changed')
        for k in COUNT_KEYS:pooled[k]+=counts[k]
    if pooled!=row['counts']:raise ValueError('Source counts do not pool to condition')
    expected={key[len(cid)+1:]:x for key,x in manifest['stratum_counts'].items() if key.startswith(cid+'|')}
    if {k:x['samples'] for k,x in row['joint_strata'].items()}!=expected:
        raise ValueError('Frozen joint-stratum support changed')
    recalculated=counts_report([pooled[k] for k in COUNT_KEYS])
    if any(row['metrics'][k]!=recalculated['metrics'][k] for k in METRIC_KEYS):
        raise ValueError('Condition metric arithmetic differs from exact counts')


class Progress:
    def __init__(self,loader,session,seed,condition,completed):
        self.loader,self.session,self.seed,self.condition,self.completed=loader,session,seed,condition,completed
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for step,batch in enumerate(self.loader,1):
            yield batch
            atomic(self.session/'progress.json',dict(stage='fixed_development_validation',seed=self.seed,
                condition=self.condition,completed_conditions=self.completed,step=step,updated_utc=common.now()))


def evaluate(session,guard):
    v,out=load();tv,refs,train_out=training.load_study(ROOT/v['training_study'])
    prepared=read(out/'prepared.json');_verify_input_hashes(read(out/'authorization.json')['bound_sha256'])
    audit=read(train_out/'data_audit.json');results={}
    for seed,ref in refs.items():
        job,metadata,best,last,expected=strict_context(prepared['jobs'][str(seed)]['config'],ref,
            tv['runs'][str(seed)]['initialization_sha256'],audit)
        dataset,_=build_validation_dataset(metadata.bundle)
        if dataset.split!='val' or ref.base.run_test_after_training:raise ValueError('Development validation only')
        _bind_validation_files(metadata.input_hashes,metadata.bundle,dataset,best);_verify_input_hashes(metadata.input_hashes)
        guard();device=setup_runtime(ref.base);target=job.output_dir;target.mkdir(exist_ok=False)
        model=move_model(build_model(ref.base,initialize_encoder=False),device,'channels_last');model.load_state_dict(best['model'],strict=True)
        before=tensor_state_sha256(model.state_dict());del last
        atomic(target/'environment.json',environment_record(ref.base,device))
        atomic(target/'provenance.json',dict(training_seed=seed,training_signature=best['training_signature'],
            initial_model_tensor_sha256=best['initial_model_tensor_sha256'],candidate_checkpoint_sha256=prepared['jobs'][str(seed)]['checkpoint_sha256'],
            benchmark_fingerprints=metadata.bundle.fingerprints,input_file_sha256=metadata.input_hashes,
            wrapper='explicit_external_supervision_fixed_b0e_validation_20261026',test_accessed=False))
        clean=run_condition(model,Progress(make_loader(dataset,ref.base,training=False),session,seed,'clean',0),ref.base,device,guard_callback=guard)
        check_replay(clean,expected);clean['matches_candidate_checkpoint']=True;clean['checkpoint_metrics']=expected['metrics']
        atomic(target/'clean_metrics.json',clean)
        groups={cid:[] for cid in metadata.bundle.manifest['condition_counts']}
        for instance in metadata.bundle.instances:groups[instance['condition_id']].append(instance)
        if len(groups)!=45:raise ValueError('Exactly45 original conditions required')
        (target/'conditions').mkdir();conditions={}
        for cid,instances in groups.items():
            cell=FrozenInstanceDataset(dataset,instances)
            row=run_condition(model,Progress(make_loader(cell,ref.base,training=False),session,seed,cid,len(conditions)),ref.base,device,guard_callback=guard)
            check_condition(row,cid,clean,metadata.bundle.manifest)
            atomic(target/'conditions'/(cid+'.json'),row);conditions[cid]=row
            atomic(target/'execution_status.json',dict(status='evaluating_original45',seed=seed,completed_conditions=len(conditions),test_evaluated=False,updated_utc=common.now()))
            print(f'b0e_seed{seed}/{cid} complete={len(conditions)}/45',flush=True)
        if tensor_state_sha256(model.state_dict())!=before:raise ValueError('Inference changed model parameters/buffers')
        summary=summarize_conditions(conditions,list(groups))
        summary.update(candidate_mode='b0e',clean_f1=clean['metrics']['f1'],
            clean_minus_corruption_mf1=clean['metrics']['f1']-summary['corruption_mf1'],
            candidate_diagnostics=_condition_diagnostics(conditions,list(groups)),
            split='val',training_started=False,test_evaluated=False,
            execution='raw_scores_pending_external_completion_audit')
        atomic(target/'conditions.json',conditions);atomic(target/'raw_summary.json',summary)
        atomic(target/'execution_status.json',dict(status='scores_complete_pending_external_completion_audit',seed=seed,completed_conditions=45,test_evaluated=False))
        _verify_input_hashes(metadata.input_hashes);results[str(seed)]=dict(clean_f1=summary['clean_f1'],corruption_mf1=summary['corruption_mf1'])
        del model,best;torch.cuda.empty_cache();guard()
    atomic(out/'raw_evaluation_completed.json',dict(status='completed_raw_scores_pending_external_audit',runs=results,test_accessed=False))
    return dict(status='completed',optimizer_updates=0,test_accessed=False)


def cpu_smoke():
    v,out=load()
    # An explicit CPU forward/count smoke; no CUDA runtime setup or data files.
    _,refs,_=training.load_study();torch.set_num_threads(2);torch.manual_seed(20261026)
    model=build_model(refs[43].base,initialize_encoder=False).cpu().eval()
    with torch.no_grad():
        logits=model(torch.rand(2,3,32,32),torch.rand(2,3,32,32))['logits']
    if logits.shape!=(2,1,32,32) or not torch.isfinite(logits).all():raise ValueError('CPU model smoke failed')
    from pcdc.evaluate_corruptions import per_sample_confusions
    counts,_=per_sample_confusions(logits,torch.zeros_like(logits),threshold=.5)
    if int(counts.sum())!=2048:raise ValueError('CPU count conservation failed')
    result=dict(status='passed_cpu_synthetic_engineering_only',optimizer_updates=0,research_result=False,actual_evaluation_remains_original_FP16=True)
    atomic(out/'smoke.json',result);return result


def certify():
    v,out=load();states,attachment=verify_training_session(out/'evaluation_session',phase='evaluation')
    if any((out/f'seed{s}'/'summary.json').exists() for s in v['seeds']):
        raise FileExistsError('Preserve existing partial or completed certification')
    if read(out/'raw_evaluation_completed.json').get('status')!='completed_raw_scores_pending_external_audit':
        raise ValueError('Both raw scoring rows required')
    _verify_input_hashes(read(out/'authorization.json')['bound_sha256'])
    bundle=load_bundle(load_evaluation_config(ROOT/v['benchmark_config']),require_review=True)
    prepared=read(out/'prepared.json')
    role_paths={r:str(out/'evaluation_session'/(r+'_state.json' if r!='worker' else 'worker_result.json')) for r in states}
    from scripts import analyze_b0e_replication as analysis
    av=analysis.load_config();analysis_out=ROOT/av['output_dir'];analysis.verify_rules(analysis_out)
    historical_path=analysis_out/'historical_manifest.json';manifest=copy.deepcopy(read(historical_path))
    manifest.update(scope='completed_development_grid_with_two_new_b0e_controls',
        historical_manifest_sha256=file_sha256(historical_path),new_completion_proofs={})
    for seed in v['seeds']:
        target=out/f'seed{seed}';summary=read(target/'raw_summary.json')
        conditions=read(target/'conditions.json');clean=read(target/'clean_metrics.json')
        if len(conditions)!=45 or clean.get('matches_candidate_checkpoint') is not True:
            raise ValueError('Incomplete raw result coverage/replay')
        check_replay(clean,prepared['jobs'][str(seed)]['expected_clean'])
        expected_ids=list(bundle.manifest['condition_counts'])
        if set(conditions)!=set(expected_ids):raise ValueError('Saved original condition identities changed')
        for cid in expected_ids:
            check_condition(conditions[cid],cid,clean,bundle.manifest)
            if conditions[cid]!=read(target/'conditions'/(cid+'.json')):
                raise ValueError('Saved condition copies disagree')
        recalculated=summarize_conditions(conditions,expected_ids)
        if any(summary[k]!=x for k,x in recalculated.items()):
            raise ValueError('Saved aggregate differs from original condition arithmetic')
        if summary['candidate_diagnostics']!=_condition_diagnostics(conditions,expected_ids):
            raise ValueError('Saved pooled diagnostic arithmetic differs')
        summary.update(status='completed',guarded_completion_required=True,
            temperature_guard=dict(status='completed',execution='derived_external_completion_summary_raw_records_retained'))
        atomic(target/'summary.json',summary);atomic(target/'status.json',summary)
        derived=dict(status='completed',decision='ok',test_evaluated=False,
            execution='derived_external_completion_certificate_not_raw_sensor_state',
            decision_semantics='Normal completed execution without guard stop under the frozen policy',
            raw_final_decision=states['guard'].get('decision'),
            maximum_observed_temperature_c=states['guard']['maximum_observed_temperature_c'],
            original_state_sha256={r:file_sha256(p) for r,p in role_paths.items()})
        atomic(target/'temperature_guard_state.json',derived)
        atomic(target/'guard_completion.json',dict(status='completed',execution='derived_external_supervision',
            summary_sha256=file_sha256(target/'summary.json'),guard_state_sha256=file_sha256(target/'temperature_guard_state.json')))
        bound={p:file_sha256(p) for p in role_paths.values()}
        for p in (out/'evaluation_session'/'attachment.json',target/'summary.json',target/'conditions.json',
                  target/'clean_metrics.json',target/'provenance.json',out/'raw_evaluation_completed.json'):
            bound[str(p)]=file_sha256(p)
        certificate=dict(status='completed_independent_external_supervision_audit',seed=seed,test_accessed=False,
            roles=role_paths,bound_sha256=bound,owner_exit_evidence='Owner recorded normal completion and bound process is inactive; only worker OS exit0 is explicitly recorded.')
        atomic(target/'evaluation_completion.json',certificate)
        analysis.check_new_completion(target/'evaluation_completion.json',file_sha256(target/'evaluation_completion.json'),seed)
        manifest['runs'][str(seed)]['b0e']=analysis.bind_run(target)
        manifest['new_completion_proofs'][str(seed)]=dict(path=str(target/'evaluation_completion.json'),sha256=file_sha256(target/'evaluation_completion.json'))
    atomic(out/'analysis_manifest.json',manifest)
    return dict(status='two_fixed_evaluations_certified',manifest=str(out/'analysis_manifest.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','smoke','certify'])
    a=p.parse_args();started=time.monotonic();status='failed_preserve'
    try:
        result=cpu_smoke() if a.action=='smoke' else globals()[a.action]()
        status='completed'
    finally:
        _,out=load()
        if out.exists():
            from scripts.fixed_development_evaluation_supervision import record_cost
            record_cost(a.action,time.monotonic()-started+3,
                        'Measured CPU metadata/engineering wall plus3second import/startup allowance; '+status)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
