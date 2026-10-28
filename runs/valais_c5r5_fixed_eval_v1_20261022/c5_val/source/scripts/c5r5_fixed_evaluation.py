"""Explicit 40ep BF16-trained/FP32-evaluated checkpoint adapter; old checks stay fixed."""
from __future__ import annotations
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

from PIL import features
import torch
import yaml

from pcdc.config import Config, UniqueKeyLoader, load_config, protocol_signature
from pcdc.data import file_sha256, make_loader
from pcdc.engine import check_checkpoint, move_model, read_checkpoint
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json
from pcdc.robust_baselines import tensor_state_sha256
from pcdc import evaluate_candidate as candidate
from pcdc.evaluate_corruptions import (COUNT_KEYS, METRIC_KEYS, _bound_input_hashes,
    _bind_validation_files, _expected_clean, _verify_input_hashes, run_condition, summarize_conditions)
from pcdc.frozen_validation import (load_bundle, load_evaluation_config, build_validation_dataset, FrozenInstanceDataset)
from scripts import five_view_baselines as f
from scripts import five_view_40ep_runtime as trainer
from scripts import valais_c5r5_runtime as training
from scripts import c5r5_runtime_identity as identity
from scripts import evaluate_coupling_p0 as p0
from scripts import valais_p1_posttraining as saved
from scripts.evaluate_guarded import EvaluationTemperatureGuard, load_guard_config, _atomic_json
from scripts.p1_process_state import process_snapshot
from scripts.run_valais_clean_guarded import creation_time_ticks

ROOT=f.ROOT
ROLES=('c5','r5')
FLAVOR=identity.FLAVOR


def read(path):return f.read(path)
def now():return f.now()


def load(path):
    source=Path(path).resolve();value=yaml.load(source.read_bytes(),Loader=UniqueKeyLoader)
    expected={'schema_version':1,'parent_study':'valais_c5r5_40ep_runtime_v1_20261020.yaml',
        'training_supervision':'../runs/valais_c5r5_40ep_runtime_v1_20261020/supervision_attach_20261021',
        'output_dir':'../runs/valais_c5r5_fixed_eval_v1_20261022',
        'benchmark_config':'evaluate_valais_baseline_corruption_v1_20261002.yaml',
        'baseline_evaluation':'../runs/valais_baseline_corruption_val_v1_seed42_20261002',
        'b0e_evaluation':'../runs/valais_b0e_fp32_seed42_v1_20261002_val',
        'p0_manifest':'../runs/valais_coupling_p0_train_qa_v1r1_20261007/manifest.json',
        'p0_review':'../runs/valais_coupling_p0_train_qa_v1r1_20261007/visual_review.json',
        'protocol_document':'../docs/C5R5_FIXED_EVALUATION_PROTOCOL_2026-10-22.md',
        'minimum_unittests':421,'validation_max_seconds_per_model':21600,'residual_max_seconds':21600,
        'smoke_max_seconds':600}
    if value!=expected:raise ValueError('Unreviewed fixed evaluation scope, gates, budget or inputs')
    paths={k:(source.parent/value[k]).resolve() for k in ('parent_study','training_supervision','output_dir',
        'benchmark_config','baseline_evaluation','b0e_evaluation','p0_manifest','p0_review','protocol_document')}
    _,settings,study_paths,_=identity.load_study(paths['parent_study'])
    if settings['optimization_mode']!='bf16_training_fp32_validation':raise ValueError('Wrong numerical training protocol')
    study_paths=dict(study_paths,benchmark_config=paths['benchmark_config'],baseline_evaluation=paths['baseline_evaluation'])
    runs={role:SimpleNamespace(base=load_config(study_paths['output_dir']/role/'base_config.yaml'),
        output_dir=study_paths['output_dir']/role,mode=role) for role in ROLES}
    paths['output_dir'].relative_to(ROOT/'runs')
    return source,value,paths,settings,study_paths,runs


def adapter_sources():
    files=identity.implementation_files()
    for name in ('scripts/c5r5_fixed_evaluation.py','scripts/c5r5_fixed_residual.py',
        'scripts/valais_p1_posttraining.py','scripts/valais_p1_residual.py','scripts/audit_coupling_p0.py',
        'scripts/c5r5_supervision_attach.py'):
        files[name]=ROOT/name
    return files


def inference_mapping(training_config,benchmark_config):
    """Validate the explicit precision/horizon differences without editing a checkpoint."""
    a,b=training_config,benchmark_config
    if (a.values['train']['epochs']!=40 or a.values['train']['amp'] is not True
            or b.values['train']['amp'] is not False or a.run_test_after_training or b.run_test_after_training
            or a.data_root!=b.data_root
            or a.path(a.values['model']['weights_cache'])!=b.path(b.values['model']['weights_cache'])):
        raise ValueError('Unexpected train/eval numerical or data identity')
    projected=copy.deepcopy(a.values)
    for key in ('name','output_dir'):projected['experiment'][key]=b.values['experiment'][key]
    projected['data']['root']=b.values['data']['root']
    projected['model']['weights_cache']=b.values['model']['weights_cache']
    projected['train']['epochs']=b.values['train']['epochs']
    projected['train']['amp']=False
    if projected!=b.values:raise ValueError('An undeclared prediction, model, loss or training setting differs')
    if protocol_signature(projected)!=protocol_signature(b.values):raise ValueError('FP32 inference signature differs')
    return {'training_protocol_signature':protocol_signature(a.values),
        'fp32_evaluation_protocol_signature':protocol_signature(projected),
        'explicit_differences':['training_autocast_bfloat16_vs_evaluation_float32','40ep_vs_historical100ep',
            'experiment_name_and_output_paths'],'checkpoint_fields_modified':False}


def validate_selection(last,best,actual,plan):
    if last['epoch']!=40 or len(last['history'])!=40 or len(plan)!=40 or actual!=plan:
        raise ValueError('Incomplete committed 40ep history or actual sampling stream')
    selected=max(last['history'],key=lambda r:r['val']['metrics']['f1'])
    if (best['epoch']!=selected['epoch'] or best['best_epoch']!=selected['epoch']
            or last['best_epoch']!=selected['epoch'] or best['best_val_f1']!=selected['val']['metrics']['f1']
            or last['best_val_f1']!=best['best_val_f1'] or best['history']!=last['history'][:best['epoch']]):
        raise ValueError('Checkpoint is not first highest clean-val F1')
    for cp in (last,best):
        if cp['training_flavor']!=FLAVOR or set(cp['file_sha256'])!={'train','val'} or any('test' in r for r in cp['history']):
            raise ValueError('Checkpoint flavor or development-only scope changed')


def validate_row(role,cfg,settings,study_paths,*,bundle=None,planned=None):
    status=training.check_completion(study_paths,role)
    plan=planned if planned is not None else read(study_paths['output_dir']/'sampling_plan.json')
    actual=read(cfg.output_dir/'sampling_actual.json')['rows']
    ident=read(cfg.output_dir/'identity.json')
    best,last=(read_checkpoint(cfg.output_dir/name) for name in ('best.pt','last.pt'))
    validate_selection(last,best,actual,plan)
    torch.set_num_threads(4)
    expected_sources={k:file_sha256(p) for k,p in identity.implementation_files().items()}
    metadata=read(study_paths['output_dir']/'metadata.json')
    for cp in (best,last):
        trainer.validate_checkpoint(cp,cfg.base,ident,plan)
        check_checkpoint(cfg.base,cp)
        if (cp['source_sha256']!=expected_sources or cp['role']!=role or cp['kind']!='research'
                or cp['identity']['optimization_mode']!='bf16_training_fp32_validation'
                or cp['identity']['training_autocast_dtype']!='bfloat16'
                or cp['identity']['parameter_optimizer_loss_dtype']!='float32'
                or cp['identity']['validation_dtype']!='float32' or cp['identity']['grad_scaler'] is not None
                or cp['file_sha256']!={k:x['file_sha256'] for k,x in metadata['audit']['splits'].items()}
                or cp['manifests']!={k:{kk:xx for kk,xx in x.items() if kk!='file_sha256'} for k,x in metadata['audit']['splits'].items()}):
            raise ValueError('40ep BF16 identity, source or complete data audit changed')
    if last['history']!=read(cfg.output_dir/'history.json')['rows'] or best['epoch']!=status['best_epoch']:
        raise ValueError('Saved status/history does not commit these checkpoints')
    mapping=None
    if bundle is not None:
        mapping=inference_mapping(cfg.base,bundle.protocol.base)
        for cp in (best,last):
            if cp['file_sha256']['val']!=bundle.manifest['file_sha256'] or cp['manifests']['val']!=bundle.manifest['inventory']:
                raise ValueError('Checkpoint val differs from the original frozen benchmark')
    files=[cfg.output_dir/n for n in ('best.pt','last.pt','status.json','guard_completion.json',
        'temperature_guard_state.json','history.json','sampling_actual.json','identity.json','base_config.yaml')]
    metrics=_expected_clean(best)['metrics']
    del last
    return best,{'role':role,'epochs':40,'optimizer_steps':110280,'pair_views':2205000,'source_draws':441000,
        'best_epoch':best['epoch'],'clean_metrics':metrics,'training_flavor':FLAVOR,
        'numerical_mapping':mapping,'bound_sha256':{str(p.resolve()):file_sha256(p) for p in files}}


def prepare(path):
    source,value,paths,settings,study_paths,runs=load(path)
    f.verify_prepared(study_paths,sealed=True)
    supervision=paths['training_supervision'];queue=read(supervision/'status.json');owner=read(supervision/'owner.json')
    if queue['status']!='completed_training_ready_for_explicit_evaluation' or queue['completed_roles']!=list(ROLES):
        raise ValueError('Training supervision has not completed both rows')
    if process_snapshot(owner['pid'],owner['creation_ticks'])['owned_active']:raise RuntimeError('Training supervisor is active')
    metadata=read(study_paths['output_dir']/'metadata.json')
    sampler=f.UniformEpochSampler(sorted(metadata['audit']['splits']['train']['file_sha256']),settings['source_seed_template'])
    planned=[]
    for epoch in range(1,41):
        sampler.set_epoch(epoch);planned.append(sampler.summary(settings,metadata['corruptions']))
    if planned!=read(study_paths['output_dir']/'sampling_plan.json'):raise ValueError('Regenerated first40 plans differ')
    bundle=load_bundle(load_evaluation_config(paths['benchmark_config']),require_review=True)
    manifest=read(paths['p0_manifest']);p0.validate_manifest(manifest);p0.validate_review(manifest,paths['p0_manifest'],paths['p0_review'])
    candidate._validate_baseline(bundle,read(paths['baseline_evaluation']/'summary.json'),read(paths['baseline_evaluation']/'conditions.json'))
    rows={};bindings=_bound_input_hashes(bundle)
    for role in ROLES:
        best,rows[role]=validate_row(role,runs[role],settings,study_paths,bundle=bundle,planned=planned)
        record=read(supervision/f'{role}.exit.json')
        if record['exit_code']!=0 or record['status_sha256']!=file_sha256(runs[role].output_dir/'status.json'):
            raise ValueError('Observed worker exit does not commit the current status')
        bindings.update(rows[role]['bound_sha256']);del best
    a,b=(read(runs[r].output_dir/'history.json')['rows'] for r in ROLES)
    if any(x['train']['candidate_pixel_stream_sha256']!=y['train']['candidate_pixel_stream_sha256'] for x,y in zip(a,b)):
        raise ValueError('Actual full candidate pixel streams differ')
    dataset,reference=build_validation_dataset(bundle)
    _bind_validation_files(bindings,bundle,dataset,reference);_verify_input_hashes(bindings)
    output=paths['output_dir'];output.mkdir(exist_ok=False)
    write_json(output/'regenerated_sampling_plan.json',planned)
    write_json(output/'training_completion_audit.json',{'status':'passed_completed_40ep_bf16_training_audit','rows':rows,
        'matched_full_actual_source_instance_candidate_streams':True,'test_accessed':False,
        'checkpoint_fields_modified':False,'extra_training_physical_steps':210,'extra_training_pair_forward_views':4158})
    saved.audit_validation(paths['b0e_evaluation'],audit_destination=output/'b0e_saved_result_audit.json')
    if read(paths['b0e_evaluation']/'provenance.json')['benchmark_fingerprints']!=bundle.fingerprints:
        raise ValueError('Historical B0-E evaluated another benchmark')
    files=[source,paths['protocol_document'],paths['parent_study'],study_paths['output_dir']/'authorization.json',
        study_paths['output_dir']/'human_precision_approval.json',study_paths['output_dir']/'metadata.json',
        study_paths['output_dir']/'sampling_plan.json',output/'regenerated_sampling_plan.json',output/'training_completion_audit.json',
        paths['p0_manifest'],paths['p0_review'],study_paths['guard_config'],ROOT/'tests/test_c5r5_fixed_evaluation.py',
        *adapter_sources().values(),*[p for p in supervision.glob('*.json')],
        *[paths['b0e_evaluation']/n for n in ('summary.json','conditions.json','clean_metrics.json','guard_completion.json','temperature_guard_state.json')],
        *[paths['baseline_evaluation']/n for n in ('summary.json','conditions.json')]]
    for sample in manifest['samples']:
        for entry in sample['files'].values():
            p0.checked_hash(entry['path'],entry['sha256']);bindings[entry['path']]=entry['sha256']
    bindings.update({str(p.resolve()):file_sha256(p) for p in files})
    write_json(output/'prepared_identity.json',{'status':'prepared_fixed_evaluation_no_inference',
        'bound_sha256':bindings,'test_accessed':False,'training_authorized':False})
    return {'status':'prepared_fixed_40ep_evaluation','files':len(bindings),'roles':rows}


def verify_seal(paths):
    proof=read(paths['output_dir']/'authorization.json');_verify_input_hashes(proof['bound_sha256']);return proof


@torch.inference_mode()
def evaluate_row(path,role):
    started=time.monotonic();source,value,paths,settings,study_paths,runs=load(path);verify_seal(paths)
    bundle=load_bundle(load_evaluation_config(paths['benchmark_config']),require_review=True)
    best,_=validate_row(role,runs[role],settings,study_paths,bundle=bundle)
    expected_clean=_expected_clean(best);dataset,_=build_validation_dataset(bundle);config=bundle.protocol.base
    bindings=_bound_input_hashes(bundle);_bind_validation_files(bindings,bundle,dataset,best);_verify_input_hashes(bindings)
    output=paths['output_dir']/(role+'_val');output.mkdir(exist_ok=False)
    files=adapter_sources();hashes={k:file_sha256(p) for k,p in files.items()}
    shutil.copyfile(source,output/'evaluation_source.yaml')
    for name,p in files.items():
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write_json(output/'provenance.json',{'training_flavor':FLAVOR,'candidate_role':role,
        'training_signature':best['training_signature'],'training_identity':best['identity'],
        'benchmark_fingerprints':bundle.fingerprints,'input_file_sha256':bindings,'adapter_source_sha256':hashes,
        'checkpoint_fields_modified':False,'training_autocast_dtype':'bfloat16','evaluation_dtype':'float32','test_evaluated':False})
    guard=EvaluationTemperatureGuard(load_guard_config(study_paths['guard_config']),output)
    def guarded():
        if time.monotonic()-started>=21600:raise RuntimeError('Fixed val stage budget exhausted; no continuation')
        guard()
    try:
        guard(force=True);device=setup_runtime(config)
        environment=environment_record(config,device);environment.update(jpeg_codec=features.version_codec('jpg'),libjpeg_turbo=features.version_feature('libjpeg_turbo'))
        write_json(output/'environment.json',environment)
        model=move_model(build_model(config,initialize_encoder=False),device,'channels_last').float()
        model.load_state_dict(best['model'],strict=True);model.requires_grad_(False)
        before=tensor_state_sha256(model.state_dict())
        _atomic_json(output/'status.json',{'status':'evaluating_clean','role':role,'test_evaluated':False})
        clean=run_condition(model,make_loader(dataset,config,training=False),config,device,guard_callback=guarded)
        clean['matches_candidate_checkpoint']=(clean['samples']==expected_clean['samples'] and
            all(clean['counts'][k]==expected_clean['metrics'][k] for k in COUNT_KEYS) and
            all(clean['metrics'][k]==expected_clean['metrics'][k] for k in METRIC_KEYS))
        write_json(output/'clean_metrics.json',clean)
        if not clean['matches_candidate_checkpoint']:raise ValueError('FP32 clean counts/metrics do not exactly reproduce best')
        grouped={}
        for instance in bundle.instances:grouped.setdefault(instance['condition_id'],[]).append(instance)
        ids=list(bundle.manifest['condition_counts'])
        if len(ids)!=45 or set(grouped)!=set(ids):raise ValueError('Original45 condition coverage changed')
        results={};(output/'conditions').mkdir()
        for condition_id in ids:
            cell=FrozenInstanceDataset(dataset,grouped[condition_id])
            row=run_condition(model,make_loader(cell,config,training=False),config,device,guard_callback=guarded)
            strata={k[len(condition_id)+1:]:n for k,n in bundle.manifest['stratum_counts'].items() if k.startswith(condition_id+'|')}
            if (row['condition_id']!=condition_id or row['samples']!=bundle.manifest['condition_counts'][condition_id]
                    or row['pixels']!=row['samples']*dataset.tile_size**2 or {k:v['samples'] for k,v in row['joint_strata'].items()}!=strata):
                raise ValueError('Incomplete condition/source/stratum coverage')
            row.update(clean_reference_f1=clean['metrics']['f1'],clean_minus_f1=clean['metrics']['f1']-row['metrics']['f1'])
            write_json(output/'conditions'/(condition_id+'.json'),row);results[condition_id]=row
            _atomic_json(output/'status.json',{'status':'evaluating_corruptions','role':role,'completed_conditions':len(results),
                'condition_count':45,'updated_utc':now(),'test_evaluated':False})
            print(f'{role}/{condition_id}: f1={row["metrics"]["f1"]:.6f} ({len(results)}/45)',flush=True)
        write_json(output/'conditions.json',results)
        by_condition,paired=candidate.compare_results(results,read(paths['baseline_evaluation']/'conditions.json'),ids)
        write_json(output/'comparison_by_condition.json',by_condition);write_json(output/'paired_by_source.json',paired)
        if before!=tensor_state_sha256(model.state_dict()):raise ValueError('Evaluation changed model parameters/buffers')
        _verify_input_hashes(bindings);verify_seal(paths);guarded()
        if hashes!={k:file_sha256(p) for k,p in adapter_sources().items()}:raise ValueError('Adapter source changed during scoring')
        summary={'status':'completed','kind':'research','mode':'explicit_40ep_bf16_fp32_validation_v1',
            'candidate_mode':role,'candidate_role':role,'training_flavor':FLAVOR,'split':'val',
            'candidate_checkpoint_epoch':best['epoch'],'selection':'clean_validation_f1_only','clean_f1':clean['metrics']['f1'],
            **summarize_conditions(results,ids),'candidate_diagnostics':candidate._condition_diagnostics(results,ids),
            'test_evaluated':False,'training_started':False,'guarded_completion_required':True,'evaluation_amp':False,
            'model_state_unchanged':True,'elapsed_seconds':time.monotonic()-started}
        summary['temperature_guard']=guard.finish(summary)
        if time.monotonic()-started>=21600:raise RuntimeError('Fixed val stage budget exhausted during completion')
        _atomic_json(output/'summary.json',summary);_atomic_json(output/'status.json',summary)
        _atomic_json(output/'guard_completion.json',{'status':'completed','summary_sha256':file_sha256(output/'summary.json'),
            'guard_state_sha256':file_sha256(output/'temperature_guard_state.json'),'adapter_source_sha256':hashes})
        saved.audit_validation(output);return summary
    except BaseException as error:
        if guard.attached:guard.abort(error)
        _atomic_json(output/'status.json',{'status':'failed_preserve','error':f'{type(error).__name__}: {error}',
            'updated_utc':now(),'automatic_retry':False,'test_evaluated':False});raise


def gate_comparison(scores,gates):
    keys={'clean_f1','clean_recall','mf1','corruption_recall','clean_empty_fpr','corruption_empty_fpr'}
    if (set(scores)!={'c5','r5','b0e'} or any(set(row)!=keys for row in scores.values())
            or any(not math.isfinite(x) or not 0<=x<=1 for row in scores.values() for x in row.values())):
        raise ValueError('Complete finite six-metric scores required')
    deltas={name:{k:scores[a][k]-scores[b][k] for k in keys}
        for name,a,b in (('c5_vs_b0e','c5','b0e'),('r5_vs_c5','r5','c5'))}
    checks={}
    for name,d in deltas.items():
        g=gates[name]
        checks[name]={
            'clean_f1':d['clean_f1']>=-g['maximum_clean_f1_drop']-1e-12,
            'clean_recall':d['clean_recall']>=-g['maximum_clean_recall_drop']-1e-12,
            'mf1':d['mf1']>=g.get('minimum_corruption_mf1_gain',-g.get('maximum_corruption_mf1_drop',0))-1e-12,
            'corruption_recall':d['corruption_recall']>=-g['maximum_corruption_recall_drop']-1e-12,
            'clean_empty_fpr':d['clean_empty_fpr']<=g['maximum_clean_empty_fpr_increase']+1e-12,
            'corruption_empty_fpr':d['corruption_empty_fpr']<=g['maximum_corruption_empty_fpr_increase']+1e-12}
    return {'deltas':deltas,'checks':checks,'passed':{k:all(v.values()) for k,v in checks.items()},
        'historical_b0e_is_unmatched_100ep_fp32_background':True,'single_seed_development_only':True,
        'p2_training_authorized':False}


def validate_smoke(report,observations):
    if (report['status']!='passed_train_only_c5r5_cuda_smoke' or report['smoke_only'] is not True
            or report['research_result'] is not False or report['predictions']!=102 or report['main_predictions']!=98
            or report['records']!=0 or report['training'] is not False or report['gradients_enabled'] is not False
            or report['amp'] is not False or report['test_accessed'] is not False or report['val_accessed'] is not False
            or not 0<=report['elapsed_seconds']<=600 or not observations
            or any(type(t) is not int or not 0<=t<85 for t in observations)
            or max(observations)!=report['maximum_observed_temperature_c']
            or set(report['models'])!=set(ROLES)
            or any(not x['model_state_unchanged'] for x in report['models'].values())
            or len(report['self_checks'])!=2 or {c['model'] for c in report['self_checks']}!=set(ROLES)
            or any(c['repeat_bitwise_equal'] is not True or c['temporal_swap_allclose'] is not True
                or not math.isfinite(c['max_swap_abs_error']) or c['max_swap_abs_error']<0 for c in report['self_checks'])):
        raise ValueError('Incomplete real FP32 CUDA smoke or temperature/numerical/scope failure')


def seal(path,unittest_log):
    source,value,paths,_,_,_=load(path);output=paths['output_dir']
    if (output/'authorization.json').exists():raise FileExistsError('Fixed evaluation already sealed')
    proof=read(output/'prepared_identity.json');_verify_input_hashes(proof['bound_sha256'])
    count=training.passing_test_count(unittest_log,value['minimum_unittests'])
    smoke=read(output/'cuda_smoke/report.json')
    validate_smoke(smoke,read(output/'cuda_smoke/guard_observations.json')['temperatures_c'])
    bound=dict(proof['bound_sha256'])
    for p in (Path(unittest_log).resolve(),output/'prepared_identity.json',output/'cuda_smoke/report.json',
            output/'cuda_smoke/guard_observations.json'):
        bound[str(p.resolve())]=file_sha256(p)
    write_json(output/'authorization.json',{'status':'authorized_fixed_c5r5_evaluation',
        'authorized_utc':now(),'human_instruction':'40轮独立C5/R5及其原45条件val与固定64 train残余；已选择BF16训练/FP32评价',
        'bound_sha256':bound,'tests':count,'checkpoint_fields_modified':False,'training_authorized':False,
        'test_accessed':False,'queue':['c5','r5','residual'],'seconds_per_stage':21600,'automatic_retry':False,
        'additional_smoke_pair_predictions':102})
    return {'status':'sealed_fixed_evaluation','tests':count,'files':len(bound)}


def run(path):
    source,value,paths,settings,study_paths,_=load(path);output=paths['output_dir'];verify_seal(paths)
    if (output/'owner.json').exists() or (output/'status.json').exists():
        raise FileExistsError('Fixed queue once-only launch already used; preserve existing state')
    write_json(output/'owner.json',{'pid':os.getpid(),'creation_ticks':creation_time_ticks(os.getpid()),
        'started_utc':now(),'config':str(source),'automatic_retry':False})
    role,child='c5',None;completed=[]
    try:
        for role in (*ROLES,'residual'):
            verify_seal(paths)
            command=[str(study_paths['python']),'-u','-m','scripts.c5r5_fixed_evaluation','action',
                '--config',str(source),'--stage',role]
            with (output/(role+'.stdout.log')).open('x',encoding='utf-8') as stdout, \
                    (output/(role+'.stderr.log')).open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                child_ticks=creation_time_ticks(child.pid)
                _atomic_json(output/'status.json',{'status':'running','stage':role,'completed_stages':completed,
                    'child_pid':child.pid,'child_creation_ticks':child_ticks,'updated_utc':now(),'test_accessed':False})
                try:code=child.wait(timeout=21600)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30)
                    write_json(output/(role+'.exit.json'),{'exit_code':child.returncode,'budget_stop':True,
                        'pid':child.pid,'creation_ticks':child_ticks,'observed_utc':now()})
                    raise RuntimeError('Fixed stage exhausted six-hour wall budget; no continuation')
            write_json(output/(role+'.exit.json'),{'exit_code':code,'pid':child.pid,'creation_ticks':child_ticks,
                'observed_utc':now(),'automatic_retry':False})
            if code:raise RuntimeError(f'{role} exited {code}; preserve failure, no automatic retry')
            if role in ROLES:saved.audit_validation(output/(role+'_val'))
            else:
                from scripts.c5r5_fixed_residual import audit_saved
                audit_saved(paths)
            completed.append(role)
        verify_seal(paths)
        scores={r:saved.audit_validation(output/(r+'_val')) for r in ROLES}
        scores['b0e']=saved.audit_validation(paths['b0e_evaluation'],audit_destination=output/'b0e_saved_result_audit.json')
        write_json(output/'comparison.json',{'status':'completed_fixed_c5r5_development_comparison','scores':scores,
            'gates':gate_comparison(scores,settings['gates']),'residual_gate':read(output/'residual/report.json')['gate'],
            'test_accessed':False,'training_authorized':False,
            'physical_compute_note':'Training 210 extra optimizer computations / 4158 pair forwards; evaluation smoke102. '
                'Prior P1 includes discarded C-S95 epochs, A-S97 >=1100 updates and earlier unknown uncommitted steps; '
                'prior screen/update and restore smokes remain additional physical costs, not credited research updates.'})
        _atomic_json(output/'status.json',{'status':'completed_fixed_evaluation_ready_for_review',
            'completed_stages':completed,'completed_utc':now(),'test_accessed':False,'p2_training_authorized':False})
    except BaseException as error:
        _atomic_json(output/'status.json',{'status':'failed_preserve','stage':role,'completed_stages':completed,
            'error':f'{type(error).__name__}: {error}','updated_utc':now(),'test_accessed':False,'automatic_retry':False})
        raise


def monitor(path):
    _,_,paths,_,study_paths,_=load(path);output=paths['output_dir']
    result={'checked_utc':now(),'test_accessed':False}
    queue=read(output/'status.json') if (output/'status.json').exists() else {}
    result['queue']=queue
    if (output/'owner.json').exists():
        owner=read(output/'owner.json');result['owner']=process_snapshot(owner['pid'],owner['creation_ticks'])
    if queue.get('child_pid') and queue.get('child_creation_ticks'):
        result['worker']=process_snapshot(queue['child_pid'],queue['child_creation_ticks'])
    for stage in (*ROLES,'residual'):
        folder=output/(stage+'_val') if stage in ROLES else output/'residual'
        for name in ('status.json','progress.json'):
            p=folder/name
            if p.is_file():
                row=read(p);result[stage]={k:row[k] for k in ('status','completed_conditions','condition_count',
                    'model','sample','records','error') if k in row}
                result[stage]['progress_age_seconds']=round(time.time()-p.stat().st_mtime,1);break
        log=output/(stage+'.stdout.log')
        if log.exists():
            result.setdefault(stage,{})['log_age_seconds']=round(time.time()-log.stat().st_mtime,1)
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-1200))
                result[stage]['log_tail']=stream.read().decode('utf-8',errors='replace').splitlines()[-2:]
    if queue.get('status')=='running':
        if not result.get('owner',{}).get('owned_active') or not result.get('worker',{}).get('owned_active'):
            result['attention']='Owner/worker creation identity or wait/exit state is not active; diagnose without retry.'
        stage=result.get(queue['stage'],{})
        if stage.get('progress_age_seconds',0)>900 and stage.get('log_age_seconds',0)>900:
            result['attention']='No actual condition/sample progress for >15 minutes; diagnose without retry.'
    guard=load_guard_config(study_paths['guard_config'])
    fields=subprocess.run([guard.values['nvidia_smi'],'--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().split(',')
    if fields[0].strip()!=guard.values['gpu_uuid']:raise ValueError('Monitored GPU identity changed')
    result['gpu']=dict(zip(('temperature_c','utilization_percent','memory_used_mib'),map(int,fields[1:])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','smoke','seal','run','action','monitor'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--stage',choices=(*ROLES,'residual'))
    parser.add_argument('--unittest-log',type=Path);args=parser.parse_args()
    if args.command=='run':
        output=load(args.config)[2]['output_dir']
        with (output/'queue.stdout.log').open('x',encoding='utf-8') as out,(output/'queue.stderr.log').open('x',encoding='utf-8') as err:
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                try:run(args.config)
                except BaseException:traceback.print_exc();raise
        return
    if args.command in ('smoke','action'):
        from scripts import c5r5_fixed_residual as residual
        if args.command=='smoke':result=residual.run(args.config,smoke=True)
        elif args.stage=='residual':result=residual.run(args.config)
        elif args.stage in ROLES:result=evaluate_row(args.config,args.stage)
        else:parser.error('action requires --stage')
    elif args.command=='seal':result=seal(args.config,args.unittest_log)
    else:result=globals()[args.command](args.config)
    if result is not None:
        print(json.dumps(result if args.command=='monitor' else {k:result[k] for k in ('status','tests','files','predictions','records') if k in result},
            ensure_ascii=True,allow_nan=False))


if __name__=='__main__':main()
