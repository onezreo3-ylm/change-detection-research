"""Frozen train-only upper-bound falsification; no action predictor or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import yaml

from pcdc.local_composition_protocol import canonical_sha, f1_fraction, optimal_finite_actions
from pcdc.local_composition_resource import amortized_condition_prediction
from pcdc.temporal_action_protocol import (ASSIGNMENTS, assignment_counts, expanded_menu,
    old_two_action_menu, require_old_menu_replay, joint_resource_gates)
from scripts.windows_shared_json import atomic_json as atomic, read_json as read
from scripts.run_local_composition_probe import SelectedTrainTiles, sha, verify

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/temporal_action_probe_20261027.yaml'
CONSTANTS=('a0_counts','full_A_only_counts','full_B_only_counts','full_both_counts',
           'local_A_only_counts','local_B_only_counts','local_both_counts')


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=dict(schema_version=1,kind='train_only_temporal_placement_feasibility',
        output_dir='runs/temporal_action_probe_20261027',device='cuda:0',amp=True,amp_dtype='float16',
        memory_format='channels_last',threshold=.5,original_eval_batch_size=32,selected_original_batches=8,
        conditions='original_clean_and_all45',edited_stage=2,
        action_order=['none','A_only','B_only','both'],
        action_boxes=[[2,8,2,8],[2,8,24,30],[24,30,2,8],[24,30,24,30]],
        mixed_main_assignments=[[0,1,2,3],[3,2,1,0],[1,1,2,2],[2,3,0,1]],
        whole_probe_limit_seconds=3600,engineering_reserve_seconds=480,postprocessing_reserve_seconds=120,
        pilot_original_batches=2,pilot_conditions=['clean','brightness_l1'],pilot_exhaustive_batches_per_condition=1,
        pilot_wall_multiplier=1.5,main_fixed_overhead_reserve_seconds=180,
        minimum_oracle_mf1_gain_over_best_seven_constants=.005,
        minimum_oracle_mf1_gain_over_old_two_action_oracle=.002,
        minimum_positive_action_fraction=.05,minimum_negative_action_fraction=.05,minimum_families_each_sign=3,
        new_model_fitting=False,test_or_martigny_access=False,automatic_retry=False,
        require_exact_original_counts=True,require_exact_all16_old_menus=True,require_exact_noop_logits=True,
        require_exact_joint_composition=True,require_exact_per_sample_count_additivity=True)
    if any(v.get(k)!=x for k,x in expected.items()):raise ValueError('Frozen temporal probe scope changed')
    old=yaml.safe_load((ROOT/'configs/local_composition_probe_v2_20261027.yaml').read_text(encoding='utf-8'))
    for key in ('phase56_config','selector_partition','historical_diagnostic','shared_ledger','global_lease',
                'memory_format','amp','amp_dtype','threshold','original_eval_batch_size','selected_original_batches','action_boxes'):
        if v[key]!=old[key]:raise ValueError('Original matched scientific setting changed: '+key)
    return v,ROOT/v['output_dir']


def source_paths():
    return [ROOT/p for p in ('pcdc/bound_temporal_actions.py','pcdc/temporal_action_protocol.py',
        'scripts/run_temporal_action_probe.py','scripts/temporal_action_probe_supervision.py',
        'configs/temporal_action_probe_20261027.yaml','configs/temporal_action_probe_runtime_20261027.yaml',
        'docs/TEMPORAL_ACTION_FEASIBILITY_PROTOCOL_2026-10-27.md',
        'docs/TEMPORAL_ACTION_CLOSE_METHOD_ALIGNMENT_2026-10-27.md',
        'tests/test_temporal_action_probe.py','tests/test_bound_temporal_actions.py')]


def prepare_metadata():
    v,out=load()
    if out.exists():raise FileExistsError('Preserve previous temporal probe preparation')
    prior_path=ROOT/v['prior_probe_metadata'];prior=read(prior_path)
    verify(prior['bound_sha256'])
    completion=read(ROOT/v['prior_probe_completion']);verify(completion['input_sha256'])
    old_result=read(ROOT/v['prior_probe_result'])
    if (completion['status']!='completed_probe' or not completion['actual_all_roles_inactive']
            or old_result['next_fitting_resource_gate'] is not False):
        raise ValueError('Close the completed failed old candidate before this distinct hypothesis')
    paths=[*source_paths(),prior_path,ROOT/v['prior_probe_completion'],ROOT/v['prior_probe_result'],ROOT/v['prior_probe_rows'],
        ROOT/'runs/bound_temporal_actions_engineering_20261027/full_verification.json',
        ROOT/'docs/METHOD_TEMPORAL_ALTERNATIVES_PROPOSAL_2026-10-27.md',
        ROOT/'runs/local_composition_equal_cost_control_20261027/temporal_alternatives_proposal_timing.json']
    if any(not p.is_file() for p in paths):raise FileNotFoundError('Complete all declared implementation and evidence paths')
    bound=dict(prior['bound_sha256']);bound.update({str(p.resolve()):sha(p) for p in paths})
    out.mkdir()
    for name in ('selected_batches.json','original_replay_counts.json'):
        (out/name).write_bytes((prior_path.parent/name).read_bytes())
        bound[str(out/name)]=sha(out/name)
    (out/'config_snapshot.yaml').write_bytes(CONFIG.read_bytes())
    bound[str(out/'config_snapshot.yaml')]=sha(out/'config_snapshot.yaml')
    result=dict(status='temporal_metadata_prepared_no_GPU_admission',
        created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        selected=prior['selected'],condition_order=prior['condition_order'],phase56_fingerprint=prior['phase56_fingerprint'],
        bound_sha256=bound,old_fixed_candidate_closed=True,same_selected_samples_instances=True,
        dataset_opened=False,model_initialized=False,inference_started=False,fitting_started=False,test_accessed=False)
    atomic(out/'prepared_metadata.json',result)
    return dict(status=result['status'],selected_samples=result['selected']['selected_samples'],source_coverage=result['selected']['source_coverage'])


def diagnose_batch(model,a,b,target,identity,v,check,*,exhaustive=False,engineering_only=False,
                   expected_counts=None,expected_old_rows=None,loss_kwargs=None):
    import torch
    from pcdc.bound_temporal_actions import BoundTemporalActionSession
    from pcdc.local_action_composition import compose_disjoint_outputs
    from pcdc.phase6_diagnostic import per_sample_confusion,per_sample_transitions,COUNT_KEYS
    from pcdc.phase5_models import per_sample_change_loss
    if not engineering_only and (expected_counts is None or expected_old_rows is None or loss_kwargs is None):
        raise ValueError('Original actual counts, all16 old menus and fixed loss are required')
    started=time.perf_counter();extra=0.
    session=BoundTemporalActionSession(model,a,b,identity,amp=not engineering_only,engineering_only=engineering_only)
    evidence=dict(session.evidence)
    with session:
        binding=session.binding;baseline=session.baseline(binding=binding)
        if not torch.equal(session.decode((),binding=binding,arm='reference_replay'),baseline):
            raise ArithmeticError('No-op changed the actual logits')
        h,w=session._base_features[1].shape[-2:]
        if (h,w)!=(32,32):raise ValueError('Fixed four-region action geometry requires original32x32 features')
        for region,(r0,r1,c0,c1) in enumerate(v['action_boxes']):
            mask=torch.zeros(a.shape[0],1,h,w,dtype=torch.bool,device=a.device);mask[:,:,r0:r1,c0:c1]=True
            for kind in v['action_order'][1:]:session.register(f'{region}/{kind}',kind,mask,binding=binding)
        for kind in v['action_order'][1:]:
            session.register('whole/'+kind,kind,torch.ones(a.shape[0],1,h,w,dtype=torch.bool,device=a.device),binding=binding)
        supports=[session.support(f'{r}/both',binding=binding) for r in range(4)]
        union=torch.zeros_like(supports[0])
        for support in supports:
            if (union&support).any():raise ValueError('Actual propagated supports intersect')
            union|=support
        if union.all():raise ValueError('Require a nonempty unaffected outside support')
        counts=lambda x:per_sample_confusion(x,target,v['threshold'])
        base=counts(baseline);fixed={'a0_counts':base.tolist()}
        for kind in v['action_order'][1:]:
            check()
            fixed['full_'+kind+'_counts']=counts(session.decode(('whole/'+kind,),binding=binding,arm='ordinary_GN')).tolist()
        if expected_counts is not None:
            for index,sid in enumerate(identity.sample_ids):
                for key,old in [('a0_counts','a0_counts'),('full_both_counts','a1_counts')]:
                    if fixed[key][index]!=[expected_counts[sid][old][k] for k in COUNT_KEYS]:
                        raise ArithmeticError('Original a0/full-both counts failed before local measurement: '+sid)
        outputs={};delta={};effects={}
        base_loss=per_sample_change_loss(baseline,target,**loss_kwargs)['total']
        for action,kind in enumerate(v['action_order'][1:],1):
            check()
            logits=session.decode(tuple(f'{r}/{kind}' for r in range(4)),binding=binding,arm='reference_replay')
            if torch.count_nonzero((logits-baseline)[~union]):raise ArithmeticError('Reference output changed outside all supports')
            outputs[action]=logits;fixed['local_'+kind+'_counts']=counts(logits).tolist()
            additive=base.clone()
            for r,support in enumerate(supports):
                single=torch.where(support,logits,baseline)
                delta[r,action]=counts(single)-base;additive+=delta[r,action]
                effects[r,action]=dict(loss_utility=(base_loss-per_sample_change_loss(single,target,**loss_kwargs)['total']).cpu().tolist(),
                    gross_transitions=per_sample_transitions(baseline,single,target,v['threshold']).tolist())
            if not torch.equal(additive,counts(logits)):raise ArithmeticError('Three-output reconstruction counts are not additive')
        rows=[]
        for index,sid in enumerate(identity.sample_ids):
            row=dict(sample_id=sid,source_id=sid.split('@',1)[0],
                **{key:value[index] for key,value in fixed.items()},
                regional_deltas=[[delta[r,k][index].tolist() for k in range(1,4)] for r in range(4)],
                singles=[dict(region=r,action=k,loss_utility=effects[r,k]['loss_utility'][index],
                    gross_transitions=effects[r,k]['gross_transitions'][index]) for r in range(4) for k in range(1,4)])
            if expected_old_rows is not None:require_old_menu_replay(row,expected_old_rows[sid])
            rows.append(row)
        main_set={tuple(x) for x in v['mixed_main_assignments']}
        assignments=ASSIGNMENTS if exhaustive else tuple(tuple(x) for x in v['mixed_main_assignments'])
        for assignment in assignments:
            check()
            if a.device.type=='cuda':torch.cuda.synchronize(a.device)
            t=time.perf_counter()
            selected=[r for r,k in enumerate(assignment) if k]
            names=tuple(f'{r}/{v["action_order"][assignment[r]]}' for r in selected)
            actual=session.decode(names,binding=binding,arm='reference_replay')
            pasted=compose_disjoint_outputs(baseline,[outputs[assignment[r]] for r in selected],
                [supports[r] for r in selected]) if selected else baseline
            if not torch.equal(actual,pasted):raise ArithmeticError('Actual mixed placement differs from postselection reconstruction')
            expected=[assignment_counts(row['a0_counts'],row['regional_deltas'],assignment) for row in rows]
            if counts(actual).tolist()!=expected:raise ArithmeticError('Actual mixed placement per-sample counts differ')
            if a.device.type=='cuda':torch.cuda.synchronize(a.device)
            if exhaustive and assignment not in main_set:extra+=time.perf_counter()-t
        calls=dict(session._calls)
    return dict(rows=rows,evidence=evidence,actual_forwards=calls,actual_assignments_checked=len(assignments),
        all16_old_menus_replayed=expected_old_rows is not None,exact_noop=True,exact_joint_composition=True,
        exhaustive_additional_wall_seconds=extra,seconds=time.perf_counter()-started,full_final_identity_checks=True)


def summarize(rows,v,split,condition_order):
    groups={}
    for group,names in [('all',set(split['fit_names'])|set(split['calibration_names'])),
                        ('fit',set(split['fit_names'])),('calibration',set(split['calibration_names']))]:
        cells={};signs=dict(positive=0,negative=0,zero=0);families={}
        for cid in condition_order:
            cell=[r for r in rows if r['condition_id']==cid and r['source_id'].removeprefix('train/') in names]
            if not cell:raise ValueError('Missing frozen group/condition cell')
            pool=lambda key:[sum(r[key][k] for r in cell) for k in range(4)]
            old=optimal_finite_actions([old_two_action_menu(r) for r in cell])
            new=optimal_finite_actions([expanded_menu(r) for r in cell])
            if new['f1']<old['f1']:raise ArithmeticError('Expanded action menu lost its old feasible subset')
            cells[cid]=dict(fixed_f1={key:float(f1_fraction(pool(key))) for key in CONSTANTS},
                old16_truth_informed_oracle=old,new256_truth_informed_oracle=new)
            if cid!='clean':
                family=cid.rsplit('_l',1)[0];families.setdefault(family,set())
                for row in cell:
                    for e in row['singles']:
                        sign='positive' if e['loss_utility']>0 else 'negative' if e['loss_utility']<0 else 'zero'
                        signs[sign]+=1;families[family].add(sign)
        corrupt=[c for c in condition_order if c!='clean']
        mean=lambda key:sum(cells[c][key]['f1'] for c in corrupt)/len(corrupt)
        constants={key:sum(cells[c]['fixed_f1'][key] for c in corrupt)/len(corrupt) for key in CONSTANTS}
        fractions={s:n/sum(signs.values()) for s,n in signs.items()}
        new,old=mean('new256_truth_informed_oracle'),mean('old16_truth_informed_oracle')
        gates=joint_resource_gates(new,old,constants,fractions,families,v)
        groups[group]=dict(constant_mf1=constants,old16_oracle_mf1=old,new256_oracle_mf1=new,
            oracle_gain_over_best_fixed=new-max(constants.values()),temporal_oracle_increment=new-old,
            single_action_loss_sign_counts=signs,single_action_loss_sign_fractions=fractions,
            gates=gates,all_gates_passed=all(gates.values()),cells=cells)
    return dict(status='completed_train_only_temporal_action_upper_bound',groups=groups,
        next_fitting_resource_gate=all(groups[g]['all_gates_passed'] for g in ('fit','calibration')),
        actual_method_effectiveness=False,method_novelty_verified=False,head_fitting_authorized=False,
        oracle_uses_ground_truth=True,development_sources_previously_used=True,
        new_optimizer_updates=0,test_accessed=False)


def probe(session,check):
    import torch
    from pcdc.bound_local_actions import ObservedBatchIdentity
    from pcdc.data import LEVIRDataset,make_loader
    from pcdc.phase5_protocol import load_phase56_config,load_frozen_manifests,verify_partition_files
    from pcdc.phase5_training import SourceSubsetDataset,_validate_actor_completion
    from pcdc.phase6_diagnostic import _load_models,_artifact_hashes,GateInstanceDataset,_transfer
    from pcdc.runtime import setup_runtime,environment_record
    v,out=load();prepared=read(out/'prepared_metadata.json');verify(read(out/'authorization.json')['bound_sha256'])
    if (out/'probe_rows.jsonl').exists() or (out/'result.json').exists():raise FileExistsError('No probe rerun')
    cfg=load_phase56_config(ROOT/v['phase56_config']);partition,gate=load_frozen_manifests(cfg)
    check();device=setup_runtime(cfg.base)
    if device.type!='cuda' or device.index not in (None,0):raise ValueError('Original CUDA device required')
    dataset=LEVIRDataset(cfg.base,'train');verify_partition_files(cfg,dataset,partition)
    base=SourceSubsetDataset(dataset,partition['gate_names'],partition='gate')
    if [f'train/{base.records[i].name}@x{x}_y{y}' for i,x,y in base.tiles]!=gate['sample_ids']:
        raise ValueError('Actual original loader ordering changed')
    _validate_actor_completion(cfg);hashes=_artifact_hashes(cfg);model,_,_=_load_models(cfg,hashes,device)
    atomic(out/'environment.json',environment_record(cfg.base,device))
    refs=read(out/'original_replay_counts.json');old_rows={cid:{} for cid in prepared['condition_order']}
    with (ROOT/v['prior_probe_rows']).open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);cid,sid=row['condition_id'],row['sample_id']
            if sid in old_rows[cid]:raise ValueError('Duplicate original local record')
            old_rows[cid][sid]=row
    source_map={x['name']:x['file_sha256'] for x in partition['sources']};batches=prepared['selected']['batches']
    selected_ids={s for b in batches for s in b['sample_ids']}
    if any(set(rs)!=selected_ids for rs in old_rows.values()):raise ValueError('Original local records incomplete')
    all_indices=[i for batch in batches for i in range(batch['original_start'],batch['original_start']+32)]
    loss_kwargs={k:float(cfg.base.values['loss'][k]) for k in ('bce_weight','dice_weight','dice_smooth')}
    session=Path(session)
    def run(cid,indices,label,stream):
        actual=base if cid=='clean' else GateInstanceDataset(base,[x for x in gate['instances'] if x['condition_id']==cid])
        loader=make_loader(SelectedTrainTiles(actual,indices),cfg.base,training=False)
        collected=[];timings=[];previous=time.perf_counter()
        for batch_index,batch in enumerate(loader):
            check();a,b,target=_transfer(batch,device,v['memory_format']);ids=tuple(batch['sample_id'])
            identity=ObservedBatchIdentity(ids,
                tuple((source_map[s.split('@',1)[0].removeprefix('train/')]['image_a'],
                       source_map[s.split('@',1)[0].removeprefix('train/')]['image_b']) for s in ids),
                tuple(canonical_sha(dict(kind='clean',sample_id=s,phase56_fingerprint=cfg.fingerprint)) for s in ids)
                  if cid=='clean' else tuple(hashlib.sha256(x.encode('utf-8')).hexdigest() for x in batch['instance_json']),
                hashes['actor_best'],hashes['adapter_best'],sha(ROOT/v['protocol']))
            r=diagnose_batch(model,a,b,target,identity,v,check,
                exhaustive=label=='pilot' and batch_index<v['pilot_exhaustive_batches_per_condition'],
                expected_counts=refs[cid],expected_old_rows=old_rows[cid],loss_kwargs=loss_kwargs)
            for row in r['rows']:
                row['condition_id']=cid;stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();collected.extend(r['rows'])
            equivalent=time.perf_counter()-previous-r['exhaustive_additional_wall_seconds']
            if equivalent<=0:raise ArithmeticError('Invalid measured pilot cost decomposition')
            timings.append(equivalent)
            progress=dict(stage=label+'_actual_batch_completed',condition_id=cid,committed_condition_rows=len(collected),
                actual_forwards=r['actual_forwards'],actual_assignments_checked=r['actual_assignments_checked'],
                exact_all16_old_counts=True,identity_evidence_sha256=canonical_sha(r['evidence']),
                equivalent_main_batch_seconds=equivalent,exhaustive_extra_seconds=r['exhaustive_additional_wall_seconds'],
                updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
            atomic(out/(label+'_progress.json'),progress);atomic(session/'progress.json',progress);check()
            previous=time.perf_counter()
        return collected,timings
    pilot_indices=[i for batch in batches[:v['pilot_original_batches']] for i in range(batch['original_start'],batch['original_start']+32)]
    pilot_by_condition=[]
    with (out/'pilot_rows.jsonl').open('x',encoding='utf-8') as stream:
        for cid in v['pilot_conditions']:
            _,times=run(cid,pilot_indices,'pilot',stream);pilot_by_condition.append(times)
    estimate=amortized_condition_prediction(pilot_by_condition,conditions=len(prepared['condition_order']),
        batches_per_condition=len(batches),multiplier=v['pilot_wall_multiplier'],fixed_overhead=v['main_fixed_overhead_reserve_seconds'])
    state=read(session/'owner_state.json');remaining=state['allowed_wall_seconds']-state.get('elapsed_seconds',0.)
    admission=dict(status='passed' if estimate['predicted_main_seconds']<=remaining else 'budget_rejected_no_main_probe',
        passed=estimate['predicted_main_seconds']<=remaining,pilot_by_condition=pilot_by_condition,
        predicted_main_seconds=estimate['predicted_main_seconds'],actual_owner_remaining_seconds=remaining,
        resource_estimate=estimate,exhaustive_pilot_already_charged_in_owner_wall=True)
    atomic(out/'pilot_admission.json',admission)
    if not admission['passed']:return dict(status='pilot_completed_main_budget_rejected',optimizer_updates=0)
    rows=[];times=[]
    with (out/'probe_rows.jsonl').open('x',encoding='utf-8') as stream:
        for cid in prepared['condition_order']:
            rs,ts=run(cid,all_indices,'probe',stream);rows.extend(rs);times.extend(ts)
    result=summarize(rows,v,read(ROOT/v['selector_partition']),prepared['condition_order'])
    result.update(actual_batch_count=len(times),actual_equivalent_batch_seconds=times,
        actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        actual_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        probe_rows_sha256=sha(out/'probe_rows.jsonl'),pilot_rows_sha256=sha(out/'pilot_rows.jsonl'),
        actual_original_counts_and_all16_old_menus_exact=True,actual_FP16_mixed_composition_exact=True,
        actual_pilot_exhaustive_assignments_per_batch=256,actual_main_checked_assignments_per_batch=4,
        remaining252_main_assignments_derived_not_actually_decoded=True)
    atomic(out/'result.json',result);return result


def smoke():
    import torch
    from pcdc.bound_local_actions import ObservedBatchIdentity
    from pcdc.model import SiameseChangeDetector
    from pcdc.phase5_models import AdapterCorrectedDetector
    v,out=load()
    if not (out/'prepared_metadata.json').exists():raise ValueError('Prepare metadata first')
    if (out/'smoke.json').exists():raise FileExistsError('Preserve smoke evidence')
    torch.set_num_threads(v['smoke']['threads']);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(v['smoke']['seed']);model=AdapterCorrectedDetector(SiameseChangeDetector()).eval().requires_grad_(False)
        torch.nn.init.normal_(model.adapter.conv3.weight,std=v['smoke']['nonzero_adapter_final_std'])
        a,b=torch.rand(1,3,256,256),torch.rand(1,3,256,256)
        target=(torch.rand(1,1,256,256)<v['smoke']['synthetic_foreground_probability']).float()
    identity=ObservedBatchIdentity(('train/explicit_synthetic@x0_y0',),(('a'*64,'b'*64),),('c'*64,),'d'*64,'e'*64,'f'*64)
    loss_kwargs=v['smoke']['loss']
    r=diagnose_batch(model,a,b,target,identity,v,lambda:None,engineering_only=True,exhaustive=True,loss_kwargs=loss_kwargs)
    result=dict(status='passed_CPU_temporal_all256_combinations',actual_forwards=r['actual_forwards'],
        actual_assignments_checked=r['actual_assignments_checked'],seconds=r['seconds'],
        exact_joint_composition=True,full_final_identity_checks=True,real_data_or_checkpoint=False,optimizer_updates=0)
    atomic(out/'smoke.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare-metadata','smoke']);a=p.parse_args()
    print(json.dumps(prepare_metadata() if a.action=='prepare-metadata' else smoke(),ensure_ascii=True))

if __name__=='__main__':main()
