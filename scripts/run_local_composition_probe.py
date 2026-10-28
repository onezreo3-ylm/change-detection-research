"""Fixed train-only mechanism probe, separate from all archived evaluators."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import yaml

from pcdc.local_composition_protocol import canonical_sha, optimal_finite_actions
from pcdc.local_composition_protocol import select_original_batches
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/local_composition_probe_20261027.yaml'


class SelectedTrainTiles:
    """Picklable map-style subset; preserves original training identities/order."""
    def __init__(self,actual,indices):
        if getattr(actual,'split',None)!='train': raise ValueError('Only original train data are admitted')
        if (not indices or len(set(indices))!=len(indices)
                or any(type(i) is not int or not 0<=i<len(actual) for i in indices)):
            raise ValueError('Require unique existing original train tile indices')
        self.actual,self.indices,self.split=actual,tuple(indices),'train'
    def __len__(self): return len(self.indices)
    def __getitem__(self,i): return self.actual[self.indices[i]]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=dict(schema_version=1,kind='train_only_fixed_local_action_composition_probe',device='cuda:0',
        amp=True,amp_dtype='float16',memory_format='channels_last',threshold=.5,original_eval_batch_size=32,
        selected_original_batches=8,calibration_containing_batches_first=4,minimum_fit_sources=4,
        minimum_calibration_sources=4,conditions='original_clean_and_all45',edited_stage=2,
        action_subsets='all16_including_noop',action_boxes=[[2,8,2,8],[2,8,24,30],[24,30,2,8],[24,30,24,30]],
        whole_probe_limit_seconds=3600,engineering_reserve_seconds=300,postprocessing_reserve_seconds=120,
        pilot_original_batches=1,pilot_conditions=['clean','brightness_l1'],pilot_wall_multiplier=1.5,
        main_fixed_overhead_reserve_seconds=180,new_model_fitting=False,test_or_martigny_access=False,automatic_retry=False,
        require_exact_original_counts=True,require_exact_noop_logits=True,require_exact_joint_composition=True,
        require_exact_per_sample_count_additivity=True,minimum_oracle_mf1_gain_over_better_noop_or_full_adapter=.005,
        minimum_positive_action_fraction=.05,minimum_negative_action_fraction=.05,minimum_families_each_sign=3)
    if any(v.get(k)!=x for k,x in expected.items()): raise ValueError('Frozen mechanism scope changed')
    if v['output_dir']!='runs/local_composition_probe_20261027': raise ValueError('Unexpected probe root')
    original=yaml.safe_load((ROOT/'configs/b0e_replication_20261026.yaml').read_text(encoding='utf-8'))
    if any(v[k]!=original[k] for k in ('shared_ledger','global_lease')): raise ValueError('Reuse the actual original shared budget and GPU lease')
    return v,ROOT/v['output_dir']


def source_paths():
    names=['pcdc/bound_local_actions.py','pcdc/local_feature_replay.py','pcdc/local_action_composition.py',
        'pcdc/local_composition_protocol.py','scripts/run_local_composition_probe.py',
        'scripts/local_composition_probe_supervision.py','scripts/temperature_guard.py',
        'scripts/levir_tail_risk_supervision.py','scripts/windows_shared_json.py',
        'configs/local_composition_probe_runtime_20261027.yaml','configs/b0e_replication_20261026.yaml',
        'tests/test_bound_local_actions.py','tests/test_local_composition_protocol.py',
        'tests/test_local_composition_probe.py']
    return [ROOT/x for x in names]


def verify(bound):
    for name,expected in bound.items():
        if sha(name)!=expected: raise ValueError('Fixed probe input changed: '+name)


def prepare_metadata():
    from pcdc.phase5_protocol import load_phase56_config,load_frozen_manifests
    from pcdc.phase6_diagnostic import preflight,phase6_source_files
    v,out=load()
    if out.exists(): raise FileExistsError('Preserve existing probe preparation')
    cfg=load_phase56_config(ROOT/v['phase56_config'])
    if (cfg.base.values['train']['eval_batch_size']!=v['original_eval_batch_size'] or
            cfg.base.values['train']['amp'] is not True or
            cfg.base.values['runtime']['memory_format']!=v['memory_format']):
        raise ValueError('Original real-data numerical protocol changed')
    partition,gate=load_frozen_manifests(cfg)
    split=read(ROOT/v['selector_partition'])
    if set(split['fit_names'])|set(split['calibration_names'])!=set(partition['gate_names']):
        raise ValueError('Original train-gate selector identity changed')
    selected=select_original_batches(gate['sample_ids'],split['fit_names'],split['calibration_names'],
        batch_size=v['original_eval_batch_size'],count=v['selected_original_batches'],
        calibration_first=v['calibration_containing_batches_first'],salt=v['selection_salt'],
        minimum_fit=v['minimum_fit_sources'],minimum_calibration=v['minimum_calibration_sources'])
    historical=ROOT/v['historical_diagnostic']; summary=read(historical/'summary.json')
    pf=preflight(cfg)
    if pf['artifact_sha256']!=summary['artifact_sha256']: raise ValueError('Original frozen actor/adapter artifacts changed')
    old_sources=read(historical/'source_sha256.json')
    if old_sources!={name:sha(p) for name,p in phase6_source_files().items()}:
        raise ValueError('Original Phase6 implementation changed')
    ids={sid for batch in selected['batches'] for sid in batch['sample_ids']}
    conditions=list(summary['actions']['a0_no_correction']['conditions'])
    if len(conditions)!=45 or len(set(conditions))!=45: raise ValueError('Original45 conditions required')
    for cid in conditions:
        actual=[x['sample_id'] for x in gate['instances'] if x['condition_id']==cid]
        if actual!=gate['sample_ids']: raise ValueError('Original corruption batch ordering differs')
    references={cid:{} for cid in ['clean',*conditions]}
    for name,hash_key in [('clean_actions.jsonl','clean_actions_sha256'),('action_utility.jsonl','raw_action_utility_sha256')]:
        p=historical/name
        if sha(p)!=summary[hash_key]: raise ValueError('Historical train action records changed')
        with p.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['sample_id'] not in ids: continue
                cid=row.get('condition_id','clean')
                if cid not in references or row['sample_id'] in references[cid]: raise ValueError('Duplicate or unknown reference cell')
                references[cid][row['sample_id']]={key:row[key] for key in ('a0_counts','a1_counts')}
    if any(set(rows)!=ids for rows in references.values()): raise ValueError('Incomplete fixed replay reference')
    paths=[CONFIG,ROOT/v['protocol'],ROOT/v['selector_partition'],historical/'summary.json',
        historical/'source_sha256.json',historical/'clean_actions.jsonl',historical/'action_utility.jsonl',
        *source_paths(),*phase6_source_files().values(),cfg.source,*cfg.paths.values(),cfg.path('actor_last_checkpoint')]
    from pcdc.phase6_diagnostic import _artifact_paths
    paths.extend(_artifact_paths(cfg).values())
    bound={str(p.resolve()):sha(p) for p in paths if p.is_file()}
    if any(not p.is_file() for p in source_paths()): raise FileNotFoundError('Complete the probe implementation before metadata preparation')
    out.mkdir()
    atomic(out/'selected_batches.json',selected)
    atomic(out/'original_replay_counts.json',references)
    (out/'config_snapshot.yaml').write_bytes(CONFIG.read_bytes())
    for name in ('selected_batches.json','original_replay_counts.json','config_snapshot.yaml'):
        bound[str(out/name)]=sha(out/name)
    result=dict(status='metadata_prepared_no_GPU_admission',created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        selected=selected,condition_order=['clean',*conditions],phase56_fingerprint=cfg.fingerprint,
        bound_sha256=bound,dataset_opened=False,model_initialized=False,inference_started=False,
        fitting_started=False,test_accessed=False)
    atomic(out/'prepared_metadata.json',result)
    return dict(status=result['status'],selected_samples=len(ids),source_coverage=selected['source_coverage'])


def diagnose_batch(model,a,b,target,identity,v,check,*,engineering_only=False,expected_counts=None,loss_kwargs=None):
    import torch
    from pcdc.bound_local_actions import BoundLocalActionSession
    from pcdc.local_action_composition import compose_disjoint_outputs
    from pcdc.phase6_diagnostic import per_sample_confusion,per_sample_transitions,COUNT_KEYS
    from pcdc.phase5_models import per_sample_change_loss
    started=time.perf_counter(); outputs={}; effects=[]
    session=BoundLocalActionSession(model,a,b,identity,amp=not engineering_only,engineering_only=engineering_only)
    evidence=dict(session.evidence)
    with session:
        binding=session.binding; baseline=session.baseline(binding=binding)
        noop=session.decode((),binding=binding,arm='reference_replay')
        if not torch.equal(noop,baseline): raise ArithmeticError('Actual no-op logits differ')
        h,w=session._base_features[1].shape[-2:]
        if (h,w)!=(32,32): raise ValueError('Frozen real action geometry requires32x32 stage2')
        for i,(r0,r1,c0,c1) in enumerate(v['action_boxes']):
            mask=torch.zeros(a.shape[0],1,h,w,dtype=torch.bool,device=a.device)
            mask[:,:,r0:r1,c0:c1]=True
            session.register(str(i),mask,binding=binding)
        session.register('whole_adapter',torch.ones(a.shape[0],1,h,w,dtype=torch.bool,device=a.device),binding=binding)
        supports=[session.support(str(i),binding=binding) for i in range(4)]
        union=torch.zeros_like(supports[0])
        for support in supports:
            if (union&support).any(): raise ValueError('Fixed action supports intersect')
            union|=support
        if union.all(): raise ValueError('No outside support remains')
        counts=lambda x:per_sample_confusion(x,target,.5)
        base_counts=counts(baseline)
        full=session.decode(('whole_adapter',),binding=binding,arm='ordinary_GN')
        outputs['a0_counts']=base_counts.tolist();outputs['full_adapter_counts']=counts(full).tolist()
        if not engineering_only and (expected_counts is None or loss_kwargs is None):
            raise ValueError('Real actions require original count references and the frozen loss')
        if expected_counts is not None:
            for index,sid in enumerate(identity.sample_ids):
                for new,old in [('a0_counts','a0_counts'),('full_adapter_counts','a1_counts')]:
                    if outputs[new][index]!=[expected_counts[sid][old][key] for key in COUNT_KEYS]:
                        raise ArithmeticError('Original counts must replay before local actions: '+sid)
        menus={arm:[base_counts.tolist()] for arm in v['arms']}
        gn_singles={};replay_singles={};cost=dict(ordinary_GN_joint=[],reference_replay_joint=[])
        criterion=loss_kwargs if loss_kwargs is not None else dict(bce_weight=1.,dice_weight=1.,dice_smooth=1.)
        base_loss=per_sample_change_loss(baseline,target,**criterion)['total']
        for subset in range(1,16):
            check()
            indices=[i for i in range(4) if subset&(1<<i)]
            names=tuple(str(i) for i in indices)
            selected_support=[supports[i] for i in indices]
            subset_union=torch.zeros_like(union)
            for support in selected_support: subset_union|=support
            for arm,implementation in [('ordinary_GN_joint','ordinary_GN'),('reference_replay_joint','reference_replay')]:
                if a.device.type=='cuda': torch.cuda.synchronize(a.device)
                t=time.perf_counter()
                logits=session.decode(names,binding=binding,arm=implementation)
                if a.device.type=='cuda': torch.cuda.synchronize(a.device)
                cost[arm].append(time.perf_counter()-t)
                current_counts=counts(logits)
                menus[arm].append(current_counts.tolist())
                if len(indices)==1:
                    (gn_singles if arm=='ordinary_GN_joint' else replay_singles)[indices[0]]=logits
                    if arm=='reference_replay_joint':
                        utility=(base_loss-per_sample_change_loss(logits,target,**criterion)['total']).cpu().tolist()
                        effects.append(dict(action=indices[0],loss_utility=utility,
                            gross_transitions=per_sample_transitions(baseline,logits,target,.5).tolist()))
                if arm=='ordinary_GN_joint':
                    masked=torch.where(subset_union,logits,baseline)
                    menus['ordinary_GN_joint_output_mask'].append(counts(masked).tolist())
                    pasted=compose_disjoint_outputs(baseline,[gn_singles[i] for i in indices],selected_support)
                    menus['ordinary_GN_singles_pasted'].append(counts(pasted).tolist())
                else:
                    pasted=compose_disjoint_outputs(baseline,[replay_singles[i] for i in indices],selected_support)
                    if not torch.equal(logits,pasted): raise ArithmeticError('Actual FP16 joint action differs from its individual composition')
                    if torch.count_nonzero((logits-baseline)[~subset_union]): raise ArithmeticError('Actual reference action changed outside its support')
                    expected=base_counts.clone()
                    for i in indices: expected+=counts(replay_singles[i])-base_counts
                    if not torch.equal(expected,current_counts): raise ArithmeticError('Per-sample joint count increments are not additive')
        actual_forwards=dict(session._calls)
    rows=[]
    for index,sid in enumerate(identity.sample_ids):
        rows.append(dict(sample_id=sid,source_id=sid.split('@',1)[0],a0_counts=outputs['a0_counts'][index],
            full_adapter_counts=outputs['full_adapter_counts'][index],
            menus={arm:[subset[index] for subset in menu] for arm,menu in menus.items()},
            singles=[dict(action=e['action'],loss_utility=e['loss_utility'][index],gross_transitions=e['gross_transitions'][index]) for e in effects]))
    return dict(rows=rows,actual_forwards=actual_forwards,evidence=evidence,
        exact_noop=True,exact_joint_composition=True,exact_per_sample_counts=True,
        support_pixels_per_sample=[int(s[0].sum()) for s in supports],actual_decoder_seconds=cost,
        seconds=time.perf_counter()-started,full_final_identity_checks=True)


def summarize(rows,v,split,condition_order):
    result={}
    for group,names in [('all',set(split['fit_names'])|set(split['calibration_names'])),('fit',set(split['fit_names'])),('calibration',set(split['calibration_names']))]:
        selected=[r for r in rows if r['source_id'].removeprefix('train/') in names]
        cells={}; signs=dict(positive=0,negative=0,zero=0);families={}
        for cid in condition_order:
            cell=[r for r in selected if r['condition_id']==cid]
            if not cell: raise ValueError('Empty fixed source/condition diagnostic cell')
            pool=lambda key:[sum(r[key][k] for r in cell) for k in range(4)]
            from pcdc.local_composition_protocol import f1_fraction
            constants={k:float(f1_fraction(pool(k))) for k in ('a0_counts','full_adapter_counts')}
            oracle={arm:optimal_finite_actions([r['menus'][arm] for r in cell]) for arm in v['arms']}
            cells[cid]=dict(fixed_f1=constants,truth_informed_oracles=oracle)
            if cid!='clean':
                family=cid.rsplit('_l',1)[0];families.setdefault(family,set())
                for row in cell:
                    for e in row['singles']:
                        sign='positive' if e['loss_utility']>0 else 'negative' if e['loss_utility']<0 else 'zero'
                        signs[sign]+=1;families[family].add(sign)
        corrupt=[cid for cid in condition_order if cid!='clean']
        means={k:sum(cells[c]['fixed_f1'][k] for c in corrupt)/len(corrupt) for k in ('a0_counts','full_adapter_counts')}
        oracle_means={arm:sum(cells[c]['truth_informed_oracles'][arm]['f1'] for c in corrupt)/len(corrupt) for arm in v['arms']}
        gain=oracle_means['reference_replay_joint']-max(means.values())
        total=sum(signs.values());fractions={k:n/total for k,n in signs.items()}
        gates=dict(oracle_margin=gain>=v['minimum_oracle_mf1_gain_over_better_noop_or_full_adapter'],
            positive_fraction=fractions['positive']>=v['minimum_positive_action_fraction'],
            negative_fraction=fractions['negative']>=v['minimum_negative_action_fraction'],
            positive_families=sum('positive' in s for s in families.values())>=v['minimum_families_each_sign'],
            negative_families=sum('negative' in s for s in families.values())>=v['minimum_families_each_sign'])
        result[group]=dict(sources=sorted({r['source_id'] for r in selected}),sample_condition_rows=len(selected),
            conditions=cells,constant_mf1=means,truth_informed_oracle_mf1=oracle_means,
            reference_oracle_gain_over_better_fixed=gain,action_sign_counts=signs,action_sign_fractions=fractions,
            gates=gates,passed=all(gates.values()))
    return dict(status='completed_fixed_train_only_mechanism_probe',groups=result,
        next_fitting_resource_gate=all(result[g]['passed'] for g in ('fit','calibration')),
        actual_method_effectiveness=False,method_novelty_verified=False,test_accessed=False,optimizer_updates=0)


def probe(session,check):
    import torch
    from pcdc.bound_local_actions import ObservedBatchIdentity
    from pcdc.data import LEVIRDataset,make_loader
    from pcdc.phase5_protocol import load_phase56_config,load_frozen_manifests,verify_partition_files
    from pcdc.phase5_training import SourceSubsetDataset,_validate_actor_completion
    from pcdc.phase6_diagnostic import _load_models,_artifact_hashes,GateInstanceDataset,_transfer,COUNT_KEYS
    from pcdc.runtime import setup_runtime,environment_record
    v,out=load();prepared=read(out/'prepared_metadata.json');verify(read(out/'authorization.json')['bound_sha256'])
    if (out/'probe_rows.jsonl').exists() or (out/'result.json').exists(): raise FileExistsError('No probe rerun')
    cfg=load_phase56_config(ROOT/v['phase56_config']);partition,gate=load_frozen_manifests(cfg)
    check();device=setup_runtime(cfg.base)
    if device.type!='cuda' or device.index not in (None,0): raise ValueError('Original CUDA device required')
    dataset=LEVIRDataset(cfg.base,'train');verify_partition_files(cfg,dataset,partition)
    base=SourceSubsetDataset(dataset,partition['gate_names'],partition='gate')
    expected_ids=[f'train/{base.records[i].name}@x{x}_y{y}' for i,x,y in base.tiles]
    if expected_ids!=gate['sample_ids']: raise ValueError('Actual clean loader ordering changed')
    _validate_actor_completion(cfg)
    hashes=_artifact_hashes(cfg);model,_,_=_load_models(cfg,hashes,device)
    atomic(out/'environment.json',environment_record(cfg.base,device))
    refs=read(out/'original_replay_counts.json');source_map={x['name']:x['file_sha256'] for x in partition['sources']}
    batches=prepared['selected']['batches'];all_indices=[i for batch in batches for i in range(batch['original_start'],batch['original_start']+32)]
    def run(cid,indices,label,stream):
        instances=None
        if cid=='clean': actual=base
        else:
            instances=[x for x in gate['instances'] if x['condition_id']==cid]
            actual=GateInstanceDataset(base,instances)
        loader=make_loader(SelectedTrainTiles(actual,indices),cfg.base,training=False)
        collected=[];timings=[]
        previous=time.perf_counter()
        for batch in loader:
            check();a,b,target=_transfer(batch,device,v['memory_format'])
            ids=tuple(batch['sample_id'])
            identity=ObservedBatchIdentity(ids,tuple((source_map[s.split('@',1)[0].removeprefix('train/')]['image_a'],source_map[s.split('@',1)[0].removeprefix('train/')]['image_b']) for s in ids),
                tuple(canonical_sha(dict(kind='clean',sample_id=s,phase56_fingerprint=cfg.fingerprint)) for s in ids) if cid=='clean' else tuple(hashlib.sha256(x.encode('utf-8')).hexdigest() for x in batch['instance_json']),
                hashes['actor_best'],hashes['adapter_best'],sha(ROOT/v['protocol']))
            record=diagnose_batch(model,a,b,target,identity,v,check,expected_counts=refs[cid],
                loss_kwargs={k:float(cfg.base.values['loss'][k]) for k in ('bce_weight','dice_weight','dice_smooth')})
            for row in record['rows']:
                expected=refs[cid][row['sample_id']]
                for new,old in [('a0_counts','a0_counts'),('full_adapter_counts','a1_counts')]:
                    if row[new]!=[expected[old][key] for key in COUNT_KEYS]: raise ArithmeticError('Actual original train-count replay failed: '+cid+'/'+row['sample_id'])
                row['condition_id']=cid
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();collected.extend(record['rows']);timings.append(time.perf_counter()-previous)
            atomic(out/(label+'_progress.json'),dict(condition_id=cid,completed_rows=len(collected),batch_seconds=record['seconds'],
                actual_forwards=record['actual_forwards'],identity_evidence_sha256=canonical_sha(record['evidence']),
                exact_original_counts=True,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
            atomic(Path(session)/'progress.json',dict(stage=label+'_actual_batch_completed',condition_id=cid,
                committed_condition_rows=len(collected),exact_original_counts=True,
                updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
            check()
            previous=time.perf_counter()
        return collected,timings
    pilot_indices=list(range(batches[0]['original_start'],batches[0]['original_start']+32));pilot_times=[]
    with (out/'pilot_rows.jsonl').open('x',encoding='utf-8') as stream:
        for cid in v['pilot_conditions']:
            _,times=run(cid,pilot_indices,'pilot',stream);pilot_times.extend(times)
    session=Path(session);state=read(session/'owner_state.json')
    predicted=sum(pilot_times)/len(pilot_times)*v['pilot_wall_multiplier']*len(batches)*len(prepared['condition_order'])+v['main_fixed_overhead_reserve_seconds']
    remaining=state['allowed_wall_seconds']-state.get('elapsed_seconds',0)
    admission=dict(status='passed' if predicted<=remaining else 'budget_rejected_no_main_probe',pilot_batch_seconds=pilot_times,
        predicted_main_seconds=predicted,actual_owner_remaining_seconds=remaining,passed=predicted<=remaining)
    atomic(out/'pilot_admission.json',admission)
    if not admission['passed']: return dict(status='pilot_completed_main_budget_rejected',optimizer_updates=0)
    rows=[];times=[]
    with (out/'probe_rows.jsonl').open('x',encoding='utf-8') as stream:
        for cid in prepared['condition_order']:
            rs,ts=run(cid,all_indices,'probe',stream);rows.extend(rs);times.extend(ts)
    result=summarize(rows,v,read(ROOT/v['selector_partition']),prepared['condition_order'])
    result.update(actual_batch_seconds=times,actual_batch_count=len(times),
        actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        actual_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        probe_rows_sha256=sha(out/'probe_rows.jsonl'),pilot_rows_sha256=sha(out/'pilot_rows.jsonl'),
        actual_original_replay_exact=True,actual_FP16_joint_composition_exact=True)
    atomic(out/'result.json',result)
    return result


def smoke():
    import torch
    from pcdc.bound_local_actions import ObservedBatchIdentity
    from pcdc.model import SiameseChangeDetector
    from pcdc.phase5_models import AdapterCorrectedDetector
    v,out=load()
    if not (out/'prepared_metadata.json').exists(): raise ValueError('Prepare metadata before smoke')
    if (out/'smoke.json').exists(): raise FileExistsError('Preserve existing probe smoke')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261027)
        model=AdapterCorrectedDetector(SiameseChangeDetector()).eval().requires_grad_(False)
        torch.nn.init.normal_(model.adapter.conv3.weight,std=.01)
        a,b=torch.rand(1,3,256,256),torch.rand(1,3,256,256);target=(torch.rand(1,1,256,256)<.25).float()
    identity=ObservedBatchIdentity(('train/explicit_synthetic@x0_y0',),(('a'*64,'b'*64),),('c'*64,),'d'*64,'e'*64,'f'*64)
    r=diagnose_batch(model,a,b,target,identity,v,lambda:None,engineering_only=True)
    result=dict(status='passed_CPU_synthetic_all15_actual_combinations',real_data_or_checkpoint=False,
        exact_composition=r['exact_joint_composition'],exact_per_sample_counts=r['exact_per_sample_counts'],
        actual_forwards=r['actual_forwards'],seconds=r['seconds'],optimizer_updates=0)
    atomic(out/'smoke.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-metadata','smoke']);a=parser.parse_args()
    result=prepare_metadata() if a.action=='prepare-metadata' else smoke()
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
