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
CONFIG=ROOT/'configs/local_composition_probe_v2_20261027.yaml'
from scripts import run_local_composition_probe as parent
from scripts.run_local_composition_probe import diagnose_batch,summarize,SelectedTrainTiles
from pcdc.local_composition_resource import require_resource_only_revision,amortized_condition_prediction




def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));old,_=parent.load()
    require_resource_only_revision(v,old)
    if (v['kind']!='train_only_fixed_local_action_composition_probe_resource_v2' or
            v['output_dir']!='runs/local_composition_probe_v2_20261027' or
            v['pilot_estimator']!='max_startup_plus_max_steady_per_condition'):
        raise ValueError('Only the explicit resource v2 identity is admitted')
    return v,ROOT/v['output_dir']


def source_paths():
    return parent.source_paths()+[Path(__file__).resolve(),CONFIG,
        ROOT/'scripts/local_composition_probe_supervision_v2.py',
        ROOT/'configs/local_composition_probe_runtime_v2_20261027.yaml',
        ROOT/'pcdc/local_composition_resource.py',ROOT/'tests/test_local_composition_resource_v2.py']


def verify(bound):
    for name,expected in bound.items():
        if sha(name)!=expected: raise ValueError('Fixed probe input changed: '+name)


def prepare_metadata():
    v,out=load()
    if out.exists():raise FileExistsError('Preserve v2 resource preparation')
    original=ROOT/v['prior_probe_metadata'];prior=read(original);verify(prior['bound_sha256'])
    completion=read(ROOT/v['prior_probe_completion'])
    if completion['status']!='pilot_completed_main_budget_rejected' or not completion['actual_all_roles_inactive']:
        raise ValueError('Actual completed resource-only v1 stop required')
    verify(completion['input_sha256'])
    if (original.parent/'result.json').exists():raise ValueError('Resource v2 is not admitted after observing v1 main scores')
    paths=[*source_paths(),CONFIG,ROOT/v['protocol'],original,ROOT/v['prior_probe_completion']]
    if any(not p.is_file() for p in paths):raise FileNotFoundError('Complete explicit v2 sources before staging')
    bound=dict(prior['bound_sha256']);bound.update({str(p.resolve()):sha(p) for p in paths})
    out.mkdir()
    for name in ('selected_batches.json','original_replay_counts.json'):
        old=original.parent/name;(out/name).write_bytes(old.read_bytes())
        if sha(out/name)!=sha(old):raise ValueError('Scientific metadata byte identity changed')
        bound[str(out/name)]=sha(out/name)
    (out/'config_snapshot.yaml').write_bytes(CONFIG.read_bytes());bound[str(out/'config_snapshot.yaml')]=sha(out/'config_snapshot.yaml')
    result=dict(prior,status='metadata_prepared_resource_v2_no_GPU_admission',
        created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        prior_metadata_created_utc=prior['created_utc'],
        bound_sha256=bound,resource_parent_metadata_sha256=sha(original),
        same_selected_samples_instances_and_scientific_rules=True,
        cumulative_probe_limit_seconds=3600)
    atomic(out/'prepared_metadata.json',result)
    return dict(status=result['status'],selected_samples=prior['selected']['selected_samples'],source_coverage=prior['selected']['source_coverage'])








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
    pilot_indices=[i for batch in batches[:v['pilot_original_batches']] for i in range(batch['original_start'],batch['original_start']+32)];pilot_times=[];pilot_by_condition=[]
    with (out/'pilot_rows.jsonl').open('x',encoding='utf-8') as stream:
        for cid in v['pilot_conditions']:
            _,times=run(cid,pilot_indices,'pilot',stream);pilot_times.extend(times);pilot_by_condition.append(times)
    session=Path(session);state=read(session/'owner_state.json')
    estimate=amortized_condition_prediction(pilot_by_condition,conditions=len(prepared['condition_order']),batches_per_condition=len(batches),multiplier=v['pilot_wall_multiplier'],fixed_overhead=v['main_fixed_overhead_reserve_seconds'])
    predicted=estimate['predicted_main_seconds']
    remaining=state['allowed_wall_seconds']-state.get('elapsed_seconds',0)
    admission=dict(status='passed' if predicted<=remaining else 'budget_rejected_no_main_probe',pilot_batch_seconds=pilot_times,
        predicted_main_seconds=predicted,actual_owner_remaining_seconds=remaining,passed=predicted<=remaining,pilot_by_condition=pilot_by_condition,resource_estimate=estimate)
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
