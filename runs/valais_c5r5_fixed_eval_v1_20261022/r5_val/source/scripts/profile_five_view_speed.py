"""Bounded real-data speed comparison, requiring bitwise training equivalence."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader
import yaml

from pcdc.data import file_sha256, seed_worker
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.metrics import BinaryMetrics
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256, _rng_state, _restore_rng
from pcdc.runtime import setup_runtime, write_json
from scripts import five_view_baselines as f
from scripts import five_view_speed as fast
from scripts.c5r5_40ep_identity import load_study
from scripts.evaluate_guarded import EvaluationTemperatureGuard, load_guard_config


def profile(path):
    source=Path(path).resolve();s=yaml.safe_load(source.read_bytes())
    expected={'schema_version':1,'parent_config':'valais_c5r5_40ep_seed42_v1_20261020.yaml',
        'output_dir':'../runs/valais_c5r5_speed_pilot_v1_20261020','steps_per_variant':16,'warmup_steps':4,
        'maximum_seconds_per_role':600,'minimum_speedup_fraction_each_role':.03,
        'orders':{'c5':['reference','candidate'],'r5':['candidate','reference']},
        'engineering_only':True,'no_validation_scoring':True}
    if s!=expected:raise ValueError('Unreviewed speed pilot scope')
    _,v,paths,_=load_study(source.parent/s['parent_config'])
    old=f.read(paths['output_dir']/'authorization.json')
    for name,sha in old['bound_sha256'].items():
        # The full data byte audit just passed. Each actually used raw sample is
        # checked again by the original dataset; check every non-data input now.
        if not name.startswith(str(paths['output_dir'])) and '/prepared_v2/' in name.replace('\\','/'):
            continue
        if Path(name).suffix=='.pt':continue
        if file_sha256(Path(name))!=sha:raise ValueError('Prior sealed input changed: '+name)
    metadata=f.read(paths['output_dir']/'metadata.json')
    output=(source.parent/s['output_dir']).resolve();output.mkdir(exist_ok=False)
    report={'status':'incomplete','engineering_only':True,'test_accessed':False,'validation_scored':False,'roles':{}}
    try:
        for role in f.ROLES:
            started=time.monotonic();row=output/role;row.mkdir()
            guard=EvaluationTemperatureGuard(load_guard_config(paths['guard_config']),row)
            guard.state.update(training_started=True,execution='bitwise_five_view_speed_pilot')
            physical=0
            def budget():
                if time.monotonic()-started>=600:raise RuntimeError('Speed pilot budget exhausted')
            try:
                guard(force=True)
                from pcdc.config import load_config
                cfg=load_config(paths['output_dir']/role/'base_config.yaml');device=setup_runtime(cfg)
                model=move_model(build_model(cfg,initialize_encoder=False),device,'channels_last').float()
                assert tensor_state_sha256(model.state_dict())==v['initial_model_tensor_sha256']
                optimizer=f.optimizer_for(model,cfg);loss=ChangeDetectionLoss(**cfg.values['loss'])
                dataset=f.FiveViewDataset(metadata,v,metadata['corruptions'],role)
                initial={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
                initial_optimizer=copy.deepcopy(optimizer.state_dict())
                results={};equality={}
                for mode in s['orders'][role]:
                    budget();model.load_state_dict(initial,strict=True);optimizer.load_state_dict(initial_optimizer)
                    sampler=f.UniformEpochSampler(dataset.names,v['source_seed_template'])
                    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=4,pin_memory=True,
                        worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42),
                        collate_fn=fast.collate_view_major if mode=='candidate' else None)
                    if not equality:
                        initial_rng=_rng_state(loader)
                    _restore_rng(initial_rng,loader)
                    iterator=iter(loader);model.train();before_bn=f.bn_counts(model)
                    clean=(fast.DeviceMetrics if mode=='candidate' else BinaryMetrics)(.5)
                    all_views=(fast.DeviceMetrics if mode=='candidate' else BinaryMetrics)(.5)
                    timings=[];loading=[];computing=[];losses=[];batches=[]
                    ih,ph=(hashlib.sha256() for _ in range(2));input_hash=None
                    for step in range(16):
                        budget();guard();tick=time.perf_counter();batch=next(iterator);loaded=time.perf_counter()
                        update=fast.update if mode=='candidate' else f.update
                        values,norm,logits,target=update(model,optimizer,loss,batch,device,cfg.values['train']['grad_clip_norm'])
                        physical+=1
                        n=len(target);clean.update(logits[:n],target);all_views.update(logits,torch.cat([target]*5))
                        torch.cuda.synchronize(device);ended=time.perf_counter()
                        timings.append(ended-tick);loading.append(loaded-tick);computing.append(ended-loaded)
                        losses.append({'losses':values,'norm':norm});batches.append(list(batch['sample_id']))
                        for sid,inst,pixels in zip(batch['sample_id'],batch['instance_json'],batch['candidate_pixels']):
                            f.stream_update(ih,sid,inst);f.stream_update(ph,sid,pixels)
                        if step==0:
                            if mode=='candidate':a,b=batch['a'],batch['b']
                            else:
                                a=batch['a'].transpose(0,1).reshape(-1,3,256,256)
                                b=batch['b'].transpose(0,1).reshape(-1,3,256,256)
                            input_hash=tensor_state_sha256({'a':a,'b':b,'mask':batch['mask']})
                            if mode=='candidate' and (not a.is_pinned() or not b.is_pinned()):
                                raise ValueError('Candidate prepacked CPU inputs are not pinned')
                    metrics={'clean':clean.compute(),'all_views':all_views.compute()}
                    if any(f.bn_counts(model)[k]-x!=16 for k,x in before_bn.items()):raise ValueError('Joint BN count changed')
                    equality[mode]={'model':tensor_state_sha256(model.state_dict()),
                        'optimizer':f.nested_fingerprint(optimizer.state_dict()),'rng':f.nested_fingerprint(_rng_state(loader)),
                        'losses':losses,'metrics':metrics,'batches':f.digest(batches),'instances':ih.hexdigest(),
                        'candidate_pixels':ph.hexdigest(),'first_actual_inputs':input_hash}
                    results[mode]={'timings_seconds':timings,'loading_seconds':loading,'compute_seconds':computing,
                        'steady_mean_seconds':statistics.mean(timings[4:]),
                        'steady_loading_seconds':statistics.mean(loading[4:]),
                        'steady_compute_seconds':statistics.mean(computing[4:])}
                    del iterator,loader,batch,logits,target
                    print(f'{role} {mode}: {results[mode]["steady_mean_seconds"]:.5f}s/step',flush=True)
                exact=equality['reference']==equality['candidate']
                # Preserve evidence even if a candidate fails exact equality. No retries.
                speedup=1-results['candidate']['steady_mean_seconds']/results['reference']['steady_mean_seconds']
                guard.finish({'status':'completed_speed_comparison'});budget()
                elapsed=time.monotonic()-started
                result={'status':'passed_exact_comparison' if exact else 'rejected_numerical_mismatch',
                    'bitwise_model_optimizer_loss_rng_metrics_and_inputs_equal':exact,
                    'speedup_fraction':speedup,'timings':results,'equality_receipts':equality,
                    'physical_optimizer_steps':physical,'research_optimizer_steps':0,'elapsed_seconds':elapsed,
                    'total_engineering_debit_seconds':old['engineering_seconds_debited'][role]+elapsed,
                    'candidate_meets_speed_gate':exact and speedup>=.03}
                write_json(row/'report.json',result);report['roles'][role]=result
                del model,optimizer,dataset,initial,initial_optimizer
                torch.cuda.empty_cache()
            except BaseException as error:
                guard.abort(error)
                write_json(row/'failure.json',{'error':f'{type(error).__name__}: {error}',
                    'elapsed_seconds':time.monotonic()-started,'physical_optimizer_steps':physical,'automatic_retry':False})
                raise
        report.update(status='completed_bounded_speed_pilot',
            candidate_admitted=all(x['candidate_meets_speed_gate'] for x in report['roles'].values()),
            physical_optimizer_steps=sum(x['physical_optimizer_steps'] for x in report['roles'].values()),
            research_optimizer_steps=0)
    finally:write_json(output/'report.json',report)
    return {k:report[k] for k in ('status','candidate_admitted','physical_optimizer_steps')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(profile(args.config)))
