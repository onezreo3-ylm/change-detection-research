"""Independent 40-epoch worker; original 100-epoch code and artifacts stay sealed."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import torch

from pcdc.config import load_config, protocol_signature
from pcdc.data import LEVIRDataset, make_loader, file_sha256
from pcdc.engine import move_model, run_epoch
from pcdc.losses import ChangeDetectionLoss
from pcdc.metrics import BinaryMetrics
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256, _rng_state
from pcdc.runtime import setup_runtime, environment_record, save_checkpoint, write_json
from scripts.evaluate_guarded import EvaluationTemperatureGuard, load_guard_config, _atomic_json
from scripts.five_view_baselines import (ROLES, read, digest, now, canonical, verify_prepared,
    bn_counts, make_train_loader, optimizer_for, stream_update, update, CheckedValLoader)
from scripts.c5r5_runtime_identity import FLAVOR, load_study, implementation_files, role_debit

from scripts import five_view_speed as fast
from scripts import five_view_bf16 as bf16
from scripts import five_view_baselines as f
from pcdc.data import seed_worker

def runtime_loader(metadata,settings,corruptions,role):
    if settings['optimization_mode']!='pinned_layout_device_metrics':
        return make_train_loader(metadata,settings,corruptions,role)
    dataset=f.FiveViewDataset(metadata,settings,corruptions,role)
    sampler=f.UniformEpochSampler(dataset.names,settings['source_seed_template'])
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,num_workers=4,pin_memory=True,
        worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42),collate_fn=fast.collate_view_major)
    return dataset,sampler,loader


def validate_checkpoint(cp,cfg,identity,plan):
    if (cp.get('format_version')!=1 or cp.get('kind')!='research' or cp.get('training_flavor')!=FLAVOR
            or cp.get('identity')!=identity or cp.get('protocol_signature')!=protocol_signature(cfg.values)
            or cp.get('config')!=cfg.values or cp.get('role')!=identity['role']
            or cp.get('training_signature')!=digest(identity)
            or cp.get('source_sha256')!=identity['implementation_sha256']
            or cp.get('initial_model_tensor_sha256')!=identity['initial_model_tensor_sha256']
            or digest(cp.get('file_sha256',{}))!=identity['data_files_sha256']
            or digest(cp.get('manifests',{}))!=identity['manifests_sha256']
            or set(cp.get('file_sha256',{}))!={'train','val'}
            or len(cp.get('history',[]))!=cp.get('epoch') or not 1<=cp.get('epoch',0)<=40):
        raise ValueError('Five-view checkpoint scope/identity changed')
    if (cfg.values['train']['epochs']!=40 or cp['scheduler']['T_max']!=40
            or cp['scheduler']['last_epoch']!=cp['epoch']):
        raise ValueError('40-epoch cosine schedule identity changed')
    if cfg.values['train']['amp']!=(identity.get('optimization_mode')=='bf16_training_fp32_validation'):
        raise ValueError('Checkpoint numerical precision identity changed')
    history=cp['history']
    best=max(history,key=lambda r:r['val']['metrics']['f1'])
    if cp['best_epoch']!=best['epoch'] or cp['best_val_f1']!=best['val']['metrics']['f1']:
        raise ValueError('Not first highest clean-val selection')
    for epoch,row in enumerate(history,1):
        train=row['train']
        observed={k:train[k] for k in plan[epoch-1]}
        if (row['epoch']!=epoch or observed!=plan[epoch-1] or train['optimizer_steps']!=2757
                or train['amp_skipped_steps']!=0 or not train['actual_streams_verified']):
            raise ValueError('Incomplete successful budget or source/instance stream')
    tensors=list(cp['model'].values())+[v for state in cp['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in tensors):
        raise ValueError('Parameters or optimizer state no longer FP32')
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError('Nonfinite completed five-view state')


def train(path,role):
    source,v,paths,_=load_study(path)
    update_fn=(bf16.update if v['optimization_mode']=='bf16_training_fp32_validation'
        else fast.update if v['optimization_mode']=='pinned_layout_device_metrics' else update)
    metric_class=fast.DeviceMetrics if v['optimization_mode']=='pinned_layout_device_metrics' else BinaryMetrics
    if role not in ROLES: raise ValueError('Unknown ordinary control')
    output=paths['output_dir']/role
    if (output/'last.pt').exists() or (output/'status.json').exists():
        raise FileExistsError('Fresh worker only; failed/paused runs require an explicit recovery record')
    started=time.monotonic()
    cfg=load_config(output/'base_config.yaml')
    _atomic_json(output/'progress.json',{'status':'verifying_inputs','role':role,'updated_utc':now(),'epoch':0,'step':0})
    verify_prepared(paths,sealed=True)
    metadata=read(paths['output_dir']/'metadata.json')
    plan=read(paths['output_dir']/'sampling_plan.json')
    identity=read(output/'identity.json')
    pilot_debit=role_debit(paths,role)
    remaining_seconds=v['maximum_training_seconds_per_role']-pilot_debit
    if not 0<remaining_seconds<=86400: raise ValueError('User time budget has no remaining training allocation')
    guard=EvaluationTemperatureGuard(load_guard_config(paths['guard_config']),output)
    guard.state.update(training_started=True,execution='five_view_training_between_batches')
    history=[];actual=[];best_f1=-1.;best_epoch=0
    def budget():
        if time.monotonic()-started>=remaining_seconds:
            raise RuntimeError('Frozen worker time budget exhausted; no automatic extension')
    try:
        guard(force=True)
        device=setup_runtime(cfg)
        model=move_model(build_model(cfg,initialize_encoder=False),device,'channels_last').float()
        if tensor_state_sha256(model.state_dict())!=v['initial_model_tensor_sha256']:
            raise ValueError('Changed original seed42 initialization')
        optimizer=optimizer_for(model,cfg)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40,eta_min=cfg.values['train']['min_learning_rate'])
        loss_fn=ChangeDetectionLoss(**cfg.values['loss'])
        dataset,sampler,loader=runtime_loader(metadata,v,metadata['corruptions'],role)
        val_dataset=LEVIRDataset(cfg,'val')
        val_loader=make_loader(val_dataset,cfg,training=False)
        if val_dataset.manifest()!={k:x for k,x in metadata['audit']['splits']['val'].items() if k!='file_sha256'}:
            raise ValueError('Validation inventory changed')
        write_json(output/'environment.json',environment_record(cfg,device))
        print(f'{role} source/identity verified; FP32 parameters / {v["optimization_mode"]} CUDA training starts',flush=True)
        for epoch in range(1,41):
            budget();guard();model.train();dataset.set_epoch(epoch);sampler.set_epoch(epoch)
            before_bn=bn_counts(model);metrics=metric_class(.5);combined=metric_class(.5)
            sums={k:0. for k in ('total','bce','dice')};seen=0;steps=0;batch_ids=[]
            inst_h,pixels_h,used_h=(hashlib.sha256() for _ in range(3));counts=Counter();degenerate=0
            expected_batches=sampler.batches()
            epoch_started=time.monotonic();lr=optimizer.param_groups[0]['lr']
            for step,batch in enumerate(loader,1):
                budget();guard()
                expected=[f'train/{sampler.names[i]}@x0_y0' for i in expected_batches[step-1]]
                ids=list(batch['sample_id'])
                if ids!=expected: raise ValueError('Actual loaded batch differs from sealed source sampler')
                batch_ids.append(ids)
                losses,norm,logits,target=update_fn(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
                steps+=1;n=len(ids);seen+=n
                for k in sums: sums[k]+=losses[k]*n
                metrics.update(logits[:n],target);combined.update(logits,torch.cat([target]*5))
                for sid,instance,pixels,used in zip(ids,batch['instance_json'],batch['candidate_pixels'],batch['used_pixels']):
                    stream_update(inst_h,sid,instance);stream_update(pixels_h,sid,pixels);stream_update(used_h,sid,used)
                    item=json.loads(instance);counts[f"{item['family']}|l{item['level']}"]+=1
                degenerate+=int(batch['degenerate'].sum())
                if step==1 or step%v['progress_every_steps']==0 or step==len(loader):
                    state={'status':'training','role':role,'epoch':epoch,'step':step,'steps_per_epoch':2757,
                        'successful_steps_in_epoch':steps,'committed_epochs':epoch-1,'updated_utc':now(),
                        'elapsed_seconds':time.monotonic()-started,'loss':sums['total']/seen}
                    _atomic_json(output/'progress.json',state)
                    print(f'{role} epoch={epoch}/40 step={step}/2757 loss={state["loss"]:.6f}',flush=True)
            observed={'epoch':epoch,'source_draws':seen,'unique_sources':len({x for b in batch_ids for x in b}),
                'batches':len(batch_ids),'tail_sources':len(batch_ids[-1]),'pair_views':seen*5,
                'source_batch_sha256':digest(batch_ids),'instance_stream_sha256':inst_h.hexdigest(),
                'instance_counts':dict(sorted(counts.items()))}
            if observed!=plan[epoch-1] or any(bn_counts(model)[n]-x!=steps for n,x in before_bn.items()):
                raise ValueError('Actual epoch source/instance/BN budget differs from sealed plan')
            train_row={**observed,'loss':{k:x/seen for k,x in sums.items()},'clean_metrics':metrics.compute(),
                'equal_view_pooled_metrics':combined.compute(),'optimizer_steps':steps,'amp_skipped_steps':0,
                'actual_streams_verified':True,'candidate_pixel_stream_sha256':pixels_h.hexdigest(),
                'actual_view_pixel_stream_sha256':used_h.hexdigest(),'degenerate_sources':degenerate,
                'seconds':time.monotonic()-epoch_started}
            _atomic_json(output/'progress.json',{'status':'validation','role':role,'epoch':epoch,'step':2757,
                'committed_epochs':epoch-1,'updated_utc':now(),'elapsed_seconds':time.monotonic()-started})
            guard(force=True)
            val=run_epoch(model,CheckedValLoader(val_loader,guard,budget),loss_fn,device,threshold=.5,
                amp=False,memory_format='channels_last',progress=f'{role}/epoch{epoch}/val')
            guard(force=True);budget()
            history.append({'epoch':epoch,'learning_rate':lr,'train':train_row,'val':val})
            actual.append(observed)
            improved=val['metrics']['f1']>best_f1
            if improved: best_f1=val['metrics']['f1'];best_epoch=epoch
            scheduler.step()
            current_sources={k:file_sha256(x) for k,x in implementation_files().items()}
            if current_sources!=identity['implementation_sha256']:
                raise ValueError('Frozen training implementation changed during execution')
            cp={'format_version':1,'kind':'research','training_flavor':FLAVOR,'role':role,'identity':identity,
                'epoch':epoch,'best_epoch':best_epoch,'best_val_f1':best_f1,'config':cfg.values,
                'protocol_signature':protocol_signature(cfg.values),'training_signature':digest(identity),
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':_rng_state(loader),'history':history,'initial_model_tensor_sha256':v['initial_model_tensor_sha256'],
                'source_sha256':current_sources,
                'file_sha256':{k:x['file_sha256'] for k,x in metadata['audit']['splits'].items()},
                'manifests':{k:{kk:xx for kk,xx in x.items() if kk!='file_sha256'} for k,x in metadata['audit']['splits'].items()},
                'elapsed_seconds':time.monotonic()-started,'pilot_seconds_debited':pilot_debit,
                'worker_time_limit_seconds':remaining_seconds,'selection':'first_highest_clean_val_f1'}
            save_checkpoint(output/'last.pt',cp)
            if improved: save_checkpoint(output/'best.pt',cp)
            _atomic_json(output/'history.json',{'rows':history})
            _atomic_json(output/'sampling_actual.json',{'rows':actual})
            status={'status':'running','role':role,'epoch':epoch,'epochs':40,'best_epoch':best_epoch,'best_val_f1':best_f1,
                'optimizer_steps':2757*epoch,'pair_views':55125*epoch,'source_draws':11025*epoch,
                'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_evaluated':False,'training_flavor':FLAVOR,'optimization_mode':v['optimization_mode']}
            status.update(pilot_seconds_debited=pilot_debit,worker_time_limit_seconds=remaining_seconds,
                cumulative_role_seconds_including_pilot=status['elapsed_seconds']+pilot_debit)
            _atomic_json(output/'status.json',status)
            _atomic_json(output/'progress.json',dict(status,committed_epochs=epoch,step=2757))
            print(f'{role} epoch={epoch} clean_val_f1={val["metrics"]["f1"]:.9f} best_epoch={best_epoch}',flush=True)
        budget()
        validate_checkpoint(cp,cfg,identity,plan)
        guard.finish({'status':'completed_five_view_training'})
        budget()
        status.update(status='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-started,
            cumulative_role_seconds_including_pilot=time.monotonic()-started+pilot_debit)
        _atomic_json(output/'status.json',status)
        write_json(output/'guard_completion.json',{'status':'completed','training_status_sha256':file_sha256(output/'status.json'),
            'guard_state_sha256':file_sha256(output/'temperature_guard_state.json'),
            'history_sha256':file_sha256(output/'history.json'),'sampling_sha256':file_sha256(output/'sampling_actual.json'),
            'best_sha256':file_sha256(output/'best.pt'),'last_sha256':file_sha256(output/'last.pt')})
        return status
    except BaseException as error:
        guard.abort(error)
        _atomic_json(output/'status.json',{'status':'failed_preserve','role':role,'committed_epochs':len(history),
            'error':f'{type(error).__name__}: {error}','elapsed_seconds':time.monotonic()-started,
            'updated_utc':now(),'automatic_retry':False,'test_evaluated':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--role',choices=ROLES,required=True)
    args=parser.parse_args()
    print(canonical(train(args.config,args.role)))


if __name__=='__main__': main()
