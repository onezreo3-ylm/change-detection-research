"""Explicit, matched C5/R5 ordinary training; archived trainers remain immutable."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from pcdc.config import Config, UniqueKeyLoader, load_config, protocol_signature
from pcdc.data import file_sha256, seed_worker, LEVIRDataset, make_loader
from pcdc.engine import move_model, source_files, run_epoch
from pcdc.losses import ChangeDetectionLoss
from pcdc.metrics import BinaryMetrics
from pcdc.model import build_model
from pcdc.perturbations import corrupt_rgb
from pcdc.robust_baselines import tensor_state_sha256, _rng_state, _restore_rng
from pcdc.runtime import environment_record, setup_runtime, save_checkpoint, write_json
from scripts.evaluate_guarded import EvaluationTemperatureGuard, _atomic_json, load_guard_config
from scripts import evaluate_coupling_p0 as p0
from scripts.prepare_coupling_p0 import draw_instance, train_path
from scripts.coupling_diagnostic import image_sha256, validate_marginal_hashes

ROOT=Path(__file__).resolve().parents[1]
FLAVOR='matched_five_view_baseline_v1'
ROLES=('c5','r5')
FAMILIES=('brightness','contrast','gamma','color_bias')
LEVELS=(1,3,5)
VIEWS=('clean','uu','uv','vu','vv')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


def load_study(path):
    source=Path(path).resolve()
    v=yaml.load(source.read_bytes(),Loader=UniqueKeyLoader)
    expected={'schema_version':1,'roles':list(ROLES),'epochs':100,'source_batch_size':4,
        'source_tiles':11025,'batches_per_epoch':2757,'seed':42,'pair_seed':2026102002,
        'families':list(FAMILIES),'levels':list(LEVELS),'view_order':list(VIEWS),
        'view_weights':[.5,.125,.125,.125,.125],'pilot_steps_per_role':32,'pilot_warmup_steps':8,
        'pilot_max_seconds_per_role':1800,'pilot_training_time_multiplier':1.25,
        'pilot_extra_seconds_per_epoch':60,'progress_every_steps':100,'check_interval_minutes':90,
        'validation_seconds_per_role':21600,'residual_seconds':21600}
    if any(v.get(k)!=x for k,x in expected.items()) or v['maximum_training_seconds_per_role'] not in (86400,172800):
        raise ValueError('Changed C5/R5 finite comparison or budget')
    expected_gates={'r5_vs_c5':{'minimum_corruption_mf1_gain':.005,'maximum_clean_f1_drop':.005,
        'maximum_clean_recall_drop':.01,'maximum_corruption_recall_drop':.01,
        'maximum_clean_empty_fpr_increase':.0002,'maximum_corruption_empty_fpr_increase':.0002},
        'c5_vs_b0e':{'maximum_clean_f1_drop':.005,'maximum_clean_recall_drop':.01,
        'maximum_corruption_mf1_drop':.005,'maximum_corruption_recall_drop':.01,
        'maximum_clean_empty_fpr_increase':.0002,'maximum_corruption_empty_fpr_increase':.0002}}
    if v['gates']!=expected_gates: raise ValueError('Changed predeclared gates')
    keys=('output_dir','python','base_config','data_audit','label_report','corruption_config',
        'p0_manifest','p0_review','guard_config','protocol_document','prior_screen_closeout')
    paths={k:(source.parent/v[k]).resolve() for k in keys}
    paths['output_dir'].relative_to(ROOT/'runs')
    base=load_config(paths['base_config'])
    values=base.values
    if (base.run_test_after_training or values['runtime']!={'device':'cuda','deterministic':True,'num_threads':4,'memory_format':'channels_last'}
            or values['train']['amp'] or values['train']['epochs']!=100 or values['train']['batch_size']!=4
            or values['experiment']['seed']!=42 or values['model']['encoder_weights'] is not None
            or values['evaluation']['threshold']!=.5 or v['minimum_unittests']<377):
        raise ValueError('Base scope, seed, numerical or device settings changed')
    return source,v,paths,base


def implementation_files():
    files=dict(source_files())
    for name in ('scripts/five_view_baselines.py','scripts/valais_c5r5_study.py',
        'scripts/evaluate_coupling_p0.py','scripts/prepare_coupling_p0.py','scripts/coupling_diagnostic.py',
        'scripts/evaluate_guarded.py','scripts/temperature_guard.py','scripts/p1_process_state.py',
        'scripts/run_valais_clean_guarded.py'):
        files[name]=ROOT/name
    return files


def verify_prepared(paths,*,sealed=False):
    filename='authorization.json' if sealed else 'prepared_identity.json'
    proof=read(paths['output_dir']/filename)
    for name,sha in proof['bound_sha256'].items(): p0.checked_hash(name,sha)
    return proof


def instance_for(sid,epoch,settings,corruptions):
    if not sid.startswith('train/') or type(epoch) is not int or not 1<=epoch<=100:
        raise ValueError('Pair instances are train-only within the frozen epochs')
    seed=int(digest(['five-view-family-level-v1',settings['pair_seed'],epoch,sid])[:16],16)
    family=FAMILIES[seed%4]
    level=LEVELS[(seed//4)%3]
    key=f'epoch:{epoch}|{sid}'
    return {'family':family,'level':level,
        'u':draw_instance(corruptions,family,level,settings['pair_seed'],key,'u'),
        'v':draw_instance(corruptions,family,level,settings['pair_seed'],key,'v')}


def stream_update(hasher,sid,instance):
    hasher.update((canonical([sid,instance])+'\n').encode())


class UniformEpochSampler(Sampler):
    def __init__(self,names,seed_template,epoch=1,batch_size=4):
        self.names=tuple(names);self.template=seed_template;self.batch_size=batch_size
        if len(set(names))!=len(names) or not names or batch_size!=4:
            raise ValueError('Unique complete source names and batch4 required')
        self.set_epoch(epoch)

    def set_epoch(self,epoch):
        if type(epoch) is not int or not 1<=epoch<=100: raise ValueError('Epoch outside finite budget')
        self.epoch=epoch

    def batches(self):
        indices=list(range(len(self.names)))
        random.Random(self.template.format(epoch=self.epoch)).shuffle(indices)
        return tuple(tuple(indices[i:i+4]) for i in range(0,len(indices),4))

    def __len__(self): return (len(self.names)+3)//4
    def __iter__(self): return iter(self.batches())

    def summary(self,settings,corruptions):
        batches=self.batches()
        ids=[[f'train/{self.names[i]}@x0_y0' for i in b] for b in batches]
        h=hashlib.sha256();counts=Counter()
        for batch in ids:
            for sid in batch:
                inst=instance_for(sid,self.epoch,settings,corruptions)
                stream_update(h,sid,canonical(inst));counts[f"{inst['family']}|l{inst['level']}"]+=1
        return {'epoch':self.epoch,'source_draws':len(self.names),'unique_sources':len(self.names),
            'batches':len(batches),'tail_sources':len(batches[-1]),'pair_views':5*len(self.names),
            'source_batch_sha256':digest(ids),'instance_stream_sha256':h.hexdigest(),
            'instance_counts':dict(sorted(counts.items()))}


class FiveViewDataset(Dataset):
    split='train'
    def __init__(self,metadata,settings,corruptions,role):
        if role not in ROLES: raise ValueError('Expected explicit C5 or R5')
        self.metadata,self.settings,self.corruptions,self.role=metadata,settings,corruptions,role
        self.names=tuple(sorted(metadata['audit']['splits']['train']['file_sha256']))
        self.epoch=1
        data=metadata['base_values']['data']
        self.mean=torch.tensor(data['mean']).view(3,1,1)
        self.std=torch.tensor(data['std']).view(3,1,1)
        self.config=Config(metadata['base_values'],Path(metadata['base_config_source']))
        if self.config.run_test_after_training or self.config.data_root!=Path(metadata['data_root']).resolve():
            raise ValueError('Dataset base/root changed')
        if (self.config.data_root/'SYNTHETIC_ONLY.json').exists() and self.config.values['experiment']['kind']!='smoke':
            raise ValueError('Synthetic fixtures cannot enter research training')
        for name in self.names:
            for folder in ('A','B','label'): train_path(self.config.data_root,folder,name)

    def set_epoch(self,epoch):
        if type(epoch) is not int or not 1<=epoch<=100: raise ValueError('Invalid epoch')
        self.epoch=epoch

    def __len__(self): return len(self.names)

    def __getitem__(self,index):
        name=self.names[index];sid=f'train/{name}@x0_y0'
        expected=self.metadata['audit']['splits']['train']['file_sha256'][name]
        sample={'name':name,'id':sid,'foreground_pixels':self.metadata['positive_pixels'].get(name,0),
            'files':{k:{'path':str(train_path(self.config.data_root,f,name)),'sha256':expected[k]}
                for k,f in (('image_a','A'),('image_b','B'),('mask','label'))}}
        a,b,target=p0.read_sample(sample,self.config)
        inst=instance_for(sid,self.epoch,self.settings,self.corruptions)
        candidate={}
        for date,raw in (('a',a),('b',b)):
            for variant in ('u','v'):
                x=inst[variant]
                candidate[date+variant]=corrupt_rgb(raw,inst['family'],x['parameters'],seed=x['seed'])
        hashes={k:image_sha256(x) for k,x in candidate.items()}
        marginal={cell:[hashes[x],hashes[y]] for cell,(x,y) in p0.CELLS.items()}
        check=validate_marginal_hashes(marginal)
        if not check['matched']: raise ValueError('Actual candidate marginal mismatch')
        clean=(a,b)
        pairs=[clean]*5 if self.role=='c5' else [clean]+[(candidate[x],candidate[y]) for x,y in p0.CELLS.values()]
        def norm(raw):
            t=torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1))).float()/255
            return (t-self.mean)/self.std
        used=[[image_sha256(x),image_sha256(y)] for x,y in pairs]
        return {'a':torch.stack([norm(x) for x,_ in pairs]),'b':torch.stack([norm(y) for _,y in pairs]),
            'mask':target.squeeze(0),'sample_id':sid,'instance_json':canonical(inst),
            'candidate_pixels':canonical(hashes),'used_pixels':canonical(used),
            'degenerate':check['fully_degenerate']}


def make_train_loader(metadata,settings,corruptions,role,*,workers=None):
    dataset=FiveViewDataset(metadata,settings,corruptions,role)
    sampler=UniformEpochSampler(dataset.names,settings['source_seed_template'])
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=4 if workers is None else workers,
        pin_memory=True,worker_init_fn=seed_worker,generator=generator)
    return dataset,sampler,loader


def transfer(batch,device):
    a,b=batch['a'],batch['b']
    if a.shape!=b.shape or a.ndim!=5 or a.shape[1:]!=(5,3,256,256):
        raise ValueError('Five actual aligned 256px views required')
    a=a.transpose(0,1).reshape(-1,3,256,256).to(device,memory_format=torch.channels_last,non_blocking=True)
    b=b.transpose(0,1).reshape(-1,3,256,256).to(device,memory_format=torch.channels_last,non_blocking=True)
    return a,b,batch['mask'].to(device,non_blocking=True)


def weighted_loss(logits,target,loss_fn):
    n=len(target)
    if logits.shape!=(5*n,1,*target.shape[-2:]): raise ValueError('Wrong five-view output shape')
    rows=[loss_fn(logits[i*n:(i+1)*n],target) for i in range(5)]
    weights=(.5,.125,.125,.125,.125)
    return {k:sum(w*r[k] for w,r in zip(weights,rows)) for k in ('total','bce','dice')}


def optimizer_for(model,cfg):
    t=cfg.values['train']
    return torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],betas=t['adam_betas'],weight_decay=t['weight_decay'])


def update(model,optimizer,loss_fn,batch,device,clip):
    a,b,target=transfer(batch,device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',enabled=False):
        logits=model(a,b)['logits'];losses=weighted_loss(logits,target,loss_fn)
    if logits.dtype!=torch.float32 or not all(torch.isfinite(x).item() for x in losses.values()):
        raise FloatingPointError('Non-FP32 or nonfinite five-view loss')
    losses['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    return {k:float(x.detach()) for k,x in losses.items()},float(norm),logits.detach(),target


def bn_counts(model):
    return {n:int(m.num_batches_tracked) for n,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}


def nested_fingerprint(value):
    def convert(x):
        if isinstance(x,torch.Tensor): return {'tensor':tensor_state_sha256({'v':x})}
        if isinstance(x,dict): return [[str(k),convert(v)] for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))]
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        return x
    return digest(convert(value))


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
            or len(cp.get('history',[]))!=cp.get('epoch') or not 1<=cp.get('epoch',0)<=100):
        raise ValueError('Five-view checkpoint scope/identity changed')
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
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError('Nonfinite completed five-view state')


class CheckedValLoader:
    def __init__(self,loader,guard,budget): self.loader,self.guard,self.budget=loader,guard,budget
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for row in self.loader:
            self.budget();self.guard();yield row


def train(path,role):
    source,v,paths,_=load_study(path)
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
    pilot_debit=read(paths['output_dir']/'multi_batch_cuda_pilot/report.json')['roles'][role]['elapsed_seconds']
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
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=cfg.values['train']['min_learning_rate'])
        loss_fn=ChangeDetectionLoss(**cfg.values['loss'])
        dataset,sampler,loader=make_train_loader(metadata,v,metadata['corruptions'],role)
        val_dataset=LEVIRDataset(cfg,'val')
        val_loader=make_loader(val_dataset,cfg,training=False)
        if val_dataset.manifest()!={k:x for k,x in metadata['audit']['splits']['val'].items() if k!='file_sha256'}:
            raise ValueError('Validation inventory changed')
        write_json(output/'environment.json',environment_record(cfg,device))
        print(f'{role} source/identity verified; FP32 CUDA training starts',flush=True)
        for epoch in range(1,101):
            budget();guard();model.train();dataset.set_epoch(epoch);sampler.set_epoch(epoch)
            before_bn=bn_counts(model);metrics=BinaryMetrics(.5);combined=BinaryMetrics(.5)
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
                losses,norm,logits,target=update(model,optimizer,loss_fn,batch,device,cfg.values['train']['grad_clip_norm'])
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
                    print(f'{role} epoch={epoch}/100 step={step}/2757 loss={state["loss"]:.6f}',flush=True)
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
            status={'status':'running','role':role,'epoch':epoch,'epochs':100,'best_epoch':best_epoch,'best_val_f1':best_f1,
                'optimizer_steps':2757*epoch,'pair_views':55125*epoch,'source_draws':11025*epoch,
                'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_evaluated':False,'training_flavor':FLAVOR}
            status.update(pilot_seconds_debited=pilot_debit,worker_time_limit_seconds=remaining_seconds,
                cumulative_role_seconds_including_pilot=status['elapsed_seconds']+pilot_debit)
            _atomic_json(output/'status.json',status)
            _atomic_json(output/'progress.json',dict(status,committed_epochs=epoch,step=2757))
            print(f'{role} epoch={epoch} clean_val_f1={val["metrics"]["f1"]:.9f} best_epoch={best_epoch}',flush=True)
        validate_checkpoint(cp,cfg,identity,plan)
        guard.finish({'status':'completed_five_view_training'})
        status.update(status='completed',completed_utc=now())
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
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--role',choices=ROLES,required=True)
    args=parser.parse_args();print(canonical(train(args.config,args.role)))


if __name__=='__main__': main()
