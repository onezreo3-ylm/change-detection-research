"""Preserve a pre-forward API failure and spend only the unused single-step budget."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.data import file_sha256
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import tensor_state_sha256
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts import valais_feasibility_screen as s
from scripts import evaluate_coupling_p0 as p0


def cuda_index(device):
    device=torch.device(device)
    if device.type!='cuda':
        raise ValueError('Pilot requires the original CUDA device')
    return device.index if device.index is not None else torch.cuda.current_device()


def check_unused_budget(failed,initial_sha):
    expected="ValueError: Expected a torch.device with a specified index or an integer, but got:cuda"
    if (failed.get('status')!='incomplete' or failed.get('error')!=expected
            or failed.get('optimizer_steps')!=0 or failed.get('pair_views')!=0
            or failed.get('initial_model_tensor_sha256')!=initial_sha
            or failed.get('test_accessed') is not False or failed.get('new_val_inference') is not False
            or not 0<=failed.get('elapsed_seconds',float('inf'))<600):
        raise ValueError('Only the exact pre-forward API error has an unused engineering step')
    return 600-failed['elapsed_seconds']


def load(path):
    source=Path(path).resolve()
    v=yaml.load(source.read_bytes(),Loader=UniqueKeyLoader)
    if (v['schema_version']!=1 or v['output_subdirectory']!='r5_engineering_pilot_cuda_index_fix'
            or v['optimizer_steps_remaining']!=1 or v['maximum_total_pilot_seconds']!=600):
        raise ValueError('Changed remaining pilot budget')
    screen=(source.parent/v['screen_config']).resolve()
    _,settings,paths,models=s.load(screen)
    output=paths['output_dir']/v['output_subdirectory']
    protocol=(source.parent/v['protocol_document']).resolve()
    return source,screen,settings,paths,models,output,protocol


def prepare(path):
    source,screen,v,paths,models,output,protocol=load(path)
    s.verify(paths)
    failed=s.read(paths['output_dir']/'r5_engineering_pilot/report.json')
    remaining=check_unused_budget(failed,v['initial_model_tensor_sha256'])
    if s.read(paths['output_dir']/'bn_diagnostic/report.json')['status']!='completed_train_only_bn_diagnostic':
        raise ValueError('Complete BN diagnostic required')
    inputs=[source,screen,protocol,Path(__file__),s.ROOT/'tests/test_r5_engineering_pilot.py',
            paths['output_dir']/'authorization.json', paths['output_dir']/'r5_engineering_pilot/report.json',
            paths['output_dir']/'r5_engineering_pilot/guard_observations.json',
            paths['output_dir']/'bn_diagnostic/report.json', paths['output_dir']/'bn_diagnostic/records.jsonl',
            paths['output_dir']/'bn_diagnostic/guard_observations.json']
    output.mkdir(exist_ok=False)
    write_json(output/'authorization.json',{'status':'same_authorized_unused_engineering_step_api_correction',
        'optimizer_steps_remaining':1,'seconds_remaining':remaining,
        'original_forward_or_optimizer_repeated':False,
        'bound_sha256':{str(p.resolve()):file_sha256(p) for p in inputs}})
    return {'status':'prepared_cuda_query_correction','remaining_steps':1,'remaining_seconds':remaining}


def verify(paths,output):
    manifest=s.verify(paths)
    auth=s.read(output/'authorization.json')
    for name,sha in auth['bound_sha256'].items(): p0.checked_hash(name,sha)
    return manifest,auth


def run(path):
    _,_,v,paths,models,output,_=load(path)
    manifest,auth=verify(paths,output)
    if (output/'report.json').exists(): raise FileExistsError(output/'report.json')
    failed=s.read(paths['output_dir']/'r5_engineering_pilot/report.json')
    remaining=check_unused_budget(failed,v['initial_model_tensor_sha256'])
    if remaining!=auth['seconds_remaining']: raise ValueError('Remaining budget changed')
    guard=p0.Guard(dict(manifest['settings'],max_seconds=remaining))
    report={'status':'incomplete','engineering_only':True,'research_score_produced':False,
        'test_accessed':False,'new_val_inference':False,'optimizer_steps':0,'automatic_retry':False,
        'source_batch_size':4,'views_per_source':5,'amp':False,'batchnorm':'single_joint_forward',
        'prior_pre_forward_failure':str(paths['output_dir']/'r5_engineering_pilot/report.json'),
        'prior_pre_forward_seconds':failed['elapsed_seconds'],'cumulative_budget_seconds':600}
    try:
        guard.check()
        cfg=load_config(models['b0e']['base_config'])
        device=setup_runtime(cfg)
        index=cuda_index(device)
        free,total=torch.cuda.mem_get_info(index)
        model=move_model(build_model(cfg,initialize_encoder=False),device,'channels_last').float().train()
        before=tensor_state_sha256(model.state_dict())
        if before!=v['initial_model_tensor_sha256']: raise ValueError('Changed original initialization')
        report['initial_model_tensor_sha256']=before
        samples={row['name']:row for row in manifest['probe']}
        av,bv,targets=[[] for _ in range(5)],[[] for _ in range(5)],[]
        for name in manifest['pilot_sources']:
            sample=samples[name]
            a,b,target=p0.read_sample(sample,cfg)
            c=next(c for c in sample['conditions'] if {'family':c['family'],'level':c['level']}==v['pilot_condition'])
            views,_=p0.render_views(a,b,c)
            pairs=[(a,b)]+[(views[x],views[y]) for x,y in p0.CELLS.values()]
            for j,(x,y) in enumerate(pairs):
                av[j].append(s.tensorize(x,cfg,device))
                bv[j].append(s.tensorize(y,cfg,device))
            targets.append(target)
        a,b=torch.cat([torch.cat(x) for x in av]),torch.cat([torch.cat(x) for x in bv])
        target=torch.cat(targets).to(device)
        opt=cfg.values['train']
        optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],betas=opt['adam_betas'],weight_decay=opt['weight_decay'])
        loss_fn=ChangeDetectionLoss(**cfg.values['loss'])
        report.update(environment=environment_record(cfg,device),cuda_device_index=index,
            free_cuda_bytes_before_initialization=free,total_cuda_bytes=total,
            sources=manifest['pilot_sources'],view_weights=v['pilot_view_weights'])
        torch.cuda.reset_peak_memory_stats(index)
        guard.check()
        started=time.monotonic()
        with torch.autocast(device_type='cuda',enabled=False):
            logits=model(a,b)['logits']
            loss=s.r5_loss(logits,target,loss_fn)
        guard.predictions+=20
        if logits.dtype!=torch.float32 or not torch.isfinite(loss): raise FloatingPointError('Invalid R5 FP32 loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clip_norm'],error_if_nonfinite=True)
        guard.check()
        optimizer.step()
        report['optimizer_steps']=1
        torch.cuda.synchronize(device)
        guard.check()
        if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite updated parameters')
        if any(int(m.num_batches_tracked)!=1 for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)):
            raise ValueError('R5 requires one joint BN update')
        if tensor_state_sha256(model.state_dict())==before: raise ValueError('No engineering update occurred')
        report.update(status='passed_one_step_r5_engineering_pilot',loss=float(loss.detach()),
            gradient_norm_before_clip=float(norm),single_step_seconds=time.monotonic()-started,
            all_bn_updates_exactly_one=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(index),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(index))
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='r5_single_batch_memory_infeasible',error=str(error),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        elapsed=time.monotonic()-guard.started
        report.update(elapsed_seconds=elapsed,cumulative_pilot_seconds=elapsed+failed['elapsed_seconds'],
            pair_views=guard.predictions,maximum_observed_temperature_c=max(guard.temperatures) if guard.temperatures else None)
        write_json(output/'report.json',report)
        write_json(output/'guard_observations.json',{'temperatures_c':guard.temperatures,'automatic_retry':False})
    return {k:report[k] for k in ('status','optimizer_steps','cumulative_pilot_seconds')}


def audit(path):
    _,_,_,paths,_,output,_=load(path)
    manifest,_=verify(paths,output)
    directory=paths['output_dir']/'bn_diagnostic'
    report=s.read(directory/'report.json')
    rows=[json.loads(line) for line in (directory/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(r,a,sample['id']) for r in s.ROLES for a in s.ARMS for sample in manifest['probe']}
    keys=[(r['model'],r['arm'],r['source_id']) for r in rows]
    if set(keys)!=expected or len(keys)!=576 or report['pair_views']!=3666 or report['records']!=576:
        raise ValueError('Incomplete/duplicate BN coverage')
    for role in s.ROLES:
        old=s.original_clean(paths,role)
        original={r['source_id']:r for r in rows if r['model']==role and r['arm']=='original'}
        if {sid:r['counts'] for sid,r in original.items()}!=old:
            raise ValueError('Original clean saved counts not exactly reproduced')
        for arm in s.ARMS:
            selected=[r for r in rows if r['model']==role and r['arm']==arm]
            arm_report=report['models'][role][arm]
            if s.report_counts(selected)!=arm_report['probe'] or not arm_report['only_bn_buffers_changed']:
                raise ValueError('BN result pooling/state check failed')
            if arm_report['calibration_steps']!=(0 if arm=='original' else 64):
                raise ValueError('Calibration budget changed')
            for r in selected:
                c,oldc,t=r['counts'],original[r['source_id']]['counts'],r['transitions']
                if (sum(c.values())!=65536 or c['tp']+c['fn']!=r['foreground_pixels']
                        or c['fn']-oldc['fn']!=t['tp_to_fn']-t['fn_to_tp']
                        or c['fp']-oldc['fp']!=t['tn_to_fp']-t['fp_to_tn']):
                    raise ValueError('Pixel/transition conservation failed')
    temps=s.read(directory/'guard_observations.json')['temperatures_c']
    if (report['status']!='completed_train_only_bn_diagnostic' or not temps or max(temps)>=85
            or report['elapsed_seconds']>=3600 or not all(report['models'][r]['original_restored'] for r in s.ROLES)):
        raise ValueError('Incomplete BN numerical/budget/restore checks')
    p=s.read(output/'report.json')
    if p['status'] not in ('passed_one_step_r5_engineering_pilot','r5_single_batch_memory_infeasible'):
        raise ValueError('No bounded pilot outcome')
    if p['cumulative_pilot_seconds']>=600 or p['optimizer_steps']>1 or p['new_val_inference'] or p['test_accessed']:
        raise ValueError('Pilot exceeded cumulative scope')
    if p['status']=='passed_one_step_r5_engineering_pilot' and (p['optimizer_steps']!=1
            or p['pair_views']!=20 or not p['all_bn_updates_exactly_one'] or not math.isfinite(p['loss'])
            or not math.isfinite(p['gradient_norm_before_clip']) or p['maximum_observed_temperature_c']>=85):
        raise ValueError('One finite FP32 update not confirmed')
    files=[directory/'report.json',directory/'records.jsonl',directory/'guard_observations.json',
        paths['output_dir']/'saved_analysis.json',output/'report.json',output/'guard_observations.json',
        output/'authorization.json', paths['output_dir']/'r5_engineering_pilot/report.json']
    result={'status':'passed_finite_screen_saved_audit','probe_records':576,'probe_predictions':594,
        'calibration_pair_views':3072,'calibration_forwards':384,'new_val_inference':False,'test_accessed':False,
        'old_checkpoints_and_bound_inputs_unchanged':True,'pilot_status':p['status'],
        'prior_pre_forward_failure_preserved':True,'total_actual_engineering_optimizer_steps':p['optimizer_steps'],
        'research_training_admitted':False,'result_sha256':{str(f):file_sha256(f) for f in files}}
    write_json(paths['output_dir']/'final_audit.json',result)
    return {k:result[k] for k in ('status','probe_records','pilot_status','total_actual_engineering_optimizer_steps')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](args.config)))


if __name__=='__main__': main()
