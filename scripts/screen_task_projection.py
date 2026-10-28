"""Small algebraic obstruction checks, never model/data effectiveness tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from scripts.windows_shared_json import atomic_json as atomic

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/task_projection_screen_20261027.yaml'


def load():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (v['schema_version']!=1 or v['kind']!='algebraic_task_projection_screen_only'
            or v['device']!='cpu' or v['dtype']!='float64'
            or v['real_data_or_checkpoint'] is not False or v['optimizer_updates']!=0):
        raise ValueError('Only the explicit CPU algebraic screen is admitted')
    return v


def null_projection(jacobian,residual,*,rtol):
    if (jacobian.device.type!='cpu' or residual.device.type!='cpu'
            or jacobian.dtype!=torch.float64 or residual.dtype!=torch.float64
            or jacobian.ndim!=2 or residual.ndim!=1 or jacobian.shape[1]!=len(residual)
            or min(jacobian.shape)==0 or not torch.isfinite(jacobian).all()
            or not torch.isfinite(residual).all() or type(rtol) not in (int,float)
            or not 0<rtol<1):
        raise ValueError('Finite, matching CPU float64 J/r and explicit pinv tolerance required')
    # Standard orthogonal projection, used only to audit a proposed guarantee.
    return residual-torch.linalg.pinv(jacobian,rtol=rtol)@(jacobian@residual)


def hard_counts(logits,truth):
    p=logits>=0;t=truth.bool()
    return [int((~t&~p).sum()),int((~t&p).sum()),int((t&~p).sum()),int((t&p).sum())]


def witnesses(v):
    t=lambda x:torch.tensor(x,dtype=torch.float64,device='cpu')
    project=lambda j,r:null_projection(j,r,rtol=v['pinv_rtol'])
    close=lambda x:bool(torch.allclose(x,torch.zeros_like(x),atol=v['assert_atol'],rtol=0))
    c=v['full_vector'];j,r=t(c['jacobian']),t(c['residual']);p=project(j,r)
    full=dict(projected_residual=p.tolist(),nonzero_feature_residual=not close(p),
        all_linear_logits_unchanged=close(j@p))
    c=v['mean_only'];j,r=t(c['jacobian']),t(c['residual']);p=project(j,r)
    logits,truth=t(c['base_logits']),t(c['truth'])
    mean=dict(mean_linear_logit_unchanged=close(j@p),
        individual_linear_logit_changes=p.tolist(),
        before_counts=hard_counts(logits,truth),after_counts=hard_counts(logits+p,truth))
    c=v['positive_mask'];logits,truth,r=t(c['base_logits']),t(c['truth']),t(c['requested_residual'])
    identity=torch.eye(len(logits),dtype=torch.float64)
    predicted_positive=logits>=0;true_positive=predicted_positive&truth.bool()
    protected=project(identity[predicted_positive],r)
    privileged=project(identity[true_positive],r)
    mask=dict(predicted_positive_mask=predicted_positive.tolist(),truth_dependent_TP_mask=true_positive.tolist(),
        before_counts=hard_counts(logits,truth),
        predicted_positive_protected_counts=hard_counts(logits+protected,truth),
        GT_TP_protected_counts=hard_counts(logits+privileged,truth),
        GT_mask_is_deployment_input=False)
    c=v['teacher_shift'];clean,observed,r=t(c['clean_jacobian']),t(c['observed_jacobian']),t(c['residual']);p=project(clean,r)
    shifted=dict(clean_linear_change_zero=close(clean@p),observed_linear_change=(observed@p).tolist())
    c=v['finite_step'];x,r=t(c['base_feature']),t(c['residual'])
    if c['function']!='offset_plus_x0_plus_x1_squared':raise ValueError('Unknown explicit scalar witness')
    f=lambda z:c['offset']+z[0]+z[1].square()
    j=torch.stack((torch.ones_like(x[0]),2*x[1])).reshape(1,-1);p=project(j,r)
    finite=dict(first_order_change_zero=close(j@p),before_logit=float(f(x)),after_logit=float(f(x+p)),
        finite_step_changed_hard_prediction=bool((f(x)>=0)!=(f(x+p)>=0)))
    c=v['dense_size_accounting'];gram=c['output_rows']**2*c['bytes_per_element'];jac=c['output_rows']*c['stage2_feature_columns']*c['bytes_per_element']
    memory=dict(explicit_dense_gram_bytes=gram,explicit_dense_jacobian_bytes=jac,
        dense_gram_exceeds_specified_device=gram>c['device_capacity_bytes'],
        actual_measured_GPU_memory=False,does_not_rule_out_matrix_free_or_subset_methods=True)
    if not (full['nonzero_feature_residual'] and full['all_linear_logits_unchanged']
            and mean['mean_linear_logit_unchanged'] and mean['before_counts']==[1,0,0,1]
            and mean['after_counts']==[0,1,1,0] and shifted['clean_linear_change_zero']
            and not close(t(shifted['observed_linear_change'])) and finite['first_order_change_zero']
            and finite['finite_step_changed_hard_prediction']
            and mask['predicted_positive_protected_counts'][1]==mask['before_counts'][1]
            and mask['GT_TP_protected_counts'][1]==0):
        raise AssertionError('Frozen obstruction witness did not reproduce')
    return dict(full_vector=full,mean_only=mean,positive_mask=mask,teacher_shift=shifted,finite_step=finite,dense_size=memory)


def main():
    started=time.perf_counter();v=load();out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Preserve completed algebraic screen')
    result=dict(status='algebraic_counterexamples_reproduced',witnesses=witnesses(v),
        research_result=False,method_effectiveness=False,novelty_verified=False,
        new_training_admitted=False,real_data_or_checkpoint=False,optimizer_updates=0,device='cpu',dtype='float64',
        elapsed_seconds=time.perf_counter()-started,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (CONFIG,Path(__file__).resolve(),ROOT/v['scope'])})
    out.mkdir();atomic(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('status','research_result','method_effectiveness','elapsed_seconds')}))

if __name__=='__main__':main()
