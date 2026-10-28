"""Explicit BF16 engineering candidate, with FP32 parameters/loss/optimizer."""
import torch
from scripts import five_view_baselines as f


def require_bf16(device):
    if device.type!='cuda' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Explicit CUDA BF16 support required; no precision/device substitution')


def update(model,optimizer,loss_fn,batch,device,clip):
    require_bf16(device)
    a,b,target=f.transfer(batch,device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
        logits=model(a,b)['logits']
    if logits.dtype!=torch.bfloat16 or not torch.isfinite(logits).all().item():
        raise FloatingPointError('BF16 candidate output is wrong dtype or nonfinite')
    with torch.autocast(device_type='cuda',enabled=False):
        losses=f.weighted_loss(logits.float(),target.float(),loss_fn)
    if any(x.dtype!=torch.float32 or not torch.isfinite(x).item() for x in losses.values()):
        raise FloatingPointError('BF16 candidate requires finite FP32 loss accumulation')
    losses['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    return {k:float(x.detach()) for k,x in losses.items()},float(norm),logits.detach(),target
