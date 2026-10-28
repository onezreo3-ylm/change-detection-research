"""CPU layout and fewer synchronizations, with the unchanged FP32 objective."""
from __future__ import annotations

import torch
from torch.utils.data import default_collate

from pcdc.metrics import BinaryMetrics
from scripts import five_view_baselines as f


def collate_view_major(rows):
    batch = default_collate(rows)
    for key in ('a', 'b'):
        x = batch[key]
        batch[key] = x.transpose(0, 1).reshape(-1, 3, 256, 256).contiguous(memory_format=torch.channels_last)
    return batch


def transfer(batch, device):
    a, b = batch['a'], batch['b']
    if a.shape != b.shape or a.shape != (5*len(batch['mask']), 3, 256, 256):
        raise ValueError('Incorrect prepacked five-view input')
    if not a.is_contiguous(memory_format=torch.channels_last) or not b.is_contiguous(memory_format=torch.channels_last):
        raise ValueError('Prepacked channel-last layout required')
    return a.to(device, non_blocking=True), b.to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)


def update(model, optimizer, loss_fn, batch, device, clip):
    a, b, target = transfer(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', enabled=False):
        logits = model(a, b)['logits']
        losses = f.weighted_loss(logits, target, loss_fn)
    values = torch.stack([losses[k] for k in ('total', 'bce', 'dice')])
    if logits.dtype != torch.float32 or not torch.isfinite(values).all().item():
        raise FloatingPointError('Non-FP32 or nonfinite five-view loss')
    losses['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    numbers = torch.cat([values.detach(), norm.detach().reshape(1)]).cpu().tolist()
    return dict(zip(('total', 'bce', 'dice'), numbers[:3])), numbers[3], logits.detach(), target


class DeviceMetrics:
    """Same confusion counts; keep int64 sums on the input device until compute."""
    def __init__(self, threshold=.5):
        if threshold != .5:
            raise ValueError('Frozen threshold .5 required')
        self.counts = None
        self.valid = None

    @torch.no_grad()
    def update(self, logits, target):
        if logits.shape != target.shape:
            raise ValueError('Metric shapes differ')
        predicted = logits.float().sigmoid() >= .5
        truth = target == 1
        tp = (predicted & truth).sum(dtype=torch.int64)
        pp, gt = predicted.sum(dtype=torch.int64), truth.sum(dtype=torch.int64)
        counts = torch.stack([logits.numel()-pp-gt+tp, pp-tp, gt-tp, tp])
        valid = torch.isfinite(logits).all() & ((target == 0) | (target == 1)).all()
        if self.counts is None:
            self.counts, self.valid = counts, valid
        else:
            self.counts += counts
            self.valid &= valid

    def compute(self):
        if self.counts is None:
            raise ValueError('No metric samples')
        if not self.valid.item():
            raise ValueError('Nonfinite logits or nonbinary target')
        result = BinaryMetrics(.5)
        result.counts = self.counts.cpu()
        return result.compute()
