"""Diagnostics for disjoint local actions; no learned policy or safety claim."""
from __future__ import annotations

import torch
from torch import Tensor


def compose_disjoint_outputs(base: Tensor, singles: list[Tensor], supports: list[Tensor]) -> Tensor:
    if not singles or len(singles) != len(supports) or base.ndim != 4 or base.shape[1] != 1:
        raise ValueError('Require nonempty matching action outputs and N1HW supports')
    if not torch.isfinite(base).all(): raise ValueError('Nonfinite baseline')
    union = torch.zeros_like(base, dtype=torch.bool); result = base.clone()
    for value, support in zip(singles, supports):
        if value.shape != base.shape or value.dtype != base.dtype or value.device != base.device or not torch.isfinite(value).all():
            raise ValueError('Action output identity differs from baseline')
        if support.shape != base.shape or support.dtype != torch.bool or support.device != base.device:
            raise ValueError('Boolean support identity differs from output')
        if (union & support).any(): raise ValueError('Intersecting influence supports cannot certify independent actions')
        union |= support; result = torch.where(support, value, result)
    return result


def confusion_vector(logits: Tensor, target: Tensor) -> Tensor:
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1 or logits.device != target.device:
        raise ValueError('Require matching N1HW logits/labels')
    if not torch.isfinite(logits).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError('Nonfinite logits or nonbinary target')
    pred = logits.float().sigmoid() >= .5
    index = target.to(torch.int64).reshape(-1) * 2 + pred.to(torch.int64).reshape(-1)
    return torch.stack([(index == i).sum(dtype=torch.int64) for i in range(4)])


def compare_joint(base: Tensor, singles: list[Tensor], joint: Tensor,
                  supports: list[Tensor], target: Tensor) -> dict:
    pasted = compose_disjoint_outputs(base, singles, supports)
    if joint.shape != base.shape or joint.dtype != base.dtype or joint.device != base.device or not torch.isfinite(joint).all():
        raise ValueError('Joint output identity differs')
    base_counts = confusion_vector(base, target)
    single_counts = [confusion_vector(x, target) for x in singles]
    actual = confusion_vector(joint, target)
    expected = base_counts.clone()
    for row in single_counts: expected += row - base_counts
    union = torch.zeros_like(base, dtype=torch.bool)
    for support in supports: union |= support
    if union.all(): raise ValueError('Need nonempty outside support for this diagnostic')
    return dict(joint_equals_pasted_individuals=bool(torch.equal(joint, pasted)),
        joint_minus_pasted_max_abs=float((joint - pasted).abs().max()),
        joint_minus_pasted_changed_positions=int(torch.count_nonzero(joint - pasted)),
        count_increment_additivity=bool(torch.equal(actual, expected)),
        base_counts=base_counts.tolist(), single_counts=[x.tolist() for x in single_counts],
        joint_counts=actual.tolist(), sum_of_individual_count_increments=expected.tolist(),
        outside_joint_changed_positions=int(torch.count_nonzero((joint - base)[~union])),
        individual_locality=[int(torch.count_nonzero((x - base)[~s])) == 0 for x, s in zip(singles, supports)],
        hard_prediction_changes=[int(torch.count_nonzero((x.float().sigmoid() >= .5) != (base.float().sigmoid() >= .5))) for x in singles])
