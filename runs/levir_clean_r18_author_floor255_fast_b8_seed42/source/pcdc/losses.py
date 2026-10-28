"""Binary change supervision; no perturbation consistency is implemented."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChangeDetectionLoss(nn.Module):
    def __init__(self, bce_weight: float, dice_weight: float, dice_smooth: float):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth

    def forward(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Loss expects matching [N,1,H,W] logits and masks")
        # Accumulate Dice in float32 even if convolution uses CUDA AMP.
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probabilities = logits.sigmoid().flatten(1)
        labels = target.flatten(1)
        intersection = (probabilities * labels).sum(dim=1)
        dice = (1 - (2 * intersection + self.dice_smooth) /
                (probabilities.sum(dim=1) + labels.sum(dim=1) + self.dice_smooth)).mean()
        total = self.bce_weight * bce + self.dice_weight * dice
        return {"total": total, "bce": bce, "dice": dice}

