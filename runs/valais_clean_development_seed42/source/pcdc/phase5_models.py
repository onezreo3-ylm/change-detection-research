"""The one user-approved Phase 5 correction action and exact task loss."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import SiameseChangeDetector


class Stage2ResidualAdapter(nn.Module):
    """Shared 128->32->32->128 residual branch with an exact zero start."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != 128:
            raise ValueError("Stage-2 adapter expects [N,128,H,W] features")
        hidden = F.relu(self.norm1(self.conv1(features)), inplace=False)
        hidden = F.relu(self.norm2(self.conv2(hidden)), inplace=False)
        return self.conv3(hidden)


def freeze_detector(detector: SiameseChangeDetector) -> None:
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)


class AdapterCorrectedDetector(nn.Module):
    """Frozen detector plus a shared stage-2 adapter for both dates.

    The encoder is evaluated without an activation graph.  The frozen decoder
    remains in the graph so gradients can flow from task loss to the adapter.
    Calling ``train(True)`` never puts detector BatchNorm layers in training mode.
    """

    def __init__(self, detector: SiameseChangeDetector,
                 adapter: Stage2ResidualAdapter | None = None) -> None:
        super().__init__()
        self.detector = detector
        self.adapter = adapter if adapter is not None else Stage2ResidualAdapter()
        freeze_detector(self.detector)

    def train(self, mode: bool = True) -> "AdapterCorrectedDetector":
        super().train(mode)
        self.detector.eval()
        self.adapter.train(mode)
        return self

    def _encode(self, image_a: Tensor, image_b: Tensor):
        self.detector.eval()
        with torch.no_grad():
            return self.detector.encode_pair(image_a, image_b)

    def _decode(self, features_a: list[Tensor], features_b: list[Tensor],
                output_size: tuple[int, int], *, apply_adapter: bool) -> Tensor:
        if apply_adapter:
            corrected_a, corrected_b = list(features_a), list(features_b)
            corrected_a[1] = corrected_a[1] + self.adapter(corrected_a[1])
            corrected_b[1] = corrected_b[1] + self.adapter(corrected_b[1])
        else:
            corrected_a, corrected_b = features_a, features_b
        differences = self.detector.difference_features(corrected_a, corrected_b)
        return self.detector.decoder(differences, output_size)

    def forward(self, image_a: Tensor, image_b: Tensor, *,
                apply_adapter: bool = True, return_features: bool = False) -> dict[str, Any]:
        features_a, features_b = self._encode(image_a, image_b)
        logits = self._decode(features_a, features_b, tuple(image_a.shape[-2:]),
                              apply_adapter=apply_adapter)
        output: dict[str, Any] = {"logits": logits}
        if return_features:
            output.update(features_a=features_a, features_b=features_b)
        return output

    def action_logits(self, image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]:
        """Actually decode both a0 and a1 from one frozen encoder execution."""
        features_a, features_b = self._encode(image_a, image_b)
        output_size = tuple(image_a.shape[-2:])
        a0 = self._decode(features_a, features_b, output_size, apply_adapter=False)
        a1 = self._decode(features_a, features_b, output_size, apply_adapter=True)
        return a0, a1


def per_sample_change_loss(logits: Tensor, target: Tensor, *, bce_weight: float,
                           dice_weight: float, dice_smooth: float) -> dict[str, Tensor]:
    """Exact per-sample form of the existing BCE plus soft-Dice objective."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Per-sample loss expects matching [N,1,H,W] tensors")
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
    probabilities, labels = logits.sigmoid().flatten(1), target.flatten(1)
    intersection = (probabilities * labels).sum(1)
    dice = 1 - (2 * intersection + dice_smooth) / (
        probabilities.sum(1) + labels.sum(1) + dice_smooth)
    total = bce_weight * bce + dice_weight * dice
    return {"total": total, "bce": bce, "dice": dice}
