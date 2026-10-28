"""Pixel-pooled binary metrics with explicit zero-denominator conventions."""

from __future__ import annotations

import torch
from torch import Tensor


class BinaryMetrics:
    def __init__(self, threshold: float = 0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between zero and one")
        self.threshold = threshold
        self.counts = torch.zeros(4, dtype=torch.int64)  # TN, FP, FN, TP

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor) -> None:
        if logits.shape != target.shape:
            raise ValueError("Metric logits and targets must have the same shape")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite metric logits")
        if not torch.all((target == 0) | (target == 1)):
            raise ValueError("Metric targets must be binary")
        predicted = logits.float().sigmoid() >= self.threshold
        indices = target.to(torch.int64).reshape(-1) * 2 + predicted.to(torch.int64).reshape(-1)
        # CUDA bincount is rejected by strict deterministic mode in torch 2.0.
        # Integer reductions give the identical pooled confusion counts on CUDA.
        counts = torch.stack([(indices == value).sum(dtype=torch.int64) for value in range(4)])
        self.counts += counts.cpu()

    def compute(self) -> dict[str, float | int]:
        tn, fp, fn, tp = self.counts.tolist()
        total = tn + fp + fn + tp
        if not total:
            raise ValueError("Cannot compute metrics for an empty split")

        def divide(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        return {"precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
                "f1": divide(2 * tp, 2 * tp + fp + fn), "iou": divide(tp, tp + fp + fn),
                "oa": (tp + tn) / total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "pixels": total}
