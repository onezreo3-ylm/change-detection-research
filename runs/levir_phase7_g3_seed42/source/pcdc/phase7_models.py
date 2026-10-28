"""Frozen Phase 7 descriptor and matched selector network."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .phase5_models import Stage2ResidualAdapter
from .robust_baselines import tensor_state_sha256


class Phase7Selector(nn.Module):
    """The exact 1024->128->64->2 MLP shared by C1/G1/G2/G3."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1024, 128, bias=True)
        self.norm1 = nn.LayerNorm(128)
        self.linear2 = nn.Linear(128, 64, bias=True)
        self.norm2 = nn.LayerNorm(64)
        self.linear3 = nn.Linear(64, 2, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.linear1, self.linear2, self.linear3):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (self.norm1, self.norm2):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != 1024:
            raise ValueError("Phase 7 selector expects [N,1024] descriptors")
        hidden = torch.nn.functional.gelu(self.norm1(self.linear1(descriptor)))
        hidden = torch.nn.functional.gelu(self.norm2(self.linear2(hidden)))
        return self.linear3(hidden)


def build_phase7_selector(seed: int = 42) -> Phase7Selector:
    """Initialize without consuming the caller's CPU/CUDA RNG streams."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        model = Phase7Selector()
    if sum(parameter.numel() for parameter in model.parameters()) != 139970:
        raise RuntimeError("Phase 7 selector parameter count changed")
    return model


def selector_initial_sha256(seed: int = 42) -> str:
    return tensor_state_sha256(build_phase7_selector(seed).state_dict())


def _mean_std(tensor: Tensor) -> tuple[Tensor, Tensor]:
    if tensor.ndim != 4 or tensor.shape[1] != 128:
        raise ValueError("Phase 7 descriptor tensors must be [N,128,H,W]")
    flat = tensor.float().flatten(2)
    return flat.mean(dim=2), flat.std(dim=2, unbiased=False)


def descriptor_from_stage2(stage2_a: Tensor, stage2_b: Tensor,
                           residual_a: Tensor, residual_b: Tensor) -> Tensor:
    """Return the approved date-symmetric 1,024-dimensional descriptor."""
    tensors = (
        (stage2_a.float() + stage2_b.float()) * 0.5,
        (stage2_a.float() - stage2_b.float()).abs(),
        (residual_a.float() + residual_b.float()) * 0.5,
        (residual_a.float() - residual_b.float()).abs(),
    )
    pieces: list[Tensor] = []
    for tensor in tensors:
        mean, std = _mean_std(tensor)
        pieces.extend((mean, std))
    descriptor = torch.cat(pieces, dim=1)
    if descriptor.shape[1] != 1024 or not torch.isfinite(descriptor).all():
        raise ValueError("Invalid Phase 7 descriptor")
    return descriptor


def phase7_descriptor(features_a: Iterable[Tensor], features_b: Iterable[Tensor],
                      adapter: Stage2ResidualAdapter) -> Tensor:
    a, b = list(features_a), list(features_b)
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Phase 7 requires four encoder stages")
    residual_a = adapter(a[1])
    residual_b = adapter(b[1])
    descriptor = descriptor_from_stage2(a[1], b[1], residual_a, residual_b)
    swapped = descriptor_from_stage2(b[1], a[1], residual_b, residual_a)
    if not torch.equal(descriptor, swapped):
        raise ValueError("Phase 7 descriptor is not bitwise date-swap invariant")
    return descriptor


def normalize_descriptor(descriptor: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    if mean.shape != (1024,) or std.shape != (1024,):
        raise ValueError("Descriptor normalization must contain 1,024 values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std < 1e-6):
        raise ValueError("Descriptor normalization is non-finite or below the frozen floor")
    return (descriptor.float() - mean) / std
