"""Experimental reference-anchored GroupNorm replay for a frozen decoder.

Isolated mechanism prototype. It never changes the original normalization,
parameters, buffers, model files, device, or training code.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Frame:
    x: Tensor
    y: Tensor
    scale: Tensor
    weight: Tensor | None
    bias: Tensor | None
    eps: float
    groups: int


class ReferenceGroupNormReplay:
    """Capture one eval reference, then replay its affine normalization.

    All decoder parameters must be frozen. An action/input residual can still
    require gradients. Each selected GroupNorm must execute exactly once per
    context; cached tensors are detached copies, safe from later inplace ReLU.
    """
    def __init__(self, model: nn.Module, names: tuple[str, ...] | None = None):
        modules = {name: m for name, m in model.named_modules() if isinstance(m, nn.GroupNorm)}
        if names is not None:
            if not names or len(set(names)) != len(names) or not set(names) <= set(modules):
                raise ValueError('Require unique, existing GroupNorm module names')
            modules = {name: modules[name] for name in names}
        if not modules: raise ValueError('No GroupNorm layers to capture')
        self.model, self.modules = model, modules
        self.frames: dict[str, Frame] = {}
        self.active = False
        self.ready = False

    def _check_model(self):
        if self.model.training or any(m.training for m in self.model.modules()):
            raise ValueError('Reference replay requires an eval-only frozen decoder')
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Freeze decoder parameters; only action/input gradients are admitted')

    @contextmanager
    def _context(self, mode):
        self._check_model()
        if self.active: raise RuntimeError('Nested capture/replay is not allowed')
        if mode == 'replay' and not self.ready: raise RuntimeError('Complete reference capture first')
        if mode == 'capture': self.frames.clear(); self.ready = False
        self.active = True; seen = set(); handles = []

        def hook(name):
            def apply(module, args, output):
                if name in seen: raise RuntimeError('Repeated GroupNorm call invalidates the one-forward cache')
                seen.add(name)
                x = args[0]
                if not isinstance(x, Tensor) or x.ndim != 4 or not torch.isfinite(x).all():
                    raise ValueError('Require finite NCHW activations')
                if mode == 'capture':
                    if not torch.isfinite(output).all(): raise ValueError('Nonfinite reference output')
                    work = x.detach().double() if x.dtype == torch.float64 else x.detach().float()
                    n, c = x.shape[:2]
                    inv_std = torch.rsqrt(work.reshape(n, module.num_groups, -1).var(dim=2, unbiased=False) + module.eps)
                    scale = inv_std.repeat_interleave(c // module.num_groups, dim=1).reshape(n, c, 1, 1)
                    if module.weight is not None: scale = scale * module.weight.detach().to(work.dtype).reshape(1, c, 1, 1)
                    self.frames[name] = Frame(x.detach().clone(), output.detach().clone(), scale,
                        module.weight.detach().clone() if module.weight is not None else None,
                        module.bias.detach().clone() if module.bias is not None else None,
                        module.eps, module.num_groups)
                    return output
                frame = self.frames[name]
                if x.shape != frame.x.shape or x.dtype != frame.x.dtype or x.device != frame.x.device:
                    raise ValueError('Reference shape/dtype/device differs; never broadcast another sample cache')
                if module.eps != frame.eps or module.num_groups != frame.groups:
                    raise ValueError('Normalization definition changed after capture')
                for current, saved in ((module.weight, frame.weight), (module.bias, frame.bias)):
                    if (current is None) != (saved is None) or (current is not None and not torch.equal(current.detach(), saved)):
                        raise ValueError('Frozen affine parameters changed after reference capture')
                dtype = frame.scale.dtype
                # Anchoring to the actual reference output makes unchanged
                # positions bit-exact, without reconstructing y0 by reductions.
                result = frame.y.to(dtype) + (x.to(dtype) - frame.x.to(dtype)) * frame.scale
                if not torch.isfinite(result).all(): raise ValueError('Nonfinite replay result')
                return result.to(output.dtype)
            return apply

        try:
            for name, module in self.modules.items(): handles.append(module.register_forward_hook(hook(name)))
            yield self
            if seen != set(self.modules): raise RuntimeError('Incomplete GroupNorm execution; cache cannot certify one forward')
            if mode == 'capture': self.ready = True
        except BaseException:
            if mode == 'capture': self.frames.clear(); self.ready = False
            raise
        finally:
            for handle in handles: handle.remove()
            self.active = False

    def capture(self): return self._context('capture')
    def replay(self): return self._context('replay')


def fpn_convolution_support(masks: list[Tensor], output_size: tuple[int, int]) -> Tensor:
    """Conservative dependency support for the unchanged original FPN graph.

    This support assumes replay at all six GroupNorms. Ordinary GroupNorm adds
    global dependencies and is intentionally not represented by this mask.
    """
    if len(masks) != 4 or any(m.dtype != torch.bool or m.ndim != 4 or m.shape[1] != 1 for m in masks):
        raise ValueError('Require four N1HW boolean feature-support masks')
    if any(m.shape[0] != masks[0].shape[0] or m.device != masks[0].device for m in masks):
        raise ValueError('Support sample/device axes differ')
    def expand(x): return F.max_pool2d(x.float(), 3, stride=1, padding=1) > 0
    def resize(x, size): return F.interpolate(x.float(), size=size, mode='nearest') > 0
    support = expand(masks[3])
    for level in (2, 1, 0): support = expand(masks[level] | resize(support, masks[level].shape[-2:]))
    support = expand(resize(support, tuple((s + 1) // 2 for s in output_size)))
    return expand(resize(support, output_size))
