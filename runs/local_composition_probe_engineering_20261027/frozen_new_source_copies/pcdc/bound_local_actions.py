"""Sample-bound inference of fixed stage-2 actions on the original frozen model.

This wrapper owns encoding, residual construction and the normalization reference.
Callers cannot supply a second same-shaped feature batch to a captured session.
It is an experimental mechanism probe, not a learned policy or safety guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import secrets

import torch
from torch import Tensor

from .local_feature_replay import ReferenceGroupNormReplay, fpn_convolution_support
from .phase5_models import AdapterCorrectedDetector


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    """Hash actual values and their axes/dtype; never serialize image data to chat."""
    detached = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(json.dumps([list(detached.shape), str(detached.dtype)]).encode())
    h.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_tensor_sha256(model) -> str:
    return _json_sha({name: tensor_sha256(tensor) for name, tensor in model.state_dict().items()})


@dataclass(frozen=True)
class ObservedBatchIdentity:
    sample_ids: tuple[str, ...]
    source_pair_sha256: tuple[tuple[str, str], ...]
    instance_sha256: tuple[str, ...]
    actor_checkpoint_sha256: str
    adapter_checkpoint_sha256: str
    protocol_sha256: str

    def __post_init__(self):
        if (not isinstance(self.sample_ids, tuple) or not self.sample_ids
                or any(not isinstance(x, str) or not x.startswith('train/') for x in self.sample_ids)
                or len(set(self.sample_ids)) != len(self.sample_ids)):
            raise ValueError('Require unique ordered train sample identities')
        if (not isinstance(self.source_pair_sha256, tuple)
                or len(self.source_pair_sha256) != len(self.sample_ids)
                or any(not isinstance(x, tuple) or len(x) != 2 or not all(_sha(y) for y in x)
                       for x in self.source_pair_sha256)):
            raise ValueError('Require ordered original A/B source hashes for each sample')
        if (not isinstance(self.instance_sha256, tuple)
                or len(self.instance_sha256) != len(self.sample_ids)
                or not all(_sha(x) for x in self.instance_sha256)):
            raise ValueError('Require an explicit corruption or clean-instance hash per sample')
        if not all(_sha(x) for x in (self.actor_checkpoint_sha256,
                                    self.adapter_checkpoint_sha256, self.protocol_sha256)):
            raise ValueError('Require checkpoint and protocol SHA256 identities')


@dataclass(frozen=True)
class ReferenceBinding:
    session_id: str
    evidence_sha256: str


class BoundLocalActionSession:
    """One observed A/B batch, fixed adapter outputs, and one decoder reference.

    A binding is checked on every action/decode. The wrapper clones real inputs,
    encodes them itself, and only exposes cloned outputs. The outer loader must
    validate the claimed file/checkpoint hashes; this records additional hashes
    of the actual normalized input tensors and actual loaded model tensors.
    CPU execution requires explicit ``engineering_only=True``. Precision is
    fixed for the entire session; no automatic CPU/FP32 fallback is provided.
    """

    def __init__(self, model: AdapterCorrectedDetector, image_a: Tensor, image_b: Tensor,
                 identity: ObservedBatchIdentity, *, amp: bool, engineering_only: bool = False):
        if not isinstance(model, AdapterCorrectedDetector):
            raise TypeError('Require the unchanged original adapter/detector implementation')
        if (image_a.shape != image_b.shape or image_a.ndim != 4 or image_a.shape[1] != 3
                or image_a.shape[0] != len(identity.sample_ids) or min(image_a.shape[-2:]) < 32
                or image_a.dtype != image_b.dtype or image_a.device != image_b.device
                or not image_a.is_floating_point() or not torch.isfinite(image_a).all()
                or not torch.isfinite(image_b).all()):
            raise ValueError('Actual A/B inputs must match the ordered batch identity')
        if image_a.device.type != 'cuda' and (not engineering_only or amp):
            raise ValueError('CPU is allowed only as explicit engineering without AMP')
        if type(amp) is not bool or type(engineering_only) is not bool:
            raise TypeError('Explicit Boolean precision and engineering flags are required')
        if any(t.device != image_a.device for t in model.state_dict().values()):
            raise ValueError('Model and actual inputs must use the same explicit device')
        self.model, self.identity, self.amp = model, identity, amp
        self.device_type = image_a.device.type
        self.output_size = tuple(image_a.shape[-2:])
        self._closed = False
        self._actions: dict[str, tuple[Tensor, Tensor]] = {}
        self._calls = dict(encoder=0, adapter=0, decoder=0)
        self._assert_frozen()
        self._model_sha = model_tensor_sha256(model)
        self._model_signature = self._signature()
        own_a, own_b = image_a.detach().clone(), image_b.detach().clone()
        evidence = dict(identity=asdict(identity), actual_model_tensor_sha256=self._model_sha,
                        normalized_a_sha256=[tensor_sha256(t) for t in own_a],
                        normalized_b_sha256=[tensor_sha256(t) for t in own_b],
                        amp=amp, actual_device=str(image_a.device), input_dtype=str(image_a.dtype),
                        engineering_only=engineering_only)
        self.binding = ReferenceBinding(secrets.token_hex(16), _json_sha(evidence))
        self.evidence = evidence
        self._replay = ReferenceGroupNormReplay(model.detector.decoder)
        if len(self._replay.modules) != 6:
            raise ValueError('The original FPN must have exactly six captured GroupNorms')
        with torch.no_grad(), self._precision():
            a, b = model._encode(own_a, own_b)
            self._calls['encoder'] += 1
            self._base_features = [t.detach().clone() for t in model.detector.difference_features(a, b)]
            ra, rb = model.adapter(a[1]), model.adapter(b[1])
            self._calls['adapter'] += 2
            self._corrected_stage2 = torch.abs((a[1] + ra) - (b[1] + rb)).detach().clone()
            if any(not torch.isfinite(t).all() for t in self._base_features + [self._corrected_stage2]):
                raise ValueError('Nonfinite actual encoded features/residuals')
            with self._replay.capture():
                self._base_logits = self._forward(self._base_features).detach().clone()
        self._cached_signature = self._cache_signature()
        self._captured_values_sha = self._captured_value_hash()

    def _precision(self):
        return torch.autocast(device_type=self.device_type, enabled=self.amp,
                              dtype=torch.float16 if self.device_type == 'cuda' else torch.bfloat16)

    def _assert_frozen(self):
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Every original model module must stay frozen and in eval mode')

    def _signature(self):
        return tuple((name, id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
                     for name, t in self.model.named_parameters()) + tuple(
            (name, id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
            for name, t in self.model.named_buffers()) + tuple(
            (name, type(m).__qualname__, repr(m.extra_repr())) for name, m in self.model.named_modules())

    def _cache_signature(self):
        tensors = self._base_features + [self._corrected_stage2, self._base_logits]
        for frame in self._replay.frames.values():
            tensors.extend(t for t in (frame.x, frame.y, frame.scale, frame.weight, frame.bias) if t is not None)
        for mask, support in self._actions.values():
            tensors.extend((mask, support))
        return tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device) for t in tensors)

    def _captured_value_hash(self):
        tensors = self._base_features + [self._corrected_stage2, self._base_logits]
        for frame in self._replay.frames.values():
            tensors.extend(t for t in (frame.x, frame.y, frame.scale, frame.weight, frame.bias) if t is not None)
        return _json_sha([tensor_sha256(t) for t in tensors])

    def _check(self, binding: ReferenceBinding):
        if self._closed:
            raise RuntimeError('A closed sample/reference session cannot be reused')
        if binding != self.binding:
            raise ValueError('Wrong ordered sample/model/reference session binding')
        if (_json_sha(self.evidence) != self.binding.evidence_sha256
                or asdict(self.identity) != self.evidence['identity']):
            raise ValueError('Ordered source/instance/model evidence changed after capture')
        self._assert_frozen()
        if self._signature() != self._model_signature:
            raise ValueError('Frozen model state or definition changed after reference capture')
        if self._cache_signature() != self._cached_signature:
            raise ValueError('Owned features/reference/action tensors changed after capture')

    def _forward(self, features):
        self._calls['decoder'] += 1
        logits = self.model.detector.decoder(features, self.output_size)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite actual decoder output')
        return logits

    def register(self, name: str, stage2_mask: Tensor, *, binding: ReferenceBinding) -> None:
        self._check(binding)
        shape = (self._base_features[1].shape[0], 1, *self._base_features[1].shape[-2:])
        if not isinstance(name, str) or not name or name in self._actions:
            raise ValueError('Require a new nonempty action name')
        if (stage2_mask.dtype != torch.bool or tuple(stage2_mask.shape) != shape
                or stage2_mask.device != self._base_features[1].device or not stage2_mask.any()):
            raise ValueError('Require a nonempty Boolean stage-2 support for this actual batch')
        mask = stage2_mask.detach().clone()
        masks = [torch.zeros((x.shape[0], 1, *x.shape[-2:]), dtype=torch.bool, device=x.device)
                 for x in self._base_features]
        masks[1] = mask
        support = fpn_convolution_support(masks, self.output_size)
        self._actions[name] = (mask, support)
        self._cached_signature = self._cache_signature()

    def baseline(self, *, binding: ReferenceBinding) -> Tensor:
        self._check(binding)
        return self._base_logits.clone()

    def support(self, name: str, *, binding: ReferenceBinding) -> Tensor:
        self._check(binding)
        return self._actions[name][1].clone()

    def decode(self, names: tuple[str, ...], *, binding: ReferenceBinding, arm: str) -> Tensor:
        self._check(binding)
        if not isinstance(names, tuple) or len(set(names)) != len(names) or not set(names) <= self._actions.keys():
            raise ValueError('Require unique registered action names from this session')
        if arm not in ('reference_replay', 'ordinary_GN', 'ordinary_GN_output_mask'):
            raise ValueError('Unknown explicitly named comparison arm')
        mask = torch.zeros_like(self._base_features[1][:, :1], dtype=torch.bool)
        union = torch.zeros_like(self._base_logits, dtype=torch.bool)
        for name in names:
            action_mask, support = self._actions[name]
            if (union & support).any():
                raise ValueError('Overlapping propagated supports are outside the composition protocol')
            mask |= action_mask
            union |= support
        features = list(self._base_features)
        features[1] = torch.where(mask, self._corrected_stage2, features[1])
        with torch.no_grad(), self._precision():
            if arm == 'reference_replay':
                with self._replay.replay():
                    logits = self._forward(features)
            else:
                logits = self._forward(features)
        if arm == 'ordinary_GN_output_mask':
            logits = torch.where(union, logits, self._base_logits)
        self._check(binding)
        return logits.detach().clone()

    def close(self) -> dict:
        if self._closed:
            raise RuntimeError('Session already closed')
        try:
            self._check(self.binding)
            # Full value hash also catches .data writes that bypass version counters.
            if model_tensor_sha256(self.model) != self._model_sha:
                raise ValueError('Actual frozen model values changed during the session')
            if self._captured_value_hash() != self._captured_values_sha:
                raise ValueError('Actual owned feature/reference values changed during the session')
            if any(m._forward_hooks for m in self._replay.modules.values()):
                raise RuntimeError('A normalization hook remains after the session')
            return dict(binding=asdict(self.binding), actual_forwards=dict(self._calls),
                        full_model_values_unchanged=True, hooks_removed=True)
        finally:
            self._closed = True
            self._replay.frames.clear()
            self._replay.ready = False
            self._actions.clear()
            self._base_features.clear()
            self._corrected_stage2 = None
            self._base_logits = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
