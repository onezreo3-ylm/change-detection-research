"""Engineering-only seam for mutually exclusive placements of one frozen adapter.

No new expert, gate or learned parameter. Production research admission requires
its own protocol, real precision checks and budget; this module does not launch.
"""
from __future__ import annotations

import torch

from .bound_local_actions import BoundLocalActionSession,tensor_sha256


class BoundTemporalActionSession(BoundLocalActionSession):
    KINDS=('A_only','B_only','both')

    def __init__(self,model,image_a,image_b,identity,*,amp,engineering_only=False):
        self._building_alternatives=True
        self._require_no_external_hooks(model)
        captured=[]
        def observe(module,args,output):
            if len(captured)>=2:raise RuntimeError('Only the two original baseline adapter calls are admitted')
            captured.append((args[0].detach().clone(),output.detach().clone()))
        handle=model.adapter.register_forward_hook(observe)
        try:
            super().__init__(model,image_a,image_b,identity,amp=amp,engineering_only=engineering_only)
        finally:
            handle.remove()
        try:
            if len(captured)!=2:raise RuntimeError('Original ordered A/B adapter outputs were not captured')
            (a,ra),(b,rb)=captured
            with torch.no_grad(),self._precision():
                self._variants=dict(A_only=torch.abs((a+ra)-b).detach().clone(),
                    B_only=torch.abs(a-(b+rb)).detach().clone(),both=self._corrected_stage2)
            if any(not torch.isfinite(t).all() for t in self._variants.values()):raise ValueError('Nonfinite actual temporal variant')
            self._variant_signature=self._variant_versions()
            self._variant_hashes=self._variant_value_hashes()
            self._which={};self._which_signature=()
            self._building_alternatives=False
        except BaseException:
            # The parent completed its capture; clean it even if variant assembly fails.
            try:BoundLocalActionSession.close(self)
            finally:
                if hasattr(self,'_variants'):self._variants.clear()
            raise

    @staticmethod
    def _require_no_external_hooks(model):
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
            raise ValueError('An original frozen model without external forward hooks is required')

    def _variant_versions(self):
        return tuple((name,id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,t.device)
                     for name,t in self._variants.items())

    def _variant_value_hashes(self):
        return {name:tensor_sha256(t) for name,t in self._variants.items()}

    def _check(self,binding):
        super()._check(binding)
        self._require_no_external_hooks(self.model)
        if self._building_alternatives:return
        if self._variant_versions()!=self._variant_signature:
            raise ValueError('Owned temporal variants changed after the actual reference capture')
        if tuple(sorted(self._which.items()))!=self._which_signature:
            raise ValueError('Action placement identity changed after registration')

    def register(self,name,kind,stage2_mask,*,binding):
        self._check(binding)
        if kind not in self.KINDS:raise ValueError('Unknown explicitly named temporal placement')
        super().register(name,stage2_mask,binding=binding)
        self._which[name]=kind;self._which_signature=tuple(sorted(self._which.items()))

    def decode(self,names,*,binding,arm):
        self._check(binding)
        if not isinstance(names,tuple) or len(set(names))!=len(names) or not set(names)<=self._which.keys():
            raise ValueError('Require unique registered temporal actions from this session')
        if arm not in ('reference_replay','ordinary_GN','ordinary_GN_output_mask'):raise ValueError('Unknown comparison arm')
        union=torch.zeros_like(self._base_logits,dtype=torch.bool)
        features=list(self._base_features)
        for name in names:
            mask,support=self._actions[name]
            if (union&support).any():raise ValueError('Mutually exclusive or spatially overlapping actions cannot be combined')
            union|=support
            features[1]=torch.where(mask,self._variants[self._which[name]],features[1])
        with torch.no_grad(),self._precision():
            if arm=='reference_replay':
                with self._replay.replay():logits=self._forward(features)
            else:logits=self._forward(features)
        if arm=='ordinary_GN_output_mask':logits=torch.where(union,logits,self._base_logits)
        self._check(binding)
        return logits.detach().clone()

    def close(self):
        error=None
        try:
            if self._variant_value_hashes()!=self._variant_hashes:
                error=ValueError('Actual temporal-variant values changed during the session')
        except BaseException as caught:error=caught
        try:
            result=super().close()
            if error is not None:raise error
            result.update(temporal_variant_values_unchanged=True,
                          temporal_variant_rule='same_observed_A_B_and_two_fixed_shared_adapter_outputs')
            return result
        finally:
            if hasattr(self,'_variants'):self._variants.clear()
            if hasattr(self,'_which'):self._which.clear()
