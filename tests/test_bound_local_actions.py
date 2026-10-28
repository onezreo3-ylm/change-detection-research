from dataclasses import replace
import unittest

import torch

from pcdc.bound_local_actions import (
    BoundLocalActionSession, ObservedBatchIdentity, ReferenceBinding, model_tensor_sha256,
)
from pcdc.local_action_composition import compose_disjoint_outputs
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector


class BoundLocalActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, batch=1):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(27813)
            model = AdapterCorrectedDetector(SiameseChangeDetector(width=32, groups=8))
            torch.nn.init.normal_(model.adapter.conv3.weight, std=.01)
            model.eval().requires_grad_(False)
            a, b = torch.rand(batch, 3, 128, 128), torch.rand(batch, 3, 128, 128)
        identity = ObservedBatchIdentity(
            tuple(f'train/synthetic_{i}.png@x0_y0' for i in range(batch)),
            tuple(('a'*64, 'b'*64) for _ in range(batch)),
            tuple('c'*64 for _ in range(batch)), 'd'*64, 'e'*64, 'f'*64)
        return model, a, b, identity

    def session(self, batch=1):
        model, a, b, identity = self.fixture(batch)
        return BoundLocalActionSession(model, a, b, identity, amp=False, engineering_only=True)

    def masks(self, session):
        mask = torch.zeros(session._base_features[1].shape[0], 1, 16, 16, dtype=torch.bool)
        other = mask.clone()
        mask[:, :, 1:4, 1:4] = True
        other[:, :, 12:15, 12:15] = True
        return mask, other

    def test_actual_encoding_full_mask_replays_original_a0_and_ordinary_a1(self):
        model, a, b, identity = self.fixture()
        with torch.no_grad():
            old_a0, old_a1 = model.action_logits(a, b)
        s = BoundLocalActionSession(model, a, b, identity, amp=False, engineering_only=True)
        with s:
            s.register('full', torch.ones(1, 1, 16, 16, dtype=torch.bool), binding=s.binding)
            self.assertTrue(torch.equal(old_a0, s.baseline(binding=s.binding)))
            self.assertTrue(torch.equal(old_a1, s.decode(('full',), binding=s.binding, arm='ordinary_GN')))
            self.assertTrue(torch.equal(old_a0, s.decode((), binding=s.binding, arm='reference_replay')))

    def test_joint_replay_is_actual_same_reference_composition(self):
        s = self.session(batch=2)
        with s:
            ma, mb = self.masks(s)
            s.register('A', ma, binding=s.binding)
            s.register('B', mb, binding=s.binding)
            base = s.baseline(binding=s.binding)
            a, b = [s.decode((n,), binding=s.binding, arm='reference_replay') for n in ('A', 'B')]
            joint = s.decode(('A', 'B'), binding=s.binding, arm='reference_replay')
            supports = [s.support(n, binding=s.binding) for n in ('A', 'B')]
            self.assertTrue(torch.equal(joint, compose_disjoint_outputs(base, [a, b], supports)))
            self.assertGreater(int(torch.count_nonzero(a-base)), 0)
            self.assertEqual(s._calls, {'encoder':1, 'adapter':2, 'decoder':4})

    def test_wrong_reference_and_ordered_sample_evidence_are_rejected(self):
        s = self.session(batch=2)
        with s:
            wrong = ReferenceBinding('another-session', s.binding.evidence_sha256)
            with self.assertRaises(ValueError): s.baseline(binding=wrong)
            original = s.identity
            s.identity = replace(original, sample_ids=tuple(reversed(original.sample_ids)))
            with self.assertRaises(ValueError): s.baseline(binding=s.binding)
            s.identity = original

    def test_cache_owns_input_mask_and_returned_output_copies(self):
        model, a, b, identity = self.fixture()
        s = BoundLocalActionSession(model, a, b, identity, amp=False, engineering_only=True)
        with s:
            ma, _ = self.masks(s)
            s.register('A', ma, binding=s.binding)
            support = s.support('A', binding=s.binding)
            base = s.baseline(binding=s.binding)
            a.zero_(); b.zero_(); ma.zero_()
            s.baseline(binding=s.binding).add_(100)
            s.support('A', binding=s.binding).zero_()
            self.assertTrue(torch.equal(base, s.baseline(binding=s.binding)))
            self.assertTrue(torch.equal(support, s.support('A', binding=s.binding)))

    def test_overlap_unknown_duplicate_empty_actions_and_wrong_mask_are_rejected(self):
        s = self.session()
        with s:
            ma, _ = self.masks(s)
            s.register('A', ma, binding=s.binding)
            s.register('overlap', ma.clone(), binding=s.binding)
            for names in (('A', 'overlap'), ('A', 'A'), ('missing',)):
                with self.assertRaises(ValueError): s.decode(names, binding=s.binding, arm='reference_replay')
            with self.assertRaises(ValueError): s.register('A', ma, binding=s.binding)
            with self.assertRaises(ValueError): s.register('empty', torch.zeros_like(ma), binding=s.binding)
            with self.assertRaises(ValueError): s.register('float', ma.float(), binding=s.binding)
            with self.assertRaises(ValueError): s.decode((), binding=s.binding, arm='unknown')

    def test_model_mutation_invalidates_reference_and_close_always_frees_cache(self):
        s = self.session()
        with torch.no_grad(): s.model.detector.decoder.classifier.bias.add_(1)
        with self.assertRaises(ValueError): s.baseline(binding=s.binding)
        with self.assertRaises(ValueError): s.close()
        self.assertTrue(s._closed)
        self.assertFalse(s._replay.frames)
        self.assertFalse(s._base_features)

    def test_data_mutation_bypassing_versions_fails_full_final_identity_check(self):
        s = self.session()
        s.model.detector.decoder.classifier.bias.data.add_(1)
        with self.assertRaisesRegex(ValueError, 'Actual frozen model values'): s.close()
        s = self.session()
        s._base_features[0].data.add_(1)
        with self.assertRaisesRegex(ValueError, 'Actual owned feature'): s.close()

    def test_decoder_failure_removes_all_reference_hooks(self):
        s = self.session()
        def fail(module, args): raise RuntimeError('Injected inference failure')
        handle = s.model.detector.decoder.classifier.register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Injected inference failure'):
                s.decode((), binding=s.binding, arm='reference_replay')
            self.assertFalse(s._replay.active)
            self.assertTrue(all(not m._forward_hooks for m in s._replay.modules.values()))
        finally:
            handle.remove()
            s.close()

    def test_explicit_precision_device_and_frozen_model_requirements(self):
        model, a, b, identity = self.fixture()
        with self.assertRaises(ValueError): BoundLocalActionSession(model, a, b, identity, amp=False)
        with self.assertRaises(ValueError): BoundLocalActionSession(model, a, b, identity, amp=True, engineering_only=True)
        model.train()
        with self.assertRaises(ValueError): BoundLocalActionSession(model, a, b, identity, amp=False, engineering_only=True)
        model.eval(); model.adapter.requires_grad_(True)
        with self.assertRaises(ValueError): BoundLocalActionSession(model, a, b, identity, amp=False, engineering_only=True)

    def test_malformed_identity_wrong_batch_and_closed_session_are_rejected(self):
        model, a, b, identity = self.fixture()
        with self.assertRaises(ValueError): replace(identity, sample_ids=('val/sample.png',))
        with self.assertRaises(ValueError): replace(identity, instance_sha256=())
        with self.assertRaises(ValueError): replace(identity, actor_checkpoint_sha256='unknown')
        with self.assertRaises(ValueError): BoundLocalActionSession(model, a.repeat(2,1,1,1), b.repeat(2,1,1,1), identity, amp=False, engineering_only=True)
        s = self.session()
        result = s.close()
        self.assertEqual(result['actual_forwards'], {'encoder':1, 'adapter':2, 'decoder':1})
        with self.assertRaises(RuntimeError): s.baseline(binding=s.binding)
        with self.assertRaises(RuntimeError): s.close()

    def test_nonfinite_owned_cache_and_evidence_edits_are_not_certified(self):
        s = self.session()
        s._base_features[0].fill_(float('nan'))
        with self.assertRaises(ValueError): s.decode((), binding=s.binding, arm='reference_replay')
        with self.assertRaises(ValueError): s.close()
        s = self.session()
        s.evidence['normalized_a_sha256'][0] = '0'*64
        with self.assertRaises(ValueError): s.close()


if __name__ == '__main__': unittest.main()
