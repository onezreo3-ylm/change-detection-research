import unittest

import torch
from torch import nn

from pcdc.local_feature_replay import ReferenceGroupNormReplay, fpn_convolution_support


class ReferenceReplayTests(unittest.TestCase):
    def model(self, affine=True):
        m = nn.Sequential(nn.GroupNorm(2, 4, affine=affine), nn.ReLU(inplace=True)).eval()
        m.requires_grad_(False)
        return m

    def test_zero_action_and_unchanged_positions_are_bit_exact(self):
        torch.set_num_threads(2); g = torch.Generator().manual_seed(712)
        m = self.model(); x = torch.rand(2, 4, 8, 8, generator=g)
        r = ReferenceGroupNormReplay(m)
        with r.capture(): y0 = m(x)
        with r.replay(): same = m(x)
        self.assertTrue(torch.equal(y0, same))
        edited = x.clone(); edited[:, :, 3, 3] += 1
        with r.replay(): y1 = m(edited)
        outside = torch.ones_like(x, dtype=torch.bool); outside[:, :, 3, 3] = False
        self.assertTrue(torch.equal(y0[outside], y1[outside]))
        self.assertGreater(float((y1-y0).abs().max()), 0)

    def test_actual_reference_is_not_overwritten_by_inplace_relu(self):
        m = self.model(); x = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        r = ReferenceGroupNormReplay(m)
        with r.capture(): m(x)
        self.assertLess(float(r.frames['0'].y.min()), 0)

    def test_action_gradient_has_no_gn_cross_position_coupling(self):
        m = nn.Sequential(nn.GroupNorm(2, 4)).eval().requires_grad_(False)
        x = torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 40
        r = ReferenceGroupNormReplay(m)
        with r.capture(): m(x)
        edited = x.clone().requires_grad_(True)
        with r.replay(): y = m(edited)
        grad = torch.autograd.grad(y[0, 0, 0, 0], edited)[0]
        self.assertGreater(float(grad[0, 0, 0, 0]), 0)
        grad[0, 0, 0, 0] = 0
        self.assertEqual(int(torch.count_nonzero(grad)), 0)

    def test_uncaptured_cache_changed_shape_and_nonfinite_input_are_rejected(self):
        m = self.model(); r = ReferenceGroupNormReplay(m); x = torch.ones(1, 4, 4, 4)
        with self.assertRaises(RuntimeError), r.replay(): m(x)
        with r.capture(): m(x)
        with self.assertRaises(ValueError), r.replay(): m(x.repeat(2, 1, 1, 1))
        x[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError), r.replay(): m(x)
        self.assertFalse(r.active); self.assertFalse(m[0]._forward_hooks)

    def test_exception_and_missing_forward_always_remove_hooks(self):
        m = self.model(); r = ReferenceGroupNormReplay(m)
        with self.assertRaises(RuntimeError), r.capture(): raise RuntimeError('synthetic failure')
        self.assertFalse(r.ready); self.assertFalse(r.frames); self.assertFalse(m[0]._forward_hooks)
        with self.assertRaises(RuntimeError), r.capture(): pass
        self.assertFalse(r.ready); self.assertFalse(m[0]._forward_hooks)

    def test_mutating_affine_or_training_state_cannot_reuse_reference(self):
        m = self.model(); r = ReferenceGroupNormReplay(m); x = torch.ones(1, 4, 4, 4)
        with r.capture(): m(x)
        with torch.no_grad(): m[0].weight.add_(1)
        with self.assertRaises(ValueError), r.replay(): m(x)
        m.train()
        with self.assertRaises(ValueError), r.capture(): m(x)
        m.eval().requires_grad_(True)
        with self.assertRaises(ValueError), r.capture(): m(x)

    def test_no_affine_and_float64_keep_the_explicit_dtype(self):
        m = self.model(affine=False).double(); r = ReferenceGroupNormReplay(m)
        x = torch.arange(64, dtype=torch.float64).reshape(1, 4, 4, 4)
        with r.capture(): y0 = m(x)
        with r.replay(): y1 = m(x)
        self.assertEqual(y1.dtype, torch.float64); self.assertTrue(torch.equal(y0, y1))

    def test_nested_and_repeated_forward_are_rejected_and_cleanup_runs(self):
        m = self.model(); r = ReferenceGroupNormReplay(m); x = torch.ones(1, 4, 4, 4)
        with self.assertRaises(RuntimeError), r.capture():
            with r.capture(): pass
        self.assertFalse(r.active)
        with self.assertRaises(RuntimeError), r.capture():
            m(x); m(x)
        self.assertFalse(r.ready); self.assertFalse(m[0]._forward_hooks)

    def test_convolution_support_is_local_and_empty_mask_stays_empty(self):
        masks = [torch.zeros(1, 1, s, s, dtype=torch.bool) for s in (32, 16, 8, 4)]
        self.assertFalse(fpn_convolution_support(masks, (128, 128)).any())
        masks[1][:, :, 7:10, 7:10] = True
        support = fpn_convolution_support(masks, (128, 128))
        self.assertTrue(support.any()); self.assertFalse(support.all())
        self.assertEqual(support.shape, (1, 1, 128, 128))


if __name__ == '__main__': unittest.main()
