import unittest
from unittest import mock
import torch
from pcdc.losses import ChangeDetectionLoss
from scripts import five_view_baselines as f
from scripts import five_view_bf16 as bf16


class Bf16Contracts(unittest.TestCase):
    def test_cpu_is_never_substituted_for_requested_cuda(self):
        with self.assertRaises(RuntimeError):bf16.require_bf16(torch.device('cpu'))

    def test_missing_hardware_support_is_never_substituted_with_fp16(self):
        with mock.patch('torch.cuda.is_bf16_supported',return_value=False):
            with self.assertRaises(RuntimeError):bf16.require_bf16(torch.device('cuda'))

    def test_loss_accumulates_bf16_logits_as_the_same_explicit_fp32_objective(self):
        z=torch.linspace(-3,3,40).reshape(10,1,2,2).to(torch.bfloat16).requires_grad_()
        target=torch.tensor([0.,1.,1.,0.,1.,0.,0.,1.]).reshape(2,1,2,2)
        loss=ChangeDetectionLoss(1,1,1)
        observed=f.weighted_loss(z.float(),target,loss)
        expected=sum(w*loss(z.float()[2*i:2*i+2],target)['total'] for i,w in enumerate([.5,.125,.125,.125,.125]))
        self.assertEqual(observed['total'].dtype,torch.float32)
        self.assertTrue(torch.equal(observed['total'],expected));observed['total'].backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_bf16_update_refuses_to_run_a_cpu_model(self):
        with self.assertRaises(RuntimeError):bf16.update(None,None,None,{},torch.device('cpu'),1.)
