"""Exact numerical contracts for engineering-only speed candidates."""
import unittest
import torch

from pcdc.metrics import BinaryMetrics
from pcdc.losses import ChangeDetectionLoss
from scripts import five_view_baselines as f
from scripts import five_view_speed as fast


class SpeedTests(unittest.TestCase):
    def test_cpu_packed_inputs_exactly_preserve_view_order_and_layout(self):
        for n in (1,4):
            rows=[{'a':torch.rand(5,3,256,256),'b':torch.rand(5,3,256,256),'mask':torch.zeros(1,256,256)} for _ in range(n)]
            normal=torch.utils.data.default_collate(rows)
            packed=fast.collate_view_major(rows)
            for a,b in zip(f.transfer(normal,torch.device('cpu')),fast.transfer(packed,torch.device('cpu'))):
                self.assertTrue(torch.equal(a,b))
            self.assertTrue(packed['a'].is_contiguous(memory_format=torch.channels_last))

    def test_device_confusion_counts_exactly_equal_original_including_threshold_edges(self):
        z=torch.tensor([-1e-9,0.,1e-9,-10,10]).reshape(1,1,1,5)
        target=torch.tensor([0,1,0,1,1.]).reshape_as(z)
        normal,packed=BinaryMetrics(.5),fast.DeviceMetrics(.5)
        for _ in range(3):normal.update(z,target);packed.update(z,target)
        self.assertEqual(normal.compute(),packed.compute())

    def test_invalid_metrics_are_rejected_before_saved_results(self):
        for logits,target in [(torch.tensor([float('nan')]),torch.zeros(1)),(torch.zeros(1),torch.tensor([.3]))]:
            metrics=fast.DeviceMetrics(.5);metrics.update(logits,target)
            with self.assertRaises(ValueError):metrics.compute()

    def test_wrong_packed_shape_or_threshold_is_rejected(self):
        with self.assertRaises(ValueError):fast.DeviceMetrics(.4)
        with self.assertRaises(ValueError):fast.transfer({'a':torch.zeros(1,3,256,256),'b':torch.zeros(1,3,256,256),'mask':torch.zeros(1,1,256,256)},torch.device('cpu'))
