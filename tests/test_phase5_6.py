"""Engineering checks for the approved Phase 5-6 path; never research results."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pcdc.corruption_protocol import KINDS, load_protocol
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import (
    AdapterCorrectedDetector,
    Stage2ResidualAdapter,
    per_sample_change_loss,
)
from pcdc.phase5_protocol import gate_instances, load_phase56_config
from pcdc.phase5_training import SourceSubsetDataset
from pcdc.phase6_diagnostic import (
    _bootstrap,
    per_sample_confusion,
    per_sample_transitions,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase5ModelTests(unittest.TestCase):
    def test_adapter_architecture_and_zero_action_are_exact(self):
        adapter = Stage2ResidualAdapter()
        self.assertEqual((adapter.conv1.in_channels, adapter.conv1.out_channels), (128, 32))
        self.assertEqual((adapter.conv2.kernel_size, adapter.conv2.padding), ((3, 3), (1, 1)))
        self.assertEqual((adapter.conv3.in_channels, adapter.conv3.out_channels), (32, 128))
        self.assertIsNone(adapter.conv1.bias)
        self.assertIsNone(adapter.conv2.bias)
        self.assertIsNotNone(adapter.conv3.bias)
        self.assertEqual(torch.count_nonzero(adapter.conv3.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.conv3.bias).item(), 0)
        features = torch.randn(2, 128, 8, 8)
        torch.testing.assert_close(adapter(features), torch.zeros_like(features), rtol=0, atol=0)

    def test_frozen_detector_stays_eval_and_only_adapter_receives_gradients(self):
        torch.manual_seed(7)
        detector = SiameseChangeDetector(width=32, groups=8)
        model = AdapterCorrectedDetector(detector)
        model.train(True)
        self.assertTrue(model.adapter.training)
        self.assertFalse(model.detector.training)
        tracked = [module.num_batches_tracked.clone() for module in detector.modules()
                   if isinstance(module, torch.nn.BatchNorm2d)]
        a, b = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        target = torch.zeros(1, 1, 32, 32)
        output = model(a, b)["logits"]
        ChangeDetectionLoss(1, 1, 1)(output, target)["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.detector.parameters()))
        after = [module.num_batches_tracked for module in detector.modules()
                 if isinstance(module, torch.nn.BatchNorm2d)]
        for before, current in zip(tracked, after):
            torch.testing.assert_close(before, current, rtol=0, atol=0)

    def test_a0_and_initial_a1_logits_match_bitwise(self):
        torch.manual_seed(8)
        model = AdapterCorrectedDetector(SiameseChangeDetector(width=32, groups=8)).eval()
        a, b = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            a0, a1 = model.action_logits(a, b)
        torch.testing.assert_close(a0, a1, rtol=0, atol=0)

    def test_per_sample_loss_mean_is_existing_loss(self):
        torch.manual_seed(9)
        logits = torch.randn(3, 1, 7, 5)
        target = torch.randint(0, 2, logits.shape).float()
        per = per_sample_change_loss(logits, target, bce_weight=1.3,
                                     dice_weight=0.7, dice_smooth=1.0)
        existing = ChangeDetectionLoss(1.3, 0.7, 1.0)(logits, target)
        for key in ("total", "bce", "dice"):
            torch.testing.assert_close(per[key].mean(), existing[key])


class Phase6MathTests(unittest.TestCase):
    def test_confusions_and_transitions_use_a0_to_a1_direction(self):
        target = torch.tensor([[[[0., 0., 1., 1.]]]])
        # a0: [FP,TN,FN,TP], a1: [TN,FP,TP,FN]
        a0 = torch.tensor([[[[10., -10., -10., 10.]]]])
        a1 = torch.tensor([[[[-10., 10., 10., -10.]]]])
        self.assertEqual(per_sample_confusion(a0, target, 0.5).tolist(), [[1, 1, 1, 1]])
        self.assertEqual(per_sample_confusion(a1, target, 0.5).tolist(), [[1, 1, 1, 1]])
        self.assertEqual(per_sample_transitions(a0, a1, target, 0.5).tolist(), [[1, 1, 1, 1]])

    def test_source_pair_bootstrap_is_deterministic(self):
        counts = np.zeros((3, 2, 3, 4), dtype=np.int64)
        counts[..., 0, :] = [90, 10, 20, 80]
        counts[..., 1, :] = [92, 8, 16, 84]
        counts[..., 2, :] = [94, 6, 12, 88]
        config = SimpleNamespace(values={"diagnostic": {
            "bootstrap_seed": 11, "bootstrap_replicates": 100,
            "confidence_level": 0.95}})
        first = _bootstrap(counts, config, better_constant_index=1)
        second = _bootstrap(counts, config, better_constant_index=1)
        self.assertEqual(first, second)
        self.assertGreater(first["oracle_mf1_minus_better_constant"][0], 0)


class Phase5ProtocolTests(unittest.TestCase):
    def test_workspace_config_is_bound_and_keeps_test_closed(self):
        config = load_phase56_config(ROOT / "configs/phase5_6_v1_seed42.yaml",
                                     require_approval=False)
        self.assertFalse(config.base.run_test_after_training)
        self.assertEqual(config.values["boundaries"]["held_out_test_access"], False)
        self.assertEqual(config.values["split"]["actor_sources"], 356)
        self.assertEqual(config.values["split"]["gate_sources"], 89)

    def test_gate_instances_are_full_deterministic_cells_with_paired_levels(self):
        protocol = load_protocol(ROOT / "runs/phase2_protocol_v1_20260913/candidate_resolved.yaml")
        config = SimpleNamespace(
            corruption=protocol,
            values={"diagnostic": {"manifest_salt": "phase6_gate_action_v1_seed42",
                                    "manifest_view": 1}})
        ids = [f"train/source_{index}.png@x0_y0" for index in range(8)]
        first = list(gate_instances(config, ids))
        second = list(gate_instances(config, list(reversed(ids))))
        self.assertEqual(len(first), 45 * len(ids))
        # The manifest order follows its input IDs, but every instance identity is invariant.
        by_key = lambda rows: {(row["sample_id"], row["condition_id"]): row for row in rows}
        self.assertEqual(by_key(first), by_key(second))
        self.assertEqual({row["condition_id"] for row in first},
                         {f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)})
        grouped = {}
        for row in first:
            key = (row["sample_id"], row["kind"])
            grouped.setdefault(key, []).append(row)
        for rows in grouped.values():
            self.assertEqual(len(rows), 5)
            self.assertEqual(len({(row["direction"], row["target_date"],
                                   row["spatial_mode"], row["seed"],
                                   str(row["injection_mask"])) for row in rows}), 1)

    def test_source_subset_keeps_whole_source_grids(self):
        records = [SimpleNamespace(name=f"s{i}.png") for i in range(3)]

        class Base:
            split = "train"
            settings = {}
            root = ROOT
            tile_size = 32
            mean = torch.zeros(3, 1, 1)
            std = torch.ones(3, 1, 1)
            pair_transform = None

            def __init__(self):
                self.records = records
                self.tiles = [(source, x, 0) for source in range(3) for x in (0, 32)]

            def __getitem__(self, index):
                return {"sample_id": str(self.tiles[index])}

            def read_pair(self, *args, **kwargs):
                return None

        subset = SourceSubsetDataset(Base(), ["s0.png", "s2.png"], partition="actor")
        self.assertEqual(len(subset), 4)
        self.assertEqual(subset.manifest()["source_pairs"], 2)
        self.assertEqual([subset[index]["sample_id"] for index in range(4)],
                         ["(0, 0, 0)", "(0, 32, 0)", "(2, 0, 0)", "(2, 32, 0)"])


if __name__ == "__main__":
    unittest.main()
