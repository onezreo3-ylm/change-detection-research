from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pcdc.phase5_models import Stage2ResidualAdapter
from pcdc.phase7_cache import _discard_uncommitted, _scalar_stats, _weighted_stats
from pcdc.phase7_evaluation import (
    POLICIES,
    _auc,
    _decision_matrix,
    phase7_evaluation_source_files,
)
from pcdc.phase7_models import (
    build_phase7_selector,
    descriptor_from_stage2,
    phase7_descriptor,
    selector_initial_sha256,
)
from pcdc.phase7_protocol import load_phase7_config, load_selector_split, preflight
from pcdc.phase7_training import (
    _epoch_seed,
    _hash_order,
    _scores,
    _threshold_value,
    threshold_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase7_v1_seed42.yaml"


class Phase7ProtocolTests(unittest.TestCase):
    def test_approved_split_and_metadata_preflight_are_exact(self) -> None:
        config = load_phase7_config(CONFIG)
        manifest = load_selector_split(config)
        self.assertEqual(len(manifest["fit_names"]), 71)
        self.assertEqual(len(manifest["calibration_names"]), 18)
        self.assertEqual(manifest["summary"]["gate_fit"]["tiles"], 1136)
        self.assertEqual(manifest["summary"]["gate_calibration"]["tiles"], 288)
        result = preflight(config)
        self.assertEqual(result["status"], "phase7_metadata_preflight_passed")
        self.assertFalse(result["dataset_opened"])
        self.assertFalse(result["test_evaluated"])


class Phase7ModelTests(unittest.TestCase):
    def test_selector_architecture_initialization_and_parameter_count(self) -> None:
        first = build_phase7_selector(42)
        second = build_phase7_selector(42)
        self.assertEqual(sum(x.numel() for x in first.parameters()), 139970)
        self.assertEqual(selector_initial_sha256(42), selector_initial_sha256(42))
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        output = first(torch.zeros(3, 1024))
        self.assertEqual(tuple(output.shape), (3, 2))
        with self.assertRaises(ValueError):
            first(torch.zeros(3, 1023))

    def test_descriptor_is_exactly_date_swap_invariant(self) -> None:
        generator = torch.Generator().manual_seed(7)
        stage_a = torch.randn(2, 128, 8, 8, generator=generator)
        stage_b = torch.randn(2, 128, 8, 8, generator=generator)
        adapter = Stage2ResidualAdapter()
        residual_a, residual_b = adapter(stage_a), adapter(stage_b)
        direct = descriptor_from_stage2(stage_a, stage_b, residual_a, residual_b)
        swapped = descriptor_from_stage2(stage_b, stage_a, residual_b, residual_a)
        wrapped = phase7_descriptor(
            [torch.empty(0), stage_a, torch.empty(0), torch.empty(0)],
            [torch.empty(0), stage_b, torch.empty(0), torch.empty(0)], adapter)
        self.assertEqual(tuple(direct.shape), (2, 1024))
        self.assertTrue(torch.equal(direct, swapped))
        self.assertTrue(torch.equal(direct, wrapped))


class Phase7NormalizationTests(unittest.TestCase):
    def test_equal_clean_perturbed_weighted_statistics(self) -> None:
        clean = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        perturbed = [np.asarray([[2.0, 0.0], [4.0, 2.0]], dtype=np.float32)]
        mean, std = _weighted_stats(clean, perturbed, 1e-6)
        combined = np.concatenate((clean, perturbed[0]))
        np.testing.assert_allclose(mean, combined.mean(axis=0))
        np.testing.assert_allclose(std, combined.std(axis=0))
        scalar = _scalar_stats(np.asarray([0.0, 2.0]), np.asarray([2.0, 4.0]), 1e-6)
        self.assertAlmostEqual(scalar["mean"], 2.0)
        self.assertAlmostEqual(scalar["std"], np.sqrt(2.0))

    def test_resume_cleanup_is_confined_to_uncommitted_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            final = root / "condition.npy"
            temporary = root / "condition.npy.tmp"
            final.write_bytes(b"partial-final")
            temporary.write_bytes(b"partial-temporary")
            _discard_uncommitted((final,), root)
            self.assertFalse(final.exists())
            self.assertFalse(temporary.exists())
            outside = Path(directory) / "outside.npy"
            outside.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                _discard_uncommitted((outside,), root)
            self.assertTrue(outside.exists())


class Phase7TrainingTests(unittest.TestCase):
    def test_shared_epoch_order_is_deterministic_and_epoch_specific(self) -> None:
        self.assertEqual(_epoch_seed(42, 1), _epoch_seed(42, 1))
        self.assertNotEqual(_epoch_seed(42, 1), _epoch_seed(42, 2))
        self.assertEqual(_hash_order(31, 4, 42), _hash_order(31, 4, 42))
        self.assertNotEqual(_hash_order(31, 4, 42), _hash_order(31, 4, 43))

    def test_g2_g3_harm_scores_are_information_equivalent(self) -> None:
        normalization = {"targets": {
            "u_obs": {"mean": 0.2, "std": 0.5},
            "u_clean": {"mean": -0.1, "std": 0.25},
            "a_obs": {"mean": 0.3, "std": 0.4},
        }}
        primary_standardized = np.asarray([[-0.4], [0.6]], dtype=np.float64)
        clean_standardized = np.asarray([[0.8], [-0.2]], dtype=np.float64)
        primary = primary_standardized[:, 0] * 0.5 + 0.2
        clean = clean_standardized[:, 0] * 0.25 - 0.1
        advantage = primary - clean
        advantage_standardized = ((advantage - 0.3) / 0.4)[:, None]
        g2_output = np.concatenate((primary_standardized, clean_standardized), axis=1)
        g3_output = np.concatenate((primary_standardized, advantage_standardized), axis=1)
        g2_primary, g2_harm = _scores("g2", g2_output, normalization)
        g3_primary, g3_harm = _scores("g3", g3_output, normalization)
        np.testing.assert_allclose(g2_primary, g3_primary)
        np.testing.assert_allclose(g2_harm, g3_harm)

    def test_threshold_records_round_trip_without_nonstandard_json_numbers(self) -> None:
        for value in (-np.inf, -0.25, 0.0, 2.5, np.inf):
            record = threshold_record(float(value))
            self.assertEqual(_threshold_value(record), value)
            self.assertNotIn("Infinity", __import__("json").dumps(record, allow_nan=False))


class Phase7EvaluationTests(unittest.TestCase):
    def test_evaluation_source_seal_inventory_is_complete(self) -> None:
        files = phase7_evaluation_source_files()
        self.assertIn("pcdc/phase7_evaluation.py", files)
        self.assertIn("scripts/evaluate_phase7_guarded.py", files)
        self.assertTrue(all(path.is_file() for path in files.values()))

    def test_decisions_are_whole_tile_and_thresholded_exactly(self) -> None:
        normalization = {"targets": {
            "u_obs": {"mean": 0.0, "std": 1.0},
            "u_clean": {"mean": 0.0, "std": 1.0},
            "a_obs": {"mean": 0.0, "std": 1.0},
        }}
        outputs = {
            "c1": torch.tensor([[-1.0, 99.0], [1.0, 99.0]]),
            "g1": torch.tensor([[-1.0, 99.0], [1.0, 99.0]]),
            "g2": torch.tensor([[1.0, -1.0], [1.0, 2.0]]),
            "g3": torch.tensor([[1.0, 2.0], [1.0, -1.0]]),
        }
        decisions, scores, _ = _decision_matrix(
            outputs, {name: 0.0 for name in ("c1", "g1", "g2", "g3")},
            normalization, np.asarray([-0.5, 0.5]))
        self.assertEqual(decisions.dtype, np.bool_)
        self.assertEqual(decisions.shape, (2, len(POLICIES)))
        self.assertFalse(decisions[:, POLICIES.index("A0")].any())
        self.assertTrue(decisions[:, POLICIES.index("G0")].all())
        self.assertEqual(decisions[:, POLICIES.index("O")].tolist(), [False, True])
        self.assertEqual(decisions[:, POLICIES.index("G2-H")].tolist(), [False, True])
        self.assertEqual(decisions[:, POLICIES.index("G3-H")].tolist(), [False, True])
        np.testing.assert_allclose(scores["g2"]["constrained"], [-1.0, 1.0])
        np.testing.assert_allclose(scores["g3"]["constrained"], [-1.0, 1.0])

    def test_auc_uses_average_ranks_for_ties(self) -> None:
        self.assertAlmostEqual(_auc(np.asarray([0.0, 1.0]), np.asarray([0, 1])), 1.0)
        self.assertAlmostEqual(
            _auc(np.asarray([0.0, 0.0, 1.0, 1.0]), np.asarray([0, 1, 0, 1])), 0.5)
        self.assertIsNone(_auc(np.asarray([0.0, 1.0]), np.asarray([1, 1])))


if __name__ == "__main__":
    unittest.main()
