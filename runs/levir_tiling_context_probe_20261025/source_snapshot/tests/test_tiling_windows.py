import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pcdc.config import Config, load_config, protocol_signature
from scripts.levir_tiling_probe import (ROOT, WindowPairs, check_checkpoint_identity,
    precision_context, state_digest, summarize_maps, validate_train_records)
from scripts.tiling_windows import (ProbabilityMosaic, Window, aligned_crops,
    compare_probability, confusion, count_metrics, require_archive_replay,
    score_probability, support_mask, window_plan, window_resource_gate)


def unique(size=4, origins=(0, 4)):
    return {"window_size": size, "origins_yx_1d": list(origins), "aggregation": "none_unique_coverage"}


def overlap(size=4, origins=(0, 2)):
    return dict(unique(size, origins), aggregation="uniform_arithmetic_mean_probability_float32_before_threshold")


class WindowGeometryTests(unittest.TestCase):
    def test_original_and_common_coordinates_have_exact_coverage(self):
        windows, coverage, box = window_plan(unique(), 8)
        self.assertEqual([w.box for w in windows], [(0, 0, 4, 4), (4, 0, 8, 4), (0, 4, 4, 8), (4, 4, 8, 8)])
        np.testing.assert_array_equal(coverage, np.ones((8, 8)))
        alternate = dict(unique(4, (2,)), support_xyxy_exclusive=[2, 2, 6, 6])
        _, cov, _ = window_plan(alternate, 8)
        np.testing.assert_array_equal(cov > 0, support_mask((2, 2, 6, 6), 8))

    def test_holes_out_of_bounds_duplicates_and_unordered_origins_rejected(self):
        for arm, side in [(unique(4, (0, 5)), 9), (unique(4, (0, 6)), 8),
                          (unique(4, (0, 0)), 8), (unique(4, (4, 0)), 8), (unique(4, (True,)), 8)]:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                window_plan(arm, side)

    def test_overlap_cannot_silently_become_unique_coverage(self):
        with self.assertRaises(ValueError):
            window_plan(unique(4, (0, 2)), 6)

    def test_all_frozen_real_arms_and_compute_budget_are_geometrically_valid(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "configs/levir_tiling_context_probe_20261025.yaml").read_text(encoding="utf-8"))
        total, equivalent = 16, 16
        for arm in cfg["arms"].values():
            plan, _, _ = window_plan(arm, 1024)
            total += len(plan); equivalent += len(plan) * (arm["window_size"] // 256) ** 2
        self.assertEqual(total * 16 * 2, 3168)
        self.assertEqual(equivalent * 16 * 2, 4032)

    def test_paired_crops_preserve_absolute_positions_and_uint8_pixels(self):
        a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        b = 255 - a
        ca, cb = aligned_crops(a, b, Window(2, 3, 4))
        np.testing.assert_array_equal(ca, a[3:7, 2:6])
        np.testing.assert_array_equal(cb, b[3:7, 2:6])
        with self.assertRaises(ValueError):
            aligned_crops(a, b, Window(5, 0, 4))
        with self.assertRaises(ValueError):
            aligned_crops(a, b[:-1], Window(0, 0, 4))


class ProbabilityMosaicTests(unittest.TestCase):
    def test_probability_mean_is_taken_before_threshold_not_majority_vote(self):
        mosaic = ProbabilityMosaic(overlap(), 6)
        for w, p in zip(window_plan(overlap(), 6)[0], [0.49, 0.49, 0.49, 0.99]):
            mosaic.add(w, np.full((4, 4), p, dtype=np.float32))
        result, coverage = mosaic.finish()
        self.assertEqual(int(coverage[2, 2]), 4)
        self.assertAlmostEqual(float(result[2, 2]), 0.615, places=6)
        self.assertGreaterEqual(result[2, 2], 0.5)

    def test_duplicate_missing_and_unknown_windows_are_rejected(self):
        m = ProbabilityMosaic(unique(), 8); w = Window(0, 0, 4); p = np.zeros((4, 4), dtype=np.float32)
        m.add(w, p)
        for action in [lambda: m.add(w, p), lambda: m.add(Window(1, 0, 4), p), m.finish]:
            with self.assertRaises(ValueError):
                action()

    def test_nonfinite_thresholded_out_of_range_or_wrong_shape_input_rejected(self):
        for p in [np.full((4, 4), np.nan), np.full((4, 4), np.inf), np.full((4, 4), 1.01),
                  np.zeros((4, 4), dtype=np.uint8), np.zeros((3, 4))]:
            with self.subTest(dtype=p.dtype), self.assertRaises(ValueError):
                ProbabilityMosaic(unique(), 8).add(Window(0, 0, 4), p)

    def test_missing_support_stays_nan_and_is_not_scored_as_background(self):
        arm = dict(unique(4, (2,)), support_xyxy_exclusive=[2, 2, 6, 6])
        mosaic = ProbabilityMosaic(arm, 8)
        mosaic.add(Window(2, 2, 4), np.ones((4, 4), dtype=np.float32))
        p, _ = mosaic.finish(); common = support_mask([2, 2, 6, 6], 8)
        self.assertTrue(np.isnan(p[~common]).all())
        self.assertEqual(score_probability(np.ones((8, 8)), p, common)["counts"]["tp"], 16)
        with self.assertRaises(ValueError):
            score_probability(np.ones((8, 8)), p, np.ones((8, 8), dtype=bool))


class PairedMetricsTests(unittest.TestCase):
    def test_null_denominators_preserve_missing_class_support(self):
        scores = count_metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 12})
        self.assertIsNone(scores["recall"]); self.assertIsNone(scores["f1"])
        self.assertEqual(scores["oa"], 1); self.assertEqual(scores["fpr"], 0)

    def test_counts_are_pooled_before_f1_and_threshold_is_inclusive(self):
        label = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        p = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
        scores = score_probability(label, p, np.ones((2, 2), dtype=bool))
        self.assertEqual(scores["counts"], {"tp": 2, "fp": 1, "fn": 0, "tn": 1})
        self.assertEqual(scores["f1"], 0.8)

    def test_replay_requires_identical_pixels_even_if_counts_match(self):
        label = np.ones((2, 2), dtype=np.uint8)
        original = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        p = np.array([[0.1, 0.9], [0.1, 0.1]], dtype=np.float32)
        counts = confusion(label, original, np.ones((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            require_archive_replay(p, label, original, counts)
        require_archive_replay(original.astype(np.float32), label, original, counts)

    def test_same_pixel_disagreement_uses_common_support_and_finite_values(self):
        a = np.zeros((4, 4), dtype=np.float32); b = a.copy(); b[0, 0] = 1; b[1, 1] = 0.5
        common = support_mask([1, 1, 3, 3], 4)
        c = compare_probability(a, b, common)
        self.assertEqual(c["pixels"], 4); self.assertEqual(c["binary_disagreement_pixels"], 1)
        self.assertEqual(c["binary_disagreement_fraction"], 0.25)
        b[1, 1] = np.nan
        with self.assertRaises(ValueError):
            compare_probability(a, b, common)

    def test_gate_requires_both_models_all_sources_and_numeric_margin(self):
        models, sources = ["a", "b"], ["s1", "s2", "s3", "s4"]
        rules = {"alternate_grid_disagreement_minimum_fraction": .001,
                 "numeric_control_multiple": 10., "qualifying_source_count_minimum": 4}
        rows = [{"model": m, "source": s, "alternate_disagreement_fraction": .002,
                 "batch_disagreement_fraction": .0001} for m in models for s in sources]
        self.assertTrue(window_resource_gate(rows, models, sources, rules)["passed"])
        rows[-1]["batch_disagreement_fraction"] = .0003
        self.assertFalse(window_resource_gate(rows, models, sources, rules)["passed"])
        for bad in [rows[:-1], rows[:-1] + [rows[0]]]:
            with self.assertRaises(ValueError):
                window_resource_gate(bad, models, sources, rules)

    def test_summary_pools_counts_and_keeps_full_and_common_support_separate(self):
        labels = {"s1": np.ones((4, 4), dtype=np.uint8), "s2": np.zeros((4, 4), dtype=np.uint8)}
        maps = {"s1": np.ones((4, 4), dtype=np.float32), "s2": np.ones((4, 4), dtype=np.float32)}
        summary = summarize_maps(labels, maps, [1, 1, 3, 3], full_coverage=True, threshold=.5)
        self.assertEqual(summary["pooled"]["common"]["counts"], {"tp": 4, "fp": 4, "fn": 0, "tn": 0})
        self.assertEqual(summary["pooled"]["full"]["counts"]["tp"], 16)
        self.assertAlmostEqual(summary["pooled"]["common"]["f1"], 2 / 3)
        with self.assertRaises(ValueError):
            summarize_maps(labels, {"s1": maps["s1"]}, [1, 1, 3, 3], full_coverage=False, threshold=.5)


class TrainIsolationTests(unittest.TestCase):
    def test_explicit_cpu_fp32_does_not_enter_even_disabled_autocast(self):
        with patch("torch.autocast", side_effect=AssertionError("CPU FP32 must not enter autocast")):
            with precision_context("cpu", False):
                value = torch.ones(2) + 1
            self.assertEqual(value.dtype, torch.float32)
        with self.assertRaises(ValueError):
            precision_context("cpu", True)

    def fixture(self, directory):
        base = load_config(ROOT / "runs/phase2_protocol_v1_20260913/base_config.yaml")
        values = copy.deepcopy(base.values)
        values["data"]["root"] = str(Path(directory) / "data")
        base = Config(values, base.source)
        name = "train_1.png"
        files = {role: {"path": str(base.data_root / "train" / folder / name), "sha256": "0" * 64}
                 for role, folder in values["data"]["folders"].items()}
        prepared = {"selected": {"train": [name]}, "records": {"train": [{"name": name, "files": files}]}}
        return base, prepared

    def test_only_fixed_train_paths_are_admitted_without_directory_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            base, prep = self.fixture(directory)
            with patch("PIL.Image.open", side_effect=AssertionError("No raw read during path validation")):
                self.assertEqual(list(validate_train_records(base, prep, expected_count=1)), ["train_1.png"])
                for split in ["val", "test", "Martigny"]:
                    with self.assertRaises(ValueError):
                        validate_train_records(base, prep, split=split, expected_count=1)

    def test_split_alias_and_malicious_prepared_path_rejected_before_raw_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            base, prep = self.fixture(directory)
            base.values["data"]["splits"]["train"] = "val"
            with self.assertRaises(ValueError):
                validate_train_records(base, prep, expected_count=1)
            base, prep = self.fixture(directory)
            for role in prep["records"]["train"][0]["files"]:
                bad = copy.deepcopy(prep)
                bad["records"]["train"][0]["files"][role]["path"] = str(base.data_root / "val" / "A/train_1.png")
                with patch("PIL.Image.open", side_effect=AssertionError("Forbidden raw read")), self.assertRaises(ValueError):
                    validate_train_records(base, bad, expected_count=1)

    def test_changed_whitelist_or_filename_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base, prep = self.fixture(directory)
            prep["selected"]["train"].append("train_1.png")
            with self.assertRaises(ValueError):
                validate_train_records(base, prep, expected_count=1)
            base, prep = self.fixture(directory)
            prep["selected"]["train"][0] = "../val/train_1.png"
            prep["records"]["train"][0]["name"] = prep["selected"]["train"][0]
            with self.assertRaises(ValueError):
                validate_train_records(base, prep, expected_count=1)

    def checkpoint(self):
        base = load_config(ROOT / "runs/phase2_protocol_v1_20260913/base_config.yaml")
        expected = {"mode": "b1", "seed": 42, "epoch": 68}
        cp = {"training_flavor": "matched_robust_baseline_v1", "kind": "research", "robust_mode": "b1",
              "config": copy.deepcopy(base.values), "epoch": 68, "best_epoch": 68,
              "protocol_signature": protocol_signature(base.values), "file_sha256": {"train": {}},
              "model": {"weight": torch.ones(2)}}
        return cp, base, expected

    def test_original_checkpoint_identity_precision_and_finite_state_required(self):
        cp, base, expected = self.checkpoint()
        check_checkpoint_identity(cp, base, expected, {})
        for change in ["flavor", "amp", "state", "epoch"]:
            cp, base, expected = self.checkpoint()
            if change == "flavor": cp["training_flavor"] = "b1_secondary_empirical_tail_v1"
            if change == "amp": cp["config"]["train"]["amp"] = False
            if change == "state": cp["model"]["weight"][0] = float("nan")
            if change == "epoch": cp["best_epoch"] = 67
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_checkpoint_identity(cp, base, expected, {})

    def test_window_dataset_preserves_crop_pixels_and_baseline_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3); b = 255 - a
            records = {"train_1.png": {}}
            for role, array in [("image_a", a), ("image_b", b)]:
                path = Path(directory) / (role + ".png"); Image.fromarray(array).save(path)
                records["train_1.png"][role] = {"path": str(path)}
            ds = WindowPairs(records, unique(), 8, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
            item = ds[1]
            self.assertEqual((item["x"], item["y"]), (4, 0))
            expected = torch.from_numpy(a[:4, 4:8].transpose(2, 0, 1).copy()).float().div_(255)
            expected = (expected - ds.mean) / ds.std
            torch.testing.assert_close(item["image_a"], expected, rtol=0, atol=0)
            self.assertEqual(item["source"], "train_1.png")

    def test_state_digest_detects_parameters_and_buffers(self):
        model = torch.nn.BatchNorm2d(2).eval()
        before = state_digest(model)
        model.running_mean[0] = 1
        self.assertNotEqual(before, state_digest(model))


if __name__ == "__main__":
    unittest.main()
