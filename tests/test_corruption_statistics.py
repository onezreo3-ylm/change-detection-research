"""CPU-only engineering checks for frozen-validation statistics and model inputs."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pcdc.config import Config, load_config, validate_config
from pcdc.evaluate_corruptions import (ConfusionAccumulator, counts_report,
                                       _bound_input_hashes, _bind_validation_files,
                                       _verify_input_hashes,
                                       per_sample_confusions, run_condition,
                                       summarize_conditions)
from pcdc.metrics import BinaryMetrics


ROOT = Path(__file__).resolve().parents[1]


def cpu_config() -> Config:
    values = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
    values["runtime"].update(device="cpu", memory_format="contiguous", num_threads=2)
    values["train"].update(amp=False, num_workers=0)
    values["evaluation"]["run_test_after_training"] = False
    validate_config(values)
    return Config(values, ROOT / "configs/smoke.yaml")


class RGBOnlyDetector(nn.Module):
    """Deliberately tiny test double, never substituted for a research model."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.training_flags = []
        self.gradient_flags = []

    def forward(self, a, b):
        self.calls += 1
        self.training_flags.append(self.training)
        self.gradient_flags.append(torch.is_grad_enabled())
        if a.shape != b.shape or a.shape[1] != 3:
            raise AssertionError("Detector accepts only aligned RGB tensors")
        return {"logits": a[:, :1] * self.scale}


def tiny_batch() -> dict:
    logits = torch.tensor([[[[-1., 1.], [-1., 1.]]], [[[0., -1.], [-1., -1.]]]])
    target = torch.tensor([[[[0., 0.], [1., 1.]]], [[[1., 1.], [0., 0.]]]])
    return {
        "image_a": logits.repeat(1, 3, 1, 1), "image_b": torch.zeros(2, 3, 2, 2),
        "mask": target, "sample_id": ["val/scene@part.png@x0_y0", "val/scene@part.png@x2_y0"],
        "condition_id": ["brightness_l1", "brightness_l1"], "kind": ["brightness", "brightness"],
        "level": torch.tensor([1, 1]), "direction": ["darken", "brighten"],
        "target_date": ["A", "B"], "spatial_mode": ["local", "global"],
        "injection_support": torch.tensor([[[[True, False], [True, False]]],
                                            [[[True, True], [True, True]]]]),
        # Large seeds and other oracle descriptors stay strings, never model arguments.
        "instance_json": ['{"seed":13843553303784611430}', '{"seed":18446744073709551615}'],
    }


def sliced(batch: dict, start: int, end: int) -> dict:
    return {key: value[start:end] for key, value in batch.items()}


def bound_fixture() -> SimpleNamespace:
    """In-memory identities only: no source files, images or weights are created."""
    directory = ROOT / "unused_synthetic_binding_fixture"
    job = SimpleNamespace(source=directory / "job.yaml",
                          path=lambda key: directory / {"acceptance": "missing_review.yaml"}[key])
    protocol = SimpleNamespace(source=directory / "candidate_resolved.yaml",
                               base=SimpleNamespace(source=directory / "base_config.yaml"),
                               path=lambda key: directory / {"reference_checkpoint": "best.pt"}[key])
    fingerprints = {"job_config_sha256": "1" * 64, "manifest_sha256": "2" * 64,
                    "protocol_config_sha256": "3" * 64, "archived_base_config_sha256": "4" * 64,
                    "instances_sha256": "5" * 64, "checkpoint_sha256": "6" * 64,
                    "preparation_environment_sha256": "7" * 64, "acceptance_sha256": None}
    manifest = {"instance_file": "val_instances.jsonl", "protocol_file_sha256": "8" * 64,
                "source_sha256": {"pcdc/model.py": "9" * 64},
                "file_sha256": {"val_1.png": {"image_a": "a" * 64, "image_b": "b" * 64,
                                              "mask": "c" * 64}}}
    return SimpleNamespace(job=job, protocol=protocol, fingerprints=fingerprints,
                           manifest=manifest, manifest_path=directory / "val_manifest.json", approval=None)


class InputBindingTests(unittest.TestCase):
    def test_missing_optional_acceptance_is_not_hashed_or_created(self):
        bundle = bound_fixture()
        with patch("pcdc.evaluate_corruptions.file_sha256", side_effect=AssertionError("No new baseline hashes")):
            bindings = _bound_input_hashes(bundle)
        self.assertNotIn(str(bundle.job.path("acceptance")), bindings)
        self.assertEqual(bindings[str(bundle.job.source)], bundle.fingerprints["job_config_sha256"])
        bundle.approval = {"status": "pending"}
        bundle.fingerprints["acceptance_sha256"] = "d" * 64
        self.assertEqual(_bound_input_hashes(bundle)[str(bundle.job.path("acceptance"))], "d" * 64)

    def test_missing_validated_digest_never_falls_back_to_current_bytes(self):
        bundle = bound_fixture()
        bundle.fingerprints.pop("protocol_config_sha256")
        with patch("pcdc.evaluate_corruptions.file_sha256", return_value="f" * 64) as hashes:
            with self.assertRaisesRegex(ValueError, "validated input hash"):
                _bound_input_hashes(bundle)
        hashes.assert_not_called()

    def test_mutations_after_bundle_validation_are_rejected_not_rebound(self):
        bundle = bound_fixture()
        bindings = _bound_input_hashes(bundle)
        frozen = dict(bindings)
        current = dict(bindings)
        current[str(bundle.protocol.source)] = "f" * 64
        with patch("pcdc.evaluate_corruptions.file_sha256", side_effect=lambda path: current[str(path)]):
            with self.assertRaisesRegex(ValueError, "Frozen input content changed"):
                _verify_input_hashes(bindings)
        self.assertEqual(bindings, frozen)

    def test_all_validation_rgb_and_label_paths_stay_bound_through_completion(self):
        bundle = bound_fixture()
        directory = bundle.manifest_path.parent / "data" / "val"
        record = SimpleNamespace(name="val_1.png", image_a=directory / "A" / "val_1.png",
                                 image_b=directory / "B" / "val_1.png", mask=directory / "label" / "val_1.png")
        dataset = SimpleNamespace(split="val", records=[record])
        checkpoint = {"file_sha256": {"val": copy.deepcopy(bundle.manifest["file_sha256"])}}
        bindings = _bound_input_hashes(bundle)
        before = len(bindings)
        with patch("pcdc.evaluate_corruptions.file_sha256", side_effect=AssertionError("Do not recapture image hashes")):
            _bind_validation_files(bindings, bundle, dataset, checkpoint)
        self.assertEqual(len(bindings) - before, 3)
        for key, expected in bundle.manifest["file_sha256"][record.name].items():
            self.assertEqual(bindings[str(getattr(record, key))], expected)
        current = dict(bindings)
        with patch("pcdc.evaluate_corruptions.file_sha256", side_effect=lambda path: current[str(path)]):
            _verify_input_hashes(bindings)
            # Simulate mutation after an early successful audit but before the
            # lazy condition readers finish: final verification must still fail.
            current[str(record.mask)] = "e" * 64
            with self.assertRaisesRegex(ValueError, "Frozen input content changed"):
                _verify_input_hashes(bindings)

    def test_manifest_checkpoint_mismatch_and_non_val_files_rejected(self):
        bundle = bound_fixture()
        dataset = SimpleNamespace(split="val", records=[])
        checkpoint = {"file_sha256": {"val": copy.deepcopy(bundle.manifest["file_sha256"])}}
        checkpoint["file_sha256"]["val"]["val_1.png"]["mask"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identities differ"):
            _bind_validation_files({}, bundle, dataset, checkpoint)
        dataset.split = "test"
        with self.assertRaisesRegex(ValueError, "Only validation"):
            _bind_validation_files({}, bundle, dataset, checkpoint)


class ConfusionTests(unittest.TestCase):
    def test_metrics_match_baseline_and_expose_foreground_background_denominators(self):
        meter = BinaryMetrics()
        meter.counts += torch.tensor([3, 1, 2, 4], dtype=torch.int64)
        expected = meter.compute()
        report = counts_report([3, 1, 2, 4])
        for key in ("precision", "recall", "f1", "iou", "oa"):
            self.assertEqual(report["metrics"][key], expected[key])
        self.assertEqual(report["counts"], {key: expected[key] for key in ("tn", "fp", "fn", "tp")})
        self.assertEqual(report["foreground_pixels"], 6)
        self.assertEqual(report["background_pixels"], 4)
        self.assertEqual(report["metrics"]["background_fpr"], 1 / 4)
        self.assertEqual(report["metrics"]["foreground_fn_rate"], 2 / 6)

    def test_empty_regions_are_not_measured_zero_scores(self):
        self.assertIsNone(counts_report([0, 0, 0, 0])["metrics"])
        negative = counts_report([10, 0, 0, 0])
        self.assertEqual(negative["pixels"], 10)
        self.assertEqual(negative["foreground_pixels"], 0)
        self.assertEqual(negative["metrics"]["oa"], 1.0)
        self.assertEqual(negative["metrics"]["f1"], 0.0)
        self.assertEqual(negative["metrics"]["foreground_fn_rate"], 0.0)

    def test_invalid_counts_rejected(self):
        for counts in ([0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 1.0], [0, 0, 0, True]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                counts_report(counts)

    def test_per_sample_confusions_threshold_and_region_partition(self):
        batch = tiny_batch()
        logits, target = batch["image_a"][:, :1], batch["mask"]
        counts, inside = per_sample_confusions(logits, target, threshold=0.5,
                                              injection_support=batch["injection_support"])
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(counts.tolist(), [[1, 1, 1, 1], [2, 0, 1, 1]])
        self.assertEqual(inside.tolist(), [[1, 0, 1, 0], [2, 0, 1, 1]])
        meter = BinaryMetrics()
        meter.update(logits, target)
        self.assertEqual(counts.sum(0).tolist(), meter.counts.tolist())
        total = ConfusionAccumulator(regional=True)
        total.add(counts[1].tolist(), inside[1].tolist())
        outside = total.report()["regions"]["outside_injection"]
        self.assertEqual(outside["pixels"], 0)
        self.assertIsNone(outside["metrics"])
        with self.assertRaises(ValueError):
            total.add([0, 0, 0, 0], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            total.add([1, 0, 0, 0])

    def test_nonfinite_and_misaligned_metric_inputs_rejected(self):
        batch = tiny_batch()
        logits, target = batch["image_a"][:, :1], batch["mask"]
        for bad_logits, bad_target, support in (
                (torch.full_like(logits, float("nan")), target, None),
                (logits, target + 0.5, None), (logits[:, 0], target, None),
                (logits, target, batch["injection_support"].float()),
                (logits, target, batch["injection_support"][:, 0])):
            with self.subTest(shape=bad_logits.shape), self.assertRaises(ValueError):
                per_sample_confusions(bad_logits, bad_target, threshold=0.5, injection_support=support)

    def test_equal_cell_average_is_not_pooled_f1(self):
        a, b = ConfusionAccumulator(), ConfusionAccumulator()
        a.add([0, 0, 0, 100])
        b.add([0, 1, 0, 0])
        cells = {"brightness_l1": {"kind": "brightness", **a.report()},
                 "noise_l1": {"kind": "noise", **b.report()}}
        result = summarize_conditions(cells, list(cells))
        self.assertEqual(result["corruption_mf1"], 0.5)
        self.assertNotEqual(result["corruption_mf1"], counts_report([0, 1, 0, 100])["metrics"]["f1"])
        with self.assertRaises(ValueError):
            summarize_conditions(cells, ["brightness_l1"])
        with self.assertRaises(ValueError):
            summarize_conditions(cells, ["brightness_l1", "brightness_l1"])


class ConditionRunnerTests(unittest.TestCase):
    def test_strata_marginals_regions_and_original_source_counts(self):
        model = RGBOnlyDetector()
        before = model.scale.detach().clone()
        checks = []
        batch = tiny_batch()
        result = run_condition(model, [batch], cpu_config(), torch.device("cpu"),
                               guard_callback=lambda: checks.append("checked"))
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.training_flags, [False])
        self.assertEqual(model.gradient_flags, [False])
        self.assertIsNone(model.scale.grad)
        torch.testing.assert_close(before, model.scale, rtol=0, atol=0)
        self.assertEqual(len(checks), 2)
        self.assertEqual(result["counts"], {"tn": 3, "fp": 1, "fn": 2, "tp": 2})
        self.assertEqual(set(result["by_source"]), {"val/scene@part.png"})
        self.assertEqual(result["by_source"]["val/scene@part.png"]["counts"], result["counts"])
        self.assertEqual(set(result["joint_strata"]), {"darken|A|local", "brighten|B|global"})
        for key in ("direction", "target_date", "spatial_mode"):
            groups = result["marginals"][key]
            self.assertEqual(sum(value["samples"] for value in groups.values()), 2)
            for count_key in ("tn", "fp", "fn", "tp"):
                self.assertEqual(sum(value["counts"][count_key] for value in groups.values()),
                                 result["counts"][count_key])
        inside = result["regions"]["injection_support"]
        outside = result["regions"]["outside_injection"]
        self.assertEqual(inside["pixels"] + outside["pixels"], result["pixels"])

    def test_batch_partition_does_not_change_any_aggregation(self):
        batch = tiny_batch()
        together = run_condition(RGBOnlyDetector(), [batch], cpu_config(), torch.device("cpu"))
        separate = run_condition(RGBOnlyDetector(), [sliced(batch, 0, 1), sliced(batch, 1, 2)],
                                 cpu_config(), torch.device("cpu"))
        together.pop("seconds")
        separate.pop("seconds")
        self.assertEqual(together, separate)

    def test_clean_path_has_no_injection_regions_or_strata(self):
        source = tiny_batch()
        batch = {key: source[key] for key in ("image_a", "image_b", "mask", "sample_id")}
        result = run_condition(RGBOnlyDetector(), [batch], cpu_config(), torch.device("cpu"))
        self.assertEqual(result["condition_id"], "clean")
        self.assertEqual(result["joint_strata"], {})
        self.assertIsNone(result["regions"])
        meter = BinaryMetrics()
        meter.update(batch["image_a"][:, :1], batch["mask"])
        self.assertEqual(result["metrics"]["f1"], meter.compute()["f1"])

    def test_invalid_or_mixed_metadata_rejected_before_model_forward(self):
        for key, value in (("condition_id", ["brightness_l1", "brightness_l2"]),
                           ("sample_id", ["val/a.png@x0_y0", "val/a.png@x0_y0"]),
                           ("sample_id", ["val/a.png@x0_y0", "test/a.png@x0_y0"]),
                           ("target_date", ["A", "both"]), ("level", torch.tensor([True, True]))):
            with self.subTest(key=key):
                batch = tiny_batch()
                batch[key] = value
                model = RGBOnlyDetector()
                with self.assertRaises(ValueError):
                    run_condition(model, [batch], cpu_config(), torch.device("cpu"))
                self.assertEqual(model.calls, 0)

    def test_empty_loader_and_guard_failure_never_forward(self):
        model = RGBOnlyDetector()
        with self.assertRaisesRegex(ValueError, "empty condition"):
            run_condition(model, [], cpu_config(), torch.device("cpu"))

        def abort():
            raise RuntimeError("synthetic safety guard stop")

        with self.assertRaisesRegex(RuntimeError, "safety guard"):
            run_condition(model, [tiny_batch()], cpu_config(), torch.device("cpu"), guard_callback=abort)
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
