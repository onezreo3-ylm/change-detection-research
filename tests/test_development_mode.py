"""Explicit CPU/synthetic engineering checks for the train/val-only lifecycle."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from pcdc.config import Config, load_config, protocol_signature, validate_config
from pcdc.data import LEVIRDataset, audit_data
from pcdc.engine import (check_checkpoint, evaluate, read_checkpoint, train,
                         training_signature)
from pcdc.smoke import create_fixture, main as smoke_main


ROOT = Path(__file__).resolve().parents[1]


def cpu_fixture_config(root: Path, *, include_test: bool = False) -> Config:
    """CPU is intentional for this synthetic test, never a research fallback."""
    values = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
    values["runtime"].update(device="cpu", num_threads=2, memory_format="contiguous")
    values["train"].update(amp=False, epochs=1, num_workers=0)
    values["data"].update(root=str(root / "data"), tile_size=64)
    values["experiment"]["output_dir"] = str(root / "run")
    values["evaluation"]["run_test_after_training"] = include_test
    validate_config(values)
    return Config(values, root / "fixture.yaml")


class DevelopmentConfigTests(unittest.TestCase):
    def test_legacy_default_and_exact_signature_compatibility(self):
        path = ROOT / "configs/levir_local_author_floor255_fast_stable_b4.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertNotIn("run_test_after_training", raw["evaluation"])
        legacy = load_config(path)
        self.assertIs(legacy.run_test_after_training, True)
        self.assertEqual(
            protocol_signature(legacy.values),
            "245f12b4a85fd2c2cf4fe8c9e709eabe56cda200c806d053aac08bc7fb982581")
        self.assertEqual(protocol_signature(legacy.values), protocol_signature(raw))
        # Reproduce the pre-change strict training signature algorithm without
        # calling the updated helper, so default insertion cannot change it.
        previous = copy.deepcopy(raw)
        previous["data"].pop("root")
        previous["model"].pop("weights_cache")
        previous["experiment"].pop("output_dir")
        expected = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
        self.assertEqual(training_signature(legacy), expected)
        development = Config(copy.deepcopy(legacy.values), legacy.source)
        development.values["evaluation"]["run_test_after_training"] = False
        self.assertEqual(protocol_signature(development.values), protocol_signature(legacy.values))
        self.assertNotEqual(training_signature(development), training_signature(legacy))

    def test_test_lifecycle_setting_requires_a_boolean(self):
        values = load_config(ROOT / "configs/levir_development_seed42.yaml").values
        for invalid in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=invalid):
                candidate = copy.deepcopy(values)
                candidate["evaluation"]["run_test_after_training"] = invalid
                with self.assertRaisesRegex(ValueError, "run_test_after_training must be boolean"):
                    validate_config(candidate)
        for enabled in (False, True):
            candidate = copy.deepcopy(values)
            candidate["evaluation"]["run_test_after_training"] = enabled
            validate_config(candidate)

    def test_development_yaml_preserves_stable_training_protocol(self):
        baseline = load_config(ROOT / "configs/levir_local_author_floor255_fast_stable_b4.yaml")
        development = load_config(ROOT / "configs/levir_development_seed42.yaml")
        self.assertIs(development.run_test_after_training, False)
        self.assertEqual(baseline.data_root, development.data_root)
        self.assertNotEqual(baseline.output_dir, development.output_dir)
        a, b = copy.deepcopy(baseline.values), copy.deepcopy(development.values)
        for values in (a, b):
            values["experiment"].pop("name")
            values["experiment"].pop("output_dir")
            values["evaluation"].pop("run_test_after_training")
        self.assertEqual(a, b)


class DevelopmentAccessTests(unittest.TestCase):
    def test_standalone_test_is_rejected_before_io_or_device_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = cpu_fixture_config(Path(tmp))
            with patch("pcdc.engine.setup_runtime") as setup, \
                    patch("pcdc.engine.read_checkpoint") as checkpoint, \
                    patch("pcdc.engine._evaluate_model") as inference, \
                    patch("pathlib.Path.exists", side_effect=AssertionError("Unexpected path access")):
                with self.assertRaisesRegex(ValueError, "Test access is disabled"):
                    evaluate(config, Path(tmp) / "missing.pt", "test", Path(tmp) / "forbidden.json")
            setup.assert_not_called()
            checkpoint.assert_not_called()
            inference.assert_not_called()
            self.assertFalse((Path(tmp) / "forbidden.json").exists())

    def test_dataset_guard_blocks_test_before_file_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = cpu_fixture_config(Path(tmp))
            with patch("pathlib.Path.exists", side_effect=AssertionError("Unexpected root access")), \
                    patch("pcdc.data._file_map") as files:
                with self.assertRaisesRegex(ValueError, "Test access is disabled"):
                    LEVIRDataset(config, "test")
            files.assert_not_called()

    def test_fixture_and_full_audit_use_only_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = cpu_fixture_config(Path(tmp))
            create_fixture(config)
            self.assertFalse((config.data_root / "test").exists())
            with patch("pcdc.data.LEVIRDataset", wraps=LEVIRDataset) as dataset:
                report = audit_data(config, full=True)
            self.assertEqual([call.args[1] for call in dataset.call_args_list], ["train", "val"])
            self.assertEqual(set(report["splits"]), {"train", "val"})
            self.assertEqual(report["audit_level"], "decoded_and_sha256")
            self.assertTrue(all("file_sha256" in entry for entry in report["splits"].values()))
            with self.assertRaisesRegex(ValueError, "Test access is disabled"):
                audit_data(config, full=True, splits=("test",))

    def test_resume_cannot_change_test_policy_before_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for original_policy in (False, True):
                with self.subTest(original_policy=original_policy):
                    original = cpu_fixture_config(Path(tmp), include_test=original_policy)
                    checkpoint = {"protocol_signature": protocol_signature(original.values),
                                  "training_signature": training_signature(original), "kind": "smoke"}
                    changed = Config(copy.deepcopy(original.values), original.source)
                    changed.values["evaluation"]["run_test_after_training"] = not original_policy
                    with patch("pcdc.engine.read_checkpoint", return_value=checkpoint), \
                            patch("pcdc.engine.setup_runtime") as setup, \
                            patch("pcdc.engine.audit_data") as audit:
                        with self.assertRaisesRegex(ValueError, "Resume must preserve"):
                            train(changed, resume=True)
                    setup.assert_not_called()
                    audit.assert_not_called()

    def test_enabled_config_cannot_silently_bypass_missing_test_audit(self):
        from pcdc.engine import _evaluate_model

        with tempfile.TemporaryDirectory() as tmp:
            config = cpu_fixture_config(Path(tmp), include_test=True)
            checkpoint = {"protocol_signature": protocol_signature(config.values), "kind": "smoke",
                          "manifests": {"train": {}, "val": {}},
                          "file_sha256": {"train": {}, "val": {}}}
            check_checkpoint(config, checkpoint)
            with patch("pcdc.engine.build_datasets") as datasets:
                with self.assertRaisesRegex(ValueError, "separately reviewed, explicit audited protocol"):
                    _evaluate_model(config, checkpoint, torch.device("cpu"), "test")
            datasets.assert_not_called()


class DevelopmentIntegrationTests(unittest.TestCase):
    def test_cpu_train_without_test_matches_legacy_weights_and_val_reload(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            development = cpu_fixture_config(root / "development")
            legacy = cpu_fixture_config(root / "legacy", include_test=True)
            create_fixture(development)
            create_fixture(legacy)
            self.assertFalse((development.data_root / "test").exists())
            # Prove train() cannot perform any training-end evaluation in dev mode.
            with patch("pcdc.engine._evaluate_model", side_effect=AssertionError("Unexpected evaluation")) as evaluated:
                dev_result = train(development)
            evaluated.assert_not_called()
            self.assertEqual(dev_result["status"], "completed")
            self.assertIs(dev_result["test_evaluated"], False)
            self.assertEqual(dev_result["selection"], "validation_f1_only")
            self.assertNotIn("test", dev_result)
            self.assertFalse((development.output_dir / "test_metrics.json").exists())
            saved_status = json.loads((development.output_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_status, dev_result)
            dev_checkpoint = read_checkpoint(development.output_dir / "last.pt")
            self.assertEqual(set(dev_checkpoint["manifests"]), {"train", "val"})
            self.assertEqual(set(dev_checkpoint["file_sha256"]), {"train", "val"})
            best = read_checkpoint(development.output_dir / "best.pt")
            val = evaluate(development, development.output_dir / "best.pt", "val",
                           development.output_dir / "independent_val.json")
            self.assertEqual(val["metrics"], best["history"][-1]["val"]["metrics"])
            legacy_result = train(legacy)
            self.assertIn("test", legacy_result)
            self.assertNotIn("test_evaluated", legacy_result)
            self.assertTrue((legacy.output_dir / "test_metrics.json").exists())
            legacy_checkpoint = read_checkpoint(legacy.output_dir / "last.pt")
            for name, weights in dev_checkpoint["model"].items():
                torch.testing.assert_close(weights, legacy_checkpoint["model"][name], rtol=0, atol=0)
            self.assertEqual(dev_checkpoint["history"][0]["train"]["loss"],
                             legacy_checkpoint["history"][0]["train"]["loss"])
            self.assertEqual(dev_checkpoint["history"][0]["val"]["metrics"],
                             legacy_checkpoint["history"][0]["val"]["metrics"])

    def test_smoke_development_independently_evaluates_val_only(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = cpu_fixture_config(Path(tmp))
            config.source.write_text(yaml.safe_dump(config.values, sort_keys=False), encoding="utf-8")
            with patch("sys.argv", ["pcdc.smoke", "--config", str(config.source)]), \
                    patch("pcdc.smoke.evaluate", wraps=evaluate) as independent:
                smoke_main()
            self.assertEqual(independent.call_count, 1)
            self.assertEqual(independent.call_args.args[2], "val")
            self.assertFalse((config.data_root / "test").exists())
            self.assertTrue((config.output_dir / "standalone_val.json").exists())
            self.assertFalse((config.output_dir / "standalone_test.json").exists())
            self.assertFalse((config.output_dir / "test_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
