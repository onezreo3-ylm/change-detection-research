"""Synthetic checks for matched B0-E/B1/B2 training; never research results."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from pcdc.config import Config, load_config, validate_config
from pcdc.corruption_protocol import apply_instance, canonical, load_protocol
from pcdc.data import LEVIRDataset, make_loader
from pcdc.engine import read_checkpoint
from pcdc.robust_baselines import (
    TwoViewTrainingDataset,
    load_robust_config,
    preflight,
    run_two_view_epoch,
    train_robust,
)
from pcdc.smoke import create_fixture


ROOT = Path(__file__).resolve().parents[1]


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(6)
        self.head = nn.Conv2d(6, 1, 1)

    def forward(self, a, b):
        return {"logits": self.head(self.bn(torch.cat((a, b), dim=1)))}


class RobustFixture:
    def __init__(self, root: Path, *, epochs: int = 1):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        base = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
        base["runtime"].update(device="cpu", num_threads=2, memory_format="contiguous")
        base["data"].update(
            root=str(root / "data"), expected_pairs={"train": 4, "val": 2, "test": 2},
            expected_image_size=64, tile_size=64)
        base["experiment"].update(name="robust_unit_fixture", output_dir=str(root / "unused"))
        base["train"].update(epochs=epochs, batch_size=2, eval_batch_size=2,
                             num_workers=0, amp=False)
        base["evaluation"]["run_test_after_training"] = False
        validate_config(base)
        self.base_path = root / "base.yaml"
        self.base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        self.base = Config(base, self.base_path)

        protocol = yaml.safe_load((ROOT / "configs/corruption_protocol_v1.yaml").read_text(
            encoding="utf-8"))
        protocol.update(
            name="robust_unit_corruptions", base_config=str(self.base_path),
            reference_checkpoint=str(root / "unused.pt"),
            reference_checkpoint_sha256="0" * 64,
            reference_audit=str(root / "unused_audit.json"),
            output_dir=str(root / "protocol_output"))
        self.protocol_path = root / "protocol.yaml"
        self.protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
        self.protocol = load_protocol(self.protocol_path)

        review = {
            "schema_version": 1, "status": "engineering_only",
            "approved_for": "validation_only",
            "protocol_sha256": self.protocol.fingerprint,
            "manifest_sha256": "1" * 64, "instances_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
            "visual_review": "not_applicable_synthetic",
            "reviewer": "automated synthetic unit fixture",
            "notes": "Synthetic engineering authorization only; no research severity claim.",
        }
        self.review_path = root / "review.yaml"
        self.review_path.write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")

    def config(self, mode: str, output: str | None = None):
        secondary = "clean" if mode == "b0e" else "perturbed"
        consistency = "probability_mse" if mode == "b2" else "none"
        weight = 1.0 if mode == "b2" else 0.0
        value = {
            "schema_version": 1, "name": f"fixture_{mode}", "mode": mode,
            "base_config": str(self.base_path), "protocol_config": str(self.protocol_path),
            "protocol_sha256": self.protocol.fingerprint,
            "severity_review": str(self.review_path),
            "output_dir": str(self.root / (output or mode)),
            "expected_source_tiles": 4, "expected_batches_per_epoch": 2,
            "views": {
                "primary": "clean", "secondary": secondary, "views_per_source": 2,
                "concat_order": ["clean", "secondary"],
                "optimizer_steps_per_source_batch": 1,
                "batchnorm_update": "single_forward_clean_then_secondary_then_temporal_concat",
            },
            "corruption": {
                "stream": "train", "view": 0,
                "selection": "uniform_family_level_direction_date_spatial_per_sample_epoch_view",
            },
            "consistency": {
                "kind": consistency, "weight": weight, "space": "probability",
                "gradient_flow": "symmetric",
            },
            "selection": "clean_val_f1",
        }
        path = self.root / f"{output or mode}.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return load_robust_config(path), path


class RobustConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = RobustFixture(Path(self.temp.name))

    def test_preflight_is_metadata_only_and_strict(self):
        config, path = self.fixture.config("b1")
        with patch("pcdc.data.LEVIRDataset", side_effect=AssertionError("dataset opened")), \
                patch("pcdc.robust_baselines.build_model", side_effect=AssertionError("model built")):
            result = preflight(config)
        self.assertEqual(result["status"], "robust_preflight_passed")
        self.assertFalse(result["test_access"])
        self.assertFalse(config.output_dir.exists())

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["consistency"]["weight"] = 0.5
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires consistency"):
            load_robust_config(path)

    def test_pending_or_research_mislabeled_review_is_rejected(self):
        _, path = self.fixture.config("b1")
        review = yaml.safe_load(self.fixture.review_path.read_text(encoding="utf-8"))
        review.update(status="pending", visual_review="not_completed")
        self.fixture.review_path.write_text(yaml.safe_dump(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected severity review"):
            load_robust_config(path)


class TwoViewDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = RobustFixture(Path(self.temp.name))
        create_fixture(self.fixture.base)
        self.base = LEVIRDataset(self.fixture.base, "train")

    def test_perturbed_view_is_stateless_and_clean_view_matches_baseline(self):
        dataset = TwoViewTrainingDataset(self.base, self.fixture.protocol, "perturbed")
        dataset.set_epoch(3)
        first, repeat = dataset[0], dataset[0]
        torch.testing.assert_close(first["clean_a"], self.base[0]["image_a"], rtol=0, atol=0)
        torch.testing.assert_close(first["mask"], self.base[0]["mask"], rtol=0, atol=0)
        self.assertEqual(first["instance_json"], repeat["instance_json"])
        torch.testing.assert_close(first["secondary_a"], repeat["secondary_a"], rtol=0, atol=0)
        torch.testing.assert_close(first["secondary_b"], repeat["secondary_b"], rtol=0, atol=0)
        instance = json.loads(first["instance_json"])
        self.assertEqual((instance["stream"], instance["epoch"], instance["view"]),
                         ("train", 3, 0))
        dataset.set_epoch(4)
        self.assertNotEqual(first["instance_json"], dataset[0]["instance_json"])

    def test_clean_secondary_is_an_exact_value_copy(self):
        dataset = TwoViewTrainingDataset(self.base, self.fixture.protocol, "clean")
        sample = dataset[0]
        torch.testing.assert_close(sample["clean_a"], sample["secondary_a"], rtol=0, atol=0)
        torch.testing.assert_close(sample["clean_b"], sample["secondary_b"], rtol=0, atol=0)
        self.assertEqual(sample["instance_json"], "clean")

    def test_loader_schedule_is_repeatable_for_same_seed_epoch(self):
        def collect():
            dataset = TwoViewTrainingDataset(self.base, self.fixture.protocol, "perturbed")
            dataset.set_epoch(7)
            return [(sample_id, instance) for batch in make_loader(
                dataset, self.fixture.base, training=True)
                    for sample_id, instance in zip(batch["sample_id"], batch["instance_json"])]
        self.assertEqual(collect(), collect())


class RobustTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = RobustFixture(Path(self.temp.name))
        create_fixture(self.fixture.base)

    @staticmethod
    def _tiny_model(*args, **kwargs):
        return TinyDetector()

    def test_one_forward_is_one_bn_update_and_b2_has_consistency(self):
        model = TinyDetector()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        batch = {
            "clean_a": torch.randn(2, 3, 16, 16),
            "clean_b": torch.randn(2, 3, 16, 16),
            "secondary_a": torch.randn(2, 3, 16, 16),
            "secondary_b": torch.randn(2, 3, 16, 16),
            "mask": torch.zeros(2, 1, 16, 16),
            "sample_id": ["train/a", "train/b"],
            "secondary_kind": ["noise", "blur"],
            "secondary_level": torch.tensor([1, 2]),
            "secondary_direction": ["neutral", "neutral"],
            "secondary_target_date": ["A", "B"],
            "secondary_spatial_mode": ["global", "local"],
            "instance_json": [canonical({"a": 1}), canonical({"b": 2})],
        }
        before = model.bn.num_batches_tracked.item()
        from pcdc.losses import ChangeDetectionLoss
        result = run_two_view_epoch(
            model, [batch], ChangeDetectionLoss(1, 1, 1), torch.device("cpu"),
            threshold=0.5, consistency_kind="probability_mse", consistency_weight=1.0,
            amp=False, memory_format="contiguous", optimizer=optimizer, scaler=scaler,
            grad_clip_norm=1.0)
        self.assertEqual(model.bn.num_batches_tracked.item(), before + 1)
        self.assertGreater(result["loss"]["consistency"], 0)
        self.assertEqual((result["source_samples"], result["pair_views"],
                          result["optimizer_steps"]), (2, 4, 1))

    def test_b1_b2_match_initialization_schedule_steps_and_test_closure(self):
        b1, _ = self.fixture.config("b1", "run_b1")
        b2, _ = self.fixture.config("b2", "run_b2")
        with patch("pcdc.robust_baselines.build_model", side_effect=self._tiny_model):
            first = train_robust(b1)
            second = train_robust(b2)
        self.assertEqual(first["initial_model_tensor_sha256"],
                         second["initial_model_tensor_sha256"])
        self.assertEqual(first["optimizer_steps"], second["optimizer_steps"])
        h1 = json.loads((b1.output_dir / "history.json").read_text())
        h2 = json.loads((b2.output_dir / "history.json").read_text())
        self.assertEqual(h1[0]["train"]["exposure_counts"],
                         h2[0]["train"]["exposure_counts"])
        self.assertEqual(h1[0]["train"]["exposure_stream_sha256"],
                         h2[0]["train"]["exposure_stream_sha256"])
        self.assertGreater(h2[0]["train"]["loss"]["consistency"], 0)
        self.assertFalse((self.fixture.base.data_root / "test").exists())
        checkpoint = read_checkpoint(b2.output_dir / "best.pt")
        self.assertEqual(checkpoint["robust_mode"], "b2")
        self.assertNotIn("test", checkpoint["manifests"])

    def test_epoch_boundary_resume_is_exact(self):
        fixture = RobustFixture(Path(self.temp.name) / "resume_fixture", epochs=2)
        create_fixture(fixture.base)
        interrupted, _ = fixture.config("b1", "interrupted")
        calls = 0

        def callback():
            nonlocal calls
            calls += 1
            # Initial call + two train batches + pre/post validation = five calls in epoch 1.
            if calls == 6:
                raise KeyboardInterrupt("synthetic epoch-2 boundary")

        with patch("pcdc.robust_baselines.build_model", side_effect=self._tiny_model):
            with self.assertRaises(KeyboardInterrupt):
                train_robust(interrupted, guard_callback=callback)
            resumed = train_robust(interrupted, resume=True)
            resumed_weights = read_checkpoint(interrupted.output_dir / "last.pt")["model"]
            uninterrupted, _ = fixture.config("b1", "uninterrupted")
            complete = train_robust(uninterrupted)
            complete_weights = read_checkpoint(uninterrupted.output_dir / "last.pt")["model"]
        self.assertEqual(resumed["best_val_f1"], complete["best_val_f1"])
        for name in resumed_weights:
            torch.testing.assert_close(resumed_weights[name], complete_weights[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
