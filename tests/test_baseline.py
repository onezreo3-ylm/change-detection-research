"""CPU unit/integration fixtures are explicit synthetic engineering checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from pcdc.benchmark import benchmark
from pcdc.config import Config, load_config, protocol_signature, validate_config
from pcdc.data import LEVIRDataset, audit_data, make_loader
from pcdc.engine import check_checkpoint, evaluate, read_checkpoint, run_epoch, train
from pcdc.inspect_labels import inspect_labels
from pcdc.losses import ChangeDetectionLoss
from pcdc.metrics import BinaryMetrics
from pcdc.model import build_model
from pcdc.runtime import setup_runtime
from pcdc.smoke import create_fixture


ROOT = Path(__file__).resolve().parents[1]


def fixture_config(root: Path) -> Config:
    cfg = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
    cfg["runtime"].update(device="cpu", num_threads=2)
    cfg["data"].update(root=str(root / "data"), tile_size=32)
    cfg["experiment"]["output_dir"] = str(root / "run")
    # Explicit CPU/test-sized configuration, never a research fallback.
    validate_config(cfg)
    return Config(cfg, root / "fixture.yaml")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/levir_baseline.yaml")

    def test_local_binding_preserves_protocol(self):
        local = load_config(ROOT / "configs/levir_local.yaml")
        self.assertEqual(protocol_signature(local.values), protocol_signature(self.config.values))
        a, b = copy.deepcopy(local.values), copy.deepcopy(self.config.values)
        a["data"].pop("root")
        b["data"].pop("root")
        self.assertEqual(a, b)

    def test_relative_paths_use_yaml_parent(self):
        self.assertEqual(self.config.data_root, ROOT / "data/LEVIR-CD")

    def test_legacy_config_defaults_to_original_strict_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = copy.deepcopy(self.config.values)
            values["data"].pop("label_decode")
            path = Path(tmp) / "legacy.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
            legacy = load_config(path)
            self.assertEqual(legacy.values["data"]["label_decode"], "strict_values")
            self.assertEqual(protocol_signature(legacy.values), protocol_signature(self.config.values))

            values = copy.deepcopy(self.config.values)
            values["runtime"].pop("memory_format")
            path = Path(tmp) / "legacy_runtime.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
            legacy = load_config(path)
            self.assertEqual(legacy.values["runtime"]["memory_format"], "contiguous")
            self.assertEqual(protocol_signature(legacy.values), protocol_signature(self.config.values))

    def test_duplicate_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("experiment: 1\nexperiment: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    def test_invalid_settings_and_future_modules_rejected(self):
        for section, key, value in (("train", "amp", "false"), ("runtime", "device", "auto"),
                                    ("train", "learning_rate", float("nan")),
                                    ("data", "tile_size", 300), ("evaluation", "threshold", 1),
                                    ("data", "label_decode", "threshold"),
                                    ("runtime", "memory_format", "nhwc"),
                                    ("research", "pse", True), ("model", "encoder_weights", "random")):
            with self.subTest(section=section, key=key):
                cfg = copy.deepcopy(self.config.values)
                cfg[section][key] = value
                with self.assertRaises((ValueError, NotImplementedError)):
                    validate_config(cfg)

    def test_missing_cuda_never_falls_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                setup_runtime(self.config)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = fixture_config(Path(self.temp.name))
        create_fixture(self.config)

    def test_tiling_normalization_and_binary_masks(self):
        ds = LEVIRDataset(self.config, "train")
        self.assertEqual(len(ds), 8)
        sample = ds[0]
        self.assertEqual(sample["image_a"].shape, (3, 32, 32))
        self.assertEqual(sample["mask"].shape, (1, 32, 32))
        self.assertTrue(set(sample["mask"].unique().tolist()) <= {0.0, 1.0})
        raw = np.array(Image.open(ds.records[0].image_a))[:32, :32]
        expected = (torch.tensor(raw.transpose(2, 0, 1), dtype=torch.float32) / 255 - ds.mean) / ds.std
        torch.testing.assert_close(sample["image_a"], expected, rtol=0, atol=0)
        ids = [ds[i]["sample_id"] for i in range(len(ds))]
        self.assertEqual(len(ids), len(set(ids)))

    def test_full_audit_and_heldout_loader(self):
        report = audit_data(self.config, full=True)
        self.assertEqual(report["audit_level"], "decoded_and_sha256")
        self.assertEqual(report["splits"]["val"]["tiles"], 4)
        with self.assertRaises(ValueError):
            make_loader(LEVIRDataset(self.config, "val"), self.config, training=True)

    def test_missing_pair_rejected(self):
        (self.config.data_root / "train/B/train_0.png").rename(self.config.data_root / "train/B/extra.png")
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            LEVIRDataset(self.config, "train")

    def test_unexpected_label_rejected(self):
        path = self.config.data_root / "train/label/train_0.png"
        mask = np.array(Image.open(path))
        mask[0, 0] = 123
        Image.fromarray(mask).save(path)
        with self.assertRaisesRegex(ValueError, "Invalid mask"):
            LEVIRDataset(self.config, "train")[0]

    def test_author_floor255_decode_is_explicit_and_non_mutating(self):
        path = self.config.data_root / "train/label/train_0.png"
        mask = np.array(Image.open(path))
        mask[:2, :2] = np.array([[156, 254], [255, 0]], dtype=np.uint8)
        Image.fromarray(mask).save(path)
        before = path.read_bytes()
        self.config.values["data"]["label_decode"] = "author_uint8_floor255"
        validate_config(self.config.values)
        sample = LEVIRDataset(self.config, "train")[0]["mask"][0]
        self.assertEqual(sample[0, 0].item(), 0)
        self.assertEqual(sample[0, 1].item(), 0)
        self.assertEqual(sample[1, 0].item(), 1)
        self.assertEqual(before, path.read_bytes())
        report = audit_data(self.config, full=True)
        self.assertEqual(report["label_decode"], "author_uint8_floor255")

    def test_inspection_reports_without_remapping_and_training_stops(self):
        path = self.config.data_root / "train/label/train_0.png"
        mask = np.array(Image.open(path))
        mask[0, 0], mask[0, 1] = 156, 254
        Image.fromarray(mask).save(path)
        before = path.read_bytes()
        report = inspect_labels(self.config)
        self.assertEqual(report["splits"]["train"]["invalid_masks"][0]["unexpected_values"],
                         {"156": 1, "254": 1})
        self.assertEqual(before, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Invalid mask"):
            train(self.config)
        self.assertFalse(self.config.output_dir.exists())

    def test_rgb_mask_disagreement_rejected(self):
        path = self.config.data_root / "val/label/val_0.png"
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[0, 0, 0] = 255
        Image.fromarray(mask).save(path)
        with self.assertRaisesRegex(ValueError, "channels disagree"):
            audit_data(self.config, full=True)

    def test_duplicate_temporal_pair_across_splits_rejected(self):
        root = self.config.data_root
        for source, target in (("A", "B"), ("B", "A")):
            with Image.open(root / f"train/{source}/train_0.png") as image:
                image.save(root / f"val/{target}/val_0.png")
        with self.assertRaisesRegex(ValueError, "Duplicate decoded"):
            audit_data(self.config, full=True)

    def test_no_synthetic_relabel_or_overwrite(self):
        with self.assertRaises(FileExistsError):
            create_fixture(self.config)
        self.config.values["experiment"]["kind"] = "research"
        with self.assertRaisesRegex(ValueError, "Synthetic fixture"):
            LEVIRDataset(self.config, "train")

    def test_loader_seed_reproducible(self):
        ds = LEVIRDataset(self.config, "train")
        def order():
            return [sample for batch in make_loader(ds, self.config, training=True) for sample in batch["sample_id"]]
        self.assertEqual(order(), order())


class MathTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Explicit CUDA metric check requires CUDA")
    def test_cuda_metrics_with_strict_determinism(self):
        torch.use_deterministic_algorithms(True)
        meter = BinaryMetrics()
        meter.update(torch.tensor([10., -10., 10., -10.], device="cuda"),
                     torch.tensor([1., 1., 0., 0.], device="cuda"))
        self.assertEqual(meter.counts.tolist(), [1, 1, 1, 1])

    def test_pooled_metrics(self):
        meter = BinaryMetrics()
        meter.update(torch.tensor([10., -10., 10., -10.]), torch.tensor([1., 1., 0., 0.]))
        result = meter.compute()
        self.assertEqual([result[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1])
        self.assertEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        meter.update(torch.full((4,), 10.), torch.ones(4))
        self.assertAlmostEqual(meter.compute()["f1"], 10 / 12)

    def test_empty_negative_threshold_and_nonfinite(self):
        meter = BinaryMetrics()
        with self.assertRaises(ValueError):
            meter.compute()
        meter.update(torch.full((3,), -100.), torch.zeros(3))
        self.assertEqual(meter.compute()["oa"], 1)
        self.assertEqual(meter.compute()["f1"], 0)
        threshold = BinaryMetrics()
        threshold.update(torch.zeros(1), torch.ones(1))
        self.assertEqual(threshold.compute()["tp"], 1)
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([float("nan")]), torch.zeros(1))

    def test_dice_is_per_sample_then_mean_and_backward(self):
        logits = torch.tensor([[[[2., -1.]]], [[[0., 1.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        loss = ChangeDetectionLoss(1., 1., 1.)
        result = loss(logits, target)
        expected = torch.stack([loss(logits[i:i+1], target[i:i+1])["dice"] for i in range(2)]).mean()
        torch.testing.assert_close(result["dice"], expected)
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class ModelTests(unittest.TestCase):
    def test_shared_encoder_symmetry_shapes_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            setup_runtime(config)
            model = build_model(config)
            a, b = torch.randn(2, 3, 32, 47), torch.randn(2, 3, 32, 47)
            initial_bn = model.encoder.stem[1].num_batches_tracked.clone()
            result = model(a, b, return_features=True)
            self.assertEqual(model.encoder.stem[1].num_batches_tracked.item(), initial_bn.item() + 1)
            self.assertEqual(result["logits"].shape, (2, 1, 32, 47))
            self.assertEqual([x.shape[1] for x in result["features_a"]], [64, 128, 256, 512])
            result["logits"].mean().backward()
            self.assertIsNotNone(model.encoder.stem[0].weight.grad)
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(a, b)["logits"], model(b, a)["logits"], rtol=0, atol=0)
                same = model(a, a, return_features=True)
                self.assertTrue(all(torch.count_nonzero(x) == 0 for x in same["differences"]))


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "AMP overflow check requires CUDA")
    def test_amp_overflow_is_logged_and_safely_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            config.values["runtime"].update(device="cuda", memory_format="channels_last")
            config.values["train"]["amp"] = True
            validate_config(config.values)
            device = setup_runtime(config)

            class TinyDetector(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.head = torch.nn.Conv2d(6, 1, 1)

                def forward(self, a, b):
                    return {"logits": self.head(torch.cat((a, b), dim=1))}

            model = TinyDetector().to(device=device, memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=2.0 ** 40,
                                               growth_interval=1000000)
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            batch = {
                "image_a": torch.randn(2, 3, 64, 64),
                "image_b": torch.randn(2, 3, 64, 64),
                "mask": torch.zeros(2, 1, 64, 64),
            }
            result = run_epoch(
                model, [batch], ChangeDetectionLoss(**config.values["loss"]), device, threshold=0.5,
                amp=True, memory_format="channels_last", optimizer=optimizer,
                scaler=scaler, grad_clip_norm=1.0)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["amp_skipped_steps"], 1)
            self.assertLess(scaler.get_scale(), 2.0 ** 40)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_benchmark_uses_real_pipeline_without_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            create_fixture(config)
            (config.data_root / "test").rename(config.data_root / "test_unavailable")
            config.values["runtime"]["memory_format"] = "channels_last"
            output = Path(tmp) / "benchmark.json"
            result = benchmark(config, output, warmup_steps=0, train_steps=1, eval_steps=1)
            self.assertTrue(output.is_file())
            self.assertIn("not_research_results", result["kind"])
            self.assertEqual(result["train"]["samples"], 2)
            self.assertEqual(result["val"]["samples"], 2)

    def test_training_checkpoint_standalone_and_protocol_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            create_fixture(config)
            result = train(config)
            self.assertEqual(result["status"], "completed")
            checkpoint = read_checkpoint(config.output_dir / "best.pt")
            self.assertEqual(checkpoint["kind"], "smoke")
            self.assertEqual(checkpoint["best_val_f1"], checkpoint["history"][0]["val"]["metrics"]["f1"])
            evaluated = evaluate(config, config.output_dir / "best.pt", "test", config.output_dir / "independent.json")
            self.assertEqual(evaluated["metrics"], result["test"])
            with patch("pcdc.engine.file_sha256", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "Changed dataset content"):
                    evaluate(config, config.output_dir / "best.pt", "test", config.output_dir / "bad_data.json")
            with self.assertRaises(FileExistsError):
                train(config)
            config.values["evaluation"]["threshold"] = 0.6
            with self.assertRaisesRegex(ValueError, "protocol differ"):
                check_checkpoint(config, checkpoint)
            config.values["evaluation"]["threshold"] = 0.5
            config.values["runtime"]["memory_format"] = "channels_last"
            with self.assertRaisesRegex(ValueError, "protocol differ"):
                check_checkpoint(config, checkpoint)
            config.values["runtime"]["memory_format"] = "contiguous"
            config.values["experiment"]["kind"] = "research"
            with self.assertRaisesRegex(ValueError, "relabel"):
                check_checkpoint(config, checkpoint)

    def test_exact_epoch_boundary_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            config.values["train"]["epochs"] = 2
            create_fixture(config)
            calls = 0
            def interrupt_second_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt("simulated interruption")
                return run_epoch(*args, **kwargs)
            with patch("pcdc.engine.run_epoch", side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train(config)
            self.assertEqual(json.loads((config.output_dir / "status.json").read_text())["status"], "interrupted")
            resumed = train(config, resume=True)
            weights = read_checkpoint(config.output_dir / "last.pt")["model"]
            config.values["experiment"]["output_dir"] = str(Path(tmp) / "uninterrupted")
            uninterrupted = train(config)
            other = read_checkpoint(config.output_dir / "last.pt")["model"]
            for name in weights:
                torch.testing.assert_close(weights[name], other[name], rtol=0, atol=0)
            self.assertEqual(resumed["test"], uninterrupted["test"])

    @unittest.skipUnless(torch.cuda.is_available(), "Accelerated resume check requires CUDA")
    def test_fast_cuda_exact_epoch_boundary_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture_config(Path(tmp))
            config.values["runtime"].update(device="cuda", memory_format="channels_last")
            config.values["data"]["tile_size"] = 64
            config.values["train"].update(epochs=2, num_workers=2, amp=True)
            validate_config(config.values)
            create_fixture(config)
            calls = 0

            def interrupt_second_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt("simulated accelerated interruption")
                return run_epoch(*args, **kwargs)

            with patch("pcdc.engine.run_epoch", side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train(config)
            resumed = train(config, resume=True)
            weights = read_checkpoint(config.output_dir / "last.pt")["model"]
            config.values["experiment"]["output_dir"] = str(Path(tmp) / "uninterrupted_fast")
            uninterrupted = train(config)
            other = read_checkpoint(config.output_dir / "last.pt")["model"]
            for name in weights:
                torch.testing.assert_close(weights[name], other[name], rtol=0, atol=0)
            self.assertEqual(resumed["test"], uninterrupted["test"])


if __name__ == "__main__":
    unittest.main()
