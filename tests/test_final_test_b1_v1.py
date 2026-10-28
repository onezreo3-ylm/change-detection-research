"""Synthetic-only checks for the separate, sealed final-test evaluator."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from pcdc.config import Config, load_config
from pcdc.data import LEVIRDataset, RawPair, file_sha256
from pcdc.smoke import create_fixture
from scripts.final_test_b1_v1 import (Job, TestInstanceDataset, audit_test_data,
                                       bootstrap_sources, run, score_condition,
                                       test_instances)


ROOT = Path(__file__).resolve().parents[1]


class FinalTestInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "configs" / "corruption_protocol_v1.yaml").open(encoding="utf-8") as stream:
            cls.protocol = SimpleNamespace(values=yaml.safe_load(stream))

    def test_test_stream_is_deterministic_and_level_paired(self) -> None:
        ids = [f"test/synthetic_{index:02d}.png@x0_y0" for index in range(16)]
        first = test_instances(self.protocol, ids, 310944)
        self.assertEqual(first, test_instances(self.protocol, ids, 310944))
        self.assertEqual(len(first), 45 * 16)
        for kind in ("brightness", "noise"):
            rows = [row for row in first if row["kind"] == kind]
            self.assertEqual({row["condition_id"] for row in rows},
                             {f"{kind}_l{level}" for level in range(1, 6)})
            for sample_id in ids:
                curve = [row for row in rows if row["sample_id"] == sample_id]
                self.assertEqual(len(curve), 5)
                self.assertEqual(len({(row["seed"], row["direction"], row["target_date"],
                                       row["spatial_mode"]) for row in curve}), 1)
                self.assertTrue(all(row["stream"] == "test" for row in curve))

    def test_rejects_val_id_or_wrong_seed(self) -> None:
        with self.assertRaises(ValueError):
            test_instances(self.protocol, ["val/not_test.png@x0_y0"], 310944)
        with self.assertRaises(ValueError):
            test_instances(self.protocol, ["test/a.png@x0_y0"], 310943)

    def test_corruption_replay_keeps_binary_mask_and_test_id(self) -> None:
        class MemoryTest:
            split = "test"
            pair_transform = None
            tile_size = 32
            mean = torch.zeros(3, 1, 1)
            std = torch.ones(3, 1, 1)
            records = [SimpleNamespace(name="synthetic.png")]
            tiles = [(0, 0, 0)]

            def read_pair(self, record, box=None):
                a = np.full((32, 32, 3), 64, dtype=np.uint8)
                b = np.full((32, 32, 3), 96, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[5:10, 5:10] = 1
                return RawPair(a, b, mask)

        instance = next(row for row in test_instances(
            self.protocol, ["test/synthetic.png@x0_y0"], 310944)
            if row["condition_id"] == "brightness_l1")
        sample = TestInstanceDataset(MemoryTest(), [instance])[0]
        self.assertEqual(sample["sample_id"], "test/synthetic.png@x0_y0")
        self.assertEqual(int(sample["mask"].sum()), 25)
        self.assertEqual(tuple(sample["image_a"].shape), (3, 32, 32))
        self.assertEqual(sample["injection_support"].dtype, torch.bool)


class FinalTestScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_config(ROOT / "runs" / "phase2_protocol_v1_20260913" / "base_config.yaml")
        values = copy.deepcopy(base.values)
        values["runtime"]["device"] = "cpu"
        values["train"]["amp"] = False
        values["evaluation"]["run_test_after_training"] = True
        self.config = Config(values, base.source)

    def test_pooled_test_scoring_and_rejects_val_id(self) -> None:
        class FixedModel(nn.Module):
            def forward(self, a, b):
                return {"logits": torch.zeros((a.shape[0], 1, 32, 32))}

        class Samples:
            def __init__(self, ids):
                self.ids = ids

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, index):
                mask = torch.zeros((1, 32, 32))
                if index == 0:
                    mask[:, 0, 0] = 1
                return {"image_a": torch.zeros((3, 32, 32)),
                        "image_b": torch.zeros((3, 32, 32)),
                        "mask": mask, "sample_id": self.ids[index]}

        ids = ["test/synthetic_a.png@x0_y0", "test/synthetic_b.png@x0_y0"]
        scored = score_condition(FixedModel(), DataLoader(Samples(ids), batch_size=2),
                                 self.config, torch.device("cpu"),
                                 expected_ids=ids, condition_id="clean")
        self.assertEqual(scored["counts"], {"tn": 0, "fp": 2047, "fn": 0, "tp": 1})
        self.assertEqual(set(scored["by_source"]),
                         {"test/synthetic_a.png", "test/synthetic_b.png"})
        with self.assertRaises(ValueError):
            score_condition(FixedModel(), DataLoader(Samples(["val/leak.png@x0_y0"])),
                            self.config, torch.device("cpu"),
                            expected_ids=["val/leak.png@x0_y0"], condition_id="clean")

    def test_source_bootstrap_is_deterministic_and_paired(self) -> None:
        sources = [f"test/synthetic_{index:03d}.png" for index in range(128)]
        source_record = {"counts": {"tn": 8, "fp": 0, "fn": 0, "tp": 2}}
        clean = {"by_source": {source: source_record for source in sources}}
        conditions = {f"{kind}_l{level}": clean for kind in
                      ("brightness", "contrast", "gamma", "color_bias", "haze",
                       "shadow", "noise", "blur", "jpeg") for level in range(1, 6)}
        result = bootstrap_sources(clean, conditions, seed=20260926, replicates=5000)
        self.assertEqual(result["percentile_95"]["clean_f1"], [1.0, 1.0])
        self.assertEqual(result["percentile_95"]["corruption_mf1"], [1.0, 1.0])
        self.assertEqual(result["percentile_95"]["clean_minus_corruption_mf1"], [0.0, 0.0])

    def test_missing_approval_blocks_before_test_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = Job(Path(directory) / "job.yaml", {"approval": "missing.yaml"})
            with mock.patch("scripts.final_test_b1_v1.audit_test_data") as audit:
                with self.assertRaises(FileNotFoundError):
                    run(job)
                audit.assert_not_called()

    def test_synthetic_audit_rejects_dev_pair_duplicated_in_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml").values)
            values["experiment"].update(kind="smoke", output_dir=str(root / "unused"))
            values["data"].update(root=str(root / "data"), expected_image_size=256,
                                  tile_size=256, expected_pairs={"train": 1, "val": 1, "test": 1})
            values["evaluation"]["run_test_after_training"] = True
            config = Config(values, root / "fixture.yaml")
            create_fixture(config)
            checkpoint = {"manifests": {}, "file_sha256": {}}
            for split in ("train", "val"):
                data = LEVIRDataset(config, split)
                checkpoint["manifests"][split] = data.manifest()
                checkpoint["file_sha256"][split] = {
                    record.name: {key: file_sha256(getattr(record, key))
                                  for key in ("image_a", "image_b", "mask")}
                    for record in data.records}
            _, report = audit_test_data(config, checkpoint, 1, 1)
            self.assertEqual(report["split"], "test")
            for folder in ("A", "B"):
                shutil.copyfile(root / "data" / "train" / folder / "train_0.png",
                                root / "data" / "test" / folder / "test_0.png")
            with self.assertRaisesRegex(ValueError, "duplicates train/val"):
                audit_test_data(config, checkpoint, 1, 1)


if __name__ == "__main__":
    unittest.main()
