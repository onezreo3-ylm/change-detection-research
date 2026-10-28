"""Transport safety tests; synthetic image fixtures are engineering checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from scripts.prepare_chat_previews import prepare


class ChatPreviewTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "train_qa" / "previews"
        self.source.mkdir(parents=True)
        self.output = self.root / "copies"
        self.path = self.root / "transport.yaml"
        self.values = {"schema_version": 1, "source_directories": [str(self.source)],
                       "output_dir": str(self.output), "max_long_edge": 256,
                       "minimum_long_edge": 32, "max_image_bytes": 12000,
                       "max_images_per_batch": 2, "max_batch_base64_bytes": 32000,
                       "overview_only": True}
        self.write_config()

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.values), encoding="utf-8")

    def test_originals_and_budget_and_manifest(self):
        rng = np.random.default_rng(20261002)
        hashes = {}
        for index in range(5):
            path = self.source / f"sheet-{index}.png"
            Image.fromarray(rng.integers(0, 256, (320, 256, 3), dtype=np.uint8)).save(path)
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = prepare(self.path)
        self.assertEqual(report["image_count"], 5)
        self.assertFalse(report["scientific_review_evidence"])
        self.assertFalse(report["dataset_images_or_labels_accessed"])
        self.assertTrue(report["originals_unchanged"])
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for image in report["images"]:
            preview = Path(image["preview"])
            self.assertLessEqual(preview.stat().st_size, self.values["max_image_bytes"])
            self.assertEqual(image["preview_sha256"], hashlib.sha256(preview.read_bytes()).hexdigest())
            with Image.open(preview) as opened:
                self.assertEqual(opened.format, "PNG")
        for batch in report["batches"]:
            self.assertLessEqual(len(batch["images"]), 2)
            self.assertLessEqual(batch["base64_bytes"], self.values["max_batch_base64_bytes"])
        self.assertEqual(json.loads((self.output / "transport_manifest.json").read_text()), report)

    def test_existing_output_is_preserved(self):
        Image.new("RGB", (16, 16)).save(self.source / "sheet.png")
        self.output.mkdir()
        sentinel = self.output / "user-file.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare(self.path)
        self.assertEqual(sentinel.read_text(), "keep me")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_inside_source_is_rejected(self):
        self.values["output_dir"] = str(self.source / "copies")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare(self.path)
        self.assertFalse((self.source / "copies").exists())

    def test_held_out_scope_is_rejected_before_creating_output(self):
        self.values["source_directories"] = [str(self.root / "test" / "train_qa" / "previews")]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "train QA"):
            prepare(self.path)
        self.assertFalse(self.output.exists())

    def test_unsatisfiable_byte_budget_fails(self):
        Image.new("RGB", (16, 16)).save(self.source / "sheet.png")
        self.values["max_image_bytes"] = 1
        self.write_config()
        with self.assertRaisesRegex(ValueError, "byte budget"):
            prepare(self.path)
        self.assertFalse((self.output / "transport_manifest.json").exists())
