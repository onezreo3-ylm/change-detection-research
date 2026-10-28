"""Engineering tests for fail-closed P0 scope and pooling, no dataset access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.coupling_diagnostic import four_cell_diagnostic, image_sha256
from pcdc.data import file_sha256
from scripts.evaluate_coupling_p0 import (CELLS, FAMILIES, Guard, render_views,
    summarize, validate_manifest, validate_review)


def manifest_fixture(root):
    samples = []
    for index in range(64):
        name = f"sample{index}.tif"
        conditions = []
        for family in sorted(FAMILIES):
            for level in (1, 3, 5):
                hashes = dict.fromkeys(("au", "av", "bu", "bv"), "0" * 64)
                conditions.append({"family": family, "level": level, "images": hashes,
                    "marginal_hashes": {k: [hashes[a], hashes[b]] for k, (a, b) in CELLS.items()}})
        samples.append({"id": f"train/{name}@x0_y0", "name": name,
            "foreground_pixels": int(index < 16), "stratum": "empty" if index >= 16 else "positive",
            "files": {k: {"path": str(root / "train" / folder / name), "sha256": "0" * 64}
                      for k, folder in (("image_a", "A"), ("image_b", "B"), ("mask", "label"))},
            "conditions": conditions})
    return {"schema_version": 1, "split": "train", "models": dict.fromkeys(("b0", "b1", "b0e"), {}),
        "data_root": str(root), "expected_predictions": 9408, "samples": samples,
        "settings": {"families": sorted(FAMILIES), "levels": [1, 3, 5], "threshold": .5,
                     "max_seconds": 21600, "temperature_stop_c": 85}}


class P0RunnerTests(unittest.TestCase):
    def test_scope_valid_and_rejects_test_path_without_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            m = manifest_fixture(Path(directory))
            validate_manifest(m, verify_files=False)
            m["samples"][0]["files"]["image_a"]["path"] = str(Path(directory) / "test/A/sample0.tif")
            with patch("scripts.evaluate_coupling_p0.checked_hash") as read:
                with self.assertRaisesRegex(ValueError, "outside"):
                    validate_manifest(m)
                read.assert_not_called()

    def test_rejects_duplicate_or_missing_condition(self):
        m = manifest_fixture(Path("fixture").resolve())
        m["samples"][0]["conditions"][-1] = m["samples"][0]["conditions"][0]
        with self.assertRaisesRegex(ValueError, "condition"):
            validate_manifest(m, verify_files=False)

    def test_rejects_forbidden_split_in_resolved_data_root(self):
        m = manifest_fixture(Path("fixture/test/nested").resolve())
        with self.assertRaisesRegex(ValueError, "aliases"):
            validate_manifest(m, verify_files=False)

    def test_rejects_bad_marginal(self):
        m = manifest_fixture(Path("fixture").resolve())
        m["samples"][0]["conditions"][0]["marginal_hashes"]["uv"][0] = "1" * 64
        with self.assertRaisesRegex(ValueError, "marginal"):
            validate_manifest(m, verify_files=False)

    def test_missing_review_rejects(self):
        with self.assertRaisesRegex(ValueError, "acceptance"):
            validate_review({}, Path("unused"), None)

    def test_review_binds_all_images_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, manifest, review = root / "image", root / "manifest", root / "review"
            image.write_bytes(b"unchanged-original")
            manifest.write_text("{}", encoding="utf-8")
            item = {"path": str(image), "sha256": file_sha256(image)}
            payload = {"status": "accepted_train_only_sampled_visibility", "manifest_sha256": file_sha256(manifest),
                       "reviewer": "unit-test", "limitations": "engineering only",
                       "reviewed_images": [{**item, "findings": "test"}]}
            review.write_text(json.dumps(payload), encoding="utf-8")
            validate_review({"qa_images": [item]}, manifest, review)
            payload["reviewed_images"] = []
            review.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every"):
                validate_review({"qa_images": [item]}, manifest, review)

    def test_render_reuses_pixels_and_fails_hash_change(self):
        raw = np.zeros((32, 32, 3), dtype=np.uint8)
        instance = {"parameters": {"factor": 1}, "seed": 42}
        condition = {"family": "brightness", "u": instance, "v": instance,
                     "images": dict.fromkeys(("au", "av", "bu", "bv"), image_sha256(raw))}
        condition["marginal_hashes"] = {k: [image_sha256(raw)] * 2 for k in CELLS}
        _, check = render_views(raw, raw, condition)
        self.assertTrue(check["fully_degenerate"])
        condition["images"]["au"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pixel hash"):
            render_views(raw, raw, condition)

    def test_pooling_clean_is_not_repeated_per_condition(self):
        target = torch.tensor([[[[1., 0.]]]])
        clean = torch.tensor([[[[2., -2.]]]])
        diag = four_cell_diagnostic(clean, dict.fromkeys(CELLS, clean), target)
        rows = [{"model": m, "source_id": "s", "stratum": "positive", "family": f,
                 "level": l, "diagnostic": diag} for m in ("b0", "b1", "b0e")
                for f in FAMILIES for l in (1, 3, 5)]
        report = summarize(rows)
        self.assertFalse(report["gate"]["passed"])
        for model in ("b0", "b1", "b0e"):
            self.assertEqual(report["aggregates"][model]["clean_unique_sources"]["counts"]["tp"], 1)
            self.assertEqual(report["aggregates"][model]["all"]["groups"]["D"]["counts"]["tp"], 24)

    def test_budget_guard_stops_before_sensor(self):
        guard = Guard({"max_seconds": 1})
        guard.started -= 2
        with patch("scripts.evaluate_coupling_p0.subprocess.run") as sensor:
            with self.assertRaisesRegex(RuntimeError, "budget"):
                guard.check()
            sensor.assert_not_called()

    def test_temperature_and_gpu_identity_fail_closed(self):
        guard = Guard({"max_seconds": 21600, "gpu_uuid": "GPU-expected", "temperature_stop_c": 85})
        with patch("scripts.evaluate_coupling_p0.subprocess.run") as sensor:
            sensor.return_value.stdout = "GPU-expected, 85"
            with self.assertRaisesRegex(RuntimeError, "temperature"):
                guard.check()
            sensor.return_value.stdout = "GPU-wrong, 40"
            with self.assertRaisesRegex(RuntimeError, "identity"):
                guard.check()
            sensor.return_value.stdout = "GPU-expected, 84"
            guard.check()


if __name__ == "__main__":
    unittest.main()
