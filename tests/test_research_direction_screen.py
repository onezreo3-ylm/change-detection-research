"""Research screening integrity; fixtures are synthetic engineering checks only."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.screen_research_directions import (
    common_misses, components, counts, load_train, load_validation,
    require_same_sources, summarize, train_file,
)


class DirectionScreenTests(unittest.TestCase):
    def test_pooling_and_zero_tp_do_not_mean_no_predictions(self):
        result = summarize({"a": dict(tn=8, fp=1, fn=1, tp=0),
                            "b": dict(tn=0, fp=0, fn=0, tp=10)}, 1)
        self.assertAlmostEqual(result["f1"], 20 / 22)
        self.assertEqual(result["positive_sources_zero_tp"], 1)
        self.assertEqual(result["positive_sources_zero_predictions"], 0)
        self.assertEqual(result["fraction_fn_in_zero_tp_sources"], 1)

    def test_count_conservation_rejects_negative_and_boolean_counts(self):
        for values in (dict(tn=4, fp=0, fn=0, tp=1), dict(tn=4, fp=-1, fn=0, tp=1),
                       dict(tn=3, fp=0, fn=0, tp=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                counts(values, 4)

    def test_comparison_rejects_changed_source_coverage_or_labels(self):
        a = {"train/a.tif@x0_y0": dict(tn=2, fp=1, fn=1, tp=0)}
        b = copy.deepcopy(a)
        b["train/a.tif@x0_y0"] = dict(tn=1, fp=1, fn=1, tp=1)
        with self.assertRaisesRegex(ValueError, "foreground"):
            require_same_sources({"x": a, "y": b})
        with self.assertRaisesRegex(ValueError, "coverage"):
            require_same_sources({"x": a, "y": {}})

    def test_validation_duplicate_and_test_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            header = "model,source,foreground_pixels,tn,fp,fn,tp\n"
            row = "x,val/a.tif,1,3,0,1,0\n"
            path.write_text(header + row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_validation(path, ["x"], 4)
            path.write_text(header + row.replace("val/", "test/"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                load_validation(path, ["x"], 4)

    def train_fixture(self):
        first = {"model": "x", "source_id": "train/a.tif@x0_y0", "foreground_pixels": 1,
                 "family": "gamma", "level": 1,
                 "diagnostic": {"clean": {"counts": dict(tn=3, fp=0, fn=1, tp=0)}}}
        second = copy.deepcopy(first)
        second["level"] = 3
        return [first, second]

    def test_repeated_clean_predictions_count_once_and_require_all_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = self.train_fixture()
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            loaded = load_train(path, ["x"], 4, {("gamma", 1), ("gamma", 3)})
            self.assertEqual(summarize(loaded["x"], 1)["counts"]["fn"], 1)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_train(path, ["x"], 4, {("gamma", 1), ("gamma", 3), ("gamma", 5)})

    def test_inconsistent_repeated_predictions_cannot_be_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = self.train_fixture()
            rows[1]["diagnostic"]["clean"]["counts"] = dict(tn=3, fp=0, fn=0, tp=1)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Repeated clean counts"):
                load_train(path, ["x"], 4, {("gamma", 1), ("gamma", 3)})

    def test_components_distinguish_diagonals_and_censored_edges(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = mask[1, 1] = mask[3, 3] = 1
        four, eight = components(mask, 4), components(mask, 8)
        self.assertEqual([r["area"] for r in four], [1, 1, 1])
        self.assertEqual([r["area"] for r in eight], [2, 1])
        self.assertEqual(sum(r["touches_tile_edge"] for r in eight), 1)
        self.assertEqual(sum(r["area"] for r in eight), int(mask.sum()))
        self.assertEqual(components(np.zeros((5, 5), dtype=np.uint8), 8), [])
        with self.assertRaises(ValueError):
            components(np.array([[0, 255]], dtype=np.uint8), 8)

    def test_raw_review_paths_cannot_escape_train_or_read_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {"id": "train/a.tif@x0_y0", "name": "a.tif",
                      "files": {"mask": {"path": str(root / "train" / "label" / "a.tif")}}}
            self.assertEqual(train_file(root, sample, "mask"), (root / "train" / "label" / "a.tif").resolve())
            sample["files"]["mask"]["path"] = str(root / "val" / "label" / "a.tif")
            with self.assertRaisesRegex(ValueError, "non-train"):
                train_file(root, sample, "mask")
            sample["name"] = "../a.tif"
            with self.assertRaisesRegex(ValueError, "Invalid train"):
                train_file(root, sample, "mask")

    def test_common_misses_are_positive_tiles_not_repeated_condition_counts(self):
        rows = {"empty": dict(tn=4, fp=0, fn=0, tp=0),
                "positive": dict(tn=3, fp=0, fn=1, tp=0)}
        result = common_misses({"x": rows, "y": copy.deepcopy(rows)})
        self.assertEqual(result["all_models_zero_tp_sources"], ["positive"])
        self.assertEqual(result["positive_sources"], 1)


if __name__ == "__main__":
    unittest.main()
