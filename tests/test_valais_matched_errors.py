"""Saved-count error review checks; synthetic fixtures are engineering only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_valais_matched_errors import (
    aggregate_conditions, analyze, describe_sources, load_assignments,
    paired_sources, size_groups,
)
from scripts.analyze_valais_saved_subgroups import metrics, pooled_counts
from scripts.audit_valais_matched_completion import validate_exposure


def source(tn, fp, fn, tp):
    counts = dict(tn=tn, fp=fp, fn=fn, tp=tp)
    return {"counts": counts, "pixels": tn + fp + fn + tp,
            "foreground_pixels": fn + tp, "background_pixels": tn + fp}


class MatchedErrorTests(unittest.TestCase):
    def sources(self):
        return {"val/empty.tif": source(10, 0, 0, 0),
                "val/missed.tif": source(7, 1, 2, 0),
                "val/detected.tif": source(7, 0, 2, 1)}

    def test_zero_tp_positive_tile_is_distinct_from_no_predictions(self):
        result = describe_sources(self.sources())
        self.assertEqual(result["positive_tiles_without_tp"], 1)
        self.assertEqual(result["positive_tiles_without_predictions"], 0)
        self.assertEqual(result["empty_tiles_with_predictions"], 0)
        self.assertEqual(result["foreground_pixels"], 5)
        self.assertEqual(result["counts"], dict(tn=24, fp=1, fn=4, tp=1))

    def test_paired_source_gains_and_losses_exclude_empty_tiles(self):
        reference = self.sources()
        candidate = {"val/empty.tif": source(9, 1, 0, 0),
                     "val/missed.tif": source(7, 1, 1, 1),
                     "val/detected.tif": source(7, 0, 3, 0)}
        result = paired_sources(candidate, reference)
        self.assertEqual(result["counts_delta"], dict(tn=-1, fp=1, fn=0, tp=0))
        self.assertEqual(result["previous_zero_tp_positive_tiles_now_with_tp"], 1)
        self.assertEqual(result["previous_detected_positive_tiles_now_zero_tp"], 1)
        self.assertEqual(result["sources_with_more_fn"], 1)
        self.assertEqual(result["sources_with_fewer_fn"], 1)
        self.assertEqual(result["sources_with_more_fp"], 1)

    def test_paired_sources_refuse_missing_or_changed_labels(self):
        reference = self.sources()
        candidate = copy.deepcopy(reference)
        candidate.pop("val/empty.tif")
        with self.assertRaisesRegex(ValueError, "coverage"):
            paired_sources(candidate, reference)
        candidate = copy.deepcopy(reference)
        candidate["val/missed.tif"] = source(6, 1, 3, 0)
        with self.assertRaisesRegex(ValueError, "label/pixel"):
            paired_sources(candidate, reference)

    def test_area_bins_are_disjoint_and_include_empty_and_upper_tail(self):
        records = self.sources()
        records["val/large.tif"] = source(5, 0, 0, 65)
        rows = size_groups(records, [2, 64])
        self.assertEqual([row["samples"] for row in rows], [1, 1, 1, 1])
        self.assertEqual(pooled_counts(iter(rows)), pooled_counts(records.values()))
        for boundaries in ([64, 2], [2, 2], [True], [0], []):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                size_groups(records, boundaries)

    def test_macro_condition_f1_is_not_pooled_f1_and_clean_is_repeated(self):
        def row(empty, positive):
            counts = pooled_counts(iter([empty, positive]))
            return {"counts": counts, "metrics": metrics(counts),
                    "strata": {"empty": describe_sources({"e": empty}),
                               "positive": describe_sources({"p": positive})}}
        rows = [row(source(10, 0, 0, 0), source(8, 0, 0, 9)),
                row(source(9, 1, 0, 0), source(8, 0, 9, 0))]
        result = aggregate_conditions(rows, dict(tn=15, fp=5, fn=1, tp=8))
        self.assertEqual(result["mean_condition_f1"], 0.5)
        self.assertAlmostEqual(result["pooled_metrics"]["f1"], 18 / 28)
        self.assertEqual(result["net_fp_vs_repeated_clean"], -9)
        self.assertEqual(result["net_fn_vs_repeated_clean"], 7)
        self.assertEqual(result["positive_tile_condition_appearances_without_tp"], 1)
        broken = copy.deepcopy(rows)
        broken[0]["strata"]["empty"]["counts"]["fp"] += 1
        with self.assertRaisesRegex(ValueError, "partition"):
            aggregate_conditions(broken, dict(tn=15, fp=5, fn=1, tp=8))

    def test_duplicate_and_missing_validation_instances_are_rejected(self):
        item = dict(condition_id="blur_l1", kind="blur", level=1, stream="val",
                    sample_id="val/a.tif@x0_y0", spatial_mode="global", direction="neutral", target_date="A")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "instances.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            self.assertEqual(set(load_assignments(path, {"blur_l1"}, {"val/a.tif"})), {"blur_l1"})
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_assignments(path, {"blur_l1"}, {"val/a.tif", "val/b.tif"})
            path.write_text((json.dumps(item) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_assignments(path, {"blur_l1"}, {"val/a.tif"})

    def test_changed_count_file_is_rejected_before_missing_instances(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            target = directory / "clean_metrics.json"
            target.write_text("{}", encoding="utf-8")
            audit = directory / "audit.json"
            audit.write_text(json.dumps(dict(status="completed_matched_study_independently_verified",
                                              test_accessed=False, source_hashes={str(target.resolve()): "changed"})), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the completion audit"):
                analyze(dict.fromkeys(("b0", "b1", "b0e"), directory), directory / "missing.jsonl", audit, [64])

    def test_exposure_hash_uses_canonical_epoch_records_and_checks_budget(self):
        train = dict(source_samples=11025, pair_views=22050, optimizer_steps=2757,
                     amp_skipped_steps=0, exposure_counts={"clean": 11025}, exposure_stream_sha256="a" * 64)
        history = [dict(epoch=1, train=train)]
        streams = [dict(epoch=1, sha256="a" * 64)]
        encoded = json.dumps(streams, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        exposure = dict(epochs=1, counts={"clean": 11025}, per_epoch_stream_sha256=streams,
                        combined_stream_sha256=hashlib.sha256(encoded).hexdigest())
        validate_exposure(history, exposure)
        wrong = copy.deepcopy(exposure)
        wrong["combined_stream_sha256"] = hashlib.sha256(bytes.fromhex("a" * 64)).hexdigest()
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_exposure(history, wrong)
        broken = copy.deepcopy(history)
        broken[0]["train"]["optimizer_steps"] -= 1
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_exposure(broken, exposure)


if __name__ == "__main__":
    unittest.main()
