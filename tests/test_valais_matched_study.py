"""Engineering checks for the new matched study's evidence and stopping gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pcdc.data import file_sha256
from scripts.analyze_valais_baseline import analyze, metrics, pooled_counts, source_strata
from scripts.valais_matched_study import bound_check, descriptive_gates, select_sources


class MatchedStudyTests(unittest.TestCase):
    def test_train_only_selection_is_fixed_stratified_and_score_independent(self):
        balance = {"train": {"positive_manifest": [{"name": "a", "positive_pixels": 1},
                       {"name": "b", "positive_pixels": 3}, {"name": "c", "positive_pixels": 6},
                       {"name": "d", "positive_pixels": 7}], "nonempty_positive_pixels_p50": 3}}
        settings = {"selection_seed": 7, "small_positive_tiles": 1, "large_positive_tiles": 1,
                    "empty_tiles": 2, "smoke_source_batch_size": 4}
        first = select_sources(balance, list("abcdef"), settings)
        self.assertEqual(first, select_sources(balance, list("fedcba"), settings))
        self.assertEqual([r["stratum"] for r in first], ["small_positive", "large_positive", "empty", "empty"])
        with self.assertRaises(ValueError): select_sources(balance, list("abef"), settings)

    def test_harm_gate_cannot_be_rescued_by_a_smaller_gap_or_fewer_fp(self):
        gates = {"minimum_mf1_gain_over_b0": .005, "minimum_mf1_gain_over_b0e": .005,
                 "maximum_clean_f1_drop": .005, "maximum_clean_recall_drop": .01,
                 "maximum_corruption_recall_drop": .01, "maximum_empty_tile_fpr_increase": .0002}
        base = {"mf1": .4, "clean_f1": .42, "clean_recall": .41, "corruption_recall": .35, "empty_tile_fpr": .001}
        candidate = dict(base, mf1=.41, empty_tile_fpr=.0005, clean_recall=.39)
        report = descriptive_gates({"b0": base, "b0e": base, "b1": candidate}, gates)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["b0e"]["clean_recall_protected"])
        candidate["clean_recall"] = .40
        report = descriptive_gates({"b0": base, "b0e": base, "b1": candidate}, gates)
        self.assertTrue(report["passed"])
        self.assertFalse(report["mechanism_training_authorized"])
        candidate["empty_tile_fpr"] = .00121
        self.assertFalse(descriptive_gates({"b0": base, "b0e": base, "b1": candidate}, gates)["passed"])
        candidate["mf1"] = float("nan")
        with self.assertRaises(ValueError): descriptive_gates({"b0": base, "b0e": base, "b1": candidate}, gates)

    def test_seal_rejects_changed_source_without_writing_it(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "input"
            file.write_bytes(b"original")
            seal = {str(file): file_sha256(file)}
            bound_check(seal)
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError): bound_check(seal)
            self.assertEqual(file.read_bytes(), b"changed")

    def test_empty_sources_are_not_counted_as_completely_missed_positive_tiles(self):
        sources = {"empty": {"foreground_pixels": 0, "counts": {"tn": 5, "fp": 1, "fn": 0, "tp": 0}},
                   "miss": {"foreground_pixels": 2, "counts": {"tn": 3, "fp": 1, "fn": 2, "tp": 0}},
                   "hit": {"foreground_pixels": 2, "counts": {"tn": 4, "fp": 0, "fn": 1, "tp": 1}}}
        result = source_strata(sources)
        self.assertEqual(result["positive"]["positive_tiles_without_tp"], 1)
        self.assertEqual(result["empty"]["positive_tiles_without_tp"], 0)
        self.assertEqual(result["empty"]["tiles_with_prediction"], 1)
        self.assertAlmostEqual(metrics(pooled_counts(sources.values()))["recall"], .25)

    def test_analysis_refuses_guard_hash_mismatch_before_opening_instances(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            summary = {"status": "completed", "split": "val", "test_evaluated": False,
                       "clean_matches_reference": True}
            records = {"summary.json": summary, "guard_completion.json": {"status": "completed", "summary_sha256": "wrong"},
                       "temperature_guard_state.json": {"status": "completed"}, "clean_metrics.json": {}}
            for name, record in records.items(): (folder / name).write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError): analyze(folder, folder / "unavailable_instances.jsonl")


if __name__ == "__main__": unittest.main()
