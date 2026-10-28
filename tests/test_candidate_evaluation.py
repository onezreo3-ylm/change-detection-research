from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from pcdc.evaluate_candidate import (
    _condition_diagnostics,
    _validate_training_metadata,
    compare_results,
    load_candidate_evaluation_config,
)


SHA = "a" * 64


def job_values() -> dict:
    return {
        "schema_version": 1,
        "name": "candidate-smoke",
        "expected_kind": "research",
        "expected_mode": "b2",
        "benchmark_config": "benchmark.yaml",
        "baseline_summary": "baseline-summary.json",
        "baseline_summary_sha256": SHA,
        "baseline_conditions": "baseline-conditions.json",
        "baseline_conditions_sha256": SHA,
        "candidate_best_checkpoint": "best.pt",
        "candidate_best_checkpoint_sha256": SHA,
        "candidate_last_checkpoint": "last.pt",
        "candidate_last_checkpoint_sha256": SHA,
        "candidate_status": "status.json",
        "candidate_status_sha256": SHA,
        "candidate_guard_completion": "guard-completion.json",
        "candidate_guard_completion_sha256": SHA,
        "candidate_exposure": "exposure.json",
        "candidate_exposure_sha256": SHA,
        "output_dir": "output",
    }


def metrics(f1: float, foreground_recall: float = 0.6,
            background_fpr: float = 0.1) -> dict[str, float]:
    return {
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "iou": f1 / (2 - f1),
        "oa": 0.9,
        "foreground_recall": foreground_recall,
        "background_fpr": background_fpr,
        "foreground_fn_rate": 1 - foreground_recall,
    }


def report(f1: float, counts: tuple[int, int, int, int]) -> dict:
    source = {
        "counts": dict(zip(("tn", "fp", "fn", "tp"), counts)),
        "metrics": metrics(f1),
    }
    return {
        "counts": dict(source["counts"]),
        "metrics": dict(source["metrics"]),
        "by_source": {"val/pair": source},
    }


class CandidateConfigTests(unittest.TestCase):
    def _write(self, directory: Path, values: dict) -> Path:
        path = directory / "candidate.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return path

    def test_loader_is_strict_and_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._write(root, job_values())
            job = load_candidate_evaluation_config(path)
            self.assertEqual(job.values["expected_mode"], "b2")
            self.assertEqual(job.path("output_dir"), (root / "output").resolve())
            broken = job_values()
            broken["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "keys differ"):
                load_candidate_evaluation_config(self._write(root, broken))

    def test_loader_rejects_mode_and_noncanonical_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = job_values()
            broken["expected_mode"] = "b3"
            with self.assertRaisesRegex(ValueError, "expected_mode"):
                load_candidate_evaluation_config(self._write(root, broken))
            broken = job_values()
            broken["candidate_status_sha256"] = "A" * 64
            with self.assertRaisesRegex(ValueError, "lowercase SHA256"):
                load_candidate_evaluation_config(self._write(root, broken))

    def test_completed_training_metadata_fails_closed_on_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = load_candidate_evaluation_config(self._write(root, job_values()))
            status = {
                "status": "completed", "training_started": True, "mode": "b2",
                "kind": "research", "selection": "clean_val_f1",
                "test_evaluated": False, "guarded_completion_required": True,
                "temperature_guard": {"status": "completed"},
                "epochs": 2, "best_epoch": 1, "optimizer_steps": 4,
                "amp_skipped_steps": 0, "pair_views": 16,
            }
            completion = {"status": "completed", "training_status_sha256": SHA}
            exposure = {
                "epochs": 2, "counts": {"noise|l1|neutral|A|global": 8},
                "per_epoch_stream_sha256": [
                    {"epoch": 1, "sha256": SHA}, {"epoch": 2, "sha256": SHA}],
                "combined_stream_sha256": SHA,
            }
            _validate_training_metadata(job, status, completion, exposure)
            unsafe = copy.deepcopy(status)
            unsafe["test_evaluated"] = True
            with self.assertRaisesRegex(ValueError, "completed guarded"):
                _validate_training_metadata(job, unsafe, completion, exposure)


class CandidateComparisonTests(unittest.TestCase):
    def test_condition_and_source_deltas_are_candidate_minus_baseline(self):
        candidate = {"noise_l1": report(0.8, (80, 4, 6, 10))}
        baseline = {"noise_l1": report(0.7, (78, 6, 8, 8))}
        conditions, paired = compare_results(candidate, baseline, ["noise_l1"])
        self.assertAlmostEqual(
            conditions["noise_l1"]["metric_delta_candidate_minus_baseline"]["f1"], 0.1)
        self.assertEqual(
            conditions["noise_l1"]["count_delta_candidate_minus_baseline"],
            {"tn": 2, "fp": -2, "fn": -2, "tp": 2})
        self.assertAlmostEqual(
            paired["noise_l1"]["val/pair"]
            ["metric_delta_candidate_minus_baseline"]["f1"], 0.1)

    def test_comparison_rejects_missing_condition_or_source(self):
        candidate = {"noise_l1": report(0.8, (80, 4, 6, 10))}
        baseline = {"noise_l1": report(0.7, (78, 6, 8, 8))}
        with self.assertRaisesRegex(ValueError, "condition coverage"):
            compare_results(candidate, baseline, ["noise_l1", "blur_l1"])
        del baseline["noise_l1"]["by_source"]["val/pair"]
        with self.assertRaisesRegex(ValueError, "source coverage"):
            compare_results(candidate, baseline, ["noise_l1"])

    def test_diagnostics_keep_raw_pooled_counts_and_macro_rates(self):
        conditions = {
            "a_l1": report(0.5, (8, 2, 3, 7)),
            "a_l2": report(0.6, (9, 1, 2, 8)),
        }
        conditions["a_l1"]["metrics"].update(
            foreground_recall=0.7, background_fpr=0.2)
        conditions["a_l2"]["metrics"].update(
            foreground_recall=0.8, background_fpr=0.1)
        result = _condition_diagnostics(conditions, ["a_l1", "a_l2"])
        self.assertEqual(result["pooled_over_conditions"]["counts"],
                         {"tn": 17, "fp": 3, "fn": 5, "tp": 15})
        self.assertAlmostEqual(result["macro_condition_foreground_recall"], 0.75)
        self.assertAlmostEqual(result["macro_condition_background_fpr"], 0.15)


if __name__ == "__main__":
    unittest.main()
