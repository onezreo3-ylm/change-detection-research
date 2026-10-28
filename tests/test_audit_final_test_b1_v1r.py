"""Synthetic-only tests for the independent V1R audit; never open test data."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_final_test_b1_v1r import (add_counts, check_counts_report,
                                            compare_number, json_file,
                                            keyed_seed, report_metrics, sha,
                                            synthetic_smoke, write_json_new)


class AuditOnlyRecoveryTests(unittest.TestCase):
    def test_synthetic_smoke(self) -> None:
        self.assertEqual(synthetic_smoke()["status"], "passed")

    def test_zero_denominator_convention(self) -> None:
        metrics = report_metrics({"tn": 4, "fp": 0, "fn": 0, "tp": 0})
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["oa"], 1.0)
        self.assertIsNone(report_metrics({"tn": 0, "fp": 0, "fn": 0, "tp": 0}))

    def test_rejects_corrupted_metric_and_count(self) -> None:
        counts = {"tn": 5, "fp": 2, "fn": 1, "tp": 3}
        row = {"counts": counts, "pixels": 11, "foreground_pixels": 4,
               "background_pixels": 7, "metrics": report_metrics(counts), "regions": None}
        self.assertEqual(check_counts_report(row, "fixture"), counts)
        row["metrics"]["f1"] = 0.0
        with self.assertRaisesRegex(ValueError, "Numerical mismatch"):
            check_counts_report(row, "fixture")
        row["metrics"] = report_metrics(counts)
        counts["tp"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid integer"):
            check_counts_report(row, "fixture")

    def test_regions_must_reconcile(self) -> None:
        counts = {"tn": 5, "fp": 2, "fn": 1, "tp": 3}
        inside = {"tn": 2, "fp": 1, "fn": 0, "tp": 1}
        outside = {"tn": 3, "fp": 1, "fn": 1, "tp": 2}
        def report(value):
            return {"counts": value, "pixels": sum(value.values()),
                    "foreground_pixels": value["fn"] + value["tp"],
                    "background_pixels": value["tn"] + value["fp"],
                    "metrics": report_metrics(value)}
        row = {**report(counts), "regions": {"injection_support": report(inside),
                                                "outside_injection": report(outside)}}
        check_counts_report(row, "fixture", regional=True)
        row["regions"]["outside_injection"]["counts"]["tn"] += 1
        with self.assertRaises(ValueError):
            check_counts_report(row, "fixture", regional=True)

    def test_seed_and_sum_are_deterministic(self) -> None:
        self.assertEqual(keyed_seed("fixture", 7), keyed_seed("fixture", 7))
        self.assertNotEqual(keyed_seed("fixture", 7), keyed_seed("fixture", 8))
        self.assertEqual(add_counts([{"tn": 1, "fp": 2, "fn": 3, "tp": 4},
                                     {"tn": 2, "fp": 3, "fn": 4, "tp": 5}]),
                         {"tn": 3, "fp": 5, "fn": 7, "tp": 9})

    def test_strict_json_and_immutable_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json_new(path, {"value": 1})
            self.assertEqual(json_file(path), {"value": 1})
            self.assertEqual(len(sha(path)), 64)
            with self.assertRaisesRegex(ValueError, "Refuse to overwrite"):
                write_json_new(path, {"value": 2})
            path.write_text('{"value":1,"value":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                json_file(path)
            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                json_file(path)

    def test_nonfinite_comparison_fails_closed(self) -> None:
        for bad in (float("nan"), float("inf"), "0.5"):
            with self.assertRaisesRegex(ValueError, "Numerical mismatch"):
                compare_number(bad, 0.5, "fixture")


if __name__ == "__main__":
    unittest.main()
