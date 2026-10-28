"""Synthetic-only regression for the separately frozen V1R2 correction."""

import unittest

from scripts.audit_final_test_b1_v1r2 import corrected_add_counts, synthetic_smoke


class AuditCorrectionTests(unittest.TestCase):
    def test_one_shot_generator_is_materialized_once(self) -> None:
        rows = ({"tn": value, "fp": value + 1, "fn": value + 2, "tp": value + 3}
                for value in (1, 2))
        self.assertEqual(corrected_add_counts(rows),
                         {"tn": 3, "fp": 5, "fn": 7, "tp": 9})

    def test_empty_generator_has_zero_counts(self) -> None:
        self.assertEqual(corrected_add_counts(iter(())),
                         {"tn": 0, "fp": 0, "fn": 0, "tp": 0})

    def test_synthetic_smoke(self) -> None:
        self.assertEqual(synthetic_smoke()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
