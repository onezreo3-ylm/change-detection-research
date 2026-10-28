import copy
import unittest

from scripts.audit_valais_full_fp32_completion import validate_history


class FullFp32CompletionAuditTests(unittest.TestCase):
    def test_history_requires_100_sequential_finite_rows_and_first_best(self):
        rows = [{"epoch": epoch, "val": {"metrics": {"f1": 0.4 if epoch in (53, 70)
                                                            else 0.1}}}
                for epoch in range(1, 101)]
        self.assertEqual(validate_history(rows), (53, 0.4))
        with self.assertRaises(RuntimeError):
            validate_history([rows])  # Windows PowerShell 5.1 nested-array shape.
        skipped = copy.deepcopy(rows)
        skipped[40]["epoch"] = 40
        with self.assertRaises(RuntimeError):
            validate_history(skipped)
        nonfinite = copy.deepcopy(rows)
        nonfinite[0]["val"]["metrics"]["f1"] = float("nan")
        with self.assertRaises(RuntimeError):
            validate_history(nonfinite)


if __name__ == "__main__":
    unittest.main()
