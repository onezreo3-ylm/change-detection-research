import unittest
from pathlib import Path

from scripts.audit_valais_guard_completion import inspect, validate_history


class GuardCompletionAuditTests(unittest.TestCase):
    def test_incomplete_current_run_cannot_be_certified(self):
        config = Path("configs/valais_guard_completion_audit_20261001.yaml").resolve()
        with self.assertRaises(RuntimeError):
            inspect(config)

    def test_history_selects_first_best_and_rejects_nonfinite(self):
        history = [
            {"epoch": 1, "val": {"metrics": {"f1": 0.1}}},
            {"epoch": 2, "val": {"metrics": {"f1": 0.3}}},
            {"epoch": 3, "val": {"metrics": {"f1": 0.3}}},
        ]
        self.assertEqual(validate_history(history, 3), (2, 0.3))
        with self.assertRaises(RuntimeError):
            validate_history(history, 4)
        bad = [*history]
        bad[2] = {"epoch": 3, "val": {"metrics": {"f1": float("nan")}}}
        with self.assertRaises(RuntimeError):
            validate_history(bad, 3)


if __name__ == "__main__":
    unittest.main()
