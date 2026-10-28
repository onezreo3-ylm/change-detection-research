"""Regression for exited-handle false liveness and damaged recovery prefixes."""
import copy
import unittest

from scripts.p1_process_state import classify_process
from scripts.resume_valais_p1_study import check_prefix_history


class ProcessStateTests(unittest.TestCase):
    def test_exited_process_with_matching_identity_is_not_alive(self):
        self.assertFalse(classify_process(1, 12, 12, 0, 0)["owned_active"])

    def test_exit_code_259_cannot_override_signaled_process(self):
        self.assertFalse(classify_process(1, 12, 12, 0, 259)["owned_active"])

    def test_active_process_requires_matching_creation_identity(self):
        self.assertTrue(classify_process(1, 12, 12, 258, 259)["owned_active"])
        self.assertFalse(classify_process(1, 13, 12, 258, 259)["owned_active"])

    def test_exit_racing_with_query_is_not_alive(self):
        self.assertFalse(classify_process(1, 12, 12, 258, 0)["owned_active"])

    def test_failed_wait_cannot_be_treated_as_stopped(self):
        with self.assertRaises(OSError):
            classify_process(1, 12, 12, 0xffffffff, 0)


class PrefixTests(unittest.TestCase):
    def fixture(self):
        planned = [{"epoch": epoch, "source_batch_sha256": str(epoch)} for epoch in range(1, 4)]
        actual = [dict(row, actual_source_stream_verified=True) for row in planned[:2]]
        history = [{"epoch": epoch, "train": {"sampling": actual[epoch-1],
                    "optimizer_steps": 2757, "amp_skipped_steps": 0, "pair_views": 22050}}
                   for epoch in range(1, 3)]
        return history, actual, planned

    def test_exact_committed_prefix_is_accepted(self):
        check_prefix_history(*self.fixture(), 2)

    def test_missing_or_duplicate_epoch_is_rejected(self):
        history, actual, planned = self.fixture()
        for changed in (history[1:], [history[0], history[0]]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_prefix_history(changed, actual, planned, 2)

    def test_sampler_or_update_budget_change_is_rejected(self):
        for key, value in (("optimizer_steps", 2756), ("amp_skipped_steps", 1), ("pair_views", 22048)):
            history, actual, planned = self.fixture()
            history[1]["train"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_prefix_history(history, actual, planned, 2)
        history, actual, planned = self.fixture()
        changed = copy.deepcopy(actual)
        changed[0]["source_batch_sha256"] = "changed"
        with self.assertRaises(ValueError):
            check_prefix_history(history, changed, planned, 2)


if __name__ == "__main__":
    unittest.main()
