"""Reject damaged A-S recovery evidence; fixtures are synthetic engineering data."""
import copy
from types import SimpleNamespace
import unittest

import torch

from scripts.resume_valais_p1_as import rng_digest, validate_prefix


class PausedAsPrefixTests(unittest.TestCase):
    def fixture(self):
        planned = [{"epoch": epoch} for epoch in range(1, 4)]
        actual = [dict(row, actual_source_stream_verified=True) for row in planned[:2]]
        history = [{"epoch": epoch, "val": {"metrics": {"f1": 0.5}}, "train": {
            "sampling": actual[epoch - 1], "optimizer_steps": 2757, "amp_skipped_steps": 0, "pair_views": 22050}}
            for epoch in range(1, 3)]
        last = {"epoch": 2, "robust_mode": "b1", "robust_config": {"frozen": 1},
                "file_sha256": {"train": {}, "val": {}}, "history": history,
                "best_epoch": 1, "best_val_f1": 0.5, "sampling_identity": {"frozen": 1},
                "rng": {"python": 0, "numpy": 0, "torch": 0, "loader": 0, "cuda": [0]},
                "model": {"w": torch.ones(1)}, "optimizer": {"state": {0: {"exp_avg": torch.ones(1)}}}}
        best = copy.deepcopy(last)
        best.update(epoch=1, history=copy.deepcopy(history[:1]))
        return last, best, SimpleNamespace(values={"frozen": 1}), history, actual, planned, 2

    def test_accepts_committed_prefix_and_first_clean_val_tie(self):
        validate_prefix(*self.fixture())

    def test_rejects_later_checkpoint_tied_for_best(self):
        args = list(self.fixture())
        args[0]["best_epoch"] = 2
        args[1].update(epoch=2, history=copy.deepcopy(args[3]))
        with self.assertRaises(ValueError):
            validate_prefix(*args)

    def test_rejects_changed_mode_epoch_or_test_scope(self):
        for key, changed in (("robust_mode", "b0e"), ("epoch", 3), ("file_sha256", {"train": {}, "val": {}, "test": {}})):
            args = list(self.fixture())
            args[0][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prefix(*args)

    def test_rejects_missing_cuda_or_loader_rng(self):
        for key in ("cuda", "loader"):
            args = list(self.fixture())
            del args[0]["rng"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prefix(*args)

    def test_rejects_nonfinite_model_or_optimizer(self):
        for optimizer in (False, True):
            args = list(self.fixture())
            if optimizer:
                args[0]["optimizer"]["state"][0]["exp_avg"][0] = float("inf")
            else:
                args[0]["model"]["w"][0] = float("nan")
            with self.subTest(optimizer=optimizer), self.assertRaises(FloatingPointError):
                validate_prefix(*args)

    def test_rng_hash_includes_loader_and_cuda_tensor_states(self):
        state = {"python": [1], "numpy": [2], "torch": torch.tensor([3], dtype=torch.uint8),
                 "loader": torch.tensor([4], dtype=torch.uint8), "cuda": [torch.tensor([5], dtype=torch.uint8)]}
        digest = rng_digest(state)
        self.assertEqual(digest, rng_digest(copy.deepcopy(state)))
        for name in ("torch", "loader", "cuda"):
            changed = copy.deepcopy(state)
            if name == "cuda":
                changed[name][0][0] += 1
            else:
                changed[name][0] += 1
            with self.subTest(name=name):
                self.assertNotEqual(digest, rng_digest(changed))


if __name__ == "__main__":
    unittest.main()
