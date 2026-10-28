import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from scripts.diagnose_valais_epoch46 import DiagnosticLoss, load_replay, tensor_counts


class Epoch46ReplayTests(unittest.TestCase):
    def test_rejects_changed_window_before_touching_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.yaml"
            path.write_text("schema_version: 1\nkind: isolated_train_numeric_replay\n"
                            "failure_window: [1, 2]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_replay(path)

    def test_nonfinite_logit_counts(self):
        stats = tensor_counts(torch.tensor([0.0, math.nan, math.inf, -math.inf]))
        self.assertEqual((stats["finite"], stats["nan"], stats["posinf"],
                          stats["neginf"]), (1, 1, 1, 1))

    def test_prefix_mismatch_stops_before_training_update(self):
        base = SimpleNamespace(values={"loss": {
            "bce_weight": 1.0, "dice_weight": 1.0, "dice_smooth": 1.0}})
        window = SimpleNamespace(step=1)
        loss = DiagnosticLoss(base, None, window, {1: 99.0}, "contiguous")
        with self.assertRaisesRegex(RuntimeError, "Trajectory mismatch"):
            loss(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
