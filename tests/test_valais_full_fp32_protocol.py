import copy
from pathlib import Path
import unittest

from pcdc.config import load_config
from scripts.authorize_valais_full_fp32 import _same_except_numeric


class FullFp32ProtocolTests(unittest.TestCase):
    def test_only_training_precision_and_run_identity_change(self):
        original = load_config(Path("configs/valais_clean_development_seed42.yaml"))
        proposed = load_config(Path("configs/valais_clean_full_fp32_seed42_20261002.yaml"))
        self.assertTrue(_same_except_numeric(original.values, proposed.values))
        self.assertTrue(original.values["train"]["amp"])
        self.assertFalse(proposed.values["train"]["amp"])
        changed = copy.deepcopy(proposed.values)
        changed["train"]["batch_size"] = 8
        self.assertFalse(_same_except_numeric(original.values, changed))
        self.assertFalse(proposed.run_test_after_training)


if __name__ == "__main__":
    unittest.main()
