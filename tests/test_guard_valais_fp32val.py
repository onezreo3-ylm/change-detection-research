from pathlib import Path
import tempfile
import unittest

import torch

from scripts.guard_valais_fp32val import _research_output, pause_research


class FakeProcess:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class ValaisFp32GuardTests(unittest.TestCase):
    def test_resume_session_maps_to_research_run(self):
        root = Path("runs/example")
        self.assertEqual(
            _research_output(root / "temperature_guard_sessions" / "session_0001"),
            root,
        )
        self.assertEqual(_research_output(root), root)

    def test_pause_preserves_complete_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            torch.save({
                "format_version": 2, "validation_amp": False,
                "test_evaluated": False, "epoch": 4, "best_epoch": 3,
                "best_val_f1": 0.2,
            }, output / "last.pt")
            process = FakeProcess()
            result = pause_research(
                process, {"run_dir": str(output)}, "paused_temperature_guard", 86
            )
            self.assertTrue(process.stopped)
            self.assertEqual(result["epoch"], 4)
            self.assertTrue(result["manual_resume_required"])
            self.assertTrue((output / "status.json").is_file())


if __name__ == "__main__":
    unittest.main()
