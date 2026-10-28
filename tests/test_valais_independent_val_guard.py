import json
from pathlib import Path
import tempfile
import unittest

from scripts.guard_valais_independent_val import pause_validation
from scripts.launch_valais_independent_val_guarded import preflight
from scripts.valais_independent_val_entry import wait_for_attach


class FakeProcess:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class IndependentValGuardTests(unittest.TestCase):
    def test_entry_waits_for_exact_guard_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attached.json"
            path.write_text(json.dumps({
                "status": "guard_attached", "study_fingerprint": "study-a"
            }))
            wait_for_attach(path, "study-a")
            with self.assertRaises(RuntimeError):
                wait_for_attach(path, "study-b")

    def test_pause_stops_only_validation_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = FakeProcess()
            result = pause_validation(
                process, {"run_dir": temporary}, "paused_temperature_guard", 86
            )
            self.assertTrue(process.stopped)
            self.assertFalse(result["research_training_status_modified"])
            self.assertTrue((Path(temporary) / "guard_pause.json").is_file())

    def test_current_running_study_cannot_be_independently_evaluated(self):
        config = Path("configs/valais_clean_fp32val_seed42_relaunch_20261001.yaml").resolve()
        with self.assertRaises(RuntimeError):
            preflight(config)


if __name__ == "__main__":
    unittest.main()
