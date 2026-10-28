import unittest

from scripts.guard_valais_numeric_replay import pause_engineering


class FakeProcess:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class ReplayGuardTests(unittest.TestCase):
    def test_pause_marks_engineering_only(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            process = FakeProcess()
            result = pause_engineering(
                process, {"run_dir": temporary}, "paused_temperature_guard", 86
            )
            self.assertTrue(process.stopped)
            self.assertEqual(result["kind"], "engineering_numeric_replay_only")
            self.assertTrue((Path(temporary) / "guard_pause.json").is_file())


if __name__ == "__main__":
    unittest.main()
