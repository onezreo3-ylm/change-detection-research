import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.launch_valais_fp32val import _session
from scripts.train_valais_fp32val import (
    _read_checkpoint, _snapshot, _verify_snapshot, _wait_guard, load_study,
)


class ValaisFp32ValidationTests(unittest.TestCase):
    def test_frozen_new_study_keeps_train_amp_and_val_fp32(self):
        path = Path("configs/valais_clean_fp32val_seed42_20260930.yaml").resolve()
        spec, paths, base, fingerprint = load_study(path)
        self.assertTrue(spec["training_amp"])
        self.assertFalse(spec["validation_amp"])
        self.assertFalse(base.run_test_after_training)
        self.assertEqual(spec["epochs"], 100)
        self.assertEqual(len(fingerprint), 64)

    def test_os_shutdown_relaunch_has_distinct_output_and_bound_addendum(self):
        path = Path("configs/valais_clean_fp32val_seed42_relaunch_20261001.yaml").resolve()
        spec, paths, base, fingerprint = load_study(path)
        self.assertTrue(paths["output_dir"].name.endswith("relaunch_20261001"))
        self.assertIn("prior_launch_failure_addendum", paths)
        self.assertTrue(paths["prior_launch_failure_addendum"].is_file())
        self.assertFalse(base.run_test_after_training)

    def test_guard_handshake_requires_matching_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attached.json"
            path.write_text(json.dumps({
                "status": "guard_attached", "study_fingerprint": "wrong"
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _wait_guard(path, "expected")
            path.write_text(json.dumps({
                "status": "guard_attached", "study_fingerprint": "expected"
            }), encoding="utf-8")
            _wait_guard(path, "expected")

    def test_old_checkpoint_format_cannot_resume_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.pt"
            torch.save({"format_version": 1}, path)
            with self.assertRaises(ValueError):
                _read_checkpoint(path)

    def test_source_snapshot_is_strict_for_resume(self):
        config_path = Path("configs/valais_clean_fp32val_seed42_20260930.yaml").resolve()
        _, _, base, fingerprint = load_study(config_path)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            hashes = _snapshot(output, config_path, base, fingerprint)
            _verify_snapshot(output, hashes)
            archived = output / "source" / "scripts" / "train_valais_fp32val.py"
            archived.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                _verify_snapshot(output, hashes)

    def test_guarded_resume_only_accepts_paused_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "last.pt").write_bytes(b"fixture")
            (output / "status.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(RuntimeError):
                _session(output, True)
            (output / "status.json").write_text(
                json.dumps({"status": "paused_temperature_guard"})
            )
            session = _session(output, True)
            self.assertEqual(session.name, "session_0001")
            self.assertTrue(session.is_dir())
            (output / "status.json").write_text(json.dumps({"status": "paused_user"}))
            self.assertEqual(_session(output, True).name, "session_0002")


if __name__ == "__main__":
    unittest.main()
