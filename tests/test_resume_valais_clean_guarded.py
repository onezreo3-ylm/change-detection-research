"""Synthetic-only checks for the separate PID-guarded Valais resume launcher."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.resume_valais_clean_guarded import paused_boundary, sha256


class ValaisResumeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = {"status": "paused_user", "kind": "research",
                       "manual_resume_required": True, "epoch": 6}
        self.checkpoint = {"format_version": 1, "kind": "research", "epoch": 6,
                           "history": [{"epoch": number} for number in range(1, 7)]}

    def test_accepts_complete_paused_epoch(self) -> None:
        paused_boundary(self.status, self.checkpoint, expected_epoch=6)

    def test_rejects_wrong_status_epoch_or_partial_history(self) -> None:
        for key, value in (("status", "running"), ("epoch", 5),
                           ("manual_resume_required", False)):
            with self.subTest(key=key):
                changed = {**self.status, key: value}
                with self.assertRaisesRegex(ValueError, "not paused"):
                    paused_boundary(changed, self.checkpoint, expected_epoch=6)
        changed_checkpoint = {**self.checkpoint, "history": self.checkpoint["history"][:-1]}
        with self.assertRaisesRegex(ValueError, "complete epoch"):
            paused_boundary(self.status, changed_checkpoint, expected_epoch=6)

    def test_streaming_sha_matches_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            data = b"synthetic-resume-fixture" * 100
            path.write_bytes(data)
            self.assertEqual(sha256(path), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
