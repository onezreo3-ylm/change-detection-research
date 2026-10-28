import json
from pathlib import Path
import tempfile
import unittest

from scripts.reclassify_valais_orphan_20261001 import (
    interrupted_status,
    read_scheduler_probe,
)


class OrphanRecoveryTests(unittest.TestCase):
    def test_scheduler_probe_accepts_utf8_bom_and_rejects_codex_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.json"
            probe = {
                "status": "completed", "parent_name": "svchost.exe",
                "gpu_accessed": False, "data_accessed": False,
            }
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(probe).encode())
            self.assertEqual(read_scheduler_probe(path), probe)
            probe["parent_name"] = "ChatGPT.exe"
            path.write_text(json.dumps(probe), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                read_scheduler_probe(path)

    def test_administrative_status_transition_is_explicit_and_bounded(self):
        old = {
            "status": "running", "epoch": 27, "kind": "research",
            "test_evaluated": False,
        }
        new = interrupted_status(old)
        self.assertEqual(old["status"], "running")
        self.assertEqual(new["status"], "interrupted")
        self.assertTrue(new["manual_resume_required"])
        with self.assertRaises(RuntimeError):
            interrupted_status({**old, "epoch": 26})


if __name__ == "__main__":
    unittest.main()
