import json
from pathlib import Path
import tempfile
import unittest

from scripts.authorize_valais_epoch38_resume import (
    validate_interrupted_status, validate_probe,
)


class Epoch38ResumeAuthorizationTests(unittest.TestCase):
    def test_interrupted_status_allows_omitted_test_field_but_rejects_true(self):
        status = {"status": "interrupted", "epoch": 38,
                  "manual_resume_required": True}
        validate_interrupted_status(status)
        with self.assertRaises(RuntimeError):
            validate_interrupted_status({**status, "test_evaluated": True})

    def test_liveness_probe_must_finish_over_an_hour_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "completion.json"
            record = {
                "status": "completed", "kind": "scheduler_liveness_no_gpu_no_data",
                "parent_name": "svchost.exe", "data_accessed": False,
                "gpu_accessed": False, "duration_seconds": 3900.0,
            }
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(record).encode())
            self.assertEqual(validate_probe(path, 3600), record)
            record["duration_seconds"] = 3599
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_probe(path, 3600)


if __name__ == "__main__":
    unittest.main()
