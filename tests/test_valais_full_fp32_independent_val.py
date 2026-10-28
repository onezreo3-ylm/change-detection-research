import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_valais_full_fp32 import validate_certificate, wait_for_guard


class FullFp32IndependentValTests(unittest.TestCase):
    def test_certificate_requires_explicit_audit_recovery_and_test_closure(self):
        certificate = {
            "status": "audit_recovered_guarded_completion_only",
            "trainer_completed_100_epochs": True,
            "original_launcher_completed_normally": False,
            "original_run_guard_completion_exists": False,
            "final_guard_status": "training_process_exited",
            "test_accessed": False,
            "model_inference_performed": False,
            "source_and_input_bindings_sha256": {"source": "hash"},
        }
        validate_certificate(certificate)
        certificate["original_launcher_completed_normally"] = True
        with self.assertRaises(RuntimeError):
            validate_certificate(certificate)
        certificate["original_launcher_completed_normally"] = False
        certificate["test_accessed"] = True
        with self.assertRaises(RuntimeError):
            validate_certificate(certificate)

    def test_guard_handshake_requires_exact_certificate_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guard_attached.json"
            path.write_text(json.dumps({"status": "guard_attached",
                                        "certificate_sha256": "abc"}), encoding="utf-8")
            wait_for_guard(path, "abc")
            with self.assertRaises(RuntimeError):
                wait_for_guard(path, "different")


if __name__ == "__main__":
    unittest.main()
