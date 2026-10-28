import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_valais_fp32val import verify_guarded_completion


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValaisIndependentEvaluationGuardTests(unittest.TestCase):
    def test_completion_requires_guard_seal_and_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run = parent / "valais_clean_fp32val_seed42_20260930"
            run.mkdir()
            fingerprint = "f" * 64
            source = parent / "source.py"
            source.write_bytes(b"stable")
            for name in ("best.pt", "last.pt"):
                (run / name).write_bytes(name.encode())
            (run / "status.json").write_text(json.dumps({"status": "completed"}))
            state = run / "temperature_guard_state.json"
            state.write_text(json.dumps({"status": "training_process_exited"}))
            seal = parent / f"{run.name}.preflight_seal.json"
            seal.write_text(json.dumps({
                "status": "independent_numeric_protocol_preflight_sealed",
                "fingerprint": fingerprint,
                "test_accessed": False,
                "bound_sha256": {str(source): sha(source)},
            }))
            completion = {
                "status": "guarded_independent_clean_completed",
                "test_accessed": False,
                "study_fingerprint": fingerprint,
                "guard_state_path": str(state),
                "preflight_seal_path": str(seal),
                "status_sha256": sha(run / "status.json"),
                "best_checkpoint_sha256": sha(run / "best.pt"),
                "last_checkpoint_sha256": sha(run / "last.pt"),
                "guard_state_sha256": sha(state),
                "preflight_seal_sha256": sha(seal),
            }
            (run / "guard_completion.json").write_text(json.dumps(completion))
            verify_guarded_completion(run, fingerprint)
            source.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                verify_guarded_completion(run, fingerprint)

    def test_missing_completion_refuses_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                verify_guarded_completion(Path(temporary), "f" * 64)

    def test_resume_completion_requires_matching_guard_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run = parent / "valais_clean_fp32val_seed42_20260930"
            session = run / "temperature_guard_sessions" / "session_0001"
            session.mkdir(parents=True)
            fingerprint = "a" * 64
            source = parent / "source.py"
            source.write_bytes(b"stable")
            for name in ("best.pt", "last.pt", "status.json"):
                (run / name).write_bytes(name.encode())
            state = session / "temperature_guard_state.json"
            state.write_text(json.dumps({"status": "training_process_exited"}))
            seal = parent / f"{run.name}.session_0001.preflight_seal.json"
            seal.write_text(json.dumps({
                "status": "independent_numeric_protocol_preflight_sealed",
                "fingerprint": fingerprint,
                "test_accessed": False,
                "bound_sha256": {str(source): sha(source)},
            }))
            completion = {
                "status": "guarded_independent_clean_completed",
                "test_accessed": False,
                "study_fingerprint": fingerprint,
                "guard_state_path": str(state),
                "preflight_seal_path": str(seal),
                "status_sha256": sha(run / "status.json"),
                "best_checkpoint_sha256": sha(run / "best.pt"),
                "last_checkpoint_sha256": sha(run / "last.pt"),
                "guard_state_sha256": sha(state),
                "preflight_seal_sha256": sha(seal),
            }
            marker = run / "guard_completion.json"
            marker.write_text(json.dumps(completion))
            verify_guarded_completion(run, fingerprint)
            wrong = run / "temperature_guard_state.json"
            wrong.write_bytes(state.read_bytes())
            completion["guard_state_path"] = str(wrong)
            marker.write_text(json.dumps(completion))
            with self.assertRaises(RuntimeError):
                verify_guarded_completion(run, fingerprint)


if __name__ == "__main__":
    unittest.main()
