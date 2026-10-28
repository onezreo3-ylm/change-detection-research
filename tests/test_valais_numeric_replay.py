import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from scripts.diagnose_valais_epoch34 import (
    _hash, _logit_diagnostic, load_replay_spec, parse_training_trace, verify_seal,
)


class ValaisNumericReplayTests(unittest.TestCase):
    def test_bound_spec_refuses_output_outside_independent_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = {}
            for name in ("source_config", "checkpoint", "failure_audit", "source_data_audit"):
                path = root / name
                path.write_bytes(name.encode())
                fields[name] = path.name
                fields[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = {
                "schema_version": 1,
                "kind": "engineering_numeric_replay_only",
                **fields,
                "output_dir": "new_output",
                "replay_epoch": 34,
                "expected_train_loss_rounded_6": 0.1,
                "expected_train_amp_skips": 1,
                "expected_train_optimizer_steps": 3,
                "expected_train_trace": {step: 0.1 for step in
                                         (1, 275, 550, 825, 1100, 1375, 1650,
                                          1925, 2200, 2475, 2750, 2757)},
                "expected_amp_skip": {"step": 1836, "scale_before": 131072,
                                      "scale_after": 65536},
                "expected_val_step35_loss_rounded_6": 0.2,
                "expected_val_trace": {step: 0.2 for step in
                                       (1, 5, 10, 15, 20, 25, 30, 35)},
                "first_possible_failure_step": 36,
                "last_possible_failure_step": 40,
                "guard": {"gpu_index": 0, "gpu_uuid": "GPU-fixture",
                          "confirmed_pause_c": 85, "immediate_pause_c": 86,
                          "poll_seconds": 10, "confirmation_seconds": 5,
                          "confirmation_samples": 3, "max_sensor_failures": 3},
            }
            config = root / "replay.yaml"
            config.write_text(yaml.safe_dump(spec), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_replay_spec(config)
            _hash(root / "checkpoint", fields["checkpoint_sha256"])
            (root / "checkpoint").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                _hash(root / "checkpoint", fields["checkpoint_sha256"])

    def test_trace_parser_tracks_intermediate_losses_and_skip_location(self):
        content = (
            "replay_epoch34/train step=1/2757 loss=0.000086\n"
            "replay_epoch34/train step=1836/2757 amp_overflow_skip=1 "
            "scale=131072->65536\n"
        )
        losses, skips = parse_training_trace(content)
        self.assertEqual(losses, {1: "0.000086"})
        self.assertEqual(skips, [{"step": 1836, "scale_before": 131072,
                                  "scale_after": 65536}])

    def test_logit_diagnostic_counts_nonfinite_values(self):
        values = torch.tensor([[[[1.0, float("nan"), float("inf"), float("-inf")]]]])
        result = _logit_diagnostic(values)
        self.assertEqual(result["finite_count"], 1)
        self.assertEqual(result["nan_count"], 1)
        self.assertEqual(result["posinf_count"], 1)
        self.assertEqual(result["neginf_count"], 1)
        self.assertEqual(result["finite_min"], 1.0)

    def test_preflight_seal_binds_diagnostic_source_bytes(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "spec.yaml"
            bound = root / "source.py"
            seal_path = root / "seal.json"
            spec.write_bytes(b"kind: engineering_numeric_replay_only\n")
            bound.write_bytes(b"print('stable')\n")
            seal = {
                "status": "engineering_replay_preflight_sealed",
                "test_accessed": False,
                "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                "bound_sha256": {
                    str(bound): hashlib.sha256(bound.read_bytes()).hexdigest()
                },
            }
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            self.assertEqual(verify_seal(seal_path, spec), seal)
            bound.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                verify_seal(seal_path, spec)


if __name__ == "__main__":
    unittest.main()
