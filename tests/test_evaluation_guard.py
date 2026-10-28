"""Mock-only guard engineering checks; no sensor, CUDA model, or research files."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from pcdc.frozen_validation import EvaluationConfig, ReviewRequired
from scripts import evaluate_guarded as guarded


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "guard.yaml"
        self.sensor_path = self.root / "nvidia-smi.exe"
        self.sensor_path.write_bytes(b"mock sensor executable; never execute")
        self.guard_source = self.root / "mock_guard.py"
        self.guard_source.write_bytes(b"# synthetic test-only source snapshot\n")
        self.values = {
            "schema_version": 1, "nvidia_smi": str(self.sensor_path), "gpu_index": 0,
            "gpu_uuid": "GPU-8ada1895-7628-bb88-1dda-c57679117fa2",
            "poll_seconds": 10, "confirmation_seconds": 5, "confirmed_pause_c": 85,
            "immediate_pause_c": 86, "confirmation_samples": 3, "max_sensor_failures": 3,
            "sensor_timeout_seconds": 5,
        }
        self._write_config()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.object(guarded, "_guard_sources", return_value={
            "scripts/mock_guard.py": self.guard_source}))
        self.output = self.root / "evaluation"
        self.clock = FakeClock()

    def _write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.values), encoding="utf-8")

    def _controller(self, readings):
        self.output.mkdir()
        self.reader = Mock(side_effect=readings)
        return guarded.EvaluationTemperatureGuard(
            guarded.load_guard_config(self.config_path), self.output, read=self.reader,
            clock=self.clock, sleep=self.clock.sleep)

    def _state(self):
        return json.loads((self.output / "temperature_guard_state.json").read_text(encoding="utf-8"))

    def _fake_bundle(self, *, device="cuda", approval=None):
        return SimpleNamespace(
            protocol=SimpleNamespace(base=SimpleNamespace(values={"runtime": {"device": device}})),
            fingerprints={"fixture": "synthetic metadata only"}, approval=approval)

    def _mock_run(self, evaluator, *, read=None, clean_only=False):
        job = EvaluationConfig({"output_dir": str(self.output)}, self.root / "job.yaml")
        with patch.object(guarded, "load_evaluation_config", return_value=job), \
                patch.object(guarded, "load_bundle", return_value=self._fake_bundle()), \
                patch.object(guarded, "evaluate", side_effect=evaluator):
            return guarded.guarded_evaluate(
                job.source, self.config_path, clean_only=clean_only,
                read=read if read is not None else Mock(return_value=70),
                clock=self.clock, sleep=self.clock.sleep)

    def test_normal_polling_is_throttled_without_sleep_and_records_provenance(self):
        controller = self._controller([76, 84])
        controller()
        self.clock.now = 9.99
        controller()
        self.assertEqual(self.reader.call_count, 1)
        self.clock.now = 10.0
        controller()
        self.assertEqual(self.reader.call_count, 2)
        self.assertEqual(self.clock.sleeps, [])
        state = self._state()
        self.assertEqual(state["callback_count"], 3)
        self.assertEqual([row["temperature_c"] for row in state["observations"]], [76, 84])
        self.assertEqual(state["guard_config_sha256"], guarded._sha(self.config_path.read_bytes()))
        self.assertEqual((self.output / "temperature_guard/config.yaml").read_bytes(),
                         self.config_path.read_bytes())
        self.assertEqual((self.output / "temperature_guard/source/scripts/mock_guard.py").read_bytes(),
                         self.guard_source.read_bytes())

    def test_85_requires_three_samples_five_seconds_apart_and_stops(self):
        controller = self._controller([85, 85, 85])
        with self.assertRaisesRegex(guarded.GuardStopped, "stopped_temperature"):
            controller()
        self.assertEqual(self.clock.sleeps, [5, 5])
        self.assertEqual([row["monotonic_seconds"] for row in self._state()["observations"]],
                         [0.0, 5.0, 10.0])
        self.assertEqual(self._state()["status"], "stopped_temperature")
        with self.assertRaisesRegex(guarded.GuardError, "cannot be reused"):
            controller()
        self.assertEqual(self.reader.call_count, 3)

    def test_86_stops_immediately_without_sleep(self):
        controller = self._controller([86])
        with self.assertRaises(guarded.GuardStopped):
            controller()
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(self.reader.call_count, 1)

    def test_cool_sample_resets_confirmation_and_allows_next_batch(self):
        controller = self._controller([85, 85, 84, 85, 84])
        controller()
        self.assertEqual(self.clock.sleeps, [5, 5])
        self.assertEqual(self._state()["hot_samples"], 0)
        self.clock.now += 10
        controller()
        self.assertEqual(self.clock.sleeps, [5, 5, 5])
        self.assertEqual(self._state()["status"], "monitoring")

    def test_three_sensor_failures_stop_before_allowing_a_batch(self):
        controller = self._controller([OSError("missing sensor")] * 3)
        with self.assertRaisesRegex(guarded.GuardStopped, "stopped_sensor_unavailable"):
            controller()
        self.assertEqual(self.clock.sleeps, [10, 10])
        self.assertEqual(self._state()["sensor_failures"], 3)
        self.assertTrue(all(row["temperature_c"] is None for row in self._state()["observations"]))

    def test_sensor_recovery_resets_failures_and_invalid_values_count_as_failures(self):
        controller = self._controller([True, 60.5, 70])
        controller()
        self.assertEqual(self.clock.sleeps, [10, 10])
        self.assertEqual(self._state()["sensor_failures"], 0)
        self.assertEqual(self._state()["temperature_c"], 70)

    def test_identity_error_stops_without_sensor_retries(self):
        controller = self._controller([guarded.GuardIdentityError("wrong GPU")])
        with self.assertRaisesRegex(guarded.GuardIdentityError, "wrong GPU"):
            controller()
        self.assertEqual(self.reader.call_count, 1)
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(self._state()["status"], "failed_guard")

    def test_config_tampering_between_batches_rejects_even_before_poll_deadline(self):
        controller = self._controller([70])
        controller()
        self.config_path.write_text(self.config_path.read_text() + "\n# altered\n", encoding="utf-8")
        self.clock.now = 1
        with self.assertRaisesRegex(guarded.GuardIdentityError, "configuration changed"):
            controller()
        self.assertEqual(self.reader.call_count, 1)
        self.assertEqual(self._state()["status"], "failed_guard")

    def test_config_mutation_during_sensor_read_is_caught_before_any_batch(self):
        controller = self._controller([])

        def tamper(values):
            self.config_path.write_bytes(self.config_path.read_bytes() + b"\n# changed\n")
            return 70

        controller.read = tamper
        with self.assertRaisesRegex(guarded.GuardIdentityError, "configuration changed"):
            controller()
        self.assertEqual(self._state()["status"], "failed_guard")

    def test_source_tampering_and_visibility_remapping_fail_closed(self):
        controller = self._controller([70])
        controller()
        self.guard_source.write_bytes(b"# modified mock source")
        with self.assertRaisesRegex(guarded.GuardIdentityError, "source changed"):
            controller()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(guarded.GuardIdentityError, "remapping"):
                guarded.load_guard_config(self.config_path)

    def test_persistence_failure_prevents_sensor_or_forward_work(self):
        controller = self._controller([70])
        with patch.object(guarded, "_atomic_json", side_effect=guarded.GuardPersistenceError("disk full")):
            with self.assertRaises(guarded.GuardPersistenceError):
                controller()
        self.reader.assert_not_called()
        self.assertNotEqual(controller.state["status"], "completed")

    def test_nonmonotonic_clock_and_keyboard_interrupt_abort(self):
        controller = self._controller([70])
        controller()
        self.clock.now = -1
        with self.assertRaisesRegex(guarded.GuardError, "nonmonotonic"):
            controller()
        self.assertEqual(self._state()["status"], "failed_guard")
        controller = guarded.EvaluationTemperatureGuard(
            guarded.load_guard_config(self.config_path), self.output / "second",
            read=Mock(side_effect=KeyboardInterrupt), clock=self.clock, sleep=self.clock.sleep)
        controller.output.mkdir()
        with self.assertRaises(KeyboardInterrupt):
            controller()
        self.assertEqual(controller.state["status"], "interrupted")

    def test_guard_config_schema_is_strict_and_does_not_query_sensor(self):
        invalid = [("schema_version", True), ("gpu_index", True), ("poll_seconds", 61),
                   ("confirmation_seconds", 11), ("confirmed_pause_c", 86),
                   ("immediate_pause_c", 120), ("confirmation_samples", 0),
                   ("max_sensor_failures", 0), ("sensor_timeout_seconds", 6),
                   ("gpu_uuid", "GPU-unknown"), ("nvidia_smi", "relative.exe")]
        original = dict(self.values)
        with patch.object(guarded.subprocess, "run", side_effect=AssertionError("sensor access")):
            for key, value in invalid:
                with self.subTest(key=key, value=value):
                    self.values = dict(original, **{key: value})
                    self._write_config()
                    with self.assertRaises(ValueError):
                        guarded.load_guard_config(self.config_path)
            self.values = dict(original, process_id=123)
            self._write_config()
            with self.assertRaises(ValueError):
                guarded.load_guard_config(self.config_path)
            self.values = original
            self._write_config()
            self.config_path.write_text(self.config_path.read_text() + "\npoll_seconds: 10\n",
                                        encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate YAML"):
                guarded.load_guard_config(self.config_path)

    def test_sensor_query_binds_single_uuid_and_uses_bounded_hidden_subprocess(self):
        response = SimpleNamespace(stdout=f"0, {self.values['gpu_uuid']}, 74\n")
        with patch.object(guarded.subprocess, "run", return_value=response) as execute:
            self.assertEqual(guarded.read_temperature(self.values), 74)
        self.assertEqual(execute.call_args.kwargs["timeout"], 5)
        self.assertTrue(execute.call_args.kwargs["check"])
        self.assertIn("--query-gpu=index,uuid,temperature.gpu", execute.call_args.args[0])
        for text in (response.stdout * 2, response.stdout.replace("0,", "1,", 1),
                     response.stdout.replace(self.values["gpu_uuid"], "GPU-other")):
            with self.subTest(response=text), patch.object(guarded.subprocess, "run", return_value=SimpleNamespace(stdout=text)):
                with self.assertRaises(guarded.GuardIdentityError):
                    guarded.read_temperature(self.values)
        with patch.object(guarded.subprocess, "run", side_effect=subprocess.TimeoutExpired("sensor", 5)):
            with self.assertRaises(subprocess.TimeoutExpired):
                guarded.read_temperature(self.values)

    def test_pending_acceptance_is_rejected_before_guard_hash_sensor_device_or_output(self):
        bindings = {key: "a" * 64 for key in (
            "protocol_sha256", "manifest_sha256", "instances_sha256", "checkpoint_sha256")}
        job_path = self.root / "pending_job.yaml"
        job = {"schema_version": 1, "name": "pending_mock_metadata",
               "protocol_config": "absent_protocol.yaml", "manifest": "absent_manifest.json",
               "acceptance": "pending.yaml", "output_dir": str(self.output),
               "preparation_environment_sha256": "a" * 64, **bindings}
        job_path.write_text(yaml.safe_dump(job), encoding="utf-8")
        (self.root / "pending.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "status": "pending", "approved_for": "validation_only",
            "visual_review": "not_completed", "reviewer": None, "notes": "Synthetic gate check.",
            **bindings}), encoding="utf-8")
        with patch.object(guarded, "load_guard_config", side_effect=AssertionError("guard before gate")), \
                patch.object(guarded, "evaluate", side_effect=AssertionError("evaluation before gate")), \
                patch("pcdc.frozen_validation.file_sha256", side_effect=AssertionError("hash before gate")):
            with self.assertRaises(ReviewRequired):
                guarded.guarded_evaluate(job_path, self.root / "nonexistent_guard.yaml")
        self.assertFalse(self.output.exists())

    def test_preflight_is_metadata_only_and_cuda_target_is_explicit(self):
        job = EvaluationConfig({"output_dir": str(self.output)}, self.root / "job.yaml")
        with patch.object(guarded, "load_evaluation_config", return_value=job), \
                patch.object(guarded, "load_bundle", return_value=self._fake_bundle()) as load, \
                patch.object(guarded, "read_temperature", side_effect=AssertionError("sensor")), \
                patch.object(guarded, "evaluate", side_effect=AssertionError("device/data")):
            result = guarded.guarded_evaluate(job.source, self.config_path, preflight_only=True)
        self.assertEqual(load.call_args.kwargs, {"require_review": False})
        self.assertEqual(result["status"], "guarded_preflight_passed")
        self.assertFalse(result["sensor_queried"])
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(guarded.GuardIdentityError, "runtime.device"):
            guarded._validate_target(self._fake_bundle(device="cpu"), guarded.load_guard_config(self.config_path))

    def test_manifest_gate_failure_propagates_before_any_guard_or_evaluator_work(self):
        job = EvaluationConfig({"output_dir": str(self.output)}, self.root / "job.yaml")
        with patch.object(guarded, "load_evaluation_config", return_value=job), \
                patch.object(guarded, "load_bundle", side_effect=ValueError("Frozen manifest SHA256 differs")), \
                patch.object(guarded, "load_guard_config", side_effect=AssertionError("guard before manifest")), \
                patch.object(guarded, "evaluate", side_effect=AssertionError("evaluate before manifest")):
            with self.assertRaisesRegex(ValueError, "Frozen manifest SHA256"):
                guarded.guarded_evaluate(job.source, self.config_path)
        self.assertFalse(self.output.exists())

    def test_completion_requires_last_sample_and_durable_commit_marker(self):
        def evaluator(job, *, clean_only, guard_callback):
            self.output.mkdir()
            self.assertTrue(clean_only)
            guard_callback()
            return {"status": "completed_clean_only", "kind": "smoke", "test_evaluated": False}

        read = Mock(side_effect=[70, 71])
        result = self._mock_run(evaluator, read=read, clean_only=True)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(result["temperature_guard"]["status"], "completed")
        self.assertEqual(self._state()["status"], "completed")
        completion = json.loads((self.output / "guard_completion.json").read_text(encoding="utf-8"))
        self.assertEqual(completion["summary_sha256"], guarded._sha((self.output / "summary.json").read_bytes()))
        self.assertEqual(completion["guard_state_sha256"], guarded._sha((self.output / "temperature_guard_state.json").read_bytes()))

    def test_overheat_and_evaluator_exceptions_do_not_leave_a_complete_result(self):
        def evaluator(job, *, clean_only, guard_callback):
            self.output.mkdir()
            guard_callback()
            raise ValueError("Synthetic evaluator failure")

        with self.assertRaisesRegex(ValueError, "Synthetic evaluator failure"):
            self._mock_run(evaluator)
        self.assertEqual(self._state()["status"], "failed_guard")
        self.assertFalse((self.output / "guard_completion.json").exists())
        self.assertEqual(json.loads((self.output / "status.json").read_text())["status"], "failed_guard")

    def test_final_sensor_failure_invalidates_already_written_underlying_summary(self):
        def evaluator(job, *, clean_only, guard_callback):
            self.output.mkdir()
            guard_callback()
            value = {"status": "completed", "kind": "smoke", "corruption_mf1": 0.1}
            (self.output / "summary.json").write_text(json.dumps(value), encoding="utf-8")
            return value

        with self.assertRaises(guarded.GuardStopped):
            self._mock_run(evaluator, read=Mock(side_effect=[70, 86]))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed_guard")
        self.assertFalse(summary["guarded_result_valid"])
        self.assertFalse((self.output / "guard_completion.json").exists())

    def test_final_commit_persistence_failure_invalidates_result_and_returns_nonzero(self):
        def evaluator(job, *, clean_only, guard_callback):
            self.output.mkdir()
            guard_callback()
            return {"status": "completed", "kind": "smoke"}

        original = guarded._atomic_json

        def fail_commit(path, payload):
            if path.name == "guard_completion.json":
                raise guarded.GuardPersistenceError("Synthetic commit disk error")
            return original(path, payload)

        with patch.object(guarded, "_atomic_json", side_effect=fail_commit):
            with self.assertRaisesRegex(guarded.GuardPersistenceError, "Synthetic commit disk error"):
                self._mock_run(evaluator)
        self.assertFalse((self.output / "guard_completion.json").exists())
        self.assertEqual(self._state()["status"], "failed_guard")
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "failed_guard")

    def test_existing_output_is_never_overwritten_and_no_callback_cannot_complete(self):
        self.output.mkdir()
        marker = self.output / "user_file.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._mock_run(Mock(side_effect=AssertionError("evaluate existing output")))
        self.assertEqual(marker.read_text(), "preserve")
        self.output = self.root / "other_output"
        with self.assertRaisesRegex(guarded.GuardError, "without running"):
            self._mock_run(Mock(return_value={"status": "completed"}))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
