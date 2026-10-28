"""Foreground, validation-only evaluation with a fail-closed GPU temperature guard.

Use ``python -m scripts.evaluate_guarded --config EVALUATION.yaml
--guard-config GUARD.yaml``. Preflight never queries a sensor or creates output.
The existing review gate remains mandatory for corruption scoring. No PID is
attached, no process is killed, and no checkpoint is resumed or trained.

This is a between-batch software stop policy, not a hard temperature cap. Only
the explicitly configured, single NVIDIA GPU without CUDA visibility remapping
is supported. A guarded result is complete only when guard_completion.json and
temperature_guard_state.json both record completion as well as status.json.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.evaluate_corruptions import evaluate
from pcdc.frozen_validation import load_bundle, load_evaluation_config
from scripts.temperature_guard import TemperaturePolicy


_KEYS = {"schema_version", "nvidia_smi", "gpu_index", "gpu_uuid", "poll_seconds",
         "confirmation_seconds", "confirmed_pause_c", "immediate_pause_c",
         "confirmation_samples", "max_sensor_failures", "sensor_timeout_seconds"}
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


class GuardError(RuntimeError):
    """An evaluation must not continue or be accepted as a guarded result."""


class GuardIdentityError(GuardError):
    """The sensor/device/configuration identity is not the approved one."""


class GuardPersistenceError(GuardError):
    """A durable guard record could not be written; inference must stop."""


class GuardStopped(GuardError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Evaluation stopped: {reason}; no automatic resume")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _guard_sources() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {name: root / name for name in (
        "scripts/__init__.py", "scripts/evaluate_guarded.py", "scripts/temperature_guard.py")}


@dataclass(frozen=True)
class GuardConfig:
    source: Path
    values: Mapping[str, Any]
    config_bytes: bytes
    config_sha256: str
    source_files: Mapping[str, Path]
    source_bytes: Mapping[str, bytes]
    source_sha256: Mapping[str, str]

    def verify(self) -> None:
        if self.source.read_bytes() != self.config_bytes:
            raise GuardIdentityError("Temperature guard configuration changed after loading")
        for name, path in self.source_files.items():
            if _sha(path.read_bytes()) != self.source_sha256[name]:
                raise GuardIdentityError(f"Temperature guard source changed: {name}")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise GuardIdentityError("CUDA_VISIBLE_DEVICES remapping is unsupported by this guard")


def load_guard_config(path: str | Path) -> GuardConfig:
    source = Path(path).resolve()
    raw = source.read_bytes()
    values = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(values, dict) or set(values) != _KEYS:
        raise ValueError(f"Unexpected or missing evaluation guard settings: expected {sorted(_KEYS)}")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only evaluation guard schema_version 1 is supported")
    for name in _KEYS - {"schema_version", "nvidia_smi", "gpu_uuid"}:
        minimum = 0 if name == "gpu_index" else 1
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError(f"{name} must be an explicit integer >= {minimum}")
    if not values["confirmed_pause_c"] < values["immediate_pause_c"] < 120:
        raise ValueError("Expected confirmed_pause_c < immediate_pause_c < 120")
    if not values["confirmation_seconds"] <= values["poll_seconds"] <= 60:
        raise ValueError("Expected confirmation_seconds <= poll_seconds <= 60")
    if values["sensor_timeout_seconds"] > values["confirmation_seconds"]:
        raise ValueError("Sensor timeout must not exceed the confirmation interval")
    if (not isinstance(values["gpu_uuid"], str)
            or _UUID.fullmatch(values["gpu_uuid"]) is None):
        raise ValueError("gpu_uuid must be an explicit NVIDIA GPU UUID")
    executable = values["nvidia_smi"]
    if (not isinstance(executable, str) or not Path(executable).is_absolute()
            or not Path(executable).is_file()):
        raise ValueError("nvidia_smi must be an existing absolute executable path")
    files = _guard_sources()
    contents = {name: candidate.read_bytes() for name, candidate in files.items()}
    result = GuardConfig(source, MappingProxyType(values), raw, _sha(raw),
                         MappingProxyType(files), MappingProxyType(contents),
                         MappingProxyType({name: _sha(data) for name, data in contents.items()}))
    result.verify()
    return result


def read_temperature(values: Mapping[str, Any]) -> int:
    """Query all cards to reject ambiguous multi-GPU and remapped device setups."""
    result = subprocess.run([
        values["nvidia_smi"], "--query-gpu=index,uuid,temperature.gpu",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        timeout=values["sensor_timeout_seconds"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise GuardIdentityError("This guard requires exactly one NVIDIA GPU")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 3:
        raise ValueError("Unexpected NVIDIA temperature sensor response")
    if int(fields[0]) != values["gpu_index"] or fields[1] != values["gpu_uuid"]:
        raise GuardIdentityError("NVIDIA sensor GPU index/UUID differs from guard configuration")
    temperature = int(fields[2])
    if not 0 <= temperature < 120:
        raise ValueError("NVIDIA temperature is outside the supported sensor range")
    return temperature


def _write_bytes(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as error:
        raise GuardPersistenceError(f"Cannot persist guard artifact: {path.name}") from error


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".guard-tmp")
    try:
        _write_bytes(temporary, (json.dumps(value, indent=2, ensure_ascii=False,
                                           allow_nan=False) + "\n").encode("utf-8"))
        os.replace(temporary, path)
    except Exception as error:
        raise GuardPersistenceError(f"Cannot persist guard state: {path.name}") from error


class EvaluationTemperatureGuard:
    """Synchronous callback: check before a batch; never stop another process."""

    def __init__(self, config: GuardConfig, output: Path, *,
                 read: Callable[[Mapping[str, Any]], int] | None = None,
                 clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None):
        self.config, self.output = config, output
        self.read = read if read is not None else read_temperature
        self.clock = clock if clock is not None else time.monotonic
        self.sleep = sleep if sleep is not None else time.sleep
        self.policy = TemperaturePolicy(config.values)
        self.attached = False
        self.last_probe: float | None = None
        self.last_clock: float | None = None
        self.state: dict[str, Any] = {
            "status": "starting", "started_utc": _utc_now(), "guard_pid": os.getpid(),
            "execution": "foreground_in_process_between_batches", "automatic_resume": False,
            "test_evaluated": False, "training_started": False,
            "guard_config_source": str(config.source), "guard_config_sha256": config.config_sha256,
            "guard_source_sha256": dict(config.source_sha256), "settings": dict(config.values),
            "callback_count": 0, "observations": [],
            "limitation": "Software checks between batches; not a hard temperature cap or CPU monitor.",
        }

    def _time(self) -> float:
        value = self.clock()
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (self.last_clock is not None and value < self.last_clock)):
            raise GuardError("Invalid or nonmonotonic guard clock")
        self.last_clock = float(value)
        return float(value)

    def _persist(self) -> None:
        _atomic_json(self.output / "temperature_guard_state.json", self.state)

    def _attach(self) -> None:
        # evaluate() alone creates its output, after its own full validation gate.
        if not self.output.is_dir():
            raise GuardPersistenceError("Evaluator output is missing at the first guard callback")
        directory = self.output / "temperature_guard"
        try:
            directory.mkdir(exist_ok=False)
            self.attached = True
            _write_bytes(directory / "config.yaml", self.config.config_bytes)
            for name, data in self.config.source_bytes.items():
                target = directory / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_bytes(target, data)
            _atomic_json(directory / "provenance.json", {
                "config_sha256": self.config.config_sha256,
                "source_sha256": dict(self.config.source_sha256)})
            self._persist()
        except Exception as error:
            raise GuardPersistenceError("Unable to initialize durable evaluation guard records") from error

    def __call__(self, *, force: bool = False) -> None:
        if self.state["status"] not in ("starting", "monitoring"):
            raise GuardError("A stopped or completed temperature guard cannot be reused")
        try:
            self.config.verify()
            if not self.attached:
                self._attach()
            self.state["callback_count"] += 1
            now = self._time()
            if not force and self.last_probe is not None and (
                    now - self.last_probe < self.config.values["poll_seconds"]):
                return
            # Confirmation/retry is synchronous: no new model batch runs with an
            # unresolved high reading or unavailable sensor. Normal polling never sleeps.
            while True:
                temperature, sensor_error = None, None
                try:
                    temperature = self.read(self.config.values)
                    if type(temperature) is not int or not 0 <= temperature < 120:
                        raise ValueError("Sensor must return an integer GPU temperature in [0,120)")
                except GuardIdentityError:
                    raise
                except Exception as error:
                    sensor_error = f"{type(error).__name__}: {error}"
                    temperature = None
                self.config.verify()
                self.last_probe = self._time()
                decision = self.policy.observe(temperature)
                self.state.update(status="monitoring", temperature_c=temperature,
                                  sensor_error=sensor_error, decision=decision,
                                  hot_samples=self.policy.hot_samples,
                                  sensor_failures=self.policy.sensor_failures)
                self.state["observations"].append({
                    "sample_utc": _utc_now(), "monotonic_seconds": self.last_probe,
                    "temperature_c": temperature, "sensor_error": sensor_error,
                    "decision": decision})
                self._persist()
                if decision in ("temperature_pause", "sensor_pause"):
                    raise GuardStopped("stopped_temperature" if decision == "temperature_pause"
                                       else "stopped_sensor_unavailable")
                if decision not in ("confirm", "sensor_retry"):
                    return
                interval = (self.config.values["confirmation_seconds"] if decision == "confirm"
                            else self.config.values["poll_seconds"])
                self.sleep(interval)
        except BaseException as error:
            self.abort(error)
            raise

    def abort(self, error: BaseException) -> None:
        reason = (error.reason if isinstance(error, GuardStopped) else
                  "interrupted" if isinstance(error, KeyboardInterrupt) else "failed_guard")
        self.state.update(status=reason, ended_utc=_utc_now(),
                          error=f"{type(error).__name__}: {error}")
        if self.attached:
            self._persist()

    def finish(self, result: dict[str, Any]) -> dict[str, Any]:
        self(force=True)
        self.config.verify()
        self.state.update(status="completed", ended_utc=_utc_now(), evaluation_status=result["status"])
        self._persist()
        return {"status": "completed", "config_sha256": self.config.config_sha256,
                "source_sha256": dict(self.config.source_sha256),
                "completion_file": "guard_completion.json"}


def _validate_target(bundle: Any, config: GuardConfig) -> None:
    device = bundle.protocol.base.values["runtime"]["device"]
    if device not in ("cuda", "cuda:0") or config.values["gpu_index"] != 0:
        raise GuardIdentityError("This single-GPU guard requires runtime.device cuda/cuda:0 and gpu_index 0")
    config.verify()


def _invalidate(output: Path, error: BaseException) -> None:
    failure = {"status": "failed_guard", "guarded_result_valid": False,
               "error": f"{type(error).__name__}: {error}",
               "test_evaluated": False, "training_started": False}
    summary_path = output / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        summary.update(failure)
        _atomic_json(summary_path, summary)
    _atomic_json(output / "status.json", failure)


def guarded_evaluate(config_path: str | Path, guard_path: str | Path, *,
                     preflight_only: bool = False, clean_only: bool = False,
                     read: Callable[[Mapping[str, Any]], int] | None = None,
                     clock: Callable[[], float] | None = None,
                     sleep: Callable[[float], None] | None = None) -> dict[str, Any]:
    if type(preflight_only) is not bool or type(clean_only) is not bool or (preflight_only and clean_only):
        raise ValueError("Choose at most one boolean preflight-only/clean-only mode")
    job = load_evaluation_config(config_path)
    # Missing/pending acceptance rejects before guard hashing, sensor/device work,
    # or any output. The underlying evaluator independently repeats the same gate.
    bundle = load_bundle(job, require_review=not (preflight_only or clean_only))
    config = load_guard_config(guard_path)
    _validate_target(bundle, config)
    if preflight_only:
        return {"status": "guarded_preflight_passed", "kind": "engineering_preflight_not_scoring",
                "fingerprints": bundle.fingerprints,
                "approval_status": bundle.approval.get("status") if bundle.approval else None,
                "guard_config_sha256": config.config_sha256,
                "guard_source_sha256": dict(config.source_sha256),
                "dataset_opened": False, "model_inference_run": False,
                "sensor_queried": False, "test_evaluated": False}
    output = job.path("output_dir")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite guarded evaluation: {output}")
    guard = EvaluationTemperatureGuard(config, output, read=read, clock=clock, sleep=sleep)
    try:
        result = evaluate(job, clean_only=clean_only, guard_callback=guard)
        if not guard.attached or not guard.state["observations"]:
            raise GuardError("Evaluator returned without running its temperature guard")
        result = dict(result, temperature_guard=guard.finish(result), guarded_completion_required=True)
        _atomic_json(output / "summary.json", result)
        _atomic_json(output / "status.json", result)
        # Last commit marker: missing/failed marker means no valid guarded result,
        # even if an earlier evaluator summary happened to say completed.
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": config.config_sha256,
            "guard_source_sha256": dict(config.source_sha256),
            "summary_sha256": _sha((output / "summary.json").read_bytes()),
            "guard_state_sha256": _sha((output / "temperature_guard_state.json").read_bytes())})
        return result
    except BaseException as error:
        try:
            guard.abort(error)
        finally:
            # Never write into an existing/concurrently-created output that the
            # callback has not attached to. Pre-callback evaluator errors retain
            # the evaluator's own failed status, and cannot have a completion marker.
            if guard.attached:
                _invalidate(output, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()
    result = guarded_evaluate(args.config, args.guard_config,
                              preflight_only=args.preflight_only, clean_only=args.clean_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
