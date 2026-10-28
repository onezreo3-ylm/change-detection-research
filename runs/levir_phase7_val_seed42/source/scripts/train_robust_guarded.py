"""Run one matched robust baseline with an in-process GPU temperature guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.robust_baselines import load_robust_config, preflight, train_robust
from scripts.evaluate_guarded import (
    EvaluationTemperatureGuard,
    GuardIdentityError,
    _atomic_json,
    load_guard_config,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingTemperatureGuard(EvaluationTemperatureGuard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.update(
            execution="foreground_in_process_training_between_batches",
            training_started=True,
        )

    def finish_training(self, result: dict) -> dict:
        self(force=True)
        self.config.verify()
        self.state.pop("evaluation_status", None)
        self.state.update(status="completed", ended_utc=_utc_now(),
                          training_status=result["status"])
        self._persist()
        return {
            "status": "completed",
            "config_sha256": self.config.config_sha256,
            "source_sha256": dict(self.config.source_sha256),
        }


def _validate_target(config, guard) -> None:
    device = config.base.values["runtime"]["device"]
    if device not in ("cuda", "cuda:0") or guard.values["gpu_index"] != 0:
        raise GuardIdentityError("Robust guard requires runtime.device cuda/cuda:0 and gpu_index 0")
    guard.verify()


def _guard_output(config, resume: bool) -> Path:
    """Keep every interrupted guard record; a resume gets a fresh session."""
    if not resume:
        return config.output_dir
    output = config.output_dir
    if not output.is_dir() or not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires an existing robust last.pt")
    status_path = output / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            raise ValueError("Completed robust training must not be resumed")
    sessions = output / "temperature_guard_resumes"
    sessions.mkdir(exist_ok=True)
    index = 1
    while (sessions / f"session_{index:04d}").exists():
        index += 1
    session = sessions / f"session_{index:04d}"
    session.mkdir(exist_ok=False)
    return session


def guarded_train(config_path: Path, guard_path: Path, *, resume: bool = False,
                  preflight_only: bool = False) -> dict:
    config = load_robust_config(config_path)
    guard_config = load_guard_config(guard_path)
    _validate_target(config, guard_config)
    if preflight_only:
        result = preflight(config)
        return {
            **result,
            "status": "guarded_robust_preflight_passed",
            "guard_config_sha256": guard_config.config_sha256,
            "sensor_queried": False,
        }
    guard_output = _guard_output(config, resume)
    guard = TrainingTemperatureGuard(guard_config, guard_output)
    try:
        result = train_robust(config, resume=resume, guard_callback=guard)
        guard_record = guard.finish_training(result)
        guard_record["state_path"] = str(
            (guard_output / "temperature_guard_state.json").relative_to(config.output_dir))
        status = dict(result, temperature_guard=guard_record,
                      guarded_completion_required=True)
        _atomic_json(config.output_dir / "status.json", status)
        _atomic_json(config.output_dir / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": guard_config.config_sha256,
            "training_status_sha256": _sha(config.output_dir / "status.json"),
            "guard_state_path": guard_record["state_path"],
            "guard_state_sha256": _sha(guard_output / "temperature_guard_state.json"),
            "wrapper_source_sha256": _sha(Path(__file__)),
        })
        return status
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.preflight_only:
        raise ValueError("Preflight and resume are mutually exclusive")
    result = guarded_train(args.config, args.guard_config, resume=args.resume,
                           preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
