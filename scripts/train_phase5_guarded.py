"""Run approved Phase 5 actor or adapter training under the GPU guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.phase5_protocol import load_phase56_config
from pcdc.phase5_training import (
    actor_preflight,
    adapter_preflight,
    train_actor_detector,
    train_adapter,
)
from scripts.evaluate_guarded import GuardIdentityError, _atomic_json, load_guard_config
from scripts.train_robust_guarded import TrainingTemperatureGuard


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output(config, stage: str) -> Path:
    value = config.values["actor_detector" if stage == "actor" else "adapter"]["output_dir"]
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else config.source.parent / candidate).resolve()


def _validate_target(config, guard) -> None:
    if config.base.values["runtime"]["device"] not in ("cuda", "cuda:0"):
        raise GuardIdentityError("Phase 5 training requires runtime.device cuda/cuda:0")
    if guard.values["gpu_index"] != 0:
        raise GuardIdentityError("Phase 5 training is frozen to physical GPU index 0")
    guard.verify()


def _guard_output(output: Path, resume: bool) -> Path:
    if not resume:
        return output
    if not output.is_dir() or not (output / "last.pt").is_file():
        raise FileNotFoundError("Phase 5 resume requires an existing last.pt")
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get(
            "status") == "completed":
        raise ValueError("Completed Phase 5 training must not be resumed")
    sessions = output / "temperature_guard_resumes"
    sessions.mkdir(exist_ok=True)
    index = 1
    while (sessions / f"session_{index:04d}").exists():
        index += 1
    session = sessions / f"session_{index:04d}"
    session.mkdir(exist_ok=False)
    return session


def guarded_train(config_path: Path, guard_path: Path, *, stage: str,
                  resume: bool = False, preflight_only: bool = False) -> dict:
    if stage not in ("actor", "adapter"):
        raise ValueError("stage must be actor or adapter")
    config = load_phase56_config(config_path, require_approval=True)
    guard_config = load_guard_config(guard_path)
    _validate_target(config, guard_config)
    preflight_fn = actor_preflight if stage == "actor" else adapter_preflight
    train_fn = train_actor_detector if stage == "actor" else train_adapter
    if preflight_only:
        result = preflight_fn(config)
        return {**result, "status": f"guarded_phase5_{stage}_preflight_passed",
                "guard_config_sha256": guard_config.config_sha256,
                "sensor_queried": False}
    output = _output(config, stage)
    guard_output = _guard_output(output, resume)
    guard = TrainingTemperatureGuard(guard_config, guard_output)
    try:
        result = train_fn(config, resume=resume, guard_callback=guard)
        guard_record = guard.finish_training(result)
        guard_record["state_path"] = str(
            (guard_output / "temperature_guard_state.json").relative_to(output))
        status = dict(result, temperature_guard=guard_record,
                      guarded_completion_required=True)
        _atomic_json(output / "status.json", status)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(), "stage": stage,
            "guard_config_sha256": guard_config.config_sha256,
            "training_status_sha256": _sha(output / "status.json"),
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
    parser.add_argument("--stage", choices=("actor", "adapter"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.preflight_only:
        raise ValueError("Preflight and resume are mutually exclusive")
    result = guarded_train(args.config, args.guard_config, stage=args.stage,
                           resume=args.resume, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
