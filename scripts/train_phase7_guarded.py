"""Train one approved Phase 7 selector under the in-process GPU guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.phase7_protocol import load_phase7_config
from pcdc.phase7_training import VARIANTS, selector_preflight, train_selector
from scripts.evaluate_guarded import GuardIdentityError, _atomic_json, load_guard_config
from scripts.train_robust_guarded import TrainingTemperatureGuard


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guard_output(output: Path, resume: bool) -> Path:
    if not resume:
        return output
    if not output.is_dir() or not (output / "last.pt").is_file():
        raise FileNotFoundError("Phase 7 resume requires an existing last.pt")
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text(
            encoding="utf-8")).get("status") == "completed":
        raise ValueError("Completed Phase 7 selector must not be resumed")
    sessions = output / "temperature_guard_resumes"
    sessions.mkdir(exist_ok=True)
    index = 1
    while (sessions / f"session_{index:04d}").exists():
        index += 1
    session = sessions / f"session_{index:04d}"
    session.mkdir(exist_ok=False)
    return session


def guarded_train(config_path: Path, guard_path: Path, *, variant: str,
                  resume: bool = False, preflight_only: bool = False) -> dict:
    config = load_phase7_config(config_path)
    guard_config = load_guard_config(guard_path)
    if (config.base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard_config.values["gpu_index"] != 0):
        raise GuardIdentityError("Phase 7 training is frozen to CUDA physical GPU index 0")
    guard_config.verify()
    if preflight_only:
        result = selector_preflight(config, variant)
        return {**result, "status": "guarded_phase7_selector_preflight_passed",
                "guard_config_sha256": guard_config.config_sha256,
                "sensor_queried": False}
    output = config.training_output(variant)
    guard_output = _guard_output(output, resume)
    guard = TrainingTemperatureGuard(guard_config, guard_output)
    try:
        result = train_selector(config, variant, resume=resume, guard_callback=guard)
        guard_record = guard.finish_training(result)
        guard_record["state_path"] = str(
            (guard_output / "temperature_guard_state.json").relative_to(output))
        status = dict(result, temperature_guard=guard_record,
                      guarded_completion_required=True)
        _atomic_json(output / "status.json", status)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "stage": "phase7_selector", "variant": variant,
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
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.preflight_only:
        raise ValueError("Preflight and resume are mutually exclusive")
    result = guarded_train(args.config, args.guard_config, variant=args.variant,
                           resume=args.resume, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
