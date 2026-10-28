"""Build the approved train-only Phase 7 descriptor cache under the GPU guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.phase7_cache import build_descriptor_cache, cache_preflight
from pcdc.phase7_protocol import load_phase7_config
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


def _guard_output(output: Path, resume: bool) -> Path:
    if not resume:
        return output
    if not output.is_dir() or not (output / "status.json").is_file():
        raise FileNotFoundError("Phase 7 cache resume requires an existing status")
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if status.get("status") == "completed":
        raise ValueError("Completed Phase 7 cache must not be resumed")
    sessions = output / "temperature_guard_resumes"
    sessions.mkdir(exist_ok=True)
    index = 1
    while (sessions / f"session_{index:04d}").exists():
        index += 1
    session = sessions / f"session_{index:04d}"
    session.mkdir(exist_ok=False)
    return session


def guarded_cache(config_path: Path, guard_path: Path, *, resume: bool = False,
                  preflight_only: bool = False) -> dict:
    config = load_phase7_config(config_path)
    guard_config = load_guard_config(guard_path)
    if (config.base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard_config.values["gpu_index"] != 0):
        raise GuardIdentityError("Phase 7 cache is frozen to CUDA physical GPU index 0")
    guard_config.verify()
    if preflight_only:
        result = cache_preflight(config)
        return {**result, "status": "guarded_phase7_cache_preflight_passed",
                "guard_config_sha256": guard_config.config_sha256,
                "sensor_queried": False}
    output = config.paths["descriptor_cache_dir"]
    guard_output = _guard_output(output, resume)
    guard = EvaluationTemperatureGuard(guard_config, guard_output)
    try:
        result = build_descriptor_cache(config, resume=resume, guard_callback=guard)
        record = guard.finish(result)
        record["state_path"] = str(
            (guard_output / "temperature_guard_state.json").relative_to(output))
        status = dict(result, temperature_guard=record, guarded_completion_required=True)
        _atomic_json(output / "status.json", status)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": guard_config.config_sha256,
            "cache_status_sha256": _sha(output / "status.json"),
            "guard_state_path": record["state_path"],
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
    result = guarded_cache(args.config, args.guard_config, resume=args.resume,
                           preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
