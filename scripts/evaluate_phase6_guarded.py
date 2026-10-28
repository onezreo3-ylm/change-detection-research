"""Run the approved Phase 6 train-gate diagnostic under the GPU guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.phase5_protocol import load_phase56_config
from pcdc.phase6_diagnostic import evaluate, preflight
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


def _output(config) -> Path:
    value = config.values["diagnostic"]["output_dir"]
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else config.source.parent / candidate).resolve()


def guarded_evaluate(config_path: Path, guard_path: Path, *,
                     preflight_only: bool = False) -> dict:
    config = load_phase56_config(config_path, require_approval=True)
    guard_config = load_guard_config(guard_path)
    if (config.base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard_config.values["gpu_index"] != 0):
        raise GuardIdentityError("Phase 6 is frozen to CUDA physical GPU index 0")
    guard_config.verify()
    if preflight_only:
        result = preflight(config)
        return {**result, "status": "guarded_phase6_preflight_passed",
                "guard_config_sha256": guard_config.config_sha256,
                "sensor_queried": False}
    output = _output(config)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 6 diagnostic: {output}")
    guard = EvaluationTemperatureGuard(guard_config, output)
    try:
        result = evaluate(config, guard_callback=guard)
        if not guard.attached or not guard.state["observations"]:
            raise RuntimeError("Phase 6 returned without an attached temperature guard")
        guarded = dict(result, temperature_guard=guard.finish(result),
                       guarded_completion_required=True)
        _atomic_json(output / "summary.json", guarded)
        _atomic_json(output / "status.json", guarded)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": guard_config.config_sha256,
            "summary_sha256": _sha(output / "summary.json"),
            "guard_state_sha256": _sha(output / "temperature_guard_state.json"),
            "wrapper_source_sha256": _sha(Path(__file__)),
        })
        return guarded
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
            _atomic_json(output / "status.json", {
                "status": "failed_guard", "guarded_result_valid": False,
                "error": f"{type(error).__name__}: {error}",
                "test_evaluated": False, "training_started": False,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = guarded_evaluate(args.config, args.guard_config,
                              preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
