"""Run one R1E replication validation with the frozen GPU temperature guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from pcdc.data import file_sha256
from scripts.evaluate_guarded import (
    EvaluationTemperatureGuard, GuardError, GuardIdentityError,
    _atomic_json, _invalidate, _sha, _utc_now, load_guard_config,
)
from scripts.r1e_evaluation import evaluate, load_context, preflight


def guarded_r1e_evaluate(
        approval_path: str | Path, key: str, guard_path: str | Path, *,
        preflight_only: bool = False,
        read: Callable[[Mapping[str, Any]], int] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None) -> dict[str, Any]:
    context = load_context(approval_path, key)
    # This loads and validates the seed-specific checkpoint before any dataset
    # read, GPU/sensor operation, or output directory is created.
    verified = preflight(context)
    guard_config = load_guard_config(guard_path)
    device = context.training.base.values["runtime"]["device"]
    if device not in ("cuda", "cuda:0") or guard_config.values["gpu_index"] != 0:
        raise GuardIdentityError("R1E requires the bound real CUDA GPU 0")
    guard_config.verify()
    if preflight_only:
        return dict(verified,
                    guard_config_sha256=guard_config.config_sha256,
                    guard_source_sha256=dict(guard_config.source_sha256))
    output = context.candidate.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite guarded R1E result: {output}")
    guard = EvaluationTemperatureGuard(
        guard_config, output, read=read, clock=clock, sleep=sleep)
    try:
        result = evaluate(context, guard_callback=guard)
        if not guard.attached or not guard.state["observations"]:
            raise GuardError("R1E evaluator returned without a temperature observation")
        result = dict(result, temperature_guard=guard.finish(result),
                      guarded_completion_required=True)
        _atomic_json(output / "summary.json", result)
        _atomic_json(output / "status.json", result)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": guard_config.config_sha256,
            "guard_source_sha256": dict(guard_config.source_sha256),
            "summary_sha256": _sha((output / "summary.json").read_bytes()),
            "guard_state_sha256": _sha(
                (output / "temperature_guard_state.json").read_bytes()),
            "wrapper_source_sha256": file_sha256(Path(__file__).resolve()),
            "r1e_approval_sha256": context.approval_sha256,
        })
        return result
    except BaseException as error:
        try:
            guard.abort(error)
        finally:
            if guard.attached:
                _invalidate(output, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--run", choices=("b1_43", "b2_43", "b1_44", "b2_44"),
                        required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = guarded_r1e_evaluate(
        args.approval, args.run, args.guard_config,
        preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
