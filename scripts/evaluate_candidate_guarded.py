"""Run frozen candidate validation with the in-process GPU temperature guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from pcdc.data import file_sha256
from pcdc.evaluate_candidate import (
    _load_metadata,
    evaluate,
    load_candidate_evaluation_config,
    preflight,
)
from scripts.evaluate_guarded import (
    EvaluationTemperatureGuard,
    GuardError,
    GuardIdentityError,
    _atomic_json,
    _invalidate,
    _sha,
    _utc_now,
    load_guard_config,
)


def guarded_candidate_evaluate(
        config_path: str | Path, guard_path: str | Path, *,
        preflight_only: bool = False,
        read: Callable[[Mapping[str, Any]], int] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None) -> dict[str, Any]:
    if type(preflight_only) is not bool:
        raise ValueError("preflight_only must be boolean")
    job = load_candidate_evaluation_config(config_path)
    # Validate every fixed training/evaluation artifact before a sensor query,
    # model load, dataset read, or output write.
    metadata = _load_metadata(job, require_review=True)
    guard_config = load_guard_config(guard_path)
    device = metadata.bundle.protocol.base.values["runtime"]["device"]
    if device not in ("cuda", "cuda:0") or guard_config.values["gpu_index"] != 0:
        raise GuardIdentityError(
            "This single-GPU guard requires runtime.device cuda/cuda:0 and gpu_index 0")
    guard_config.verify()
    if preflight_only:
        result = preflight(job)
        return dict(result, status="guarded_candidate_preflight_passed",
                    guard_config_sha256=guard_config.config_sha256,
                    guard_source_sha256=dict(guard_config.source_sha256))
    output = job.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite guarded candidate evaluation: {output}")
    guard = EvaluationTemperatureGuard(
        guard_config, output, read=read, clock=clock, sleep=sleep)
    try:
        result = evaluate(job, guard_callback=guard)
        if not guard.attached or not guard.state["observations"]:
            raise GuardError("Candidate evaluator returned without running its temperature guard")
        result = dict(result, temperature_guard=guard.finish(result),
                      guarded_completion_required=True)
        _atomic_json(output / "summary.json", result)
        _atomic_json(output / "status.json", result)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": _utc_now(),
            "guard_config_sha256": guard_config.config_sha256,
            "guard_source_sha256": dict(guard_config.source_sha256),
            "summary_sha256": _sha((output / "summary.json").read_bytes()),
            "guard_state_sha256": _sha((output / "temperature_guard_state.json").read_bytes()),
            "wrapper_source_sha256": file_sha256(Path(__file__).resolve()),
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = guarded_candidate_evaluate(
        args.config, args.guard_config, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
