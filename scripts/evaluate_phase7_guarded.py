"""Run the sealed one-shot Phase 7 official-val evaluation under the GPU guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pcdc.phase7_evaluation import evaluate, preflight
from pcdc.phase7_protocol import load_phase7_config
from scripts.evaluate_guarded import (
    EvaluationTemperatureGuard,
    GuardIdentityError,
    _atomic_json,
    load_guard_config,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guarded_evaluate(config_path: Path, guard_path: Path, *,
                     preflight_only: bool = False) -> dict:
    config = load_phase7_config(config_path)
    guard_config = load_guard_config(guard_path)
    if (config.base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard_config.values["gpu_index"] != 0):
        raise GuardIdentityError("Phase 7 evaluation is frozen to CUDA physical GPU index 0")
    guard_config.verify()
    if preflight_only:
        result = preflight(config)
        return {**result, "status": "guarded_phase7_val_preflight_passed",
                "guard_config_sha256": guard_config.config_sha256,
                "sensor_queried": False}
    output = config.paths["validation_output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 official val output: {output}")
    guard = EvaluationTemperatureGuard(guard_config, output)
    try:
        result = evaluate(config, guard_callback=guard)
        if not guard.attached or not guard.state["observations"]:
            raise RuntimeError("Phase 7 evaluation returned without an attached guard")
        guard_record = guard.finish(result)
        guarded = dict(result, temperature_guard=guard_record,
                       guarded_completion_required=True)
        _atomic_json(output / "summary.json", guarded)
        _atomic_json(output / "status.json", guarded)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "guard_config_sha256": guard_config.config_sha256,
            "summary_sha256": _sha(output / "summary.json"),
            "guard_state_sha256": _sha(output / "temperature_guard_state.json"),
            "wrapper_source_sha256": _sha(Path(__file__)),
        })
        return guarded
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
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
