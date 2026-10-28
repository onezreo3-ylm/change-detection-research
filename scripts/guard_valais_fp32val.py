"""PID-bound thermal guard for the independent Valais FP32-validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.temperature_guard import (
    WindowsProcess,
    load_settings,
    monitor,
    read_temperature,
    utc_now,
    write_json,
)


def _research_output(guard_dir: Path) -> Path:
    if guard_dir.parent.name == "temperature_guard_sessions":
        return guard_dir.parent.parent
    return guard_dir


def pause_research(process, settings: dict, reason: str, temperature: int | None) -> dict:
    process.stop()
    guard_dir = Path(settings["run_dir"])
    output = _research_output(guard_dir)
    payload = {
        "status": reason,
        "kind": "research",
        "paused_utc": utc_now(),
        "temperature_c": temperature,
        "manual_resume_required": True,
        "reason": "Bound Valais FP32-validation training stopped; incomplete epoch discarded.",
    }
    try:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (checkpoint.get("format_version") != 2
                or checkpoint.get("validation_amp") is not False
                or checkpoint.get("test_evaluated") is not False):
            raise ValueError("Unexpected numeric-protocol checkpoint")
        payload["epoch"] = checkpoint["epoch"]
        payload["best_epoch"] = checkpoint["best_epoch"]
        payload["best_val_f1"] = checkpoint["best_val_f1"]
    except Exception as error:
        payload["checkpoint_verification_error"] = f"{type(error).__name__}: {error}"
    write_json(output / "status.json", payload)
    write_json(guard_dir / "guard_pause.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    process = WindowsProcess(settings)
    try:
        first_temperature = read_temperature(settings)
        if first_temperature >= settings["confirmed_pause_c"]:
            pause_research(process, settings, "paused_temperature_guard", first_temperature)
            raise RuntimeError("GPU is already at the confirmed pause threshold")
        guard_dir = Path(settings["run_dir"])
        write_json(guard_dir / "guard_attached.marker.json", {
            "status": "bound_guard_attached",
            "process_id": settings["process_id"],
            "creation_time_ticks": settings["creation_time_ticks"],
            "guard_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "first_temperature_c": first_temperature,
            "created_utc": utc_now(),
        })
        result = monitor(settings, process, read=read_temperature, pause=pause_research)
        print(json.dumps({"status": result["status"], "kind": "research"}))
    finally:
        process.close()


if __name__ == "__main__":
    main()
