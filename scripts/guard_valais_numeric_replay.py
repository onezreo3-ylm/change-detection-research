"""PID-bound temperature guard for a separate engineering replay process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.temperature_guard import (
    WindowsProcess,
    load_settings,
    monitor,
    read_temperature,
    utc_now,
    write_json,
)


def pause_engineering(process, settings: dict, reason: str, temperature: int | None) -> dict:
    process.stop()
    payload = {
        "status": reason,
        "kind": "engineering_numeric_replay_only",
        "paused_utc": utc_now(),
        "temperature_c": temperature,
        "original_research_run_modified": False,
        "reason": "Bound replay process stopped; never auto-restart.",
    }
    write_json(Path(settings["run_dir"]) / "guard_pause.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    process = WindowsProcess(settings)
    try:
        first_temperature = read_temperature(settings)
        if first_temperature >= settings["immediate_pause_c"]:
            pause_engineering(process, settings, "paused_temperature_guard", first_temperature)
            raise RuntimeError("Replay GPU already exceeds immediate pause threshold")
        write_json(Path(settings["run_dir"]) / "guard_attached.marker.json", {
            "status": "bound_guard_attached",
            "process_id": settings["process_id"],
            "creation_time_ticks": settings["creation_time_ticks"],
            "guard_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "first_temperature_c": first_temperature,
            "created_utc": utc_now(),
        })
        result = monitor(settings, process, read=read_temperature, pause=pause_engineering)
        print(json.dumps({"status": result["status"], "kind": "engineering_numeric_replay_only"}))
    finally:
        process.close()


if __name__ == "__main__":
    main()
