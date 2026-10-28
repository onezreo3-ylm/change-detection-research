"""PID-bound GPU temperature guard for completed-run independent Valais val."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.temperature_guard import (
    WindowsProcess, load_settings, monitor, read_temperature, utc_now, write_json,
)


def pause_validation(process, settings: dict, reason: str, temperature: int | None) -> dict:
    process.stop()
    record = {
        "status": reason,
        "kind": "independent_validation_only",
        "paused_utc": utc_now(),
        "temperature_c": temperature,
        "test_accessed": False,
        "research_training_status_modified": False,
        "automatic_retry": False,
    }
    write_json(Path(settings["run_dir"]) / "guard_pause.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    process = WindowsProcess(settings)
    try:
        temperature = read_temperature(settings)
        if temperature >= settings["confirmed_pause_c"]:
            pause_validation(process, settings, "paused_temperature_guard", temperature)
            raise RuntimeError("GPU is too warm to start independent validation")
        write_json(Path(settings["run_dir"]) / "guard_attached.marker.json", {
            "status": "bound_guard_attached",
            "process_id": settings["process_id"],
            "creation_time_ticks": settings["creation_time_ticks"],
            "guard_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "first_temperature_c": temperature,
            "created_utc": utc_now(),
        })
        result = monitor(settings, process, read=read_temperature, pause=pause_validation)
        print(json.dumps({"status": result["status"], "kind": "independent_validation_only"}))
    finally:
        process.close()


if __name__ == "__main__":
    main()
