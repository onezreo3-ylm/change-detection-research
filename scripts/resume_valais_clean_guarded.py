"""Resume the paused Valais clean run from a complete epoch with a fresh PID guard.

This launcher does not alter the frozen model, training config or checkpoint.
It archives the prior pause/guard records before starting a new guarded session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

from pcdc.config import load_config
from pcdc.engine import check_checkpoint, read_checkpoint, source_hashes, training_signature
from scripts.run_valais_clean_guarded import creation_time_ticks, exact_command_line
from scripts.temperature_guard import load_settings


ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SESSION = re.compile(r"resume_[0-9]{8}_epoch[0-9]+\Z")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paused_boundary(status: dict, checkpoint: dict, *, expected_epoch: int) -> None:
    require(status.get("status") == "paused_user" and
            status.get("manual_resume_required") is True and
            status.get("epoch") == expected_epoch and
            status.get("kind") == "research", "Run is not paused at the expected complete epoch")
    require(checkpoint.get("format_version") == 1 and
            checkpoint.get("kind") == "research" and
            checkpoint.get("epoch") == expected_epoch and
            len(checkpoint.get("history", [])) == expected_epoch,
            "Checkpoint is not at the expected complete epoch")


def active_training_pids(config_path: Path, powershell: Path) -> list[int]:
    command = (
        "$match = Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m pcdc train*' -and "
        f"$_.CommandLine -like '*{config_path.name}*' }}; "
        "@($match | ForEach-Object { $_.ProcessId }) | ConvertTo-Json -Compress"
    )
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive",
                             "-Command", command], capture_output=True, text=True,
                            check=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    value = json.loads(result.stdout) if result.stdout.strip() else None
    return [] if value is None else ([value] if isinstance(value, int) else list(value))


def preflight(config_path: Path, expected_sha: str, expected_epoch: int,
              session: str) -> tuple[object, Path]:
    require(_SHA.fullmatch(expected_sha) is not None, "Expected SHA256 must be lowercase")
    require(_SESSION.fullmatch(session) is not None and
            session.endswith(f"_epoch{expected_epoch}"), "Invalid or mismatched resume session")
    config = load_config(config_path.resolve())
    require(config.values["experiment"]["name"] == "valais_clean_development_seed42" and
            config.values["experiment"]["kind"] == "research" and
            config.values["runtime"]["device"] == "cuda" and
            not config.run_test_after_training, "Only the CUDA, development-only Valais clean run may resume")
    output = config.output_dir
    require(output.is_dir(), "Valais output directory is missing")
    require(not (output / "temperature_guard.lock").exists(), "Existing live guard lock")
    require(not (output / "guard_completion.json").exists(), "Run is already guarded complete")
    checkpoint_path = output / "last.pt"
    require(sha256(checkpoint_path) == expected_sha, "Last checkpoint hash differs")
    checkpoint = read_checkpoint(checkpoint_path)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    paused_boundary(status, checkpoint, expected_epoch=expected_epoch)
    check_checkpoint(config, checkpoint)
    require(checkpoint["training_signature"] == training_signature(config) and
            checkpoint["source_sha256"] == source_hashes(),
            "Frozen training signature or source differs")
    require((output / "data_audit.json").is_file() and
            (output / "history.json").is_file() and
            (output / "temperature_guard_state.json").is_file(),
            "Prior audit/history/guard record missing")
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    require(not active_training_pids(config.source, powershell), "Training process is already active")
    archive = output / "temperature_guard_sessions" / session
    require(not archive.exists() and
            not (output.parent / f"{output.name}.{session}.train.log").exists() and
            not (output.parent / f"{output.name}.{session}.guard.log").exists(),
            "Resume session already exists")
    return config, archive


def run(config_path: Path, expected_sha: str, expected_epoch: int, session: str) -> None:
    if os.name != "nt":
        raise RuntimeError("This guarded resume launcher is Windows-only")
    config, archive = preflight(config_path, expected_sha, expected_epoch, session)
    output = config.output_dir
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("status.json", "temperature_guard_state.json"):
        shutil.copyfile(output / name, archive / name)
    shutil.copyfile(output / "temperature_guard.yaml", archive / "prior_temperature_guard.yaml")
    train_log = output.parent / f"{output.name}.{session}.train.log"
    guard_log = output.parent / f"{output.name}.{session}.guard.log"
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    command = [sys.executable, "-m", "pcdc", "train", "--config", str(config.source), "--resume"]
    with train_log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    print(f"training_pid={child.pid} resumed_from_epoch={expected_epoch}", flush=True)
    guard = None
    try:
        settings = {
            "run_dir": str(output), "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": exact_command_line(child.pid, powershell),
            "powershell": str(powershell), "nvidia_smi": str(nvidia_smi),
            "gpu_index": 0, "poll_seconds": 10, "confirmation_seconds": 5,
            "confirmed_pause_c": 85, "immediate_pause_c": 86,
            "confirmation_samples": 3, "max_sensor_failures": 3,
        }
        guard_config = archive / "temperature_guard.yaml"
        with guard_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        load_settings(guard_config)
        prior_guard_mtime = (output / "temperature_guard_state.json").stat().st_mtime_ns
        with guard_log.open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen(
                [sys.executable, "-m", "scripts.temperature_guard", "--config", str(guard_config)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 60
        while True:
            if guard.poll() is not None:
                raise RuntimeError(f"Temperature guard exited before attachment: {guard.returncode}")
            if child.poll() is not None:
                raise RuntimeError(f"Training exited before guard attachment: {child.returncode}")
            state = output / "temperature_guard_state.json"
            if state.stat().st_mtime_ns > prior_guard_mtime:
                try:
                    attached = json.loads(state.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    attached = {}
                if attached.get("training_pid") == child.pid and attached.get("guard_pid") == guard.pid:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("Temperature guard did not attach")
            time.sleep(1)
        print(f"guard_pid={guard.pid} launch_status=guard_attached run_dir={output}", flush=True)
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Temperature guard exited while training was active")
            time.sleep(5)
        guard_exit = guard.wait(timeout=60)
        print(f"training_exit={child.returncode} guard_exit={guard_exit}", flush=True)
        if child.returncode or guard_exit:
            raise RuntimeError("Training or guard did not complete normally")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        config, archive = preflight(args.config, args.checkpoint_sha256, args.epoch, args.session)
        print(json.dumps({"status": "preflight_passed", "run": str(config.output_dir),
                          "archive": str(archive), "epoch": args.epoch}), flush=True)
    else:
        run(args.config, args.checkpoint_sha256, args.epoch, args.session)


if __name__ == "__main__":
    main()
