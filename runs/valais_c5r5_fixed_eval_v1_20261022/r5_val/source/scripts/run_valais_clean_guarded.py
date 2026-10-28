"""Launch the independent Valais clean run with the existing PID-bound guard."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pcdc.config import load_config
from scripts.temperature_guard import load_settings
import yaml


def creation_time_ticks(pid: int) -> int:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def exact_command_line(pid: int, powershell: Path) -> str:
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "$p.CommandLine | ConvertTo-Json -Compress"
    )
    observed = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    result = json.loads(observed.stdout)
    if not isinstance(result, str) or "-m pcdc train" not in result:
        raise RuntimeError("Unexpected training command line")
    return result


def main():
    if os.name != "nt":
        raise RuntimeError("This launcher is Windows-only")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if config.run_test_after_training or config.values["runtime"]["device"] != "cuda":
        raise RuntimeError("Expected CUDA development config with test disabled")
    run_dir = config.output_dir
    if run_dir.exists():
        raise FileExistsError(run_dir)
    root = Path(__file__).resolve().parents[1]
    log = run_dir.with_suffix(".train.log")
    guard_log = run_dir.with_suffix(".guard.log")
    if log.exists() or guard_log.exists():
        raise FileExistsError("Launch logs already exist")
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    command = [sys.executable, "-m", "pcdc", "train", "--config", str(config.source)]
    with log.open("x", encoding="utf-8") as train_log:
        child = subprocess.Popen(
            command,
            cwd=root,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(f"training_pid={child.pid}", flush=True)
    try:
        deadline = time.monotonic() + 600
        while not run_dir.is_dir():
            if child.poll() is not None:
                raise RuntimeError(f"Training exited before run directory: {child.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("Training setup exceeded 10 minutes")
            time.sleep(2)
        settings = {
            "run_dir": str(run_dir),
            "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": exact_command_line(child.pid, powershell),
            "powershell": str(powershell),
            "nvidia_smi": str(nvidia_smi),
            "gpu_index": 0,
            "poll_seconds": 10,
            "confirmation_seconds": 5,
            "confirmed_pause_c": 85,
            "immediate_pause_c": 86,
            "confirmation_samples": 3,
            "max_sensor_failures": 3,
        }
        guard_config = run_dir / "temperature_guard.yaml"
        with guard_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        load_settings(guard_config)
        with guard_log.open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen(
                [sys.executable, "-m", "scripts.temperature_guard", "--config", str(guard_config)],
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        print(f"guard_pid={guard.pid} run_dir={run_dir}", flush=True)
        # The guard must attach successfully before launch is considered ready.
        deadline = time.monotonic() + 45
        while not (run_dir / "temperature_guard_state.json").is_file():
            if guard.poll() is not None:
                raise RuntimeError(f"Temperature guard exited before attachment: {guard.returncode}")
            if child.poll() is not None:
                raise RuntimeError(f"Training exited before guard attachment: {child.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("Temperature guard did not attach")
            time.sleep(1)
        print("launch_status=guard_attached", flush=True)
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Temperature guard exited while training was still active")
            time.sleep(5)
        child_exit = child.returncode
        guard_exit = guard.wait(timeout=60)
        print(f"training_exit={child_exit} guard_exit={guard_exit}", flush=True)
        if child_exit or guard_exit:
            raise RuntimeError("Training or guard did not complete normally")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise


if __name__ == "__main__":
    main()
