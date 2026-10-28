"""Run one isolated epoch-34 replay with the PID-bound temperature guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from pcdc.engine import source_files
from scripts.diagnose_valais_epoch34 import preflight, verify_original_failure_artifacts
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.temperature_guard import load_settings, read_temperature


def observed_command_line(pid: int, powershell: Path) -> str:
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "$p.CommandLine | ConvertTo-Json -Compress"
    )
    response = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    result = json.loads(response.stdout)
    if not isinstance(result, str) or "-m scripts.diagnose_valais_epoch34" not in result:
        raise RuntimeError("Unexpected replay command line")
    return result


def launch(spec_path: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("This launcher is Windows-only")
    spec, paths, config, checkpoint = preflight(spec_path)
    output = paths["output_dir"]
    root = Path(__file__).resolve().parents[1]
    evidence_root = root / "runs" / "valais_numeric_replay_preflight_20260930"
    unittest_evidence = evidence_root / "unittest_evidence.json"
    unittest_log = evidence_root / "unittest_final.log"
    smoke_evidence = evidence_root / "real_cuda_smoke.json"
    for item in (unittest_evidence, unittest_log, smoke_evidence):
        if not item.is_file():
            raise FileNotFoundError(item)
    tests = json.loads(unittest_evidence.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_evidence.read_text(encoding="utf-8"))
    if (tests.get("status") != "passed" or tests.get("exit_code") != 0
            or tests.get("test_count", 0) < 225
            or tests.get("log_sha256") != hashlib.sha256(unittest_log.read_bytes()).hexdigest()
            or smoke.get("status") != "engineering_smoke_only"
            or smoke.get("test_accessed") is not False
            or smoke.get("config_sha256") != spec["source_config_sha256"]):
        raise RuntimeError("Replay test/smoke evidence is missing or invalid")
    required_sources = {
        "scripts/diagnose_valais_epoch34.py",
        "scripts/launch_valais_numeric_replay.py",
        "scripts/guard_valais_numeric_replay.py",
        "tests/test_valais_numeric_replay.py",
        "tests/test_guard_valais_numeric_replay.py",
        "docs/VALAIS_NUMERIC_REPLAY_PROTOCOL_2026-09-30.md",
    }
    if set(tests.get("source_sha256", {})) != required_sources:
        raise RuntimeError("Unittest evidence does not bind every replay source")
    for relative, expected in tests.get("source_sha256", {}).items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed after replay unittest: {relative}")
    train_log = output.with_suffix(".stdout.log")
    guard_log = output.with_suffix(".guard.log")
    seal_path = output.with_suffix(".preflight_seal.json")
    if train_log.exists() or guard_log.exists() or seal_path.exists():
        raise FileExistsError("Replay launch logs already exist")
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    uuid_response = subprocess.run(
        [str(nvidia_smi), "-i", "0", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if uuid_response.stdout.strip() != spec["guard"]["gpu_uuid"]:
        raise RuntimeError("Bound GPU UUID changed")
    sensor_settings = {"nvidia_smi": str(nvidia_smi), "gpu_index": 0}
    if read_temperature(sensor_settings) >= spec["guard"]["confirmed_pause_c"]:
        raise RuntimeError("GPU is too warm to start the replay")
    bound = [
        spec_path,
        paths["source_config"], paths["checkpoint"], paths["failure_audit"],
        paths["source_data_audit"],
        root / "scripts" / "diagnose_valais_epoch34.py",
        root / "scripts" / "launch_valais_numeric_replay.py",
        root / "scripts" / "guard_valais_numeric_replay.py",
        root / "tests" / "test_valais_numeric_replay.py",
        root / "tests" / "test_guard_valais_numeric_replay.py",
        root / "docs" / "VALAIS_NUMERIC_REPLAY_PROTOCOL_2026-09-30.md",
        unittest_evidence, unittest_log, smoke_evidence,
        *source_files().values(),
    ]
    seal = {
        "status": "engineering_replay_preflight_sealed",
        "test_accessed": False,
        "original_run_modified": False,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "bound_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in bound
        },
        "gpu_uuid": spec["guard"]["gpu_uuid"],
    }
    with seal_path.open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")
    command = [
        sys.executable, "-m", "scripts.diagnose_valais_epoch34",
        "--config", str(spec_path), "--seal", str(seal_path),
    ]
    with train_log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(f"replay_pid={child.pid}", flush=True)
    guard = None
    try:
        deadline = time.monotonic() + 600
        while not (output / "replay.json").is_file():
            if child.poll() is not None:
                raise RuntimeError(f"Replay exited before guard setup: {child.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("Replay preflight exceeded 10 minutes")
            time.sleep(2)
        guard_settings = {
            "run_dir": str(output),
            "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": observed_command_line(child.pid, powershell),
            "powershell": str(powershell),
            "nvidia_smi": str(nvidia_smi),
            "gpu_index": spec["guard"]["gpu_index"],
            **{key: value for key, value in spec["guard"].items()
               if key not in ("gpu_index", "gpu_uuid")},
        }
        config_path = output / "temperature_guard.yaml"
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(guard_settings, stream, sort_keys=False)
        load_settings(config_path)
        with guard_log.open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen(
                [
                    sys.executable, "-m", "scripts.guard_valais_numeric_replay",
                    "--config", str(config_path),
                ],
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        deadline = time.monotonic() + 45
        marker_path = output / "guard_attached.marker.json"
        while not marker_path.is_file():
            if guard.poll() is not None:
                raise RuntimeError("Replay guard exited before attachment")
            if child.poll() is not None:
                raise RuntimeError("Replay exited before guard attachment")
            if time.monotonic() > deadline:
                raise RuntimeError("Replay guard did not attach")
            time.sleep(1)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("status") != "bound_guard_attached"
                or marker.get("process_id") != child.pid
                or marker.get("creation_time_ticks") != guard_settings["creation_time_ticks"]
                or marker.get("guard_config_sha256") !=
                hashlib.sha256(config_path.read_bytes()).hexdigest()):
            raise RuntimeError("Replay guard identity/state mismatch")
        handshake = {
            "status": "guard_attached",
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        }
        (output / "guard_attached.json").write_text(
            json.dumps(handshake, indent=2) + "\n", encoding="utf-8"
        )
        print(f"guard_pid={guard.pid} launch_status=guard_attached", flush=True)
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Replay guard exited while replay was active")
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f"Replay process exited with {child.returncode}")
        if guard.wait(timeout=45):
            raise RuntimeError("Replay guard exited with error")
        report = json.loads((output / "replay.json").read_text(encoding="utf-8"))
        state = json.loads((output / "temperature_guard_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "training_process_exited":
            raise RuntimeError("Replay guard did not complete normally")
        for name, expected in seal["bound_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Bound replay input/source changed: {name}")
        failure = json.loads(paths["failure_audit"].read_text(encoding="utf-8"))
        verify_original_failure_artifacts(failure, root)
        completion = {
            "status": "engineering_replay_attempt_complete",
            "replay_status": report["status"],
            "reproduced_within_bound_window": report["status"] == "reproduced_within_bound_window",
            "original_run_modified": False,
            "test_accessed": False,
            "preflight_seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
            "replay_sha256": hashlib.sha256((output / "replay.json").read_bytes()).hexdigest(),
            "guard_state_sha256": hashlib.sha256(
                (output / "temperature_guard_state.json").read_bytes()
            ).hexdigest(),
            "train_trace_sha256": hashlib.sha256(
                (output / "train_trace.log").read_bytes()
            ).hexdigest(),
        }
        with (output / "completion.json").open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"replay_status": report["status"], "original_run_modified": False}))
        return report
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        if guard is not None and guard.poll() is None:
            guard.wait(timeout=45)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    spec_path = args.config.resolve()
    if args.preflight_only:
        spec, paths, config, checkpoint = preflight(spec_path)
        print(json.dumps({"status": "preflight_passed", "epoch": checkpoint["epoch"],
                          "test_accessed": False}))
    else:
        launch(spec_path)


if __name__ == "__main__":
    main()
