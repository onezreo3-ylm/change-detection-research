"""Launch or explicitly resume the independent Valais numeric protocol guarded."""

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
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.temperature_guard import load_settings, read_temperature
from scripts.train_valais_fp32val import load_study


def _observed_command_line(pid: int, powershell: Path) -> str:
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "$p.CommandLine | ConvertTo-Json -Compress"
    )
    response = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, check=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    result = json.loads(response.stdout)
    if not isinstance(result, str) or "-m scripts.train_valais_fp32val" not in result:
        raise RuntimeError("Unexpected training command line")
    return result


def _verify_preflight(root: Path, spec: dict, paths: dict, fingerprint: str) -> list[Path]:
    evidence = root / "runs" / (
        "valais_fp32val_relaunch_preflight_20261001"
        if "prior_launch_failure_addendum" in paths else
        "valais_fp32val_preflight_20261001"
    )
    log = evidence / "unittest.log"
    tests_path = evidence / "unittest_evidence.json"
    smoke_path = evidence / "real_cuda_smoke.json"
    for path in (log, tests_path, smoke_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    required = {
        "scripts/train_valais_fp32val.py",
        "scripts/launch_valais_fp32val.py",
        "scripts/guard_valais_fp32val.py",
        "scripts/smoke_valais_fp32val.py",
        "scripts/evaluate_valais_fp32val.py",
        "tests/test_train_valais_fp32val.py",
        "tests/test_guard_valais_fp32val.py",
        "tests/test_evaluate_valais_fp32val.py",
        "docs/VALAIS_CLEAN_FP32_VALIDATION_PROTOCOL_2026-09-30.md",
        "configs/valais_clean_fp32val_seed42_20260930.yaml",
    }
    if "prior_launch_failure_addendum" in paths:
        required.update({
            "configs/valais_clean_fp32val_seed42_relaunch_20261001.yaml",
            "docs/VALAIS_FP32VAL_RELAUNCH_AFTER_OS_SHUTDOWN_2026-10-01.md",
        })
    if (tests.get("status") != "passed" or tests.get("exit_code") != 0
            or tests.get("test_count", 0) < 230
            or tests.get("log_sha256") != hashlib.sha256(log.read_bytes()).hexdigest()
            or set(tests.get("source_sha256", {})) != required
            or smoke.get("status") != "engineering_smoke_only"
            or smoke.get("test_accessed") is not False
            or smoke.get("training_amp") is not True
            or smoke.get("validation_amp") is not False
            or smoke.get("study_fingerprint") != fingerprint):
        raise RuntimeError("Numeric-protocol test/smoke preflight failed")
    for name, expected in tests["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed after tests: {name}")
    return [log, tests_path, smoke_path]


def _session(output: Path, resume: bool) -> Path:
    if not resume:
        if output.exists():
            raise FileExistsError(output)
        return output
    if not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume checkpoint is missing")
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if status.get("status") not in (
        "paused_temperature_guard", "paused_temperature_unavailable",
        "paused_temperature_guard_error", "paused_user", "interrupted",
    ):
        raise RuntimeError("Only a guarded or explicit interruption may resume")
    sessions = output / "temperature_guard_sessions"
    sessions.mkdir(exist_ok=True)
    index = 1
    while (sessions / f"session_{index:04d}").exists():
        index += 1
    folder = sessions / f"session_{index:04d}"
    folder.mkdir(exist_ok=False)
    return folder


def launch(path: Path, *, resume: bool = False) -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows guarded launch only")
    spec, paths, base, fingerprint = load_study(path)
    root = Path(__file__).resolve().parents[1]
    evidence_paths = _verify_preflight(root, spec, paths, fingerprint)
    output = paths["output_dir"]
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    uuid = subprocess.run(
        [str(nvidia_smi), "-i", "0", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()
    if uuid != spec["guard"]["gpu_uuid"]:
        raise RuntimeError("GPU UUID changed")
    temperature = read_temperature({"nvidia_smi": str(nvidia_smi), "gpu_index": 0})
    if temperature >= spec["guard"]["confirmed_pause_c"]:
        raise RuntimeError("GPU is too warm to launch the independent study")
    guard_dir = _session(output, resume)
    suffix = "" if not resume else "." + guard_dir.name
    stdout_path = output.with_suffix(suffix + ".train.log")
    guard_log = output.with_suffix(suffix + ".guard.log")
    seal_path = output.with_suffix(suffix + ".preflight_seal.json")
    if any(item.exists() for item in (stdout_path, guard_log, seal_path)):
        raise FileExistsError("Launch artifacts already exist")
    bound = [
        path, paths["base_config"], paths["source_data_audit"],
        paths["numeric_replay_result"], *evidence_paths,
        root / "scripts" / "train_valais_fp32val.py",
        root / "scripts" / "launch_valais_fp32val.py",
        root / "scripts" / "guard_valais_fp32val.py",
        root / "scripts" / "smoke_valais_fp32val.py",
        root / "scripts" / "evaluate_valais_fp32val.py",
        root / "docs" / "VALAIS_CLEAN_FP32_VALIDATION_PROTOCOL_2026-09-30.md",
        *source_files().values(),
    ]
    if "prior_launch_failure_addendum" in paths:
        bound.extend([
            paths["prior_launch_failure_addendum"],
            root / "docs" / "VALAIS_FP32VAL_RELAUNCH_AFTER_OS_SHUTDOWN_2026-10-01.md",
        ])
    seal = {
        "status": "independent_numeric_protocol_preflight_sealed",
        "fingerprint": fingerprint,
        "training_amp": True,
        "validation_amp": False,
        "test_accessed": False,
        "gpu_uuid": uuid,
        "bound_sha256": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in bound
        },
    }
    with seal_path.open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")
    handshake = guard_dir / "guard_attached.json"
    command = [
        sys.executable, "-m", "scripts.train_valais_fp32val", "--config", str(path),
        "--guard-handshake", str(handshake),
    ]
    if resume:
        command.append("--resume")
    with stdout_path.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(f"training_pid={child.pid}", flush=True)
    guard = None
    try:
        deadline = time.monotonic() + 600
        while True:
            status_path = output / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    status = {}
                if status.get("status") == "waiting_guard":
                    break
            if child.poll() is not None:
                raise RuntimeError(f"Training exited before guard: {child.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("Training setup exceeded 10 minutes")
            time.sleep(2)
        settings = {
            "run_dir": str(guard_dir),
            "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": _observed_command_line(child.pid, powershell),
            "powershell": str(powershell),
            "nvidia_smi": str(nvidia_smi),
            "gpu_index": 0,
            **{key: value for key, value in spec["guard"].items()
               if key not in ("gpu_index", "gpu_uuid")},
        }
        guard_config = guard_dir / "temperature_guard.yaml"
        with guard_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        load_settings(guard_config)
        with guard_log.open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen(
                [
                    sys.executable, "-m", "scripts.guard_valais_fp32val",
                    "--config", str(guard_config),
                ],
                cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        marker_path = guard_dir / "guard_attached.marker.json"
        state_path = guard_dir / "temperature_guard_state.json"
        deadline = time.monotonic() + 60
        while not (marker_path.is_file() and state_path.is_file()
                   and state_path.stat().st_mtime_ns > marker_path.stat().st_mtime_ns):
            if guard.poll() is not None or child.poll() is not None:
                raise RuntimeError("Guard/training exited before confirmed first sample")
            if time.monotonic() > deadline:
                raise RuntimeError("Temperature guard first sample not persisted")
            time.sleep(1)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("status") != "bound_guard_attached"
                or marker.get("process_id") != child.pid
                or marker.get("creation_time_ticks") != settings["creation_time_ticks"]
                or marker.get("guard_config_sha256") !=
                hashlib.sha256(guard_config.read_bytes()).hexdigest()):
            raise RuntimeError("Guard attach marker does not bind this process")
        handshake.write_text(json.dumps({
            "status": "guard_attached", "study_fingerprint": fingerprint
        }) + "\n", encoding="utf-8")
        print(f"guard_pid={guard.pid} launch_status=guard_attached", flush=True)
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Guard exited while training remained active")
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f"Training process exited with {child.returncode}")
        if guard.wait(timeout=60):
            raise RuntimeError("Temperature guard exited with error")
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed" or state.get("status") != "training_process_exited":
            raise RuntimeError("Training or guard did not complete normally")
        for name, expected in seal["bound_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Bound source/input changed: {name}")
        completion = {
            "status": "guarded_independent_clean_completed",
            "test_accessed": False,
            "study_fingerprint": fingerprint,
            "guard_state_path": str(state_path.resolve()),
            "preflight_seal_path": str(seal_path.resolve()),
            "status_sha256": hashlib.sha256((output / "status.json").read_bytes()).hexdigest(),
            "guard_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "preflight_seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
            "best_checkpoint_sha256": hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
            "last_checkpoint_sha256": hashlib.sha256((output / "last.pt").read_bytes()).hexdigest(),
        }
        with (output / "guard_completion.json").open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, indent=2)
            stream.write("\n")
        return completion
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.preflight_only:
        spec, paths, base, fingerprint = load_study(path)
        print(json.dumps({"status": "preflight_passed", "fingerprint": fingerprint,
                          "test_accessed": False}))
    else:
        result = launch(path, resume=args.resume)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
