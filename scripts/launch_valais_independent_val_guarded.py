"""One-shot PID-guarded clean-val replay after the 100-epoch study completes."""

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

from pcdc.data import file_sha256
from scripts.evaluate_valais_fp32val import verify_guarded_completion
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.temperature_guard import load_settings, read_temperature
from scripts.train_valais_fp32val import _read_checkpoint, _verify_snapshot, load_study


def _observed_command(pid: int, powershell: Path) -> str:
    command = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "$p.CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, check=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, str) or "-m scripts.valais_independent_val_entry" not in value:
        raise RuntimeError("Unexpected independent-validation command line")
    return value


def preflight(spec_path: Path):
    spec, paths, base, fingerprint = load_study(spec_path)
    run_dir = paths["output_dir"]
    if not run_dir.is_dir():
        raise FileNotFoundError("Completed independent study is missing")
    completion = verify_guarded_completion(run_dir, fingerprint)
    snapshot = json.loads((run_dir / "source_sha256.json").read_text(encoding="utf-8"))
    _verify_snapshot(run_dir, snapshot)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if (status.get("status") != "completed"
            or status.get("epochs") != 100
            or status.get("test_evaluated") is not False
            or status.get("validation_amp") is not False):
        raise RuntimeError("Study is not a complete FP32-validation run")
    best = run_dir / "best.pt"
    checkpoint = _read_checkpoint(best)
    if (checkpoint["study_fingerprint"] != fingerprint
            or checkpoint["best_epoch"] != checkpoint["epoch"]
            or checkpoint["best_val_f1"] != status["best_val_f1"]):
        raise RuntimeError("Selected checkpoint does not match completion")
    validation_dir = run_dir / "validation_guard"
    if validation_dir.exists() or (run_dir / "val_independent.json").exists():
        raise FileExistsError("One-shot independent validation already started")
    return spec, paths, base, fingerprint, completion, snapshot, checkpoint


def launch(spec_path: Path):
    if os.name != "nt":
        raise RuntimeError("Windows guarded validation only")
    spec, paths, base, fingerprint, completion, snapshot, checkpoint = preflight(spec_path)
    run_dir = paths["output_dir"]
    validation_dir = run_dir / "validation_guard"
    workspace = Path(__file__).resolve().parents[1]
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    uuid = subprocess.run(
        [str(nvidia_smi), "-i", "0", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()
    if uuid != spec["guard"]["gpu_uuid"]:
        raise RuntimeError("Validation GPU UUID changed")
    temperature = read_temperature({"nvidia_smi": str(nvidia_smi), "gpu_index": 0})
    if temperature >= spec["guard"]["confirmed_pause_c"]:
        raise RuntimeError("GPU too warm for independent validation")
    bound = [
        spec_path,
        paths["base_config"], paths["source_data_audit"],
        paths["numeric_replay_result"], run_dir / "best.pt",
        run_dir / "guard_completion.json", run_dir / "source_sha256.json",
        workspace / "scripts" / "evaluate_valais_fp32val.py",
        workspace / "scripts" / "valais_independent_val_entry.py",
        workspace / "scripts" / "launch_valais_independent_val_guarded.py",
        workspace / "scripts" / "guard_valais_independent_val.py",
        workspace / "tests" / "test_valais_independent_val_guard.py",
        workspace / "docs" / "VALAIS_INDEPENDENT_VAL_GUARD_PROTOCOL_2026-10-01.md",
    ]
    if "prior_launch_failure_addendum" in paths:
        bound.append(paths["prior_launch_failure_addendum"])
    bound.extend(run_dir / "source" / name for name in snapshot)
    seal = {
        "status": "independent_val_preflight_sealed",
        "study_fingerprint": fingerprint,
        "training_completion_sha256": file_sha256(run_dir / "guard_completion.json"),
        "best_checkpoint_sha256": file_sha256(run_dir / "best.pt"),
        "test_accessed": False,
        "gpu_uuid": uuid,
        "bound_sha256": {str(path.resolve()): file_sha256(path) for path in bound},
    }
    validation_dir.mkdir(parents=True, exist_ok=False)
    seal_path = validation_dir / "preflight_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    handshake = validation_dir / "guard_attached.json"
    output = run_dir / "val_independent.json"
    command = [
        sys.executable, "-m", "scripts.valais_independent_val_entry",
        "--config", str(spec_path), "--checkpoint", str(run_dir / "best.pt"),
        "--output", str(output), "--guard-handshake", str(handshake),
        "--fingerprint", fingerprint,
    ]
    with (validation_dir / "evaluation.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, cwd=workspace, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    guard = None
    try:
        observed = _observed_command(child.pid, powershell)
        guard_settings = {
            "run_dir": str(validation_dir),
            "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": observed,
            "powershell": str(powershell),
            "nvidia_smi": str(nvidia_smi),
            "gpu_index": 0,
            **{key: value for key, value in spec["guard"].items()
               if key not in ("gpu_index", "gpu_uuid")},
        }
        guard_config = validation_dir / "temperature_guard.yaml"
        guard_config.write_text(yaml.safe_dump(guard_settings, sort_keys=False), encoding="utf-8")
        load_settings(guard_config)
        with (validation_dir / "guard.log").open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen(
                [sys.executable, "-m", "scripts.guard_valais_independent_val",
                 "--config", str(guard_config)],
                cwd=workspace, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        marker = validation_dir / "guard_attached.marker.json"
        state_path = validation_dir / "temperature_guard_state.json"
        deadline = time.monotonic() + 60
        while not (marker.is_file() and state_path.is_file()
                   and state_path.stat().st_mtime_ns > marker.stat().st_mtime_ns):
            if child.poll() is not None or guard.poll() is not None:
                raise RuntimeError("Validation/guard exited before first persisted sample")
            if time.monotonic() > deadline:
                raise RuntimeError("Validation guard first sample not persisted")
            time.sleep(1)
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if (marker_data.get("status") != "bound_guard_attached"
                or marker_data.get("process_id") != child.pid
                or marker_data.get("creation_time_ticks") != guard_settings["creation_time_ticks"]
                or marker_data.get("guard_config_sha256") != file_sha256(guard_config)):
            raise RuntimeError("Validation guard attach marker differs")
        handshake.write_text(json.dumps({
            "status": "guard_attached", "study_fingerprint": fingerprint
        }) + "\n", encoding="utf-8")
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Validation guard exited while evaluator active")
            time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Independent validation exited with {child.returncode}")
        if guard.wait(timeout=45):
            raise RuntimeError("Validation temperature guard exited with error")
        report = json.loads(output.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (report.get("status") != "independent_validation_reproduced"
                or report.get("test_accessed") is not False
                or report.get("metrics", {}).get("f1") != checkpoint["best_val_f1"]
                or state.get("status") != "training_process_exited"):
            raise RuntimeError("Independent validation or guard did not complete normally")
        for name, expected in seal["bound_sha256"].items():
            if file_sha256(Path(name)) != expected:
                raise RuntimeError(f"Val-bound source/input changed: {name}")
        result = {
            "status": "guarded_independent_val_completed",
            "test_accessed": False,
            "study_fingerprint": fingerprint,
            "preflight_seal_sha256": file_sha256(seal_path),
            "training_completion_sha256": file_sha256(run_dir / "guard_completion.json"),
            "best_checkpoint_sha256": file_sha256(run_dir / "best.pt"),
            "evaluation_report_sha256": file_sha256(output),
            "guard_state_sha256": file_sha256(state_path),
        }
        (validation_dir / "completion.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        if guard is not None and guard.poll() is None:
            guard.wait(timeout=45)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        spec, paths, base, fingerprint, completion, snapshot, checkpoint = preflight(
            args.config.resolve()
        )
        print(json.dumps({"status": "preflight_passed", "best_epoch": checkpoint["epoch"],
                          "test_accessed": False}))
    else:
        print(json.dumps(launch(args.config.resolve())))


if __name__ == "__main__":
    main()
