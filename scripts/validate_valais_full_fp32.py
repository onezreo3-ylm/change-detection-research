"""One-shot PID-guarded independent clean-val replay for full-FP32 Valais."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from pcdc.config import load_config
from pcdc.data import file_sha256
from pcdc.runtime import write_json
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.temperature_guard import load_settings, read_temperature


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "valais_clean_full_fp32_seed42_20261002"
AUDIT = ROOT / "runs" / "valais_full_fp32_completion_audit_20261002" / "certificate.json"
CONFIG = ROOT / "configs" / "valais_clean_full_fp32_seed42_20261002.yaml"
OUTPUT = ROOT / "runs" / "valais_full_fp32_independent_val_20261002"
PREFLIGHT = ROOT / "runs" / "valais_full_fp32_independent_val_preflight_20261002"
UUID = "GPU-8ada1895-7628-bb88-1dda-c57679117fa2"


def validate_certificate(certificate: dict) -> None:
    if (certificate.get("status") != "audit_recovered_guarded_completion_only"
            or certificate.get("trainer_completed_100_epochs") is not True
            or certificate.get("original_launcher_completed_normally") is not False
            or certificate.get("original_run_guard_completion_exists") is not False
            or certificate.get("final_guard_status") != "training_process_exited"
            or certificate.get("test_accessed") is not False
            or certificate.get("model_inference_performed") is not False
            or not isinstance(certificate.get("source_and_input_bindings_sha256"), dict)):
        raise RuntimeError("Independent validation requires the exact audit-recovered certificate")


def preflight(*, require_new: bool = True):
    if require_new and (OUTPUT.exists() or (RUN / "val_independent.json").exists()):
        raise FileExistsError("Independent validation already started")
    if (RUN / "guard_completion.json").exists():
        raise RuntimeError("Original launcher marker unexpectedly appeared")
    certificate = json.loads(AUDIT.read_text(encoding="utf-8"))
    validate_certificate(certificate)
    for name, expected in certificate["source_and_input_bindings_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise RuntimeError(f"Completion-certified input changed: {name}")
    if str((RUN / "best.pt").resolve()) not in certificate["source_and_input_bindings_sha256"]:
        raise RuntimeError("Certificate does not bind selected checkpoint")
    config = load_config(CONFIG)
    if config.run_test_after_training or config.values["train"]["amp"] is not False:
        raise RuntimeError("Independent validation is FP32 and val-only")
    from pcdc.engine import check_checkpoint, read_checkpoint
    checkpoint = read_checkpoint(RUN / "best.pt")
    check_checkpoint(config, checkpoint)
    if (checkpoint.get("epoch") != certificate["best_epoch"]
            or checkpoint.get("best_val_f1") != certificate["best_clean_val_f1"]
            or checkpoint.get("kind") != "research"):
        raise RuntimeError("Selected checkpoint differs from completion certificate")
    if require_new:
        tests = (PREFLIGHT / "unittest.log").read_text(encoding="utf-8-sig")
        smoke = json.loads((PREFLIGHT / "cuda_smoke.json").read_text(encoding="utf-8"))
        if ("Ran " not in tests or not tests.rstrip().endswith("OK")
                or smoke.get("status") != "engineering_smoke_only"
                or smoke.get("test_accessed") is not False
                or smoke.get("training_amp") is not False
                or smoke.get("validation_amp") is not False):
            raise RuntimeError("Independent validation unittest/CUDA smoke gate failed")
    return config, checkpoint, certificate


def wait_for_guard(path: Path, fingerprint: str) -> None:
    deadline = time.monotonic() + 90
    while not path.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("Independent validation guard did not attach")
        time.sleep(1)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != {"status": "guard_attached", "certificate_sha256": fingerprint}:
        raise RuntimeError("Independent validation guard handshake differs")


def worker(handshake: Path, seal_path: Path) -> dict:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("status") != "full_fp32_independent_val_sealed"
            or seal.get("certificate_sha256") != file_sha256(AUDIT)
            or seal.get("test_accessed") is not False):
        raise RuntimeError("Independent validation seal differs")
    wait_for_guard(handshake, seal["certificate_sha256"])
    # Import the evaluator after the PID guard handshake.
    from pcdc.engine import evaluate
    config, checkpoint, certificate = preflight(require_new=False)
    result = evaluate(config, RUN / "best.pt", "val", OUTPUT / "val_independent.json")
    if result["metrics"]["f1"] != certificate["best_clean_val_f1"]:
        raise RuntimeError("Independent clean-val F1 did not reproduce selected checkpoint")
    return result


def _observed_command(pid: int, powershell: Path) -> str:
    script = f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; $p.CommandLine | ConvertTo-Json -Compress"
    response = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
                              capture_output=True, text=True, check=True, timeout=20,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    value = json.loads(response.stdout)
    if not isinstance(value, str) or "-m scripts.validate_valais_full_fp32" not in value:
        raise RuntimeError("Unexpected independent validation process command")
    return value


def launch():
    if os.name != "nt":
        raise RuntimeError("Windows guarded validation only")
    config, checkpoint, certificate = preflight()
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia = Path("C:/Windows/System32/nvidia-smi.exe")
    uuid = subprocess.run([str(nvidia), "-i", "0", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True, timeout=10,
                          creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    if uuid != UUID or read_temperature({"nvidia_smi": str(nvidia), "gpu_index": 0}) >= 85:
        raise RuntimeError("Validation GPU UUID/temperature differs")
    bound = [AUDIT, CONFIG, RUN / "best.pt", PREFLIGHT / "unittest.log",
             PREFLIGHT / "cuda_smoke.json", ROOT / "scripts" / "validate_valais_full_fp32.py",
             ROOT / "scripts" / "guard_valais_independent_val.py",
             ROOT / "docs" / "VALAIS_FULL_FP32_INDEPENDENT_VAL_2026-10-02.md"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    seal_path = OUTPUT / "preflight_seal.json"
    seal = {"status": "full_fp32_independent_val_sealed",
            "certificate_sha256": file_sha256(AUDIT), "test_accessed": False,
            "gpu_uuid": uuid,
            "bound_sha256": {str(path.resolve()): file_sha256(path) for path in bound}}
    write_json(seal_path, seal)
    handshake = OUTPUT / "guard_attached.json"
    with (OUTPUT / "evaluation.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, "-m", "scripts.validate_valais_full_fp32",
                                  "--worker", "--guard-handshake", str(handshake),
                                  "--seal", str(seal_path)], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    guard = None
    try:
        settings = {"run_dir": str(OUTPUT), "process_id": child.pid,
                    "creation_time_ticks": creation_time_ticks(child.pid),
                    "executable": str(Path(sys.executable).resolve()),
                    "command_line": _observed_command(child.pid, powershell),
                    "powershell": str(powershell), "nvidia_smi": str(nvidia),
                    "gpu_index": 0, "poll_seconds": 10, "confirmation_seconds": 5,
                    "confirmed_pause_c": 85, "immediate_pause_c": 86,
                    "confirmation_samples": 3, "max_sensor_failures": 3}
        guard_config = OUTPUT / "temperature_guard.yaml"
        with guard_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
        load_settings(guard_config)
        with (OUTPUT / "guard.log").open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen([sys.executable, "-m", "scripts.guard_valais_independent_val",
                                      "--config", str(guard_config)], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        marker_path = OUTPUT / "guard_attached.marker.json"
        state_path = OUTPUT / "temperature_guard_state.json"
        deadline = time.monotonic() + 60
        while not (marker_path.is_file() and state_path.is_file()
                   and state_path.stat().st_mtime_ns > marker_path.stat().st_mtime_ns):
            if child.poll() is not None or guard.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Validation/guard exited before first persisted temperature")
            time.sleep(1)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("status") != "bound_guard_attached"
                or marker.get("process_id") != child.pid
                or marker.get("creation_time_ticks") != settings["creation_time_ticks"]
                or marker.get("guard_config_sha256") != file_sha256(guard_config)):
            raise RuntimeError("Validation guard marker differs")
        write_json(handshake, {"status": "guard_attached",
                               "certificate_sha256": seal["certificate_sha256"]})
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Validation guard exited while evaluator active")
            time.sleep(2)
        if child.returncode or guard.wait(timeout=45):
            raise RuntimeError(f"Independent validation/guard failed: {child.returncode}")
        report_path = OUTPUT / "val_independent.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (report.get("split") != "val"
                or report.get("metrics", {}).get("f1") != certificate["best_clean_val_f1"]
                or state.get("status") != "training_process_exited"):
            raise RuntimeError("Independent validation result/guard differs")
        for name, expected in seal["bound_sha256"].items():
            if file_sha256(Path(name)) != expected:
                raise RuntimeError(f"Validation-bound file changed: {name}")
        result = {"status": "guarded_independent_clean_val_reproduced",
                  "test_accessed": False, "split": "val",
                  "selected_epoch": certificate["best_epoch"],
                  "selected_f1": certificate["best_clean_val_f1"],
                  "certificate_sha256": file_sha256(AUDIT),
                  "best_checkpoint_sha256": file_sha256(RUN / "best.pt"),
                  "seal_sha256": file_sha256(seal_path),
                  "report_sha256": file_sha256(report_path),
                  "guard_state_sha256": file_sha256(state_path)}
        write_json(OUTPUT / "completion.json", result)
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--guard-handshake", type=Path)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.guard_handshake is None or args.seal is None:
            raise ValueError("Worker requires handshake and seal")
        result = worker(args.guard_handshake.resolve(), args.seal.resolve())
        print(json.dumps({"status": "independent_val_worker_completed",
                          "f1": result["metrics"]["f1"], "test_accessed": False}))
    elif args.preflight_only:
        config, checkpoint, certificate = preflight()
        print(json.dumps({"status": "preflight_passed", "selected_epoch": checkpoint["epoch"],
                          "test_accessed": False}))
    else:
        print(json.dumps(launch()))


if __name__ == "__main__":
    main()
