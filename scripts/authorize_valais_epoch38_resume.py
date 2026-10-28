"""Hash-bound one-time authorization to resume after a verified supervisor loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import audit_data, file_sha256
from pcdc.engine import _digest
from scripts.train_valais_fp32val import _read_checkpoint, _verify_snapshot, load_study


KEYS = {
    "schema_version", "kind", "study_config", "study_config_sha256",
    "supervisor_loss_audit", "supervisor_loss_audit_sha256",
    "checkpoint_sha256", "interrupted_status_sha256", "liveness_probe",
    "minimum_probe_seconds", "authorization_output", "task_name", "test_access",
}
TASK_NAME = "Codex-Valais-FP32Val-Resume-Epoch38-20261001"


def validate_interrupted_status(status: dict) -> None:
    # The PID-bound pause helper records test access in the checkpoint, while its
    # administrative status omits test_evaluated. A true value always fails.
    if (status.get("status") != "interrupted" or status.get("epoch") != 38
            or status.get("manual_resume_required") is not True
            or status.get("test_evaluated") is True):
        raise RuntimeError("Run is not the audited interrupted epoch38 state")


def validate_probe(path: Path, minimum_seconds: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    probe = json.loads(path.read_text(encoding="utf-8-sig"))
    if (probe.get("status") != "completed"
            or probe.get("kind") != "scheduler_liveness_no_gpu_no_data"
            or probe.get("parent_name", "").lower() != "svchost.exe"
            or probe.get("data_accessed") is not False
            or probe.get("gpu_accessed") is not False
            or type(probe.get("duration_seconds")) not in (int, float)
            or probe["duration_seconds"] < minimum_seconds):
        raise RuntimeError("Long-duration no-trigger Scheduler probe did not pass")
    return probe


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def _matching_processes(run_name: str) -> list[int]:
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    query = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
        f"-and $_.CommandLine -match '{run_name}' "
        "-and $_.CommandLine -match 'train_valais_fp32val|guard_valais_fp32val|launch_valais_fp32val' } "
        "| Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
        capture_output=True, text=True, check=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    value = result.stdout.strip()
    if not value or value == "null":
        return []
    decoded = json.loads(value)
    return [decoded] if isinstance(decoded, int) else list(decoded)


def inspect(config_path: Path):
    cfg = yaml.load(config_path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(cfg, dict) or set(cfg) != KEYS:
        raise ValueError("Unexpected epoch38 recovery keys")
    if (cfg["schema_version"] != 1 or cfg["kind"] != "supervisor_loss_manual_resume"
            or cfg["minimum_probe_seconds"] != 3600
            or cfg["task_name"] != TASK_NAME
            or cfg["test_access"] != "forbidden"):
        raise ValueError("Wrong epoch38 recovery protocol")
    paths = {
        key: _resolve(config_path, cfg[key])
        for key in ("study_config", "supervisor_loss_audit", "liveness_probe",
                    "authorization_output")
    }
    workspace = Path(__file__).resolve().parents[1]
    if paths["authorization_output"] != (
        workspace / "runs" / "valais_fp32val_supervisor_loss_20261001"
        / "epoch38_resume_authorization.json"
    ) or paths["authorization_output"].exists():
        raise RuntimeError("Authorization output must be new and independently named")
    for key in ("study_config", "supervisor_loss_audit"):
        if not paths[key].is_file() or file_sha256(paths[key]) != cfg[key + "_sha256"]:
            raise RuntimeError(f"Bound recovery source changed: {key}")
    audit = json.loads(paths["supervisor_loss_audit"].read_text(encoding="utf-8"))
    if (audit.get("status") != "safely_interrupted_after_supervisor_loss"
            or audit.get("last_complete_epoch") != 38
            or audit.get("test_accessed") is not False
            or audit.get("artifacts_sha256", {}).get("last_checkpoint") != cfg["checkpoint_sha256"]
            or audit.get("artifacts_sha256", {}).get("status_after_stop") !=
            cfg["interrupted_status_sha256"]):
        raise RuntimeError("Supervisor-loss audit does not authorize epoch38")
    validate_probe(paths["liveness_probe"], cfg["minimum_probe_seconds"])
    spec, study_paths, base, fingerprint = load_study(paths["study_config"])
    run = study_paths["output_dir"]
    if (not run.is_dir() or (run / "guard_completion.json").exists()
            or _matching_processes(run.name)):
        raise RuntimeError("Training is active or already completed")
    status_path = run / "status.json"
    checkpoint_path = run / "last.pt"
    if (file_sha256(status_path) != cfg["interrupted_status_sha256"]
            or file_sha256(checkpoint_path) != cfg["checkpoint_sha256"]):
        raise RuntimeError("Interrupted status/checkpoint bytes changed")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    checkpoint = _read_checkpoint(checkpoint_path)
    validate_interrupted_status(status)
    if (checkpoint.get("epoch") != 38
            or checkpoint.get("study_fingerprint") != fingerprint
            or checkpoint.get("test_evaluated") is not False
            or len(checkpoint.get("history", [])) != 38
            or not all(torch.isfinite(value).all().item()
                       for value in checkpoint["model"].values()
                       if torch.is_tensor(value))):
        raise RuntimeError("Epoch38 recovery state is not intact")
    snapshot = json.loads((run / "source_sha256.json").read_text(encoding="utf-8"))
    _verify_snapshot(run, snapshot)
    live_audit = audit_data(base, full=True)
    archived_audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    if live_audit != archived_audit or _digest(live_audit) != checkpoint["data_audit_digest"]:
        raise RuntimeError("Current train/val data differs from epoch38 checkpoint")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
    result = subprocess.run(
        [str(nvidia_smi), "-i", "0", "--query-gpu=uuid,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    gpu_uuid, temperature_text = [piece.strip() for piece in result.stdout.strip().split(",")]
    temperature = int(temperature_text)
    if gpu_uuid != spec["guard"]["gpu_uuid"] or temperature >= spec["guard"]["confirmed_pause_c"]:
        raise RuntimeError("GPU UUID or temperature differs from safe recovery policy")
    return cfg, paths, spec, run, checkpoint, fingerprint, gpu_uuid, temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.authorize:
        raise ValueError("Choose exactly one mode")
    path = args.config.resolve()
    cfg, paths, spec, run, checkpoint, fingerprint, gpu_uuid, temperature = inspect(path)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "epoch": 38,
                          "probe_seconds": validate_probe(
                              paths["liveness_probe"], cfg["minimum_probe_seconds"]
                          )["duration_seconds"],
                          "test_accessed": False}))
        return
    evidence = {
        "schema_version": 1,
        "status": "manual_epoch38_resume_authorized",
        "task_name": TASK_NAME,
        "test_accessed": False,
        "automatic_retry": False,
        "last_complete_epoch": 38,
        "partial_epoch39_discarded": True,
        "study_fingerprint": fingerprint,
        "study_config_sha256": cfg["study_config_sha256"],
        "supervisor_loss_audit_sha256": cfg["supervisor_loss_audit_sha256"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "interrupted_status_sha256": cfg["interrupted_status_sha256"],
        "liveness_probe_sha256": file_sha256(paths["liveness_probe"]),
        "liveness_probe_seconds": validate_probe(
            paths["liveness_probe"], cfg["minimum_probe_seconds"]
        )["duration_seconds"],
        "gpu_uuid": gpu_uuid,
        "gpu_temperature_c": temperature,
        "recovery_config_sha256": file_sha256(path),
        "recovery_script_sha256": file_sha256(Path(__file__).resolve()),
        "recovery_protocol_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "docs"
            / "VALAIS_FP32VAL_EPOCH38_SUPERVISOR_RECOVERY_2026-10-01.md"
        ),
        "scheduler_wrapper_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "scripts"
            / "launch_valais_fp32val_resume_epoch38_task_20261001.ps1"
        ),
        "active_matching_processes": 0,
    }
    output = paths["authorization_output"]
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": evidence["status"], "epoch": 38,
                      "authorization_sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
