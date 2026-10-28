"""One-time audited administrative transition from stale running to interrupted."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from pcdc.config import load_config
from pcdc.data import audit_data, file_sha256
from pcdc.engine import _digest
from scripts.train_valais_fp32val import _read_checkpoint, _verify_snapshot, load_study


EXPECTED_AUDIT_SHA = "5a7567f2265dede5f08f62260893d7f0a828f1f323294fd3d06de7881234d8c5"
EXPECTED_LAST_SHA = "92e0d14acf9bad87cc6bfea61e36dc9d92a5aacc9c95a89e7b7b87e68b6bb607"
RUN_NAME = "valais_clean_fp32val_seed42_relaunch_20261001"


def read_scheduler_probe(path: Path) -> dict:
    probe = json.loads(path.read_text(encoding="utf-8-sig"))
    if (probe.get("status") != "completed"
            or probe.get("parent_name", "").lower() != "svchost.exe"
            or probe.get("gpu_accessed") is not False
            or probe.get("data_accessed") is not False):
        raise RuntimeError("Task Scheduler host probe did not pass")
    return probe


def interrupted_status(old_status: dict) -> dict:
    if (old_status.get("status") != "running"
            or old_status.get("epoch") != 27
            or old_status.get("test_evaluated") is not False):
        raise RuntimeError("Only the audited epoch27 stale status may be reclassified")
    return {
        **old_status,
        "status": "interrupted",
        "reason": "Administrative reclassification of a stale running status after an orphaned process tree; cause not established. The event-153/app-restart timeline is recorded in a separate audit.",
        "recovery_audit_sha256": EXPECTED_AUDIT_SHA,
        "last_complete_checkpoint_sha256": EXPECTED_LAST_SHA,
        "manual_resume_required": True,
    }


def _processes(workspace: Path) -> list[int]:
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    query = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
        f"-and $_.CommandLine -match '{RUN_NAME}' }} | "
        "Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
        cwd=workspace, capture_output=True, text=True, check=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    value = result.stdout.strip()
    if not value or value == "null":
        return []
    decoded = json.loads(value)
    return [decoded] if isinstance(decoded, int) else list(decoded)


def inspect(workspace: Path) -> tuple[dict, dict, Path, Path, dict]:
    audit_dir = workspace / "runs" / "valais_fp32val_relaunch_interrupt_20261001"
    audit_path = audit_dir / "audit.json"
    if file_sha256(audit_path) != EXPECTED_AUDIT_SHA:
        raise RuntimeError("Orphan audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    run = workspace / "runs" / RUN_NAME
    if (audit.get("status") != "orphaned_after_complete_epoch27"
            or audit.get("automatic_resume_authorized") is not False
            or audit.get("last_complete_checkpoint", {}).get("epoch") != 27):
        raise RuntimeError("Unexpected orphan audit state")
    paths = {
        "status": run / "status.json",
        "history": run / "history.json",
        "last_checkpoint": run / "last.pt",
        "best_checkpoint": run / "best.pt",
        "data_audit": run / "data_audit.json",
        "source_sha256": run / "source_sha256.json",
        "study_config": run / "study_config.yaml",
        "base_config": run / "base_config.yaml",
        "guard_state": run / "temperature_guard_state.json",
        "guard_config": run / "temperature_guard.yaml",
        "guard_attach_marker": run / "guard_attached.marker.json",
        "guard_handshake": run / "guard_attached.json",
        "preflight_seal": workspace / "runs" / f"{RUN_NAME}.preflight_seal.json",
        "train_log": workspace / "runs" / f"{RUN_NAME}.train.log",
        "guard_log": workspace / "runs" / f"{RUN_NAME}.guard.log",
        "launcher_stdout": workspace / "runs" / f"{RUN_NAME}.launcher_stdout.log",
        "launcher_stderr": workspace / "runs" / f"{RUN_NAME}.launcher_stderr.log",
        "status_snapshot": audit_dir / "status_before.json",
        "guard_snapshot": audit_dir / "guard_state_before.json",
        "train_log_snapshot": audit_dir / "train_log_before.log",
        "nvlddmkm_event_xml": audit_dir / "nvlddmkm_153.xml",
    }
    if set(paths) != set(audit["artifacts_sha256"]):
        raise RuntimeError("Orphan audit mapping is incomplete")
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != audit["artifacts_sha256"][name]:
            raise RuntimeError(f"Original orphan artifact changed: {name}")
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    if (status.get("status") != "running" or status.get("epoch") != 27
            or status.get("test_evaluated") is not False):
        raise RuntimeError("Stale running status no longer matches the audit")
    if (run / "guard_completion.json").exists() or (run / "last.pt.tmp").exists():
        raise RuntimeError("Unexpected completion or temporary checkpoint")
    checkpoint = _read_checkpoint(paths["last_checkpoint"])
    if (file_sha256(paths["last_checkpoint"]) != EXPECTED_LAST_SHA
            or checkpoint["epoch"] != 27
            or checkpoint["format_version"] != 2
            or checkpoint["test_evaluated"] is not False
            or len(checkpoint["history"]) != 27
            or not all(torch.isfinite(value).all().item()
                       for value in checkpoint["model"].values()
                       if torch.is_tensor(value))):
        raise RuntimeError("Complete epoch27 checkpoint is not intact")
    spec_path = workspace / "configs" / "valais_clean_fp32val_seed42_relaunch_20261001.yaml"
    spec, resolved, base, fingerprint = load_study(spec_path)
    if resolved["output_dir"] != run or checkpoint["study_fingerprint"] != fingerprint:
        raise RuntimeError("Resume study fingerprint differs")
    _verify_snapshot(run, json.loads(paths["source_sha256"].read_text(encoding="utf-8")))
    live_audit = audit_data(base, full=True)
    if (live_audit != json.loads(paths["data_audit"].read_text(encoding="utf-8"))
            or _digest(live_audit) != checkpoint["data_audit_digest"]):
        raise RuntimeError("Current train/val bytes differ from epoch27 audit")
    if _processes(workspace):
        raise RuntimeError("Valais training/launcher/guard process is still active")
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
        raise RuntimeError("GPU identity or temperature is unsuitable for recovery")
    read_scheduler_probe(workspace / "runs" / "valais_scheduler_probe_20261001.json")
    second_probe = read_scheduler_probe(
        workspace / "runs" / "valais_scheduler_no_trigger_probe_20261001.json"
    )
    if second_probe.get("kind") != "scheduler_no_trigger_probe_only":
        raise RuntimeError("No-trigger Task Scheduler probe identity differs")
    return audit, status, audit_dir, run, checkpoint, gpu_uuid, temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.apply:
        raise ValueError("Choose exactly one mode")
    workspace = Path(__file__).resolve().parents[1]
    audit, old_status, audit_dir, run, checkpoint, gpu_uuid, temperature = inspect(workspace)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "epoch": 27,
                          "active_matching_processes": 0, "gpu_uuid": gpu_uuid,
                          "temperature_c": temperature, "test_accessed": False}))
        return
    authorization = audit_dir / "recovery_authorization.json"
    if authorization.exists():
        raise FileExistsError(authorization)
    new_status = interrupted_status(old_status)
    status_path = run / "status.json"
    temp = run / "status.interrupted-tmp"
    if temp.exists():
        raise FileExistsError(temp)
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(new_status, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temp, status_path)
    record = {
        "schema_version": 1,
        "status": "manual_orphan_recovery_authorized",
        "test_accessed": False,
        "original_status_sha256": audit["artifacts_sha256"]["status"],
        "new_status_sha256": file_sha256(status_path),
        "orphan_audit_sha256": EXPECTED_AUDIT_SHA,
        "epoch27_checkpoint_sha256": EXPECTED_LAST_SHA,
        "recovery_script_sha256": file_sha256(Path(__file__).resolve()),
        "recovery_protocol_sha256": file_sha256(
            workspace / "docs" / "VALAIS_FP32VAL_EPOCH27_ORPHAN_RECOVERY_2026-10-01.md"
        ),
        "scheduler_probe_sha256": file_sha256(
            workspace / "runs" / "valais_scheduler_probe_20261001.json"
        ),
        "scheduler_no_trigger_probe_sha256": file_sha256(
            workspace / "runs" / "valais_scheduler_no_trigger_probe_20261001.json"
        ),
        "scheduler_no_trigger_probe_script_sha256": file_sha256(
            workspace / "scripts" / "scheduler_probe_no_trigger_20261001.ps1"
        ),
        "scheduler_resume_wrapper_sha256": file_sha256(
            workspace / "scripts" / "launch_valais_fp32val_resume_task_20261001.ps1"
        ),
        "partial_epoch28_discarded": True,
        "active_matching_processes": 0,
        "gpu_uuid": gpu_uuid,
        "temperature_c_at_transition": temperature,
        "original_running_status_preserved_at": str(audit_dir / "status_before.json"),
        "transition": "running/epoch27 -> interrupted/epoch27, administrative only",
    }
    with authorization.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "epoch": 27,
                      "authorization_sha256": file_sha256(authorization)}))


if __name__ == "__main__":
    main()
