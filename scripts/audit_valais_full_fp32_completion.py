"""Read-only certificate for completed FP32 training after a wrapper false negative."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import torch

from pcdc.config import load_config
from pcdc.data import audit_data, file_sha256
from pcdc.engine import _digest, check_checkpoint, read_checkpoint, source_files, source_hashes, training_signature
from pcdc.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "valais_clean_full_fp32_seed42_20261002"
EVIDENCE = ROOT / "runs" / "valais_full_fp32_preflight_20261002"
OUTPUT = ROOT / "runs" / "valais_full_fp32_completion_audit_20261002"
CONFIG = ROOT / "configs" / "valais_clean_full_fp32_seed42_20261002.yaml"
TASK = "Codex-Valais-FullFP32-Seed42-20261002"
UUID = "GPU-8ada1895-7628-bb88-1dda-c57679117fa2"


def _finite_tree(value) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return False


def validate_history(history: list) -> tuple[int, float]:
    if (not isinstance(history, list) or len(history) != 100
            or [row.get("epoch") for row in history] != list(range(1, 101))
            or not _finite_tree(history)):
        raise RuntimeError("Completed history is incomplete or non-finite")
    scores = [row["val"]["metrics"]["f1"] for row in history]
    if any(not 0 <= score <= 1 for score in scores):
        raise RuntimeError("Validation F1 outside [0,1]")
    highest = max(scores)
    return scores.index(highest) + 1, highest


def _no_live_processes() -> None:
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    query = (
        "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID "
        "-and $_.Name -match '^(python|powershell)\\.exe$' "
        "-and $_.CommandLine -match 'valais_clean_full_fp32_seed42_20261002|"
        "launch_valais_full_fp32_task_20261002' } "
        "| Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    )
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
                            capture_output=True, text=True, check=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip() not in ("", "null"):
        raise RuntimeError("Full-FP32 training/launcher/guard process remains active")


def _task_definition(path: Path) -> None:
    root = ET.parse(path).getroot()
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    def value(query: str):
        item = root.find(query, ns)
        return None if item is None else item.text
    if (root.find("t:Triggers", ns) is None
            or list(root.find("t:Triggers", ns))
            or value("t:Settings/t:StopIfGoingOnBatteries") != "false"
            or value("t:Settings/t:MultipleInstancesPolicy") != "IgnoreNew"
            or value("t:Actions/t:Exec/t:Command") !=
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
            or "launch_valais_full_fp32_task_20261002.ps1" not in
            (value("t:Actions/t:Exec/t:Arguments") or "")):
        raise RuntimeError("One-shot Scheduler task definition differs")


def inspect():
    if OUTPUT.exists() or (RUN / "guard_completion.json").exists():
        raise RuntimeError("Separate certificate already exists or original marker is present")
    _no_live_processes()
    config = load_config(CONFIG)
    authorization_path = EVIDENCE / "authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if (authorization.get("status") != "full_fp32_training_authorized"
            or authorization.get("task_name") != TASK
            or authorization.get("config_sha256") != file_sha256(CONFIG)
            or authorization.get("test_accessed") is not False):
        raise RuntimeError("Training authorization differs")
    for name, expected in authorization["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise RuntimeError(f"Authorized source/input changed: {name}")
    _task_definition(EVIDENCE / "scheduler_task_definition.xml")
    task_result = json.loads((EVIDENCE / "scheduler_result.json").read_text(encoding="utf-8-sig"))
    if (task_result.get("task_name") != TASK
            or task_result.get("parent_name", "").lower() != "svchost.exe"
            or task_result.get("launcher_exit_code") != 0
            or task_result.get("status") != "failed"
            or task_result.get("error") !=
            "Training/guard/history did not complete under the frozen protocol"):
        raise RuntimeError("Wrapper false-negative identity differs")
    status_path = RUN / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or status.get("epochs") != 100
            or status.get("test_evaluated") is not False
            or status.get("best_val_f1") is None
            or config.run_test_after_training
            or config.values["train"]["amp"] is not False):
        raise RuntimeError("Trainer did not complete the frozen test-closed protocol")
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    best_epoch, best_f1 = validate_history(history)
    last = read_checkpoint(RUN / "last.pt")
    best = read_checkpoint(RUN / "best.pt")
    for checkpoint in (last, best):
        check_checkpoint(config, checkpoint)
        if (checkpoint.get("training_signature") != training_signature(config)
                or checkpoint.get("source_sha256") != source_hashes()
                or checkpoint.get("kind") != "research"):
            raise RuntimeError("Checkpoint source/training identity differs")
        if not all(torch.isfinite(value).all().item() for value in checkpoint["model"].values()
                   if torch.is_tensor(value)):
            raise RuntimeError("Completed model contains non-finite tensors")
    if (last.get("epoch") != 100 or len(last.get("history", [])) != 100
            or last["history"] != history
            or best.get("epoch") != best_epoch
            or best.get("best_val_f1") != best_f1
            or last.get("best_epoch") != best_epoch
            or last.get("best_val_f1") != best_f1
            or status.get("best_epoch") != best_epoch
            or status.get("best_val_f1") != best_f1):
        raise RuntimeError("Best checkpoint/status does not match complete history")
    live_audit = audit_data(config, full=True)
    archived_audit = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))
    if (live_audit != archived_audit
            or _digest(live_audit) != last.get("audit_sha256")
            or last.get("audit_sha256") != best.get("audit_sha256")):
        raise RuntimeError("Completed train/val input audit changed")
    archived_config = load_config(RUN / "config.yaml")
    if archived_config.values != config.values:
        # Archived config uses absolute data/output paths, so compare their resolved semantics.
        if (archived_config.data_root != config.data_root
                or archived_config.output_dir != config.output_dir
                or training_signature(archived_config) != training_signature(config)):
            raise RuntimeError("Archived run config differs from frozen source")
    snapshot = json.loads((RUN / "source_sha256.json").read_text(encoding="utf-8"))
    if snapshot != source_hashes():
        raise RuntimeError("Run source hash inventory differs")
    for name, expected in snapshot.items():
        if file_sha256(RUN / "source" / name) != expected:
            raise RuntimeError(f"Run source snapshot changed: {name}")
    guard_path = RUN / "temperature_guard_state.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    settings_path = RUN / "temperature_guard.yaml"
    import yaml
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if (guard.get("status") != "training_process_exited"
            or guard.get("training_pid") != settings.get("process_id")
            or guard.get("decision") != "ok"
            or guard.get("sensor_failures") != 0
            or guard.get("temperature_c", 999) >= 85
            or guard.get("settings", {}).get("creation_time_ticks") !=
            settings.get("creation_time_ticks")
            or guard.get("settings", {}).get("command_line") != settings.get("command_line")
            or guard.get("settings", {}).get("nvidia_smi") != settings.get("nvidia_smi")
            or guard.get("settings", {}).get("run_dir") != str(RUN)
            or (RUN / "temperature_guard.lock").exists()
            or (RUN / "guard_pause.json").exists()):
        raise RuntimeError("PID-bound temperature guard did not finish normally")
    train_log = RUN.with_suffix(".train.log")
    guard_log = RUN.with_suffix(".guard.log")
    outer_stdout = EVIDENCE / "scheduler_launcher_stdout.log"
    outer_stderr = EVIDENCE / "scheduler_launcher_stderr.log"
    text = train_log.read_text(encoding="utf-8-sig")
    if ("epoch=100 val_f1=" not in text or "Traceback" in text
            or "training_exit=0 guard_exit=0" not in outer_stdout.read_text(encoding="utf-8")
            or "Traceback" in guard_log.read_text(encoding="utf-8")
            or outer_stderr.read_text(encoding="utf-8").strip()):
        raise RuntimeError("Training and guard logs do not confirm normal child exit")
    if (datetime.fromisoformat(guard["ended_utc"]).timestamp() <= 0
            or status_path.stat().st_mtime_ns > guard_path.stat().st_mtime_ns):
        raise RuntimeError("Guard completion predates trainer completion")
    files = [CONFIG, authorization_path, EVIDENCE / "scheduler_task_definition.xml",
             EVIDENCE / "scheduler_result.json", status_path, RUN / "history.json",
             RUN / "best.pt", RUN / "last.pt", RUN / "data_audit.json",
             RUN / "source_sha256.json", RUN / "config.yaml",
             guard_path, settings_path, train_log, guard_log, outer_stdout, outer_stderr,
             ROOT / "scripts" / "launch_valais_full_fp32_task_20261002.ps1",
             ROOT / "scripts" / "run_valais_clean_guarded.py",
             ROOT / "scripts" / "temperature_guard.py",
             ROOT / "scripts" / "audit_valais_full_fp32_completion.py",
             ROOT / "docs" / "VALAIS_FULL_FP32_COMPLETION_AUDIT_2026-10-02.md",
             *[RUN / "source" / name for name in source_files()]]
    bindings = {str(path.resolve()): file_sha256(path) for path in files}
    return status, guard, best_epoch, best_f1, bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--certify", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.certify:
        raise ValueError("Choose exactly one audit mode")
    status, guard, best_epoch, best_f1, bindings = inspect()
    if args.preflight_only:
        print(json.dumps({"status": "audit_preflight_passed", "epochs": 100,
                          "bound_files": len(bindings), "test_accessed": False}))
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    certificate = {
        "schema_version": 1,
        "status": "audit_recovered_guarded_completion_only",
        "original_run_guard_completion_exists": False,
        "original_launcher_completed_normally": False,
        "trainer_completed_100_epochs": True,
        "final_guard_status": guard["status"],
        "best_epoch": best_epoch,
        "best_clean_val_f1": best_f1,
        "test_accessed": False,
        "model_inference_performed": False,
        "source_and_input_bindings_sha256": bindings,
        "limitation": "Read-only certification after Windows PowerShell history-array false negative; original launcher marker remains absent.",
    }
    write_json(OUTPUT / "certificate.json", certificate)
    print(json.dumps({"status": certificate["status"], "best_epoch": best_epoch,
                      "certificate_sha256": file_sha256(OUTPUT / "certificate.json"),
                      "test_accessed": False}))


if __name__ == "__main__":
    main()
