"""Authorize a distinct, full-FP32 Valais clean run after numeric replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess

from pcdc.config import load_config
from pcdc.data import audit_data, file_sha256
from pcdc.engine import source_files
from pcdc.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
TASK = "Codex-Valais-FullFP32-Seed42-20261002"
UUID = "GPU-8ada1895-7628-bb88-1dda-c57679117fa2"
BASE_SHA = "7a4931665591a24e005b7ea3b42b1cb88855accc52343c8642ad4efc5ceea6ab"
DATA_SHA = "670d7d29361125e291d637f9fb9be44df31b67912dc2af837d719f6c4ae2cd87"
FAILURE_SHA = "58642ffca1155b5d389b03e20c7a4b0ae1426d6d84a42f17384aaabe754bd76a"
REPLAY_STATUS = "reproduced_train_nonfinite"


def _same_except_numeric(base: dict, proposed: dict) -> bool:
    original = copy.deepcopy(base)
    changed = copy.deepcopy(proposed)
    original["experiment"].pop("name")
    changed["experiment"].pop("name")
    original["experiment"].pop("output_dir")
    changed["experiment"].pop("output_dir")
    original["train"]["amp"] = False
    return original == changed


def inspect():
    config_path = ROOT / "configs" / "valais_clean_full_fp32_seed42_20261002.yaml"
    base_path = ROOT / "configs" / "valais_clean_development_seed42.yaml"
    data_path = ROOT / "runs" / "valais_clean_development_seed42" / "data_audit.json"
    failure_path = ROOT / "runs" / "valais_fp32val_train_nonfinite_20261001" / "audit.json"
    replay_dir = ROOT / "runs" / "valais_train_nonfinite_replay_epoch46_20261002"
    preflight = ROOT / "runs" / "valais_full_fp32_preflight_20261002"
    authorization = preflight / "authorization.json"
    unittest_log = preflight / "unittest_final.log"
    smoke_path = preflight / "cuda_smoke_final.json"
    run = ROOT / "runs" / "valais_clean_full_fp32_seed42_20261002"
    if run.exists() or authorization.exists():
        raise FileExistsError("Full-FP32 run/authorization already exists")
    if (file_sha256(base_path) != BASE_SHA or file_sha256(data_path) != DATA_SHA
            or file_sha256(failure_path) != FAILURE_SHA):
        raise RuntimeError("Source numeric failure, base protocol or data audit changed")
    base, proposed = load_config(base_path), load_config(config_path)
    if (not _same_except_numeric(base.values, proposed.values)
            or proposed.values["train"]["amp"] is not False
            or proposed.values["train"]["epochs"] != 100
            or proposed.values["train"]["batch_size"] != 4
            or proposed.values["evaluation"]["threshold"] != 0.5
            or proposed.run_test_after_training
            or proposed.values["runtime"]["device"] != "cuda"
            or proposed.output_dir != run):
        raise RuntimeError("Full-FP32 protocol changed more than name/output/train precision")
    if audit_data(proposed, full=True) != json.loads(data_path.read_text(encoding="utf-8")):
        raise RuntimeError("Live train/val data differs from source audit")
    completion = json.loads((replay_dir / "completion.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_dir / "replay.json").read_text(encoding="utf-8"))
    first = replay.get("first_nonfinite", {})
    if (completion.get("status") != "guarded_engineering_replay_complete"
            or completion.get("replay_status") != REPLAY_STATUS
            or completion.get("test_accessed") is not False
            or completion.get("original_run_modified") is not False
            or completion.get("replay_sha256") != file_sha256(replay_dir / "replay.json")
            or completion.get("guard_state_sha256") !=
            file_sha256(replay_dir / "temperature_guard_state.json")
            or first.get("step") != 1104
            or first.get("amp_logits", {}).get("nan") != 65536
            or first.get("fp32_logits", {}).get("nan") != 0
            or not isinstance(first.get("fp32_losses", {}).get("total"), float)):
        raise RuntimeError("Guarded train-numeric replay does not support FP32 protocol")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (smoke.get("status") != "engineering_smoke_only"
            or smoke.get("test_accessed") is not False
            or smoke.get("training_amp") is not False
            or smoke.get("validation_amp") is not False
            or smoke.get("train_batch_size") != 4
            or smoke.get("train_logits_dtype") != "torch.float32"
            or smoke.get("val_logits_dtype") != "torch.float32"
            or smoke.get("gradient_norm_finite") is not True
            or smoke.get("one_optimizer_step") is not True):
        raise RuntimeError("Real full-FP32 CUDA smoke did not pass")
    log = unittest_log.read_text(encoding="utf-8-sig")
    counts = re.findall(r"Ran (\d+) tests in", log)
    if not counts or int(counts[-1]) < 251 or not log.rstrip().endswith("OK"):
        raise RuntimeError("Full unittest did not pass")
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    query = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
             "-and $_.CommandLine -match 'pcdc train|train_valais_fp32val|guard_valais_fp32val' } "
             "| Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
    active = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
                            capture_output=True, text=True, check=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    if active and active != "null":
        raise RuntimeError("Another Valais training or guard process is active")
    task_query = (f"Get-ScheduledTask | Where-Object {{ $_.TaskName -eq '{TASK}' }} "
                  "| Select-Object -ExpandProperty TaskName | ConvertTo-Json -Compress")
    task = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", task_query],
                          capture_output=True, text=True, check=True, timeout=20,
                          creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    if task and task != "null":
        raise RuntimeError("Full-FP32 Scheduler task already exists")
    nvidia = Path("C:/Windows/System32/nvidia-smi.exe")
    value = subprocess.run([str(nvidia), "-i", "0", "--query-gpu=uuid,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, check=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    uuid, temperature = [part.strip() for part in value.split(",")]
    if uuid != UUID or int(temperature) >= 85:
        raise RuntimeError("GPU UUID or temperature differs from safe launch policy")
    bound = [
        config_path, base_path, data_path, failure_path,
        replay_dir / "completion.json", replay_dir / "replay.json",
        replay_dir / "temperature_guard_state.json",
        unittest_log, smoke_path,
        ROOT / "scripts" / "authorize_valais_full_fp32.py",
        ROOT / "scripts" / "smoke_valais_full_fp32.py",
        ROOT / "scripts" / "launch_valais_full_fp32_task_20261002.ps1",
        ROOT / "scripts" / "run_valais_clean_guarded.py",
        ROOT / "scripts" / "temperature_guard.py",
        ROOT / "docs" / "VALAIS_FULL_FP32_CLEAN_PROTOCOL_2026-10-02.md",
        *source_files().values(),
    ]
    return authorization, config_path, int(temperature), bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args()
    authorization, config_path, temperature, bound = inspect()
    if not args.authorize:
        print(json.dumps({"status": "preflight_passed", "test_accessed": False,
                          "gpu_temperature_c": temperature}))
        return
    report = {
        "schema_version": 1,
        "status": "full_fp32_training_authorized",
        "task_name": TASK,
        "test_accessed": False,
        "training_amp": False,
        "validation_amp": False,
        "epochs": 100,
        "gpu_uuid": UUID,
        "gpu_temperature_c": temperature,
        "config_sha256": file_sha256(config_path),
        "bound_sha256": {str(path.resolve()): file_sha256(path) for path in bound},
    }
    write_json(authorization, report)
    print(json.dumps({"status": report["status"],
                      "authorization_sha256": file_sha256(authorization),
                      "test_accessed": False}))


if __name__ == "__main__":
    main()
