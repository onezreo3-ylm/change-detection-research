"""Separate read-only certificate when training finishes but its launcher does not."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import torch
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import audit_data, file_sha256
from pcdc.engine import _digest
from scripts.train_valais_fp32val import (
    _read_checkpoint, _verify_snapshot, load_study,
)


KEYS = {
    "schema_version", "kind", "study_config", "study_config_sha256",
    "supervisor_loss_audit", "supervisor_loss_audit_sha256", "output_dir",
    "required_epochs", "test_access", "original_run_guard_completion",
}


def _path(source: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def load_audit_config(path: Path):
    cfg = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(cfg, dict) or set(cfg) != KEYS:
        raise ValueError("Unexpected audit-recovery config keys")
    if (cfg["schema_version"] != 1
            or cfg["kind"] != "independent_guard_completion_audit_only"
            or cfg["required_epochs"] != 100
            or cfg["test_access"] != "forbidden"
            or cfg["original_run_guard_completion"] != "absent"):
        raise ValueError("Wrong audit-recovery protocol")
    paths = {
        key: _path(path, cfg[key])
        for key in ("study_config", "supervisor_loss_audit", "output_dir")
    }
    root = Path(__file__).resolve().parents[1]
    if paths["output_dir"] != root / "runs" / "valais_fp32val_guard_completion_audit_20261001":
        raise ValueError("Audit certificate output is not independently named")
    for key in ("study_config", "supervisor_loss_audit"):
        if (not paths[key].is_file()
                or file_sha256(paths[key]) != cfg[key + "_sha256"]):
            raise RuntimeError(f"Bound audit source changed: {key}")
    loss = json.loads(paths["supervisor_loss_audit"].read_text(encoding="utf-8"))
    if (loss.get("status") != "safely_interrupted_after_supervisor_loss"
            or loss.get("complete_checkpoint", {}).get("epoch") != 38
            or loss.get("test_accessed") is not False):
        raise RuntimeError("Supervisor-loss evidence no longer matches scope")
    spec, study_paths, base, fingerprint = load_study(paths["study_config"])
    return cfg, paths, spec, study_paths, base, fingerprint


def _finite_tree(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return False


def validate_history(history: list, epochs: int) -> tuple[int, float]:
    if not isinstance(history, list) or len(history) != epochs:
        raise RuntimeError("History length differs from the complete budget")
    if [row.get("epoch") for row in history] != list(range(1, epochs + 1)):
        raise RuntimeError("History epoch sequence has a gap or duplicate")
    if not _finite_tree(history):
        raise RuntimeError("History contains non-finite or unsupported values")
    scores = [row["val"]["metrics"]["f1"] for row in history]
    if any(not 0 <= score <= 1 for score in scores):
        raise RuntimeError("Validation F1 outside the supported range")
    best = max(scores)
    best_epoch = scores.index(best) + 1
    return best_epoch, best


def final_guard_session(run: Path) -> Path:
    parent = run / "temperature_guard_sessions"
    sessions = sorted(
        path for path in parent.glob("session_*")
        if path.is_dir() and re.fullmatch(r"session_[0-9]{4}", path.name)
    )
    if not sessions:
        raise RuntimeError("No guarded resume session exists")
    return sessions[-1]


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


def _bind(path: Path, identity: dict[str, str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    identity[str(path.resolve())] = file_sha256(path)


def inspect(config_path: Path):
    cfg, paths, spec, study_paths, base, fingerprint = load_audit_config(config_path)
    run = study_paths["output_dir"]
    if paths["output_dir"].exists():
        raise FileExistsError(paths["output_dir"])
    if not run.is_dir() or (run / "guard_completion.json").exists():
        raise RuntimeError("Original launcher completion exists or study is missing")
    if _matching_processes(run.name):
        raise RuntimeError("Training, launcher or guard process is still active")
    status_path = run / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or status.get("epochs") != 100
            or status.get("test_evaluated") is not False
            or status.get("training_amp") is not True
            or status.get("validation_amp") is not False):
        raise RuntimeError("Trainer did not report complete, test-closed FP32 validation")
    last_path = run / "last.pt"
    best_path = run / "best.pt"
    last = _read_checkpoint(last_path)
    best = _read_checkpoint(best_path)
    if (last.get("epoch") != 100 or last.get("test_evaluated") is not False
            or last.get("training_amp") is not True
            or last.get("validation_amp") is not False
            or last.get("study_fingerprint") != fingerprint
            or best.get("study_fingerprint") != fingerprint
            or best.get("test_evaluated") is not False):
        raise RuntimeError("Completed checkpoint identity differs")
    best_epoch, best_f1 = validate_history(last["history"], 100)
    if (last["best_epoch"] != best_epoch or last["best_val_f1"] != best_f1
            or best["epoch"] != best_epoch or best["best_val_f1"] != best_f1
            or status.get("best_epoch") != best_epoch
            or status.get("best_val_f1") != best_f1):
        raise RuntimeError("Checkpoint selection does not match history")
    if not all(torch.isfinite(value).all().item()
               for checkpoint in (last, best)
               for value in checkpoint["model"].values()
               if torch.is_tensor(value)):
        raise RuntimeError("Completed model tensor is non-finite")
    snapshot = json.loads((run / "source_sha256.json").read_text(encoding="utf-8"))
    _verify_snapshot(run, snapshot)
    archived_audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    live_audit = audit_data(base, full=True)
    if (archived_audit != live_audit
            or _digest(live_audit) != last["data_audit_digest"]
            or last["data_audit_digest"] != best["data_audit_digest"]):
        raise RuntimeError("Train/val data changed since completed checkpoints")
    if (file_sha256(run / "study_config.yaml") != cfg["study_config_sha256"]
            or file_sha256(run / "base_config.yaml") != spec["base_config_sha256"]):
        raise RuntimeError("Archived study/base configuration differs")
    session = final_guard_session(run)
    state_path = session / "temperature_guard_state.json"
    marker_path = session / "guard_attached.marker.json"
    guard_config = session / "temperature_guard.yaml"
    guard_log = run.parent / f"{run.name}.{session.name}.guard.log"
    train_log = run.parent / f"{run.name}.{session.name}.train.log"
    seal_path = run.parent / f"{run.name}.{session.name}.preflight_seal.json"
    if (session / "temperature_guard.lock").exists() or (session / "guard_pause.json").exists():
        raise RuntimeError("Guard lock/pause artifact remains")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(guard_config.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (state.get("status") != "training_process_exited"
            or state.get("training_pid") != settings.get("process_id")
            or state.get("sensor_failures") != 0
            or not state.get("ended_utc")
            or marker.get("status") != "bound_guard_attached"
            or marker.get("process_id") != settings.get("process_id")
            or marker.get("creation_time_ticks") != settings.get("creation_time_ticks")
            or marker.get("guard_config_sha256") != file_sha256(guard_config)
            or seal.get("status") != "independent_numeric_protocol_preflight_sealed"
            or seal.get("fingerprint") != fingerprint
            or seal.get("test_accessed") is not False):
        raise RuntimeError("Final PID-bound guard or preflight seal is incomplete")
    bound = seal.get("bound_sha256")
    if not isinstance(bound, dict) or not bound:
        raise RuntimeError("Final session preflight seal has no file bindings")
    for name, expected in bound.items():
        if not Path(name).is_file() or file_sha256(Path(name)) != expected:
            raise RuntimeError(f"Final session bound input/source changed: {name}")
    contents = train_log.read_text(encoding="utf-8")
    if ("epoch=100 val_f1=" not in contents or "Traceback" in contents
            or "temperature_pause" in guard_log.read_text(encoding="utf-8")):
        raise RuntimeError("Final session logs do not show normal completion")
    if (status_path.stat().st_mtime_ns > state_path.stat().st_mtime_ns
            or datetime.fromisoformat(state["ended_utc"]).timestamp() <= 0):
        raise RuntimeError("Final guard state predates completed trainer status")
    identity: dict[str, str] = {}
    for path in (
        config_path, paths["study_config"], paths["supervisor_loss_audit"],
        status_path, last_path, best_path, run / "history.json",
        run / "data_audit.json", run / "source_sha256.json",
        run / "study_config.yaml", run / "base_config.yaml",
        state_path, marker_path, guard_config, train_log, guard_log, seal_path,
        Path(__file__).resolve(),
    ):
        _bind(path, identity)
    return cfg, paths, run, status, last, best, session, state, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--certify", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.certify:
        raise ValueError("Choose exactly one audit mode")
    cfg, paths, run, status, last, best, session, state, identity = inspect(
        args.config.resolve()
    )
    if args.preflight_only:
        print(json.dumps({"status": "audit_preflight_passed", "epoch": last["epoch"],
                          "bound_files": len(identity), "test_accessed": False}))
        return
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    certificate = {
        "schema_version": 1,
        "status": "audit_recovered_guarded_completion_only",
        "original_run_guard_completion_exists": False,
        "original_launcher_completed_normally": False,
        "trainer_completed_100_epochs": True,
        "final_guard_session": session.name,
        "final_guard_status": state["status"],
        "best_epoch": best["epoch"],
        "best_clean_val_f1": best["best_val_f1"],
        "test_accessed": False,
        "model_inference_performed": False,
        "source_and_input_bindings_sha256": identity,
        "limitation": "Read-only certification of a finished trainer and guard after launcher loss; not a retroactive original launcher completion marker.",
    }
    path = output / "certificate.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(certificate, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": certificate["status"], "best_epoch": best["epoch"],
                      "certificate_sha256": file_sha256(path), "test_accessed": False}))


if __name__ == "__main__":
    main()
