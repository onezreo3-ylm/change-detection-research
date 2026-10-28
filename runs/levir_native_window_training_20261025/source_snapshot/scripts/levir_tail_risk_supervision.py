"""Console-free CPU owner, PID-bound thermal guard and separate wall watchdog.

No torch import in the supervisor/monitor paths. The worker must see all three
live, fresh creation-bound supervisors before importing the CUDA runner.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from scripts.temperature_guard import TemperaturePolicy, WindowsProcess, read_temperature
from scripts.windows_shared_json import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return read_json(path)


def atomic(path, value):
    atomic_json(path, value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def creation_ticks(pid):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    api.GetProcessTimes.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
    finally:
        api.CloseHandle(handle)


def load_runtime(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (cfg["schema_version"] != 1 or cfg["automatic_retry"] is not False
            or cfg["confirmed_pause_c"] != 85 or cfg["immediate_pause_c"] != 86
            or cfg["poll_seconds"] > 2 or cfg["confirmation_seconds"] > 1
            or cfg["single_training_limit_seconds"] != 86400
            or cfg["heartbeat_max_age_seconds"] > 30):
        raise ValueError("Frozen supervision policy changed")
    for key in ("python", "powershell", "nvidia_smi"):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    study = (path.parent / cfg["study_config"]).resolve()
    values = yaml.safe_load(study.read_text(encoding="utf-8"))
    root = (study.parent / values["workspace_root"]).resolve()
    if root != ROOT:
        raise ValueError("Wrong study workspace")
    cfg["study_path"] = str(study)
    cfg["study_output"] = str(root / values["output_dir"])
    cfg["ledger_path"] = str(root / values["managed_ledger"])
    return path, cfg


def query_command(cfg, pid):
    expression = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
                  "$p | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run([cfg["powershell"], "-NoProfile", "-NonInteractive", "-Command", expression],
        capture_output=True, text=True, check=True, timeout=20, creationflags=HIDDEN)
    item = json.loads(result.stdout)
    if not item or not item["ExecutablePath"] or not item["CommandLine"]:
        raise RuntimeError("Missing live process identity")
    return item


def identity(cfg, pid):
    item = query_command(cfg, pid)
    return {"process_id": pid, "creation_time_ticks": creation_ticks(pid),
            "executable": item["ExecutablePath"], "command_line": item["CommandLine"],
            "powershell": cfg["powershell"]}


def gpu_state(cfg):
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("No CUDA device remapping is admitted")
    result = subprocess.run([cfg["nvidia_smi"], "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        timeout=5, creationflags=HIDDEN)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected the original single physical GPU")
    uuid, temp, util, memory = [v.strip() for v in lines[0].split(",")]
    if uuid != cfg["gpu_uuid"]:
        raise ValueError("Physical GPU identity changed")
    return {"uuid": uuid, "temperature_c": int(temp), "utilization_percent": int(util), "memory_used_mib": int(memory)}


def age_seconds(utc, current=None):
    age = ((current or datetime.now(timezone.utc)) - datetime.fromisoformat(utc)).total_seconds()
    if age < -1:
        raise ValueError("Heartbeat clock is in the future")
    return max(age, 0)


def require_fresh(path, limit):
    value = read(path)
    if age_seconds(value["updated_utc"]) > limit or value.get("status") not in ("running", "monitoring"):
        raise RuntimeError("Supervisor heartbeat stale or inactive: " + str(path))
    return value


def ledger_update(cfg, entry, elapsed, status, exit_code=None):
    path = Path(cfg["ledger_path"])
    ledger = read(path)
    matches = [v for v in ledger["entries"] if v.get("id") == entry]
    if len(matches) != 1:
        raise ValueError("Compute charge must have one persistent identity")
    value = matches[0]
    if elapsed < value["charged_seconds"]:
        raise ValueError("Compute charge cannot decrease")
    value.update(charged_seconds=elapsed, status=status, updated_utc=now())
    if exit_code is not None:
        value["worker_exit_code"] = exit_code
    ledger["total_charged_seconds"] = sum(v["charged_seconds"] for v in ledger["entries"])
    if value.get("research_training"):
        root = Path(cfg["study_output"])
        committed_path = root / "candidate_seed42/status.json"
        committed = read(committed_path) if committed_path.exists() else {}
        successful = committed.get("optimizer_steps", 0)
        progress_path = root / "training_session/progress.json"
        if progress_path.exists():
            p = read(progress_path)
            if p.get("stage") == "candidate_train" and p["epoch"] > committed.get("epoch", 0):
                successful += p["step"] - p.get("amp_skips_in_stage", 0)
        value["successful_research_optimizer_updates"] = successful
        ledger["new_research_optimizer_updates"] = sum(v.get("successful_research_optimizer_updates", 0) for v in ledger["entries"])
    atomic(path, ledger)


def reserve_charge(cfg, mode, session):
    path = Path(cfg["ledger_path"])
    ledger = read(path)
    if any(v.get("status") == "running" for v in ledger["entries"]):
        raise RuntimeError("Another live or unreconciled compute reservation exists")
    remaining = ledger["batch_upper_bound_seconds"] - ledger["total_charged_seconds"]
    prior_training = sum(v["charged_seconds"] for v in ledger["entries"]
                         if v.get("kind") == "tail_risk_train_whole_worker_process")
    limit = min(remaining, cfg["pilot_wall_limit_seconds"] if mode == "pilot" else
                cfg["single_training_limit_seconds"] - prior_training)
    if limit <= 0:
        raise RuntimeError("Original cumulative compute budget exhausted")
    entry = "tail_risk_" + mode + "_" + str(session.relative_to(ROOT)).replace("\\", "/")
    if any(v.get("id") == entry for v in ledger["entries"]):
        raise RuntimeError("No automatic rerun or budget reset")
    ledger["entries"].append({"id": entry, "kind": "tail_risk_" + mode + "_whole_worker_process",
        "status": "running", "charged_seconds": 0.0, "started_utc": now(),
        "basis": "Conservative entire worker wall time including CPU/data/I/O, not pure kernel time.",
        "research_training": mode == "train", "maximum_seconds": limit})
    atomic(path, ledger)
    return entry, limit


def external_worker_check(cfg):
    # Check processes, not GPU utilization (WDDM often cannot attribute memory).
    expression = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' } | "
                  "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run([cfg["powershell"], "-NoProfile", "-NonInteractive", "-Command", expression],
        capture_output=True, text=True, check=True, timeout=20, creationflags=HIDDEN)
    processes = json.loads(result.stdout or "null") or []
    if isinstance(processes, dict):
        processes = [processes]
    import re
    for item in processes:
        command = item.get("CommandLine") or ""
        if item["ProcessId"] == os.getpid():
            continue
        if re.search(r"-m pcdc\s+(train|evaluate)|-m scripts\.[a-z_0-9]*(train|evaluate|smoke|replay|diagnose)[a-z_0-9]*\b|levir_tail_risk_supervision worker", command):
            raise RuntimeError("Another possible GPU worker exists: " + str(item["ProcessId"]))


def launch(path, mode):
    path, cfg = load_runtime(path)
    root = Path(cfg["study_output"])
    root.mkdir(parents=True, exist_ok=True)
    session = root / ("pilot" if mode == "pilot" else "training_session")
    if session.exists() or (root / "gpu_execution.lock").exists():
        raise FileExistsError("Existing execution evidence/lease; inspect without duplicate launch")
    if not (root / ("prepared.json" if mode == "pilot" else "authorization.json")).is_file():
        raise RuntimeError("Required preparation or training authorization missing")
    external_worker_check(cfg)
    gpu = gpu_state(cfg)
    if gpu["temperature_c"] >= cfg["confirmed_pause_c"]:
        raise RuntimeError("GPU is already at the thermal pause threshold")
    session.mkdir(exist_ok=False)
    command = [cfg["python"], "-X", "utf8", "-u", "-m", "scripts.levir_tail_risk_supervision", "owner",
               "--runtime", str(path), "--mode", mode]
    with (session / "owner.stdout.log").open("x", encoding="utf-8") as out, (session / "owner.stderr.log").open("x", encoding="utf-8") as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=HIDDEN)
    return {"status": "hidden_cpu_owner_dispatched_not_yet_training", "owner_pid": child.pid,
            "owner_creation_ticks": creation_ticks(child.pid), "session": str(session), "gpu": gpu}


def owner(path, mode):
    path, cfg = load_runtime(path)
    root = Path(cfg["study_output"])
    session = root / ("pilot" if mode == "pilot" else "training_session")
    lock = root / "gpu_execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    child, guard, watchdog, entry = None, None, None, None
    started = time.monotonic()
    state = {"status": "running", "mode": mode, "updated_utc": now(), "automatic_retry": False}
    try:
        external_worker_check(cfg)
        own = identity(cfg, os.getpid())
        atomic(session / "owner_identity.json", own)
        state.update(owner_pid=own["process_id"], owner_creation_ticks=own["creation_time_ticks"])
        atomic(session / "owner_state.json", state)
        entry, limit = reserve_charge(cfg, mode, session)
        command = [cfg["python"], "-X", "utf8", "-u", "-m", "scripts.levir_tail_risk_supervision", "worker",
                   "--runtime", str(path), "--mode", mode]
        with (session/"worker.stdout.log").open("x", encoding="utf-8") as out, (session/"worker.stderr.log").open("x", encoding="utf-8") as err:
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=HIDDEN)
        worker = identity(cfg, child.pid)
        atomic(session / "worker_identity.json", worker)
        state.update(worker_pid=child.pid, worker_creation_ticks=worker["creation_time_ticks"])
        settings = {k: cfg[k] for k in ("powershell", "nvidia_smi", "gpu_index", "poll_seconds",
            "confirmation_seconds", "confirmed_pause_c", "immediate_pause_c", "confirmation_samples", "max_sensor_failures")}
        settings.update(worker, run_dir=str(session))
        (session / "thermal_bound.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        supervisors = {}
        for role in ("guard", "watchdog"):
            command = [cfg["python"], "-X", "utf8", "-u", "-m", "scripts.levir_tail_risk_supervision", role,
                       "--runtime", str(path), "--mode", mode]
            with (session/(role+".stdout.log")).open("x", encoding="utf-8") as out, (session/(role+".stderr.log")).open("x", encoding="utf-8") as err:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=HIDDEN)
            if role == "guard":
                guard = proc
            else:
                watchdog = proc
            supervisors[role] = identity(cfg, proc.pid)
        atomic(session / "attachment.json", {"owner": own, "worker": worker, **supervisors,
            "allowed_wall_seconds": limit, "created_utc": now(), "runtime_sha256": sha(path),
            "source_sha256": sha(__file__)})
        state.update(guard_pid=guard.pid, watchdog_pid=watchdog.pid, allowed_wall_seconds=limit)
        atomic(session / "owner_state.json", dict(state, updated_utc=now()))
        while child.poll() is None:
            elapsed = time.monotonic() - started
            atomic(session / "owner_state.json", dict(state, updated_utc=now(), elapsed_seconds=elapsed))
            ledger_update(cfg, entry, elapsed, "running")
            if elapsed >= limit or guard.poll() is not None or watchdog.poll() is not None:
                raise RuntimeError("Wall budget exhausted or independent guard/watchdog exited")
            progress = session / "progress.json"
            if progress.exists():
                item = read(progress)
                if item.get("numeric_alert") or age_seconds(item["updated_utc"]) > cfg["progress_stall_seconds"]:
                    raise RuntimeError("Worker numeric alert or actual batch progress stalled")
            elif elapsed > cfg["progress_stall_seconds"]:
                raise RuntimeError("No actual worker progress within frozen limit")
            time.sleep(cfg["poll_seconds"])
        if child.returncode != 0 or not (session / "worker_result.json").exists():
            raise RuntimeError("Worker failed with exit code " + str(child.returncode))
        result = read(session / "worker_result.json")
        if result["status"] != "completed":
            raise RuntimeError("Worker result did not confirm completion")
        state.update(status="completed", worker_exit_code=child.returncode)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        state.update(status="failed_preserve", error=f"{type(error).__name__}: {error}",
                     worker_exit_code=child.returncode if child else None)
        traceback.print_exc()
    finally:
        elapsed = time.monotonic() - started
        state.update(updated_utc=now(), elapsed_seconds=elapsed)
        atomic(session / "owner_state.json", state)
        if entry:
            ledger_update(cfg, entry, elapsed, state["status"], state.get("worker_exit_code"))
        # Guards exit themselves after the bound worker ends; retain their evidence.
        for proc in (guard, watchdog):
            if proc:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=10)
        lock.unlink()
    if state["status"] != "completed":
        raise RuntimeError(state.get("error", "Supervisor failed"))
    return state


def wait_attachment(session, cfg):
    deadline = time.monotonic() + cfg["startup_limit_seconds"]
    while not (session / "attachment.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Supervisor attachment not available")
        time.sleep(0.2)
    proof = read(session / "attachment.json")
    if proof["source_sha256"] != sha(__file__):
        raise ValueError("Supervision code changed after binding")
    return proof


def guard_loop(path, mode, role):
    path, cfg = load_runtime(path)
    session = Path(cfg["study_output"]) / ("pilot" if mode == "pilot" else "training_session")
    proof = wait_attachment(session, cfg)
    if proof["runtime_sha256"] != sha(path):
        raise ValueError("Runtime changed after binding")
    process = WindowsProcess(proof["worker"])
    owner_handle = peer_handle = None
    state = {"status": "monitoring", "role": role, "pid": os.getpid(), "updated_utc": now()}
    started = time.monotonic()
    policy = TemperaturePolicy(cfg)
    peer = "watchdog" if role == "guard" else "guard"
    try:
        owner_handle, peer_handle = WindowsProcess(proof["owner"]), WindowsProcess(proof[peer])
        if os.getpid() != proof[role]["process_id"] or creation_ticks(os.getpid()) != proof[role]["creation_time_ticks"]:
            raise ValueError("Guard/watchdog creation identity mismatch")
        atomic(session / (role+"_state.json"), state)
        maximum = None
        while process.is_alive():
            elapsed = time.monotonic() - started
            if not owner_handle.is_alive() or not peer_handle.is_alive():
                raise RuntimeError("CPU owner or independent peer guard process ended")
            require_fresh(session / "owner_state.json", cfg["heartbeat_max_age_seconds"])
            peer_path = session / (peer+"_state.json")
            if peer_path.exists():
                require_fresh(peer_path, cfg["heartbeat_max_age_seconds"])
            elif elapsed > cfg["startup_limit_seconds"]:
                raise RuntimeError("Peer guard never became ready")
            state.update(updated_utc=now(), elapsed_seconds=elapsed)
            if elapsed > proof["allowed_wall_seconds"] or age_seconds(proof["created_utc"]) > proof["allowed_wall_seconds"]:
                raise RuntimeError("Independent original wall budget exhausted")
            progress = session / "progress.json"
            if progress.exists():
                p = read(progress)
                if p.get("numeric_alert") or age_seconds(p["updated_utc"]) > cfg["progress_stall_seconds"]:
                    raise RuntimeError("Numeric alert or real batch progress stalled")
            elif elapsed > cfg["progress_stall_seconds"]:
                raise RuntimeError("Worker never reported actual progress")
            if role == "guard":
                temperature = None
                sensor_error = None
                try:
                    sample = gpu_state(cfg)
                    temperature = sample["temperature_c"]
                    state["gpu"] = sample
                except Exception as error:
                    sensor_error = str(error)
                decision = policy.observe(temperature)
                if temperature is not None:
                    maximum = max(maximum or temperature, temperature)
                state.update(temperature_c=temperature, maximum_observed_temperature_c=maximum,
                             decision=decision, sensor_error=sensor_error)
                if decision in ("temperature_pause", "sensor_pause"):
                    raise RuntimeError("Independent temperature policy: " + decision)
            atomic(session / (role+"_state.json"), state)
            time.sleep(cfg["confirmation_seconds"] if role == "guard" and policy.hot_samples else cfg["poll_seconds"])
        state.update(status="worker_exited", updated_utc=now())
    except BaseException as error:
        if process.is_alive():
            process.stop()
            state.update(status="stopped_worker_preserve", error=f"{type(error).__name__}: {error}", updated_utc=now())
            traceback.print_exc()
        else:
            state.update(status="worker_exited", updated_utc=now())
    finally:
        atomic(session / (role+"_state.json"), state)
        for handle in (process, owner_handle, peer_handle):
            if handle:
                handle.close()
    return state


def worker(path, mode):
    path, cfg = load_runtime(path)
    session = Path(cfg["study_output"]) / ("pilot" if mode == "pilot" else "training_session")
    handles = []
    try:
        proof = wait_attachment(session, cfg)
        if proof["runtime_sha256"] != sha(path) or os.getpid() != proof["worker"]["process_id"] or creation_ticks(os.getpid()) != proof["worker"]["creation_time_ticks"]:
            raise ValueError("Worker runtime/creation identity mismatch")
        for role in ("owner", "guard", "watchdog"):
            handles.append(WindowsProcess(proof[role]))
        deadline = time.monotonic() + cfg["startup_limit_seconds"]
        while not all((session / (r+"_state.json")).exists() for r in ("guard", "watchdog")):
            if not all(h.is_alive() for h in handles) or time.monotonic() > deadline:
                raise RuntimeError("Independent supervisors not ready before CUDA")
            time.sleep(0.2)

        def check():
            for role, handle in zip(("owner", "guard", "watchdog"), handles):
                if not handle.is_alive():
                    raise RuntimeError("Bound supervisor no longer active: " + role)
                require_fresh(session / (role+"_state.json"), cfg["heartbeat_max_age_seconds"])
        check()
        atomic(session / "progress.json", {"stage": "supervisors_attached_before_cuda_import", "step": 0,
                                           "updated_utc": now(), "numeric_alert": False})
        from scripts import levir_tail_risk_control as runner
        result = getattr(runner, mode)(cfg["study_path"], session, cfg, check)
        atomic(session / "worker_result.json", {"status": "completed", "mode": mode,
            "completed_utc": now(), "result_status": result["status"], "test_accessed": False})
        return {"status": "completed"}
    except BaseException as error:
        atomic(session / "worker_result.json", {"status": "failed_preserve", "mode": mode,
            "error": f"{type(error).__name__}: {error}", "updated_utc": now(), "automatic_retry": False})
        raise
    finally:
        for handle in handles:
            handle.close()


def snapshot_identity(cfg, item):
    try:
        handle = WindowsProcess(item)
        try:
            return {"pid": item["process_id"], "creation_time_ticks": item["creation_time_ticks"], "active": handle.is_alive()}
        finally:
            handle.close()
    except Exception as error:
        return {"pid": item["process_id"], "active": False, "identity_error": str(error)}


def monitor(path, mode):
    _, cfg = load_runtime(path)
    root = Path(cfg["study_output"])
    session = root / ("pilot" if mode == "pilot" else "training_session")
    result = {"mode": mode, "gpu": gpu_state(cfg), "checked_utc": now()}
    if not session.exists():
        return dict(result, status="not_started")
    for name in ("owner_state", "progress", "guard_state", "watchdog_state", "worker_result"):
        if (session / (name+".json")).exists():
            result[name] = read(session / (name+".json"))
    if (session / "attachment.json").exists():
        proof = read(session / "attachment.json")
        result["processes"] = {role:snapshot_identity(cfg, proof[role]) for role in ("owner", "worker", "guard", "watchdog")}
    run = root / "candidate_seed42"
    if (run / "status.json").exists():
        result["committed_training"] = read(run / "status.json")
    for name in ("best.pt", "last.pt"):
        if (run / name).exists():
            result[name] = {"bytes": (run/name).stat().st_size, "modified_unix": (run/name).stat().st_mtime}
    for kind in ("worker.stdout.log", "worker.stderr.log"):
        p = session / kind
        if p.exists():
            with p.open("rb") as stream:
                stream.seek(max(0, p.stat().st_size-1200))
                result[kind] = stream.read().decode("utf-8", errors="replace").splitlines()[-6:]
    ledger = read(cfg["ledger_path"])
    result["charged_seconds"] = ledger["total_charged_seconds"]
    result["batch_budget_seconds"] = ledger["batch_upper_bound_seconds"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "owner", "worker", "guard", "watchdog", "monitor"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "train"), required=True)
    args = parser.parse_args()
    result = guard_loop(args.runtime, args.mode, args.action) if args.action in ("guard", "watchdog") else globals()[args.action](args.runtime, args.mode)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
