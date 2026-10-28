"""CPU owner and independent guards for the frozen train-only tiling probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import yaml

from scripts import levir_tail_risk_supervision as shared
from scripts.temperature_guard import TemperaturePolicy, WindowsProcess
from scripts.windows_shared_json import atomic_json as atomic, read_json as read

ROOT = Path(__file__).resolve().parents[1]
MODULE = "scripts.tiling_probe_supervision"


def load_runtime(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    study = (path.parent / cfg["study_config"]).resolve()
    if study != ROOT / "configs/levir_tiling_context_probe_20261025.yaml":
        raise ValueError("Only the frozen tiling study is admitted")
    values = yaml.safe_load(study.read_text(encoding="utf-8"))
    if (cfg["schema_version"] != 1 or cfg["automatic_retry"] is not False
            or cfg["poll_seconds"] != 2 or cfg["confirmation_seconds"] != 1
            or cfg["confirmed_pause_c"] != 85 or cfg["immediate_pause_c"] != 86
            or cfg["confirmation_samples"] != 3 or cfg["max_sensor_failures"] != 3
            or cfg["heartbeat_max_age_seconds"] != 30 or cfg["sensor_timeout_seconds"] != 1
            or cfg["pilot_wall_limit_seconds"] != 900 or cfg["main_wall_limit_seconds"] != 1800
            or cfg["whole_probe_limit_seconds"] != 3600):
        raise ValueError("Frozen supervision limits changed")
    if (cfg["gpu_uuid"] != values["runtime"]["gpu_uuid"] or cfg["gpu_index"] != 0
            or cfg["python"] != values["runtime"]["python"]
            or values["constraints"]["raw_splits_allowed"] != ["train"]
            or values["constraints"]["optimizer_updates"] != 0):
        raise ValueError("Device, interpreter or train-only scope changed")
    for key in ["python", "powershell", "nvidia_smi"]:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    cfg.update(study_path=str(study), study_output=str(ROOT / values["output_dir"]),
               ledger_path=str(ROOT / values["budget"]["shared_ledger"]),
               global_lease=str(ROOT / values["managed_run_dir"] / "gpu_study.lock"))
    return path, cfg


def gpu_state(cfg):
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("CUDA remapping is not admitted")
    r = subprocess.run([cfg["nvidia_smi"], "--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        timeout=cfg["sensor_timeout_seconds"], creationflags=shared.HIDDEN)
    lines = r.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected the original single physical GPU")
    uuid, temp, util, memory = [v.strip() for v in lines[0].split(",")]
    if uuid != cfg["gpu_uuid"]:
        raise ValueError("Physical GPU UUID changed")
    return {"uuid": uuid, "temperature_c": int(temp), "utilization_percent": int(util), "memory_used_mib": int(memory)}


def verify(bound):
    for name, expected in bound.items():
        if shared.sha(name) != expected:
            raise ValueError("Sealed input changed: " + name)


def session_path(cfg, mode):
    if mode not in ("pilot", "main"):
        raise ValueError("Only pilot/main inference modes are admitted")
    return Path(cfg["study_output"]) / (mode + "_session")


def reserve(cfg, mode):
    if mode not in ("pilot", "main"):
        raise ValueError("No training or retry mode")
    ledger = read(cfg["ledger_path"])
    if any(e.get("status") == "running" for e in ledger["entries"]):
        raise RuntimeError("Active or unreconciled compute reservation")
    entry_id = "tiling_probe_gpu_" + mode + "_20261025"
    if any(e.get("id") == entry_id for e in ledger["entries"]):
        raise RuntimeError("One execution per mode; no retry or budget reset")
    spent = sum(e["charged_seconds"] for e in ledger["entries"] if e.get("kind") in
                ("tiling_probe_gpu_pilot", "tiling_probe_gpu_main"))
    limit = min(cfg[mode + "_wall_limit_seconds"], cfg["whole_probe_limit_seconds"] - spent,
                ledger["batch_upper_bound_seconds"] - ledger["total_charged_seconds"])
    if limit <= 0:
        raise RuntimeError("Original shared/probe budget exhausted")
    ledger["entries"].append({"id": entry_id, "kind": "tiling_probe_gpu_" + mode,
        "status": "running", "charged_seconds": 0., "started_utc": shared.now(),
        "maximum_seconds": limit, "research_training": False, "optimizer_updates": 0,
        "basis": "Conservative entire owner process wall including startup, CPU, I/O and guard cleanup."})
    atomic(cfg["ledger_path"], ledger)
    return entry_id, limit


def seal(path, checks_path, smoke_path):
    path, cfg = load_runtime(path)
    out = Path(cfg["study_output"])
    checks, smoke = read(checks_path), read(smoke_path)
    if not checks.get("passed") or smoke.get("status") != "passed" or smoke.get("synthetic_only") is not True:
        raise ValueError("Passing complete tests and explicit synthetic smoke are required")
    engineering = ROOT / "runs/levir_tiling_context_engineering_20261025/verification_engineering.json"
    prior = read(engineering)
    if prior["status"] != "inference_core_verified_supervision_adapter_pending":
        raise ValueError("Verified frozen inference core is required")
    prepared = read(out / "prepared.json")
    verify(prepared["bound_sha256"])
    files = [path, Path(__file__), ROOT / "scripts/tiling_probe_cuda.py",
        ROOT / "tests/test_tiling_probe_supervision.py", ROOT / "scripts/levir_tail_risk_supervision.py",
        ROOT / "scripts/temperature_guard.py", ROOT / "scripts/windows_shared_json.py",
        out / "prepared.json", engineering, Path(checks_path).resolve(), Path(smoke_path).resolve()]
    authority = {"status": "sealed_train_only_tiling_probe", "created_utc": shared.now(),
        "gpu_uuid": cfg["gpu_uuid"], "optimizer_updates": 0, "raw_splits_allowed": ["train"],
        "runtime": str(path), "bound_sha256": {str(p): shared.sha(p) for p in files},
        "pilot_pairs_upper_bound": 16, "main_pairs": 3168, "main_extra_self_check_pairs": 16,
        "new_training_admitted": False, "automatic_retry": False}
    authority_path = out / "authorization.json"
    with authority_path.open("x", encoding="utf-8") as stream:
        json.dump(authority, stream, indent=2, ensure_ascii=False)
    return {"status": authority["status"], "file": str(authority_path)}


def external_worker_check(cfg):
    shared.external_worker_check(cfg)
    expression = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    p = subprocess.run([cfg["powershell"], "-NoProfile", "-NonInteractive", "-Command", expression],
        capture_output=True, text=True, check=True, timeout=20, creationflags=shared.HIDDEN)
    items = json.loads(p.stdout or "null") or []
    if isinstance(items, dict):
        items = [items]
    for item in items:
        if item["ProcessId"] != os.getpid() and (MODULE + " worker") in (item.get("CommandLine") or ""):
            raise RuntimeError("Another tiling GPU worker is active")


def launch(path, mode):
    path, cfg = load_runtime(path)
    out, session = Path(cfg["study_output"]), session_path(cfg, mode)
    if session.exists() or Path(cfg["global_lease"]).exists():
        raise FileExistsError("Existing session/lease; no duplicate launch")
    authority = read(out / "authorization.json")
    if authority.get("status") != "sealed_train_only_tiling_probe" or authority.get("gpu_uuid") != cfg["gpu_uuid"]:
        raise ValueError("Explicit frozen runtime authority required")
    verify(authority["bound_sha256"])
    if mode == "main":
        prior = session_path(cfg, "pilot")
        if read(prior / "owner_state.json")["status"] != "completed" or read(prior / "worker_result.json")["status"] != "completed":
            raise RuntimeError("Completed successful pilot required")
        for role in ["guard", "watchdog"]:
            if read(prior / (role + "_state.json"))["status"] != "worker_exited":
                raise RuntimeError("Pilot guards have not completed normally")
    external_worker_check(cfg)
    sample = gpu_state(cfg)
    if sample["temperature_c"] >= cfg["confirmed_pause_c"]:
        raise RuntimeError("Initial thermal stop")
    session.mkdir(exist_ok=False)
    command = [cfg["python"], "-X", "utf8", "-u", "-m", MODULE, "owner", "--runtime", str(path), "--mode", mode]
    with (session / "owner.stdout.log").open("x") as a, (session / "owner.stderr.log").open("x") as b:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=a, stderr=b, creationflags=shared.HIDDEN)
    return {"status": "hidden_cpu_owner_dispatched", "mode": mode, "owner_pid": proc.pid,
            "owner_creation_ticks": shared.creation_ticks(proc.pid), "gpu": sample}


def owner(path, mode):
    path, cfg = load_runtime(path)
    session = session_path(cfg, mode)
    lease = Path(cfg["global_lease"])
    with lease.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "mode": mode, "created_utc": shared.now()}, stream)
    child, guards, entry = None, [], None
    start = time.monotonic()
    state = {"status": "running", "mode": mode, "optimizer_updates": 0, "updated_utc": shared.now(), "automatic_retry": False}
    try:
        external_worker_check(cfg)
        verify(read(Path(cfg["study_output"]) / "authorization.json")["bound_sha256"])
        own = shared.identity(cfg, os.getpid())
        entry, limit = reserve(cfg, mode)
        state["allowed_wall_seconds"] = limit
        atomic(session / "owner_state.json", dict(state, allowed_wall_seconds=limit))
        def spawn(role):
            command = [cfg["python"], "-X", "utf8", "-u", "-m", MODULE, role, "--runtime", str(path), "--mode", mode]
            with (session / (role + ".stdout.log")).open("x") as a, (session / (role + ".stderr.log")).open("x") as b:
                return subprocess.Popen(command, cwd=ROOT, stdout=a, stderr=b, creationflags=shared.HIDDEN)
        child = spawn("worker")
        identities = {"owner": own, "worker": shared.identity(cfg, child.pid)}
        for role in ["guard", "watchdog"]:
            proc = spawn(role); guards.append(proc); identities[role] = shared.identity(cfg, proc.pid)
        proof = dict(identities, allowed_wall_seconds=limit, created_utc=state["updated_utc"],
            source_sha256=shared.sha(shared.__file__), adapter_sha256=shared.sha(__file__),
            runtime_sha256=shared.sha(path), optimizer_updates=0, mode=mode)
        atomic(session / "attachment.json", proof)
        while child.poll() is None:
            elapsed = time.monotonic() - start
            atomic(session / "owner_state.json", dict(state, updated_utc=shared.now(), elapsed_seconds=elapsed, allowed_wall_seconds=limit))
            shared.ledger_update(cfg, entry, elapsed, "running")
            if elapsed >= limit or any(p.poll() is not None for p in guards):
                raise RuntimeError("External wall cap or independent supervisor exit")
            if (session / "progress.json").exists():
                p = read(session / "progress.json")
                if p.get("numeric_alert") or shared.age_seconds(p["updated_utc"]) > cfg["progress_stall_seconds"]:
                    raise RuntimeError("Numeric alert or stalled actual progress")
            elif elapsed > cfg["progress_stall_seconds"]:
                raise RuntimeError("Worker reported no progress")
            time.sleep(cfg["poll_seconds"])
        if child.returncode != 0 or read(session / "worker_result.json")["status"] != "completed":
            raise RuntimeError("Inference worker failed: " + str(child.returncode))
        state.update(status="completed", worker_exit_code=0)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait(timeout=15)
        state.update(status="failed_preserve", worker_exit_code=child.returncode if child else None,
                     error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        for proc in guards:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait(timeout=10)
            if proc.returncode and state["status"] == "completed":
                state.update(status="failed_preserve", error="Supervisor exit code " + str(proc.returncode))
        elapsed = time.monotonic() - start
        state.update(updated_utc=shared.now(), elapsed_seconds=elapsed)
        atomic(session / "owner_state.json", state)
        if entry:
            shared.ledger_update(cfg, entry, elapsed, state["status"], state.get("worker_exit_code"))
        lease.unlink()
    if state["status"] != "completed":
        raise RuntimeError(state.get("error", "Owner failed"))
    return state


def attachment(path, cfg, session):
    proof = shared.wait_attachment(session, cfg)
    if proof["runtime_sha256"] != shared.sha(path) or proof["adapter_sha256"] != shared.sha(__file__):
        raise ValueError("Bound runtime/adapter changed")
    return proof


def guard_loop(path, mode, role):
    path, cfg = load_runtime(path); session = session_path(cfg, mode)
    proof = attachment(path, cfg, session)
    if proof[role]["process_id"] != os.getpid() or proof[role]["creation_time_ticks"] != shared.creation_ticks(os.getpid()):
        raise ValueError("Supervisor creation identity differs")
    handles = {r: WindowsProcess(proof[r]) for r in ["worker", "owner", "watchdog" if role == "guard" else "guard"]}
    worker_handle = handles["worker"]
    state = {"status": "monitoring", "role": role, "updated_utc": shared.now()}
    policy = TemperaturePolicy(cfg); maximum = None; start = time.monotonic()
    try:
        atomic(session / (role + "_state.json"), state)
        while worker_handle.is_alive():
            elapsed = time.monotonic() - start
            for peer, handle in handles.items():
                if peer == "worker":
                    continue
                if not handle.is_alive():
                    raise RuntimeError("Bound owner/peer exited")
                file = session / (peer + "_state.json")
                if file.exists():
                    shared.require_fresh(file, cfg["heartbeat_max_age_seconds"])
                elif elapsed > cfg["startup_limit_seconds"]:
                    raise RuntimeError("Peer heartbeat never ready")
            if elapsed >= proof["allowed_wall_seconds"] or shared.age_seconds(proof["created_utc"]) >= proof["allowed_wall_seconds"]:
                raise RuntimeError("Independent wall-clock cap")
            if (session / "progress.json").exists():
                p = read(session / "progress.json")
                if p.get("numeric_alert") or shared.age_seconds(p["updated_utc"]) > cfg["progress_stall_seconds"]:
                    raise RuntimeError("Independent numeric/progress stop")
            elif elapsed > cfg["progress_stall_seconds"]:
                raise RuntimeError("No actual worker progress")
            if role == "guard":
                temperature, sensor_error = None, None
                try:
                    sample = gpu_state(cfg); temperature = sample["temperature_c"]; state["gpu"] = sample
                except Exception as error:
                    sensor_error = str(error)
                decision = policy.observe(temperature)
                if temperature is not None:
                    maximum = max(maximum if maximum is not None else temperature, temperature)
                state.update(temperature_c=temperature, maximum_observed_temperature_c=maximum,
                             sensor_error=sensor_error, decision=decision)
                if decision in ("temperature_pause", "sensor_pause"):
                    raise RuntimeError("Independent temperature/sensor stop: " + decision)
            atomic(session / (role + "_state.json"), dict(state, updated_utc=shared.now()))
            time.sleep(cfg["confirmation_seconds"] if role == "guard" and policy.hot_samples else cfg["poll_seconds"])
        state.update(status="worker_exited")
    except BaseException as error:
        if worker_handle.is_alive():
            worker_handle.stop(); state.update(status="stopped_worker_preserve", error=f"{type(error).__name__}: {error}")
        else:
            state.update(status="worker_exited")
    finally:
        atomic(session / (role + "_state.json"), dict(state, updated_utc=shared.now()))
        for handle in handles.values():
            handle.close()
    return state


def worker(path, mode):
    path, cfg = load_runtime(path); session = session_path(cfg, mode); handles = []
    try:
        proof = attachment(path, cfg, session)
        if proof["worker"]["process_id"] != os.getpid() or proof["worker"]["creation_time_ticks"] != shared.creation_ticks(os.getpid()):
            raise ValueError("Worker creation identity differs")
        handles = [WindowsProcess(proof[r]) for r in ["owner", "guard", "watchdog"]]
        start = time.monotonic()
        while not all((session / (r + "_state.json")).exists() for r in ["guard", "watchdog"]):
            if time.monotonic() - start > cfg["startup_limit_seconds"] or not all(h.is_alive() for h in handles):
                raise RuntimeError("Independent supervisors not ready before CUDA import")
            time.sleep(.2)
        def check():
            for role, handle in zip(["owner", "guard", "watchdog"], handles):
                if not handle.is_alive():
                    raise RuntimeError("Bound supervisor exited: " + role)
                shared.require_fresh(session / (role + "_state.json"), cfg["heartbeat_max_age_seconds"])
        def progress(value):
            atomic(session / "progress.json", dict(value, updated_utc=shared.now()))
        check(); gpu_state(cfg)
        progress({"stage": "independent_supervisors_ready_before_torch_import", "step": 0, "numeric_alert": False})
        verify(read(Path(cfg["study_output"]) / "authorization.json")["bound_sha256"])
        from scripts.tiling_probe_cuda import pilot
        from scripts.levir_tiling_probe import evaluate_prepared
        result = (pilot(cfg["study_path"], session, check=check, progress=progress) if mode == "pilot" else
                  evaluate_prepared(cfg["study_path"], Path(cfg["study_output"]) / "inference", check=check, progress=progress))
        atomic(session / "worker_result.json", {"status": "completed", "mode": mode,
            "result_status": result["status"], "optimizer_updates": 0, "completed_utc": shared.now()})
        return {"status": "completed"}
    except BaseException as error:
        atomic(session / "worker_result.json", {"status": "failed_preserve", "mode": mode,
            "error": f"{type(error).__name__}: {error}", "updated_utc": shared.now(), "automatic_retry": False})
        raise
    finally:
        for handle in handles:
            handle.close()


def monitor(path, mode):
    path, cfg = load_runtime(path); session = session_path(cfg, mode)
    result = {"mode": mode, "checked_utc": shared.now(), "gpu": gpu_state(cfg)}
    for name in ["owner_state", "progress", "guard_state", "watchdog_state", "worker_result"]:
        if (session / (name + ".json")).exists():
            result[name] = read(session / (name + ".json"))
    if (session / "attachment.json").exists():
        proof = read(session / "attachment.json")
        result["processes"] = {r: shared.snapshot_identity(cfg, proof[r]) for r in ["owner", "worker", "guard", "watchdog"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seal", "launch", "owner", "worker", "guard", "watchdog", "monitor"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--checks", type=Path); parser.add_argument("--smoke", type=Path)
    args = parser.parse_args()
    if args.action == "seal":
        result = seal(args.runtime, args.checks, args.smoke)
    elif args.action in ["guard", "watchdog"]:
        result = guard_loop(args.runtime, args.mode, args.action)
    else:
        result = globals()[args.action](args.runtime, args.mode)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
