"""CPU owner and independent guards for the matched native-window training study."""
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
MODULE = "scripts.native_window_supervision"


def load_runtime(path):
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
    study=(path.parent/cfg["study_config"]).resolve()
    if study!=ROOT/"configs/levir_native_window_training_20261025.yaml":
        raise ValueError("Only the frozen native-window study is admitted")
    values=yaml.safe_load(study.read_text(encoding="utf-8"))
    policy={"schema_version":1,"poll_seconds":2,"confirmation_seconds":1,"confirmed_pause_c":85,
        "immediate_pause_c":86,"confirmation_samples":3,"max_sensor_failures":3,
        "heartbeat_max_age_seconds":30,"sensor_timeout_seconds":1,"startup_limit_seconds":120,
        "progress_stall_seconds":900,"pilot_wall_limit_seconds":1800,"training_wall_limit_seconds":86400,
        "whole_study_limit_seconds":129600,"max_consecutive_amp_skips":8,"minimum_amp_scale":1.0,
        "automatic_retry":False,"gpu_index":0,"gpu_uuid":"GPU-8ada1895-7628-bb88-1dda-c57679117fa2",
        "python":"E:/Download/anaconda3/envs/geosa-basa/python.exe"}
    if any(cfg.get(k)!=v for k,v in policy.items()):raise ValueError("Frozen supervision policy changed")
    if values.get("test_or_martigny_access") is not False or values.get("arms")!=["native512","grouped256"]:
        raise ValueError("Frozen development-only arms changed")
    for key in ("python","powershell","nvidia_smi"):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    cfg.update(study_path=str(study),study_output=str(ROOT/values["output_dir"]),
        ledger_path=str(ROOT/values["shared_ledger"]),global_lease=str(ROOT/values["managed_run_dir"]/"gpu_study.lock"))
    return path,cfg


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


def session_path(cfg,mode):
    if mode not in ("pilot","native512","grouped256"):raise ValueError("No unplanned mode")
    return Path(cfg["study_output"])/(mode+"_session")


def reserve(cfg,mode):
    session_path(cfg,mode)
    ledger=read(cfg["ledger_path"])
    if any(e.get("status")=="running" for e in ledger["entries"]):raise RuntimeError("Live/unreconciled compute reservation")
    entry_id="native_window_"+mode+"_20261025"
    if any(e.get("id")==entry_id for e in ledger["entries"]):raise RuntimeError("No retries or budget reset")
    spent=sum(e["charged_seconds"] for e in ledger["entries"] if e.get("kind","").startswith("native_window_"))
    same_arm_spent=sum(e["charged_seconds"] for e in ledger["entries"] if e.get("kind","").startswith("native_window_")
        and e.get("mode") in (None,"pilot",mode))
    limit=min(cfg["pilot_wall_limit_seconds"] if mode=="pilot" else cfg["training_wall_limit_seconds"]-same_arm_spent,
        cfg["whole_study_limit_seconds"]-spent,ledger["batch_upper_bound_seconds"]-ledger["total_charged_seconds"])
    if limit<=0:raise RuntimeError("Original shared/study budget exhausted")
    ledger["entries"].append({"id":entry_id,"kind":"native_window_gpu_"+mode,"mode":mode,"status":"running",
        "charged_seconds":0.,"started_utc":shared.now(),"maximum_seconds":limit,"research_training":mode!="pilot",
        "basis":"Conservative complete owner wall including CPU/startup/I/O and guard cleanup; original shared ledger."})
    atomic(cfg["ledger_path"],ledger);return entry_id,limit


def seal(path,checks_path,smoke_path):
    path,cfg=load_runtime(path);out=Path(cfg["study_output"])
    checks,smoke=read(checks_path),read(smoke_path)
    if not checks.get("passed") or checks.get("full_unittest_count",0)<501 or smoke.get("status")!="passed" or smoke.get("synthetic_only") is not True:
        raise ValueError("Passing complete tests and explicit CPU smoke required")
    prepared=read(out/"prepared.json")
    if prepared.get("status")!="passed_data_core_initialization_and_order_identity":raise ValueError("Preparation not passed")
    verify(prepared["bound_sha256"])
    bound=dict(prepared["bound_sha256"])
    for p in (path,out/"prepared.json",Path(checks_path).resolve(),Path(smoke_path).resolve()):bound[str(p)]=shared.sha(p)
    authority={"status":"sealed_native_window_engineering","created_utc":shared.now(),"gpu_uuid":cfg["gpu_uuid"],
        "bound_sha256":bound,"raw_splits_allowed":["train","val"],"automatic_retry":False}
    target=out/"engineering_authorization.json"
    with target.open("x",encoding="utf-8") as stream:json.dump(authority,stream,ensure_ascii=False,indent=2)
    return {"status":authority["status"],"path":str(target)}


def authorize_train(path):
    path,cfg=load_runtime(path);out=Path(cfg["study_output"]);base=session_path(cfg,"pilot")
    a=read(out/"engineering_authorization.json");verify(a["bound_sha256"])
    report=read(base/"pilot_verified.json")
    if report.get("status")!="passed_engineering_only" or not report.get("repeat_full_state_exact") or not report.get("next_update_full_state_exact") or not report.get("paired_group_prefix_exact"):
        raise ValueError("Successful exact pilot required")
    if read(base/"owner_state.json")["status"]!="completed" or read(base/"owner_state.json")["worker_exit_code"]!=0:
        raise RuntimeError("Pilot owner incomplete")
    if any(read(base/(r+"_state.json"))["status"]!="worker_exited" for r in ("guard","watchdog")):
        raise RuntimeError("Pilot supervisors incomplete")
    ledger=read(cfg["ledger_path"])
    if any(e.get("status")=="running" for e in ledger["entries"]):raise RuntimeError("Active reservation")
    spent=sum(e["charged_seconds"] for e in ledger["entries"] if e.get("kind","").startswith("native_window_"))
    from scripts.native_window_training import budget_estimate
    budget=budget_estimate(report["rows"],ledger["batch_upper_bound_seconds"]-ledger["total_charged_seconds"],
        cfg["whole_study_limit_seconds"]-spent,cfg["training_wall_limit_seconds"]-spent)
    if not budget["passed"]:
        atomic(out/"budget_rejection.json",budget);raise RuntimeError("Frozen throughput/budget gate failed")
    bound=dict(a["bound_sha256"])
    for p in (base/"pilot_verified.json",base/"owner_state.json",base/"guard_state.json",base/"watchdog_state.json"):
        bound[str(p)]=shared.sha(p)
    value={"status":"authorized_native_window_training_under_explicit_user_request","created_utc":shared.now(),
        "user_request":"有提升吗，开启新的训练","gpu_uuid":cfg["gpu_uuid"],"bound_sha256":bound,"budget":budget,
        "arms":["native512","grouped256"],"automatic_retry":False,"held_out_test_open":False}
    target=out/"training_authorization.json"
    with target.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    return {"status":value["status"],"budget":budget}


def ledger_update(cfg,entry,elapsed,status,exit_code=None):
    ledger=read(cfg["ledger_path"]);matches=[e for e in ledger["entries"] if e.get("id")==entry]
    if len(matches)!=1 or elapsed<matches[0]["charged_seconds"]:raise ValueError("Persistent monotone charge required")
    e=matches[0];e.update(charged_seconds=elapsed,status=status,updated_utc=shared.now())
    if exit_code is not None:e["worker_exit_code"]=exit_code
    if e.get("research_training"):
        mode=e["mode"];out=Path(cfg["study_output"]);p=out/mode/"status.json"
        committed=read(p) if p.exists() else {};successful=committed.get("optimizer_steps",0)
        p=session_path(cfg,mode)/"progress.json"
        if p.exists():
            progress=read(p)
            if progress.get("stage")==mode+"_train" and progress.get("epoch",0)>committed.get("epoch",0):
                successful+=progress["step"]-progress.get("amp_skips_in_stage",0)
        e["successful_research_optimizer_updates"]=successful
    ledger["total_charged_seconds"]=sum(e["charged_seconds"] for e in ledger["entries"])
    ledger["new_research_optimizer_updates"]=sum(e.get("successful_research_optimizer_updates",0) for e in ledger["entries"])
    atomic(cfg["ledger_path"],ledger)


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
            raise RuntimeError("Another native-window GPU worker is active")


def launch(path,mode):
    path,cfg=load_runtime(path);out=Path(cfg["study_output"]);session=session_path(cfg,mode)
    if session.exists() or Path(cfg["global_lease"]).exists():raise FileExistsError("Existing session/lease; no duplicates")
    authority=read(out/("engineering_authorization.json" if mode=="pilot" else "training_authorization.json"))
    expected="sealed_native_window_engineering" if mode=="pilot" else "authorized_native_window_training_under_explicit_user_request"
    if authority.get("status")!=expected or authority.get("gpu_uuid")!=cfg["gpu_uuid"]:raise ValueError("Frozen runtime authority required")
    verify(authority["bound_sha256"])
    if mode!="pilot":
        prior=session_path(cfg,"pilot" if mode=="native512" else "native512")
        if read(prior/"owner_state.json")["status"]!="completed" or read(prior/"worker_result.json")["status"]!="completed":
            raise RuntimeError("Previous stage not completed normally")
        if any(read(prior/(r+"_state.json"))["status"]!="worker_exited" for r in ("guard","watchdog")):
            raise RuntimeError("Previous independent supervisors incomplete")
    external_worker_check(cfg);sample=gpu_state(cfg)
    if sample["temperature_c"]>=cfg["confirmed_pause_c"]:raise RuntimeError("Initial thermal stop")
    session.mkdir(exist_ok=False)
    command=[cfg["python"],"-X","utf8","-u","-m",MODULE,"owner","--runtime",str(path),"--mode",mode]
    with (session/"owner.stdout.log").open("x") as a,(session/"owner.stderr.log").open("x") as b:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b,creationflags=shared.HIDDEN)
    return {"status":"hidden_cpu_owner_dispatched","mode":mode,"owner_pid":proc.pid,
        "owner_creation_ticks":shared.creation_ticks(proc.pid),"gpu":sample}


def owner(path, mode):
    path, cfg = load_runtime(path)
    session = session_path(cfg, mode)
    lease = Path(cfg["global_lease"])
    with lease.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "mode": mode, "created_utc": shared.now()}, stream)
    child, guards, entry = None, [], None
    start = time.monotonic()
    state = {"status": "running", "mode": mode, "research_training": mode != "pilot", "updated_utc": shared.now(), "automatic_retry": False}
    try:
        external_worker_check(cfg)
        verify(read(Path(cfg["study_output"]) / ("engineering_authorization.json" if mode == "pilot" else "training_authorization.json"))["bound_sha256"])
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
            runtime_sha256=shared.sha(path), research_training=mode != "pilot", mode=mode)
        atomic(session / "attachment.json", proof)
        while child.poll() is None:
            elapsed = time.monotonic() - start
            atomic(session / "owner_state.json", dict(state, updated_utc=shared.now(), elapsed_seconds=elapsed, allowed_wall_seconds=limit))
            ledger_update(cfg, entry, elapsed, "running")
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
            raise RuntimeError("Native-window worker failed: " + str(child.returncode))
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
            ledger_update(cfg, entry, elapsed, state["status"], state.get("worker_exit_code"))
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


def worker(path,mode):
    path,cfg=load_runtime(path);session=session_path(cfg,mode);handles=[]
    try:
        proof=attachment(path,cfg,session)
        if proof["worker"]["process_id"]!=os.getpid() or proof["worker"]["creation_time_ticks"]!=shared.creation_ticks(os.getpid()):
            raise ValueError("Worker creation identity differs")
        handles=[WindowsProcess(proof[r]) for r in ("owner","guard","watchdog")];start=time.monotonic()
        while not all((session/(r+"_state.json")).exists() for r in ("guard","watchdog")):
            if time.monotonic()-start>cfg["startup_limit_seconds"] or not all(h.is_alive() for h in handles):
                raise RuntimeError("Independent protection not ready before CUDA import")
            time.sleep(.2)
        def check():
            for role,handle in zip(("owner","guard","watchdog"),handles):
                if not handle.is_alive():raise RuntimeError("Bound supervisor exited: "+role)
                shared.require_fresh(session/(role+"_state.json"),cfg["heartbeat_max_age_seconds"])
        check();gpu_state(cfg)
        atomic(session/"progress.json",{"stage":"supervisors_ready_before_torch_import","step":0,"updated_utc":shared.now(),"numeric_alert":False})
        authority=read(Path(cfg["study_output"])/("engineering_authorization.json" if mode=="pilot" else "training_authorization.json"))
        verify(authority["bound_sha256"])
        from scripts import native_window_training as runner
        result=runner.pilot(cfg["study_path"],session,cfg,check) if mode=="pilot" else runner.train(cfg["study_path"],session,cfg,check,mode)
        atomic(session/"worker_result.json",{"status":"completed","mode":mode,"result_status":result["status"],
            "successful_research_optimizer_updates":result.get("optimizer_steps",0),"completed_utc":shared.now(),"test_accessed":False})
        return {"status":"completed"}
    except BaseException as error:
        atomic(session/"worker_result.json",{"status":"failed_preserve","mode":mode,"error":f"{type(error).__name__}: {error}",
            "updated_utc":shared.now(),"automatic_retry":False});raise
    finally:
        for handle in handles:handle.close()


def monitor(path,mode):
    path,cfg=load_runtime(path);session=session_path(cfg,mode)
    result={"mode":mode,"checked_utc":shared.now(),"gpu":gpu_state(cfg)}
    for name in ("owner_state","progress","guard_state","watchdog_state","worker_result"):
        if (session/(name+".json")).exists():result[name]=read(session/(name+".json"))
    if (session/"attachment.json").exists():
        proof=read(session/"attachment.json");result["processes"]={r:shared.snapshot_identity(cfg,proof[r]) for r in ("owner","worker","guard","watchdog")}
    p=Path(cfg["study_output"])/mode/"status.json"
    if p.exists():result["committed_training"]=read(p)
    ledger=read(cfg["ledger_path"]);result["shared_charged_seconds"]=ledger["total_charged_seconds"]
    result["shared_upper_bound_seconds"]=ledger["batch_upper_bound_seconds"]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["seal","authorize","launch","owner","worker","guard","watchdog","monitor"])
    p.add_argument("--runtime",type=Path,required=True);p.add_argument("--mode",choices=["pilot","native512","grouped256"],default="pilot")
    p.add_argument("--checks",type=Path);p.add_argument("--smoke",type=Path);a=p.parse_args()
    if a.action=="seal":result=seal(a.runtime,a.checks,a.smoke)
    elif a.action=="authorize":result=authorize_train(a.runtime)
    elif a.action in ("guard","watchdog"):result=guard_loop(a.runtime,a.mode,a.action)
    else:result=globals()[a.action](a.runtime,a.mode)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
