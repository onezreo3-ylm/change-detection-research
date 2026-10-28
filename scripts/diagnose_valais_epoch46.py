"""One-shot, guarded engineering replay of the failed Valais train window."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import torch
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.data import audit_data, build_datasets, file_sha256, make_loader
from pcdc.engine import _restore_rng, move_model, run_epoch, source_files
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts.run_valais_clean_guarded import creation_time_ticks
from scripts.temperature_guard import load_settings, read_temperature
from scripts.train_valais_fp32val import _read_checkpoint, _verify_snapshot, load_study


ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "schema_version", "kind", "study_config", "study_config_sha256",
    "checkpoint", "checkpoint_sha256", "failure_audit", "failure_audit_sha256",
    "output_dir", "source_epoch", "replay_epoch", "max_train_step",
    "expected_prefix_loss", "expected_overflow", "failure_window", "test_access", "guard",
}
TRACE_STEPS = {1, 275, 550, 825, 1100}
GUARD_KEYS = {
    "gpu_index", "gpu_uuid", "confirmed_pause_c", "immediate_pause_c",
    "poll_seconds", "confirmation_seconds", "confirmation_samples", "max_sensor_failures",
}


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def _write(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_replay(path: Path, *, require_new: bool = True):
    cfg = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(cfg, dict) or set(cfg) != KEYS:
        raise ValueError("Unexpected epoch46 replay keys")
    if (cfg["schema_version"] != 1 or cfg["kind"] != "isolated_train_numeric_replay"
            or (cfg["source_epoch"], cfg["replay_epoch"], cfg["max_train_step"]) != (45, 46, 1375)
            or set(cfg["expected_prefix_loss"]) != TRACE_STEPS
            or cfg["expected_overflow"] != {
                "step": 480, "scale_before": 131072, "scale_after": 65536}
            or cfg["failure_window"] != [1101, 1375]
            or cfg["test_access"] != "forbidden"
            or set(cfg["guard"]) != GUARD_KEYS
            or cfg["guard"]["gpu_index"] != 0
            or (cfg["guard"]["confirmed_pause_c"], cfg["guard"]["immediate_pause_c"]) != (85, 86)):
        raise ValueError("Epoch46 replay protocol differs")
    paths = {key: _resolve(path, cfg[key]) for key in (
        "study_config", "checkpoint", "failure_audit", "output_dir")}
    if paths["output_dir"] != ROOT / "runs" / "valais_train_nonfinite_replay_epoch46_20261002":
        raise ValueError("Replay output must be isolated")
    if require_new and (paths["output_dir"].exists()
                        or paths["output_dir"].with_suffix(".seal.json").exists()):
        raise FileExistsError("Epoch46 replay already started")
    for key in ("study_config", "checkpoint", "failure_audit"):
        if file_sha256(paths[key]) != cfg[key + "_sha256"]:
            raise RuntimeError(f"Bound epoch46 input changed: {key}")
    return cfg, paths


def preflight(path: Path, *, require_new: bool = True):
    cfg, paths = load_replay(path, require_new=require_new)
    spec, study_paths, base, fingerprint = load_study(paths["study_config"])
    run = study_paths["output_dir"]
    if paths["checkpoint"] != run / "last.pt" or base.run_test_after_training:
        raise RuntimeError("Replay source is not the failed val-only run")
    audit = json.loads(paths["failure_audit"].read_text(encoding="utf-8-sig"))
    if (audit.get("status") != "training_nonfinite_failure_preserved"
            or audit.get("last_complete_epoch") != 45
            or audit.get("failing_epoch") != 46
            or audit.get("test_accessed") is not False
            or audit.get("artifacts_sha256", {}).get("last_checkpoint") != cfg["checkpoint_sha256"]):
        raise RuntimeError("Failure audit differs")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if (status.get("status") != "failed"
            or status.get("error") != "FloatingPointError: Non-finite loss; experiment stopped"
            or status.get("test_evaluated") is not False):
        raise RuntimeError("Research run is not the frozen failed state")
    for name, relative in {
        "status": run / "status.json",
        "history": run / "history.json",
        "best_checkpoint": run / "best.pt",
        "train_log": run.with_suffix(".session_0002.train.log"),
        "guard_state": run / "temperature_guard_sessions" / "session_0002" / "temperature_guard_state.json",
    }.items():
        if file_sha256(relative) != audit["artifacts_sha256"][name]:
            raise RuntimeError(f"Original failure artifact changed: {name}")
    log = run.with_suffix(".session_0002.train.log").read_text(encoding="utf-8-sig")
    for step, expected in cfg["expected_prefix_loss"].items():
        if f"epoch=46/train step={step}/2757 loss={expected:.6f}" not in log:
            raise RuntimeError("Frozen training prefix does not match the failed log")
    if ("epoch=46/train step=480/2757 amp_overflow_skip=1 scale=131072->65536"
            not in log):
        raise RuntimeError("Frozen AMP overflow differs")
    checkpoint = _read_checkpoint(paths["checkpoint"])
    if (checkpoint.get("epoch") != 45 or len(checkpoint.get("history", [])) != 45
            or checkpoint.get("study_fingerprint") != fingerprint
            or checkpoint.get("validation_amp") is not False
            or checkpoint.get("test_evaluated") is not False
            or not all(torch.isfinite(value).all().item() for value in checkpoint["model"].values()
                       if torch.is_tensor(value))):
        raise RuntimeError("Epoch45 checkpoint identity or tensors differ")
    snapshot = json.loads((run / "source_sha256.json").read_text(encoding="utf-8"))
    _verify_snapshot(run, snapshot)
    return cfg, paths, base, checkpoint, run, fingerprint


def tensor_counts(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    good = tensor[finite].float()
    return {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "nan": int(torch.isnan(tensor).sum().item()),
        "posinf": int(torch.isposinf(tensor).sum().item()),
        "neginf": int(torch.isneginf(tensor).sum().item()),
        "finite": int(finite.sum().item()),
        "finite_min": float(good.min().item()) if good.numel() else None,
        "finite_max": float(good.max().item()) if good.numel() else None,
    }


def safe_scalar(value: torch.Tensor) -> float | str:
    number = float(value.detach().float().item())
    return number if math.isfinite(number) else str(number)


class WindowLoader:
    def __init__(self, loader, model, max_step: int):
        self.loader, self.model, self.max_step = loader, model, max_step
        self.step = 0
        self.batch = None
        self.first_bad_module = None
        self.hooks = []

    def __len__(self):
        return len(self.loader)

    def _hook(self, name: str, module, inputs, output):
        if self.first_bad_module is not None:
            return
        values = [output] if torch.is_tensor(output) else (
            list(output) if isinstance(output, (tuple, list)) else [])
        for value in values:
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                self.first_bad_module = {
                    "name": name, "type": type(module).__name__,
                    "output": tensor_counts(value),
                }
                break

    def __iter__(self):
        for step, batch in enumerate(self.loader, 1):
            if step > self.max_step:
                break
            self.step, self.batch = step, batch
            if step == 1101:
                for name, module in self.model.named_modules():
                    if name and not any(module.children()):
                        self.hooks.append(module.register_forward_hook(
                            lambda mod, inp, out, name=name: self._hook(name, mod, inp, out)))
            yield batch

    def close(self):
        for hook in self.hooks:
            hook.remove()


class DiagnosticLoss(ChangeDetectionLoss):
    def __init__(self, base, model, loader: WindowLoader, expected: dict, memory: str):
        super().__init__(**base.values["loss"])
        self.base, self.model, self.loader = base, model, loader
        self.expected, self.memory = expected, memory
        self.loss_sum = 0.0
        self.samples = 0
        self.prefix = {}
        self.first_nonfinite = None

    def forward(self, logits, target):
        losses = super().forward(logits, target)
        step = self.loader.step
        if all(torch.isfinite(value).item() for value in losses.values()):
            count = target.shape[0]
            self.loss_sum += losses["total"].detach().item() * count
            self.samples += count
            if step in self.expected:
                observed = f"{self.loss_sum / self.samples:.6f}"
                expected = f"{self.expected[step]:.6f}"
                self.prefix[str(step)] = {"observed": observed, "expected": expected}
                if observed != expected:
                    raise RuntimeError(f"Trajectory mismatch at step {step}: {observed} != {expected}")
        else:
            batch = self.loader.batch
            layout = torch.channels_last if self.memory == "channels_last" else torch.contiguous_format
            a = batch["image_a"].to(device=logits.device, non_blocking=True, memory_format=layout)
            b = batch["image_b"].to(device=logits.device, non_blocking=True, memory_format=layout)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                fp32_logits = self.model(a, b)["logits"]
                fp32_losses = super().forward(fp32_logits, target)
            self.first_nonfinite = {
                "step": step,
                "sample_ids": list(batch["sample_id"]),
                "input_a_finite": bool(torch.isfinite(a).all().item()),
                "input_b_finite": bool(torch.isfinite(b).all().item()),
                "target_finite": bool(torch.isfinite(target).all().item()),
                "amp_logits": tensor_counts(logits),
                "amp_losses": {key: safe_scalar(value) for key, value in losses.items()},
                "first_visible_bad_module": self.loader.first_bad_module,
                "fp32_logits": tensor_counts(fp32_logits),
                "fp32_losses": {key: safe_scalar(value) for key, value in fp32_losses.items()},
            }
        return losses


def _check_seal(path: Path, cfg_path: Path):
    seal = json.loads(path.read_text(encoding="utf-8"))
    if (seal.get("status") != "isolated_numeric_replay_sealed"
            or seal.get("spec_sha256") != file_sha256(cfg_path)
            or seal.get("test_accessed") is not False):
        raise RuntimeError("Replay preflight seal differs")
    for name, expected in seal["bound_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise RuntimeError(f"Replay-bound file changed: {name}")
    return seal


def work(cfg_path: Path, seal_path: Path):
    seal = _check_seal(seal_path, cfg_path)
    cfg, paths, base, checkpoint, run, fingerprint = preflight(cfg_path, require_new=False)
    device = setup_runtime(base)
    live_audit = audit_data(base, full=True)
    archived_audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    if live_audit != archived_audit:
        raise RuntimeError("Live train/val bytes differ from frozen audit")
    datasets = build_datasets(base, ("train", "val"))
    loaders = {split: make_loader(ds, base, training=split == "train")
               for split, ds in datasets.items()}
    memory = base.values["runtime"]["memory_format"]
    model = move_model(build_model(base, initialize_encoder=False), device, memory)
    model.load_state_dict(checkpoint["model"], strict=True)
    settings = base.values["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  betas=tuple(settings["adam_betas"]),
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng(checkpoint["rng"], loaders["train"])
    window = WindowLoader(loaders["train"], model, cfg["max_train_step"])
    loss_fn = DiagnosticLoss(base, model, window, cfg["expected_prefix_loss"], memory)
    output = paths["output_dir"]
    output.mkdir(exist_ok=False)
    report = {
        "status": "ready_for_guard", "kind": "engineering_replay_only",
        "research_result": False, "test_accessed": False,
        "source_epoch": 45, "replay_epoch": 46,
        "spec_sha256": file_sha256(cfg_path), "seal_sha256": file_sha256(seal_path),
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "failure_audit_sha256": cfg["failure_audit_sha256"],
        "study_fingerprint": fingerprint,
        "environment": environment_record(base, device),
    }
    _write(output / "replay.json", report)
    deadline = time.monotonic() + 90
    while not (output / "guard_attached.json").is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("Replay guard did not attach")
        time.sleep(1)
    handshake = json.loads((output / "guard_attached.json").read_text(encoding="utf-8"))
    if handshake != {"status": "guard_attached", "spec_sha256": report["spec_sha256"]}:
        raise RuntimeError("Replay guard handshake differs")
    try:
        report["status"] = "running_engineering_replay"
        _write(output / "replay.json", report)
        with (output / "trace.log").open("x", encoding="utf-8") as trace, redirect_stdout(trace):
            try:
                run_epoch(model, window, loss_fn, device,
                          threshold=base.values["evaluation"]["threshold"],
                          amp=True, memory_format=memory, optimizer=optimizer,
                          scaler=scaler, grad_clip_norm=settings["grad_clip_norm"],
                          progress="epoch=46/train")
            except FloatingPointError as error:
                if str(error) != "Non-finite loss; experiment stopped":
                    raise
        trace_text = (output / "trace.log").read_text(encoding="utf-8")
        overflow = re.findall(
            r"epoch=46/train step=(\d+)/2757 amp_overflow_skip=1 scale=(\d+)->(\d+)",
            trace_text)
        report["prefix_loss"] = loss_fn.prefix
        report["observed_overflow"] = [
            {"step": int(step), "scale_before": int(before), "scale_after": int(after)}
            for step, before, after in overflow]
        report["first_nonfinite"] = loss_fn.first_nonfinite
        report["status"] = (
            "reproduced_train_nonfinite"
            if loss_fn.first_nonfinite is not None
            and cfg["failure_window"][0] <= loss_fn.first_nonfinite["step"] <= cfg["failure_window"][1]
            and report["observed_overflow"] == [cfg["expected_overflow"]]
            and len(loss_fn.prefix) == len(cfg["expected_prefix_loss"])
            else "trajectory_or_failure_not_reproduced")
        _write(output / "replay.json", report)
        return report
    except BaseException as error:
        report["status"] = "engineering_replay_failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["prefix_loss"] = loss_fn.prefix
        report["first_nonfinite"] = loss_fn.first_nonfinite
        _write(output / "replay.json", report)
        raise
    finally:
        window.close()


def _observed_command(pid: int, powershell: Path) -> str:
    command = f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; $p.CommandLine | ConvertTo-Json -Compress"
    response = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", command],
                              capture_output=True, text=True, check=True, timeout=20,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    value = json.loads(response.stdout)
    if not isinstance(value, str) or "-m scripts.diagnose_valais_epoch46" not in value:
        raise RuntimeError("Unexpected replay worker process")
    return value


def launch(cfg_path: Path):
    if os.name != "nt":
        raise RuntimeError("Windows guarded replay only")
    cfg, paths, base, checkpoint, run, fingerprint = preflight(cfg_path)
    output = paths["output_dir"]
    powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    nvidia = Path("C:/Windows/System32/nvidia-smi.exe")
    query = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
        "-and $_.CommandLine -match 'scripts.train_valais_fp32val|scripts.guard_valais_fp32val' } "
        "| Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress"
    )
    active = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", query],
                            capture_output=True, text=True, check=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    if active and active != "null":
        raise RuntimeError("Valais research training or guard is still active")
    gpu = subprocess.run([str(nvidia), "-i", "0", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=10,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    if gpu != cfg["guard"]["gpu_uuid"] or read_temperature({"nvidia_smi": str(nvidia), "gpu_index": 0}) >= 85:
        raise RuntimeError("GPU UUID or temperature differs from replay policy")
    bound = [cfg_path, paths["study_config"], paths["checkpoint"], paths["failure_audit"],
             run / "status.json", run / "history.json", run / "data_audit.json",
             run / "source_sha256.json", run.with_suffix(".session_0002.train.log"),
             ROOT / "scripts" / "diagnose_valais_epoch46.py",
             ROOT / "scripts" / "guard_valais_numeric_replay.py",
             ROOT / "docs" / "VALAIS_TRAIN_NONFINITE_REPLAY_PROTOCOL_2026-10-02.md",
             ROOT / "tests" / "test_valais_epoch46_replay.py", *source_files().values()]
    seal = {"status": "isolated_numeric_replay_sealed", "spec_sha256": file_sha256(cfg_path),
            "test_accessed": False, "gpu_uuid": gpu,
            "bound_sha256": {str(path.resolve()): file_sha256(path) for path in bound}}
    seal_path = output.with_suffix(".seal.json")
    _write(seal_path, seal)
    log_path = output.with_suffix(".stdout.log")
    guard_log = output.with_suffix(".guard.log")
    with log_path.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, "-m", "scripts.diagnose_valais_epoch46", "--config",
                                  str(cfg_path), "--seal", str(seal_path), "--worker"],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    print(f"replay_pid={child.pid}", flush=True)
    guard = None
    try:
        deadline = time.monotonic() + 600
        while not (output / "replay.json").is_file():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Replay worker exited or timed out before guard attach")
            time.sleep(2)
        guard_settings = {
            "run_dir": str(output), "process_id": child.pid,
            "creation_time_ticks": creation_time_ticks(child.pid),
            "executable": str(Path(sys.executable).resolve()),
            "command_line": _observed_command(child.pid, powershell),
            "powershell": str(powershell), "nvidia_smi": str(nvidia), "gpu_index": 0,
            **{key: value for key, value in cfg["guard"].items() if key not in ("gpu_index", "gpu_uuid")},
        }
        guard_config = output / "temperature_guard.yaml"
        with guard_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(guard_settings, stream, sort_keys=False)
        load_settings(guard_config)
        with guard_log.open("x", encoding="utf-8") as stream:
            guard = subprocess.Popen([sys.executable, "-m", "scripts.guard_valais_numeric_replay",
                                      "--config", str(guard_config)], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        marker_path = output / "guard_attached.marker.json"
        deadline = time.monotonic() + 60
        while not marker_path.is_file():
            if child.poll() is not None or guard.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Replay guard failed before attachment")
            time.sleep(1)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("status") != "bound_guard_attached"
                or marker.get("process_id") != child.pid
                or marker.get("creation_time_ticks") != guard_settings["creation_time_ticks"]
                or marker.get("guard_config_sha256") != file_sha256(guard_config)):
            raise RuntimeError("Replay guard marker differs")
        _write(output / "guard_attached.json", {"status": "guard_attached",
                                               "spec_sha256": seal["spec_sha256"]})
        while child.poll() is None:
            if guard.poll() is not None:
                raise RuntimeError("Replay guard exited while worker active")
            time.sleep(2)
        if child.returncode or guard.wait(timeout=45):
            raise RuntimeError(f"Replay/guard exited with failure: {child.returncode}")
        state = json.loads((output / "temperature_guard_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "training_process_exited":
            raise RuntimeError("Replay guard did not complete")
        _check_seal(seal_path, cfg_path)
        report = json.loads((output / "replay.json").read_text(encoding="utf-8"))
        result = {"status": "guarded_engineering_replay_complete", "replay_status": report["status"],
                  "test_accessed": False, "original_run_modified": False,
                  "seal_sha256": file_sha256(seal_path),
                  "replay_sha256": file_sha256(output / "replay.json"),
                  "guard_state_sha256": file_sha256(output / "temperature_guard_state.json"),
                  "trace_sha256": file_sha256(output / "trace.log")}
        write_json(output / "completion.json", result)
        print(json.dumps(result))
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
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    if args.worker:
        if args.seal is None:
            raise ValueError("Worker requires seal")
        print(json.dumps({"status": work(cfg_path, args.seal.resolve())["status"]}))
    elif args.preflight_only:
        cfg, paths, base, checkpoint, run, fingerprint = preflight(cfg_path)
        print(json.dumps({"status": "preflight_passed", "checkpoint_epoch": checkpoint["epoch"],
                          "test_accessed": False}))
    else:
        launch(cfg_path)


if __name__ == "__main__":
    main()
