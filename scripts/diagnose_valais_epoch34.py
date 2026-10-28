"""Isolated one-epoch engineering replay; never resumes the failed research run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from contextlib import redirect_stdout
from typing import Any

import torch
import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.data import audit_data, build_datasets, file_sha256, make_loader
from pcdc.engine import (
    _digest,
    _restore_rng,
    check_checkpoint,
    move_model,
    read_checkpoint,
    run_epoch,
    source_hashes,
    training_signature,
)
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime


_KEYS = {
    "schema_version", "kind", "source_config", "source_config_sha256",
    "checkpoint", "checkpoint_sha256", "failure_audit", "failure_audit_sha256",
    "source_data_audit", "source_data_audit_sha256", "output_dir", "replay_epoch",
    "expected_train_loss_rounded_6", "expected_train_amp_skips",
    "expected_train_optimizer_steps", "expected_train_trace", "expected_amp_skip",
    "expected_val_step35_loss_rounded_6", "expected_val_trace",
    "first_possible_failure_step", "last_possible_failure_step", "guard",
}


def _path(source: Path, value: str) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else source.parent / candidate).resolve()


def _hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"Bound file missing or changed: {path}")


def verify_seal(seal_path: Path, spec_path: Path) -> dict[str, Any]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("status") != "engineering_replay_preflight_sealed"
            or seal.get("spec_sha256") != hashlib.sha256(spec_path.read_bytes()).hexdigest()
            or seal.get("test_accessed") is not False):
        raise RuntimeError("Replay preflight seal does not match the spec")
    bindings = seal.get("bound_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Replay preflight seal has no bound files")
    for name, expected in bindings.items():
        _hash(Path(name), expected)
    return seal


def verify_original_failure_artifacts(failure: dict[str, Any], workspace: Path) -> None:
    run = workspace / "runs" / "valais_clean_development_seed42"
    prefix = workspace / "runs" / "valais_clean_development_seed42.resume_20260930_epoch6"
    paths = {
        "status": run / "status.json",
        "last_checkpoint": run / "last.pt",
        "best_checkpoint": run / "best.pt",
        "history": run / "history.json",
        "data_audit": run / "data_audit.json",
        "temperature_guard_state": run / "temperature_guard_state.json",
        "resume_train_log": Path(str(prefix) + ".train.log"),
        "resume_guard_log": Path(str(prefix) + ".guard.log"),
        "resume_launcher_log": Path(str(prefix) + ".launcher.log"),
        "resume_launcher_stderr": Path(str(prefix) + ".launcher_stderr.log"),
        "frozen_config": workspace / "configs" / "valais_clean_development_seed42.yaml",
        "frozen_engine_source": workspace / "pcdc" / "engine.py",
        "frozen_loss_source": workspace / "pcdc" / "losses.py",
    }
    expected = failure.get("artifacts_sha256")
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise RuntimeError("Incomplete original numeric-failure artifact bindings")
    for name, path in paths.items():
        _hash(path, expected[name])


def load_replay_spec(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    raw = path.read_bytes()
    spec = yaml.load(raw, Loader=UniqueKeyLoader)
    if not isinstance(spec, dict) or set(spec) != _KEYS:
        raise ValueError("Unexpected replay spec keys")
    if spec["schema_version"] != 1 or spec["kind"] != "engineering_numeric_replay_only":
        raise ValueError("Wrong replay schema or kind")
    if (spec["replay_epoch"], spec["first_possible_failure_step"],
            spec["last_possible_failure_step"]) != (34, 36, 40):
        raise ValueError("Replay is restricted to epoch 34 validation window")
    guard = spec["guard"]
    if (not isinstance(guard, dict)
            or set(guard) != {"gpu_index", "gpu_uuid", "confirmed_pause_c",
                              "immediate_pause_c", "poll_seconds", "confirmation_seconds",
                              "confirmation_samples", "max_sensor_failures"}
            or guard["gpu_index"] != 0
            or not isinstance(guard["gpu_uuid"], str)
            or not guard["gpu_uuid"].startswith("GPU-")
            or (guard["confirmed_pause_c"], guard["immediate_pause_c"]) != (85, 86)):
        raise ValueError("Unexpected bound GPU or temperature policy")
    if (not isinstance(spec["expected_train_trace"], dict)
            or set(spec["expected_train_trace"]) !=
            {1, 275, 550, 825, 1100, 1375, 1650, 1925, 2200, 2475, 2750, 2757}
            or not isinstance(spec["expected_val_trace"], dict)
            or set(spec["expected_val_trace"]) != {1, 5, 10, 15, 20, 25, 30, 35}
            or spec["expected_amp_skip"] !=
            {"step": 1836, "scale_before": 131072, "scale_after": 65536}):
        raise ValueError("Unexpected frozen training/validation trace")
    if (f"{spec['expected_train_trace'][2757]:.6f}" !=
            f"{spec['expected_train_loss_rounded_6']:.6f}" or
            f"{spec['expected_val_trace'][35]:.6f}" !=
            f"{spec['expected_val_step35_loss_rounded_6']:.6f}"):
        raise ValueError("Final trace values disagree with replay summary gates")
    paths = {
        key: _path(path, spec[key])
        for key in ("source_config", "checkpoint", "failure_audit", "source_data_audit",
                    "output_dir")
    }
    if paths["output_dir"].exists():
        raise FileExistsError(paths["output_dir"])
    workspace = Path(__file__).resolve().parents[1]
    if paths["output_dir"] != (workspace / "runs" / "valais_numeric_replay_epoch34_20260930"):
        raise ValueError("Replay output must be the independently named workspace run")
    for key in ("source_config", "checkpoint", "failure_audit", "source_data_audit"):
        _hash(paths[key], spec[key + "_sha256"])
    return spec, paths


def preflight(path: Path):
    spec, paths = load_replay_spec(path)
    config = load_config(paths["source_config"])
    if config.run_test_after_training or config.values["runtime"]["device"] != "cuda":
        raise RuntimeError("Replay requires original CUDA development-only protocol")
    checkpoint = read_checkpoint(paths["checkpoint"])
    check_checkpoint(config, checkpoint)
    if checkpoint["epoch"] != 33 or checkpoint["kind"] != "research":
        raise RuntimeError("Expected research checkpoint from complete epoch 33")
    if checkpoint["training_signature"] != training_signature(config):
        raise RuntimeError("Training signature changed")
    if checkpoint["source_sha256"] != source_hashes():
        raise RuntimeError("Frozen pcdc source changed")
    failure = json.loads(paths["failure_audit"].read_text(encoding="utf-8"))
    if (failure.get("status") != "failed_numeric_no_automatic_resume"
            or failure.get("last_complete_epoch") != 33
            or failure.get("test_evaluated") is not False):
        raise RuntimeError("Failure audit does not match the replay boundary")
    hashes = failure.get("artifacts_sha256", {})
    if (hashes.get("last_checkpoint") != spec["checkpoint_sha256"]
            or hashes.get("data_audit") != spec["source_data_audit_sha256"]
            or hashes.get("frozen_config") != spec["source_config_sha256"]):
        raise RuntimeError("Replay source identities differ from failure audit")
    verify_original_failure_artifacts(failure, Path(__file__).resolve().parents[1])
    archived_audit = json.loads(paths["source_data_audit"].read_text(encoding="utf-8"))
    if _digest(archived_audit) != checkpoint["audit_sha256"]:
        raise RuntimeError("Original data audit differs from checkpoint")
    return spec, paths, config, checkpoint


def _finite_value(value: torch.Tensor) -> float | str:
    number = float(value.detach().float().item())
    return number if math.isfinite(number) else str(number)


def _logit_diagnostic(logits: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(logits)
    valid = logits[finite].float()
    return {
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(logits).sum().item()),
        "posinf_count": int(torch.isposinf(logits).sum().item()),
        "neginf_count": int(torch.isneginf(logits).sum().item()),
        "finite_min": float(valid.min().item()) if valid.numel() else None,
        "finite_max": float(valid.max().item()) if valid.numel() else None,
    }


def _batch_to_device(batch: dict, device: torch.device, memory_format: str):
    layout = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
    a = batch["image_a"].to(device=device, non_blocking=True, memory_format=layout)
    b = batch["image_b"].to(device=device, non_blocking=True, memory_format=layout)
    target = batch["mask"].to(device, non_blocking=True)
    return a, b, target


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_training_trace(contents: str) -> tuple[dict[int, str], list[dict[str, int]]]:
    losses: dict[int, str] = {}
    skips: list[dict[str, int]] = []
    for line in contents.splitlines():
        match = re.fullmatch(
            r"replay_epoch34/train step=(\d+)/2757 loss=(\d+\.\d{6})", line
        )
        if match:
            step = int(match.group(1))
            if step in losses:
                raise RuntimeError("Duplicate replay train trace step")
            losses[step] = match.group(2)
            continue
        match = re.fullmatch(
            r"replay_epoch34/train step=(\d+)/2757 amp_overflow_skip=1 "
            r"scale=([0-9.e+]+)->([0-9.e+]+)", line
        )
        if match:
            skips.append({
                "step": int(match.group(1)),
                "scale_before": int(float(match.group(2))),
                "scale_after": int(float(match.group(3))),
            })
    return losses, skips


def run(spec_path: Path, seal_path: Path) -> dict[str, Any]:
    seal = verify_seal(seal_path, spec_path)
    spec, paths, config, checkpoint = preflight(spec_path)
    device = setup_runtime(config)
    # Recompute decoded-label/file audit before touching the GPU model.
    live_audit = audit_data(config, full=True)
    if _digest(live_audit) != checkpoint["audit_sha256"]:
        raise RuntimeError("Live train/val data differs from the checkpoint")
    datasets = build_datasets(config, ("train", "val"))
    loaders = {split: make_loader(dataset, config, training=split == "train")
               for split, dataset in datasets.items()}
    memory = config.values["runtime"]["memory_format"]
    model = move_model(build_model(config, initialize_encoder=False), device, memory)
    model.load_state_dict(checkpoint["model"], strict=True)
    train_settings = config.values["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_settings["learning_rate"],
        betas=tuple(train_settings["adam_betas"]),
        weight_decay=train_settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_settings["epochs"],
        eta_min=train_settings["min_learning_rate"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_settings["amp"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng(checkpoint["rng"], loaders["train"])
    loss_fn = ChangeDetectionLoss(**config.values["loss"])

    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running_engineering_replay",
        "research_result": False,
        "original_run_modified": False,
        "test_accessed": False,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "preflight_seal_sha256": file_sha256(seal_path),
        "source_config_sha256": spec["source_config_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "failure_audit_sha256": spec["failure_audit_sha256"],
        "environment": environment_record(config, device),
    }
    _write(output / "replay.json", report)
    try:
        ready = output / "guard_attached.json"
        deadline = time.monotonic() + 90
        while not ready.is_file():
            if time.monotonic() > deadline:
                raise RuntimeError("Bound temperature guard did not attach")
            time.sleep(1)
        handshake = json.loads(ready.read_text(encoding="utf-8"))
        if handshake != {"status": "guard_attached", "spec_sha256": report["spec_sha256"]}:
            raise RuntimeError("Temperature guard handshake differs from replay spec")
        trace_path = output / "train_trace.log"
        with trace_path.open("x", encoding="utf-8") as trace, redirect_stdout(
            _Tee(sys.stdout, trace)
        ):
            training = run_epoch(
                model, loaders["train"], loss_fn, device,
                threshold=config.values["evaluation"]["threshold"],
                amp=train_settings["amp"], memory_format=memory,
                optimizer=optimizer, scaler=scaler,
                grad_clip_norm=train_settings["grad_clip_norm"],
                progress="replay_epoch34/train",
            )
        observed_losses, observed_skips = parse_training_trace(
            trace_path.read_text(encoding="utf-8")
        )
        expected_losses = {
            step: f"{value:.6f}" for step, value in spec["expected_train_trace"].items()
        }
        report["training"] = {
            "loss_total": training["loss"]["total"],
            "loss_rounded_6": f"{training['loss']['total']:.6f}",
            "optimizer_steps": training["optimizer_steps"],
            "amp_skipped_steps": training["amp_skipped_steps"],
            "loss_match": f"{training['loss']['total']:.6f}" ==
                          f"{spec['expected_train_loss_rounded_6']:.6f}",
            "optimizer_steps_match": training["optimizer_steps"] ==
                                     spec["expected_train_optimizer_steps"],
            "amp_skips_match": training["amp_skipped_steps"] ==
                               spec["expected_train_amp_skips"],
            "trace_match": observed_losses == expected_losses,
            "observed_trace": observed_losses,
            "expected_trace": expected_losses,
            "overflow_location_match": observed_skips == [spec["expected_amp_skip"]],
            "observed_overflow": observed_skips,
        }
        _write(output / "replay.json", report)
        if not all(report["training"][key] for key in
                   ("loss_match", "optimizer_steps_match", "amp_skips_match",
                    "trace_match", "overflow_location_match")):
            report["status"] = "replay_trajectory_mismatch_stop"
            _write(output / "replay.json", report)
            return report

        model.eval()
        total_loss = 0.0
        total_samples = 0
        report["validation_trace"] = {}
        for step, batch in enumerate(loaders["val"], 1):
            a, b, target = _batch_to_device(batch, device, memory)
            with torch.set_grad_enabled(False), torch.autocast(device_type="cuda", enabled=True):
                logits = model(a, b)["logits"]
                losses = loss_fn(logits, target)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                failure = {
                    "step": step,
                    "sample_ids": list(batch["sample_id"]),
                    "amp_logits": _logit_diagnostic(logits),
                    "amp_losses": {key: _finite_value(value) for key, value in losses.items()},
                }
                # Same frozen model and batch; diagnostic FP32 does not update any state.
                with torch.set_grad_enabled(False), torch.autocast(device_type="cuda", enabled=False):
                    fp32_logits = model(a, b)["logits"]
                    fp32_losses = loss_fn(fp32_logits, target)
                failure["fp32_logits"] = _logit_diagnostic(fp32_logits)
                failure["fp32_losses"] = {
                    key: _finite_value(value) for key, value in fp32_losses.items()
                }
                report["first_nonfinite"] = failure
                report["status"] = (
                    "reproduced_within_bound_window"
                    if spec["first_possible_failure_step"] <= step <=
                    spec["last_possible_failure_step"]
                    else "replay_step_mismatch_stop"
                )
                _write(output / "replay.json", report)
                return report
            total_samples += target.shape[0]
            total_loss += float(losses["total"].item()) * target.shape[0]
            if step in spec["expected_val_trace"]:
                mean = total_loss / total_samples
                entry = {
                    "mean_loss": mean,
                    "rounded_6": f"{mean:.6f}",
                    "match": f"{mean:.6f}" ==
                             f"{spec['expected_val_trace'][step]:.6f}",
                }
                report["validation_trace"][str(step)] = entry
                _write(output / "replay.json", report)
                if not entry["match"]:
                    report["status"] = "replay_validation_mismatch_stop"
                    _write(output / "replay.json", report)
                    return report
            if step == spec["last_possible_failure_step"]:
                report["status"] = "failure_not_reproduced_by_step40"
                _write(output / "replay.json", report)
                return report
        report["status"] = "failure_not_reproduced"
        _write(output / "replay.json", report)
        return report
    except BaseException as error:
        report["status"] = "engineering_replay_failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _write(output / "replay.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.preflight_only:
        spec, paths, config, checkpoint = preflight(path)
        print(json.dumps({"status": "preflight_passed", "epoch": checkpoint["epoch"],
                          "output": str(paths["output_dir"]), "test_accessed": False}))
    else:
        if args.seal is None:
            raise ValueError("Executing replay requires a bound preflight seal")
        result = run(path, args.seal.resolve())
        print(json.dumps({"status": result["status"], "test_accessed": False}))


if __name__ == "__main__":
    main()
