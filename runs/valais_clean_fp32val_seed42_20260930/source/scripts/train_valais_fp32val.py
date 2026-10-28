"""Independent Valais clean trainer: AMP training, FP32 validation, no test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.data import audit_data, build_datasets, file_sha256, make_loader
from pcdc.engine import (
    _digest,
    _restore_rng,
    _rng_state,
    move_model,
    run_epoch,
    source_files,
    source_hashes,
    training_signature,
)
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import (
    environment_record,
    save_checkpoint,
    setup_runtime,
    write_json,
)


SPEC_KEYS = {
    "schema_version", "kind", "name", "base_config", "base_config_sha256",
    "source_data_audit", "source_data_audit_sha256", "numeric_replay_result",
    "numeric_replay_result_sha256", "output_dir", "training_amp",
    "validation_amp", "seed", "epochs", "selection", "test_access", "guard",
}
GUARD_KEYS = {
    "gpu_index", "gpu_uuid", "confirmed_pause_c", "immediate_pause_c",
    "poll_seconds", "confirmation_seconds", "confirmation_samples",
    "max_sensor_failures",
}


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def load_study(path: Path):
    raw = path.read_bytes()
    spec = yaml.load(raw, Loader=UniqueKeyLoader)
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError("Unexpected independent numeric-study config")
    if (spec["schema_version"] != 1
            or spec["kind"] != "independent_clean_numeric_protocol"
            or spec["training_amp"] is not True
            or spec["validation_amp"] is not False
            or spec["seed"] != 42
            or spec["epochs"] != 100
            or spec["selection"] != "clean_val_f1_at_threshold_0_5"
            or spec["test_access"] != "forbidden"):
        raise ValueError("Unexpected study identity or numeric settings")
    guard = spec["guard"]
    if (not isinstance(guard, dict) or set(guard) != GUARD_KEYS
            or guard["gpu_index"] != 0
            or not isinstance(guard["gpu_uuid"], str)
            or not guard["gpu_uuid"].startswith("GPU-")
            or (guard["confirmed_pause_c"], guard["immediate_pause_c"]) != (85, 86)):
        raise ValueError("Unexpected GPU or temperature guard policy")
    paths = {
        key: _resolve(path, spec[key])
        for key in ("base_config", "source_data_audit", "numeric_replay_result", "output_dir")
    }
    workspace = Path(__file__).resolve().parents[1]
    if paths["output_dir"] != workspace / "runs" / "valais_clean_fp32val_seed42_20260930":
        raise ValueError("Output must use the independently named study directory")
    for key in ("base_config", "source_data_audit", "numeric_replay_result"):
        if not paths[key].is_file() or file_sha256(paths[key]) != spec[key + "_sha256"]:
            raise RuntimeError(f"Bound study input changed: {key}")
    base = load_config(paths["base_config"])
    if (base.run_test_after_training
            or base.values["experiment"]["seed"] != spec["seed"]
            or base.values["train"]["epochs"] != spec["epochs"]
            or base.values["train"]["amp"] is not spec["training_amp"]
            or base.values["runtime"]["device"] != "cuda"
            or base.values["evaluation"]["threshold"] != 0.5):
        raise ValueError("Base protocol differs from frozen new-study assumptions")
    replay = json.loads(paths["numeric_replay_result"].read_text(encoding="utf-8"))
    if (replay.get("status") != "reproduced_within_bound_window"
            or replay.get("test_accessed") is not False
            or replay.get("first_nonfinite", {}).get("step") != 36
            or replay.get("first_nonfinite", {}).get("amp_logits", {}).get("nan_count") != 65536
            or replay.get("first_nonfinite", {}).get("fp32_logits", {}).get("nan_count") != 0):
        raise RuntimeError("Numeric evidence no longer supports the proposed protocol")
    fingerprint = hashlib.sha256(json.dumps({
        "spec_sha256": hashlib.sha256(raw).hexdigest(),
        "base_training_signature": training_signature(base),
        "base_source_sha256": source_hashes(),
        "trainer_sha256": file_sha256(Path(__file__).resolve()),
        "validation_amp": False,
    }, sort_keys=True).encode()).hexdigest()
    return spec, paths, base, fingerprint


def _read_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Not an independent Valais FP32-validation checkpoint")
    return checkpoint


def _wait_guard(handshake_path: Path, fingerprint: str) -> None:
    deadline = time.monotonic() + 90
    while not handshake_path.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("PID-bound temperature guard did not attach")
        time.sleep(1)
    handshake = json.loads(handshake_path.read_text(encoding="utf-8"))
    if handshake != {"status": "guard_attached", "study_fingerprint": fingerprint}:
        raise RuntimeError("Temperature guard handshake does not match study")


def _snapshot(output: Path, spec_path: Path, base, fingerprint: str) -> dict[str, str]:
    workspace = Path(__file__).resolve().parents[1]
    files = {
        **source_files(),
        "scripts/train_valais_fp32val.py": Path(__file__).resolve(),
        "scripts/launch_valais_fp32val.py": workspace / "scripts" / "launch_valais_fp32val.py",
        "scripts/guard_valais_fp32val.py": workspace / "scripts" / "guard_valais_fp32val.py",
        "scripts/smoke_valais_fp32val.py": workspace / "scripts" / "smoke_valais_fp32val.py",
        "scripts/evaluate_valais_fp32val.py": workspace / "scripts" / "evaluate_valais_fp32val.py",
        "docs/VALAIS_CLEAN_FP32_VALIDATION_PROTOCOL_2026-09-30.md":
            workspace / "docs" / "VALAIS_CLEAN_FP32_VALIDATION_PROTOCOL_2026-09-30.md",
    }
    hashes = {name: file_sha256(path) for name, path in files.items()}
    for name, path in files.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(output / "source_sha256.json", hashes)
    shutil.copyfile(spec_path, output / "study_config.yaml")
    shutil.copyfile(base.source, output / "base_config.yaml")
    write_json(output / "protocol_identity.json", {
        "fingerprint": fingerprint,
        "training_amp": True,
        "validation_amp": False,
        "test_accessed": False,
    })
    return hashes


def _verify_snapshot(output: Path, expected: dict[str, str]) -> None:
    workspace = Path(__file__).resolve().parents[1]
    for name, sha in expected.items():
        if (file_sha256(workspace / name) != sha
                or file_sha256(output / "source" / name) != sha):
            raise RuntimeError(f"Source snapshot changed: {name}")


def train(spec_path: Path, *, resume: bool = False, handshake_path: Path) -> dict:
    spec, paths, base, fingerprint = load_study(spec_path)
    output = paths["output_dir"]
    if resume:
        if not output.is_dir() or not (output / "last.pt").is_file():
            raise FileNotFoundError("Resume requires an existing complete checkpoint")
        checkpoint = _read_checkpoint(output / "last.pt")
        if checkpoint["study_fingerprint"] != fingerprint:
            raise RuntimeError("Resume protocol/source fingerprint differs")
        if not (output / "status.json").is_file():
            raise RuntimeError("Resume status is missing")
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if status.get("status") not in (
            "paused_temperature_guard", "paused_temperature_unavailable",
            "paused_temperature_guard_error", "paused_user", "interrupted",
        ):
            raise RuntimeError("Only a guarded or explicit interruption may resume")
        snapshot = json.loads((output / "source_sha256.json").read_text(encoding="utf-8"))
        _verify_snapshot(output, snapshot)
    else:
        if output.exists():
            raise FileExistsError(output)
        checkpoint = None
    device = setup_runtime(base)
    audit = audit_data(base, full=True)
    source_audit = json.loads(paths["source_data_audit"].read_text(encoding="utf-8"))
    if audit != source_audit:
        raise RuntimeError("Current train/val bytes differ from frozen source audit")
    audit_sha = _digest(audit)
    if resume and checkpoint["data_audit_digest"] != audit_sha:
        raise RuntimeError("Resume data audit differs")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "environment.json", environment_record(base, device))
        write_json(output / "data_audit.json", audit)
        snapshot = _snapshot(output, spec_path, base, fingerprint)
    try:
        write_json(output / "status.json", {
            "status": "waiting_guard", "kind": "research",
            "training_amp": True, "validation_amp": False,
            "test_evaluated": False,
        }, overwrite=True)
        datasets = build_datasets(base, ("train", "val"))
        loaders = {
            split: make_loader(ds, base, training=split == "train")
            for split, ds in datasets.items()
        }
        model = move_model(
            build_model(base, initialize_encoder=not resume), device,
            base.values["runtime"]["memory_format"],
        )
        settings = base.values["train"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"],
            betas=tuple(settings["adam_betas"]),
            weight_decay=settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"]
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        loss_fn = ChangeDetectionLoss(**base.values["loss"])
        history: list[dict] = []
        best_val_f1, best_epoch, start_epoch = -1.0, 0, 1
        if resume:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            best_val_f1 = checkpoint["best_val_f1"]
            best_epoch = checkpoint["best_epoch"]
            start_epoch = checkpoint["epoch"] + 1
            _restore_rng(checkpoint["rng"], loaders["train"])
        _wait_guard(handshake_path, fingerprint)
        write_json(output / "status.json", {
            "status": "running", "epoch": start_epoch - 1,
            "epochs": settings["epochs"], "kind": "research",
            "training_amp": True, "validation_amp": False,
            "test_evaluated": False,
        }, overwrite=True)
        common = dict(
            threshold=base.values["evaluation"]["threshold"],
            memory_format=base.values["runtime"]["memory_format"],
        )
        for epoch in range(start_epoch, settings["epochs"] + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            training = run_epoch(
                model, loaders["train"], loss_fn, device,
                amp=True, optimizer=optimizer, scaler=scaler,
                grad_clip_norm=settings["grad_clip_norm"],
                progress=f"epoch={epoch}/train", **common,
            )
            validation = run_epoch(
                model, loaders["val"], loss_fn, device,
                amp=False, progress=f"epoch={epoch}/val", **common,
            )
            improved = validation["metrics"]["f1"] > best_val_f1
            if improved:
                best_val_f1 = validation["metrics"]["f1"]
                best_epoch = epoch
            scheduler.step()
            history.append({
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": training, "val": validation,
            })
            payload = {
                "format_version": 2, "kind": "research", "epoch": epoch,
                "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "study_fingerprint": fingerprint, "data_audit_digest": audit_sha,
                "training_amp": True, "validation_amp": False,
                "test_evaluated": False,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": _rng_state(loaders["train"]), "history": history,
            }
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "history.json", history, overwrite=True)
            write_json(output / "status.json", {
                "status": "running", "kind": "research", "epoch": epoch,
                "epochs": settings["epochs"], "best_epoch": best_epoch,
                "best_val_f1": best_val_f1, "training_amp": True,
                "validation_amp": False, "test_evaluated": False,
            }, overwrite=True)
            print(f"epoch={epoch} val_f1={validation['metrics']['f1']:.6f} best={best_val_f1:.6f}", flush=True)
        result = {
            "status": "completed", "kind": "research", "epochs": settings["epochs"],
            "best_epoch": best_epoch, "best_val_f1": best_val_f1,
            "training_amp": True, "validation_amp": False, "test_evaluated": False,
        }
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "kind": "research", "error": f"{type(error).__name__}: {error}",
            "training_amp": True, "validation_amp": False, "test_evaluated": False,
        }, overwrite=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-handshake", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.preflight_only:
        spec, paths, base, fingerprint = load_study(path)
        print(json.dumps({"status": "preflight_passed", "fingerprint": fingerprint,
                          "test_accessed": False}))
    else:
        result = train(path, resume=args.resume, handshake_path=args.guard_handshake.resolve())
        print(json.dumps(result))


if __name__ == "__main__":
    main()
