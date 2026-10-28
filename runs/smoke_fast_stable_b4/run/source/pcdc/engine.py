"""Phase 1 train/evaluate loop with explicit protocol and checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, protocol_signature
from .data import audit_data, build_datasets, file_sha256, make_loader
from .losses import ChangeDetectionLoss
from .metrics import BinaryMetrics
from .model import build_model
from .runtime import environment_record, save_checkpoint, setup_runtime, write_json


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def training_signature(config: Config) -> str:
    # Resume is stricter than inference: optimizer, seed, runtime, budget all match.
    values = json.loads(json.dumps(config.values))
    values["data"].pop("root")
    values["model"].pop("weights_cache")
    values["experiment"].pop("output_dir")
    return _digest(values)


def source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)).replace("\\", "/"): path
            for path in sorted((root / "pcdc").glob("*.py"))}


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in source_files().items()}


def move_model(model: nn.Module, device: torch.device, memory_format: str) -> nn.Module:
    if memory_format == "channels_last":
        return model.to(device=device, memory_format=torch.channels_last)
    if memory_format != "contiguous":
        raise ValueError(f"Unsupported memory format: {memory_format}")
    return model.to(device=device)


def read_checkpoint(path: Path) -> dict[str, Any]:
    # Never fall back to arbitrary pickle execution for incompatible checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Not a supported Phase 1 checkpoint")
    return checkpoint


def check_checkpoint(config: Config, checkpoint: dict[str, Any]) -> None:
    if checkpoint["protocol_signature"] != protocol_signature(config.values):
        raise ValueError("Checkpoint and evaluation protocol differ; no silent override")
    if checkpoint["kind"] != config.values["experiment"]["kind"]:
        raise ValueError("Cannot relabel a smoke checkpoint as a research experiment")


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: ChangeDetectionLoss,
              device: torch.device, *, threshold: float, amp: bool = False,
              memory_format: str = "contiguous",
              optimizer: torch.optim.Optimizer | None = None,
              scaler: Any = None, grad_clip_norm: float | None = None,
              progress: str | None = None) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = BinaryMetrics(threshold)
    sums = {key: 0.0 for key in ("total", "bce", "dice")}
    samples = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    started = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        transfer_format = (torch.channels_last if memory_format == "channels_last"
                           else torch.contiguous_format)
        a = batch["image_a"].to(device=device, non_blocking=True,
                                memory_format=transfer_format)
        b = batch["image_b"].to(device=device, non_blocking=True,
                                memory_format=transfer_format)
        target = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(a, b)["logits"]
                losses = loss_fn(logits, target)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Non-finite loss; experiment stopped")
            if training:
                if scaler is None:
                    raise ValueError("Training requires an explicitly configured GradScaler")
                scale_before = float(scaler.get_scale())
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                # GradScaler is specifically designed to skip an occasional FP16
                # overflow and reduce its loss scale. FP32 non-finite gradients
                # remain a hard failure. For AMP, verify that a non-finite norm is
                # actually recognized by GradScaler before allowing the skipped step.
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm or float("inf"),
                    error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                if torch.isfinite(total_norm).item():
                    optimizer_steps += 1
                elif amp:
                    scale_after = float(scaler.get_scale())
                    if not scale_after < scale_before:
                        raise FloatingPointError(
                            "Non-finite AMP gradients were not handled by GradScaler")
                    amp_skipped_steps += 1
                    if progress:
                        print(f"{progress} step={step}/{len(loader)} amp_overflow_skip=1 "
                              f"scale={scale_before:g}->{scale_after:g}", flush=True)
        count = target.shape[0]
        samples += count
        for key, value in losses.items():
            sums[key] += value.detach().item() * count
        metrics.update(logits.detach(), target)
        if progress and (step == 1 or step % max(1, len(loader) // 10) == 0 or step == len(loader)):
            print(f"{progress} step={step}/{len(loader)} loss={sums['total']/samples:.6f}", flush=True)
    if samples == 0:
        raise ValueError("Empty loader")
    return {"loss": {key: value / samples for key, value in sums.items()},
            "metrics": metrics.compute(), "samples": samples,
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": amp_skipped_steps,
            "seconds": time.monotonic() - started}


def _rng_state(loader: DataLoader) -> dict[str, Any]:
    numpy_rng = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def _restore_rng(state: dict[str, Any], loader: DataLoader) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def _evaluate_model(config: Config, checkpoint: dict[str, Any], device: torch.device,
                    split: str) -> dict[str, Any]:
    check_checkpoint(config, checkpoint)
    dataset = build_datasets(config, (split,))[split]
    if dataset.manifest() != checkpoint["manifests"][split]:
        raise ValueError(f"{split} inventory differs from the training audit")
    # Inventory byte counts alone cannot detect same-size content changes.
    for record in dataset.records:
        expected = checkpoint["file_sha256"][split][record.name]
        for key, checksum in expected.items():
            if file_sha256(getattr(record, key)) != checksum:
                raise ValueError(f"Changed dataset content: {split}/{record.name}/{key}")
    model = move_model(build_model(config, initialize_encoder=False), device,
                       config.values["runtime"]["memory_format"])
    model.load_state_dict(checkpoint["model"], strict=True)
    result = run_epoch(model, make_loader(dataset, config, training=False),
                       ChangeDetectionLoss(**config.values["loss"]), device,
                       threshold=config.values["evaluation"]["threshold"],
                       amp=config.values["train"]["amp"],
                       memory_format=config.values["runtime"]["memory_format"],
                       progress=f"evaluate/{split}")
    return {"kind": checkpoint["kind"], "split": split, "epoch": checkpoint["epoch"],
            "selection": "validation_f1_only", "best_val_f1": checkpoint["best_val_f1"],
            "protocol_signature": checkpoint["protocol_signature"],
            "training_source_sha256": checkpoint["source_sha256"],
            "evaluation_source_sha256": source_hashes(),
            "environment": environment_record(config, device), **result}


def evaluate(config: Config, checkpoint_path: Path, split: str, output: Path) -> dict[str, Any]:
    if split not in ("val", "test"):
        raise ValueError("Standalone evaluation supports val or test")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    device = setup_runtime(config)
    checkpoint = read_checkpoint(checkpoint_path)
    result = _evaluate_model(config, checkpoint, device, split)
    result.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=file_sha256(checkpoint_path))
    write_json(output, result)
    return result


def train(config: Config, *, resume: bool = False) -> dict[str, Any]:
    output = config.output_dir
    if not resume and output.exists():
        raise FileExistsError(f"Run already exists: {output}. Use --resume or a new YAML output_dir.")
    device = setup_runtime(config)
    print("Auditing all three real/smoke splits, decoded labels and SHA256...", flush=True)
    audit = audit_data(config, full=True)
    manifests = {split: {k: v for k, v in report.items() if k != "file_sha256"}
                 for split, report in audit["splits"].items()}
    content = {split: report["file_sha256"] for split, report in audit["splits"].items()}
    code_hashes = source_hashes()
    checkpoint = read_checkpoint(output / "last.pt") if resume else None
    if checkpoint:
        check_checkpoint(config, checkpoint)
        if checkpoint["training_signature"] != training_signature(config):
            raise ValueError("Resume must preserve all training and runtime settings")
        if checkpoint["audit_sha256"] != _digest(audit):
            raise ValueError("Dataset changed since checkpoint; resume refused")
        if checkpoint["source_sha256"] != code_hashes:
            raise ValueError("Source changed since checkpoint; use a new run, not resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "environment.json", environment_record(config, device))
        write_json(output / "data_audit.json", audit)
        write_json(output / "source_sha256.json", code_hashes)
        with (output / "config.yaml").open("x", encoding="utf-8") as stream:
            # Absolute paths let the archived config be used directly for evaluation.
            archived = json.loads(json.dumps(config.values))
            archived["data"]["root"] = str(config.data_root)
            archived["experiment"]["output_dir"] = str(output)
            archived["model"]["weights_cache"] = str(config.path(archived["model"]["weights_cache"]))
            yaml.safe_dump(archived, stream, sort_keys=False, allow_unicode=True)
        for name, path in source_files().items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    try:
        write_json(output / "status.json", {"status": "running", "kind": config.values["experiment"]["kind"]}, overwrite=True)
        datasets = build_datasets(config, ("train", "val"))
        loaders = {split: make_loader(ds, config, training=split == "train") for split, ds in datasets.items()}
        model = move_model(build_model(config, initialize_encoder=not resume), device,
                           config.values["runtime"]["memory_format"])
        settings = config.values["train"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
        loss_fn = ChangeDetectionLoss(**config.values["loss"])
        history, best_val_f1, best_epoch, start = [], -1.0, 0, 1
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history = checkpoint["history"]
            best_val_f1, best_epoch = checkpoint["best_val_f1"], checkpoint["best_epoch"]
            start = checkpoint["epoch"] + 1
            _restore_rng(checkpoint["rng"], loaders["train"])
        for epoch in range(start, settings["epochs"] + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            common = dict(threshold=config.values["evaluation"]["threshold"], amp=settings["amp"],
                          memory_format=config.values["runtime"]["memory_format"])
            training = run_epoch(model, loaders["train"], loss_fn, device, optimizer=optimizer,
                                 scaler=scaler, grad_clip_norm=settings["grad_clip_norm"],
                                 progress=f"epoch={epoch}/train", **common)
            validation = run_epoch(model, loaders["val"], loss_fn, device,
                                   progress=f"epoch={epoch}/val", **common)
            improved = validation["metrics"]["f1"] > best_val_f1
            if improved:
                best_val_f1, best_epoch = validation["metrics"]["f1"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": learning_rate,
                            "train": training, "val": validation})
            payload = {"format_version": 1, "kind": config.values["experiment"]["kind"],
                       "epoch": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "config": config.values, "protocol_signature": protocol_signature(config.values),
                       "training_signature": training_signature(config), "audit_sha256": _digest(audit),
                       "manifests": manifests, "file_sha256": content, "source_sha256": code_hashes,
                       "rng": _rng_state(loaders["train"]), "history": history}
            if improved:
                save_checkpoint(output / "best.pt", payload)
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "history.json", history, overwrite=True)
            write_json(output / "status.json", {"status": "running", "epoch": epoch,
                       "epochs": settings["epochs"], "best_epoch": best_epoch,
                       "best_val_f1": best_val_f1, "kind": payload["kind"]}, overwrite=True)
            print(f"epoch={epoch} val_f1={validation['metrics']['f1']:.6f} best={best_val_f1:.6f}", flush=True)
        # Held-out test is visited for prediction only after the training budget is complete.
        del model, optimizer, scheduler, scaler, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best = read_checkpoint(output / "best.pt")
        result = _evaluate_model(config, best, device, "test")
        result.update(checkpoint=str(output / "best.pt"), checkpoint_sha256=file_sha256(output / "best.pt"))
        write_json(output / "test_metrics.json", result, overwrite=True)
        summary = {"status": "completed", "kind": config.values["experiment"]["kind"],
                   "epochs": settings["epochs"], "best_epoch": best_epoch,
                   "best_val_f1": best_val_f1, "test": result["metrics"]}
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "kind": config.values["experiment"]["kind"], "error": f"{type(error).__name__}: {error}"}, overwrite=True)
        raise
