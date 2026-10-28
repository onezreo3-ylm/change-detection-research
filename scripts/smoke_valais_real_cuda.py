"""One real-data CUDA train/eval step for engineering only, never a score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from pcdc.config import load_config
from pcdc.data import build_datasets
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json


def _batch(dataset, count: int, device: torch.device, memory_format: str):
    rows = [dataset[index] for index in range(count)]
    layout = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format
    a = torch.stack([row["image_a"] for row in rows]).to(device=device, memory_format=layout)
    b = torch.stack([row["image_b"] for row in rows]).to(device=device, memory_format=layout)
    mask = torch.stack([row["mask"] for row in rows]).to(device=device)
    return a, b, mask, [row["sample_id"] for row in rows]


def run(config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    if config.run_test_after_training:
        raise RuntimeError("Development smoke requires test access disabled")
    device = setup_runtime(config)
    if device.type != "cuda":
        raise RuntimeError("This smoke requires real CUDA")
    datasets = build_datasets(config, ("train", "val"))
    settings = config.values
    memory = settings["runtime"]["memory_format"]
    a, b, target, train_ids = _batch(datasets["train"], 2, device, memory)
    model = move_model(build_model(config), device, memory)
    loss_fn = ChangeDetectionLoss(**settings["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["train"]["learning_rate"],
        weight_decay=settings["train"]["weight_decay"],
        betas=tuple(settings["train"]["adam_betas"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=settings["train"]["amp"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=settings["train"]["amp"]):
        logits = model(a, b)["logits"]
        losses = loss_fn(logits, target)
    if logits.shape != target.shape or not torch.isfinite(losses["total"]):
        raise RuntimeError("Invalid CUDA train forward")
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["train"]["grad_clip_norm"]
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite CUDA train gradients")
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before:
        raise RuntimeError("AMP skipped the engineering update")
    va, vb, vt, val_ids = _batch(datasets["val"], 2, device, memory)
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", enabled=settings["train"]["amp"]
    ):
        val_logits = model(va, vb)["logits"]
        val_loss = loss_fn(val_logits, vt)["total"]
    if val_logits.shape != vt.shape or not torch.isfinite(val_loss):
        raise RuntimeError("Invalid CUDA validation forward")
    payload = {
        "status": "engineering_smoke_only",
        "research_score": False,
        "test_accessed": False,
        "model_initialized_randomly": settings["model"]["encoder_weights"] is None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_shape": list(logits.shape),
        "val_shape": list(val_logits.shape),
        "train_loss_finite": True,
        "val_loss_finite": True,
        "grad_norm_finite": True,
        "optimizer_steps": 1,
        "environment": environment_record(config, device),
    }
    write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "test_accessed": False}))


if __name__ == "__main__":
    main()
