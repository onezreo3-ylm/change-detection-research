"""One real Valais batch at the frozen full-FP32 training batch size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from pcdc.config import load_config
from pcdc.data import build_datasets
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json


GPU_UUID = "GPU-8ada1895-7628-bb88-1dda-c57679117fa2"


def gpu_temperature() -> int:
    output = subprocess.run(
        ["C:/Windows/System32/nvidia-smi.exe", "-i", "0",
         "--query-gpu=uuid,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()
    uuid, value = [part.strip() for part in output.split(",")]
    if uuid != GPU_UUID:
        raise RuntimeError("FP32 smoke GPU UUID changed")
    temperature = int(value)
    if temperature >= 85:
        raise RuntimeError("GPU is too warm for FP32 smoke")
    return temperature


def _batch(dataset, count: int, device: torch.device):
    samples = [dataset[index] for index in range(count)]
    return (
        torch.stack([item["image_a"] for item in samples]).to(
            device=device, memory_format=torch.channels_last),
        torch.stack([item["image_b"] for item in samples]).to(
            device=device, memory_format=torch.channels_last),
        torch.stack([item["mask"] for item in samples]).to(device=device),
        [item["sample_id"] for item in samples],
    )


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    if (config.run_test_after_training or config.values["runtime"]["device"] != "cuda"
            or config.values["train"]["amp"] is not False
            or config.values["train"]["batch_size"] != 4
            or config.values["train"]["epochs"] != 100):
        raise RuntimeError("Smoke requires frozen full-FP32 val-only protocol")
    device = setup_runtime(config)
    first = gpu_temperature()
    datasets = build_datasets(config, ("train", "val"))
    a, b, target, train_ids = _batch(datasets["train"], 4, device)
    model = move_model(build_model(config), device, "channels_last")
    loss_fn = ChangeDetectionLoss(**config.values["loss"])
    settings = config.values["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    model.train()
    logits = model(a, b)["logits"]
    losses = loss_fn(logits, target)
    if not all(torch.isfinite(value).item() for value in losses.values()):
        raise RuntimeError("Full-FP32 training smoke loss is non-finite")
    losses["total"].backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    va, vb, vt, val_ids = _batch(datasets["val"], 4, device)
    model.eval()
    with torch.no_grad():
        val_logits = model(va, vb)["logits"]
        val_losses = loss_fn(val_logits, vt)
    if not all(torch.isfinite(value).item() for value in val_losses.values()):
        raise RuntimeError("Full-FP32 validation smoke loss is non-finite")
    last = gpu_temperature()
    report = {
        "status": "engineering_smoke_only", "research_score": False,
        "test_accessed": False, "training_amp": False, "validation_amp": False,
        "train_batch_size": 4, "train_ids": train_ids, "val_ids": val_ids,
        "train_logits_dtype": str(logits.dtype),
        "val_logits_dtype": str(val_logits.dtype),
        "gradient_norm_finite": bool(torch.isfinite(norm).item()),
        "one_optimizer_step": True, "first_temperature_c": first,
        "last_temperature_c": last,
        "environment": environment_record(config, device),
    }
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "test_accessed": False}))


if __name__ == "__main__":
    main()
