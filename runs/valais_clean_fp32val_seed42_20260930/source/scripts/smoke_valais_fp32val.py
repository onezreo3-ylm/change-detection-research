"""Real train/val single-batch engineering smoke for the new numeric protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from pcdc.data import build_datasets
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts.train_valais_fp32val import load_study


def _batch(dataset, count: int, device: torch.device):
    rows = [dataset[index] for index in range(count)]
    a = torch.stack([row["image_a"] for row in rows]).to(
        device=device, memory_format=torch.channels_last
    )
    b = torch.stack([row["image_b"] for row in rows]).to(
        device=device, memory_format=torch.channels_last
    )
    target = torch.stack([row["mask"] for row in rows]).to(device=device)
    return a, b, target, [row["sample_id"] for row in rows]


def run(spec_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    spec, paths, base, fingerprint = load_study(spec_path)
    device = setup_runtime(base)
    if device.type != "cuda":
        raise RuntimeError("Real CUDA is required")
    nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")

    def temperature_and_uuid():
        response = subprocess.run(
            [str(nvidia_smi), "-i", "0", "--query-gpu=uuid,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        uuid, temperature = [value.strip() for value in response.stdout.strip().split(",")]
        if uuid != spec["guard"]["gpu_uuid"]:
            raise RuntimeError("GPU UUID changed")
        return int(temperature)

    first_temperature = temperature_and_uuid()
    if first_temperature >= spec["guard"]["confirmed_pause_c"]:
        raise RuntimeError("GPU too warm for the engineering smoke")
    datasets = build_datasets(base, ("train", "val"))
    train_a, train_b, train_target, train_ids = _batch(datasets["train"], 2, device)
    model = move_model(build_model(base), device, "channels_last")
    loss_fn = ChangeDetectionLoss(**base.values["loss"])
    settings = base.values["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True):
        train_logits = model(train_a, train_b)["logits"]
        train_losses = loss_fn(train_logits, train_target)
    if not all(torch.isfinite(value).item() for value in train_losses.values()):
        raise RuntimeError("Non-finite AMP training smoke loss")
    scaler.scale(train_losses["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["grad_clip_norm"]
    )
    if not torch.isfinite(grad_norm).item():
        raise RuntimeError("Non-finite AMP training smoke gradient")
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < before:
        raise RuntimeError("Engineering smoke AMP update skipped")
    val_a, val_b, val_target, val_ids = _batch(datasets["val"], 2, device)
    model.eval()
    with torch.set_grad_enabled(False), torch.autocast(device_type="cuda", enabled=False):
        val_logits = model(val_a, val_b)["logits"]
        val_losses = loss_fn(val_logits, val_target)
    if not all(torch.isfinite(value).item() for value in val_losses.values()):
        raise RuntimeError("Non-finite FP32 validation smoke loss")
    last_temperature = temperature_and_uuid()
    report = {
        "status": "engineering_smoke_only",
        "test_accessed": False,
        "research_score": False,
        "training_amp": True,
        "validation_amp": False,
        "study_fingerprint": fingerprint,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_logits_dtype": str(train_logits.dtype),
        "val_logits_dtype": str(val_logits.dtype),
        "train_logits_shape": list(train_logits.shape),
        "val_logits_shape": list(val_logits.shape),
        "finite_train_loss": True,
        "finite_val_loss": True,
        "one_optimizer_step": True,
        "first_temperature_c": first_temperature,
        "last_temperature_c": last_temperature,
        "environment": environment_record(base, device),
    }
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "test_accessed": False}))


if __name__ == "__main__":
    main()
