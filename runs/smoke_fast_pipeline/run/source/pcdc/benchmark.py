"""Real train/val throughput benchmark; never accesses test or reports research scores."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import Config, protocol_signature
from .data import build_datasets, make_loader
from .engine import move_model, run_epoch, source_hashes
from .losses import ChangeDetectionLoss
from .model import build_model
from .runtime import environment_record, setup_runtime, write_json


class _IteratorSlice:
    """A fixed-length view over one existing loader iterator."""

    def __init__(self, iterator: Iterator, steps: int):
        self.iterator = iterator
        self.steps = steps

    def __iter__(self):
        return islice(self.iterator, self.steps)

    def __len__(self) -> int:
        return self.steps


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(config: Config, output: Path, *, warmup_steps: int,
              train_steps: int, eval_steps: int) -> dict[str, Any]:
    """Measure the configured pipeline on real train/val tiles only."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    for name, value, minimum in (("warmup_steps", warmup_steps, 0),
                                 ("train_steps", train_steps, 1),
                                 ("eval_steps", eval_steps, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    device = setup_runtime(config)
    datasets = build_datasets(config, ("train", "val"))
    loaders = {split: make_loader(dataset, config, training=split == "train")
               for split, dataset in datasets.items()}
    requested = {"train": warmup_steps + train_steps, "val": warmup_steps + eval_steps}
    for split, steps in requested.items():
        if steps > len(loaders[split]):
            raise ValueError(f"{split} benchmark needs {steps} batches, loader has {len(loaders[split])}")

    settings = config.values["train"]
    memory_format = config.values["runtime"]["memory_format"]
    model = move_model(build_model(config), device, memory_format)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  betas=tuple(settings["adam_betas"]),
                                  weight_decay=settings["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
    loss_fn = ChangeDetectionLoss(**config.values["loss"])
    common = {"threshold": config.values["evaluation"]["threshold"],
              "amp": settings["amp"], "memory_format": memory_format}

    train_iterator = iter(loaders["train"])
    if warmup_steps:
        run_epoch(model, _IteratorSlice(train_iterator, warmup_steps), loss_fn, device,
                  optimizer=optimizer, scaler=scaler,
                  grad_clip_norm=settings["grad_clip_norm"], **common)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_result = run_epoch(model, _IteratorSlice(train_iterator, train_steps), loss_fn, device,
                             optimizer=optimizer, scaler=scaler,
                             grad_clip_norm=settings["grad_clip_norm"], **common)
    _synchronize(device)
    train_peak = ({"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                   "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
                  if device.type == "cuda" else None)

    val_iterator = iter(loaders["val"])
    if warmup_steps:
        run_epoch(model, _IteratorSlice(val_iterator, warmup_steps), loss_fn, device, **common)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_result = run_epoch(model, _IteratorSlice(val_iterator, eval_steps), loss_fn, device, **common)
    _synchronize(device)
    eval_peak = ({"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                  "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
                 if device.type == "cuda" else None)

    def throughput(result: dict[str, Any], steps: int) -> dict[str, float]:
        return {"steps_per_second": steps / result["seconds"],
                "samples_per_second": result["samples"] / result["seconds"]}

    report = {
        "kind": "engineering_throughput_benchmark_not_research_results",
        "warning": "Uses real train/val tiles for speed only; test was not opened and metrics are not research results.",
        "config": str(config.source),
        "protocol_signature": protocol_signature(config.values),
        "settings": {
            "batch_size": settings["batch_size"],
            "eval_batch_size": settings["eval_batch_size"],
            "num_workers": settings["num_workers"],
            "amp": settings["amp"],
            "memory_format": memory_format,
            "warmup_steps": warmup_steps,
            "train_steps": train_steps,
            "eval_steps": eval_steps,
        },
        "inventory": {split: dataset.manifest() for split, dataset in datasets.items()},
        "train": {"seconds": train_result["seconds"], "samples": train_result["samples"],
                  **throughput(train_result, train_steps), "cuda_peak": train_peak},
        "val": {"seconds": eval_result["seconds"], "samples": eval_result["samples"],
                **throughput(eval_result, eval_steps), "cuda_peak": eval_peak},
        "environment": environment_record(config, device),
        "source_sha256": source_hashes(),
    }
    write_json(output, report)
    return report
