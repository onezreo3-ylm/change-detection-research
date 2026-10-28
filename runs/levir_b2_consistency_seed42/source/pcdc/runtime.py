"""Reproducible execution, environment records, and safe artifact writes."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision
import yaml

from .config import Config


def setup_runtime(config: Config) -> torch.device:
    cfg = config.values
    strict = cfg["runtime"]["deterministic"]
    if strict:
        value = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if value not in (":4096:8", ":16:8"):
            raise ValueError("Strict reproducibility requires a supported CUBLAS_WORKSPACE_CONFIG")
    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Explicitly set runtime.device: cpu if intended.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device does not exist: {device}")
    return device


def environment_record(config: Config, device: torch.device) -> dict[str, Any]:
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "numpy": str(np.__version__), "pillow": str(PIL.__version__), "pyyaml": str(yaml.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": config.values["experiment"]["seed"],
            "deterministic": config.values["runtime"]["deterministic"],
            "num_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "source_config": str(config.source)}


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Replace only this run's checkpoint after a complete write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

