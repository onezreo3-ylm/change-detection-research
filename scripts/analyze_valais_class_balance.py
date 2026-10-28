"""Train/validation label balance audit; no model or held-out test access."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from math import comb
from pathlib import Path

import numpy as np
from PIL import Image

from pcdc.config import load_config


def empty_batch_probability(empty: int, total: int, batch_size: int) -> float:
    """Exact probability of no positive tile in a uniform batch without replacement."""
    if not 0 <= empty <= total or not 0 < batch_size <= total:
        raise ValueError("Invalid population or batch size")
    if empty < batch_size:
        return 0.0
    return comb(empty, batch_size) / comb(total, batch_size)


def summarize_labels(folder: Path, expected: int, batch_size: int) -> dict:
    paths = sorted(folder.glob("*.tif"))
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} labels in {folder}, found {len(paths)}")
    digest = hashlib.sha256()
    positive = []
    total_positive_pixels = 0
    for path in paths:
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + data)
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (256, 256) or image.mode != "L":
                raise RuntimeError(f"Unexpected Valais label format: {path}")
            array = np.asarray(image)
        if not np.isin(array, (0, 1)).all():
            raise RuntimeError(f"Unexpected label values: {path}")
        count = int(array.sum())
        total_positive_pixels += count
        if count:
            positive.append({"name": path.name, "positive_pixels": count})
    nonempty = len(positive)
    empty = expected - nonempty
    counts = sorted(item["positive_pixels"] for item in positive)

    def nearest_rank(proportion: float) -> int | None:
        if not counts:
            return None
        position = max(0, min(len(counts) - 1, int(np.ceil(len(counts) * proportion)) - 1))
        return counts[position]

    return {
        "pairs": expected,
        "positive_pairs": nonempty,
        "empty_pairs": empty,
        "positive_pair_fraction": nonempty / expected,
        "positive_pixels": total_positive_pixels,
        "total_pixels": expected * 256 * 256,
        "positive_pixel_fraction": total_positive_pixels / (expected * 256 * 256),
        "natural_batch_size": batch_size,
        "natural_empty_batch_probability": empty_batch_probability(empty, expected, batch_size),
        "nonempty_positive_pixels_p50": nearest_rank(0.50),
        "nonempty_positive_pixels_p90": nearest_rank(0.90),
        "nonempty_positive_pixels_max": counts[-1] if counts else None,
        "labels_name_and_bytes_sha256": digest.hexdigest(),
        "positive_manifest": positive,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if config.run_test_after_training:
        raise RuntimeError("Balance analysis requires test disabled")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data = config.values["data"]
    root = config.data_root
    report = {
        "status": "development_label_balance_only",
        "model_inference": False,
        "test_accessed": False,
        "config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "train": summarize_labels(
            root / data["splits"]["train"] / data["folders"]["mask"],
            data["expected_pairs"]["train"],
            config.values["train"]["batch_size"],
        ),
        "val": summarize_labels(
            root / data["splits"]["val"] / data["folders"]["mask"],
            data["expected_pairs"]["val"],
            config.values["train"]["batch_size"],
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        split: {key: report[split][key] for key in
                ("pairs", "positive_pairs", "empty_pairs", "positive_pixel_fraction",
                 "natural_empty_batch_probability")}
        for split in ("train", "val")
    }))


if __name__ == "__main__":
    main()
