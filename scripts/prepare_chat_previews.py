"""Make bounded, overview-only transport copies of existing TRAIN QA sheets.

These copies are for chat navigation. Scientific visibility review must still
use the unchanged original sheets/tiles, especially for noise, blur and JPEG.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import yaml

from pcdc.config import UniqueKeyLoader


KEYS = {"schema_version", "source_directories", "output_dir", "max_long_edge",
        "minimum_long_edge", "max_image_bytes", "max_images_per_batch",
        "max_batch_base64_bytes", "overview_only"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(config: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config.parent / path).resolve()


def _encode(image: Image.Image, max_edge: int, min_edge: int,
            byte_limit: int) -> tuple[bytes, tuple[int, int]]:
    edge = min(max(image.size), max_edge)
    while True:
        ratio = edge / max(image.size)
        size = tuple(max(1, round(value * ratio)) for value in image.size)
        overview = image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image.copy()
        buffer = BytesIO()
        overview.save(buffer, format="PNG", optimize=True, compress_level=9)
        data = buffer.getvalue()
        if len(data) <= byte_limit:
            return data, size
        if edge <= min_edge:
            raise ValueError("Preview cannot meet the byte budget at the minimum size")
        edge = max(min_edge, int(edge * 0.85))


def prepare(config_path: Path) -> dict:
    config_path = config_path.resolve()
    settings = yaml.load(config_path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(settings, dict) or set(settings) != KEYS:
        raise ValueError("Unexpected chat preview config keys")
    if settings["schema_version"] != 1 or settings["overview_only"] is not True:
        raise ValueError("Transport copies must be explicitly overview-only")
    for key in ("max_long_edge", "minimum_long_edge", "max_image_bytes",
                "max_images_per_batch", "max_batch_base64_bytes"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if settings["minimum_long_edge"] > settings["max_long_edge"]:
        raise ValueError("Minimum image size exceeds the maximum")
    if settings["max_images_per_batch"] > 2:
        raise ValueError("A chat preview batch may contain at most two images")
    values = settings["source_directories"]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise ValueError("Explicit train QA source directories are required")
    sources = [_resolve(config_path, value) for value in values]
    destination = _resolve(config_path, settings["output_dir"])
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate source directories")
    files = []
    for index, source in enumerate(sources, 1):
        # Validate scope before opening any image. Never traverse dataset splits.
        if (source.name != "previews" or "train" not in source.parent.name.lower()
                or "qa" not in source.parent.name.lower()
                or any(part.lower() in {"test", "val", "validation"} for part in source.parts)):
            raise ValueError("Only existing train QA previews directories are allowed")
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Output and source directories must be separate")
        if not source.is_dir():
            raise FileNotFoundError(source)
        selected = sorted(source.glob("*.png"))
        if not selected:
            raise ValueError(f"No existing PNG QA sheets in {source}")
        if any(path.is_symlink() or path.resolve().parent != source for path in selected):
            raise ValueError("QA image links outside the source directory are not allowed")
        files.extend((index, path) for path in selected)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    batches = []
    current_batch = []
    current_bytes = 0
    for index, source in files:
        source_hash = _sha256(source)
        with Image.open(source) as image:
            image.load()
            if image.mode not in {"RGB", "L"}:
                raise ValueError("Only opaque RGB/grayscale QA sheets are supported")
            original_size = image.size
            data, size = _encode(image, settings["max_long_edge"],
                                 settings["minimum_long_edge"], settings["max_image_bytes"])
        if _sha256(source) != source_hash:
            raise RuntimeError(f"Source changed while copying: {source}")
        # Include the PNG data-URL prefix in the transport budget.
        encoded_bytes = 22 + 4 * ((len(data) + 2) // 3)
        if encoded_bytes > settings["max_batch_base64_bytes"]:
            raise ValueError("One preview exceeds the entire batch budget")
        if current_batch and (len(current_batch) >= settings["max_images_per_batch"]
                              or current_bytes + encoded_bytes > settings["max_batch_base64_bytes"]):
            batches.append({"images": current_batch, "base64_bytes": current_bytes})
            current_batch, current_bytes = [], 0
        target = destination / f"source-{index:02d}" / source.name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        records.append({"source": str(source), "source_sha256": source_hash,
                        "source_bytes": source.stat().st_size, "source_size": list(original_size),
                        "preview": str(target), "preview_sha256": hashlib.sha256(data).hexdigest(),
                        "preview_bytes": len(data), "preview_size": list(size),
                        "base64_bytes": encoded_bytes, "overview_only": True})
        current_batch.append(str(target))
        current_bytes += encoded_bytes
    if current_batch:
        batches.append({"images": current_batch, "base64_bytes": current_bytes})
    if any(_sha256(Path(record["source"])) != record["source_sha256"] for record in records):
        raise RuntimeError("A source QA sheet changed during generation")
    report = {"schema_version": 1, "overview_only": True,
              "scientific_review_evidence": False, "originals_unchanged": True,
              "dataset_images_or_labels_accessed": False,
              "config_sha256": _sha256(config_path), "settings": settings,
              "source_bytes": sum(record["source_bytes"] for record in records),
              "preview_bytes": sum(record["preview_bytes"] for record in records),
              "image_count": len(records), "images": records, "batches": batches}
    with (destination / "transport_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.config)
    print(json.dumps({key: report[key] for key in ("image_count", "source_bytes", "preview_bytes",
                                                 "overview_only", "originals_unchanged")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
