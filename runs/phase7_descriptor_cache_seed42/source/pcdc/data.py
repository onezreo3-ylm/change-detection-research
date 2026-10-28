"""Strict LEVIR-CD pairs with deterministic, non-overlapping grid tiling."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import Config


@dataclass
class RawPair:
    """Phase 2 hook: cropped uint8 RGB arrays, uint8 binary mask, and metadata.

    A future pair transform belongs BEFORE normalization. Phase 1 supplies None.
    Any transform must preserve aligned image/mask dimensions and binary labels.
    """

    image_a: np.ndarray
    image_b: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


PairTransform = Callable[[RawPair], RawPair]


@dataclass(frozen=True)
class PairRecord:
    name: str
    image_a: Path
    image_b: Path
    mask: Path
    size: tuple[int, int]  # width, height


def _file_map(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing LEVIR-CD directory: {folder}")
    return {path.name: path for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in extensions}


def _read_rgb(path: Path, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image, got {image.mode}: {path}")
        return np.array(image.crop(box) if box is not None else image, dtype=np.uint8)


def _read_mask(path: Path, values: list[int], decode: str,
               box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "1", "P"):
            raise ValueError(f"Unsupported mask mode {image.mode}: {path}")
        array = np.array(image.crop(box) if box is not None else image)
    if array.ndim == 3:
        if array.shape[2] != 3 or not np.all(array == array[..., :1]):
            raise ValueError(f"RGB mask channels disagree: {path}")
        array = array[..., 0]
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 mask, got {array.dtype}: {path}")
    if decode == "author_uint8_floor255":
        # Mirrors the LEVIR-CD author's public BIT_CD `label // 255` loader.
        # The raw file is never modified; all uint8 values below 255 decode to 0.
        return (array // 255).astype(np.uint8)
    if decode != "strict_values":
        raise ValueError(f"Unsupported label decode strategy: {decode}")
    unique = np.unique(array)
    if not np.isin(unique, values).all():
        raise ValueError(f"Invalid mask values {unique.tolist()} in {path}; expected {values}")
    return (array == values[1]).astype(np.uint8)


class LEVIRDataset(Dataset):
    """A/B/label file names must match exactly. No inferred split or resizing."""

    def __init__(self, config: Config, split: str, pair_transform: PairTransform | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if split == "test" and not config.run_test_after_training:
            raise ValueError("Test access is disabled by evaluation.run_test_after_training: false")
        if (config.data_root / "SYNTHETIC_ONLY.json").exists() and config.values["experiment"]["kind"] != "smoke":
            raise ValueError("Synthetic fixture cannot be used as a research dataset")
        self.settings = config.values["data"]
        self.split = split
        self.root = (config.data_root / self.settings["splits"][split]).resolve()
        self.tile_size = self.settings["tile_size"]
        self.pair_transform = pair_transform
        self.mean = torch.tensor(self.settings["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(self.settings["std"], dtype=torch.float32)[:, None, None]
        extensions = {ext.lower() for ext in self.settings["extensions"]}
        maps = {key: _file_map(self.root / folder, extensions)
                for key, folder in self.settings["folders"].items()}
        names = set(maps["image_a"])
        if not names:
            raise ValueError(f"No image pairs in {self.root}")
        for key in ("image_b", "mask"):
            other = set(maps[key])
            if names != other:
                raise ValueError(f"Unmatched {split}/{key}: missing={sorted(names-other)[:5]}, "
                                 f"extra={sorted(other-names)[:5]}")
        expected = self.settings["expected_pairs"][split]
        if expected is not None and len(names) != expected:
            raise ValueError(f"{split}: expected {expected} source pairs, found {len(names)}")
        self.records: list[PairRecord] = []
        self.tiles: list[tuple[int, int, int]] = []  # record index, x, y
        for name in sorted(names):
            paths = [maps[key][name] for key in ("image_a", "image_b", "mask")]
            sizes = []
            for path in paths:
                with Image.open(path) as image:
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError(f"Unaligned A/B/mask sizes for {split}/{name}: {sizes}")
            width, height = sizes[0]
            expected_size = self.settings["expected_image_size"]
            if expected_size is not None and (width, height) != (expected_size, expected_size):
                raise ValueError(f"{split}/{name}: expected {expected_size}x{expected_size}, got {sizes[0]}")
            if width % self.tile_size or height % self.tile_size:
                raise ValueError(f"{split}/{name} is not divisible by tile_size; no pixels may be dropped")
            index = len(self.records)
            self.records.append(PairRecord(name, *paths, (width, height)))
            self.tiles.extend((index, x, y) for y in range(0, height, self.tile_size)
                              for x in range(0, width, self.tile_size))

    def __len__(self) -> int:
        return len(self.tiles)

    def read_pair(self, record: PairRecord, box: tuple[int, int, int, int] | None = None) -> RawPair:
        return RawPair(_read_rgb(record.image_a, box), _read_rgb(record.image_b, box),
                       _read_mask(record.mask, self.settings["label_values"],
                                  self.settings["label_decode"], box))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, x, y = self.tiles[index]
        record = self.records[record_index]
        box = (x, y, x + self.tile_size, y + self.tile_size)
        pair = self.read_pair(record, box)
        if self.pair_transform is not None:
            pair = self.pair_transform(pair)
        shape = (self.tile_size, self.tile_size)
        if pair.image_a.shape != (*shape, 3) or pair.image_b.shape != (*shape, 3) or pair.mask.shape != shape:
            raise ValueError("Pair transform changed image/mask alignment")
        if pair.image_a.dtype != np.uint8 or pair.image_b.dtype != np.uint8:
            raise ValueError("Pair transform must return uint8 RGB images before normalization")
        if not np.isin(pair.mask, (0, 1)).all():
            raise ValueError("Pair transform must preserve binary labels")

        def normalize(image: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
            return (tensor - self.mean) / self.std

        return {
            "image_a": normalize(pair.image_a),
            "image_b": normalize(pair.image_b),
            "mask": torch.from_numpy(np.ascontiguousarray(pair.mask)).float().unsqueeze(0),
            "sample_id": f"{self.split}/{record.name}@x{x}_y{y}",
            "metadata": pair.metadata,
        }

    def manifest(self) -> dict[str, Any]:
        """Portable inventory fingerprint, NOT a claim of full content verification."""
        entries = [{"name": record.name, "size": list(record.size),
                    "bytes": [path.stat().st_size for path in (record.image_a, record.image_b, record.mask)]}
                   for record in self.records]
        payload = {"split": self.split, "tile_size": self.tile_size, "entries": entries}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {**payload, "source_pairs": len(self.records), "tiles": len(self), "inventory_sha256": digest}


def build_datasets(config: Config, splits: tuple[str, ...]) -> dict[str, LEVIRDataset]:
    datasets = {split: LEVIRDataset(config, split) for split in splits}
    seen: dict[Path, str] = {}
    for split, dataset in datasets.items():
        for record in dataset.records:
            for path in (record.image_a, record.image_b, record.mask):
                resolved = path.resolve()
                if resolved in seen:
                    raise ValueError(f"Aliased data path in {seen[resolved]} and {split}: {resolved}")
                seen[resolved] = split
    return datasets


def seed_worker(worker_id: int) -> None:
    """Top-level callable so Windows spawn workers can import it."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: LEVIRDataset, config: Config, *, training: bool) -> DataLoader:
    cfg = config.values
    if training and dataset.split != "train":
        raise ValueError("Only the training split may be shuffled for optimization")
    generator = torch.Generator()
    generator.manual_seed(cfg["experiment"]["seed"] + {"train": 0, "val": 1, "test": 2}[dataset.split])
    return DataLoader(dataset,
                      batch_size=cfg["train"]["batch_size" if training else "eval_batch_size"],
                      shuffle=training, num_workers=cfg["train"]["num_workers"],
                      pin_memory=cfg["runtime"]["device"].startswith("cuda"),
                      drop_last=False, worker_init_fn=seed_worker, generator=generator)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_data(config: Config, *, full: bool = False,
               splits: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Full audit decodes labels and detects identical image pairs across splits.

    A full audit also saves encoded-file SHA256 hashes. Duplicate checking uses
    decoded RGB pixels and ignores temporal order. Neither check proves geographic
    independence between different-looking neighboring scenes. Development
    configurations audit train/val only, without opening the test directory.
    """
    if splits is None:
        splits = ("train", "val", "test") if config.run_test_after_training else ("train", "val")
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Audit splits must be nonempty and unique")
    datasets = build_datasets(config, splits)
    reports = {split: dataset.manifest() for split, dataset in datasets.items()}
    if full:
        seen_pairs: dict[tuple[str, str], str] = {}
        for split, dataset in datasets.items():
            content = {}
            for record in dataset.records:
                pair = dataset.read_pair(record)
                image_hashes = tuple(sorted(hashlib.sha256(image.tobytes()).hexdigest()
                                            for image in (pair.image_a, pair.image_b)))
                if image_hashes in seen_pairs and seen_pairs[image_hashes] != split:
                    raise ValueError(f"Duplicate decoded image pair across {seen_pairs[image_hashes]} and {split}: {record.name}")
                seen_pairs[image_hashes] = split
                content[record.name] = {key: file_sha256(getattr(record, key))
                                        for key in ("image_a", "image_b", "mask")}
            reports[split]["file_sha256"] = content
    return {"audit_level": "decoded_and_sha256" if full else "headers_and_inventory",
            "label_decode": config.values["data"]["label_decode"],
            "splits": reports,
            "limitation": "Inventory counts alone do not verify official split identity or geographic independence."}
