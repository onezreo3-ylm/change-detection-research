"""Strict YAML configuration; all relative paths are config-relative."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys, which otherwise silently change experiment settings."""


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True)
class Config:
    values: dict[str, Any]
    source: Path

    def path(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else self.source.parent / candidate).resolve()

    @property
    def data_root(self) -> Path:
        return self.path(self.values["data"]["root"])

    @property
    def output_dir(self) -> Path:
        return self.path(self.values["experiment"]["output_dir"])


SCHEMA = {
    "experiment": {"name", "kind", "seed", "output_dir"},
    "runtime": {"device", "deterministic", "num_threads"},
    "data": {"root", "splits", "folders", "extensions", "expected_pairs",
             "expected_image_size", "tile_size", "label_values", "label_decode",
             "mean", "std"},
    "model": {"encoder_weights", "weights_cache", "decoder_channels", "normalization_groups"},
    "research": {"pse", "pcdc", "cpc"},
    "train": {"epochs", "batch_size", "eval_batch_size", "num_workers", "learning_rate",
              "weight_decay", "adam_betas", "min_learning_rate", "amp", "grad_clip_norm"},
    "loss": {"bce_weight", "dice_weight", "dice_smooth"},
    "evaluation": {"threshold", "checkpoint_metric"},
}


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label}: missing={expected - actual}, unexpected={actual - expected}")


def _number(value: Any, label: str, minimum: float = 0, integer: bool = False) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'} >= {minimum}")


def validate_config(cfg: dict[str, Any]) -> None:
    _keys(cfg, set(SCHEMA), "config")
    for section, keys in SCHEMA.items():
        _keys(cfg[section], keys, section)
    data, train, model = cfg["data"], cfg["train"], cfg["model"]
    for key in ("splits", "expected_pairs"):
        _keys(data[key], {"train", "val", "test"}, f"data.{key}")
    _keys(data["folders"], {"image_a", "image_b", "mask"}, "data.folders")
    for section, key in (("experiment", "seed"), ("train", "num_workers")):
        _number(cfg[section][key], f"{section}.{key}", integer=True)
    if cfg["experiment"]["seed"] >= 2**32:
        raise ValueError("experiment.seed must be < 2**32")
    for section, keys in (("train", ("epochs", "batch_size", "eval_batch_size")),
                          ("runtime", ("num_threads",)),
                          ("model", ("decoder_channels", "normalization_groups")),
                          ("data", ("tile_size",))):
        for key in keys:
            _number(cfg[section][key], f"{section}.{key}", 1, integer=True)
    if data["tile_size"] < 32:
        raise ValueError("data.tile_size must be >= 32 for ResNet-18")
    size = data["expected_image_size"]
    if size is not None:
        _number(size, "data.expected_image_size", 32, integer=True)
        if size % data["tile_size"]:
            raise ValueError("Image size must be divisible by tile_size; no silent border cropping")
    for split, count in data["expected_pairs"].items():
        if count is not None:
            _number(count, f"data.expected_pairs.{split}", 1, integer=True)
    if len(set(data["splits"].values())) != 3:
        raise ValueError("Train/val/test split directories must differ")
    if len(set(data["folders"].values())) != 3:
        raise ValueError("A/B/mask directories must differ")
    for key in ("splits", "folders"):
        for value in data[key].values():
            if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"data.{key} entries must be nonempty relative subdirectories")
    for value in (data["root"], cfg["experiment"]["output_dir"], model["weights_cache"], cfg["experiment"]["name"]):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Paths and experiment.name must be nonempty strings")
    if not isinstance(data["extensions"], list) or not data["extensions"] or any(
        not isinstance(ext, str) or not ext.startswith(".") for ext in data["extensions"]
    ):
        raise ValueError("data.extensions must be a nonempty list such as [.png]")
    for key in ("mean", "std"):
        if not isinstance(data[key], list) or len(data[key]) != 3:
            raise ValueError(f"data.{key} needs three RGB values")
        for value in data[key]:
            _number(value, f"data.{key}", 1e-12 if key == "std" else 0)
    labels = data["label_values"]
    if not isinstance(labels, list) or len(labels) != 2 or len(set(labels)) != 2 or any(
        type(v) is not int or not 0 <= v <= 255 for v in labels
    ):
        raise ValueError("data.label_values must contain distinct uint8 background and foreground")
    if data["label_decode"] not in ("strict_values", "author_uint8_floor255"):
        raise ValueError("data.label_decode must be strict_values or author_uint8_floor255")
    if data["label_decode"] == "author_uint8_floor255" and labels != [0, 255]:
        raise ValueError("author_uint8_floor255 requires data.label_values: [0, 255]")
    if model["encoder_weights"] not in (None, "IMAGENET1K_V1"):
        raise ValueError("model.encoder_weights must be null or IMAGENET1K_V1")
    if model["decoder_channels"] % (4 * model["normalization_groups"]):
        raise ValueError("decoder_channels must be divisible by 4 * normalization_groups")
    if cfg["experiment"]["kind"] not in ("research", "smoke"):
        raise ValueError("experiment.kind must be research or smoke")
    device = cfg["runtime"]["device"]
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or
        (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("runtime.device must be cpu, cuda, or cuda:N")
    for section, key in (("runtime", "deterministic"), ("train", "amp")):
        if type(cfg[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be boolean")
    if train["amp"] and device == "cpu":
        raise ValueError("AMP is CUDA-only in this baseline; do not silently disable it")
    for key, enabled in cfg["research"].items():
        if type(enabled) is not bool:
            raise ValueError(f"research.{key} must be boolean")
        if enabled:
            raise NotImplementedError(f"{key} is outside Phase 1 and is not implemented")
    for key in ("learning_rate", "min_learning_rate", "weight_decay"):
        _number(train[key], f"train.{key}", 1e-12 if key == "learning_rate" else 0)
    if train["min_learning_rate"] > train["learning_rate"]:
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    if train["grad_clip_norm"] is not None:
        _number(train["grad_clip_norm"], "train.grad_clip_norm", 1e-12)
    betas = train["adam_betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("train.adam_betas requires two values")
    for beta in betas:
        _number(beta, "train.adam_betas")
        if beta >= 1:
            raise ValueError("Adam betas must be < 1")
    for key, value in cfg["loss"].items():
        _number(value, f"loss.{key}", 1e-12 if key == "dice_smooth" else 0)
    if cfg["loss"]["bce_weight"] + cfg["loss"]["dice_weight"] <= 0:
        raise ValueError("At least one change-detection loss must have positive weight")
    _number(cfg["evaluation"]["threshold"], "evaluation.threshold", 1e-12)
    if cfg["evaluation"]["threshold"] >= 1:
        raise ValueError("evaluation.threshold must be strictly between 0 and 1")
    if cfg["evaluation"]["checkpoint_metric"] != "f1":
        raise ValueError("Phase 1 selects checkpoints by validation f1 only")


def load_config(path: str | Path) -> Config:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        cfg = yaml.load(stream, Loader=UniqueKeyLoader)
    # Backward-compatible reading for archived Phase 1 configs created before
    # the decoder was named. Their behavior was strict exact-value decoding.
    if isinstance(cfg, dict) and isinstance(cfg.get("data"), dict):
        cfg["data"].setdefault("label_decode", "strict_values")
    validate_config(cfg)
    return Config(cfg, source)


def protocol_signature(cfg: dict[str, Any]) -> str:
    """Hash inference/metric semantics while permitting data/output relocation."""
    protocol = copy.deepcopy({key: cfg[key] for key in ("data", "model", "research", "loss", "evaluation")})
    protocol["data"].pop("root")
    # Preserve the signature of earlier archived strict-decoding checkpoints.
    if protocol["data"].get("label_decode") == "strict_values":
        protocol["data"].pop("label_decode")
    protocol["model"].pop("weights_cache")
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
