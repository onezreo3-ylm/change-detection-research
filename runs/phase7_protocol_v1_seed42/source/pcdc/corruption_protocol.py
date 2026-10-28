"""Versioned, label-independent corruption instances; no model training here."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from .config import Config, UniqueKeyLoader, load_config, protocol_signature
from .data import LEVIRDataset, RawPair
from .perturbations import corrupt_rgb, make_injection_mask


KINDS = ("brightness", "contrast", "gamma", "color_bias", "haze", "shadow",
         "noise", "blur", "jpeg")
_KEYS = {"schema_version", "name", "status", "base_config", "reference_checkpoint",
         "reference_checkpoint_sha256", "reference_audit", "output_dir", "seeds",
         "placement", "spatial_modes", "local_mask", "calibration", "corruptions"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def keyed_seed(*parts: Any) -> int:
    """Portable SHA256-derived uint64, independent of Python hash/global RNG."""
    return int.from_bytes(hashlib.sha256(canonical(parts).encode("utf-8")).digest()[:8], "big")


def _keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Invalid {label} schema; expected {sorted(keys)}")


def _integer(value: Any, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


@dataclass(frozen=True)
class CorruptionProtocol:
    values: dict[str, Any]
    source: Path
    base: Config

    def path(self, key: str) -> Path:
        path = Path(self.values[key])
        return (path if path.is_absolute() else self.source.parent / path).resolve()

    @property
    def fingerprint(self) -> str:
        # Relocation is allowed; settings, seeds, base inference and reference
        # checkpoint content are not. Implementation hashes are recorded separately.
        settings = {k: v for k, v in self.values.items() if k not in (
            "base_config", "reference_checkpoint", "reference_audit", "output_dir")}
        return digest({"corruptions": settings,
                       "base_protocol_signature": protocol_signature(self.base.values),
                       "base_runtime": self.base.values["runtime"],
                       "base_seed": self.base.values["experiment"]["seed"],
                       "evaluation_batch_size": self.base.values["train"]["eval_batch_size"],
                       "num_workers": self.base.values["train"]["num_workers"]})


def load_protocol(path: str | Path) -> CorruptionProtocol:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        values = yaml.load(stream, Loader=UniqueKeyLoader)
    _keys(values, _KEYS, "corruption protocol")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only corruption schema_version 1 is supported")
    if values["status"] != "candidate_pending_review":
        raise ValueError("This preparation tool only emits candidate protocols, not accepted benchmarks")
    if values["placement"] != "after_tiling_before_normalization":
        raise ValueError("Only tile-level pre-normalization injection is supported")
    if values["spatial_modes"] != ["global", "local"]:
        raise ValueError("v1 requires equally represented [global, local] injection")
    for key in ("name", "base_config", "reference_checkpoint", "reference_audit", "output_dir"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    checksum = values["reference_checkpoint_sha256"]
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)):
        raise ValueError("reference_checkpoint_sha256 must be a lowercase SHA256")
    _keys(values["seeds"], {"train", "val", "test"}, "seeds")
    for key, seed in values["seeds"].items():
        _integer(seed, f"seeds.{key}")
    if len(set(values["seeds"].values())) != 3:
        raise ValueError("Train, validation and reserved test seeds must differ")
    mask = values["local_mask"]
    _keys(mask, {"center_x_range", "center_y_range", "radius_x_range", "radius_y_range", "feather"},
          "local_mask")
    for key, bounds in mask.items():
        if key == "feather":
            bounds = [bounds, bounds]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 <= bounds[0] <= bounds[1] <= 1):
            raise ValueError(f"Invalid normalized {key}")
        if (key.startswith("radius") or key == "feather") and bounds[0] == 0:
            raise ValueError(f"{key} must be positive")
    calibration = values["calibration"]
    _keys(calibration, {"source_pairs", "selection_seed", "preview_levels", "preview_tile_pixels"},
          "calibration")
    _integer(calibration["source_pairs"], "calibration.source_pairs", 1)
    _integer(calibration["selection_seed"], "calibration.selection_seed")
    _integer(calibration["preview_tile_pixels"], "calibration.preview_tile_pixels", 32)
    levels = calibration["preview_levels"]
    if (not isinstance(levels, list) or not levels
            or any(type(n) is not int or not 1 <= n <= 5 for n in levels)
            or len(set(levels)) != len(levels)):
        raise ValueError("preview_levels must be distinct integers within 1..5")
    _keys(values["corruptions"], set(KINDS), "corruptions")
    # Validate every operator's exact schema, including on identity/empty paths.
    probe = np.zeros((2, 2, 3), dtype=np.uint8)
    empty = np.zeros((2, 2), dtype=np.float32)
    for kind, directions in values["corruptions"].items():
        if not isinstance(directions, dict) or not directions:
            raise ValueError(f"{kind} must have named directions")
        for direction, parameters in directions.items():
            if not isinstance(direction, str) or not direction:
                raise ValueError("Direction names must be nonempty strings")
            if not isinstance(parameters, list) or len(parameters) != 5:
                raise ValueError(f"{kind}/{direction} needs exactly five severity levels")
            for parameter in parameters:
                corrupt_rgb(probe, kind, parameter, seed=0, mask=empty)
            if any(parameter == previous for i, parameter in enumerate(parameters)
                   for previous in parameters[:i]):
                raise ValueError(f"{kind}/{direction} has duplicate severity levels")
    base_path = Path(values["base_config"])
    base = load_config(base_path if base_path.is_absolute() else source.parent / base_path)
    if base.run_test_after_training:
        raise ValueError("Corruption development requires run_test_after_training: false")
    return CorruptionProtocol(values, source, base)


def sample_ids(dataset: LEVIRDataset) -> list[str]:
    return [f"{dataset.split}/{dataset.records[r].name}@x{x}_y{y}" for r, x, y in dataset.tiles]


def _mask_spec(protocol: CorruptionProtocol, seed: int, spatial: str) -> dict[str, Any]:
    if spatial == "global":
        return {"mode": "global"}
    if spatial != "local":
        raise ValueError("Unknown spatial mode")
    rng = np.random.Generator(np.random.PCG64(seed))
    ranges = protocol.values["local_mask"]
    return {"mode": "ellipse", "feather": ranges["feather"],
            **{key: float(rng.uniform(*ranges[f"{key}_range"]))
               for key in ("center_x", "center_y", "radius_x", "radius_y")}}


def make_instance(protocol: CorruptionProtocol, sample_id: str, kind: str, level: int, *,
                  stream: str, direction: str, target_date: str, spatial: str,
                  epoch: int = 0, view: int = 0) -> dict[str, Any]:
    if stream not in ("train", "val"):
        raise ValueError("Held-out test instances are reserved for a separately reviewed final protocol")
    if not isinstance(sample_id, str) or not sample_id.startswith(stream + "/"):
        raise ValueError("Instance sample must belong to its declared RNG/data stream")
    _integer(epoch, "epoch")
    _integer(view, "view")
    if stream != "train" and (epoch or view):
        raise ValueError("Validation instances cannot vary by training epoch/view")
    if type(level) is not int or not 1 <= level <= 5:
        raise ValueError("Severity level must be an integer in 1..5")
    if target_date not in ("A", "B"):
        raise ValueError("Exactly one of A/B must be targeted")
    if kind not in KINDS or direction not in protocol.values["corruptions"][kind]:
        raise ValueError("Unknown corruption or direction")
    # Exclude level: paired severity curves share the noise field/mask/direction.
    seed = keyed_seed(protocol.values["name"], protocol.values["seeds"][stream],
                      stream, sample_id, kind, direction, target_date, spatial, epoch, view)
    return {"sample_id": sample_id, "condition_id": f"{kind}_l{level}",
            "stream": stream, "epoch": epoch, "view": view,
            "kind": kind, "level": level, "direction": direction,
            "target_date": target_date, "spatial_mode": spatial, "seed": seed,
            "parameters": copy.deepcopy(protocol.values["corruptions"][kind][direction][level - 1]),
            "injection_mask": _mask_spec(protocol, keyed_seed(seed, "mask"), spatial)}


def validation_instances(protocol: CorruptionProtocol, ids: list[str]) -> Iterator[dict[str, Any]]:
    """45 cells, one instance per tile/cell, with explicit balanced nuisance strata.

    All dates/directions/spatial modes form a Cartesian product. Assignments are
    balanced to within one tile; for LEVIR's 1024 val tiles they balance exactly.
    The sample order never depends on image content, labels, or model predictions.
    """
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(s, str) or not s.startswith("val/") for s in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Validation requires unique nonempty val sample IDs")
    for kind in KINDS:
        directions = sorted(protocol.values["corruptions"][kind])
        strata = list(itertools.product(directions, ("A", "B"), ("global", "local")))
        order = sorted(ids, key=lambda s: (keyed_seed(protocol.values["seeds"]["val"], kind, s), s))
        assignment = {s: strata[index % len(strata)] for index, s in enumerate(order)}
        for level in range(1, 6):
            for sample_id in ids:
                direction, target, spatial = assignment[sample_id]
                yield make_instance(protocol, sample_id, kind, level, stream="val",
                                    direction=direction, target_date=target, spatial=spatial)


def sample_training_instance(protocol: CorruptionProtocol, sample_id: str, *,
                             epoch: int, view: int = 0) -> dict[str, Any]:
    """Future integration hook only: uniform family/level/direction/date/support.

    Stateless per sample/epoch/view, independent of worker scheduling. It is NOT
    attached to the Phase-1 loader or to a training experiment in this stage.
    """
    _integer(epoch, "epoch")
    _integer(view, "view")
    rng = np.random.Generator(np.random.PCG64(keyed_seed(
        protocol.values["seeds"]["train"], "training_selection", sample_id, epoch, view)))
    kind = KINDS[int(rng.integers(len(KINDS)))]
    directions = sorted(protocol.values["corruptions"][kind])
    return make_instance(protocol, sample_id, kind, int(rng.integers(1, 6)), stream="train",
                         direction=directions[int(rng.integers(len(directions)))],
                         target_date=("A", "B")[int(rng.integers(2))],
                         spatial=("global", "local")[int(rng.integers(2))], epoch=epoch, view=view)


def apply_instance(pair: RawPair, instance: dict[str, Any]) -> RawPair:
    """Apply one RGB-only action; semantic mask and other date are copied intact."""
    if (pair.image_a.dtype != np.uint8 or pair.image_b.dtype != np.uint8
            or pair.image_a.shape != pair.image_b.shape or pair.image_a.ndim != 3
            or pair.image_a.shape[2] != 3
            or pair.mask.shape != pair.image_a.shape[:2]
            or pair.mask.dtype != np.uint8 or not np.isin(pair.mask, (0, 1)).all()):
        raise ValueError("Expected an aligned RGB pair and decoded uint8 binary semantic mask")
    target = instance["target_date"]
    if target not in ("A", "B"):
        raise ValueError("Exactly one date must be targeted")
    alpha = make_injection_mask(pair.mask.shape, instance["injection_mask"])
    a, b = pair.image_a.copy(), pair.image_b.copy()
    transformed = corrupt_rgb(a if target == "A" else b, instance["kind"],
                              instance["parameters"], seed=instance["seed"], mask=alpha)
    if target == "A":
        a = transformed
    else:
        b = transformed
    return RawPair(a, b, pair.mask.copy(), {**copy.deepcopy(pair.metadata),
                                          "corruption_instance_json": canonical(instance)})
