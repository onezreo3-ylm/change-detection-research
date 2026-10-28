"""Deterministic, non-geometric RGB corruptions before model normalization.

The injection mask is an appearance-edit alpha, never a semantic label or a
causal feature-damage target. Every operator preserves the input pixel grid.
Float64 transforms are blended before one final clip/round to uint8; JPEG's
codec quantization is inherent to the explicitly defined JPEG operator.
Severity values and sample/seed selection belong to the calling YAML protocol.
"""

from __future__ import annotations

import io
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from PIL import Image


_PARAMETER_KEYS = {
    "none": set(),
    "brightness": {"factor"},
    "contrast": {"factor", "pivot"},
    "gamma": {"exponent"},
    "color_bias": {"offset_rgb"},
    "haze": {"opacity", "airlight_rgb"},
    "shadow": {"attenuation"},
    "noise": {"sigma"},
    "blur": {"radius"},
    "jpeg": {"quality", "subsampling"},
}


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)}, "
            f"extra={sorted(map(repr, extra))}"
        )


def _real(value: Any, name: str, minimum: float | None = None,
          maximum: float | None = None, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _integer(value: Any, name: str, minimum: int,
             maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _rgb(value: Any, name: str, minimum: float, maximum: float) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three RGB values")
    return tuple(_real(item, f"{name}[{index}]", minimum, maximum)
                 for index, item in enumerate(value))


def _parameters(kind: str, parameters: dict) -> dict[str, Any]:
    if not isinstance(kind, str) or kind not in _PARAMETER_KEYS:
        raise ValueError(f"Unsupported corruption kind: {kind!r}")
    _keys(parameters, _PARAMETER_KEYS[kind], f"{kind} parameters")
    if kind == "none":
        return {}
    if kind == "brightness":
        return {"factor": _real(parameters["factor"], "factor", positive=True)}
    if kind == "contrast":
        pivot = _real(parameters["pivot"], "pivot")
        if pivot != 0.5:
            raise ValueError("contrast pivot must be exactly 0.5")
        return {"factor": _real(parameters["factor"], "factor", 0), "pivot": pivot}
    if kind == "gamma":
        return {"exponent": _real(parameters["exponent"], "exponent", positive=True)}
    if kind == "color_bias":
        return {"offset_rgb": _rgb(parameters["offset_rgb"], "offset_rgb", -1, 1)}
    if kind == "haze":
        return {"opacity": _real(parameters["opacity"], "opacity", 0, 1),
                "airlight_rgb": _rgb(parameters["airlight_rgb"], "airlight_rgb", 0, 1)}
    if kind == "shadow":
        return {"attenuation": _real(parameters["attenuation"], "attenuation", 0, 1)}
    if kind == "noise":
        return {"sigma": _real(parameters["sigma"], "sigma", 0, 1)}
    if kind == "blur":
        return {"radius": _real(parameters["radius"], "radius", 0)}
    return {"quality": _integer(parameters["quality"], "quality", 1, 100),
            "subsampling": _integer(parameters["subsampling"], "subsampling", 0, 2)}


def _neutral(kind: str, parameters: dict[str, Any]) -> bool:
    if kind == "none":
        return True
    if kind in ("brightness", "contrast"):
        return parameters["factor"] == 1
    if kind == "gamma":
        return parameters["exponent"] == 1
    if kind == "color_bias":
        return all(value == 0 for value in parameters["offset_rgb"])
    key = {"haze": "opacity", "shadow": "attenuation",
           "noise": "sigma", "blur": "radius"}.get(kind)
    # JPEG quality=100 is still a codec round trip, not an identity operator.
    return key is not None and parameters[key] == 0


def _alpha(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=np.float64)
    if not isinstance(mask, np.ndarray) or mask.shape != shape:
        raise ValueError(f"injection mask must be a NumPy array with shape {shape}")
    if not np.issubdtype(mask.dtype, np.floating):
        raise ValueError("injection mask must have floating dtype, not semantic label dtype")
    if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("injection mask must contain finite values in [0, 1]")
    return mask.astype(np.float64, copy=False)


def _gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Separable float64 Gaussian, sigma=radius pixels, ceil(3*sigma) support.

    Each axis uses NumPy ``reflect`` padding (edge pixel is not repeated).
    Radius zero is handled as an exact identity before reaching this function.
    No spatial resampling or intermediate uint8 conversion is performed.
    """
    if not math.isfinite(3 * radius):
        raise ValueError("blur radius is too large to construct a finite kernel")
    half_width = math.ceil(3 * radius)
    positions = np.arange(-half_width, half_width + 1, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-0.5 * (positions / radius) ** 2)
    weights /= weights.sum()
    result = image
    for axis in (0, 1):
        padding = [(0, 0)] * image.ndim
        padding[axis] = (half_width, half_width)
        padded = np.pad(result, padding, mode="reflect")
        filtered = np.zeros_like(result)
        for index, weight in enumerate(weights):
            selection = [slice(None)] * image.ndim
            selection[axis] = slice(index, index + image.shape[axis])
            filtered += weight * padded[tuple(selection)]
        result = filtered
    return result


def _jpeg(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    # Constructing an image from the RGB array drops all source EXIF/ICC metadata.
    with io.BytesIO() as buffer:
        Image.fromarray(image).save(
            buffer, format="JPEG", quality=parameters["quality"],
            subsampling=parameters["subsampling"], optimize=False, progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.array(decoded.convert("RGB"), dtype=np.float64) / 255.0


def corrupt_rgb(image: np.ndarray, kind: str, parameters: dict, *, seed: int,
                mask: np.ndarray | None = None) -> np.ndarray:
    """Return a new uint8 HWC RGB image without changing input or global RNGs.

    ``mask=None`` means full-image support. A supplied float HxW mask blends the
    corrupted RGB with the original; pixels with alpha zero remain bitwise equal.
    Parameters are exact-schema dictionaries, including the fixed contrast pivot
    0.5 and explicit JPEG subsampling. Seed is a nonnegative integer; only noise
    consumes it, through a local NumPy Generator(PCG64(seed)). No operator consults
    semantic labels. Even identity paths validate all arguments first.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] == 0 or image.shape[1] == 0):
        raise ValueError("image must be a nonempty uint8 HWC RGB NumPy array")
    settings = _parameters(kind, parameters)
    checked_seed = _integer(seed, "seed", 0)
    alpha = _alpha(mask, image.shape[:2])
    if _neutral(kind, settings) or not np.any(alpha):
        return image.copy()

    original = image.astype(np.float64) / 255.0
    if kind == "brightness":
        transformed = original * settings["factor"]
    elif kind == "contrast":
        transformed = (original - settings["pivot"]) * settings["factor"] + settings["pivot"]
    elif kind == "gamma":
        transformed = original ** settings["exponent"]
    elif kind == "color_bias":
        transformed = original + np.array(settings["offset_rgb"], dtype=np.float64)
    elif kind == "haze":
        transformed = ((1 - settings["opacity"]) * original
                       + settings["opacity"] * np.array(settings["airlight_rgb"], dtype=np.float64))
    elif kind == "shadow":
        transformed = original * (1 - settings["attenuation"])
    elif kind == "noise":
        generator = np.random.Generator(np.random.PCG64(checked_seed))
        transformed = original + generator.normal(0.0, settings["sigma"], size=image.shape)
    elif kind == "blur":
        transformed = _gaussian_blur(original, settings["radius"])
    else:
        transformed = _jpeg(image, settings)

    blended = original * (1 - alpha[..., None]) + transformed * alpha[..., None]
    result = np.rint(np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
    result[alpha == 0] = image[alpha == 0]
    return result


def make_injection_mask(shape: tuple[int, int], spec: dict) -> np.ndarray:
    """Build a float32 alpha from geometry alone, independent of image/labels.

    ``global`` and ``empty`` accept only their mode. ``ellipse`` additionally
    requires center_x/y, radius_x/y, and feather. Centers and radii are normalized
    to image width/height; radii and feather are positive and at most one. Pixel
    centers have coordinates (column+0.5)/width and (row+0.5)/height. With elliptical
    distance r, alpha=clip((1-r)/feather, 0, 1); outside support is exactly zero.
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be an (height, width) tuple")
    height = _integer(shape[0], "height", 1)
    width = _integer(shape[1], "width", 1)
    if not isinstance(spec, dict) or not isinstance(spec.get("mode"), str):
        raise ValueError("mask specification must be a dictionary with a string mode")
    mode = spec["mode"]
    if mode in ("global", "empty"):
        _keys(spec, {"mode"}, f"{mode} mask")
        return np.full((height, width), 1 if mode == "global" else 0, dtype=np.float32)
    if mode != "ellipse":
        raise ValueError(f"Unsupported injection mask mode: {mode!r}")
    _keys(spec, {"mode", "center_x", "center_y", "radius_x", "radius_y", "feather"},
          "ellipse mask")
    center_x = _real(spec["center_x"], "center_x", 0, 1)
    center_y = _real(spec["center_y"], "center_y", 0, 1)
    radius_x = _real(spec["radius_x"], "radius_x", maximum=1, positive=True)
    radius_y = _real(spec["radius_y"], "radius_y", maximum=1, positive=True)
    feather = _real(spec["feather"], "feather", maximum=1, positive=True)
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    with np.errstate(over="ignore"):
        distance = np.hypot((x[None, :] - center_x) / radius_x,
                            (y[:, None] - center_y) / radius_y)
        alpha = np.clip((1 - distance) / feather, 0.0, 1.0)
    return alpha.astype(np.float32)
