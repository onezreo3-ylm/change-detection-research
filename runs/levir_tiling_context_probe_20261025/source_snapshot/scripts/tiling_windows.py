"""Window geometry, probability mosaics and paired counts; no model or data I/O."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    size: int

    @property
    def box(self):
        return self.x, self.y, self.x + self.size, self.y + self.size


def window_plan(arm, side):
    size, origins = arm["window_size"], arm["origins_yx_1d"]
    if type(side) is not int or type(size) is not int or not 0 < size <= side:
        raise ValueError("Window and source sizes must be positive integers")
    if (not isinstance(origins, (list, tuple)) or not origins
            or any(type(v) is not int or v < 0 or v + size > side for v in origins)
            or list(origins) != sorted(set(origins))):
        raise ValueError("Window origins must be sorted, unique and inside the source")
    windows = tuple(Window(x, y, size) for y, x in product(origins, repeat=2))
    coverage = np.zeros((side, side), dtype=np.uint16)
    for w in windows:
        coverage[w.y:w.y + size, w.x:w.x + size] += 1
    support = tuple(arm.get("support_xyxy_exclusive", (0, 0, side, side)))
    allowed = support_mask(support, side)
    if not np.array_equal(coverage > 0, allowed):
        raise ValueError("Actual window coverage differs from declared support")
    aggregation = arm["aggregation"]
    if aggregation not in ("none_unique_coverage", "uniform_arithmetic_mean_probability_float32_before_threshold"):
        raise ValueError("Unsupported aggregation; probability averaging must precede thresholding")
    if aggregation == "none_unique_coverage" and np.any(coverage[allowed] != 1):
        raise ValueError("Unique-coverage arm contains overlapping windows")
    return windows, coverage, support


def support_mask(box, side):
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or not 0 <= box[0] < box[2] <= side or not 0 <= box[1] < box[3] <= side):
        raise ValueError("Expected a nonempty absolute xyxy support inside the source")
    result = np.zeros((side, side), dtype=bool)
    result[box[1]:box[3], box[0]:box[2]] = True
    return result


def aligned_crops(a, b, window):
    if (a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3
            or a.dtype != np.uint8 or b.dtype != np.uint8):
        raise ValueError("Expected aligned uint8 RGB dates")
    x, y, right, bottom = window.box
    if x < 0 or y < 0 or right > a.shape[1] or bottom > a.shape[0]:
        raise ValueError("No out-of-source padding, resizing or clipped crop is allowed")
    return a[y:bottom, x:right], b[y:bottom, x:right]


class ProbabilityMosaic:
    def __init__(self, arm, side):
        windows, self.expected_coverage, self.support = window_plan(arm, side)
        self.expected = frozenset(windows)
        self.seen = set()
        self.total = np.zeros((side, side), dtype=np.float32)
        self.coverage = np.zeros((side, side), dtype=np.uint16)

    def add(self, window, probability):
        p = np.asarray(probability)
        if window not in self.expected or window in self.seen:
            raise ValueError("Unexpected or duplicate window")
        if (p.shape != (window.size, window.size) or not np.issubdtype(p.dtype, np.floating)
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
            raise ValueError("Expected finite unthresholded probabilities in [0,1]")
        sl = np.s_[window.y:window.y + window.size, window.x:window.x + window.size]
        self.total[sl] += p.astype(np.float32)
        self.coverage[sl] += 1
        self.seen.add(window)

    def finish(self):
        if self.seen != self.expected or not np.array_equal(self.coverage, self.expected_coverage):
            raise ValueError("Incomplete or inconsistent probability coverage")
        # Uncovered pixels stay undefined; they must never be scored as background.
        result = np.full(self.total.shape, np.nan, dtype=np.float32)
        np.divide(self.total, self.coverage, out=result, where=self.coverage > 0)
        if not np.isfinite(result[self.coverage > 0]).all():
            raise ValueError("Nonfinite fused probability")
        return result, self.coverage.copy()


def _selected(array, support):
    a = np.asarray(array)
    if a.ndim != 2 or a.shape != support.shape or support.dtype != bool or not support.any():
        raise ValueError("Map and nonempty scoring support must align")
    return a[support]


def confusion(label, predicted, support):
    a, b = _selected(label, support), _selected(predicted, support)
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Expected binary labels and predictions")
    a, b = a.astype(bool), b.astype(bool)
    return {"tp": int(np.count_nonzero(a & b)), "fp": int(np.count_nonzero(~a & b)),
            "fn": int(np.count_nonzero(a & ~b)), "tn": int(np.count_nonzero(~a & ~b))}


def count_metrics(counts):
    if set(counts) != {"tp", "fp", "fn", "tn"} or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, fn, tn = (counts[k] for k in ["tp", "fp", "fn", "tn"])
    def divide(a, b):
        return a / b if b else None
    return {"counts": counts, "f1": divide(2 * tp, 2 * tp + fp + fn),
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "fpr": divide(fp, fp + tn), "iou": divide(tp, tp + fp + fn),
            "oa": divide(tp + tn, tp + fp + fn + tn)}


def score_probability(label, probability, support, threshold=0.5):
    p = _selected(probability, support)
    if not 0 < threshold < 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Scored probabilities and threshold must be valid")
    return count_metrics(confusion(label, np.asarray(probability) >= threshold, support))


def compare_probability(reference, candidate, support, threshold=0.5):
    a, b = _selected(reference, support), _selected(candidate, support)
    if (not 0 < threshold < 1 or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1))):
        raise ValueError("Comparison cannot include missing or invalid probabilities")
    delta = np.abs(a - b)
    changed = int(np.count_nonzero((a >= threshold) != (b >= threshold)))
    return {"pixels": int(a.size), "binary_disagreement_pixels": changed,
            "binary_disagreement_fraction": changed / a.size,
            "probability_absolute_difference_mean": float(delta.mean(dtype=np.float64)),
            "probability_absolute_difference_p99": float(np.quantile(delta, 0.99)),
            "probability_absolute_difference_max": float(delta.max())}


def require_archive_replay(probability, label, archived_binary, archived_counts, threshold=0.5):
    full = np.ones(np.shape(label), dtype=bool)
    score = score_probability(label, probability, full, threshold)
    if (not np.array_equal(probability >= threshold, archived_binary)
            or score["counts"] != archived_counts):
        raise ValueError("Archived binary map/count replay failed; no tolerance or threshold fallback")
    return score


def window_resource_gate(rows, models, sources, rules):
    expected = {(m, s) for m in models for s in sources}
    if len(rows) != len(expected) or {(r["model"], r["source"]) for r in rows} != expected:
        raise ValueError("Resource gate requires all model/source comparisons exactly once")
    result = {}
    for model in models:
        details = []
        for r in rows:
            if r["model"] != model:
                continue
            signal, numeric = r["alternate_disagreement_fraction"], r["batch_disagreement_fraction"]
            if not np.isfinite([signal, numeric]).all() or not 0 <= min(signal, numeric) <= max(signal, numeric) <= 1:
                raise ValueError("Disagreement fractions must lie in [0,1]")
            minimum = max(rules["alternate_grid_disagreement_minimum_fraction"],
                          rules["numeric_control_multiple"] * numeric)
            details.append({"source": r["source"], "minimum": minimum,
                            "measured": signal, "qualified": signal >= minimum})
        n = sum(r["qualified"] for r in details)
        result[model] = {"qualifying_sources": n,
                         "passed": n >= rules["qualifying_source_count_minimum"], "sources": details}
    return {"models": result, "passed": all(v["passed"] for v in result.values()),
            "new_training_admitted": False, "causal_or_method_effect_claim": False}
