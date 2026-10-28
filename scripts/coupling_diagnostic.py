"""Pure P0 coupling diagnostics; no I/O, model execution, or data access.

Four-cell effects are descriptive finite differences, not causal damage maps.
All task metrics pool integer pixel counts; zero denominators return zero.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from numbers import Integral
from typing import Iterable, Mapping

import numpy as np
import torch

CELL_NAMES = ("uu", "uv", "vu", "vv")
COUNT_NAMES = ("tn", "fp", "fn", "tp")
TRANSITION_NAMES = ("tp_to_fn", "fn_to_tp", "tn_to_fp", "fp_to_tn",
                    "clean_tp", "clean_fn", "clean_tn", "clean_fp")


def _natural(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def area_stratum(foreground_pixels, area_boundaries=(64, 256, 1024, 4096)):
    n = _natural(foreground_pixels, "foreground_pixels")
    boundaries = [_natural(v, "boundary") for v in area_boundaries]
    if not boundaries or boundaries[0] == 0 or boundaries != sorted(set(boundaries)):
        raise ValueError("area boundaries must be strictly increasing positive integers")
    return "empty" if n == 0 else f"positive_{1 + sum(n > v for v in boundaries)}"


def select_train_samples(records, *, seed, positive_quotas=(4, 3, 3, 3, 3),
                         empty_count=48, area_boundaries=(64, 256, 1024, 4096)):
    """Hash-rank within strata, then fill positive shortfalls globally by hash.

    Requires explicit train provenance, unique IDs, and enough positive/empty
    tiles. Never replaces missing positive samples by empty samples or vice versa.
    Records retain caller metadata; the selection hash binds IDs, areas, ranks,
    strata and protocol, independent of input ordering.
    """
    area_stratum(0, area_boundaries)
    quotas = [_natural(q, "quota") for q in positive_quotas]
    empty_count = _natural(empty_count, "empty_count")
    if len(quotas) != len(area_boundaries) + 1:
        raise ValueError("one positive quota required per area stratum")
    rows, seen = [], set()
    for original in records:
        row = dict(original)
        sid = row.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("sample IDs must be unique nonempty strings")
        if row.get("split") != "train":
            raise ValueError("selection permits explicit train records only")
        seen.add(sid)
        row["area_stratum"] = area_stratum(row["foreground_pixels"], area_boundaries)
        row["selection_rank"] = hashlib.sha256(json.dumps(
            ["p0-train-selection-v1", seed, sid], ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest()
        rows.append(row)
    rows.sort(key=lambda r: (r["selection_rank"], r["id"]))
    empty = [r for r in rows if r["area_stratum"] == "empty"]
    positive = [r for r in rows if r["area_stratum"] != "empty"]
    if len(empty) < empty_count or len(positive) < sum(quotas):
        raise ValueError("insufficient train positive or empty samples; no substitution")
    chosen = empty[:empty_count]
    deficits = {}
    for index, quota in enumerate(quotas, 1):
        name = f"positive_{index}"
        selected = [r for r in positive if r["area_stratum"] == name][:quota]
        chosen.extend(selected)
        deficits[name] = quota - len(selected)
    chosen_ids = {r["id"] for r in chosen}
    fill = [r for r in positive if r["id"] not in chosen_ids][:sum(deficits.values())]
    chosen.extend(fill)
    chosen.sort(key=lambda r: (r["selection_rank"], r["id"]))
    protocol = {"seed": seed, "positive_quotas": quotas, "empty_count": empty_count,
                "area_boundaries": list(area_boundaries),
                "shortfall_rule": "remaining_positive_global_hash_order"}
    binding = {"protocol": protocol, "samples": [
        {k: r[k] for k in ("id", "foreground_pixels", "area_stratum", "selection_rank")}
        for r in chosen]}
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"samples": chosen, "selection_sha256": digest, "protocol": protocol,
            "strata": dict(Counter(r["area_stratum"] for r in chosen)),
            "initial_deficits": deficits, "fill_ids": [r["id"] for r in fill]}


def pool_counts(rows: Iterable[Mapping[str, int]]):
    result = dict.fromkeys(COUNT_NAMES, 0)
    for row in rows:
        for key in COUNT_NAMES:
            result[key] += _natural(row[key], key)
    return result


def metrics_from_counts(counts):
    c = pool_counts([counts])
    tn, fp, fn, tp = (c[k] for k in COUNT_NAMES)
    divide = lambda a, b: a / b if b else 0.0
    return {"precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "f1": divide(2 * tp, 2 * tp + fp + fn), "iou": divide(tp, tp + fp + fn),
            "oa": divide(tn + tp, tn + fp + fn + tp), "fpr": divide(fp, tn + fp),
            "pixels": tn + fp + fn + tp}


def pool_transitions(rows):
    result = dict.fromkeys(TRANSITION_NAMES, 0)
    for row in rows:
        for key in TRANSITION_NAMES:
            result[key] += _natural(row[key], key)
    return result


def _prediction(logits, target, threshold):
    if logits.shape != target.shape or logits.numel() == 0:
        raise ValueError("logit/target shapes must agree and be nonempty")
    if not torch.isfinite(logits).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("finite logits and binary targets required")
    return logits.float().sigmoid() >= threshold


def _counts(pred, target):
    truth = target.bool()
    return dict(zip(COUNT_NAMES, (int(m.sum().item()) for m in
        (~truth & ~pred, ~truth & pred, truth & ~pred, truth & pred))))


@torch.no_grad()
def four_cell_diagnostic(clean_logits, logits, target, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one")
    if set(logits) != set(CELL_NAMES):
        raise ValueError("exactly uu, uv, vu, vv logits required")
    clean = _prediction(clean_logits, target, threshold)
    clean_counts = _counts(clean, target)
    truth, cells = target.bool(), {}
    for name in CELL_NAMES:
        pred = _prediction(logits[name], target, threshold)
        transitions = dict(zip(TRANSITION_NAMES[:4], (int(m.sum().item()) for m in (
            truth & clean & ~pred, truth & ~clean & pred,
            ~truth & ~clean & pred, ~truth & clean & ~pred))))
        transitions.update({f"clean_{k}": clean_counts[k] for k in ("tp", "fn", "tn", "fp")})
        counts = _counts(pred, target)
        cells[name] = {"counts": counts, "metrics": metrics_from_counts(counts),
                       "transitions": transitions}
    groups = {}
    for group, names in (("D", ("uu", "vv")), ("O", ("uv", "vu"))):
        counts = pool_counts(cells[k]["counts"] for k in names)
        groups[group] = {"counts": counts, "metrics": metrics_from_counts(counts),
                         "transitions": pool_transitions(cells[k]["transitions"] for k in names)}
    uu, uv, vu, vv = (logits[k].detach().double() for k in CELL_NAMES)
    effects = {"m": (uu + uv + vu + vv) / 4, "a": (uu + uv - vu - vv) / 4,
               "b": (uu - uv + vu - vv) / 4, "i": (uu - uv - vu + vv) / 4}
    def describe(values):
        if not values.numel():
            return {"pixels": 0, "sum": 0.0, "abs_sum": 0.0, "square_sum": 0.0,
                    "mean": None, "mean_abs": None, "rms": None}
        return {"pixels": values.numel(), "sum": values.sum().item(),
                "abs_sum": values.abs().sum().item(), "square_sum": values.square().sum().item(),
                "mean": values.mean().item(), "mean_abs": values.abs().mean().item(),
                "rms": values.square().mean().sqrt().item()}
    return {"clean": {"counts": clean_counts, "metrics": metrics_from_counts(clean_counts)},
            "cells": cells, "groups": groups,
            "D_minus_O": {k: groups["D"]["metrics"][k] - groups["O"]["metrics"][k]
                          for k in ("f1", "recall", "fpr")},
            "logit_effects": {k: {"all": describe(v), "foreground": describe(v[truth]),
                                   "background": describe(v[~truth])} for k, v in effects.items()}}


def image_sha256(image):
    """Hash dimensions, dtype and exact unchanged RGB bytes (not a PNG codec)."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HWC uint8 RGB")
    digest = hashlib.sha256(json.dumps([list(image.shape), str(image.dtype)]).encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def validate_marginal_hashes(cell_hashes):
    """Validate actual per-cell A/B hashes, allowing recorded duplicate instances."""
    if set(cell_hashes) != set(CELL_NAMES):
        raise ValueError("exactly four cell hashes required")
    for hashes in cell_hashes.values():
        if len(hashes) != 2 or any(not isinstance(h, str) or len(h) != 64 or
                any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("each cell requires two SHA256 hex hashes")
    uu, uv, vu, vv = (cell_hashes[k] for k in CELL_NAMES)
    if not (uu[0] == uv[0] and vu[0] == vv[0] and uu[1] == vu[1] and uv[1] == vv[1]):
        raise ValueError("four-cell reuse identity mismatch")
    for date in (0, 1):
        if Counter((uu[date], vv[date])) != Counter((uv[date], vu[date])):
            raise ValueError("D/O marginal hashes mismatch")
    return {"matched": True, "a_instances_identical": uu[0] == vu[0],
            "b_instances_identical": uu[1] == uv[1],
            "fully_degenerate": uu[0] == vu[0] and uu[1] == uv[1]}
