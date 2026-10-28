"""Matched C1/G1/G2/G3 selector training and calibration for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import file_sha256
from .phase6_diagnostic import COUNT_KEYS, _condition_order, _metrics_from_counts
from .phase7_cache import _load_npy, _read_jsonl, load_descriptor_cache
from .phase7_models import build_phase7_selector, selector_initial_sha256
from .phase7_protocol import Phase7Config, load_selector_split
from .runtime import environment_record, save_checkpoint, setup_runtime, write_json


VARIANTS = ("c1", "g1", "g2", "g3")


def _sha(path: Path) -> str:
    return file_sha256(path)


def _hash_order(length: int, epochs: int, seed: int) -> str:
    digest = hashlib.sha256()
    for epoch in range(1, epochs + 1):
        generator = np.random.Generator(np.random.PCG64(_epoch_seed(seed, epoch)))
        digest.update(generator.permutation(length).astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _epoch_seed(seed: int, epoch: int) -> int:
    payload = f"phase7_selector_order_v1_seed{seed}\\0{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _atomic_copy_checkpoint(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def phase7_training_source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pcdc/config.py", "pcdc/data.py", "pcdc/phase7_protocol.py",
        "pcdc/phase7_models.py", "pcdc/phase7_cache.py",
        "pcdc/phase7_training.py", "pcdc/runtime.py",
    )
    return {name: root / name for name in names}


@dataclass
class SelectorArrays:
    fit_perturbed: np.ndarray
    fit_clean: np.ndarray
    fit_clean_index: np.ndarray
    fit_u_t: np.ndarray
    fit_u_0: np.ndarray
    fit_a: np.ndarray
    calibration_perturbed: np.ndarray
    calibration_clean: np.ndarray
    calibration_clean_index: np.ndarray
    calibration_u_t: np.ndarray
    calibration_u_0: np.ndarray
    calibration_a: np.ndarray
    calibration_condition: np.ndarray
    calibration_a0_counts: np.ndarray
    calibration_a1_counts: np.ndarray
    calibration_clean_u0: np.ndarray
    calibration_clean_a0_counts: np.ndarray
    calibration_clean_a1_counts: np.ndarray


def _counts(row: Mapping[str, Any], key: str) -> list[int]:
    value = row[key]
    if not isinstance(value, dict) or set(value) != set(COUNT_KEYS):
        raise ValueError(f"Invalid action counts: {key}")
    return [int(value[name]) for name in COUNT_KEYS]


def load_selector_arrays(config: Phase7Config) -> SelectorArrays:
    cache = load_descriptor_cache(config)
    output, status = cache["output"], cache["status"]
    clean_info = status["clean"]
    clean_x = np.asarray(_load_npy(
        output / clean_info["descriptor_path"], (1424, 1024)), dtype=np.float32)
    clean_rows = _read_jsonl(output / clean_info["records_path"])
    partitions = [row["phase7_partition"] for row in clean_rows]
    fit_clean_global = [index for index, part in enumerate(partitions) if part == "gate_fit"]
    cal_clean_global = [index for index, part in enumerate(partitions)
                        if part == "gate_calibration"]
    if len(fit_clean_global) != 1136 or len(cal_clean_global) != 288:
        raise ValueError("Clean selector cache partition counts differ")
    fit_global_to_local = {global_index: local for local, global_index in enumerate(
        fit_clean_global)}
    cal_global_to_local = {global_index: local for local, global_index in enumerate(
        cal_clean_global)}
    sample_to_global = {row["sample_id"]: index for index, row in enumerate(clean_rows)}
    if len(sample_to_global) != 1424:
        raise ValueError("Clean selector sample IDs are not unique")

    fit_x: list[np.ndarray] = []
    cal_x: list[np.ndarray] = []
    fit_clean_index: list[int] = []
    cal_clean_index: list[int] = []
    fit_u_t: list[float] = []
    fit_u_0: list[float] = []
    fit_a: list[float] = []
    cal_u_t: list[float] = []
    cal_u_0: list[float] = []
    cal_a: list[float] = []
    cal_condition: list[int] = []
    cal_a0_counts: list[list[int]] = []
    cal_a1_counts: list[list[int]] = []
    for condition_index, condition in enumerate(_condition_order()):
        shard = status["condition_shards"][condition]
        descriptor = np.asarray(_load_npy(
            output / shard["descriptor_path"], (1424, 1024)), dtype=np.float32)
        rows = _read_jsonl(output / shard["records_path"])
        fit_indices = [index for index, row in enumerate(rows)
                       if row["phase7_partition"] == "gate_fit"]
        cal_indices = [index for index, row in enumerate(rows)
                       if row["phase7_partition"] == "gate_calibration"]
        if len(fit_indices) != 1136 or len(cal_indices) != 288:
            raise ValueError(f"Selector shard partition counts differ: {condition}")
        fit_x.append(descriptor[fit_indices])
        cal_x.append(descriptor[cal_indices])
        for index in fit_indices:
            row = rows[index]
            global_index = sample_to_global[row["sample_id"]]
            fit_clean_index.append(fit_global_to_local[global_index])
            fit_u_t.append(float(row["uT"]))
            fit_u_0.append(float(row["u0"]))
            fit_a.append(float(row["a"]))
        for index in cal_indices:
            row = rows[index]
            global_index = sample_to_global[row["sample_id"]]
            cal_clean_index.append(cal_global_to_local[global_index])
            cal_u_t.append(float(row["uT"]))
            cal_u_0.append(float(row["u0"]))
            cal_a.append(float(row["a"]))
            cal_condition.append(condition_index)
            cal_a0_counts.append(_counts(row, "a0_counts"))
            cal_a1_counts.append(_counts(row, "a1_counts"))
    result = SelectorArrays(
        fit_perturbed=np.concatenate(fit_x).astype(np.float32, copy=False),
        fit_clean=clean_x[fit_clean_global].astype(np.float32, copy=False),
        fit_clean_index=np.asarray(fit_clean_index, dtype=np.int64),
        fit_u_t=np.asarray(fit_u_t, dtype=np.float32),
        fit_u_0=np.asarray(fit_u_0, dtype=np.float32),
        fit_a=np.asarray(fit_a, dtype=np.float32),
        calibration_perturbed=np.concatenate(cal_x).astype(np.float32, copy=False),
        calibration_clean=clean_x[cal_clean_global].astype(np.float32, copy=False),
        calibration_clean_index=np.asarray(cal_clean_index, dtype=np.int64),
        calibration_u_t=np.asarray(cal_u_t, dtype=np.float32),
        calibration_u_0=np.asarray(cal_u_0, dtype=np.float32),
        calibration_a=np.asarray(cal_a, dtype=np.float32),
        calibration_condition=np.asarray(cal_condition, dtype=np.int64),
        calibration_a0_counts=np.asarray(cal_a0_counts, dtype=np.int64),
        calibration_a1_counts=np.asarray(cal_a1_counts, dtype=np.int64),
        calibration_clean_u0=np.asarray(
            [clean_rows[index]["u0"] for index in cal_clean_global], dtype=np.float32),
        calibration_clean_a0_counts=np.asarray(
            [_counts(clean_rows[index], "a0_counts") for index in cal_clean_global],
            dtype=np.int64),
        calibration_clean_a1_counts=np.asarray(
            [_counts(clean_rows[index], "a1_counts") for index in cal_clean_global],
            dtype=np.int64),
    )
    if (result.fit_perturbed.shape != (51120, 1024)
            or result.calibration_perturbed.shape != (12960, 1024)
            or result.fit_clean_index.shape != (51120,)
            or result.calibration_clean_index.shape != (12960,)):
        raise ValueError("Loaded selector arrays differ from the frozen counts")
    return result


@dataclass
class DeviceArrays:
    fit_perturbed: Tensor
    fit_clean: Tensor
    fit_clean_index: Tensor
    fit_u_t: Tensor
    fit_u_0: Tensor
    fit_a: Tensor
    calibration_perturbed: Tensor
    calibration_clean: Tensor
    calibration_clean_index: Tensor
    calibration_u_t: Tensor
    calibration_u_0: Tensor
    calibration_a: Tensor


def _to_device(arrays: SelectorArrays, normalization: Mapping[str, Any],
               device: torch.device) -> DeviceArrays:
    descriptor = normalization["descriptor"]
    mean = torch.tensor(descriptor["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(descriptor["std"], dtype=torch.float32, device=device)

    def x(value: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)
        return (tensor - mean) / std

    def f(value: np.ndarray) -> Tensor:
        return torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)

    def i(value: np.ndarray) -> Tensor:
        return torch.from_numpy(np.array(value, dtype=np.int64, copy=True)).to(device)

    return DeviceArrays(
        x(arrays.fit_perturbed), x(arrays.fit_clean), i(arrays.fit_clean_index),
        f(arrays.fit_u_t), f(arrays.fit_u_0), f(arrays.fit_a),
        x(arrays.calibration_perturbed), x(arrays.calibration_clean),
        i(arrays.calibration_clean_index), f(arrays.calibration_u_t),
        f(arrays.calibration_u_0), f(arrays.calibration_a),
    )


def _target(value: Tensor, normalization: Mapping[str, Any], name: str) -> Tensor:
    row = normalization["targets"][name]
    return (value - float(row["mean"])) / float(row["std"])


def _regression_loss(output: Tensor, primary: Tensor, secondary: Tensor | None,
                     beta: float) -> Tensor:
    first = F.smooth_l1_loss(output[:, 0], primary, beta=beta)
    if secondary is None:
        return first
    second = F.smooth_l1_loss(output[:, 1], secondary, beta=beta)
    return 0.5 * (first + second)


def _classification_loss(logit: Tensor, utility: Tensor, weights: Mapping[str, float],
                         *, weighted: bool) -> Tensor:
    target = (utility > 0).to(torch.float32)
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if weighted:
        weight = torch.where(target > 0.5, float(weights["positive"]),
                             float(weights["nonpositive"]))
        loss = loss * weight
    return loss.mean()


def _variant_loss(variant: str, output: Tensor, utility: Tensor,
                   clean_utility: Tensor, advantage: Tensor,
                   normalization: Mapping[str, Any], *, weighted_c1: bool,
                   smooth_l1_beta: float = 1.0) -> Tensor:
    if variant == "c1":
        return _classification_loss(output[:, 0], utility,
                                    normalization["c1"]["class_weights"],
                                    weighted=weighted_c1)
    primary = _target(utility, normalization, "u_obs")
    secondary = None
    if variant == "g2":
        secondary = _target(clean_utility, normalization, "u_clean")
    elif variant == "g3":
        secondary = _target(advantage, normalization, "a_obs")
    elif variant != "g1":
        raise ValueError(f"Unknown Phase 7 selector variant: {variant}")
    return _regression_loss(output, primary, secondary, float(smooth_l1_beta))


def _epoch_loss(model: nn.Module, arrays: DeviceArrays, normalization: Mapping[str, Any],
                variant: str, optimizer: torch.optim.Optimizer, epoch: int,
                settings: Mapping[str, Any], guard_callback: Callable[[], None] | None) -> float:
    model.train()
    generator = np.random.Generator(np.random.PCG64(_epoch_seed(42, epoch)))
    order = generator.permutation(arrays.fit_perturbed.shape[0])
    batch_size = int(settings["batch_perturbed_records"])
    total = 0.0
    batches = 0
    for start in range(0, len(order), batch_size):
        if guard_callback:
            guard_callback()
        index = torch.from_numpy(order[start:start + batch_size].astype(
            np.int64, copy=False)).to(arrays.fit_perturbed.device)
        clean_index = arrays.fit_clean_index[index]
        perturbed_output = model(arrays.fit_perturbed[index])
        clean_output = model(arrays.fit_clean[clean_index])
        perturbed_loss = _variant_loss(
            variant, perturbed_output, arrays.fit_u_t[index], arrays.fit_u_0[index],
            arrays.fit_a[index], normalization, weighted_c1=True,
            smooth_l1_beta=float(settings["smooth_l1_beta"]))
        zeros = torch.zeros_like(arrays.fit_u_t[index])
        clean_loss = _variant_loss(
            variant, clean_output, arrays.fit_u_0[index], arrays.fit_u_0[index], zeros,
            normalization, weighted_c1=True,
            smooth_l1_beta=float(settings["smooth_l1_beta"]))
        loss = 0.5 * (perturbed_loss + clean_loss)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite Phase 7 selector loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]),
                                       error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    if batches != settings["attempted_updates_per_epoch"]:
        raise ValueError("Phase 7 attempted updates per epoch differ")
    return total / batches


def _calibration_metric(model: nn.Module, arrays: DeviceArrays,
                        normalization: Mapping[str, Any], variant: str,
                        *, smooth_l1_beta: float = 1.0,
                        batch_size: int = 2048) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, arrays.calibration_perturbed.shape[0], batch_size):
            stop = min(start + batch_size, arrays.calibration_perturbed.shape[0])
            index = torch.arange(start, stop, device=arrays.calibration_perturbed.device)
            clean_index = arrays.calibration_clean_index[index]
            perturbed = model(arrays.calibration_perturbed[index])
            clean = model(arrays.calibration_clean[clean_index])
            if variant == "c1":
                p = _classification_loss(perturbed[:, 0], arrays.calibration_u_t[index],
                                         normalization["c1"]["class_weights"], weighted=False)
                c = _classification_loss(clean[:, 0], arrays.calibration_u_0[index],
                                         normalization["c1"]["class_weights"], weighted=False)
            else:
                p = F.smooth_l1_loss(
                    perturbed[:, 0], _target(arrays.calibration_u_t[index],
                                             normalization, "u_obs"),
                    beta=float(smooth_l1_beta))
                c = F.smooth_l1_loss(
                    clean[:, 0], _target(arrays.calibration_u_0[index],
                                         normalization, "u_obs"),
                    beta=float(smooth_l1_beta))
            value = 0.5 * (p + c)
            size = stop - start
            total += float(value) * size
            count += size
    return total / count


def _predict(model: nn.Module, x: Tensor, batch_size: int = 4096) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            outputs.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def _scores(variant: str, output: np.ndarray, normalization: Mapping[str, Any]) -> tuple[
        np.ndarray, np.ndarray]:
    if variant == "c1":
        primary = output[:, 0]
        return primary, primary
    primary_stats = normalization["targets"]["u_obs"]
    primary = output[:, 0] * primary_stats["std"] + primary_stats["mean"]
    if variant == "g2":
        secondary_stats = normalization["targets"]["u_clean"]
        clean = output[:, 1] * secondary_stats["std"] + secondary_stats["mean"]
        constrained = np.minimum(primary, clean)
    elif variant == "g3":
        secondary_stats = normalization["targets"]["a_obs"]
        advantage = output[:, 1] * secondary_stats["std"] + secondary_stats["mean"]
        constrained = np.minimum(primary, primary - advantage)
    elif variant == "g1":
        constrained = primary
    else:
        raise ValueError(f"Unknown selector variant: {variant}")
    return primary, constrained


def _mf1(counts: np.ndarray, condition: np.ndarray, selected: np.ndarray) -> float:
    chosen = np.where(selected[:, None], counts[:, 1, :], counts[:, 0, :])
    values = []
    for index in range(45):
        pooled = chosen[condition == index].sum(axis=0)
        values.append(_metrics_from_counts(pooled)["f1"])
    return float(np.mean(values))


def _threshold_value(record: Mapping[str, Any]) -> float:
    kind = record["kind"]
    if kind == "finite":
        return float(record["value"])
    if kind == "positive_infinity":
        return math.inf
    if kind == "negative_infinity":
        return -math.inf
    raise ValueError("Invalid serialized Phase 7 threshold")


def threshold_record(value: float) -> dict[str, Any]:
    if value == math.inf:
        return {"kind": "positive_infinity", "value": None}
    if value == -math.inf:
        return {"kind": "negative_infinity", "value": None}
    if not math.isfinite(value):
        raise ValueError("Unexpected non-finite threshold")
    return {"kind": "finite", "value": float(value)}


def _policy_summary(arrays: SelectorArrays, perturbed_score: np.ndarray,
                    clean_score: np.ndarray, threshold: float) -> dict[str, Any]:
    perturbed_selected = perturbed_score > threshold
    clean_selected = clean_score > threshold
    clean_counts = np.where(
        clean_selected[:, None], arrays.calibration_clean_a1_counts,
        arrays.calibration_clean_a0_counts).sum(axis=0)
    clean_metrics = _metrics_from_counts(clean_counts)
    negative_mass = float(np.maximum(-arrays.calibration_u_t[perturbed_selected], 0).sum())
    positive_mass = float(np.maximum(arrays.calibration_u_t[perturbed_selected], 0).sum())
    return {
        "threshold": threshold_record(threshold),
        "corruption_mf1": _mf1(
            np.stack((arrays.calibration_a0_counts, arrays.calibration_a1_counts), axis=1),
            arrays.calibration_condition, perturbed_selected),
        "clean": clean_metrics,
        "perturbed_a1_selected": int(perturbed_selected.sum()),
        "perturbed_selection_rate": float(perturbed_selected.mean()),
        "clean_a1_selected": int(clean_selected.sum()),
        "clean_selection_rate": float(clean_selected.mean()),
        "selected_negative_utility_mass": negative_mass,
        "captured_positive_utility_mass": positive_mass,
    }


def calibrate_policies(config: Phase7Config, variant: str, model: nn.Module,
                       device_arrays: DeviceArrays, arrays: SelectorArrays,
                       normalization: Mapping[str, Any]) -> dict[str, Any]:
    perturbed_output = _predict(model, device_arrays.calibration_perturbed)
    clean_output = _predict(model, device_arrays.calibration_clean)
    primary_perturbed, constrained_perturbed = _scores(
        variant, perturbed_output, normalization)
    primary_clean, constrained_clean = _scores(variant, clean_output, normalization)
    unconstrained = _policy_summary(arrays, primary_perturbed, primary_clean, 0.0)
    repeated_clean = constrained_clean[arrays.calibration_clean_index]
    quantile_input = np.concatenate((constrained_perturbed, repeated_clean))
    probabilities = np.linspace(0.0, 1.0, config.values["policies"][
        "calibration_quantiles"])
    finite = np.quantile(quantile_input.astype(np.float64), probabilities)
    candidates = np.concatenate(([-math.inf], np.unique(finite), [math.inf]))
    a0_clean = _metrics_from_counts(arrays.calibration_clean_a0_counts.sum(axis=0))
    f1_floor = a0_clean["f1"] - config.values["policies"]["clean_f1_tolerance"]
    recall_floor = a0_clean["recall"] - config.values["policies"][
        "clean_recall_tolerance"]
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for threshold in candidates:
        row = _policy_summary(
            arrays, constrained_perturbed, constrained_clean, float(threshold))
        if row["clean"]["f1"] < f1_floor or row["clean"]["recall"] < recall_floor:
            continue
        key = (row["corruption_mf1"], row["clean"]["recall"], row["clean"]["f1"],
               -row["selected_negative_utility_mass"], float(threshold))
        if best_key is None or key > best_key:
            best_key, best = key, row
    if best is None:
        raise ValueError("No feasible Phase 7 harm-constrained threshold")
    return {
        "status": "calibrated", "variant": variant,
        "unconstrained": unconstrained, "harm_constrained": best,
        "candidate_thresholds": int(len(candidates)),
        "clean_a0_reference": a0_clean,
        "clean_f1_floor": f1_floor, "clean_recall_floor": recall_floor,
        "score_rule": ("min_predicted_u_obs_and_u_clean"
                       if variant in ("g2", "g3") else "primary_benefit_score"),
        "test_evaluated": False,
    }


def selector_preflight(config: Phase7Config, variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("Unknown Phase 7 selector variant")
    load_selector_split(config)
    cache = load_descriptor_cache(config)
    output = config.training_output(variant)
    return {
        "status": "phase7_selector_preflight_passed", "variant": variant,
        "kind": "metadata_only_no_model_or_device",
        "phase7_fingerprint": config.fingerprint,
        "cache_status_sha256": _sha(cache["output"] / "status.json"),
        "cache_guard_sha256": _sha(cache["output"] / "guard_completion.json"),
        "output_exists": output.exists(), "model_initialized": False,
        "training_started": False, "validation_evaluated": False,
        "sensor_queried": False, "test_evaluated": False,
    }


def _snapshot(config: Phase7Config, variant: str, output: Path,
              device: torch.device, source_hashes: Mapping[str, str],
              cache: Mapping[str, Any]) -> dict[str, str]:
    (output / "phase7_config.yaml").write_bytes(config.raw)
    for key, name in (("approval", "approval.yaml"),
                      ("frozen_protocol_document", "frozen_protocol_document.md"),
                      ("selector_split_manifest", "selector_split_manifest.json")):
        shutil.copyfile(config.paths[key], output / name)
    shutil.copyfile(cache["output"] / "normalization.json", output / "normalization.json")
    cache_hashes = {
        "status": _sha(cache["output"] / "status.json"),
        "guard_completion": _sha(cache["output"] / "guard_completion.json"),
        "normalization": _sha(cache["output"] / "normalization.json"),
    }
    write_json(output / "cache_sha256.json", cache_hashes)
    write_json(output / "source_sha256.json", dict(source_hashes))
    write_json(output / "environment.json", environment_record(config.base, device))
    write_json(output / "run_binding.json", {
        "variant": variant, "phase7_fingerprint": config.fingerprint,
        "phase7_config_sha256": config.config_sha256,
        "approval_sha256": _sha(config.paths["approval"]),
        "selector_split_manifest_sha256": _sha(config.paths["selector_split_manifest"]),
        "descriptor_cache_sha256": cache_hashes, "test_evaluated": False,
    })
    for name, path in phase7_training_source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if _sha(target) != source_hashes[name]:
            raise ValueError(f"Phase 7 training source changed while copying: {name}")
    return cache_hashes


def _load_resume(output: Path, config: Phase7Config, variant: str,
                 initialized_sha: str, order_sha: str) -> dict[str, Any]:
    checkpoint_path = output / "last.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Phase 7 resume requires last.pt")
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text(
            encoding="utf-8")).get("status") == "completed":
        raise ValueError("Completed Phase 7 selector must not be resumed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (checkpoint.get("training_flavor") != "phase7_selector_v1"
            or checkpoint.get("variant") != variant
            or checkpoint.get("phase7_fingerprint") != config.fingerprint
            or checkpoint.get("initial_selector_tensor_sha256") != initialized_sha
            or checkpoint.get("training_order_sha256") != order_sha):
        raise ValueError("Phase 7 selector resume checkpoint identity differs")
    return checkpoint


def train_selector(config: Phase7Config, variant: str, *, resume: bool = False,
                   guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("variant must be one of c1/g1/g2/g3")
    cache = load_descriptor_cache(config)
    arrays = load_selector_arrays(config)
    normalization = cache["normalization"]
    settings = config.values["training"]
    epoch_settings = dict(settings, smooth_l1_beta=config.values["targets"][
        "smooth_l1_beta"])
    output = config.training_output(variant)
    source_hashes = {name: _sha(path) for name, path in phase7_training_source_files().items()}
    device = setup_runtime(config.base)
    model = build_phase7_selector(config.values["selector"]["seed"]).to(device)
    initialized_sha = selector_initial_sha256(config.values["selector"]["seed"])
    order_sha = _hash_order(len(arrays.fit_u_t), settings["epochs"],
                            config.values["selector"]["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"], betas=tuple(settings["adam_betas"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"])
    checkpoint = None
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("Phase 7 resume requires an existing output")
        checkpoint = _load_resume(output, config, variant, initialized_sha, order_sha)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
    elif output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 selector: {output}")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        cache_hashes = _snapshot(
            config, variant, output, device, source_hashes, cache)
    else:
        cache_hashes = json.loads((output / "cache_sha256.json").read_text(
            encoding="utf-8"))
        current_cache_hashes = {
            "status": _sha(cache["output"] / "status.json"),
            "guard_completion": _sha(cache["output"] / "guard_completion.json"),
            "normalization": _sha(cache["output"] / "normalization.json"),
        }
        if current_cache_hashes != cache_hashes:
            raise ValueError("Phase 7 cache changed before selector resume")
        saved_sources = json.loads((output / "source_sha256.json").read_text(
            encoding="utf-8"))
        if saved_sources != source_hashes:
            raise ValueError("Phase 7 training source changed before selector resume")
    device_arrays = _to_device(arrays, normalization, device)
    start_epoch = int(checkpoint["epoch"] + 1) if checkpoint else 1
    best_epoch = int(checkpoint["best_epoch"]) if checkpoint else 0
    best_metric = float(checkpoint["best_metric"]) if checkpoint else math.inf
    attempted = int(checkpoint["attempted_updates"]) if checkpoint else 0
    started = time.monotonic()
    try:
        for epoch in range(start_epoch, settings["epochs"] + 1):
            train_loss = _epoch_loss(model, device_arrays, normalization, variant,
                                     optimizer, epoch, epoch_settings, guard_callback)
            calibration_metric = _calibration_metric(
                model, device_arrays, normalization, variant,
                smooth_l1_beta=config.values["targets"]["smooth_l1_beta"])
            scheduler.step()
            attempted += settings["attempted_updates_per_epoch"]
            improved = calibration_metric < best_metric
            if improved:
                best_epoch, best_metric = epoch, calibration_metric
            payload = {
                "format_version": 1, "training_flavor": "phase7_selector_v1",
                "variant": variant, "phase7_fingerprint": config.fingerprint,
                "phase7_config_sha256": config.config_sha256,
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_epoch": best_epoch, "best_metric": best_metric,
                "train_loss": train_loss, "calibration_primary_metric": calibration_metric,
                "attempted_updates": attempted,
                "initial_selector_tensor_sha256": initialized_sha,
                "training_order_sha256": order_sha,
                "cache_sha256": cache_hashes,
                "rng": {"torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all()},
                "test_evaluated": False,
            }
            save_checkpoint(output / "last.pt", payload)
            if improved:
                _atomic_copy_checkpoint(output / "last.pt", output / "best.pt")
            status = {
                "status": "training", "kind": "research", "stage": "phase7_selector",
                "variant": variant, "epoch": epoch, "epochs": settings["epochs"],
                "best_epoch": best_epoch, "best_calibration_primary_metric": best_metric,
                "train_loss": train_loss, "attempted_updates": attempted,
                "initial_selector_tensor_sha256": initialized_sha,
                "training_order_sha256": order_sha, "phase7_fingerprint": config.fingerprint,
                "validation_evaluated": False, "test_evaluated": False,
            }
            write_json(output / "status.json", status, overwrite=True)
            print(f"phase7/{variant}/epoch={epoch} train={train_loss:.8f} "
                  f"cal={calibration_metric:.8f} best_epoch={best_epoch}", flush=True)
        best = torch.load(output / "best.pt", map_location=device)
        model.load_state_dict(best["model"], strict=True)
        calibration = calibrate_policies(
            config, variant, model, device_arrays, arrays, normalization)
        write_json(output / "policy_calibration.json", calibration)
        if ({name: _sha(path) for name, path in phase7_training_source_files().items()}
                != source_hashes):
            raise ValueError("Phase 7 training source changed during selector fitting")
        result = {
            "status": "completed", "kind": "research", "stage": "phase7_selector",
            "variant": variant, "epochs": settings["epochs"],
            "best_epoch": best_epoch, "best_calibration_primary_metric": best_metric,
            "attempted_updates": attempted,
            "successful_updates": attempted,
            "initial_selector_tensor_sha256": initialized_sha,
            "training_order_sha256": order_sha,
            "phase7_fingerprint": config.fingerprint,
            "cache_sha256": cache_hashes,
            "best_checkpoint_sha256": _sha(output / "best.pt"),
            "last_checkpoint_sha256": _sha(output / "last.pt"),
            "policy_calibration_sha256": _sha(output / "policy_calibration.json"),
            "calibration": calibration, "seconds": time.monotonic() - started,
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "source_sha256": source_hashes, "training_started": True,
            "validation_evaluated": False, "test_evaluated": False,
            "guarded_completion_required": True,
        }
        write_json(output / "status.json", result, overwrite=True)
        return result
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": f"{type(error).__name__}: {error}", "variant": variant,
            "phase7_fingerprint": config.fingerprint, "training_started": True,
            "validation_evaluated": False, "test_evaluated": False,
        }, overwrite=True)
        raise
