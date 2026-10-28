"""Real-CUDA train-only engineering smoke for the approved Phase 7 path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pcdc.data import LEVIRDataset, make_loader
from pcdc.phase5_models import per_sample_change_loss
from pcdc.phase5_protocol import load_frozen_manifests, verify_partition_files
from pcdc.phase5_training import SourceSubsetDataset
from pcdc.phase6_diagnostic import (
    GateInstanceDataset,
    _artifact_hashes,
    _load_models,
    per_sample_confusion,
)
from pcdc.phase7_cache import _scalar_stats, _weighted_stats
from pcdc.phase7_models import build_phase7_selector, phase7_descriptor, selector_initial_sha256
from pcdc.phase7_protocol import load_phase7_config
from pcdc.phase7_training import VARIANTS, _variant_loss
from pcdc.robust_baselines import tensor_state_sha256
from pcdc.runtime import environment_record, save_checkpoint, setup_runtime, write_json
from scripts.evaluate_guarded import GuardIdentityError, _atomic_json, load_guard_config
from scripts.train_robust_guarded import TrainingTemperatureGuard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stack_clean(gate: SourceSubsetDataset, sample_ids: list[str]) -> dict[str, torch.Tensor]:
    lookup = {}
    for index, (record_index, x, y) in enumerate(gate.tiles):
        name = gate.records[record_index].name
        lookup[f"train/{name}@x{x}_y{y}"] = index
    if any(sample_id not in lookup for sample_id in sample_ids):
        raise ValueError("Phase 7 smoke cannot match perturbed tiles to clean tiles")
    rows = [gate[lookup[sample_id]] for sample_id in sample_ids]
    return {key: torch.stack([row[key] for row in rows])
            for key in ("image_a", "image_b", "mask")}


def _normalization(clean_descriptor: torch.Tensor, perturbed_descriptor: torch.Tensor,
                   clean_u: torch.Tensor, perturbed_u: torch.Tensor) -> dict:
    clean_x = clean_descriptor.detach().cpu().numpy().astype(np.float32)
    perturbed_x = perturbed_descriptor.detach().cpu().numpy().astype(np.float32)
    mean, std = _weighted_stats(clean_x, [perturbed_x], 1e-6)
    clean = clean_u.detach().cpu().numpy().astype(np.float64)
    perturbed = perturbed_u.detach().cpu().numpy().astype(np.float64)
    advantage = perturbed - clean
    positive = 0.5 * (float((clean > 0).mean()) + float((perturbed > 0).mean()))
    if not 0 < positive < 1:
        # A smoke batch may be accidentally single-class. This deterministic fallback
        # remains engineering-only and is never used by the research cache or training.
        positive = 0.5
    return {
        "descriptor": {"mean": mean.tolist(), "std": std.tolist()},
        "targets": {
            "u_obs": _scalar_stats(clean, perturbed, 1e-6),
            "u_clean": _scalar_stats(clean, clean, 1e-6),
            "a_obs": _scalar_stats(np.zeros_like(clean), advantage, 1e-6),
        },
        "c1": {"class_weights": {
            "positive": 1.0 / (2.0 * positive),
            "nonpositive": 1.0 / (2.0 * (1.0 - positive)),
        }},
    }


def _step(model, optimizer, variant: str, perturbed_x: torch.Tensor,
          clean_x: torch.Tensor, perturbed_u: torch.Tensor, clean_u: torch.Tensor,
          normalization: dict) -> float:
    advantage = perturbed_u - clean_u
    zeros = torch.zeros_like(clean_u)
    perturbed_loss = _variant_loss(
        variant, model(perturbed_x), perturbed_u, clean_u, advantage,
        normalization, weighted_c1=True, smooth_l1_beta=1.0)
    clean_loss = _variant_loss(
        variant, model(clean_x), clean_u, clean_u, zeros,
        normalization, weighted_c1=True, smooth_l1_beta=1.0)
    loss = 0.5 * (perturbed_loss + clean_loss)
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite Phase 7 smoke loss for {variant}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_phase7_config(args.config)
    guard_config = load_guard_config(args.guard_config)
    if (config.base.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or guard_config.values["gpu_index"] != 0):
        raise GuardIdentityError("Phase 7 smoke is frozen to CUDA physical GPU index 0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7 smoke: {output}")
    output.mkdir(parents=True, exist_ok=False)
    guard = TrainingTemperatureGuard(guard_config, output)
    try:
        device = setup_runtime(config.base)
        if device.type != "cuda":
            raise RuntimeError("Phase 7 smoke requires real CUDA")
        partition, gate_manifest = load_frozen_manifests(config.phase56)
        train = LEVIRDataset(config.base, "train")
        verify_partition_files(config.phase56, train, partition)
        gate = SourceSubsetDataset(train, partition["gate_names"], partition="gate")
        condition = "brightness_l1"
        instances = [row for row in gate_manifest["instances"]
                     if row["condition_id"] == condition]
        perturbed_batch = next(iter(make_loader(
            GateInstanceDataset(gate, instances), config.base, training=False)))
        clean_batch = _stack_clean(gate, list(perturbed_batch["sample_id"]))
        memory_format = config.base.values["runtime"]["memory_format"]
        fmt = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format

        def transfer(batch):
            return (batch["image_a"].to(device=device, memory_format=fmt),
                    batch["image_b"].to(device=device, memory_format=fmt),
                    batch["mask"].to(device=device))

        perturbed_a, perturbed_b, target = transfer(perturbed_batch)
        clean_a, clean_b, clean_target = transfer(clean_batch)
        if not torch.equal(target, clean_target):
            raise ValueError("Phase 7 smoke clean/perturbed masks differ")
        artifacts = _artifact_hashes(config.phase56)
        detector, _, _ = _load_models(config.phase56, artifacts, device)
        detector.eval()
        guard()
        train_settings = config.base.values["train"]
        with torch.no_grad(), torch.autocast(
                device_type="cuda", enabled=train_settings["amp"]):
            perturbed_features = detector._encode(perturbed_a, perturbed_b)
            clean_features = detector._encode(clean_a, clean_b)
            perturbed_descriptor = phase7_descriptor(
                *perturbed_features, detector.adapter)
            clean_descriptor = phase7_descriptor(*clean_features, detector.adapter)
        loss_kwargs = {key: float(config.base.values["loss"][key])
                       for key in ("bce_weight", "dice_weight", "dice_smooth")}
        with torch.no_grad(), torch.autocast(
                device_type="cuda", enabled=train_settings["amp"]):
            perturbed_a0, perturbed_a1 = detector.action_logits(perturbed_a, perturbed_b)
            clean_a0, clean_a1 = detector.action_logits(clean_a, clean_b)
        perturbed_u = (per_sample_change_loss(perturbed_a0, target, **loss_kwargs)["total"]
                       - per_sample_change_loss(perturbed_a1, target, **loss_kwargs)["total"])
        clean_u = (per_sample_change_loss(clean_a0, target, **loss_kwargs)["total"]
                   - per_sample_change_loss(clean_a1, target, **loss_kwargs)["total"])
        normalization = _normalization(
            clean_descriptor, perturbed_descriptor, clean_u, perturbed_u)
        mean = torch.tensor(normalization["descriptor"]["mean"], device=device)
        std = torch.tensor(normalization["descriptor"]["std"], device=device)
        perturbed_x = (perturbed_descriptor - mean) / std
        clean_x = (clean_descriptor - mean) / std
        variants = {}
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir()
        for variant in VARIANTS:
            guard()
            model = build_phase7_selector(42).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4,
                betas=(0.9, 0.999))
            first_loss = _step(model, optimizer, variant, perturbed_x, clean_x,
                               perturbed_u, clean_u, normalization)
            checkpoint = checkpoint_dir / f"{variant}.pt"
            save_checkpoint(checkpoint, {
                "variant": variant, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "completed_steps": 1,
                "test_evaluated": False,
            })
            resumed = build_phase7_selector(42).to(device)
            resumed_optimizer = torch.optim.AdamW(
                resumed.parameters(), lr=1e-3, weight_decay=1e-4,
                betas=(0.9, 0.999))
            payload = torch.load(checkpoint, map_location=device)
            resumed.load_state_dict(payload["model"], strict=True)
            resumed_optimizer.load_state_dict(payload["optimizer"])
            if tensor_state_sha256(model.state_dict()) != tensor_state_sha256(
                    resumed.state_dict()):
                raise ValueError(f"Phase 7 smoke resume state differs for {variant}")
            second_loss = _step(resumed, resumed_optimizer, variant, perturbed_x,
                                clean_x, perturbed_u, clean_u, normalization)
            variants[variant] = {
                "first_step_loss": first_loss, "resumed_step_loss": second_loss,
                "checkpoint_sha256": _sha(checkpoint), "resume_state_equal": True,
            }
        counts_a0 = per_sample_confusion(
            perturbed_a0, target, config.base.values["evaluation"]["threshold"])
        counts_a1 = per_sample_confusion(
            perturbed_a1, target, config.base.values["evaluation"]["threshold"])
        result = {
            "status": "completed", "kind": "engineering_smoke_not_research",
            "stage": "phase7", "split": "train_gate_only", "condition": condition,
            "samples": len(perturbed_batch["sample_id"]), "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "phase7_fingerprint": config.fingerprint,
            "initial_selector_tensor_sha256": selector_initial_sha256(42),
            "descriptor_shape": list(perturbed_descriptor.shape),
            "descriptor_finite": bool(torch.isfinite(perturbed_descriptor).all()),
            "utility_finite": bool(torch.isfinite(perturbed_u).all()),
            "both_actions_executed": bool(counts_a0.numel() and counts_a1.numel()),
            "variants": variants, "research_result": False,
            "validation_evaluated": False, "test_evaluated": False,
        }
        if (result["descriptor_shape"][1] != 1024 or not result["descriptor_finite"]
                or not result["utility_finite"] or not result["both_actions_executed"]):
            raise ValueError("Phase 7 smoke invariants failed")
        write_json(output / "environment.json", environment_record(config.base, device))
        write_json(output / "status.json", result)
        guard_record = guard.finish_training(result)
        guarded = dict(result, temperature_guard=guard_record,
                       guarded_completion_required=True)
        _atomic_json(output / "status.json", guarded)
        _atomic_json(output / "guard_completion.json", {
            "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "guard_config_sha256": guard_config.config_sha256,
            "status_sha256": _sha(output / "status.json"),
            "guard_state_sha256": _sha(output / "temperature_guard_state.json"),
            "wrapper_source_sha256": _sha(Path(__file__)),
        })
        print(json.dumps(guarded, indent=2, ensure_ascii=False, allow_nan=False))
    except BaseException as error:
        if guard.attached:
            guard.abort(error)
        _atomic_json(output / "status.json", {
            "status": "failed", "kind": "engineering_smoke_not_research",
            "error": f"{type(error).__name__}: {error}",
            "validation_evaluated": False, "test_evaluated": False,
        })
        raise


if __name__ == "__main__":
    main()
