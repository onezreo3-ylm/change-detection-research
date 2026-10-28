"""One-batch real-data/CUDA engineering smoke for the approved Phase 5-6 path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from pcdc.data import LEVIRDataset, make_loader
from pcdc.engine import move_model
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.phase5_models import AdapterCorrectedDetector, Stage2ResidualAdapter, per_sample_change_loss
from pcdc.phase5_protocol import load_frozen_manifests, load_phase56_config, verify_partition_files
from pcdc.phase5_training import SourceSubsetDataset, run_adapter_epoch
from pcdc.phase6_diagnostic import GateInstanceDataset, per_sample_confusion
from pcdc.robust_baselines import TwoViewTrainingDataset, run_two_view_epoch, tensor_state_sha256
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts.evaluate_guarded import _atomic_json, load_guard_config
from scripts.train_robust_guarded import TrainingTemperatureGuard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_phase56_config(args.config, require_approval=True)
    guard_config = load_guard_config(args.guard_config)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    guard = TrainingTemperatureGuard(guard_config, output)
    try:
        device = setup_runtime(config.base)
        if device.type != "cuda":
            raise RuntimeError("Phase 5-6 smoke requires real CUDA")
        partition, gate_manifest = load_frozen_manifests(config)
        train = LEVIRDataset(config.base, "train")
        verify_partition_files(config, train, partition)
        actor = SourceSubsetDataset(train, partition["actor_names"], partition="actor")
        actor_views = TwoViewTrainingDataset(actor, config.corruption, "perturbed")
        actor_views.set_epoch(1)
        batch = next(iter(make_loader(actor_views, config.base, training=True)))
        loss_fn = ChangeDetectionLoss(**config.base.values["loss"])
        detector = move_model(build_model(config.base), device,
                              config.base.values["runtime"]["memory_format"])
        initial_sha = tensor_state_sha256(detector.state_dict())
        if initial_sha != config.values["actor_detector"]["expected_initial_model_tensor_sha256"]:
            raise ValueError("Smoke actor initialization differs from frozen seed 42")
        settings = config.base.values["train"]
        actor_optimizer = torch.optim.AdamW(
            detector.parameters(), lr=settings["learning_rate"],
            betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        actor_scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
        actor_result = run_two_view_epoch(
            detector, [batch], loss_fn, device,
            threshold=config.base.values["evaluation"]["threshold"],
            consistency_kind="none", consistency_weight=0.0, amp=settings["amp"],
            memory_format=config.base.values["runtime"]["memory_format"],
            optimizer=actor_optimizer, scaler=actor_scaler,
            grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard,
            progress="phase5-smoke/actor")
        for parameter in detector.parameters():
            parameter.grad = None
        corrected = AdapterCorrectedDetector(detector, Stage2ResidualAdapter())
        corrected = move_model(corrected, device, config.base.values["runtime"]["memory_format"])
        fmt = (torch.channels_last if config.base.values["runtime"]["memory_format"]
               == "channels_last" else torch.contiguous_format)
        clean_a = batch["clean_a"].to(device=device, memory_format=fmt)
        clean_b = batch["clean_b"].to(device=device, memory_format=fmt)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=settings["amp"]):
            initial_a0, initial_a1 = corrected.action_logits(clean_a, clean_b)
        if not torch.equal(initial_a0, initial_a1):
            raise ValueError("Zero-initialized adapter is not an exact no-op")
        adapter_optimizer = torch.optim.AdamW(
            corrected.adapter.parameters(), lr=settings["learning_rate"],
            betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        adapter_scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
        adapter_result = run_adapter_epoch(
            corrected, [batch], loss_fn, device,
            threshold=config.base.values["evaluation"]["threshold"], amp=settings["amp"],
            memory_format=config.base.values["runtime"]["memory_format"],
            optimizer=adapter_optimizer, scaler=adapter_scaler,
            grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard,
            progress="phase5-smoke/adapter")
        gate = SourceSubsetDataset(train, partition["gate_names"], partition="gate")
        first_condition = "brightness_l1"
        instances = [row for row in gate_manifest["instances"]
                     if row["condition_id"] == first_condition]
        gate_batch = next(iter(make_loader(
            GateInstanceDataset(gate, instances), config.base, training=False)))
        image_a = gate_batch["image_a"].to(device=device, memory_format=fmt)
        image_b = gate_batch["image_b"].to(device=device, memory_format=fmt)
        target = gate_batch["mask"].to(device)
        guard()
        corrected.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=settings["amp"]):
            a0, a1 = corrected.action_logits(image_a, image_b)
        utility = (per_sample_change_loss(a0, target, **config.base.values["loss"])["total"]
                   - per_sample_change_loss(a1, target, **config.base.values["loss"])["total"])
        counts = per_sample_confusion(a1, target,
                                      config.base.values["evaluation"]["threshold"])
        result = {
            "status": "completed", "kind": "engineering_smoke_not_research",
            "real_data_split": "train_actor_and_train_gate_only", "device": str(device),
            "gpu": torch.cuda.get_device_name(device), "phase56_fingerprint": config.fingerprint,
            "initial_model_tensor_sha256": initial_sha,
            "actor_batch": {"samples": actor_result["source_samples"],
                            "optimizer_steps": actor_result["optimizer_steps"]},
            "adapter_batch": {"samples": adapter_result["source_samples"],
                              "optimizer_steps": adapter_result["optimizer_steps"]},
            "phase6_batch": {"condition": first_condition,
                             "samples": len(gate_batch["sample_id"]),
                             "finite_utilities": bool(torch.isfinite(utility).all()),
                             "pixels": int(counts.sum())},
            "detector_gradients_after_adapter": sum(
                parameter.grad is not None for parameter in corrected.detector.parameters()),
            "test_evaluated": False, "research_result": False,
        }
        if (result["detector_gradients_after_adapter"] != 0
                or not result["phase6_batch"]["finite_utilities"]):
            raise ValueError("Phase 5-6 smoke invariants failed")
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
            "error": f"{type(error).__name__}: {error}", "test_evaluated": False})
        raise


if __name__ == "__main__":
    main()
