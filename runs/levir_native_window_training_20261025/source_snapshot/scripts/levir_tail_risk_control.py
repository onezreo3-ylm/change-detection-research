"""Separate ordinary B1 empirical-tail control; the archived core is unchanged."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate
import yaml

from pcdc.config import UniqueKeyLoader, protocol_signature
from pcdc.corruption_protocol import canonical, sample_training_instance
from pcdc.data import audit_data, build_datasets, file_sha256, make_loader
from pcdc.engine import move_model, read_checkpoint, run_epoch
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.robust_baselines import (
    TwoViewTrainingDataset, _exposure_summary, _restore_rng, _rng_state,
    _update_stream_hash, load_robust_config, robust_source_files,
    robust_source_hashes, run_two_view_epoch, tensor_state_sha256,
)
from pcdc.runtime import environment_record, save_checkpoint, setup_runtime
from scripts.windows_shared_json import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]
FLAVOR = "b1_secondary_empirical_tail_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return read_json(path)


def atomic(path, value):
    atomic_json(path, value)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class EmpiricalTailLoss(nn.Module):
    """Half clean mean + half stable upper-tail secondary sample loss."""
    def __init__(self, bce_weight, dice_weight, dice_smooth, q=0.5):
        super().__init__()
        if not math.isfinite(q) or not 0 < q <= 1:
            raise ValueError("q must lie in (0,1]")
        self.q = q
        self.original = ChangeDetectionLoss(bce_weight, dice_weight, dice_smooth)
        self.bce_weight, self.dice_weight, self.dice_smooth = bce_weight, dice_weight, dice_smooth

    def components(self, logits, target):
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Expected matching N,1,H,W tensors")
        if logits.shape[0] < 2 or logits.shape[0] % 2:
            raise ValueError("Expected ordered N clean followed by N secondary views")
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
        probabilities, labels = logits.sigmoid().flatten(1), target.flatten(1)
        dice = 1 - (2 * (probabilities * labels).sum(1) + self.dice_smooth) / (
            probabilities.sum(1) + labels.sum(1) + self.dice_smooth)
        return bce, dice

    def forward(self, logits, target):
        bce, dice = self.components(logits, target)
        if self.q == 1:
            # Preserve the original floating point reduction, not just its algebra.
            return self.original(logits, target)
        count = logits.shape[0] // 2
        total = self.bce_weight * bce + self.dice_weight * dice
        chosen = torch.argsort(total[count:].detach(), descending=True, stable=True)[:math.ceil(self.q * count)]
        reduced_bce = 0.5 * bce[:count].mean() + 0.5 * bce[count:][chosen].mean()
        reduced_dice = 0.5 * dice[:count].mean() + 0.5 * dice[count:][chosen].mean()
        return {"total": self.bce_weight * reduced_bce + self.dice_weight * reduced_dice,
                "bce": reduced_bce, "dice": reduced_dice}


def load_study(path):
    path = Path(path).resolve()
    values = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    root = (path.parent / values["workspace_root"]).resolve()
    if root != ROOT or values["schema_version"] != 1:
        raise ValueError("Wrong study workspace or schema")
    ref = load_robust_config(root / values["reference_robust_config"])
    candidate, cfg = values["candidate"], ref.base.values
    loss = candidate["loss"]
    if (ref.mode != "b1" or ref.base.run_test_after_training or cfg["experiment"]["seed"] != 42
            or candidate["seed"] != 42 or cfg["runtime"]["device"] != "cuda"
            or cfg["model"]["encoder_weights"] is not None or not cfg["train"]["amp"]
            or cfg["train"]["epochs"] != candidate["epochs"] or candidate["epochs"] != 100
            or cfg["train"]["batch_size"] != 4 or cfg["train"]["num_workers"] != 4
            or cfg["evaluation"]["threshold"] != 0.5 or candidate["pair_views"] != 1424000
            or candidate["total_attempted_steps"] != 178000
            or loss["upper_tail_fraction"] != 0.5 or loss["primary_clean_weight"] != 0.5
            or loss["secondary_weight"] != 0.5
            or values["engineering"]["real_cuda_steps_per_loss"] != 32
            or values["budgets"]["single_training_upper_bound_hours"] != 24
            or values["budgets"]["batch_upper_bound_gpu_hours"] != 48
            or any(values["boundaries"][k] for k in (
                "new_model_architecture", "new_corruption_operators_or_severities",
                "original_test_and_martigny_access", "new_datasets", "new_method_or_novelty_admitted"))):
        raise ValueError("Frozen scope or scientific settings changed")
    output = root / values["output_dir"]
    if output.resolve() in (ref.output_dir.resolve(), ref.base.data_root.resolve()):
        raise ValueError("Output must be separate from reference and dataset")
    return path, values, ref, output


def source_files():
    files = dict(robust_source_files())
    for name in ("scripts/levir_tail_risk_control.py", "scripts/levir_tail_risk_supervision.py",
                 "scripts/windows_shared_json.py", "scripts/temperature_guard.py", "tests/test_levir_tail_risk_control.py"):
        files[name] = ROOT / name
    return files


def bound_inputs(path, values, ref):
    files = dict(source_files())
    for item in [path, ROOT / values["protocol"], ROOT / values.get("runtime_config", "configs/levir_tail_risk_runtime_20261025.yaml"),
                 ref.source, *[ref.paths[k] for k in ("base_config", "protocol_config", "severity_review")],
                 *[ROOT / values["reference_run"] / n for n in ("source_sha256.json", "history.json", "data_audit.json", "status.json")]]:
        files[str(item.relative_to(ROOT)).replace("\\", "/")] = item
    return {name: file_sha256(p) for name, p in sorted(files.items())}


def verify_bound(bound):
    for name, expected in bound.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)


class StreamMetadata(Dataset):
    split = "train"

    def __init__(self, base, protocol):
        self.base, self.protocol = base, protocol

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        record, x, y = self.base.tiles[index]
        sample_id = f"train/{self.base.records[record].name}@x{x}_y{y}"
        instance = sample_training_instance(self.protocol, sample_id, epoch=1, view=0)
        return {"sample_id": sample_id, "instance_json": canonical(instance), "index": index}


def metadata_stream(base, ref):
    # DataLoader always draws a base seed, even with zero workers. Worker count
    # and pinning do not alter this sampler's independent generator sequence.
    generator = torch.Generator().manual_seed(ref.base.values["experiment"]["seed"])
    loader = DataLoader(StreamMetadata(base, ref.protocol), batch_size=4, shuffle=True,
                        num_workers=0, generator=generator, drop_last=False)
    stream, ids, last = hashlib.sha256(), [], None
    for batch in loader:
        _update_stream_hash(stream, list(batch["sample_id"]), list(batch["instance_json"]))
        ids.extend(batch["sample_id"])
        last = batch
    return {"sha256": stream.hexdigest(), "first_132_sample_ids": ids[:132],
            "last_sample_ids": list(last["sample_id"]), "last_indices": last["index"].tolist(),
            "source_samples": len(ids), "tail_source_batch_size": len(last["sample_id"])}


def prepare(path):
    path, values, ref, output = load_study(path)
    if (output / "prepared.json").exists():
        raise FileExistsError("Preparation is one-shot; preserve existing proof")
    original = ROOT / values["reference_run"]
    if robust_source_hashes() != read(original / "source_sha256.json"):
        raise ValueError("Original B1 core sources differ")
    audit = audit_data(ref.base, full=True)
    if set(audit["splits"]) != {"train", "val"} or audit != read(original / "data_audit.json"):
        raise ValueError("Current decoded train/val audit differs from original B1")
    datasets = build_datasets(ref.base, ("train", "val"))
    stream = metadata_stream(datasets["train"], ref)
    history = read(original / "history.json")
    if len(history) != 100 or stream["sha256"] != history[0]["train"]["exposure_stream_sha256"]:
        raise ValueError("Original source/instance sampler identity failed")
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "data_audit.json", audit)
    for name, src in source_files().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    proof = {"status": "data_core_and_sampler_identity_passed", "created_utc": now(),
             "bound_sha256": bound_inputs(path, values, ref), "audit_sha256": digest(audit),
             "original_epoch1_stream": stream, "original_core_sha256": robust_source_hashes(),
             "held_out_opened": False, "research_training_started": False}
    atomic(output / "prepared.json", proof)
    return proof


def state_digest(value):
    """Device-independent exact digest including tensors, optimizer, scaler and RNG."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            h.update(canonical(["tensor", str(data.dtype), list(data.shape)]).encode())
            h.update(data.numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for k in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(k)
                visit(item[k])
        elif isinstance(item, (list, tuple)):
            h.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            h.update(canonical([type(item).__name__, item]).encode())
    visit(value)
    return h.hexdigest()


class LimitedLoader:
    def __init__(self, loader, count):
        self.loader, self.count, self.iterator = loader, count, None

    def __len__(self):
        return self.count

    def __iter__(self):
        self.iterator = iter(self.loader)
        for _ in range(self.count):
            yield next(self.iterator)


class SingleBatch:
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


class ObservedLoader:
    """Record completion after each original runner iteration, without changing it."""
    def __init__(self, loader, session, scaler, stage, epoch, guard, runtime, prior_steps=0):
        self.loader, self.session, self.scaler = loader, session, scaler
        self.stage, self.epoch, self.guard, self.runtime = stage, epoch, guard, runtime
        self.prior_steps = prior_steps
        self.completion_seconds, self.first_ids, self.last_ids = [], [], []
        self.skips = self.consecutive_skips = 0
        self.started = time.monotonic()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for step, batch in enumerate(self.loader, 1):
            self.guard()
            scale = float(self.scaler.get_scale()) if self.scaler else None
            if step == 1:
                self.first_ids = list(batch["sample_id"])
            yield batch
            self.last_ids = list(batch["sample_id"])
            current = float(self.scaler.get_scale()) if self.scaler else None
            skipped = self.scaler is not None and current < scale
            self.skips += int(skipped)
            self.consecutive_skips = self.consecutive_skips + 1 if skipped else 0
            elapsed = time.monotonic() - self.started
            self.completion_seconds.append(elapsed)
            alert = self.scaler is not None and (current < self.runtime["minimum_amp_scale"] or
                       self.consecutive_skips > self.runtime["max_consecutive_amp_skips"])
            atomic(self.session / "progress.json", {"stage": self.stage, "epoch": self.epoch,
                "step": step, "steps_in_stage": len(self),
                "completed_training_attempts": self.prior_steps + (step if self.scaler else 0),
                "amp_skips_in_stage": self.skips, "consecutive_amp_skips": self.consecutive_skips,
                "amp_scale": current, "numeric_alert": alert, "updated_utc": now(),
                "stage_elapsed_seconds": elapsed, "actual_source_batch_shape": list(batch["mask"].shape)})
            if alert:
                raise FloatingPointError("Frozen AMP scale/consecutive-skip protection failed")


def construct(ref):
    device = setup_runtime(ref.base)
    datasets = build_datasets(ref.base, ("train", "val"))
    train_dataset = TwoViewTrainingDataset(datasets["train"], ref.protocol, "perturbed")
    loader = make_loader(train_dataset, ref.base, training=True)
    val_loader = make_loader(datasets["val"], ref.base, training=False)
    model = move_model(build_model(ref.base), device, ref.base.values["runtime"]["memory_format"])
    settings = ref.base.values["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
        betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
    return device, model, optimizer, scaler, train_dataset, loader, val_loader


def training_epoch(ref, model, loader, loss, device, optimizer, scaler, guard):
    settings = ref.base.values["train"]
    return run_two_view_epoch(model, loader, loss, device, threshold=0.5,
        consistency_kind="none", consistency_weight=0.0, amp=settings["amp"],
        memory_format=ref.base.values["runtime"]["memory_format"], optimizer=optimizer,
        scaler=scaler, grad_clip_norm=settings["grad_clip_norm"], guard_callback=guard)


def training_state(model, optimizer, scaler, loader):
    return {"format_version": 1, "kind": "engineering_only", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(loader)}


def estimate_budget(seconds_per_step, values, spent, prior_training_seconds=0):
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0 or spent < 0:
        raise ValueError("Invalid measured throughput or cumulative spend")
    engineering = values["engineering"]
    estimate = (seconds_per_step * values["candidate"]["total_attempted_steps"] +
        values["candidate"]["epochs"] * engineering["per_epoch_validation_save_allowance_seconds"]) * (
            1 + engineering["allowance_fraction"])
    available = min(values["budgets"]["single_training_upper_bound_hours"] * 3600 - prior_training_seconds,
                    values["budgets"]["batch_upper_bound_gpu_hours"] * 3600 - spent)
    return {"estimated_training_seconds_with_margin": estimate, "available_seconds": available,
            "passed": estimate <= available, "measured_seconds_per_step": seconds_per_step}


def pilot(path, session, runtime, guard):
    path, values, ref, output = load_study(path)
    proof = read(output / "prepared.json")
    verify_bound(proof["bound_sha256"])
    rows, initial_hashes = [], []
    saved, next_batch = None, None
    q1_proof = None
    for label, loss in (("original_mean", ChangeDetectionLoss(**ref.base.values["loss"])),
                        ("proposed_top_half", EmpiricalTailLoss(**ref.base.values["loss"])),
                        ("proposed_repeat", EmpiricalTailLoss(**ref.base.values["loss"]))):
        guard()
        device, model, optimizer, scaler, dataset, loader, val_loader = construct(ref)
        initial = tensor_state_sha256(model.state_dict())
        initial_hashes.append(initial)
        if initial != values["candidate"]["initialization_sha256"]:
            raise ValueError("Original initialization tensor identity failed")
        torch.cuda.reset_peak_memory_stats(device)
        limited = LimitedLoader(loader, 32)
        observed = ObservedLoader(limited, session, scaler, label, 1, guard, runtime)
        report = training_epoch(ref, model, observed, loss, device, optimizer, scaler, guard)
        torch.cuda.synchronize(device)
        batch33 = next(limited.iterator)
        expected = proof["original_epoch1_stream"]["first_132_sample_ids"]
        if observed.first_ids != expected[:4] or list(batch33["sample_id"]) != expected[128:132]:
            raise ValueError("Actual original sampler prefix/next batch differs")
        boundary = training_state(model, optimizer, scaler, loader)
        boundary_sha = state_digest(boundary)
        intervals = [b-a for a, b in zip(observed.completion_seconds, observed.completion_seconds[1:])]
        row = {"label": label, "report": report, "initial_sha256": initial,
            "boundary_state_sha256": boundary_sha, "warmup_and_first_batch_seconds": observed.completion_seconds[0],
            "mean_seconds_per_step_after_first": sum(intervals)/len(intervals),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "actual_first_source_ids": observed.first_ids, "actual_next_batch_sha256": state_digest(batch33)}
        rows.append(row)
        if label == "proposed_top_half":
            save_checkpoint(session / "boundary.pt", boundary)
            saved, next_batch = read_checkpoint(session / "boundary.pt"), batch33
            # Actual CUDA loss values and gradients on the original real target.
            logits = torch.linspace(-5, 5, 8*256*256, device=device).reshape(8,1,256,256).requires_grad_()
            target = torch.cat((batch33["mask"], batch33["mask"])).to(device)
            a = ChangeDetectionLoss(**ref.base.values["loss"])(logits, target)
            b = EmpiricalTailLoss(**ref.base.values["loss"], q=1)(logits, target)
            ga, = torch.autograd.grad(a["total"], logits)
            gb, = torch.autograd.grad(b["total"], logits)
            q1_proof = all(torch.equal(a[k], b[k]) for k in a) and torch.equal(ga, gb)
            if not q1_proof:
                raise ValueError("q1 original mean CUDA value/gradient identity failed")
            continuation = ObservedLoader(SingleBatch(batch33), session, scaler, "next_update_original", 1, guard, runtime)
            next_report = training_epoch(ref, model, continuation, loss, device, optimizer, scaler, guard)
            next_sha = state_digest(training_state(model, optimizer, scaler, loader))
            row["next_update"] = {"report": next_report, "state_sha256": next_sha}
            del logits, target, a, b, ga, gb
        elif label == "proposed_repeat":
            if boundary_sha != rows[1]["boundary_state_sha256"] or state_digest(batch33) != state_digest(next_batch):
                raise ValueError("32-step independent repeat/model/optimizer/scaler/RNG/input identity failed")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            _restore_rng(saved["rng"], loader)
            replay = ObservedLoader(SingleBatch(batch33), session, scaler, "next_update_replay", 1, guard, runtime)
            replay_report = training_epoch(ref, model, replay, loss, device, optimizer, scaler, guard)
            if state_digest(training_state(model, optimizer, scaler, loader)) != rows[1]["next_update"]["state_sha256"]:
                raise ValueError("Exact next update replay failed")
            row["next_update_replay"] = replay_report
            tail = default_collate([dataset[i] for i in proof["original_epoch1_stream"]["last_indices"]])
            if list(tail["sample_id"]) != proof["original_epoch1_stream"]["last_sample_ids"]:
                raise ValueError("Actual last batch identity differs")
            tail_loader = ObservedLoader(SingleBatch(tail), session, scaler, "actual_epoch1_tail_shape", 1, guard, runtime)
            row["actual_tail_batch"] = {"shape": list(tail["mask"].shape),
                "report": training_epoch(ref, model, tail_loader, loss, device, optimizer, scaler, guard)}
        row["max_allocated_bytes_including_replays"] = torch.cuda.max_memory_allocated(device)
        row["max_reserved_bytes_including_replays"] = torch.cuda.max_memory_reserved(device)
        del boundary, model, optimizer, scaler, dataset, loader, val_loader, observed, limited, batch33
        torch.cuda.empty_cache()
    if len({r["report"]["exposure_stream_sha256"] for r in rows}) != 1:
        raise ValueError("Physical 32-step source/instance streams differ")
    report = {"status": "passed_engineering_only", "completed_utc": now(), "rows": rows,
        "q1_cuda_values_and_gradients_exact": q1_proof, "initial_hashes": initial_hashes,
        "repeat_full_state_exact": True, "next_update_full_state_exact": True,
        "actual_epoch1_tail_batch_size": proof["original_epoch1_stream"]["tail_source_batch_size"],
        "physical_optimizer_attempts": 99, "physical_training_pair_views": 99*8,
        "research_optimizer_updates": 0, "test_accessed": False,
        "environment": environment_record(ref.base, device), "bound_sha256": proof["bound_sha256"]}
    atomic(session / "engineering_report.json", report)
    return report


def seal(path, unittest_log):
    path, values, ref, output = load_study(path)
    proof = read(output / "prepared.json")
    verify_bound(proof["bound_sha256"])
    session = output / "pilot"
    owner = read(session / "owner_state.json")
    report = read(session / "engineering_report.json")
    log = Path(unittest_log).resolve()
    text = log.read_text(encoding="utf-8")
    import re
    count = re.search(r"Ran (\d+) tests? in ", text)
    if (not count or int(count.group(1)) < 450 or not text.rstrip().endswith("OK")
            or owner["status"] != "completed" or owner["worker_exit_code"] != 0
            or report["status"] != "passed_engineering_only" or read(output / "smoke.json")["status"] != "passed"
            or any(read(session / (role+"_state.json"))["status"] != "worker_exited" for role in ("guard", "watchdog"))):
        raise ValueError("Full tests, smoke and guarded engineering completion required")
    ledger = read(ROOT / values["managed_ledger"])
    if any(e.get("status") == "running" for e in ledger["entries"]):
        raise ValueError("A compute session is still active")
    seconds = max(row["mean_seconds_per_step_after_first"] for row in report["rows"][:2])
    prior = sum(e["charged_seconds"] for e in ledger["entries"]
                if e.get("kind") == "tail_risk_train_whole_worker_process")
    budget = estimate_budget(seconds, values, ledger["total_charged_seconds"], prior)
    if not budget["passed"]:
        atomic(output / "budget_rejection.json", budget)
        raise RuntimeError("Frozen throughput/budget gate failed; no training or precision substitution")
    bound = dict(proof["bound_sha256"])
    for p in [log, output / "smoke.json", output / "data_audit.json", output / "prepared.json",
              session / "engineering_report.json", session / "owner_state.json"]:
        bound[str(p.relative_to(ROOT)).replace("\\", "/")] = file_sha256(p)
    if values.get("infrastructure_repair"):
        io_path = output / "windows_file_io_stress.json"
        io = read(io_path)
        if (io["status"] != "passed" or not io["old_crt_reader_blocks_replacement"]
                or io["errors"] or io["writes"] < 100 or io["reads"] < 100):
            raise RuntimeError("The diagnosed Windows IO repair has not passed its admission checks")
        bound[str(io_path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(io_path)
    authority = {"status": "authorized_under_user_managed_training_request", "created_utc": now(),
        "training_flavor": FLAVOR, "risk_training_signature": digest({"study": values, "bound": proof["bound_sha256"]}),
        "bound_sha256": bound, "budget": budget, "full_test_count": int(count.group(1)),
        "test_open": False, "automatic_retry": False, "precision": "original_fp16"}
    target = output / "authorization.json"
    if target.exists():
        raise FileExistsError(target)
    atomic(target, authority)
    return authority


def train(path, session, runtime, guard):
    path, values, ref, output = load_study(path)
    authority = read(output / "authorization.json")
    verify_bound(authority["bound_sha256"])
    run = output / "candidate_seed42"
    run.mkdir(exist_ok=False)
    audit = read(output / "data_audit.json")
    device, model, optimizer, scaler, dataset, loader, val_loader = construct(ref)
    initial = tensor_state_sha256(model.state_dict())
    if initial != values["candidate"]["initialization_sha256"]:
        raise ValueError("Frozen original initialization changed")
    if len(dataset) != 7120 or len(loader) != 1780:
        raise ValueError("Original batch/exposure size changed")
    settings = ref.base.values["train"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=settings["min_learning_rate"])
    risk, original_loss = EmpiricalTailLoss(**ref.base.values["loss"]), ChangeDetectionLoss(**ref.base.values["loss"])
    reference_history = read(ROOT / values["reference_run"] / "history.json")
    atomic(run / "environment.json", environment_record(ref.base, device))
    atomic(run / "provenance.json", {"training_flavor": FLAVOR, "authority_sha256": file_sha256(output/"authorization.json"),
                                   "initial_model_tensor_sha256": initial, "test_open": False})
    history, best, best_epoch = [], -1.0, 0
    for epoch in range(1, 101):
        dataset.set_epoch(epoch)
        lr = optimizer.param_groups[0]["lr"]
        observed = ObservedLoader(loader, session, scaler, "candidate_train", epoch, guard, runtime, (epoch-1)*1780)
        training = training_epoch(ref, model, observed, risk, device, optimizer, scaler, guard)
        expected = reference_history[epoch-1]["train"]
        for field in ("exposure_stream_sha256", "exposure_counts", "source_samples", "pair_views"):
            if training[field] != expected[field]:
                raise ValueError(f"Original epoch {epoch} physical exposure mismatch: {field}")
        guard()
        val_observed = ObservedLoader(val_loader, session, None, "clean_validation", epoch, guard, runtime, epoch*1780)
        validation = run_epoch(model, val_observed, original_loss, device, threshold=0.5, amp=settings["amp"],
                               memory_format=ref.base.values["runtime"]["memory_format"])
        guard()
        improved = validation["metrics"]["f1"] > best
        if improved:
            best, best_epoch = validation["metrics"]["f1"], epoch
        scheduler.step()
        history.append({"epoch": epoch, "learning_rate": lr, "train": training, "val": validation})
        payload = {"format_version": 1, "training_flavor": FLAVOR, "kind": "research", "epoch": epoch,
            "best_epoch": best_epoch, "best_val_f1": best, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
            "config": ref.base.values, "reference_robust_config": dict(ref.values), "risk_config": values,
            "protocol_signature": protocol_signature(ref.base.values),
            "training_signature": authority["risk_training_signature"],
            "corruption_protocol_sha256": ref.protocol.fingerprint,
            "severity_review_sha256": file_sha256(ref.paths["severity_review"]),
            "initial_model_tensor_sha256": initial, "audit_sha256": digest(audit),
            "manifests": {s: {k:v for k,v in a.items() if k != "file_sha256"} for s,a in audit["splits"].items()},
            "file_sha256": {s:a["file_sha256"] for s,a in audit["splits"].items()},
            "source_sha256": {n:file_sha256(p) for n,p in source_files().items()},
            "authority_sha256": file_sha256(output / "authorization.json"),
            "rng": _rng_state(loader), "history": history}
        if improved:
            save_checkpoint(run / "best.pt", payload)
        save_checkpoint(run / "last.pt", payload)
        atomic(run / "history.json", history)
        atomic(run / "training_exposure_summary.json", _exposure_summary(history))
        atomic(run / "status.json", {"status": "running", "epoch": epoch, "epochs": 100,
            "best_epoch": best_epoch, "best_val_f1": best, "updated_utc": now(),
            "optimizer_steps": sum(r["train"]["optimizer_steps"] for r in history),
            "amp_skipped_steps": sum(r["train"]["amp_skipped_steps"] for r in history),
            "training_flavor": FLAVOR, "test_evaluated": False})
        print(f"risk/epoch={epoch} val_f1={validation['metrics']['f1']:.8f} best={best:.8f}", flush=True)
    result = read(run / "status.json")
    result.update(status="completed_training_requires_explicit_validation", updated_utc=now(), pair_views=1424000)
    atomic(run / "status.json", result)
    return result


def smoke(path):
    _, _, ref, output = load_study(path)
    # Small synthetic CPU optimization with the same model/loss; never a result.
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = build_model(ref.base).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    a, b = torch.randn(4,3,32,32), torch.randn(4,3,32,32)
    target = torch.randint(0, 2, (4,1,32,32)).float()
    logits = model(a,b)["logits"]
    losses = EmpiricalTailLoss(**ref.base.values["loss"])(logits,target)
    losses["total"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    result = {"status": "passed", "kind": "synthetic_cpu_engineering_only", "gradient_norm": float(norm),
              "loss": float(losses["total"]), "physical_gpu_compute": False, "research_result": False}
    atomic(output / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "seal"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unittest-log", type=Path)
    args = parser.parse_args()
    result = seal(args.config, args.unittest_log) if args.action == "seal" else globals()[args.action](args.config)
    print(json.dumps({k:v for k,v in result.items() if k not in ("bound_sha256", "original_core_sha256", "original_epoch1_stream")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
