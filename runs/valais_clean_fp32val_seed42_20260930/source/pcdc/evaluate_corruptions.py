"""Frozen, validation-only scoring; candidate corruption scoring requires review.

``--preflight-only`` inspects the frozen bundle without a dataset or model.
``--clean-only`` verifies the clean validation checkpoint without corrupt scoring.
Neither path changes candidate acceptance or permits held-out test access.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Sequence, TYPE_CHECKING

from PIL import features
import torch
from torch import Tensor, nn
import yaml

from .config import Config
from .corruption_protocol import KINDS
from .data import file_sha256, make_loader
from .engine import move_model, source_files, source_hashes
from .model import build_model
from .runtime import environment_record, setup_runtime, write_json

if TYPE_CHECKING:
    from .frozen_validation import EvaluationConfig


COUNT_KEYS = ("tn", "fp", "fn", "tp")
METRIC_KEYS = ("precision", "recall", "f1", "iou", "oa")
STRATUM_KEYS = ("direction", "target_date", "spatial_mode")
_SAMPLE_ID = re.compile(r"^(val/.+)@x(0|[1-9][0-9]*)_y(0|[1-9][0-9]*)$")


def _checked_counts(counts: Sequence[int]) -> list[int]:
    values = list(counts)
    if len(values) != 4 or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Counts must be four nonnegative integers in TN/FP/FN/TP order")
    return values


def counts_report(counts: Sequence[int]) -> dict[str, Any]:
    """Keep raw denominators; an empty region has no measured metrics."""
    tn, fp, fn, tp = _checked_counts(counts)
    total, foreground, background = tn + fp + fn + tp, tp + fn, tn + fp

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    metrics = None if total == 0 else {
        "precision": divide(tp, tp + fp), "recall": divide(tp, foreground),
        "f1": divide(2 * tp, 2 * tp + fp + fn), "iou": divide(tp, tp + fp + fn),
        "oa": (tp + tn) / total, "foreground_recall": divide(tp, foreground),
        "background_fpr": divide(fp, background), "foreground_fn_rate": divide(fn, foreground),
    }
    return {"counts": dict(zip(COUNT_KEYS, (tn, fp, fn, tp))), "pixels": total,
            "foreground_pixels": foreground, "background_pixels": background, "metrics": metrics}


@torch.no_grad()
def per_sample_confusions(logits: Tensor, target: Tensor, *, threshold: float,
                          injection_support: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    """One prediction and deterministic int64 reductions, never CUDA bincount."""
    if (logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1
            or any(size == 0 for size in logits.shape)):
        raise ValueError("Expected matching nonempty N/1/H/W logits and binary targets")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between zero and one")
    if not torch.isfinite(logits).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Metric logits must be finite and targets must be binary")
    if injection_support is not None and (
            injection_support.shape != target.shape or injection_support.dtype != torch.bool
            or injection_support.device != target.device):
        raise ValueError("Injection support must be an aligned boolean tensor on the target device")
    predicted = logits.float().sigmoid() >= threshold
    truth = target == 1
    indicators = (~truth & ~predicted, ~truth & predicted, truth & ~predicted, truth & predicted)
    all_counts = torch.stack([part.flatten(1).sum(1, dtype=torch.int64)
                              for part in indicators], dim=1)
    if injection_support is None:
        return all_counts.cpu(), None
    inside = torch.stack([(part & injection_support).flatten(1).sum(1, dtype=torch.int64)
                           for part in indicators], dim=1)
    # A single small transfer; outside counts are the exact integer complement.
    transferred = torch.stack((all_counts, inside), dim=1).cpu()
    return transferred[:, 0], transferred[:, 1]


class ConfusionAccumulator:
    """A pixel-pooled sample group, with optional injection-region diagnostics."""

    def __init__(self, *, regional: bool = False):
        self.regional = regional
        self.samples = 0
        self.counts = [0, 0, 0, 0]
        self.inside = [0, 0, 0, 0]

    def add(self, counts: Sequence[int], inside: Sequence[int] | None = None) -> None:
        values = _checked_counts(counts)
        if (inside is not None) != self.regional:
            raise ValueError("Injection-region availability changed within a sample group")
        regional = _checked_counts(inside) if inside is not None else [0, 0, 0, 0]
        if any(part > whole for part, whole in zip(regional, values)):
            raise ValueError("Injection-region counts cannot exceed full-image counts")
        self.samples += 1
        self.counts = [previous + value for previous, value in zip(self.counts, values)]
        self.inside = [previous + value for previous, value in zip(self.inside, regional)]

    def report(self) -> dict[str, Any]:
        regions = None
        if self.regional:
            regions = {"injection_support": counts_report(self.inside),
                       "outside_injection": counts_report([a - b for a, b in zip(self.counts, self.inside)])}
        return {"samples": self.samples, **counts_report(self.counts), "regions": regions}


def _batch_values(batch: dict[str, Any], key: str, samples: int) -> list[Any]:
    value = batch[key]
    if isinstance(value, Tensor):
        value = value.cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != samples:
        raise ValueError(f"Invalid collated batch metadata: {key}")
    return list(value)


@torch.no_grad()
def run_condition(model: nn.Module, loader: Any, config: Config, device: torch.device,
                  guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    """Evaluate one clean or corrupted condition; no oracle metadata enters the model."""
    model.eval()
    started = time.monotonic()
    overall: ConfusionAccumulator | None = None
    joint: dict[str, ConfusionAccumulator] = {}
    marginal: dict[str, dict[str, ConfusionAccumulator]] = {key: {} for key in STRATUM_KEYS}
    sources: dict[str, ConfusionAccumulator] = {}
    seen_ids: set[str] = set()
    condition_id, kind, level = "clean", "clean", 0
    corruption_mode: bool | None = None
    transfer_format = (torch.channels_last if config.values["runtime"]["memory_format"] == "channels_last"
                       else torch.contiguous_format)
    for batch in loader:
        if guard_callback is not None:
            guard_callback()
        is_corrupted = "condition_id" in batch
        if corruption_mode is None:
            corruption_mode = is_corrupted
            overall = ConfusionAccumulator(regional=is_corrupted)
        if is_corrupted != corruption_mode:
            raise ValueError("A condition loader cannot mix clean and corrupted samples")
        batch_size = batch["mask"].shape[0]
        sample_ids = _batch_values(batch, "sample_id", batch_size)
        metadata = {key: _batch_values(batch, key, batch_size) for key in
                    ("condition_id", "kind", "level", *STRATUM_KEYS)} if is_corrupted else {}
        source_ids, stratum_labels = [], []
        # Reject malformed or mixed batches before even one model forward.
        for index, sample_id in enumerate(sample_ids):
            match = _SAMPLE_ID.fullmatch(sample_id) if isinstance(sample_id, str) else None
            if match is None or sample_id in seen_ids:
                raise ValueError("Condition requires unique, valid validation tile IDs")
            seen_ids.add(sample_id)
            source_ids.append(sample_id.rsplit("@x", 1)[0])
            if is_corrupted:
                current = (metadata["condition_id"][index], metadata["kind"][index], metadata["level"][index])
                if current[1] not in KINDS or type(current[2]) is not int or not 1 <= current[2] <= 5:
                    raise ValueError("Invalid corruption kind or severity level")
                if condition_id == "clean":
                    condition_id, kind, level = current
                if current != (condition_id, kind, level) or condition_id != f"{kind}_l{level}":
                    raise ValueError("A model batch/condition loader cannot mix corruption conditions")
                labels = [metadata[key][index] for key in STRATUM_KEYS]
                if (not all(isinstance(value, str) and value and "|" not in value for value in labels)
                        or labels[1] not in ("A", "B") or labels[2] not in ("global", "local")):
                    raise ValueError("Invalid corruption stratum metadata")
                stratum_labels.append(labels)
        a = batch["image_a"].to(device=device, non_blocking=True, memory_format=transfer_format)
        b = batch["image_b"].to(device=device, non_blocking=True, memory_format=transfer_format)
        target = batch["mask"].to(device=device, non_blocking=True)
        support = batch["injection_support"].to(device=device, non_blocking=True) if is_corrupted else None
        # Explicit positional RGB-only interface: labels, alpha, corruption type,
        # severity, seeds and clean references are never passed to the detector.
        with torch.autocast(device_type=device.type, enabled=config.values["train"]["amp"]):
            logits = model(a, b)["logits"]
        counts, inside = per_sample_confusions(
            logits, target, threshold=config.values["evaluation"]["threshold"], injection_support=support)
        for index, sample_id in enumerate(sample_ids):
            values, region = counts[index].tolist(), inside[index].tolist() if inside is not None else None
            assert overall is not None
            overall.add(values, region)
            source_id = source_ids[index]
            sources.setdefault(source_id, ConfusionAccumulator(regional=is_corrupted)).add(values, region)
            if is_corrupted:
                labels = stratum_labels[index]
                joint.setdefault("|".join(labels), ConfusionAccumulator(regional=True)).add(values, region)
                for key, value in zip(STRATUM_KEYS, labels):
                    marginal[key].setdefault(value, ConfusionAccumulator(regional=True)).add(values, region)
    if overall is None or overall.samples == 0:
        raise ValueError("Cannot evaluate an empty condition")
    if guard_callback is not None:
        guard_callback()
    return {"condition_id": condition_id, "kind": kind, "level": level, **overall.report(),
            "joint_strata": {key: value.report() for key, value in sorted(joint.items())},
            "marginals": {key: {label: value.report() for label, value in sorted(groups.items())}
                          for key, groups in marginal.items()},
            "by_source": {key: value.report() for key, value in sorted(sources.items())},
            "seconds": time.monotonic() - started}


def summarize_conditions(conditions: dict[str, dict[str, Any]],
                         expected_ids: Sequence[str]) -> dict[str, Any]:
    """An equal mean of cell F1s, never a pooled-condition or mean-tile F1."""
    expected = list(expected_ids)
    if not expected or len(set(expected)) != len(expected) or set(conditions) != set(expected):
        raise ValueError("Scoring must contain every frozen condition exactly once")
    f1s = []
    families: dict[str, list[float]] = defaultdict(list)
    for condition_id in expected:
        result = conditions[condition_id]
        metrics = result["metrics"]
        if metrics is None or result["samples"] == 0:
            raise ValueError("Cannot aggregate an empty condition")
        f1 = metrics["f1"]
        if not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError("Condition F1 must be finite and in [0, 1]")
        f1s.append(f1)
        families[result["kind"]].append(f1)
    return {"condition_count": len(expected), "corruption_mf1": sum(f1s) / len(f1s),
            "family_mf1": {kind: sum(values) / len(values) for kind, values in families.items()},
            "aggregation_rule": "Pool pixels within each condition, then average condition F1s equally."}


def _expected_clean(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise ValueError("Frozen evaluation requires the validation-selected best checkpoint")
    selected = [row for row in checkpoint["history"] if row["epoch"] == checkpoint["epoch"]]
    if len(selected) != 1 or selected[0]["val"]["metrics"]["f1"] != checkpoint["best_val_f1"]:
        raise ValueError("Checkpoint clean validation selection/history is inconsistent")
    return selected[0]["val"]


def _snapshot(output: Path, job: Any, hashes: dict[str, str], files: dict[str, Path]) -> None:
    shutil.copyfile(job.source, output / "evaluation_source.yaml")
    write_json(output / "evaluation_config.json", {"source": str(job.source), "values": job.values,
                                                   "resolved_output_dir": str(output)})
    resolved = json.loads(json.dumps(job.values))
    for key in ("protocol_config", "manifest", "acceptance", "output_dir"):
        if resolved.get(key) is not None:
            resolved[key] = str(job.path(key))
    with (output / "evaluation_resolved.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    for name, path in files.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_sha256(target) != hashes[name]:
            raise ValueError(f"Source changed during evaluator snapshot: {name}")
    write_json(output / "source_sha256.json", hashes)


def _bind_hash(bindings: dict[str, str], path: Path, expected: Any) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError(f"Missing or invalid validated input hash: {path}")
    key = str(path)
    if key in bindings and bindings[key] != expected:
        raise ValueError(f"Conflicting frozen input identities: {path}")
    bindings[key] = expected


def _bound_input_hashes(bundle: Any) -> dict[str, str]:
    """Use validated identities, never recapture current bytes as new baselines."""
    fingerprint, manifest = bundle.fingerprints, bundle.manifest
    directory = bundle.manifest_path.parent
    bindings: dict[str, str] = {}
    for path, expected in (
            (bundle.job.source, fingerprint.get("job_config_sha256")),
            (bundle.manifest_path, fingerprint.get("manifest_sha256")),
            (bundle.protocol.source, fingerprint.get("protocol_config_sha256")),
            (bundle.protocol.base.source, fingerprint.get("archived_base_config_sha256")),
            (directory / manifest["instance_file"], fingerprint.get("instances_sha256")),
            (bundle.protocol.path("reference_checkpoint"), fingerprint.get("checkpoint_sha256")),
            (directory / "environment.json", fingerprint.get("preparation_environment_sha256")),
            (directory / "candidate_source.yaml", manifest.get("protocol_file_sha256"))):
        _bind_hash(bindings, path, expected)
    # A missing review is legitimate only for preflight/clean-only; don't try to
    # hash a nonexistent path or manufacture an approval record for that case.
    if bundle.approval is not None:
        _bind_hash(bindings, bundle.job.path("acceptance"), fingerprint.get("acceptance_sha256"))
    for name, checksum in manifest["source_sha256"].items():
        _bind_hash(bindings, directory / "source" / name, checksum)
    return bindings


def _bind_validation_files(bindings: dict[str, str], bundle: Any, dataset: Any,
                           checkpoint: dict[str, Any]) -> None:
    if dataset.split != "val":
        raise ValueError("Only validation source files may enter scoring provenance")
    expected_content = bundle.manifest["file_sha256"]
    if expected_content != checkpoint["file_sha256"]["val"]:
        raise ValueError("Validation file identities differ between manifest and checkpoint")
    names = [record.name for record in dataset.records]
    if len(set(names)) != len(names) or set(names) != set(expected_content):
        raise ValueError("Validation record coverage differs from the frozen content audit")
    for record in dataset.records:
        expected = expected_content[record.name]
        if set(expected) != {"image_a", "image_b", "mask"}:
            raise ValueError(f"Incomplete frozen validation file identities: {record.name}")
        for key, checksum in expected.items():
            _bind_hash(bindings, getattr(record, key), checksum)


def _verify_input_hashes(bindings: dict[str, str]) -> None:
    for path, expected in bindings.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Frozen input content changed: {path}")


def preflight(job: EvaluationConfig) -> dict[str, Any]:
    from .frozen_validation import load_bundle

    bundle = load_bundle(job, require_review=False)
    return {"status": "preflight_passed", "kind": "engineering_preflight_not_scoring",
            "fingerprints": bundle.fingerprints, "approval_present": bundle.approval is not None,
            "approval_status": bundle.approval.get("status") if bundle.approval else None,
            "dataset_opened": False, "model_inference_run": False, "test_evaluated": False}


def evaluate(job: EvaluationConfig, *, clean_only: bool = False,
             guard_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    from .frozen_validation import FrozenInstanceDataset, build_validation_dataset, load_bundle

    if type(clean_only) is not bool:
        raise ValueError("clean_only must be boolean")
    # For corrupt scoring, approval is enforced before device/data access or writes.
    bundle = load_bundle(job, require_review=not clean_only)
    output = job.path("output_dir")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validation evaluation: {output}")
    input_hashes = _bound_input_hashes(bundle)
    _verify_input_hashes(input_hashes)
    files = source_files()
    code_hashes = {name: file_sha256(path) for name, path in files.items()}
    if any(code_hashes.get(name) != expected
           for name, expected in bundle.manifest["source_sha256"].items()):
        raise ValueError("Prepared source changed after frozen-bundle validation")
    dataset, checkpoint = build_validation_dataset(bundle)
    config = bundle.protocol.base
    if dataset.split != "val" or config.run_test_after_training:
        raise ValueError("Frozen corruption evaluation is restricted to development validation")
    _bind_validation_files(input_hashes, bundle, dataset, checkpoint)
    expected_clean = _expected_clean(checkpoint)
    _verify_input_hashes(input_hashes)
    device = setup_runtime(config)
    checkpoint_path = bundle.protocol.path("reference_checkpoint")
    mode = "clean_validation_only" if clean_only else "frozen_corruption_validation"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "evaluating_clean", "mode": mode,
                                       "test_evaluated": False, "training_started": False})
    try:
        _snapshot(output, job, code_hashes, files)
        if file_sha256(output / "evaluation_source.yaml") != input_hashes[str(job.source)]:
            raise ValueError("Evaluation configuration changed while it was archived")
        write_json(output / "provenance.json", {
            "fingerprints": bundle.fingerprints, "input_file_sha256": input_hashes,
            "approval": bundle.approval, "mode": mode,
            "prepared_source_sha256": bundle.manifest["source_sha256"],
            "training_source_sha256": checkpoint["source_sha256"],
            "evaluation_source_sha256": code_hashes, "selection": "validation_f1_only",
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": input_hashes[str(checkpoint_path)],
        })
        environment = environment_record(config, device)
        environment.update(jpeg_codec=features.version_codec("jpg"),
                           libjpeg_turbo=features.version_feature("libjpeg_turbo"))
        write_json(output / "environment.json", environment)
        model = move_model(build_model(config, initialize_encoder=False), device,
                           config.values["runtime"]["memory_format"])
        model.load_state_dict(checkpoint["model"], strict=True)
        _verify_input_hashes(input_hashes)
        print("Evaluating the frozen clean validation checkpoint...", flush=True)
        clean = run_condition(model, make_loader(dataset, config, training=False), config, device,
                              guard_callback=guard_callback)
        clean["matches_reference"] = (
            clean["samples"] == expected_clean["samples"]
            and all(clean["counts"][key] == expected_clean["metrics"][key] for key in COUNT_KEYS)
            and all(clean["metrics"][key] == expected_clean["metrics"][key] for key in METRIC_KEYS))
        clean["reference_metrics"] = expected_clean["metrics"]
        write_json(output / "clean_metrics.json", clean)
        if not clean["matches_reference"]:
            raise ValueError("Clean validation no longer exactly matches the frozen checkpoint reference")
        summary = {"status": "completed_clean_only" if clean_only else "completed",
                   "kind": checkpoint["kind"], "mode": mode, "split": "val",
                   "checkpoint_epoch": checkpoint["epoch"], "selection": "validation_f1_only",
                   "clean_f1": clean["metrics"]["f1"], "clean_matches_reference": True,
                   "corruption_scoring_run": not clean_only, "test_evaluated": False,
                   "training_started": False, "fingerprints": bundle.fingerprints,
                   "approval_status": bundle.approval.get("status") if bundle.approval else None,
                   "purpose": "clean_checkpoint_verification" if clean_only else "frozen_validation_degradation"}
        if not clean_only:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for instance in bundle.instances:
                grouped[instance["condition_id"]].append(instance)
            expected_ids = list(bundle.manifest["condition_counts"])
            if set(grouped) != set(expected_ids):
                raise ValueError("Instance condition coverage differs from the frozen manifest")
            results = {}
            (output / "conditions").mkdir()
            for condition_id in expected_ids:
                cell = FrozenInstanceDataset(dataset, grouped[condition_id])
                # Keep each condition's own eval-batch boundaries, including its
                # final partial batch. No cross-condition model batch is allowed.
                result = run_condition(model, make_loader(cell, config, training=False), config, device,
                                       guard_callback=guard_callback)
                if (result["condition_id"] != condition_id or
                        result["samples"] != bundle.manifest["condition_counts"][condition_id] or
                        result["pixels"] != result["samples"] * dataset.tile_size ** 2):
                    raise ValueError(f"Incomplete or mismatched condition: {condition_id}")
                expected_strata = {key[len(condition_id) + 1:]: value
                                   for key, value in bundle.manifest["stratum_counts"].items()
                                   if key.startswith(condition_id + "|")}
                actual_strata = {key: value["samples"] for key, value in result["joint_strata"].items()}
                if actual_strata != expected_strata:
                    raise ValueError(f"Stratum coverage differs from the manifest: {condition_id}")
                result["clean_reference_f1"] = clean["metrics"]["f1"]
                result["clean_minus_f1"] = clean["metrics"]["f1"] - result["metrics"]["f1"]
                write_json(output / "conditions" / f"{condition_id}.json", result)
                results[condition_id] = result
                write_json(output / "status.json", {"status": "evaluating_corruptions", "mode": mode,
                           "completed_conditions": len(results), "condition_count": len(expected_ids),
                           "test_evaluated": False, "training_started": False}, overwrite=True)
                print(f"{condition_id}: f1={result['metrics']['f1']:.6f} ({len(results)}/{len(expected_ids)})", flush=True)
            write_json(output / "conditions.json", results)
            summary.update(summarize_conditions(results, expected_ids))
            summary["clean_minus_corruption_mf1"] = summary["clean_f1"] - summary["corruption_mf1"]
        if source_hashes() != code_hashes:
            raise ValueError("Evaluation source changed during scoring")
        _verify_input_hashes(input_hashes)
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "mode": mode,
            "error": f"{type(error).__name__}: {error}", "test_evaluated": False, "training_started": False,
        }, overwrite=True)
        raise


def main() -> None:
    from .frozen_validation import load_evaluation_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()
    job = load_evaluation_config(args.config)
    result = preflight(job) if args.preflight_only else evaluate(job, clean_only=args.clean_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
