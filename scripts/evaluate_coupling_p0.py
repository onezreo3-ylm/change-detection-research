"""Hash-sealed, train-only frozen-checkpoint P0 inference; never trains a model."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from pcdc.config import load_config
from scripts.coupling_diagnostic import (four_cell_diagnostic, image_sha256,
    metrics_from_counts, pool_counts, pool_transitions, validate_marginal_hashes)
from pcdc.data import _read_mask, _read_rgb, file_sha256
from pcdc.engine import check_checkpoint, move_model
from pcdc.model import build_model
from pcdc.perturbations import corrupt_rgb
from pcdc.runtime import environment_record, setup_runtime, write_json


FAMILIES = {"brightness", "contrast", "gamma", "color_bias"}
CELLS = {"uu": ("au", "bu"), "uv": ("au", "bv"),
         "vu": ("av", "bu"), "vv": ("av", "bv")}


def checked_hash(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Sealed file changed or missing: {path}")
    return path


def validate_manifest(manifest, *, verify_files=True):
    """Reject non-train paths before opening any image, including their headers."""
    s = manifest["settings"]
    if (manifest.get("schema_version") != 1 or manifest.get("split") != "train"
            or set(manifest["models"]) != {"b0", "b1", "b0e"}
            or set(s["families"]) != FAMILIES or s["levels"] != [1, 3, 5]
            or s["threshold"] != .5 or not 0 < s["max_seconds"] <= 21600
            or s["temperature_stop_c"] != 85):
        raise ValueError("P0 frozen scope/settings mismatch")
    data_root = Path(manifest["data_root"]).resolve()
    train_root = (data_root / "train").resolve()
    if train_root != data_root / "train" or {p.lower() for p in train_root.parts} & {"val", "test"}:
        raise ValueError("Train root aliases another split or location")
    samples = manifest["samples"]
    ids = [sample["id"] for sample in samples]
    if (len(samples) != 64 or len(set(ids)) != 64 or len({x["name"] for x in samples}) != 64
            or sum(x["foreground_pixels"] > 0 for x in samples) != 16
            or manifest["expected_predictions"] != 9408):
        raise ValueError("Expected frozen 16-positive/48-empty 64-source P0")
    expected = {(f, l) for f in FAMILIES for l in (1, 3, 5)}
    for sample in samples:
        if not sample["id"].startswith("train/"):
            raise ValueError("Non-train sample identity")
        if sample["id"] != f"train/{sample['name']}@x0_y0":
            raise ValueError("P0 only reads the original whole 256px train tile")
        if (len(sample["conditions"]) != 12 or
                {(c["family"], c["level"]) for c in sample["conditions"]} != expected):
            raise ValueError("Incomplete or duplicated condition grid")
        for key, folder in (("image_a", "A"), ("image_b", "B"), ("mask", "label")):
            item = sample["files"][key]
            path = Path(item["path"]).resolve()
            if (path.parent != train_root / folder or path.name != sample["name"]
                    or path.suffix.lower() != ".tif"
                    or {p.lower() for p in path.parts} & {"val", "test"}):
                raise ValueError("Image path is outside the exact train folder or differs in identity")
        for condition in sample["conditions"]:
            hashes = condition["images"]
            marginal = {cell: [hashes[a], hashes[b]] for cell, (a, b) in CELLS.items()}
            if marginal != condition["marginal_hashes"]:
                raise ValueError("Recorded marginal hashes differ from image hashes")
            validate_marginal_hashes(marginal)
    if verify_files:
        checked_hash(manifest["config_path"], manifest["config_sha256"])
        bindings = manifest["source_hashes"]
        for required in (Path(__file__).resolve(), Path(manifest["base_config"]).resolve(),
                         Path(__file__).resolve().parents[1] / "scripts/coupling_diagnostic.py"):
            if str(required) not in bindings and required.as_posix() not in bindings:
                raise ValueError(f"Required evaluator code absent from seal: {required}")
        for path, sha in bindings.items():
            checked_hash(path, sha)
        for model in manifest["models"].values():
            checked_hash(model["config"], model["config_sha256"])
            checked_hash(model["checkpoint"], model["checkpoint_sha256"])
    return train_root


def validate_review(manifest, manifest_path, review_path):
    if review_path is None:
        raise ValueError("Train-only sampled visibility acceptance is required before scoring")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (review.get("status") != "accepted_train_only_sampled_visibility"
            or review.get("manifest_sha256") != file_sha256(Path(manifest_path))
            or not review.get("reviewer") or not review.get("limitations")):
        raise ValueError("Review has not accepted this exact manifest")
    expected = {(str(Path(x["path"]).resolve()), x["sha256"]) for x in manifest["qa_images"]}
    observed = {(str(Path(x["path"]).resolve()), x["sha256"]) for x in review["reviewed_images"]
                if x.get("findings")}
    if not expected or expected != observed:
        raise ValueError("Review must cover every sealed original QA image")
    for path, sha in expected:
        checked_hash(path, sha)
    return review


def read_sample(sample, config):
    # Caller has validated ALL image paths before this first file read.
    paths = {k: checked_hash(v["path"], v["sha256"]) for k, v in sample["files"].items()}
    a, b = _read_rgb(paths["image_a"]), _read_rgb(paths["image_b"])
    d = config.values["data"]
    mask = _read_mask(paths["mask"], d["label_values"], d["label_decode"])
    if a.shape != (256, 256, 3) or b.shape != a.shape or mask.shape != (256, 256):
        raise ValueError("P0 requires original aligned 256-pixel tiles without resize")
    if int(mask.sum()) != sample["foreground_pixels"]:
        raise ValueError("Sealed foreground count differs")
    return a, b, torch.from_numpy(mask.copy()).float().unsqueeze(0).unsqueeze(0)


def render_views(a, b, condition):
    views = {}
    for date, raw in (("a", a), ("b", b)):
        for variant in ("u", "v"):
            instance = condition[variant]
            key = date + variant
            views[key] = corrupt_rgb(raw, condition["family"], instance["parameters"],
                                     seed=instance["seed"])
            if image_sha256(views[key]) != condition["images"][key]:
                raise ValueError("Regenerated pixel hash differs from the sealed manifest")
    hashes = {key: image_sha256(value) for key, value in views.items()}
    marginal = {cell: [hashes[x], hashes[y]] for cell, (x, y) in CELLS.items()}
    if marginal != condition["marginal_hashes"]:
        raise ValueError("Actual four-cell marginal identity changed")
    return views, validate_marginal_hashes(marginal)


class Guard:
    """Fail closed on sensor errors; checks before and after every CUDA forward."""
    def __init__(self, settings):
        self.settings = settings
        self.started = time.monotonic()
        self.temperatures = []
        self.predictions = 0

    def check(self):
        if time.monotonic() - self.started >= self.settings["max_seconds"]:
            raise RuntimeError("P0 budget exhausted; incomplete, manual continuation required")
        response = subprocess.run(
            [self.settings.get("nvidia_smi", "C:/Windows/System32/nvidia-smi.exe"),
             "-i", "0", "--query-gpu=uuid,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        uuid, value = [v.strip() for v in response.stdout.strip().split(",")]
        temperature = int(value)
        if uuid != self.settings["gpu_uuid"] or not 0 <= temperature < 120:
            raise RuntimeError("P0 GPU/sensor identity invalid")
        self.temperatures.append(temperature)
        if temperature >= self.settings["temperature_stop_c"]:
            raise RuntimeError("P0 temperature stop at >=85C; incomplete")


def summarize(rows):
    result = {}
    for model in ("b0", "b1", "b0e"):
        model_rows = [r for r in rows if r["model"] == model]
        strata = sorted({r["stratum"] for r in model_rows})
        sets = {"all": model_rows}
        sets.update({f"family:{f}": [r for r in model_rows if r["family"] == f]
                     for f in sorted(FAMILIES)})
        sets.update({f"stratum:{s}": [r for r in model_rows if r["stratum"] == s] for s in strata})
        sets.update({f"condition:{f}:{level}": [r for r in model_rows
                     if r["family"] == f and r["level"] == level]
                     for f in sorted(FAMILIES) for level in (1, 3, 5)})
        model_result = {}
        for label, selected in sets.items():
            groups = {}
            for group in ("D", "O"):
                counts = pool_counts(r["diagnostic"]["groups"][group]["counts"] for r in selected)
                transitions = pool_transitions(r["diagnostic"]["groups"][group]["transitions"] for r in selected)
                groups[group] = {"counts": counts, "metrics": metrics_from_counts(counts),
                                 "transitions": transitions}
            effects = {}
            for effect in ("m", "a", "b", "i"):
                effects[effect] = {}
                for area in ("all", "foreground", "background"):
                    stats = {k: sum(r["diagnostic"]["logit_effects"][effect][area][k]
                                   for r in selected) for k in ("pixels", "sum", "abs_sum", "square_sum")}
                    n = stats["pixels"]
                    stats.update(mean=stats["sum"] / n if n else None,
                                 mean_abs=stats["abs_sum"] / n if n else None,
                                 rms=math.sqrt(stats["square_sum"] / n) if n else None)
                    effects[effect][area] = stats
            delta = {k: groups["D"]["metrics"][k] - groups["O"]["metrics"][k]
                     for k in ("f1", "recall", "fpr")}
            model_result[label] = {"records": len(selected), "groups": groups,
                                   "D_minus_O": delta, "logit_effects": effects}
        clean_rows = {r["source_id"]: r["diagnostic"]["clean"]["counts"] for r in model_rows}
        clean_counts = pool_counts(clean_rows.values())
        model_result["clean_unique_sources"] = {"sources": len(clean_rows), "counts": clean_counts,
                                                 "metrics": metrics_from_counts(clean_counts)}
        result[model] = model_result
    passed = {}
    for model in ("b1", "b0e"):
        accepted = []
        for family in sorted(FAMILIES):
            row = result[model][f"family:{family}"]
            harmful = sum(row["groups"][g]["transitions"][k]
                          for g in ("D", "O") for k in ("tp_to_fn", "tn_to_fp"))
            if abs(row["D_minus_O"]["f1"]) >= .005 and harmful > 0:
                accepted.append(family)
        passed[model] = {"qualifying_families": accepted, "passed": len(accepted) >= 2}
    return {"aggregates": result, "gate": {"models": passed,
            "passed": all(x["passed"] for x in passed.values()),
            "harmful_exchange_definition": "Nonzero clean-to-cell TP-to-FN or TN-to-FP; not a D-to-O net-harm estimate",
            "interpretation": "Practical train-enriched screening only; not significance, causality or method authorization"}}


def run(manifest_path, review_path, output, *, smoke=False):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if not smoke:
        validate_review(manifest, manifest_path, review_path)
    config = load_config(manifest["base_config"])
    if (config.data_root != Path(manifest["data_root"]).resolve()
            or config.values["runtime"]["device"] not in ("cuda", "cuda:0")
            or config.run_test_after_training or not config.values["runtime"]["deterministic"]
            or config.values["evaluation"]["threshold"] != .5):
        raise ValueError("Base config violates train-only deterministic CUDA scope")
    output.mkdir(parents=True)
    guard = Guard(manifest["settings"])
    rows, checks = [], []
    main_predictions = 0
    report = {"status": "incomplete", "smoke_only": smoke, "train_only": True,
              "val_accessed": False, "test_accessed": False, "gradients_enabled": False,
              "training": False, "amp": False, "manifest_sha256": file_sha256(manifest_path),
              "review_sha256": file_sha256(Path(review_path)) if review_path and not smoke else None}
    try:
        guard.check()
        device = setup_runtime(config)
        report["environment"] = environment_record(config, device)
        d = config.values["data"]
        mean = torch.tensor(d["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(d["std"], dtype=torch.float32).view(3, 1, 1)
        def tensor(raw):
            value = torch.from_numpy(np.ascontiguousarray(raw.transpose(2, 0, 1))).float() / 255
            return ((value - mean) / std).unsqueeze(0).to(device, memory_format=torch.channels_last)
        for model_name, model_spec in manifest["models"].items():
            guard.check()
            checked_hash(model_spec["checkpoint"], model_spec["checkpoint_sha256"])
            checkpoint = torch.load(model_spec["checkpoint"], map_location="cpu", weights_only=True)
            if checkpoint.get("kind") != "research" or checkpoint.get("format_version") != 1:
                raise ValueError("A genuine research checkpoint is required")
            check_checkpoint(config, checkpoint)
            model = build_model(config, initialize_encoder=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            model = move_model(model, device, "channels_last").float().eval()
            model.requires_grad_(False)
            def infer(a, b):
                guard.check()
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                    logits = model(a, b)["logits"]
                torch.cuda.synchronize(device)
                guard.predictions += len(a)
                guard.check()
                if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite or non-FP32 inference")
                return logits.cpu()
            for sample_index, sample in enumerate(manifest["samples"][:1] if smoke else manifest["samples"]):
                raw_a, raw_b, target = read_sample(sample, config)
                a, b = tensor(raw_a), tensor(raw_b)
                clean = infer(a, b)
                main_predictions += 1
                if sample_index == 0:
                    repeated, swapped = infer(a, b), infer(b, a)
                    if not torch.equal(clean, repeated) or not torch.allclose(clean, swapped, atol=1e-5, rtol=1e-5):
                        raise RuntimeError("Repeat determinism or temporal symmetry self-check failed")
                    checks.append({"model": model_name, "source_id": sample["id"],
                                   "repeat_bitwise_equal": True, "temporal_swap_allclose": True,
                                   "max_swap_abs_error": (clean - swapped).abs().max().item()})
                if smoke:
                    continue
                for condition in sample["conditions"]:
                    views, marginal = render_views(raw_a, raw_b, condition)
                    inputs = {k: tensor(v) for k, v in views.items()}
                    logits = {cell: infer(inputs[x], inputs[y]) for cell, (x, y) in CELLS.items()}
                    main_predictions += 4
                    if marginal["fully_degenerate"] and any(not torch.equal(logits["uu"], z) for z in logits.values()):
                        raise RuntimeError("Degenerate four-cell equality self-check failed")
                    row = {"model": model_name, "source_id": sample["id"], "stratum": sample["stratum"],
                           "foreground_pixels": sample["foreground_pixels"], "family": condition["family"],
                           "level": condition["level"], "marginal_check": marginal,
                           "diagnostic": four_cell_diagnostic(clean, logits, target)}
                    rows.append(row)
                    with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                write_json(output / "progress.json", {"model": model_name, "sample": sample_index + 1,
                           "predictions": guard.predictions, "records": len(rows)}, overwrite=True)
            del model
            torch.cuda.empty_cache()
        if not smoke:
            if len(rows) != 2304 or guard.predictions != manifest["expected_predictions"] + 6:
                raise RuntimeError("Full P0 coverage was not reached")
            report.update(summarize(rows))
        report["status"] = "engineering_clean_cuda_smoke_passed" if smoke else "completed_train_p0_diagnostic"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(predictions=guard.predictions, main_predictions=main_predictions,
                      records=len(rows), self_checks=checks,
                      elapsed_seconds=time.monotonic() - guard.started,
                      maximum_observed_temperature_c=max(guard.temperatures) if guard.temperatures else None,
                      budget_seconds=manifest["settings"]["max_seconds"])
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Clean-only train engineering check; never outputs task scores")
    args = parser.parse_args()
    result = run(args.manifest, args.review, args.output, smoke=args.smoke)
    print(json.dumps({k: result[k] for k in ("status", "predictions", "val_accessed", "test_accessed")}))


if __name__ == "__main__":
    main()
