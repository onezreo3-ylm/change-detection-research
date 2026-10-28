"""Prepare isolated train windows and exercise the inference core; no launch CLI.

Real CUDA execution additionally needs an independently bound owner/guard adapter.
The prepare command does not grant that authority and never runs a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from pcdc.config import Config, UniqueKeyLoader, load_config, protocol_signature
from pcdc.data import _read_mask, _read_rgb, file_sha256
from pcdc.engine import read_checkpoint
from pcdc.model import build_model
from pcdc.runtime import setup_runtime
from scripts.tiling_windows import (ProbabilityMosaic, Window, compare_probability,
    count_metrics, require_archive_replay, score_probability, support_mask,
    window_plan, window_resource_gate)
from scripts.windows_shared_json import read_json


ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_train_records(base, prepared, *, split="train", expected_count=16):
    """Validate the entire whitelist before hashing/opening any source file."""
    d = base.values["data"]
    if (split != "train" or d["splits"] != {"train": "train", "val": "val", "test": "test"}
            or base.run_test_after_training or d["folders"] != {"image_a": "A", "image_b": "B", "mask": "label"}):
        raise ValueError("Only canonical, selected train A/B/label paths are admitted")
    train_root = base.data_root / "train"
    if train_root.resolve() != train_root:
        raise ValueError("Train directory alias/junction is not admitted")
    names = prepared["selected"]["train"]
    entries = prepared["records"]["train"]
    if (len(names) != expected_count or len(set(names)) != len(names) or names != sorted(names)
            or sorted(e["name"] for e in entries) != names or len(entries) != expected_count):
        raise ValueError("Exact fixed selected-train whitelist is required")
    result = {}
    for entry in entries:
        name = entry["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"train_[0-9]+[.]png", name)
                or set(entry["files"]) != set(d["folders"])):
            raise ValueError("Invalid selected train filename or source roles")
        files = {}
        for role, folder in d["folders"].items():
            info = entry["files"][role]
            expected = train_root / folder / name
            path = Path(info["path"])
            if (not path.is_absolute() or path.resolve() != expected
                    or expected.resolve() != expected or not re.fullmatch("[0-9a-f]{64}", info["sha256"])):
                raise ValueError("Selected source path alias, split escape or invalid hash")
            files[role] = {"path": str(expected), "sha256": info["sha256"]}
        result[name] = files
    return {name: result[name] for name in names}


def check_checkpoint_identity(checkpoint, base, expected, source_records):
    if (checkpoint.get("training_flavor") != "matched_robust_baseline_v1"
            or checkpoint.get("kind") != "research" or checkpoint.get("robust_mode") != expected["mode"]
            or checkpoint["config"]["experiment"]["seed"] != expected["seed"]
            or checkpoint.get("epoch") != checkpoint.get("best_epoch")
            or checkpoint.get("epoch") != expected["epoch"]
            or checkpoint.get("protocol_signature") != protocol_signature(base.values)
            or checkpoint["config"]["train"]["amp"] is not True
            or checkpoint["config"]["runtime"]["memory_format"] != "channels_last"):
        raise ValueError("Frozen matched checkpoint identity or precision differs")
    for name, files in source_records.items():
        if checkpoint["file_sha256"]["train"][name] != {r: f["sha256"] for r, f in files.items()}:
            raise ValueError("Checkpoint and selected train source identities differ")
    for value in checkpoint["model"].values():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Checkpoint model state must be finite tensors")


def load_settings(path):
    path = Path(path).resolve()
    cfg = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    root = (path.parent / cfg["workspace_root"]).resolve()
    if root != ROOT:
        raise ValueError("Research probe is bound to this workspace")
    seal = read_json(root / "runs/research_question_screen_20261025/artifact_sha256.json")
    key = path.relative_to(root).as_posix()
    if file_sha256(path) != seal["files"].get(key):
        raise ValueError("Frozen diagnostic configuration changed")
    base = load_config(root / cfg["base_config"])
    limits = cfg["constraints"]
    if (cfg["source_split"] != "train" or cfg["source_count"] != 16 or cfg["source_size"] != 1024
            or limits["raw_splits_allowed"] != ["train"] or limits["optimizer_updates"] != 0
            or limits["new_training_admitted"] or limits["new_method_admitted"]
            or limits["raw_validation_test_or_martigny_access"] or base.run_test_after_training):
        raise ValueError("Frozen train-only/no-training scope is required")
    if cfg["runtime"]["device"] != "cuda:0" or cfg["runtime"]["amp_dtype"] != "float16":
        raise ValueError("No device or precision substitution")
    for arm in cfg["arms"].values():
        window_plan(arm, cfg["source_size"])
    return path, cfg, base


def prepare(path):
    path, cfg, base = load_settings(path)
    prepared_path = ROOT / cfg["prepared_inventory"]
    screen_inputs = read_json(ROOT / "runs/research_question_screen_20261025/inputs_sha256.json")
    if file_sha256(prepared_path) != screen_inputs[prepared_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Fixed source selection changed")
    prior = read_json(prepared_path)
    records = validate_train_records(base, prior, expected_count=cfg["source_count"])
    out = ROOT / cfg["output_dir"]
    if out.exists():
        raise FileExistsError("Preserve existing probe preparation; inspect before another attempt")
    # All split/path checks above precede any raw selected-file read.
    bound = {str(path): file_sha256(path), str(base.source): file_sha256(base.source),
             str(prepared_path): file_sha256(prepared_path)}
    for files in records.values():
        for info in files.values():
            p = Path(info["path"])
            if file_sha256(p) != info["sha256"]:
                raise ValueError("Selected train source content changed: " + str(p))
            with Image.open(p) as image:
                if image.size != (cfg["source_size"],) * 2:
                    raise ValueError("Original selected train dimensions differ")
            bound[str(p)] = info["sha256"]
    models = {}
    for name, model_cfg in cfg["models"].items():
        expected = prior["models"][model_cfg["prepared_key"]]
        p = Path(expected["checkpoint"])
        if file_sha256(p) != expected["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint bytes changed")
        checkpoint = read_checkpoint(p)
        check_checkpoint_identity(checkpoint, base, expected, records)
        del checkpoint
        models[name] = expected
        bound[str(p)] = expected["checkpoint_sha256"]
    cached = {}
    inference = prepared_path.parent / "inference"
    for name in ["train_labels.npz"] + [f"{m}_train_{suffix}" for m in models
                                             for suffix in ["predictions.npz", "counts.json"]]:
        p = inference / name
        if file_sha256(p) != screen_inputs[p.relative_to(ROOT).as_posix()]:
            raise ValueError("Archived train replay evidence changed")
        cached[name] = str(p)
        bound[str(p)] = screen_inputs[p.relative_to(ROOT).as_posix()]
    with np.load(cached["train_labels.npz"], allow_pickle=False) as labels:
        if sorted(labels.files) != list(records):
            raise ValueError("Archived train label coverage differs")
        for name, files in records.items():
            mask = _read_mask(Path(files["mask"]["path"]), base.values["data"]["label_values"],
                              base.values["data"]["label_decode"])
            if not np.array_equal(mask, np.unpackbits(labels[name], axis=1, bitorder="big")):
                raise ValueError("Decoded original train label differs from archive")
    sources = list((ROOT / "pcdc").glob("*.py")) + [Path(__file__), ROOT / "scripts/tiling_windows.py",
        ROOT / "tests/test_tiling_windows.py", ROOT / cfg["protocol"],
        ROOT / "configs/smoke_tiling_windows_20261025.yaml"]
    for p in sources:
        bound[str(p)] = file_sha256(p)
    out.mkdir(exist_ok=False)
    for p in sources:
        target = out / "source_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    result = {"status": "train_only_prepared_not_gpu_authorized", "source_records": records,
              "models": models, "cached": cached, "bound_sha256": bound,
              "raw_train_files_hashed_and_headers_checked": 3 * len(records),
              "decoded_train_masks_exact_to_archive": len(records),
              "raw_validation_test_or_martigny_opened": False,
              "research_model_forwards": 0, "optimizer_updates": 0,
              "runtime_supervision_pending": True, "gpu_execution_authorized": False}
    write_new(out / "prepared.json", result)
    return {k: v for k, v in result.items() if k not in ("bound_sha256", "source_records", "cached", "models")}


class WindowPairs(Dataset):
    """No directory scans: only prevalidated per-source paths and exact boxes."""
    def __init__(self, records, arm, side, mean, std):
        self.records = records
        self.plan = window_plan(arm, side)[0]
        self.index = [(name, w) for name in records for w in self.plan]
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, window = self.index[index]
        files = self.records[name]
        result = {"source": name, "x": window.x, "y": window.y, "size": window.size}
        for role in ("image_a", "image_b"):
            image = _read_rgb(Path(files[role]["path"]), window.box)
            if image.shape != (window.size, window.size, 3):
                raise ValueError("Crop size changed; no implicit padding is allowed")
            result[role + "_pixels_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
            result[role] = (tensor - self.mean) / self.std
        return result


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def infer_windows(model, dataset, arm, side, *, device, batch_size, amp, check, progress):
    """Inference core; the real worker must supply its live supervisor checks."""
    device = torch.device(device)
    if any(m.training for m in model.modules()) or not callable(check) or not callable(progress) or (amp and device.type != "cuda"):
        raise ValueError("Eval, explicit supervisor callbacks and requested precision are required")
    mosaics = {name: ProbabilityMosaic(arm, side) for name in dataset.records}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    exposures, pairs = [], 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            check()
            a = batch["image_a"].to(device, memory_format=torch.channels_last)
            b = batch["image_b"].to(device, memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(a, b)["logits"]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits; no partial probability scoring")
            probabilities = logits.float().sigmoid().cpu().numpy()[:, 0]
            for i, name in enumerate(batch["source"]):
                window = Window(int(batch["x"][i]), int(batch["y"][i]), int(batch["size"][i]))
                mosaics[name].add(window, probabilities[i])
                exposures.append({"source": name, "box_xyxy": list(window.box),
                    "image_a_pixels_sha256": batch["image_a_pixels_sha256"][i],
                    "image_b_pixels_sha256": batch["image_b_pixels_sha256"][i]})
            pairs += len(probabilities)
            progress({"step": step, "pair_predictions": pairs})
            check()
    maps, coverage = {}, {}
    for name, mosaic in mosaics.items():
        check(); maps[name], coverage[name] = mosaic.finish()
    return maps, coverage, exposures


def summarize_maps(labels, maps, common_box, *, full_coverage, threshold):
    if set(labels) != set(maps) or not labels:
        raise ValueError("All fixed sources are required before computing pooled metrics")
    side = next(iter(labels.values())).shape[0]
    supports = {"common": support_mask(common_box, side)}
    if full_coverage:
        supports["full"] = np.ones((side, side), dtype=bool)
    rows, pooled = {}, {}
    for scope, support in supports.items():
        counts = {k: 0 for k in ["tp", "fp", "fn", "tn"]}
        for name in labels:
            score = score_probability(labels[name], maps[name], support, threshold)
            rows.setdefault(name, {})[scope] = score
            for k, v in score["counts"].items():
                counts[k] += v
        pooled[scope] = count_metrics(counts)
    return {"by_source": rows, "pooled": pooled}


def _repeat_and_swap(model, dataset, maps, *, device, amp, check):
    examples = [dataset[i] for i in range(4)]
    a = torch.stack([e["image_a"] for e in examples]).to(device, memory_format=torch.channels_last)
    b = torch.stack([e["image_b"] for e in examples]).to(device, memory_format=torch.channels_last)
    expected = np.stack([maps[e["source"]][e["y"]:e["y"] + e["size"], e["x"]:e["x"] + e["size"]]
                         for e in examples])
    pairs = 0
    with torch.inference_mode():
        for first, second in [(a, b), (b, a)]:
            check()
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                logits = model(first, second)["logits"]
            actual = logits.float().sigmoid().cpu().numpy()[:, 0]
            pairs += len(actual)
            if not np.array_equal(actual, expected):
                raise ValueError("Fixed-grid duplicate/date-swap probability replay differs")
            check()
    return {"duplicate_and_temporal_swap_exact": True, "extra_pair_predictions": pairs}


def evaluate_prepared(path, execution, *, check, progress):
    """Future bound worker entry; unavailable until an owner seals authorization.

    No CLI command calls this function. A real owner must establish independent
    live process handles and implement the supplied check/progress callbacks.
    """
    path, cfg, base = load_settings(path)
    out = ROOT / cfg["output_dir"]
    authority = read_json(out / "authorization.json")
    if (authority.get("status") != "sealed_train_only_tiling_probe"
            or authority.get("optimizer_updates") != 0 or not callable(check) or not callable(progress)):
        raise ValueError("Independent supervisor authorization is required before real inference")
    check()
    for filename, expected in authority["bound_sha256"].items():
        if file_sha256(Path(filename)) != expected:
            raise ValueError("Authorized input changed: " + filename)
    prepared = read_json(out / "prepared.json")
    if prepared.get("runtime_supervision_pending") is not True:
        raise ValueError("Expected the original engineering preparation")
    fixed = read_json(ROOT / cfg["prepared_inventory"])
    records = validate_train_records(base, fixed, expected_count=cfg["source_count"])
    if records != prepared["source_records"]:
        raise ValueError("Prepared whitelist changed")
    for filename, expected in prepared["bound_sha256"].items():
        check()
        if file_sha256(Path(filename)) != expected:
            raise ValueError("Prepared source/code changed: " + filename)
    execution = Path(execution).resolve()
    if execution != out / "inference":
        raise ValueError("Inference output must be a new directory of this probe")
    execution.mkdir(exist_ok=False)
    device = setup_runtime(base)
    if device.type != "cuda" or torch.cuda.current_device() != 0:
        raise RuntimeError("The requested original CUDA device is required")
    with np.load(prepared["cached"]["train_labels.npz"], allow_pickle=False) as z:
        labels = {name: np.unpackbits(z[name], axis=1, bitorder="big") for name in records}
    common = support_mask(cfg["evaluation"]["common_support_xyxy_exclusive"], cfg["source_size"])
    threshold = cfg["evaluation"]["threshold"]
    results, gate_rows, main_pairs, extra_pairs = {}, [], 0, 0
    started = time.monotonic()
    for model_name, identity in prepared["models"].items():
        check()
        checkpoint = read_checkpoint(Path(identity["checkpoint"]))
        check_checkpoint_identity(checkpoint, base, identity, records)
        model = build_model(base, initialize_encoder=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model = model.to(device, memory_format=torch.channels_last).eval().requires_grad_(False)
        before = state_digest(model)
        cached_counts = read_json(prepared["cached"][f"{model_name}_train_counts.json"])
        with np.load(prepared["cached"][f"{model_name}_train_predictions.npz"], allow_pickle=False) as z:
            cached_predictions = {name: np.unpackbits(z[name], axis=1, bitorder="big") for name in records}
        model_results, archive, grid, batch_differences = {}, None, None, None
        definitions = [("archive_grid256_batch32", cfg["arms"]["grid256"],
                        cfg["runtime"]["archived_grid_replay_batch_size"])] + [
            (name, arm, cfg["runtime"]["batch_size_all_comparison_arms"]) for name, arm in cfg["arms"].items()]
        for arm_name, arm, batch_size in definitions:
            ds = WindowPairs(records, arm, cfg["source_size"], base.values["data"]["mean"], base.values["data"]["std"])
            base_pairs = main_pairs
            def report(update):
                progress(dict(update, stage="fixed_train_tiling_inference", model=model_name, arm=arm_name,
                              main_pair_predictions=base_pairs + update["pair_predictions"],
                              extra_pair_predictions=extra_pairs, optimizer_updates=0, numeric_alert=False))
            maps, coverage, exposures = infer_windows(model, ds, arm, cfg["source_size"], device=device,
                batch_size=batch_size, amp=True, check=check, progress=report)
            main_pairs += len(exposures)
            write_new(execution / f"{model_name}_{arm_name}_exposures.json", exposures)
            np.savez_compressed(execution / f"{model_name}_{arm_name}_probabilities.npz", **maps)
            # Coverage is explicit even for the alternate arm's undefined exterior.
            np.savez_compressed(execution / f"{model_name}_{arm_name}_coverage.npz", **coverage)
            model_results[arm_name] = summarize_maps(labels, maps, cfg["evaluation"]["common_support_xyxy_exclusive"],
                full_coverage="support_xyxy_exclusive" not in arm, threshold=threshold)
            if arm_name == "archive_grid256_batch32":
                for name in records:
                    require_archive_replay(maps[name], labels[name], cached_predictions[name], cached_counts[name], threshold)
                archive = maps
                model_results[arm_name]["binary_and_confusion_replay_exact"] = True
            elif arm_name == "grid256":
                grid = maps
                batch_differences = {name: compare_probability(archive[name], grid[name], common, threshold) for name in records}
                model_results[arm_name]["batch4_vs32_by_source"] = batch_differences
                repeated = _repeat_and_swap(model, ds, grid, device=device, amp=True, check=check)
                extra_pairs += repeated["extra_pair_predictions"]
                model_results[arm_name]["self_checks"] = repeated
                archive = None
            else:
                comparisons = {name: compare_probability(grid[name], maps[name], common, threshold) for name in records}
                model_results[arm_name]["vs_grid256_by_source"] = comparisons
                if arm_name == "alternate_origins256":
                    gate_rows.extend({"model": model_name, "source": name,
                        "alternate_disagreement_fraction": comparisons[name]["binary_disagreement_fraction"],
                        "batch_disagreement_fraction": batch_differences[name]["binary_disagreement_fraction"]} for name in records)
            check()
        if before != state_digest(model):
            raise ValueError("Real inference changed model parameters or buffers")
        model_results["model_state_unchanged"] = True
        results[model_name] = model_results
        del model, maps, grid
    if main_pairs != cfg["budget"]["primary_patch_pair_predictions_maximum"]:
        raise ValueError("Frozen patch-pair exposure count differs")
    gate = window_resource_gate(gate_rows, list(prepared["models"]), list(records), cfg["resource_gate"])
    result = {"status": "completed_train_only_diagnostic", "models": results, "resource_gate": gate,
              "main_pair_predictions": main_pairs, "extra_pair_predictions": extra_pairs,
              "tile256_pixel_equivalent_main_predictions": cfg["budget"]["tile256_pixel_equivalent_predictions"],
              "source_count": len(records), "optimizer_updates": 0, "raw_validation_or_heldout_access": False,
              "new_method_or_training_admitted": False, "seconds": time.monotonic() - started}
    write_new(execution / "analysis.json", result)
    return result


def smoke(path):
    cfg = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if cfg["synthetic_only"] is not True or cfg["device"] != "cpu" or cfg["amp"] is not False:
        raise ValueError("This explicitly CPU FP32 synthetic smoke cannot stand in for a real probe")
    out = ROOT / cfg["output_dir"]
    out.mkdir(exist_ok=False)
    base = load_config(ROOT / cfg["base_config"])
    values = copy.deepcopy(base.values)
    values["experiment"].update(kind="smoke", seed=cfg["seed"])
    values["runtime"].update(device="cpu")
    values["train"]["amp"] = False
    torch.set_num_threads(cfg["num_threads"])
    torch.manual_seed(cfg["seed"]); torch.use_deterministic_algorithms(True)
    model = build_model(Config(values, base.source), initialize_encoder=False).eval().to(memory_format=torch.channels_last)
    before = state_digest(model)
    rng = np.random.default_rng(cfg["seed"])
    records, masks = {}, {}
    side = cfg["source_size"]
    for i in range(cfg["source_count"]):
        name = f"train_{i}.png"
        a = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
        b, label = a.copy(), np.zeros((side, side), dtype=np.uint8)
        label[side // 4:side // 2, side // 4:3 * side // 4] = 1
        b[label == 1] = 255 - b[label == 1]
        masks[name] = label; records[name] = {}
        for role, array in [("image_a", a), ("image_b", b), ("mask", label * 255)]:
            p = out / "synthetic_data" / "train" / role / name
            p.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(array).save(p)
            records[name][role] = {"path": str(p)}
    write_new(out / "synthetic_data/SYNTHETIC_ONLY.json", {"research_data": False})
    results, reference, total, callbacks = {}, None, 0, []
    for name, arm in cfg["arms"].items():
        ds = WindowPairs(records, arm, side, values["data"]["mean"], values["data"]["std"])
        maps, coverage, exposures = infer_windows(model, ds, arm, side, device="cpu", batch_size=cfg["batch_size"],
            amp=False, check=lambda: None, progress=callbacks.append)
        total += len(exposures)
        common = support_mask(cfg["common_support_xyxy_exclusive"], side)
        if reference is None:
            reference = maps
        for source in records:
            score_probability(masks[source], maps[source], common)
            compare_probability(reference[source], maps[source], common)
        results[name] = {"pair_predictions": len(exposures), "finite_common_support": True,
                         "coverage_verified": True, "uncovered_pixels_not_scored": True}
    after = state_digest(model)
    if before != after:
        raise ValueError("Smoke modified model parameters or buffers")
    result = {"status": "passed", "synthetic_only": True, "device": "cpu", "amp": False,
              "same_resnet18_fpn_width": values["model"]["decoder_channels"], "arms": results,
              "pair_predictions": total, "model_state_unchanged": True,
              "optimizer_updates": 0, "real_cuda_or_archive_replay_claim": False}
    write_new(out / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.config), ensure_ascii=False))


if __name__ == "__main__":
    main()
