"""Exploratory saved-count screening and unchanged, train-only image review copies."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

KEYS = ("tn", "fp", "fn", "tp")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def counts(value, pixels):
    result = {key: value[key] for key in KEYS}
    if any(type(v) is not int or v < 0 for v in result.values()) or sum(result.values()) != pixels:
        raise ValueError("Nonnegative integer count/pixel conservation failed")
    return result


def require_same_sources(models):
    reference = next(iter(models.values()))
    for rows in models.values():
        if set(rows) != set(reference):
            raise ValueError("Model source coverage differs")
        for source, c in rows.items():
            old = reference[source]
            if c["tp"] + c["fn"] != old["tp"] + old["fn"]:
                raise ValueError("Source foreground support differs")


def load_validation(path, expected_models, pixels):
    result = {model: {} for model in expected_models}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model, source = row["model"], row["source"]
            if model not in result or not re.fullmatch(r"val/[^/\\]+\.tif", source):
                raise ValueError("Unexpected validation model/source")
            if source in result[model]:
                raise ValueError("Duplicate validation source")
            c = counts({key: int(row[key]) for key in KEYS}, pixels)
            if int(row["foreground_pixels"]) != c["tp"] + c["fn"]:
                raise ValueError("Validation foreground support differs")
            result[model][source] = c
    require_same_sources(result)
    return result


def load_train(path, expected_models, pixels, expected_conditions):
    result = {model: {} for model in expected_models}
    seen = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            model, source = row["model"], row["source_id"]
            if model not in result or not re.fullmatch(r"train/[^/\\]+\.tif@x0_y0", source):
                raise ValueError("Expected train-only model/source")
            key = (model, source)
            condition = (row["family"], row["level"])
            if condition not in expected_conditions or condition in seen.setdefault(key, set()):
                raise ValueError("Unexpected or duplicate train condition")
            seen[key].add(condition)
            c = counts(row["diagnostic"]["clean"]["counts"], pixels)
            if row["foreground_pixels"] != c["tp"] + c["fn"]:
                raise ValueError("Train foreground support differs")
            if source in result[model] and result[model][source] != c:
                raise ValueError("Repeated clean counts differ")
            result[model][source] = c
    if not seen or any(v != expected_conditions for v in seen.values()):
        raise ValueError("Incomplete train condition coverage")
    require_same_sources(result)
    return result


def summarize(rows, top_k):
    pooled = {key: sum(c[key] for c in rows.values()) for key in KEYS}
    positive = {s: c for s, c in rows.items() if c["tp"] + c["fn"] > 0}
    missed = {s: c for s, c in positive.items() if c["tp"] == 0}
    fn_missed = sum(c["fn"] for c in missed.values())
    denominator = 2 * pooled["tp"] + pooled["fp"] + pooled["fn"]
    fg = pooled["tp"] + pooled["fn"]
    return {
        "sources": len(rows), "positive_sources": len(positive), "counts": pooled,
        "f1": 2 * pooled["tp"] / denominator if denominator else 0.0,
        "recall": pooled["tp"] / fg if fg else 0.0,
        "positive_sources_zero_tp": len(missed),
        "positive_sources_zero_predictions": sum(c["tp"] + c["fp"] == 0 for c in positive.values()),
        "positive_source_zero_tp_fraction": len(missed) / len(positive) if positive else 0.0,
        "fn_in_zero_tp_sources": fn_missed,
        "fraction_fn_in_zero_tp_sources": fn_missed / pooled["fn"] if pooled["fn"] else 0.0,
        "empty_sources_with_fp": sum(c["fp"] > 0 for s, c in rows.items() if s not in positive),
        "top_fn_sources": sorted(positive, key=lambda s: (-positive[s]["fn"], s))[:top_k],
    }


def common_misses(models):
    require_same_sources(models)
    reference = next(iter(models.values()))
    positive = [s for s, c in reference.items() if c["tp"] + c["fn"] > 0]
    common = sorted(s for s in positive if all(rows[s]["tp"] == 0 for rows in models.values()))
    return {"positive_sources": len(positive), "all_models_zero_tp_sources": common,
            "all_models_zero_tp_count": len(common),
            "warning": "Same architecture and single seed; checkpoints are not independent replications."}


def train_file(root, sample, role):
    folder = {"image_a": "A", "image_b": "B", "mask": "label"}[role]
    name = sample["name"]
    if not re.fullmatch(r"[^/\\]+\.tif", name) or sample["id"] != f"train/{name}@x0_y0":
        raise ValueError("Invalid train source name/identity")
    expected = (Path(root) / "train" / folder / name).resolve()
    supplied = Path(sample["files"][role]["path"]).resolve()
    if expected != supplied or not expected.is_relative_to(Path(root).resolve()):
        raise ValueError("Refusing a non-train or unexpected raw data path")
    return expected


def components(mask, connectivity):
    if mask.ndim != 2 or connectivity not in (4, 8) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a strict 2D binary mask and connectivity 4 or 8")
    remaining = mask.astype(bool, copy=True)
    height, width = remaining.shape
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)]
    result = []
    for y, x in np.argwhere(remaining):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        stack = [(int(y), int(x))]
        area, ymin, ymax, xmin, xmax = 0, int(y), int(y), int(x), int(x)
        while stack:
            cy, cx = stack.pop()
            area += 1
            ymin, ymax = min(ymin, cy), max(ymax, cy)
            xmin, xmax = min(xmin, cx), max(xmax, cx)
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((ny, nx))
        result.append({"area": area, "bbox_xyxy_inclusive": [xmin, ymin, xmax, ymax],
                       "touches_tile_edge": ymin == 0 or xmin == 0 or ymax == height - 1 or xmax == width - 1})
    if sum(c["area"] for c in result) != int(mask.sum()):
        raise AssertionError("Connected component pixel conservation failed")
    return result


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg["kind"] != "exploratory_research_direction_screen":
        raise ValueError("Unexpected study kind")
    if any(cfg["limits"][k] for k in ("model_inference", "training", "raw_validation_access", "test_access")):
        raise ValueError("This tool permits only saved counts and train review")
    if not cfg["review"]["include_all_positive_sources"]:
        raise ValueError("The review must retain all positive sources in the original probe")
    root = (Path(path).resolve().parent / cfg["workspace_root"]).resolve()
    return cfg, root, root / cfg["output_dir"]


def analyze(path):
    started = time.monotonic()
    cfg, root, out = settings(path)
    files = [Path(path).resolve(), root / cfg["validation_csv"], root / cfg["source_manifest"], Path(__file__).resolve()]
    files += [root / item["path"] for item in cfg["train_records"]]
    before = {str(p): digest(p) for p in files}
    val = load_validation(root / cfg["validation_csv"], cfg["validation_models"], cfg["pixels_per_source"])
    train = {}
    conditions = {(family, level) for family in cfg["families"] for level in cfg["levels"]}
    for item in cfg["train_records"]:
        loaded = load_train(root / item["path"], item["models"], cfg["pixels_per_source"], conditions)
        if train.keys() & loaded.keys():
            raise ValueError("Repeated train model")
        train.update(loaded)
    require_same_sources(train)
    top_k = cfg["review"]["top_fn_sources"]
    val_stats = {m: summarize(rows, top_k) for m, rows in val.items()}
    train_stats = {m: summarize(rows, top_k) for m, rows in train.items()}
    for split, stats in (("validation", val_stats), ("train", train_stats)):
        if any(s["sources"] != cfg[f"{split}_sources"] or
               s["positive_sources"] != cfg[f"{split}_positive_sources"] for s in stats.values()):
            raise ValueError("Unexpected saved source coverage")
    manifest = json.loads((root / cfg["source_manifest"]).read_text(encoding="utf-8"))
    if manifest["split"] != "train":
        raise ValueError("Expected original train manifest")
    samples = manifest["samples"]
    if len(samples) != len({s["id"] for s in samples}) or {s["id"] for s in samples} != set(next(iter(train.values()))):
        raise ValueError("Original train manifest coverage differs")
    reference = next(iter(train.values()))
    for s in samples:
        if s["foreground_pixels"] != reference[s["id"]]["tp"] + reference[s["id"]]["fn"]:
            raise ValueError("Manifest foreground support differs")
        for role in cfg["review"]["panels"]:
            train_file(cfg["train_data_root"], s, role)
    positives = [s for s in samples if s["foreground_pixels"] > 0]
    empty = [s for s in samples if s["foreground_pixels"] == 0]
    rank = lambda s: hashlib.sha256((cfg["review"]["selection_salt"] + "|" + s["id"]).encode()).hexdigest()
    selected = sorted(positives, key=rank) + sorted(empty, key=rank)[:cfg["review"]["empty_sources"]]
    if len(positives) != cfg["train_positive_sources"] or len(empty) < cfg["review"]["empty_sources"]:
        raise ValueError("Incomplete review pool")
    review = [{k: s[k] for k in ("id", "name", "foreground_pixels", "files")} for s in selected]
    for s in review:
        s["saved_clean_counts"] = {m: rows[s["id"]] for m, rows in train.items()}
        s["zero_tp_models"] = [m for m, c in s["saved_clean_counts"].items()
                               if s["foreground_pixels"] > 0 and c["tp"] == 0]
        s["independent_annotation_verdict"] = "pending"
    if {str(p): digest(p) for p in files} != before:
        raise ValueError("Saved input changed during analysis")
    elapsed = time.monotonic() - started
    if elapsed > cfg["limits"]["cpu_wall_seconds"]:
        raise TimeoutError("CPU screening budget exceeded")
    report = {"status": "completed_exploratory_saved_count_screen", "created_utc": datetime.now(timezone.utc).isoformat(),
              "validation": val_stats, "train_fixed_enriched_subset": train_stats,
              "validation_common_misses": common_misses(val), "train_common_misses": common_misses(train),
              "review_samples": len(review), "elapsed_seconds": elapsed,
              "new_inference": False, "training": False, "raw_validation_read": False, "test_read": False,
              "limitations": ["Exploratory analysis after scores were known; no preregistered method gate.",
                              "A zero-TP tile is not a count of missed buildings or instances.",
                              "All seven train checkpoints use one architecture and seed; no independent replication.",
                              "Sixteen positive training probes are enriched; no prevalence or geographic significance.",
                              "Disagreement with a label does not establish annotation error."]}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "saved_count_screen.json", report)
    write_json(out / "review_samples.json", review)
    with (out / "review_pending.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "foreground_pixels", "zero_tp_models", "verdict", "reason", "reviewer"])
        writer.writerows((s["id"], s["foreground_pixels"], "|".join(s["zero_tp_models"]), "pending", "", "") for s in review)
    write_json(out / "input_manifest.json", {"inputs_sha256": before,
               "outputs_sha256": {name: digest(out / name) for name in
                                  ("saved_count_screen.json", "review_samples.json", "review_pending.csv")}})
    return report


def render_review(path):
    started = time.monotonic()
    cfg, root, out = settings(path)
    seal = json.loads((out / "input_manifest.json").read_text(encoding="utf-8"))
    for name, sha in seal["inputs_sha256"].items():
        if digest(name) != sha:
            raise ValueError("Screen source changed before train review")
    if digest(out / "review_samples.json") != seal["outputs_sha256"]["review_samples.json"]:
        raise ValueError("Review list changed")
    samples = json.loads((out / "review_samples.json").read_text(encoding="utf-8"))
    target = out / "train_review_native"
    target.mkdir(parents=True, exist_ok=False)
    results, preview_bytes = [], 0
    size = cfg["image_size"]
    header, footer = cfg["review"]["panel_header_height"], cfg["review"]["footer_height"]
    for sample in samples:
        if time.monotonic() - started > cfg["limits"]["cpu_wall_seconds"]:
            raise TimeoutError("CPU train review budget exceeded; partial outputs preserved")
        arrays = {}
        for role in cfg["review"]["panels"]:
            source = train_file(cfg["train_data_root"], sample, role)
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sample["files"][role]["sha256"]:
                raise ValueError("Original train file hash differs")
            with Image.open(io.BytesIO(raw)) as im:
                arrays[role] = np.array(im)
        mask = arrays["mask"]
        if mask.ndim == 3 and mask.shape[2] == 3 and np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 0], mask[..., 2]):
            mask = mask[..., 0]
        if mask.shape != (size, size) or not np.isin(mask, [0, 1]).all() or int(mask.sum()) != sample["foreground_pixels"]:
            raise ValueError("Original strict train label differs")
        cc = {str(n): components(mask, n) for n in cfg["review"]["component_connectivities"]}
        canvas = Image.new("RGB", (size * 3, size + header + footer), "white")
        draw = ImageDraw.Draw(canvas)
        for index, role in enumerate(cfg["review"]["panels"]):
            array = arrays[role]
            if role == "mask":
                low, high = cfg["review"]["mask_display_values"]
                array = np.where(mask, high, low).astype(np.uint8)
            elif array.shape != (size, size, 3) or array.dtype != np.uint8:
                raise ValueError("Expected unchanged RGB uint8 train images")
            canvas.paste(Image.fromarray(array).convert("RGB"), (index * size, header))
            title = {"image_a": "A 2017 - original pixels", "image_b": "B 2023 - original pixels",
                     "mask": "label: 0/1 displayed as 0/255"}[role]
            draw.text((index * size + 4, 8), title, fill="black")
        draw.text((4, header + size + 7), f"TRAIN {sample['name']} | label pixels={int(mask.sum())} | original scale 1:1", fill="black")
        destination = target / (Path(sample["name"]).stem + ".png")
        canvas.save(destination)
        payload = 4 * ((destination.stat().st_size + 2) // 3) + len("data:image/png;base64,")
        preview_bytes += payload
        results.append({"source": sample["id"], "foreground_pixels": int(mask.sum()),
                        "zero_tp_models": sample["zero_tp_models"], "components": cc,
                        "preview_path": str(destination.resolve()), "preview_sha256": digest(destination),
                        "estimated_data_url_bytes": payload, "original_files": sample["files"],
                        "independent_annotation_verdict": "pending"})
    common = set(json.loads((out / "saved_count_screen.json").read_text(encoding="utf-8"))["train_common_misses"]["all_models_zero_tp_sources"])
    report = {"status": "completed_train_only_native_review_preparation", "sources": results,
              "rendered_sources": len(results), "raw_validation_read": False, "test_read": False,
              "model_inference": False, "training": False, "elapsed_seconds": time.monotonic() - started,
              "common_miss_label_pixels": {s["source"]: s["foreground_pixels"] for s in results if s["source"] in common},
              "all_previews_estimated_data_url_bytes": preview_bytes,
              "chat_budget_bytes": cfg["review"]["chat_base64_budget_bytes"],
              "chat_max_images_per_call": cfg["review"]["chat_max_images_per_call"],
              "warning": "Preparation does not accept labels or establish object identity. Do not load all images into chat."}
    write_json(out / "train_review_preparation.json", report)
    return {k: v for k, v in report.items() if k != "sources"}


def smoke():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert len(components(mask, 4)) == 2 and len(components(mask, 8)) == 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "counts.csv"
        path.write_text("model,source,foreground_pixels,tn,fp,fn,tp\nx,val/a.tif,1,2,1,1,0\n", encoding="utf-8")
        result = summarize(load_validation(path, ["x"], 4)["x"], 1)
        assert result["positive_sources_zero_tp"] == 1 and result["positive_sources_zero_predictions"] == 0
    return {"smoke": "passed", "synthetic_engineering_only": True, "model_inference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "render-train-review", "smoke"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command != "smoke" and args.config is None:
        parser.error("--config is required")
    result = smoke() if args.command == "smoke" else (analyze(args.config) if args.command == "analyze" else render_review(args.config))
    if args.command == "analyze":
        result = {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                  "validation_zero_tp": {m: s["positive_sources_zero_tp"] for m, s in result["validation"].items()},
                  "train_common_misses": result["train_common_misses"], "review_samples": result["review_samples"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
