"""Prepare candidate val manifests and TRAIN-only visual QA, without inference.

Run: python -m pcdc.prepare_corruptions --config configs/corruption_protocol_v1.yaml
Never reads held-out test image/label files, changes the source data, trains a
model, chooses severities from scores, or declares the candidate accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageDraw, features
import yaml

from .config import protocol_signature
from .corruption_protocol import (KINDS, CorruptionProtocol, apply_instance, canonical,
                                  keyed_seed, load_protocol, make_instance, sample_ids,
                                  validation_instances)
from .data import LEVIRDataset, PairRecord, RawPair, file_sha256
from .engine import check_checkpoint, read_checkpoint, source_files, source_hashes
from .perturbations import make_injection_mask
from .runtime import write_json


def _check_inventory(dataset: LEVIRDataset, checkpoint: dict[str, Any]) -> None:
    if dataset.manifest() != checkpoint["manifests"][dataset.split]:
        raise ValueError(f"{dataset.split} inventory differs from the frozen baseline")


def _check_content(records: list[PairRecord], split: str, checkpoint: dict[str, Any]) -> dict:
    verified = {}
    for record in records:
        expected = checkpoint["file_sha256"][split][record.name]
        actual = {key: file_sha256(getattr(record, key)) for key in ("image_a", "image_b", "mask")}
        if actual != expected:
            raise ValueError(f"Changed source content: {split}/{record.name}")
        verified[record.name] = actual
    return verified


def select_calibration_tiles(dataset: LEVIRDataset, protocol: CorruptionProtocol) -> list[int]:
    """One tile from each of N source pairs, selected before reading any labels."""
    if dataset.split != "train":
        raise ValueError("Severity calibration is restricted to the training split")
    settings = protocol.values["calibration"]
    count, seed = settings["source_pairs"], settings["selection_seed"]
    if count > len(dataset.records):
        raise ValueError("Not enough distinct training source pairs for calibration")
    order = sorted(range(len(dataset.records)),
                   key=lambda i: (keyed_seed(seed, dataset.records[i].name), dataset.records[i].name))
    selected = []
    for record_index in order[:count]:
        candidates = [i for i, (r, _, _) in enumerate(dataset.tiles) if r == record_index]
        choice = keyed_seed(seed, dataset.records[record_index].name, "tile") % len(candidates)
        selected.append(candidates[choice])
    return selected


def _raw_tile(dataset: LEVIRDataset, index: int) -> RawPair:
    r, x, y = dataset.tiles[index]
    return dataset.read_pair(dataset.records[r], (x, y, x + dataset.tile_size, y + dataset.tile_size))


def _thumbnail(array: np.ndarray, pixels: int) -> Image.Image:
    # Display only. The experiment operator always sees the original tile size.
    return Image.fromarray(array).convert("RGB").resize((pixels, pixels), Image.Resampling.NEAREST)


def calibration_previews(protocol: CorruptionProtocol, dataset: LEVIRDataset,
                         indices: list[int], output: Path) -> dict[str, Any]:
    settings = protocol.values["calibration"]
    pixels = settings["preview_tile_pixels"]
    levels = settings["preview_levels"]
    ids = sample_ids(dataset)
    pairs = [_raw_tile(dataset, index) for index in indices]
    previews = output / "previews"
    full_size = output / "calibration_tiles"
    previews.mkdir(exist_ok=False)
    full_size.mkdir(exist_ok=False)
    for row, pair in enumerate(pairs):
        for suffix, array in (("A", pair.image_a), ("B", pair.image_b), ("label", pair.mask * 255)):
            Image.fromarray(array).save(full_size / f"sample{row}_{suffix}.png")
    records, sheets = [], []
    label_width, gap, row_header, top = 195, 6, 30, 65
    columns = ["clean A", "clean B", "semantic label", "injection alpha"] + [f"level {n}" for n in levels]
    width = label_width + (pixels + gap) * len(columns)
    height = top + (pixels + row_header + gap) * len(indices)
    for kind in KINDS:
        directions = sorted(protocol.values["corruptions"][kind])
        for spatial in protocol.values["spatial_modes"]:
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 7), f"CANDIDATE / TRAIN ONLY - {kind} - {spatial}", fill="black")
            draw.text((8, 24), f"Display thumbnails only; original {dataset.tile_size}px tiles are not resized by the operator.", fill="black")
            for column, title in enumerate(columns):
                draw.text((label_width + column * (pixels + gap), 46), title, fill="black")
            for row, (index, pair) in enumerate(zip(indices, pairs)):
                direction = directions[row % len(directions)]
                target = ("A", "B")[(row + (spatial == "local")) % 2]
                instance = make_instance(protocol, ids[index], kind, 1, stream="train",
                                         direction=direction, target_date=target, spatial=spatial)
                alpha = make_injection_mask(pair.mask.shape, instance["injection_mask"])
                originals = [pair.image_a.copy(), pair.image_b.copy(), pair.mask.copy()]
                source = pair.image_a if target == "A" else pair.image_b
                untouched = pair.image_b if target == "A" else pair.image_a
                tiles = [pair.image_a, pair.image_b, pair.mask * 255,
                         np.rint(alpha * 255).astype(np.uint8)]
                by_level = {}
                for level in range(1, 6):
                    current = make_instance(protocol, ids[index], kind, level, stream="train",
                                            direction=direction, target_date=target, spatial=spatial)
                    changed = apply_instance(pair, current)
                    transformed = changed.image_a if target == "A" else changed.image_b
                    other = changed.image_b if target == "A" else changed.image_a
                    invariants = {
                        "label_unchanged": bool(np.array_equal(changed.mask, pair.mask)),
                        "other_date_unchanged": bool(np.array_equal(other, untouched)),
                        "outside_support_unchanged": bool(np.array_equal(transformed[alpha == 0], source[alpha == 0])),
                        "inputs_not_mutated": all(np.array_equal(before, after) for before, after in
                                                  zip(originals, (pair.image_a, pair.image_b, pair.mask))),
                    }
                    if not all(invariants.values()):
                        raise AssertionError(f"Corruption invariant failed: {current}")
                    by_level[level] = transformed
                    extreme_path = None
                    if level == 5:
                        extreme_path = f"calibration_tiles/sample{row}_{kind}_{spatial}_l5.png"
                        Image.fromarray(transformed).save(output / extreme_path)
                    # Boundary occupancy is not the same thing as clipping or
                    # observability; the report deliberately uses precise names.
                    records.append({**current, **invariants,
                                    "injection_support_fraction": float((alpha > 0).mean()),
                                    "changed_rgb_pixel_fraction": float(np.any(transformed != source, axis=2).mean()),
                                    "mean_absolute_channel_delta_0_1": float(np.abs(transformed.astype(np.float64) - source).mean() / 255),
                                    "output_boundary_channel_fraction": float(((transformed == 0) | (transformed == 255)).mean()),
                                    "new_boundary_channel_fraction": float(((((transformed == 0) & (source != 0)) |
                                                                              ((transformed == 255) & (source != 255)))).mean()),
                                    "output_rgb_sha256": hashlib.sha256(transformed.tobytes()).hexdigest(),
                                    "full_size_level5_image": extreme_path})
                tiles.extend(by_level[level] for level in levels)
                y = top + row * (pixels + row_header + gap)
                draw.text((8, y), ids[index].replace("@", "\n@"), fill="black")
                draw.text((8, y + 36), f"direction: {direction}\ntarget: {target}", fill="black")
                for column, array in enumerate(tiles):
                    canvas.paste(_thumbnail(array, pixels), (label_width + column * (pixels + gap), y))
            relative = f"previews/{kind}_{spatial}.png"
            canvas.save(output / relative)
            sheets.append(relative)
            print(f"train-only preview: {kind}/{spatial}", flush=True)
    return {"scope": "train_only_candidate_visual_QA_not_research_scores",
            "selection_uses_labels": False, "source_pair_count": len(indices),
            "samples": [{"sample_id": ids[index], "index": index,
                         "source_prefix": f"calibration_tiles/sample{row}"}
                        for row, index in enumerate(indices)],
            "all_invariants_passed": True, "instances": records, "preview_sheets": sheets,
            "limitation": "Limited label-independent train sample QA does not establish observability for every source image."}


def prepare(protocol: CorruptionProtocol) -> dict[str, Any]:
    output = protocol.path("output_dir")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate preparation: {output}")
    sources = source_files()
    code_hashes = {name: file_sha256(path) for name, path in sources.items()}
    protocol_file_hash = file_sha256(protocol.source)
    base_file_hash = file_sha256(protocol.base.source)
    if load_protocol(protocol.source).fingerprint != protocol.fingerprint:
        raise ValueError("Protocol/base configuration changed after it was loaded")
    checkpoint_path = protocol.path("reference_checkpoint")
    checksum = file_sha256(checkpoint_path)
    if checksum != protocol.values["reference_checkpoint_sha256"]:
        raise ValueError("Reference baseline checkpoint content changed")
    checkpoint = read_checkpoint(checkpoint_path)
    check_checkpoint(protocol.base, checkpoint)
    with protocol.path("reference_audit").open(encoding="utf-8") as stream:
        audit = json.load(stream)
    # Use the original checkpoint's exact audit encoding, not this module's compact JSON.
    audit_digest = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    if audit_digest != checkpoint["audit_sha256"]:
        raise ValueError("Reference audit is not the audit bound to this baseline checkpoint")
    datasets = {split: LEVIRDataset(protocol.base, split) for split in ("train", "val")}
    for dataset in datasets.values():
        _check_inventory(dataset, checkpoint)
    indices = select_calibration_tiles(datasets["train"], protocol)
    selected_records = [datasets["train"].records[datasets["train"].tiles[index][0]] for index in indices]
    print("Checking frozen train-calibration and full val file SHA256 (no test files)...", flush=True)
    content = {"train_calibration": _check_content(selected_records, "train", checkpoint),
               "val": _check_content(datasets["val"].records, "val", checkpoint)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "preparing", "kind": "protocol_preparation"})
    try:
        shutil.copyfile(protocol.source, output / "candidate_source.yaml")
        resolved = dict(protocol.values)
        for key in ("base_config", "reference_checkpoint", "reference_audit", "output_dir"):
            resolved[key] = str(protocol.path(key))
        resolved["base_config"] = str(output / "base_config.yaml")
        with (output / "candidate_resolved.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(resolved, stream, sort_keys=False)
        with (output / "base_config.yaml").open("x", encoding="utf-8") as stream:
            base = json.loads(json.dumps(protocol.base.values))
            base["data"]["root"] = str(protocol.base.data_root)
            base["model"]["weights_cache"] = str(protocol.base.path(base["model"]["weights_cache"]))
            base["experiment"]["output_dir"] = str(protocol.base.output_dir)
            yaml.safe_dump(base, stream, sort_keys=False)
        for name, path in sources.items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_sha256(target) != code_hashes[name]:
                raise ValueError(f"Source changed while creating snapshot: {name}")
        write_json(output / "source_sha256.json", code_hashes)
        counts, strata_counts = Counter(), Counter()
        val_ids = sample_ids(datasets["val"])
        instances_path = output / "val_instances.jsonl"
        with instances_path.open("x", encoding="utf-8", newline="\n") as stream:
            for instance in validation_instances(protocol, val_ids):
                stream.write(canonical(instance) + "\n")
                condition = instance["condition_id"]
                counts[condition] += 1
                strata_counts["|".join((condition, instance["direction"], instance["target_date"],
                                        instance["spatial_mode"]))] += 1
        manifest = {"schema_version": 1, "status": "candidate_pending_review", "split": "val",
                    "protocol_sha256": protocol.fingerprint,
                    "base_protocol_signature": protocol_signature(protocol.base.values),
                    "base_config_sha256": base_file_hash,
                    "archived_base_config_sha256": file_sha256(output / "base_config.yaml"),
                    "protocol_file_sha256": protocol_file_hash,
                    "reference_checkpoint": str(checkpoint_path), "reference_checkpoint_sha256": checksum,
                    "source_sha256": code_hashes, "inventory": datasets["val"].manifest(),
                    "file_sha256": content["val"], "instance_file": instances_path.name,
                    "instance_file_sha256": file_sha256(instances_path), "condition_counts": dict(counts),
                    "stratum_counts": dict(strata_counts), "instance_count": sum(counts.values()),
                    "aggregation_rule": "Pool pixel confusion per cell first; average 45 cell F1s equally; report direction/date/support strata separately.",
                    "severity_pairing": "One direction/date/mask/noise field per tile/family is shared across all five levels.",
                    "test_manifest": "not_created_development_only"}
        write_json(output / "val_manifest.json", manifest)
        calibration = calibration_previews(protocol, datasets["train"], indices, output)
        calibration["verified_source_file_sha256"] = content["train_calibration"]
        write_json(output / "train_calibration.json", calibration)
        environment = {"python": sys.version, "executable": sys.executable,
                       "platform": platform.platform(), "numpy": np.__version__, "pillow": PIL.__version__,
                       "computation": "CPU image preparation only; no model or device substitution",
                       "jpeg_codec": features.version_codec("jpg"),
                       "libjpeg_turbo": features.version_feature("libjpeg_turbo")}
        write_json(output / "environment.json", environment)
        if (source_hashes() != code_hashes or file_sha256(protocol.source) != protocol_file_hash
                or file_sha256(protocol.base.source) != base_file_hash):
            raise ValueError("Source or configuration changed during preparation; candidate not completed")
        summary = {"status": "completed_preparation", "protocol_status": "candidate_pending_review",
                   "kind": "protocol_preparation", "protocol_sha256": protocol.fingerprint,
                   "output_dir": str(output), "validation_cells": len(counts),
                   "validation_tiles": len(val_ids), "validation_instances": sum(counts.values()),
                   "train_calibration_pairs": len(indices), "train_calibration_instances": len(calibration["instances"]),
                   "preview_sheets": len(calibration["preview_sheets"]), "all_invariants_passed": True,
                   "test_files_read": 0, "model_inference_run": False, "new_training_started": False,
                   "next_gate": "Review train-only severity examples before accepting and scoring the candidate benchmark."}
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "kind": "protocol_preparation",
                                           "error": f"{type(error).__name__}: {error}"}, overwrite=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(load_protocol(args.config)), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
