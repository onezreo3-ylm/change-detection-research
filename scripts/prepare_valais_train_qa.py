"""Deterministic TRAIN-only visual severity QA for a separate Valais candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.corruption_protocol import CorruptionProtocol, keyed_seed, load_protocol
from pcdc.data import LEVIRDataset, file_sha256
from pcdc.engine import check_checkpoint, read_checkpoint, source_hashes
from pcdc.prepare_corruptions import _check_content, calibration_previews
from pcdc.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "schema_version", "name", "status", "base_config", "source_corruption_config",
    "source_corruption_config_sha256", "reference_checkpoint",
    "reference_checkpoint_sha256", "independent_val_completion",
    "independent_val_completion_sha256", "train_label_balance_report",
    "train_label_balance_report_sha256", "output_dir", "selection", "seeds",
    "preview_levels", "preview_tile_pixels", "placement", "spatial_modes",
}


def _path(source: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def select_tiles(dataset: LEVIRDataset, positive_names: set[str], *, seed: int,
                 positive_count: int, empty_count: int) -> list[int]:
    if dataset.split != "train" or positive_count != empty_count or positive_count != 4:
        raise ValueError("Valais QA requires four positive/four empty train tiles")
    names = {record.name for record in dataset.records}
    if not positive_names or not positive_names < names:
        raise ValueError("Positive train manifest is empty or has unknown names")
    empty_names = names - positive_names
    positives = sorted(positive_names, key=lambda name: (keyed_seed(seed, "positive", name), name))[:4]
    negatives = sorted(empty_names, key=lambda name: (keyed_seed(seed, "empty", name), name))[:4]
    indices = {dataset.records[record].name: index
               for index, (record, _, _) in enumerate(dataset.tiles)}
    return [indices[name] for pair in zip(positives, negatives) for name in pair]


def load_candidate(path: Path, *, require_new: bool = True):
    values = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(values, dict) or set(values) != KEYS:
        raise ValueError("Unexpected Valais QA candidate keys")
    if (values["schema_version"] != 1
            or values["name"] != "valais_train_severity_qa_candidate_v1"
            or values["status"] != "candidate_pending_visual_review"
            or values["placement"] != "after_tiling_before_normalization"
            or values["spatial_modes"] != ["global", "local"]
            or values["selection"] != {
                "seed": 810207, "positive_tiles": 4, "empty_tiles": 4,
                "order": "alternating_positive_empty",
                "label_source": "frozen_train_positive_manifest_only"}
            or values["seeds"] != {"train": 510942, "val": 510943, "test": 510944}
            or values["preview_levels"] != [1, 3, 5]
            or values["preview_tile_pixels"] != 192):
        raise ValueError("Valais QA candidate settings differ")
    paths = {key: _path(path, values[key]) for key in (
        "base_config", "source_corruption_config", "reference_checkpoint",
        "independent_val_completion", "train_label_balance_report", "output_dir")}
    if paths["output_dir"] != ROOT / "runs" / "valais_train_severity_qa_candidate_v1_20261002":
        raise ValueError("Candidate output path differs")
    if require_new and paths["output_dir"].exists():
        raise FileExistsError(paths["output_dir"])
    for key in ("source_corruption_config", "reference_checkpoint",
                "independent_val_completion", "train_label_balance_report"):
        if file_sha256(paths[key]) != values[key + "_sha256"]:
            raise RuntimeError(f"Bound Valais candidate input changed: {key}")
    base = load_config(paths["base_config"])
    if base.run_test_after_training or base.values["train"]["amp"] is not False:
        raise RuntimeError("Valais QA requires full-FP32 val-only baseline")
    complete = json.loads(paths["independent_val_completion"].read_text(encoding="utf-8"))
    if (complete.get("status") != "guarded_independent_clean_val_reproduced"
            or complete.get("test_accessed") is not False
            or complete.get("best_checkpoint_sha256") != values["reference_checkpoint_sha256"]
            or complete.get("selected_f1", 0) <= 0):
        raise RuntimeError("Clean baseline gate not satisfied")
    checkpoint = read_checkpoint(paths["reference_checkpoint"])
    check_checkpoint(base, checkpoint)
    if (checkpoint["source_sha256"] != source_hashes()
            or checkpoint["epoch"] != complete["selected_epoch"]):
        raise RuntimeError("Selected baseline source/epoch differs")
    source = load_protocol(paths["source_corruption_config"])
    settings = copy.deepcopy(source.values)
    settings.update({
        "name": values["name"], "base_config": values["base_config"],
        "reference_checkpoint": values["reference_checkpoint"],
        "reference_checkpoint_sha256": values["reference_checkpoint_sha256"],
        "reference_audit": "../runs/valais_clean_full_fp32_seed42_20261002/data_audit.json",
        "output_dir": values["output_dir"], "seeds": values["seeds"],
        "placement": values["placement"], "spatial_modes": values["spatial_modes"],
    })
    settings["calibration"] = {
        "source_pairs": 8, "selection_seed": values["selection"]["seed"],
        "preview_levels": values["preview_levels"],
        "preview_tile_pixels": values["preview_tile_pixels"],
    }
    return values, paths, base, checkpoint, CorruptionProtocol(settings, path, base)


def prepare(path: Path) -> dict:
    values, paths, base, checkpoint, protocol = load_candidate(path)
    dataset = LEVIRDataset(base, "train")
    balance = json.loads(paths["train_label_balance_report"].read_text(encoding="utf-8"))
    if (balance.get("status") != "development_label_balance_only"
            or balance.get("test_accessed") is not False
            or balance.get("train", {}).get("pairs") != len(dataset.records)):
        raise RuntimeError("Frozen train-only label report differs")
    positive = {item["name"] for item in balance["train"]["positive_manifest"]}
    if len(positive) != balance["train"]["positive_pairs"]:
        raise RuntimeError("Positive train manifest count differs")
    selected = select_tiles(dataset, positive, seed=values["selection"]["seed"],
                            positive_count=4, empty_count=4)
    records = [dataset.records[dataset.tiles[index][0]] for index in selected]
    checked = _check_content(records, "train", checkpoint)
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    previews = calibration_previews(protocol, dataset, selected, output)
    ids = [previews["samples"][index]["sample_id"] for index in range(8)]
    if (len(previews["instances"]) != 8 * 9 * 2 * 5
            or not previews["all_invariants_passed"]
            or len(previews["preview_sheets"]) != 18):
        raise RuntimeError("Candidate QA coverage/invariants differ")
    report = {
        "schema_version": 1,
        "status": "candidate_pending_visual_review",
        "scope": "Valais Sion train only; engineering visual QA, not research scores",
        "model_inference_performed": False,
        "val_test_image_or_label_accessed": False,
        "source_corruption_config_sha256": values["source_corruption_config_sha256"],
        "candidate_config_sha256": file_sha256(path),
        "selected_checkpoint_sha256": values["reference_checkpoint_sha256"],
        "independent_val_completion_sha256": values["independent_val_completion_sha256"],
        "selected_train_file_sha256": checked,
        "selection": values["selection"],
        "selected_sample_ids": ids,
        "selected_label_class": ["positive" if i % 2 == 0 else "empty" for i in range(8)],
        "protocol_fingerprint": protocol.fingerprint,
        **previews,
    }
    write_json(output / "train_visual_qa.json", report)
    print(json.dumps({"status": report["status"], "instances": len(previews["instances"]),
                      "sheets": len(previews["preview_sheets"]),
                      "val_test_image_or_label_accessed": False}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.preflight_only:
        values, paths, base, checkpoint, protocol = load_candidate(path)
        print(json.dumps({"status": "preflight_passed", "checkpoint_epoch": checkpoint["epoch"],
                          "val_test_image_or_label_accessed": False}))
    else:
        prepare(path)


if __name__ == "__main__":
    main()
