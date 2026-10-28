"""Prespecified, TRAIN-only supplementary candidate visual QA; never inference.

Run as ``python -m scripts.supplement_train_qa --config <supplement YAML>``.
The original candidate, severities, implementation, manifest and acceptance state
are read-only references. This script only creates a new, non-overwriting QA run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import PIL
from PIL import features
import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.corruption_protocol import KINDS, CorruptionProtocol, keyed_seed, load_protocol, sample_ids
from pcdc.data import LEVIRDataset, RawPair, file_sha256
from pcdc.engine import check_checkpoint, read_checkpoint, source_files, source_hashes
from pcdc.perturbations import make_injection_mask
from pcdc.prepare_corruptions import (
    _check_content, _check_inventory, _raw_tile, calibration_previews, select_calibration_tiles,
)
from pcdc.runtime import write_json


ENVIRONMENT_FIELDS = ("python", "executable", "platform", "numpy", "pillow",
                      "jpeg_codec", "libjpeg_turbo")
_PATH_FIELDS = ("protocol_config", "reference_preparation", "output_dir")
_FIELDS = {"schema_version", "name", "status", *_PATH_FIELDS,
           "expected_protocol_sha256", "expected_checkpoint_sha256", "selection",
           "rendering", "verification"}


@dataclass(frozen=True)
class SupplementConfig:
    values: dict[str, Any]
    source: Path

    def path(self, key: str) -> Path:
        value = Path(self.values[key])
        return (value if value.is_absolute() else self.source.parent / value).resolve()


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Unexpected {label} fields; expected {sorted(expected)}")


def _integer(value: Any, label: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def load_supplement_config(path: str | Path) -> SupplementConfig:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        values = yaml.load(stream, Loader=UniqueKeyLoader)
    _keys(values, _FIELDS, "supplement")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise ValueError("Only supplement schema_version 1 is supported")
    if values["status"] != "awaiting_visual_review":
        raise ValueError("Supplementary QA cannot declare the candidate accepted")
    for key in ("name", *_PATH_FIELDS):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("expected_protocol_sha256", "expected_checkpoint_sha256"):
        value = values[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"{key} must be a lowercase SHA256")
    selection = values["selection"]
    _keys(selection, {"seed", "skip_source_pairs", "source_pairs", "tiles_per_source"}, "selection")
    for key, value in selection.items():
        _integer(value, f"selection.{key}", 0 if key == "seed" else 1)
    rendering = values["rendering"]
    _keys(rendering, {"group_size", "row_rotations", "preview_levels"}, "rendering")
    _integer(rendering["group_size"], "rendering.group_size")
    if rendering["group_size"] != 4 or rendering["row_rotations"] != [0, 1]:
        raise ValueError("This supplement uses groups of four and left row rotations [0, 1]")
    if any(type(value) is not int for value in rendering["row_rotations"]):
        raise ValueError("Row rotations must be integers, not booleans")
    if (rendering["preview_levels"] != [1, 3, 5]
            or any(type(value) is not int for value in rendering["preview_levels"])):
        raise ValueError("Display levels are [1, 3, 5]; numeric QA still covers all five")
    if selection["source_pairs"] % rendering["group_size"]:
        raise ValueError("Supplement source count must form complete four-source groups")
    verification = values["verification"]
    _keys(verification, {"frozen_source_count", "prepare_environment_fields"}, "verification")
    _integer(verification["frozen_source_count"], "verification.frozen_source_count")
    if verification["prepare_environment_fields"] != list(ENVIRONMENT_FIELDS):
        raise ValueError("All original preparation environment fields must be checked")
    return SupplementConfig(values, source)


def select_supplement_tiles(dataset: LEVIRDataset, *, seed: int, skip_source_pairs: int,
                            source_pairs: int, tiles_per_source: int) -> list[int]:
    """Extend the original hash order; do not read pixels, labels or scores."""
    if dataset.split != "train":
        raise ValueError("Supplementary severity QA is restricted to the training split")
    for key, value in (("seed", seed), ("skip_source_pairs", skip_source_pairs),
                       ("source_pairs", source_pairs), ("tiles_per_source", tiles_per_source)):
        _integer(value, key, 0 if key == "seed" else 1)
    names = [record.name for record in dataset.records]
    if len(names) != len(set(names)):
        raise ValueError("Training source filenames must be unique")
    if skip_source_pairs + source_pairs > len(names):
        raise ValueError("Not enough distinct training source pairs for the supplement")
    order = sorted(range(len(names)), key=lambda index: (keyed_seed(seed, names[index]), names[index]))
    selected = []
    for record_index in order[skip_source_pairs:skip_source_pairs + source_pairs]:
        candidates = [index for index, (record, _, _) in enumerate(dataset.tiles)
                      if record == record_index]
        if len(candidates) != tiles_per_source:
            raise ValueError(f"{names[record_index]} does not have the specified {tiles_per_source} tiles")
        choice = keyed_seed(seed, names[record_index], "tile") % len(candidates)
        selected.append(candidates[choice])
    return selected


def preview_batches(indices: list[int], *, group_size: int, row_rotations: list[int]) -> list[dict]:
    """Even four-row rotations flip target-date parity for every sample."""
    if (group_size != 4 or row_rotations != [0, 1]
            or any(type(value) is not int for value in row_rotations)
            or not indices or len(indices) % group_size or len(set(indices)) != len(indices)):
        raise ValueError("Expected distinct indices in complete groups of four, rotations [0, 1]")
    batches = []
    for group_index, start in enumerate(range(0, len(indices), group_size), 1):
        group = indices[start:start + group_size]
        for pass_index, rotation in enumerate(row_rotations, 1):
            batches.append({"batch_id": f"group{group_index:02d}_pass{pass_index}_rotation{rotation}",
                            "group": group_index, "pass": pass_index, "left_row_rotation": rotation,
                            "indices": group[rotation:] + group[:rotation]})
    return batches


def posthoc_overlap(mask: np.ndarray, injection_spec: dict[str, Any]) -> dict[str, Any]:
    """Semantic overlap is measured AFTER the label-blind injection is fixed."""
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("Posthoc analysis expects a decoded uint8 binary semantic mask")
    # Only shape and the already-recorded mask specification enter the generator.
    support = make_injection_mask(mask.shape, injection_spec) > 0
    positive = mask == 1
    positive_pixels = int(positive.sum())
    support_pixels = int(support.sum())
    overlap_pixels = int((positive & support).sum())
    return {"computed_posthoc_not_used_for_selection_or_injection": True,
            "positive_pixels": positive_pixels, "injection_support_pixels": support_pixels,
            "positive_pixels_in_injection_support": overlap_pixels,
            "positive_pixels_outside_injection_support": positive_pixels - overlap_pixels,
            "positive_coverage_fraction": overlap_pixels / positive_pixels if positive_pixels else None,
            "positive_fraction_of_support": overlap_pixels / support_pixels if support_pixels else None,
            "support_definition": "alpha > 0; not a causal feature-damage annotation"}


def annotate_report(report: dict[str, Any], pairs: dict[str, RawPair], batch_id: str) -> dict[str, Any]:
    """Keep the frozen preview helper's invariants and RGB digests intact."""
    instances = []
    for instance in report["instances"]:
        if instance["stream"] != "train" or instance["sample_id"] not in pairs:
            raise ValueError("Only the prespecified selected training samples may be analysed")
        instances.append({**instance, "batch_id": batch_id,
                          "posthoc_semantic_overlap": posthoc_overlap(
                              pairs[instance["sample_id"]].mask, instance["injection_mask"])})
    return {**report, "batch_id": batch_id, "instances": instances}


def verify_coverage(instances: list[dict], selected_ids: list[str]) -> dict[str, Any]:
    expected = {(sample_id, kind, spatial, target)
                for sample_id in selected_ids for kind in KINDS
                for spatial in ("global", "local") for target in ("A", "B")}
    levels = defaultdict(list)
    for instance in instances:
        key = (instance["sample_id"], instance["kind"], instance["spatial_mode"], instance["target_date"])
        levels[key].append(instance["level"])
    if set(levels) != expected or any(sorted(value) != [1, 2, 3, 4, 5] for value in levels.values()):
        raise AssertionError("Supplement must cover both dates at all five levels for each sample/family/support")
    return {"both_dates_per_sample_family_spatial_mode": True,
            "numeric_levels": [1, 2, 3, 4, 5], "instance_count": len(instances),
            "direction_rotation_note": "Row rotation also rotates multi-direction families; this is not a date-only matched comparison.",
            "injection_rotation_note": "Direction and target are seed inputs; mask/noise fields can change across passes, but are paired across levels within a pass."}


def preparation_environment() -> dict[str, Any]:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "pillow": PIL.__version__,
            "jpeg_codec": features.version_codec("jpg"),
            "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
            "computation": "CPU image preparation and checkpoint metadata verification only; no model forward or device substitution"}


def verify_environment(reference: dict[str, Any], actual: dict[str, Any]) -> None:
    mismatches = {key: {"reference": reference.get(key), "current": actual.get(key)}
                  for key in ENVIRONMENT_FIELDS
                  if key not in reference or key not in actual or reference[key] != actual[key]}
    if mismatches:
        raise ValueError(f"Original preparation environment changed: {mismatches}")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_frozen_sources(frozen: dict[str, str], manifest_sources: dict[str, str],
                          expected_count: int) -> dict[str, str]:
    """Keep original operators fixed while allowing separately added evaluators.

    Return the full current Python source index so additions or changes DURING
    this QA run are still detected. New files are not retroactively called part
    of the original preparation snapshot.
    """
    _integer(expected_count, "expected frozen source count")
    current = source_hashes()
    if (not isinstance(frozen, dict) or len(frozen) != expected_count
            or frozen != manifest_sources):
        raise ValueError("Frozen preparation source SHA256 identities changed")
    for name, checksum in frozen.items():
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
                or name not in current or current[name] != checksum):
            raise ValueError(f"Frozen preparation source SHA256 identity changed: {name}")
    return current


def verify_references(config: SupplementConfig, protocol: CorruptionProtocol) -> dict[str, Any]:
    """Read reference metadata only, never a val/test image or semantic label."""
    directory = config.path("reference_preparation")
    names = ("status.json", "val_manifest.json", "train_calibration.json",
             "source_sha256.json", "environment.json")
    paths = {name: directory / name for name in names}
    records = {name: _read_json(path) for name, path in paths.items()}
    status, manifest = records["status.json"], records["val_manifest.json"]
    expected_protocol = config.values["expected_protocol_sha256"]
    expected_checkpoint = config.values["expected_checkpoint_sha256"]
    if (protocol.fingerprint != expected_protocol
            or status.get("protocol_sha256") != expected_protocol
            or manifest.get("protocol_sha256") != expected_protocol):
        raise ValueError("Original candidate protocol fingerprint changed")
    if status.get("status") != "completed_preparation":
        raise ValueError("Original candidate preparation was not completed")
    if (protocol.values["reference_checkpoint_sha256"] != expected_checkpoint
            or manifest.get("reference_checkpoint_sha256") != expected_checkpoint):
        raise ValueError("Original reference checkpoint binding changed")
    if (manifest.get("protocol_file_sha256") != file_sha256(protocol.source)
            or manifest.get("base_config_sha256") != file_sha256(protocol.base.source)):
        raise ValueError("Original protocol/base YAML content changed")
    frozen = records["source_sha256.json"]
    execution_sources = verify_frozen_sources(
        frozen, manifest.get("source_sha256"), config.values["verification"]["frozen_source_count"])
    for name, checksum in frozen.items():
        if file_sha256(directory / "source" / name) != checksum:
            raise ValueError(f"Original frozen source snapshot changed: {name}")
    instance_name = manifest.get("instance_file")
    if (not isinstance(instance_name, str) or not instance_name
            or Path(instance_name).name != instance_name
            or "/" in instance_name or "\\" in instance_name):
        raise ValueError("Original validation instance reference must be an adjacent filename")
    instance_path = directory / instance_name
    if file_sha256(instance_path) != manifest.get("instance_file_sha256"):
        raise ValueError("Original validation instance JSONL content changed")
    # This is a hash of the existing metadata file, not access to validation pixels.
    paths[instance_name] = instance_path
    actual_environment = preparation_environment()
    verify_environment(records["environment.json"], actual_environment)
    selection = config.values["selection"]
    original_selection = protocol.values["calibration"]
    if (selection["seed"] != original_selection["selection_seed"]
            or selection["skip_source_pairs"] != original_selection["source_pairs"]):
        raise ValueError("Supplement must continue the original selection seed and skip its original source count")
    if config.values["rendering"]["preview_levels"] != original_selection["preview_levels"]:
        raise ValueError("Original display-level settings changed")
    return {"paths": paths, "records": records, "source_sha256": frozen,
            "execution_source_sha256": execution_sources,
            "environment": actual_environment,
            "reference_artifact_sha256": {name: file_sha256(path) for name, path in paths.items()}}


def prepare_supplement(config: SupplementConfig) -> dict[str, Any]:
    output = config.path("output_dir")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite supplementary QA: {output}")
    protocol = load_protocol(config.path("protocol_config"))
    for protected in (protocol.base.data_root, protocol.base.output_dir, config.path("reference_preparation")):
        if output == protected or protected in output.parents:
            raise ValueError(f"Supplement output cannot be inside a protected data/reference directory: {protected}")
    references = verify_references(config, protocol)
    checkpoint_path = protocol.path("reference_checkpoint")
    checksum = file_sha256(checkpoint_path)
    if checksum != config.values["expected_checkpoint_sha256"]:
        raise ValueError("Reference checkpoint content changed")
    # The frozen helper uses weights_only=True and map_location='cpu'. No model is constructed.
    checkpoint = read_checkpoint(checkpoint_path)
    check_checkpoint(protocol.base, checkpoint)
    dataset = LEVIRDataset(protocol.base, "train")
    _check_inventory(dataset, checkpoint)
    original_indices = select_calibration_tiles(dataset, protocol)
    original_samples = references["records"]["train_calibration.json"]["samples"]
    ids = sample_ids(dataset)
    if ([sample["sample_id"] for sample in original_samples] != [ids[index] for index in original_indices]
            or [sample["index"] for sample in original_samples] != original_indices):
        raise ValueError("Original calibration selection does not match the frozen label-blind rule")
    indices = select_supplement_tiles(dataset, **config.values["selection"])
    if {dataset.tiles[index][0] for index in original_indices} & {dataset.tiles[index][0] for index in indices}:
        raise AssertionError("Supplement reuses an original QA source pair")
    selected_ids = [ids[index] for index in indices]
    batches = preview_batches(indices, group_size=config.values["rendering"]["group_size"],
                              row_rotations=config.values["rendering"]["row_rotations"])
    script_path = Path(__file__).resolve()
    watched_paths = {"supplement_config": config.source, "protocol_config": protocol.source,
                     "base_config": protocol.base.source, "script": script_path,
                     "checkpoint": checkpoint_path,
                     **{f"reference/{name}": path for name, path in references["paths"].items()}}
    before_hashes = {name: file_sha256(path) for name, path in watched_paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "preparing", "kind": "train_only_supplementary_visual_QA"})
    try:
        # Persist the prespecified identities before decoded-label counts or image generation.
        selection = {"selection_uses_labels": False, "selection_uses_model_scores": False,
                     "rule": "Original keyed_seed(seed, filename) order; skip original sources, then choose keyed_seed(seed, filename, 'tile') modulo tile count.",
                     "settings": config.values["selection"], "original_samples_skipped": original_samples,
                     "samples": [{"sample_id": ids[index], "index": index,
                                  "source_filename": dataset.records[dataset.tiles[index][0]].name}
                                 for index in indices],
                     "batches": [{**batch, "sample_ids": [ids[index] for index in batch["indices"]]}
                                 for batch in batches],
                     "saved_before_decoded_label_analysis_and_rendering": True}
        write_json(output / "selection.json", selection)
        copies = {"supplement_source.yaml": config.source, "reference_protocol_source.yaml": protocol.source,
                  "reference_base_config.yaml": protocol.base.source,
                  "source/scripts/supplement_train_qa.py": script_path}
        for relative, source in copies.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_sha256(target) != file_sha256(source):
                raise ValueError(f"Supplement snapshot copy changed: {relative}")
        resolved = dict(config.values)
        for field in _PATH_FIELDS:
            resolved[field] = str(config.path(field))
        with (output / "supplement_resolved.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(resolved, stream, sort_keys=False)
        write_json(output / "source_sha256.json", {
            "original_frozen_sources": references["source_sha256"],
            "execution_pcdc_sources": references["execution_source_sha256"],
            "supplement_sources": {"scripts/supplement_train_qa.py": before_hashes["script"]},
            "archived_file_sha256": {relative: file_sha256(output / relative) for relative in copies}})
        selected_records = [dataset.records[dataset.tiles[index][0]] for index in indices]
        content_before = _check_content(selected_records, "train", checkpoint)
        pairs = {ids[index]: _raw_tile(dataset, index) for index in indices}
        tile_counts = [{"sample_id": sample_id, "positive_pixels": int(pair.mask.sum()),
                        "total_pixels": int(pair.mask.size),
                        "positive_fraction": float(pair.mask.mean()),
                        "counted_posthoc_not_used_for_selection": True}
                       for sample_id, pair in pairs.items()]
        write_json(output / "posthoc_tile_counts.json", tile_counts)
        all_instances, sheets, batch_reports = [], [], []
        for batch in batches:
            relative = Path("batches") / batch["batch_id"]
            batch_output = output / relative
            batch_output.mkdir(parents=True, exist_ok=False)
            generated = calibration_previews(protocol, dataset, batch["indices"], batch_output)
            report = annotate_report(generated, pairs, batch["batch_id"])
            write_json(batch_output / "train_calibration.json", report)
            for instance in report["instances"]:
                # Preserve the helper's original relative path in its per-batch report.
                image_path = instance["full_size_level5_image"]
                all_instances.append({**instance, "full_size_level5_image": (
                    (relative / image_path).as_posix() if image_path else None)})
            batch_sheets = [(relative / sheet).as_posix() for sheet in report["preview_sheets"]]
            sheets.extend(batch_sheets)
            batch_reports.append({**batch, "report": (relative / "train_calibration.json").as_posix(),
                                  "preview_sheets": batch_sheets,
                                  "instance_count": len(report["instances"])})
        coverage = verify_coverage(all_instances, selected_ids)
        content_after = _check_content(selected_records, "train", checkpoint)
        if content_before != content_after:
            raise ValueError("Selected training source content changed during supplementary QA")
        after_hashes = {name: file_sha256(path) for name, path in watched_paths.items()}
        if before_hashes != after_hashes or source_hashes() != references["execution_source_sha256"]:
            raise ValueError("Frozen source, script, configuration or reference artifact changed during QA")
        verify_environment(references["records"]["environment.json"], preparation_environment())
        provenance = {"protocol_sha256": protocol.fingerprint, "reference_checkpoint_sha256": checksum,
                      "checkpoint_loaded_on": "cpu_metadata_only_no_model_constructed",
                      "checkpoint_metadata": {key: checkpoint.get(key) for key in
                                              ("kind", "epoch", "best_epoch", "protocol_signature", "audit_sha256")},
                      "original_frozen_source_sha256": references["source_sha256"],
                      "execution_pcdc_source_sha256": references["execution_source_sha256"],
                      "watched_file_sha256_before": before_hashes, "watched_file_sha256_after": after_hashes,
                      "reference_artifact_sha256": references["reference_artifact_sha256"],
                      "selected_train_file_sha256_before": content_before,
                      "selected_train_file_sha256_after": content_after,
                      "train_header_inventory_checked": True,
                      "unselected_train_content_hashes_read": False,
                      "val_image_or_label_files_read": 0, "test_image_or_label_files_read": 0,
                      "model_inference_run": False, "new_training_started": False}
        report = {"schema_version": 1, "status": "awaiting_visual_review",
                  "scope": "train_only_candidate_visual_QA_not_research_scores",
                  "selection": selection, "posthoc_tile_counts": tile_counts,
                  "coverage": coverage, "batches": batch_reports, "preview_sheets": sheets,
                  "instances": all_instances, "all_invariants_passed": True,
                  "provenance": provenance,
                  "limitation": "Label-independent supplementary samples and numeric invariants do not establish universal observability or constitute visual acceptance. No model scores were computed."}
        write_json(output / "train_qa_supplement.json", report)
        write_json(output / "environment.json", references["environment"])
        summary = {"status": "awaiting_visual_review", "kind": "train_only_supplementary_visual_QA",
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol_status": "candidate_pending_review", "protocol_sha256": protocol.fingerprint,
                   "reference_checkpoint_sha256": checksum, "output_dir": str(output),
                   "source_pairs": len(indices), "batches": len(batches),
                   "instances": len(all_instances), "preview_sheets": len(sheets),
                   "positive_tiles_posthoc": sum(item["positive_pixels"] > 0 for item in tile_counts),
                   "all_invariants_passed": True, "both_dates_per_sample_family_spatial_mode": True,
                   "numeric_levels": [1, 2, 3, 4, 5], "display_levels": config.values["rendering"]["preview_levels"],
                   "val_image_or_label_files_read": 0, "test_image_or_label_files_read": 0,
                   "model_inference_run": False, "new_training_started": False,
                   "acceptance_record_written": False,
                   "next_gate": "Actually inspect the unchanged candidate's original and supplementary train-only images; record visual findings separately before any corruption scoring."}
        write_json(output / "status.json", summary, overwrite=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "kind": "train_only_supplementary_visual_QA",
                                           "error": f"{type(error).__name__}: {error}",
                                           "acceptance_record_written": False}, overwrite=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_supplement(load_supplement_config(args.config)),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
