"""Independent TRAIN-only visual QA to balance label class and direction parity."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from pcdc.config import UniqueKeyLoader
from pcdc.corruption_protocol import CorruptionProtocol, keyed_seed
from pcdc.data import LEVIRDataset, file_sha256
from pcdc.prepare_corruptions import _check_content, calibration_previews
from pcdc.runtime import write_json
from scripts.prepare_valais_train_qa import ROOT, load_candidate


KEYS = {"schema_version", "kind", "status", "base_candidate",
        "base_candidate_sha256", "base_report", "base_report_sha256",
        "output_dir", "selection", "seeds"}


def balanced_names(dataset: LEVIRDataset, positive: set[str], excluded: set[str],
                   *, seed: int) -> list[str]:
    if dataset.split != "train":
        raise ValueError("Supplement is train-only")
    names = {record.name for record in dataset.records} - excluded
    if not positive <= {record.name for record in dataset.records}:
        raise ValueError("Positive manifest contains unknown train names")
    positives = sorted(names & positive,
                       key=lambda name: (keyed_seed(seed, "positive", name), name))[:4]
    empties = sorted(names - positive,
                     key=lambda name: (keyed_seed(seed, "empty", name), name))[:4]
    if len(positives) != 4 or len(empties) != 4:
        raise ValueError("Insufficient distinct positive/empty train sources")
    # Two positive and two empty samples occupy each direction parity.
    return [positives[0], empties[0], positives[1], empties[1],
            empties[2], positives[2], empties[3], positives[3]]


def preflight(path: Path):
    values = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    if not isinstance(values, dict) or set(values) != KEYS:
        raise ValueError("Unexpected Valais supplement keys")
    if (values["schema_version"] != 1
            or values["kind"] != "train_only_direction_balance_supplement"
            or values["status"] != "candidate_pending_visual_review"
            or values["selection"] != {
                "seed": 810208, "positive_tiles": 4, "empty_tiles": 4,
                "exclude_base_sources": True, "row_order": "balanced_direction_parity"}
            or values["seeds"] != {"train": 510945, "val": 510946, "test": 510947}):
        raise ValueError("Valais supplement parameters differ")
    paths = {}
    for key in ("base_candidate", "base_report", "output_dir"):
        item = Path(values[key])
        paths[key] = (item if item.is_absolute() else path.parent / item).resolve()
    if paths["output_dir"] != ROOT / "runs" / "valais_train_severity_qa_supplement_v1_20261002":
        raise ValueError("Supplement output path differs")
    if paths["output_dir"].exists():
        raise FileExistsError(paths["output_dir"])
    for key in ("base_candidate", "base_report"):
        if file_sha256(paths[key]) != values[key + "_sha256"]:
            raise RuntimeError(f"Bound Valais supplement reference changed: {key}")
    # The base candidate's output now exists, so validate its contents without
    # invoking its one-shot output-nonexistence preflight.
    from scripts.prepare_valais_train_qa import load_candidate as load_base
    base_values, base_paths, base, checkpoint, base_protocol = load_base(
        paths["base_candidate"], require_new=False)
    original = json.loads(paths["base_report"].read_text(encoding="utf-8"))
    if (original.get("status") != "candidate_pending_visual_review"
            or original.get("val_test_image_or_label_accessed") is not False
            or len(original.get("selected_sample_ids", [])) != 8):
        raise RuntimeError("Base QA report identity differs")
    settings = copy.deepcopy(base_protocol.values)
    settings["name"] = "valais_train_severity_qa_supplement_v1"
    settings["output_dir"] = values["output_dir"]
    settings["seeds"] = values["seeds"]
    protocol = CorruptionProtocol(settings, path, base)
    return values, paths, base, checkpoint, protocol, original, base_paths


def prepare(path: Path):
    values, paths, base, checkpoint, protocol, original, base_paths = preflight(path)
    dataset = LEVIRDataset(base, "train")
    balance = json.loads(base_paths["train_label_balance_report"].read_text(encoding="utf-8"))
    positive = {item["name"] for item in balance["train"]["positive_manifest"]}
    excluded = {sample.split("/", 1)[1].split("@", 1)[0]
                for sample in original["selected_sample_ids"]}
    ordered = balanced_names(dataset, positive, excluded, seed=values["selection"]["seed"])
    index = {record.name: position for position, (record_id, _, _) in enumerate(dataset.tiles)
             for record in [dataset.records[record_id]]}
    selected = [index[name] for name in ordered]
    checked = _check_content([dataset.records[dataset.tiles[item][0]] for item in selected],
                             "train", checkpoint)
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    previews = calibration_previews(protocol, dataset, selected, output)
    if (len(previews["instances"]) != 720 or len(previews["preview_sheets"]) != 18
            or not previews["all_invariants_passed"]):
        raise RuntimeError("Supplement QA coverage/invariants differ")
    classes = ["positive" if name in positive else "empty" for name in ordered]
    if classes != ["positive", "empty", "positive", "empty",
                   "empty", "positive", "empty", "positive"]:
        raise RuntimeError("Direction parity is not balanced by label class")
    report = {"schema_version": 1, "status": "candidate_pending_visual_review",
              "scope": "Sion train-only direction parity supplement; no research scores",
              "model_inference_performed": False,
              "val_test_image_or_label_accessed": False,
              "supplement_config_sha256": file_sha256(path),
              "base_candidate_sha256": values["base_candidate_sha256"],
              "base_report_sha256": values["base_report_sha256"],
              "selected_checkpoint_sha256": file_sha256(base_paths["reference_checkpoint"]),
              "selected_train_file_sha256": checked,
              "selected_source_names": ordered,
              "selected_label_class": classes,
              "selection": values["selection"],
              "protocol_fingerprint": protocol.fingerprint,
              **previews}
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
        values, paths, base, checkpoint, protocol, original, base_paths = preflight(path)
        print(json.dumps({"status": "preflight_passed", "test_accessed": False}))
    else:
        prepare(path)


if __name__ == "__main__":
    main()
