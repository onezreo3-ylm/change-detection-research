"""Read an immutable human submission and join only previously saved evidence.

The configuration is JSON syntax, a YAML subset. This script never imports a
spreadsheet writer or research model, evaluates Excel formulas, or reads raw data.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile


NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_sheet(data, sheet_name):
    """Extract stored values and formulas without recalculating or saving XLSX."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        matches = [s for s in book.find("s:sheets", NS) if s.get("name") == sheet_name]
        if len(matches) != 1:
            raise ValueError("Expected exactly one matching worksheet")
        rid = matches[0].get("{" + REL_NS + "}id")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in relations if r.get("Id") == rid)
        part = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in item.iter("{" + NS["s"] + "}t"))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        root = ET.fromstring(archive.read(part))
        values, formulas = {}, {}
        for cell in root.findall(".//s:sheetData/s:row/s:c", NS):
            address = cell.get("r")
            value = cell.find("s:v", NS)
            if cell.get("t") == "inlineStr":
                text = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
            elif value is None:
                text = None
            elif cell.get("t") == "s":
                text = strings[int(value.text)]
            else:
                text = value.text
            values[address] = text
            formula = cell.find("s:f", NS)
            if formula is not None:
                formulas[address] = formula.text
        return values, formulas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / cfg["workspace_root"]).resolve()
    output = root / cfg["output_dir"]
    source = root / cfg["source_workbook"]
    original_stat = source.stat()
    workbook_bytes = source.read_bytes()
    if digest(workbook_bytes) != cfg["expected_source_sha256"]:
        raise ValueError("Submitted workbook changed; use a new dated ingest configuration")

    inputs = {
        str(config_path): digest(config_path.read_bytes()),
        str(Path(__file__).resolve()): digest(Path(__file__).read_bytes()),
    }

    def load_json(relative):
        path = root / relative
        data = path.read_bytes()
        inputs[str(path)] = digest(data)
        return json.loads(data)

    # Verify sealed development evidence before using it. No raw dataset paths
    # from the source manifest are opened.
    sealed = {}
    for relative in cfg["prior_closeouts"]:
        sealed.update(load_json(relative)["artifacts_sha256"])
    sealed.update(load_json(cfg["packet_verification"])["outputs_sha256"])
    for path, expected in sealed.items():
        actual = digest(Path(path).read_bytes())
        if actual != expected:
            raise ValueError("Sealed evidence changed: " + path)
        inputs[path] = actual

    packet_cfg = load_json(cfg["packet_config"])
    samples = load_json(cfg["source_manifest"])
    previews = load_json(cfg["preview_manifest"])["sources"]
    ai_cases = load_json(cfg["preliminary_observations"])["cases"]
    sample_map = {s["id"]: s for s in samples}
    preview_map = {s["source"]: s for s in previews}
    ai_map = {s["source_id"]: s for s in ai_cases}
    verdict_map = {v["label"]: v["code"] for v in packet_cfg["verdicts"]}
    pending = set(cfg["user_clarification"]["pending_source_ids"])
    values, formulas = read_sheet(workbook_bytes, cfg["sheet_name"])
    rows, panel_hashes = [], {}
    for number in range(cfg["data_first_row"], cfg["data_first_row"] + cfg["expected_rows"]):
        row = {field: values.get(column + str(number)) for field, column in cfg["columns"].items()}
        row["sheet_row"] = number
        if any(row[field] is None or not row[field].strip() for field in cfg["columns"]):
            raise ValueError("Missing required human field in row " + str(number))
        if any(column + str(number) in formulas for column in cfg["columns"].values()):
            raise ValueError("Human input/source ID must be a stored value, not a formula")
        if row["verdict"] not in verdict_map or row["prior_context"] not in packet_cfg["prior_context_options"]:
            raise ValueError("Invalid category/context in row " + str(number))
        sid = row["source_id"]
        if sid not in sample_map:
            raise ValueError("Unrecognized source: " + sid)
        sample, preview = sample_map[sid], preview_map[sid]
        for model, counts in sample["saved_clean_counts"].items():
            if set(counts) != {"tp", "tn", "fp", "fn"} or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("Invalid saved confusion counts: " + sid + " " + model)
            if sum(counts.values()) != cfg["expected_tile_pixels"] or counts["tp"] + counts["fn"] != sample["foreground_pixels"]:
                raise ValueError("Saved counts do not reconcile to original support: " + sid)
        # Only hash existing rendered train panels, never decode/display images.
        panel = source.parent / "images" / (Path(sample["name"]).stem + ".png")
        panel_hash = digest(panel.read_bytes())
        if panel_hash != preview["preview_sha256"]:
            raise ValueError("Submitted packet image changed: " + str(panel))
        panel_hashes[str(panel)] = panel_hash
        row.update({
            "raw_verdict_code": verdict_map[row["verdict"]],
            "effective_verdict_code": None if sid in pending else verdict_map[row["verdict"]],
            "review_status": "pending_user_requested_refill" if sid in pending else "human_nonblind_submission_recorded",
            "user_clarification_applies": sid in pending,
            "foreground_pixels": sample["foreground_pixels"],
            "components_from_saved_mask_audit": preview["components"],
            "saved_clean_counts": sample["saved_clean_counts"],
            "same_assistant_preliminary_verdict": ai_map[sid]["preliminary_verdict"],
            "raw_verdict_empty_label_semantic_conflict": (
                sample["foreground_pixels"] > 0
                and verdict_map[row["verdict"]] == "unchanged_consistent_with_empty_label"
            ),
            "strict_blind_independent_confirmation": False,
        })
        rows.append(row)

    ids = [r["source_id"] for r in rows]
    if len(set(ids)) != cfg["expected_rows"] or set(ids) != set(sample_map):
        raise ValueError("Duplicate, missing, or extra review source")
    if not pending <= set(ids):
        raise ValueError("User clarification refers to an unknown source")

    consistent = [r for r in rows if r["effective_verdict_code"] == "observable_change_consistent_with_label"]
    models = list(samples[0]["saved_clean_counts"])
    common_miss = [r for r in consistent if all(r["saved_clean_counts"][m]["tp"] == 0 for m in models)]
    internal_miss = [r for r in common_miss if not any(
        c["touches_tile_edge"] for c in r["components_from_saved_mask_audit"]["8"]
    )]
    summary = {
        "date": cfg["date"],
        "status": "human_nonblind_review_ingested_two_rows_pending_no_method_admitted",
        "required_fields_complete_rows": len(rows),
        "source_ids_verified": len(set(ids)),
        "positive_label_sources": sum(r["foreground_pixels"] > 0 for r in rows),
        "empty_label_sources": sum(r["foreground_pixels"] == 0 for r in rows),
        "raw_workbook_category_counts": dict(Counter(r["verdict"] for r in rows)),
        "effective_recorded_verdict_counts": dict(Counter(r["effective_verdict_code"] for r in rows if r["effective_verdict_code"])),
        "effective_recorded_rows": sum(r["effective_verdict_code"] is not None for r in rows),
        "pending_refill_source_ids": sorted(pending),
        "reviewer_field_tokens": dict(Counter(r["reviewer"] for r in rows)),
        "independent_human_identity_verified": False,
        "prior_context_counts": dict(Counter(r["prior_context"] for r in rows)),
        "strict_blind_confirmation_established": False,
        "confirmed_annotation_errors": None,
        "population_noise_rate": None,
        "common_zero_tp_human_consistent_sources": [r["source_id"] for r in common_miss],
        "internal_common_zero_tp_human_consistent_sources": [r["source_id"] for r in internal_miss],
        "zero_tp_by_model_in_human_consistent_subset": {
            m: [r["source_id"] for r in consistent if r["saved_clean_counts"][m]["tp"] == 0]
            for m in models
        },
        "subset_performance_or_improvement_claim": False,
        "new_method_training_admitted": False,
        "recommended_next_step": "resolve_two_pending_rows_and_obtain_blind_case_adjudication_before_any_method_training",
        "execution_boundaries": cfg["execution_boundaries"],
    }

    def write_json(name, data):
        destination = output / name
        if destination.exists():
            raise FileExistsError("Do not overwrite prior ingest: " + str(destination))
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # A byte-for-byte snapshot is provenance, not a workbook edit/re-export.
    snapshot = output / "submitted_workbook_snapshot.xlsx"
    if snapshot.exists():
        raise FileExistsError(str(snapshot))
    snapshot.write_bytes(workbook_bytes)
    write_json("submission_rows.json", {"sheet": cfg["sheet_name"], "rows": rows})
    write_json("user_clarification.json", cfg["user_clarification"])
    write_json("review_summary.json", summary)
    with (output / "review_rows.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["sheet_row", "source_id", "verdict", "reason", "reviewer", "prior_context", "effective_verdict_code", "review_status", "foreground_pixels"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    after = source.stat()
    if source.read_bytes() != workbook_bytes or after.st_mtime_ns != original_stat.st_mtime_ns:
        raise ValueError("Source workbook changed during analysis")
    write_json("input_manifest.json", {
        "source_workbook": str(source),
        "source_sha256": digest(workbook_bytes),
        "source_bytes": len(workbook_bytes),
        "source_mtime_ns": original_stat.st_mtime_ns,
        "snapshot_sha256": digest(snapshot.read_bytes()),
        "source_unchanged": True,
        "sealed_input_files_verified": len(sealed),
        "inputs_sha256": inputs,
        "submitted_panel_hashes": panel_hashes,
        "submitted_panel_hashes_verified": len(panel_hashes),
        "formulas_not_recalculated": True,
        "B3_cached_not_used_for_completeness": values.get("B3"),
        "execution_boundaries": cfg["execution_boundaries"],
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
