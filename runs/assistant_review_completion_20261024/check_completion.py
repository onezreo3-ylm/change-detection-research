"""Verify delegated workbook completion and preserve the distinction from truth."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import openpyxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / cfg["workspace_root"]).resolve()
    run = root / cfg["run_dir"]
    build = root / cfg["build_dir"]
    output = root / cfg["output_dir"]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig"))
    source = root / cfg["source_workbook"]
    target = output / cfg["workbook_name"]
    assert sha(source) == cfg["source_sha256"]
    prior = read(root / cfg["previous_verification"])
    for p, expected in prior["artifacts_sha256"].items():
        assert sha(p) == expected, p

    before = read(build / "native_source.json")
    after = read(build / "native_reopened.json")
    expected_edits = dict(cfg["header_edits"])
    for entry in cfg["fills"]:
        for col, value in zip("DEFG", [entry["verdict"], entry["reason"], cfg["assistant_reviewer_label"], cfg["assistant_prior_context"]]):
            expected_edits[col + str(entry["row"])] = value
    for address, cell in before["cells"].items():
        expected = expected_edits.get(address, cell["formula"])
        assert after["cells"][address]["formula"] == expected, address
    for field in ["sheetCount", "validations", "splitColumn", "splitRow", "freezePanes"]:
        assert before[field] == after[field], field
    assert after["cells"]["B3"]["value"] == 24
    assert all(after["cells"][f"H{row}"]["value"] == "已填写" for row in range(8,32))
    assert all(after["cells"][f"C{row}"]["value"] == "打开三联图" for row in range(8,32))

    # Second reader checks persisted XLSX input values and Excel-calculated caches.
    book = openpyxl.load_workbook(target, read_only=True, data_only=True)
    sheet = book[cfg["sheet_name"]]
    rows = [dict(zip(["number","source_id","link","verdict","reason","reviewer","context","status"], row))
            for row in sheet.iter_rows(min_row=8,max_row=31,min_col=1,max_col=8,values_only=True)]
    assert sheet["B3"].value == 24
    book.close()
    verdicts = Counter(row["verdict"] for row in rows)
    assert verdicts == {"变化与标注相符":10,"疑似标注或时点问题":4,"空标签基本合理":8,"触边难判断":2}
    assert Counter(row["reviewer"] for row in rows) == {"本人":22,cfg["assistant_reviewer_label"]:2}
    assert len({row["source_id"] for row in rows}) == 24

    samples = read(root / cfg["source_manifest"])
    sample_map = {s["id"]:s for s in samples}
    preparation = read(root / cfg["preview_manifest"])["sources"]
    preview_map = {s["source"]:s for s in preparation}
    for row in rows:
        panel = output / "images" / (Path(sample_map[row["source_id"]]["name"]).stem + ".png")
        assert sha(panel) == preview_map[row["source_id"]]["preview_sha256"]
    focused = read(run / "focused_assistant_review.json")
    assert len(focused["cases"]) == 9 and not focused["independent_human"]
    assert {Path(sample_map[c["source_id"]]["name"]).stem for c in focused["cases"]} == set(cfg["reviewed_source_stems"])

    models = list(samples[0]["saved_clean_counts"])
    consistent = [row for row in rows if row["verdict"] == "变化与标注相符"]
    common = [row["source_id"] for row in consistent if all(sample_map[row["source_id"]]["saved_clean_counts"][m]["tp"] == 0 for m in models)]
    assert set(common) == {"train/291.tif@x0_y0","train/3175.tif@x0_y0"}
    result = {
        "status":"passed_delegated_completion_and_saved_evidence_diagnostic",
        "workbook":str(target), "workbook_sha256":sha(target),
        "source_sha256":sha(source), "source_unchanged":True,
        "records":24, "human_original_rows_retained":22, "assistant_completed_rows":2,
        "mixed_workbook_verdict_counts":dict(verdicts),
        "independent_human_truth_established":False,
        "human_291_verdict":"observable_change_consistent_with_label",
        "assistant_291_verdict":"insufficient_visual_evidence",
        "common_zero_tp_human_consistent_source_ids":common,
        "new_method_admitted":False,
        "previous_sealed_files_verified":len(prior["artifacts_sha256"]),
        "native_cells_scope_verified":len(before["cells"]),
        "unchanged_24_relative_image_links_verified":True,
        "unchanged_validation_and_frozen_panes_verified":True,
        "verified_unchanged_image_copies":24,
        "actually_viewed_native_panels":9,
        "native_export_recovery":read(build / "native_open_status.json"),
        "raw_validation_or_test_read":False,
        "new_research_inference_or_training_executed_in_this_diagnostic":False,
        "execution_boundaries":cfg["execution_boundaries"],
        "next_action":"Continue delegated hourly research/training preparation under docs/MANAGED_CONTINUATION_2026-10-24.md; do not use assistant completion as independent truth."
    }
    (run / "completion_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run / "mixed_submission_rows.json").write_text(json.dumps({"rows":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
