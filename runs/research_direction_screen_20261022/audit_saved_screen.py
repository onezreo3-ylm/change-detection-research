"""Independent check of saved counts and unchanged train review panel pixels."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
report = json.loads((ROOT / "saved_count_screen.json").read_text(encoding="utf-8"))
preparation = json.loads((ROOT / "train_review_preparation.json").read_text(encoding="utf-8"))
seal = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, digest in seal["inputs_sha256"].items():
    assert sha(path) == digest
for name, digest in seal["outputs_sha256"].items():
    assert sha(ROOT / name) == digest

with (PROJECT / "runs/valais_matched_error_review_20261004/clean_by_source.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
val_common = None
for model, actual in report["validation"].items():
    selected = [r for r in rows if r["model"] == model]
    totals = {k: sum(int(r[k]) for r in selected) for k in ("tn", "fp", "fn", "tp")}
    missed = {r["source"] for r in selected if int(r["foreground_pixels"]) > 0 and int(r["tp"]) == 0}
    assert totals == actual["counts"]
    assert len(missed) == actual["positive_sources_zero_tp"]
    assert sum(int(r["foreground_pixels"]) > 0 and int(r["tp"]) + int(r["fp"]) == 0
               for r in selected) == actual["positive_sources_zero_predictions"]
    assert sum(int(r["fn"]) for r in selected if r["source"] in missed) == actual["fn_in_zero_tp_sources"]
    assert actual["f1"] == 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"])
    val_common = missed if val_common is None else val_common & missed
assert sorted(val_common) == report["validation_common_misses"]["all_models_zero_tp_sources"]

train = {}
for rel in ("runs/valais_coupling_p0_diagnostic_20261007/records.jsonl",
            "runs/valais_p1_posttraining_v1_20261019/residual/records.jsonl",
            "runs/valais_c5r5_fixed_eval_v1_20261022/residual/records.jsonl"):
    with (PROJECT / rel).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["model"], row["source_id"]
            value = row["diagnostic"]["clean"]["counts"]
            if key in train:
                assert train[key] == value
            train[key] = value
train_common = None
for model, actual in report["train_fixed_enriched_subset"].items():
    selected = {s: c for (m, s), c in train.items() if m == model}
    assert len(selected) == 64
    totals = {k: sum(c[k] for c in selected.values()) for k in ("tn", "fp", "fn", "tp")}
    assert totals == actual["counts"]
    missed = {s for s, c in selected.items() if c["fn"] > 0 and c["tp"] == 0}
    assert len(missed) == actual["positive_sources_zero_tp"]
    train_common = missed if train_common is None else train_common & missed
assert sorted(train_common) == report["train_common_misses"]["all_models_zero_tp_sources"]

for sample in preparation["sources"]:
    with Image.open(sample["preview_path"]) as im:
        preview = np.array(im)
    for index, role in enumerate(("image_a", "image_b", "mask")):
        original = sample["original_files"][role]
        assert sha(original["path"]) == original["sha256"]
        with Image.open(original["path"]) as im:
            array = np.array(im)
        if role == "mask":
            if array.ndim == 3:
                assert np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])
                array = array[..., 0]
            assert int(array.sum()) == sample["foreground_pixels"]
            array = np.repeat((array * 255)[..., None], 3, axis=2)
        assert np.array_equal(preview[30:286, index * 256:(index + 1) * 256], array)
    for connected in sample["components"].values():
        assert sum(c["area"] for c in connected) == sample["foreground_pixels"]

common = [s for s in preparation["sources"] if s["source"] in train_common]
all_positive = [s for s in preparation["sources"] if s["foreground_pixels"] > 0]
support = sum(s["foreground_pixels"] for s in all_positive)
common_support = sum(s["foreground_pixels"] for s in common)
border_only = [s for s in common if all(c["touches_tile_edge"] for c in s["components"]["8"])]
result = {"status": "passed_independent_saved_count_and_native_panel_audit",
          "validation_models": 3, "validation_sources_per_model": 1715,
          "train_models": 7, "train_sources_per_model_after_deduplication": 64,
          "verified_native_train_panels": len(preparation["sources"]) * 3,
          "train_probe_foreground_pixels": support,
          "common_zero_tp_sources": len(common), "common_zero_tp_foreground_pixels": common_support,
          "common_zero_tp_fraction_of_probe_foreground": common_support / support,
          "common_zero_tp_all_components_touch_border": len(border_only),
          "common_zero_tp_border_only_foreground_pixels": sum(s["foreground_pixels"] for s in border_only),
          "raw_validation_read": False, "test_read": False, "model_inference": False,
          "audit_source_sha256": sha(__file__)}
(ROOT / "independent_saved_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
