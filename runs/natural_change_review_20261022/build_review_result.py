"""Join recorded preliminary observations to existing saved train counts only."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert cfg["kind"] == "preliminary_natural_change_train_review"
    assert not any(cfg[k] for k in ("reviewer_is_independent", "new_training", "new_model_inference", "raw_validation_or_test_access"))
    root = (args.config.resolve().parent / cfg["workspace_root"]).resolve()
    output = root / cfg["output_dir"]
    if (output / "review_summary.json").exists():
        raise FileExistsError("Preserve completed review results; do not overwrite")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda name: json.loads((root / cfg[name]).read_text(encoding="utf-8"))
    closeout = read("previous_closeout")
    for path, digest in closeout["artifacts_sha256"].items():
        assert sha(path) == digest, path
    observation_hash_before_join = sha(root / cfg["observations"])
    observed = read("observations")
    assert not observed["independent_reviewer"] and not observed["blind_to_prior_model_results"]
    cases = {s["source_id"]: s for s in observed["cases"]}
    samples = {s["id"]: s for s in read("previous_samples")}
    prepared = {s["source"]: s for s in read("previous_preparation")["sources"]}
    assert len(cases) == len(observed["cases"]) == cfg["expected_sources"]
    assert set(cases) == set(samples) == set(prepared)
    assert all(s.startswith("train/") for s in cases)
    assert all(c["preliminary_verdict"] in cfg["allowed_preliminary_verdicts"] for c in cases.values())
    assert all(c["confirmed_annotation_error"] is None and c["independent_verdict"] == "pending" for c in cases.values())
    models = set(next(iter(samples.values()))["saved_clean_counts"])
    groups = Counter(c["preliminary_verdict"] for c in cases.values())
    positive_groups = Counter()
    positive_pixels = Counter()
    common = []
    by_model = {model: defaultdict(lambda: {"positive_sources": 0, "zero_tp_sources": 0, "fn": 0, "tp": 0}) for model in sorted(models)}
    rows = []
    for source, case in cases.items():
        sample, prep = samples[source], prepared[source]
        assert sha(prep["preview_path"]) == prep["preview_sha256"]
        assert set(sample["saved_clean_counts"]) == models
        support = sample["foreground_pixels"]
        verdict = case["preliminary_verdict"]
        if support:
            positive_groups[verdict] += 1
            positive_pixels[verdict] += support
        for model, counts in sample["saved_clean_counts"].items():
            assert counts["tp"] + counts["fn"] == support
            if support:
                group = by_model[model][verdict]
                group["positive_sources"] += 1
                group["zero_tp_sources"] += counts["tp"] == 0
                group["fn"] += counts["fn"]
                group["tp"] += counts["tp"]
        if support and all(c["tp"] == 0 for c in sample["saved_clean_counts"].values()):
            common.append(source)
        ccs = prep["components"]["8"]
        rows.append({"source": source, "foreground_pixels": support,
                     "preliminary_verdict": verdict,
                     "components_touching_tile_edge": sum(c["touches_tile_edge"] for c in ccs),
                     "components_total": len(ccs), "independent_verdict": "pending",
                     "confirmed_annotation_error": "not_established", "evidence": case["evidence"]})
    positives = sum(positive_groups.values())
    assert positives == cfg["expected_positive_sources"]
    assert len(cases) - positives == cfg["expected_empty_sources"]
    assert sha(root / cfg["observations"]) == observation_hash_before_join
    for path, digest in closeout["artifacts_sha256"].items():
        assert sha(path) == digest
    summary = {
        "status": "completed_same_assistant_preliminary_visual_review_independent_truth_pending",
        "sources": len(cases), "positive_sources": positives, "empty_sources": len(cases) - positives,
        "preliminary_groups": dict(groups), "positive_groups": dict(positive_groups),
        "foreground_pixels_by_positive_group": dict(positive_pixels),
        "saved_train_counts_by_model_and_positive_group": by_model,
        "all_model_zero_tp_sources": sorted(common),
        "all_model_zero_tp_preliminary_groups": dict(Counter(cases[s]["preliminary_verdict"] for s in common)),
        "independent_annotation_verdicts_pending": len(cases), "confirmed_label_errors": None,
        "label_noise_rate": None, "new_research_training": False, "new_research_inference": False,
        "raw_validation_or_test_read": False, "previous_artifact_hashes_unchanged": True,
        "observations_sha256_before_saved_model_count_join": observation_hash_before_join,
        "estimated_cumulative_image_data_url_bytes": observed["estimated_cumulative_image_data_url_bytes"],
        "elapsed_seconds": time.monotonic() - started,
        "decision": "Do not admit a new method: confirmed complete internal true-change omissions are not established in this fixed probe.",
        "limitations": ["Same assistant had prior exposure to selected model results; this is not independent blind adjudication.",
                        "Categories are preliminary image interpretations, not annotation ground truth.",
                        "A positive tile with TP>0 may still contain completely missed components; saved tile totals cannot resolve this.",
                        "One enriched training probe and one architecture/seed cannot establish population frequency or geographic replication.",
                        "No threshold, split, label or original full-validation score has been changed."]}
    (output / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "preliminary_review.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Copy an empty independent-review template; never substitute assistant categories for a reviewer.
    blind = (root / cfg["previous_blind_template"]).read_bytes()
    with (output / "independent_review_blank.csv").open("xb") as f:
        f.write(blind)
    inputs = [args.config.resolve(), root / cfg["observations"], root / cfg["previous_samples"],
              root / cfg["previous_preparation"], root / cfg["previous_closeout"],
              root / cfg["previous_blind_template"], Path(__file__).resolve()]
    (output / "input_manifest.json").write_text(json.dumps({"inputs_sha256": {str(p): sha(p) for p in inputs},
        "outputs_sha256": {name: sha(output / name) for name in ("review_summary.json", "preliminary_review.csv", "independent_review_blank.csv")}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("saved_train_counts_by_model_and_positive_group", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
