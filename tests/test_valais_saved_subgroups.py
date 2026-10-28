"""CPU-only count audits; fixture counts are engineering evidence only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_valais_saved_subgroups import analyze, condition_subgroups, metrics, pooled_counts


def source(tn, fp, fn, tp):
    counts = dict(tn=tn, fp=fp, fn=fn, tp=tp)
    return {"counts": counts, "pixels": tn + fp + fn + tp,
            "foreground_pixels": fn + tp, "background_pixels": tn + fp}


def fixture():
    clean_sources = {"val/positive.tif": source(7, 0, 1, 2),
                     "val/empty.tif": source(10, 0, 0, 0)}
    perturbed_sources = {"val/positive.tif": source(6, 1, 1, 2),
                         "val/empty.tif": source(9, 1, 0, 0)}
    def record(sources):
        counts = pooled_counts(sources.values())
        return {"by_source": sources, "samples": 2, "pixels": 20,
                "counts": counts, "metrics": metrics(counts)}
    clean, condition = record(clean_sources), record(perturbed_sources)
    condition.update(condition_id="brightness_l1")
    assignments = {"val/positive.tif": dict(spatial_mode="local", direction="brighten", target_date="A"),
                   "val/empty.tif": dict(spatial_mode="global", direction="darken", target_date="B")}
    return clean, condition, assignments


class SavedSubgroupTests(unittest.TestCase):
    def test_one_shot_iterator_retains_all_confusion_fields(self):
        rows = [source(7, 2, 3, 4), source(11, 5, 6, 8)]
        self.assertEqual(pooled_counts(iter(rows)), dict(tn=18, fp=7, fn=9, tp=12))

    def test_invalid_counts_are_rejected(self):
        for bad in (-1, True, 1.5):
            row = source(1, 1, 1, 1)
            row["counts"]["fp"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pooled_counts(iter([row]))

    def test_same_sources_clean_and_partition_totals_are_recovered(self):
        clean, condition, assignments = fixture()
        rows = condition_subgroups(condition, clean, assignments)
        for axis in ("spatial_mode", "direction", "target_date"):
            selected = [row for row in rows if row["axis"] == axis]
            self.assertEqual(pooled_counts(iter(selected)), condition["counts"])
            self.assertEqual(pooled_counts({"counts": row["same_sources_clean_counts"]}
                                           for row in selected), clean["counts"])
            self.assertEqual(sum(row["net_fp"] for row in selected), 2)
        local = next(row for row in rows if row["axis"] == "spatial_mode" and row["value"] == "local")
        self.assertAlmostEqual(local["perturbed"]["f1"], 2 / 3)
        self.assertAlmostEqual(local["same_sources_clean"]["f1"], 4 / 5)

    def test_missing_source_assignment_is_rejected(self):
        clean, condition, assignments = fixture()
        assignments.pop("val/empty.tif")
        with self.assertRaisesRegex(ValueError, "coverage"):
            condition_subgroups(condition, clean, assignments)

    def test_source_totals_and_label_changes_are_rejected(self):
        clean, condition, assignments = fixture()
        broken = copy.deepcopy(condition)
        broken["counts"]["fp"] += 1
        with self.assertRaisesRegex(ValueError, "counts"):
            condition_subgroups(broken, clean, assignments)
        changed = copy.deepcopy(condition)
        changed["by_source"]["val/positive.tif"] = source(6, 1, 2, 1)
        changed["by_source"]["val/empty.tif"] = source(8, 1, 1, 0)
        changed["counts"] = pooled_counts(changed["by_source"].values())
        with self.assertRaisesRegex(ValueError, "Foreground labels"):
            condition_subgroups(changed, clean, assignments)

    def test_guard_mismatch_rejected_before_instances_or_analysis_access(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            summary = dict(status="completed", split="val", test_evaluated=False,
                           clean_matches_reference=True)
            records = {"summary.json": summary,
                       "guard_completion.json": dict(status="completed", summary_sha256="wrong"),
                       "temperature_guard_state.json": dict(status="completed", decision="ok")}
            for name, data in records.items():
                (directory / name).write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "guarded validation"):
                analyze(directory, directory / "missing_instances.jsonl", directory / "missing_analysis.json")


if __name__ == "__main__":
    unittest.main()
