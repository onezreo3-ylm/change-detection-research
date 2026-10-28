"""Synthetic-only QA supplement tests; no real data, model, inference or GPU."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pcdc.corruption_protocol import KINDS, keyed_seed, load_protocol, sample_ids
from pcdc.data import RawPair
from pcdc.prepare_corruptions import calibration_previews, select_calibration_tiles
from scripts.supplement_train_qa import (
    ENVIRONMENT_FIELDS, SupplementConfig, annotate_report, load_supplement_config,
    posthoc_overlap, preparation_environment, prepare_supplement, preview_batches,
    select_supplement_tiles, verify_coverage, verify_environment,
    verify_frozen_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_dataset(count=20, tiles_per_side=4, split="train"):
    """No file-backed dataset; even read_pair returns explicitly synthetic arrays."""
    records = [SimpleNamespace(name=f"synthetic_{index:03d}.png") for index in range(count)]
    tiles = [(index, x * 32, y * 32) for index in range(count)
             for y in range(tiles_per_side) for x in range(tiles_per_side)]

    def read_pair(record, box):
        index = int(record.name.split("_")[1].split(".")[0])
        grid = np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3)
        image_a = ((grid + 11 * index) % 256).astype(np.uint8)
        image_b = ((grid * 3 + 17 * index) % 256).astype(np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        if index % 2:
            mask[8:17, 10:21] = 1
        return RawPair(image_a, image_b, mask)

    return SimpleNamespace(split=split, records=records, tiles=tiles, tile_size=32, read_pair=read_pair)


class SupplementTrainQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Loading these YAMLs only checks schemas; it never opens dataset files.
        self.settings = load_supplement_config(ROOT / "configs/train_qa_supplement_20260913.yaml")
        self.protocol = copy.deepcopy(load_protocol(ROOT / "configs/corruption_protocol_v1.yaml"))
        self.protocol.values["calibration"]["preview_tile_pixels"] = 32
        self.dataset = synthetic_dataset()

    def test_selection_continues_original_hash_order_without_pixel_reads(self):
        with patch.object(self.dataset, "read_pair", side_effect=AssertionError("pixel read")), \
                patch("PIL.Image.open", side_effect=AssertionError("file read")):
            original = select_calibration_tiles(self.dataset, self.protocol)
            selected = select_supplement_tiles(self.dataset, **self.settings.values["selection"])
            repeated = select_supplement_tiles(self.dataset, **self.settings.values["selection"])
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 12)
        expected_order = sorted(range(20), key=lambda index: (
            keyed_seed(730194, self.dataset.records[index].name), self.dataset.records[index].name))
        self.assertEqual([self.dataset.tiles[index][0] for index in original], expected_order[:4])
        self.assertEqual([self.dataset.tiles[index][0] for index in selected], expected_order[4:16])
        for tile_index in selected:
            record_index = self.dataset.tiles[tile_index][0]
            name = self.dataset.records[record_index].name
            self.assertEqual(tile_index, record_index * 16 + keyed_seed(730194, name, "tile") % 16)
        self.assertFalse(set(original) & set(selected))

    def test_selection_rejects_other_splits_wrong_grid_and_insufficient_sources(self):
        for split in ("val", "test"):
            dataset = SimpleNamespace(split=split)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "training split"):
                select_supplement_tiles(dataset, **self.settings.values["selection"])
        with self.assertRaisesRegex(ValueError, "distinct training source pairs"):
            select_supplement_tiles(synthetic_dataset(count=15), **self.settings.values["selection"])
        with self.assertRaisesRegex(ValueError, "specified 16 tiles"):
            select_supplement_tiles(synthetic_dataset(tiles_per_side=2), **self.settings.values["selection"])
        dataset = synthetic_dataset()
        dataset.records[1].name = dataset.records[0].name
        with self.assertRaisesRegex(ValueError, "filenames must be unique"):
            select_supplement_tiles(dataset, **self.settings.values["selection"])

    def test_six_batches_flip_every_rows_date_parity(self):
        indices = list(range(12))
        batches = preview_batches(indices, group_size=4, row_rotations=[0, 1])
        self.assertEqual(len(batches), 6)
        self.assertEqual(len({item["batch_id"] for item in batches}), 6)
        self.assertEqual(batches[0]["indices"], [0, 1, 2, 3])
        self.assertEqual(batches[1]["indices"], [1, 2, 3, 0])
        targets = {}
        for batch in batches:
            for row, index in enumerate(batch["indices"]):
                for spatial in ("global", "local"):
                    targets.setdefault((index, spatial), []).append(("A", "B")[(row + (spatial == "local")) % 2])
        self.assertEqual(set(targets), {(index, spatial) for index in indices for spatial in ("global", "local")})
        self.assertTrue(all(sorted(value) == ["A", "B"] for value in targets.values()))
        for bad_indices, group, rotations in (([0, 1, 2], 4, [0, 1]), ([0, 0, 1, 2], 4, [0, 1]),
                                             ([0, 1, 2, 3], 4, [0, 2]), ([0, 1, 2, 3], 2, [0, 1])):
            with self.subTest(indices=bad_indices, rotations=rotations), self.assertRaises(ValueError):
                preview_batches(bad_indices, group_size=group, row_rotations=rotations)

    def test_posthoc_overlap_has_explicit_denominators_and_zero_positive_null(self):
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        alpha = np.array([[0.1, 1], [0, 0]], dtype=np.float32)
        with patch("scripts.supplement_train_qa.make_injection_mask", return_value=alpha) as generator:
            result = posthoc_overlap(mask, {"mode": "test_only"})
        generator.assert_called_once_with(mask.shape, {"mode": "test_only"})
        self.assertEqual(result["positive_pixels"], 2)
        self.assertEqual(result["injection_support_pixels"], 2)
        self.assertEqual(result["positive_pixels_in_injection_support"], 1)
        self.assertEqual(result["positive_pixels_outside_injection_support"], 1)
        self.assertEqual(result["positive_coverage_fraction"], 0.5)
        self.assertEqual(result["positive_fraction_of_support"], 0.5)
        result = posthoc_overlap(np.zeros((2, 2), dtype=np.uint8), {"mode": "global"})
        self.assertIsNone(result["positive_coverage_fraction"])
        self.assertEqual(result["positive_fraction_of_support"], 0)
        for bad in (mask.astype(np.float32), mask * 255, mask[..., None]):
            with self.assertRaises(ValueError):
                posthoc_overlap(bad, {"mode": "global"})

    def test_real_preview_helper_on_synthetic_pairs_preserves_invariants_and_all_levels(self):
        indices = [0, 16, 32, 48]
        ids = sample_ids(self.dataset)
        pairs = {ids[index]: self.dataset.read_pair(self.dataset.records[index // 16], None)
                 for index in indices}
        reports, instances = [], []
        for batch in preview_batches(indices, group_size=4, row_rotations=[0, 1]):
            output = self.root / batch["batch_id"]
            output.mkdir()
            with patch("builtins.print"):
                original = calibration_previews(self.protocol, self.dataset, batch["indices"], output)
            report = annotate_report(original, pairs, batch["batch_id"])
            self.assertEqual(len(report["preview_sheets"]), 18)
            self.assertEqual(len(report["instances"]), 360)
            for before, after in zip(original["instances"], report["instances"]):
                self.assertEqual({key: after[key] for key in before}, before)
                self.assertEqual(len(after["output_rgb_sha256"]), 64)
                self.assertTrue(all(after[key] for key in (
                    "label_unchanged", "other_date_unchanged", "outside_support_unchanged", "inputs_not_mutated")))
                self.assertIn("posthoc_semantic_overlap", after)
            reports.append(report)
            instances.extend(report["instances"])
        coverage = verify_coverage(instances, list(pairs))
        self.assertTrue(coverage["both_dates_per_sample_family_spatial_mode"])
        self.assertEqual(coverage["instance_count"], 720)
        self.assertEqual(coverage["numeric_levels"], [1, 2, 3, 4, 5])
        self.assertEqual(sum(len(report["preview_sheets"]) for report in reports) * 3, 108)
        with self.assertRaisesRegex(AssertionError, "both dates"):
            verify_coverage(instances[:-1], list(pairs))
        with self.assertRaisesRegex(AssertionError, "both dates"):
            verify_coverage(instances + instances[:1], list(pairs))

    def test_annotation_does_not_allow_unselected_or_nontrain_instances(self):
        pair = RawPair(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8),
                       np.zeros((2, 2), dtype=np.uint8))
        for instance in ({"stream": "val", "sample_id": "train/selected"},
                         {"stream": "train", "sample_id": "train/unselected"}):
            with self.assertRaisesRegex(ValueError, "prespecified selected training"):
                annotate_report({"instances": [instance]}, {"train/selected": pair}, "synthetic")

    def test_environment_version_or_missing_field_fails_closed(self):
        reference = preparation_environment()
        verify_environment(reference, dict(reference))
        for key in ENVIRONMENT_FIELDS:
            actual = dict(reference)
            actual[key] = "different"
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "environment changed"):
                verify_environment(reference, actual)
            actual = dict(reference)
            del actual[key]
            with self.assertRaises(ValueError):
                verify_environment(reference, actual)

    def test_frozen_subset_allows_new_evaluator_files_but_records_current_graph(self):
        frozen = {"pcdc/operator.py": "a" * 64}
        current = {**frozen, "pcdc/new_evaluator.py": "b" * 64}
        with patch("scripts.supplement_train_qa.source_hashes", return_value=current):
            result = verify_frozen_sources(frozen, dict(frozen), 1)
        self.assertEqual(result, current)
        self.assertEqual(frozen, {"pcdc/operator.py": "a" * 64})

    def test_frozen_subset_rejects_changed_missing_or_invalid_original_identities(self):
        frozen = {"pcdc/operator.py": "a" * 64}
        for current in ({}, {"pcdc/operator.py": "b" * 64}):
            with self.subTest(current=current), \
                    patch("scripts.supplement_train_qa.source_hashes", return_value=current), \
                    self.assertRaisesRegex(ValueError, "SHA256 identity changed"):
                verify_frozen_sources(frozen, dict(frozen), 1)
        with patch("scripts.supplement_train_qa.source_hashes", return_value=frozen):
            with self.assertRaisesRegex(ValueError, "identities changed"):
                verify_frozen_sources(frozen, {"pcdc/operator.py": "b" * 64}, 1)
            with self.assertRaisesRegex(ValueError, "identities changed"):
                verify_frozen_sources(frozen, dict(frozen), 2)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                verify_frozen_sources({"pcdc/operator.py": "bad"}, {"pcdc/operator.py": "bad"}, 1)

    def test_nonoverwrite_refuses_before_protocol_checkpoint_or_dataset_access(self):
        values = copy.deepcopy(self.settings.values)
        values["output_dir"] = str(self.root)
        config = SupplementConfig(values, self.root / "synthetic_config.yaml")
        with patch("scripts.supplement_train_qa.load_protocol", side_effect=AssertionError("protocol read")), \
                patch("scripts.supplement_train_qa.read_checkpoint", side_effect=AssertionError("checkpoint read")), \
                patch("scripts.supplement_train_qa.LEVIRDataset", side_effect=AssertionError("dataset read")):
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                prepare_supplement(config)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_config_requires_hash_bindings_pending_review_and_fixed_coverage(self):
        self.assertEqual(self.settings.values["selection"], {
            "seed": 730194, "skip_source_pairs": 4, "source_pairs": 12, "tiles_per_source": 16})
        self.assertEqual(self.settings.values["expected_protocol_sha256"],
                         "98f5e62efb60fa0cefd17c0191522cb11b0083755f1215484206a8b5e8469296")
        self.assertEqual(self.settings.values["expected_checkpoint_sha256"],
                         "033a0aa14becabc97c60b368b6d7c2e49bc8b386daa61dcc77e761c81ae747f3")
        invalid = [(('schema_version',), True), (('status',), 'accepted'),
                   (('expected_protocol_sha256',), 'bad'), (('expected_checkpoint_sha256',), 'A' * 64),
                   (('selection', 'seed'), True), (('selection', 'source_pairs'), 11),
                   (('rendering', 'group_size'), 2), (('rendering', 'row_rotations'), [0, 2]),
                   (('rendering', 'row_rotations'), [False, True]),
                   (('rendering', 'preview_levels'), [1, 2, 5]),
                   (('verification', 'prepare_environment_fields'), ['numpy'])]
        path = self.root / "invalid.yaml"
        for keys, value in invalid:
            values = copy.deepcopy(self.settings.values)
            parent = values
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
            with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                load_supplement_config(path)
        path.write_text(yaml.safe_dump(self.settings.values) + "\nname: duplicate\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            load_supplement_config(path)


if __name__ == "__main__":
    unittest.main()
