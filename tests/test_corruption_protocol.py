"""CPU-only synthetic protocol checks; no research data, scoring, or training."""

import copy
import hashlib
import itertools
import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from pcdc.config import load_config
from pcdc.corruption_protocol import (
    KINDS, CorruptionProtocol, apply_instance, canonical, digest, keyed_seed,
    load_protocol, make_instance, sample_ids, sample_training_instance,
    validation_instances,
)
from pcdc.data import LEVIRDataset, RawPair
from pcdc.perturbations import make_injection_mask
from pcdc.prepare_corruptions import select_calibration_tiles
from pcdc.smoke import create_fixture


ROOT = Path(__file__).resolve().parents[1]


class CorruptionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # This is an explicit CPU smoke configuration, never a resource fallback.
        base = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
        base["runtime"].update(device="cpu", num_threads=2)
        base["data"].update(root=str(self.root / "data"), tile_size=32,
                            expected_pairs={"train": 4, "val": 1, "test": 1})
        base["experiment"]["output_dir"] = str(self.root / "unused_training_output")
        base["evaluation"]["run_test_after_training"] = False
        self.base_path = self.root / "base.yaml"
        self.base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        self.values = yaml.safe_load((ROOT / "configs/corruption_protocol_v1.yaml").read_text(
            encoding="utf-8"
        ))
        self.values.update(name="synthetic_protocol_fixture", base_config="base.yaml",
                           reference_checkpoint="unused_checkpoint.pt",
                           reference_checkpoint_sha256="0" * 64,
                           reference_audit="unused_audit.json", output_dir="candidate_outputs")
        self.values["calibration"]["source_pairs"] = 4
        self.protocol = self._load(self.values)
        self.ids = [f"val/source_{i // 4:02d}.png@x{(i % 2) * 32}_y{((i // 2) % 2) * 32}"
                    for i in range(16)]

    def _load(self, values, name="candidate.yaml"):
        path = self.root / name
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return load_protocol(path)

    def _dataset(self, split="train"):
        create_fixture(self.protocol.base)
        return LEVIRDataset(self.protocol.base, split)

    def _instance(self, **updates):
        arguments = dict(sample_id=self.ids[0], kind="noise", level=3, stream="val",
                         direction="neutral", target_date="A", spatial="local")
        arguments.update(updates)
        return make_instance(self.protocol, **arguments)

    @staticmethod
    def _pair():
        rng = np.random.Generator(np.random.PCG64(195))
        return RawPair(rng.integers(0, 256, (17, 19, 3), dtype=np.uint8),
                       rng.integers(0, 256, (17, 19, 3), dtype=np.uint8),
                       (np.indices((17, 19)).sum(axis=0) % 2).astype(np.uint8),
                       {"fixture": {"values": [1, 2]}})

    def test_relative_paths_and_fingerprint_capture_protocol_not_relocation(self):
        self.assertEqual(self.protocol.path("base_config"), self.base_path)
        self.assertEqual(self.protocol.path("output_dir"), self.root / "candidate_outputs")
        self.assertEqual(self.protocol.base.values["runtime"]["device"], "cpu")
        self.assertFalse(self.protocol.base.run_test_after_training)
        relocated = copy.deepcopy(self.values)
        relocated.update(base_config=str(self.base_path), reference_checkpoint="other/model.pt",
                         reference_audit="other/audit.json", output_dir="other/output")
        other = self._load(relocated, "relocated.yaml")
        self.assertEqual(self.protocol.fingerprint, other.fingerprint)
        for field, replacement in (("reference_checkpoint_sha256", "1" * 64),
                                   ("name", "different_protocol_name")):
            values = copy.deepcopy(self.values)
            values[field] = replacement
            self.assertNotEqual(self.protocol.fingerprint, self._load(values).fingerprint)
        values = copy.deepcopy(self.values)
        values["corruptions"]["noise"]["neutral"][0]["sigma"] = 0.011
        self.assertNotEqual(self.protocol.fingerprint, self._load(values).fingerprint)
        base = copy.deepcopy(self.protocol.base)
        base.values["data"]["mean"][0] += 0.01
        changed = CorruptionProtocol(copy.deepcopy(self.values), self.protocol.source, base)
        self.assertNotEqual(self.protocol.fingerprint, changed.fingerprint)
        for section, key in (("runtime", "num_threads"), ("experiment", "seed"),
                             ("train", "eval_batch_size"), ("train", "num_workers")):
            base = copy.deepcopy(self.protocol.base)
            base.values[section][key] += 1
            changed = CorruptionProtocol(copy.deepcopy(self.values), self.protocol.source, base)
            with self.subTest(base_setting=f"{section}.{key}"):
                self.assertNotEqual(self.protocol.fingerprint, changed.fingerprint)

    def test_canonical_hashes_and_seed_encoding_are_stable(self):
        first, second = {"b": [3, 2], "a": 1}, {"a": 1, "b": [3, 2]}
        self.assertEqual(canonical(first), '{"a":1,"b":[3,2]}')
        self.assertEqual(digest(first), digest(second))
        parts = ("fixture", first, 7)
        expected = int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8], "big")
        self.assertEqual(keyed_seed(*parts), expected)
        self.assertEqual(keyed_seed("fixture", second, 7), expected)
        self.assertNotEqual(keyed_seed("ab", "c"), keyed_seed("a", "bc"))
        self.assertGreaterEqual(expected, 0)
        self.assertLess(expected, 2**64)
        with self.assertRaises(ValueError):
            canonical({"nonfinite": float("nan")})

    def test_sample_ids_match_loader_and_require_no_pixel_reads(self):
        dataset = self._dataset()
        with patch.object(dataset, "read_pair", side_effect=AssertionError("unexpected pixel read")):
            ids = sample_ids(dataset)
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 16)
        self.assertEqual(ids[:4], ["train/train_0.png@x0_y0", "train/train_0.png@x32_y0",
                                  "train/train_0.png@x0_y32", "train/train_0.png@x32_y32"])
        for index in (0, 3, 7, 15):
            self.assertEqual(ids[index], dataset[index]["sample_id"])
        self.assertFalse((self.protocol.base.data_root / "test").exists())

    def test_validation_manifest_has_45_exactly_balanced_cells(self):
        instances = list(validation_instances(self.protocol, self.ids))
        self.assertEqual(len(instances), 45 * len(self.ids))
        counts = Counter(item["condition_id"] for item in instances)
        self.assertEqual(set(counts), {f"{kind}_l{level}" for kind in KINDS for level in range(1, 6)})
        self.assertEqual(set(counts.values()), {len(self.ids)})
        per_cell = defaultdict(list)
        for item in instances:
            per_cell[item["condition_id"]].append(item)
            self.assertEqual((item["stream"], item["epoch"], item["view"]), ("val", 0, 0))
            self.assertEqual(item["parameters"], self.values["corruptions"][item["kind"]][
                item["direction"]][item["level"] - 1])
        for condition, cell in per_cell.items():
            with self.subTest(condition=condition):
                self.assertEqual({item["sample_id"] for item in cell}, set(self.ids))
                directions = sorted(self.values["corruptions"][cell[0]["kind"]])
                strata = Counter((item["direction"], item["target_date"], item["spatial_mode"])
                                 for item in cell)
                expected_strata = set(itertools.product(directions, ("A", "B"), ("global", "local")))
                self.assertEqual(set(strata), expected_strata)
                self.assertEqual(set(strata.values()), {len(self.ids) // len(expected_strata)})
        # The actual JSONL representation is byte-repeatable and round-trippable.
        manifest = "".join(canonical(item) + "\n" for item in instances)
        repeated = "".join(canonical(item) + "\n" for item in validation_instances(self.protocol, self.ids))
        self.assertEqual(manifest.encode(), repeated.encode())
        self.assertEqual([json.loads(line) for line in manifest.splitlines()], instances)

    def test_validation_assignment_is_independent_of_sample_iteration_order(self):
        forward = list(validation_instances(self.protocol, self.ids))
        reverse = list(validation_instances(self.protocol, list(reversed(self.ids))))
        key = lambda item: (item["sample_id"], item["condition_id"])
        self.assertEqual({key(item): item for item in forward}, {key(item): item for item in reverse})
        # Non-divisible counts still differ by at most one sample per stratum.
        small = list(validation_instances(self.protocol, self.ids[:13]))
        for kind in KINDS:
            strata = Counter((item["direction"], item["target_date"], item["spatial_mode"])
                             for item in small if item["kind"] == kind and item["level"] == 1)
            expected = itertools.product(self.values["corruptions"][kind], ("A", "B"), ("global", "local"))
            frequencies = [strata[group] for group in expected]
            self.assertLessEqual(max(frequencies) - min(frequencies), 1)

    def test_all_severity_levels_pair_masks_and_noise_fields(self):
        for kind in KINDS:
            direction = sorted(self.values["corruptions"][kind])[0]
            instances = [self._instance(kind=kind, direction=direction, level=level)
                         for level in range(1, 6)]
            with self.subTest(kind=kind):
                self.assertEqual(len({item["seed"] for item in instances}), 1)
                self.assertEqual(len({canonical(item["injection_mask"]) for item in instances}), 1)
                self.assertEqual(len({canonical(item["parameters"]) for item in instances}), 5)
                masks = [make_injection_mask((17, 19), item["injection_mask"]) for item in instances]
                for mask in masks[1:]:
                    np.testing.assert_array_equal(mask, masks[0])
        noise = [self._instance(level=level) for level in range(1, 6)]
        fields = [np.random.Generator(np.random.PCG64(item["seed"])).standard_normal((17, 19, 3))
                  for item in noise]
        for field in fields[1:]:
            np.testing.assert_array_equal(field, fields[0])

    def test_train_stream_epoch_view_and_selection_are_stateless(self):
        sample_id = self.ids[0].replace("val/", "train/", 1)
        train = self._instance(sample_id=sample_id, stream="train")
        self.assertEqual(train, self._instance(sample_id=sample_id, stream="train"))
        self.assertNotEqual(train["seed"], self._instance()["seed"])
        epoch = self._instance(sample_id=sample_id, stream="train", epoch=1)
        view = self._instance(sample_id=sample_id, stream="train", view=1)
        self.assertEqual(len({train["seed"], epoch["seed"], view["seed"]}), 3)
        self.assertNotEqual(train["injection_mask"], epoch["injection_mask"])
        sampled = sample_training_instance(self.protocol, sample_id, epoch=4, view=2)
        self.assertEqual((sampled["stream"], sampled["epoch"], sampled["view"]), ("train", 4, 2))
        for i in range(8):
            sample_training_instance(self.protocol, f"train/other_{i}.png@x0_y0", epoch=i)
        self.assertEqual(sampled, sample_training_instance(self.protocol, sample_id, epoch=4, view=2))
        self.assertNotEqual(sampled["seed"], sample_training_instance(
            self.protocol, sample_id, epoch=5, view=2)["seed"])
        self.assertNotEqual(sampled["seed"], sample_training_instance(
            self.protocol, sample_id, epoch=4, view=3)["seed"])
        targets = {sample_training_instance(self.protocol, f"train/fixture_{i}.png@x0_y0", epoch=0)[
            "target_date"] for i in range(64)}
        self.assertEqual(targets, {"A", "B"})

    def test_protocol_and_application_do_not_consume_global_rng_state(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        list(validation_instances(self.protocol, self.ids))
        sample_training_instance(self.protocol, "train/fixture.png@x0_y0", epoch=2, view=1)
        apply_instance(self._pair(), self._instance())
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_application_preserves_labels_other_date_inputs_and_metadata(self):
        pair = self._pair()
        originals = copy.deepcopy(pair)
        for array in (pair.image_a, pair.image_b, pair.mask):
            array.flags.writeable = False
        for kind, target, spatial in itertools.product(KINDS, ("A", "B"), ("global", "local")):
            direction = sorted(self.values["corruptions"][kind])[0]
            instance = self._instance(kind=kind, direction=direction, target_date=target, spatial=spatial)
            with self.subTest(kind=kind, target=target, spatial=spatial):
                output = apply_instance(pair, instance)
                self.assertEqual(output.metadata["corruption_instance_json"], canonical(instance))
                np.testing.assert_array_equal(output.mask, originals.mask)
                self.assertFalse(np.shares_memory(output.mask, pair.mask))
                changed = output.image_a if target == "A" else output.image_b
                source = pair.image_a if target == "A" else pair.image_b
                other = output.image_b if target == "A" else output.image_a
                untouched = pair.image_b if target == "A" else pair.image_a
                np.testing.assert_array_equal(other, untouched)
                self.assertFalse(np.shares_memory(other, untouched))
                alpha = make_injection_mask(pair.mask.shape, instance["injection_mask"])
                np.testing.assert_array_equal(changed[alpha == 0], source[alpha == 0])
                self.assertTrue(np.any(changed != source))
                different_labels = RawPair(pair.image_a, pair.image_b, 1 - pair.mask, pair.metadata)
                relabeled = apply_instance(different_labels, instance)
                np.testing.assert_array_equal(relabeled.image_a, output.image_a)
                np.testing.assert_array_equal(relabeled.image_b, output.image_b)
                output.metadata["fixture"]["values"].append(99)
        for name in ("image_a", "image_b", "mask"):
            np.testing.assert_array_equal(getattr(pair, name), getattr(originals, name))
        self.assertEqual(pair.metadata, originals.metadata)

    def test_application_rejects_invalid_pair_or_target(self):
        pair = self._pair()
        invalid = [RawPair(pair.image_a, pair.image_b[:-1], pair.mask),
                   RawPair(pair.image_a, pair.image_b, pair.mask.astype(np.float32)),
                   RawPair(pair.image_a, pair.image_b, pair.mask * 255),
                   RawPair(pair.image_a, pair.image_b, pair.mask[..., None]),
                   RawPair(pair.image_a.astype(np.float32), pair.image_b, pair.mask),
                   RawPair(pair.image_a, pair.image_b.astype(np.float32), pair.mask)]
        for index, bad_pair in enumerate(invalid):
            for target in ("A", "B"):
                with self.subTest(pair=index, target=target), self.assertRaises(ValueError):
                    apply_instance(bad_pair, self._instance(target_date=target))
        instance = self._instance()
        instance["target_date"] = "both"
        with self.assertRaises(ValueError):
            apply_instance(pair, instance)

    def test_calibration_selection_is_label_independent_distinct_and_train_only(self):
        dataset = self._dataset()
        with patch.object(dataset, "read_pair", side_effect=AssertionError("label/image read forbidden")), \
                patch("PIL.Image.open", side_effect=AssertionError("file read forbidden")):
            selected = select_calibration_tiles(dataset, self.protocol)
            repeated = select_calibration_tiles(dataset, self.protocol)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({dataset.tiles[index][0] for index in selected}), 4)
        for split in ("val", "test"):
            no_files = SimpleNamespace(split=split, records=dataset.records, tiles=dataset.tiles)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "training split"):
                select_calibration_tiles(no_files, self.protocol)
        too_many = copy.deepcopy(self.protocol)
        too_many.values["calibration"]["source_pairs"] = 5
        with self.assertRaisesRegex(ValueError, "distinct training source pairs"):
            select_calibration_tiles(dataset, too_many)
        self.assertFalse((self.protocol.base.data_root / "test").exists())

    def test_duplicate_yaml_duplicate_levels_and_missing_parameters_are_rejected(self):
        path = self.root / "duplicate.yaml"
        text = yaml.safe_dump(self.values, sort_keys=False)
        path.write_text(text + "\nname: overwritten\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            load_protocol(path)
        path.write_text("seeds:\n  val: 1\n  val: 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            load_protocol(path)
        bad = copy.deepcopy(self.values)
        bad["corruptions"]["noise"]["neutral"][1] = copy.deepcopy(
            bad["corruptions"]["noise"]["neutral"][0])
        with self.assertRaisesRegex(ValueError, "duplicate severity levels"):
            self._load(bad)
        bad = copy.deepcopy(self.values)
        bad["corruptions"]["brightness"]["darken"][:2] = [{"factor": 1}, {"factor": 1.0}]
        with self.assertRaisesRegex(ValueError, "duplicate severity levels"):
            self._load(bad)
        for parameters in ({}, {"sigma": 0.01, "unrecognized": 1}):
            bad = copy.deepcopy(self.values)
            bad["corruptions"]["noise"]["neutral"][0] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self._load(bad)

    def test_invalid_protocol_fields_seeds_masks_and_calibration_rejected(self):
        replacements = [
            (("schema_version",), True), (("schema_version",), 2),
            (("status",), "accepted"), (("name",), " "),
            (("placement",), "after_normalization"), (("spatial_modes",), ["global"]),
            (("reference_checkpoint_sha256",), "bad"),
            (("reference_checkpoint_sha256",), "A" * 64),
            (("seeds", "train"), -1), (("seeds", "train"), True),
            (("seeds", "train"), 3.0), (("seeds", "train"), "3"),
            (("seeds", "train"), self.values["seeds"]["val"]),
            (("local_mask", "center_x_range"), [0.7, 0.3]),
            (("local_mask", "center_y_range"), [-0.1, 0.3]),
            (("local_mask", "radius_x_range"), [0, 0.3]),
            (("local_mask", "radius_y_range"), [0.1, 1.1]),
            (("local_mask", "feather"), 0), (("local_mask", "feather"), float("nan")),
            (("calibration", "source_pairs"), 0), (("calibration", "selection_seed"), True),
            (("calibration", "preview_tile_pixels"), 31),
            (("calibration", "preview_levels"), [1, 1, 5]),
            (("calibration", "preview_levels"), [0, 3, 5]),
            (("calibration", "preview_levels"), [True, 3, 5]),
            (("calibration", "preview_levels"), [[1], 3, 5]),
            (("corruptions", "noise", "neutral"), [{"sigma": 0.01}]),
        ]
        for keys, value in replacements:
            bad = copy.deepcopy(self.values)
            parent = bad
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            with self.subTest(path=keys, value=value), self.assertRaises(ValueError):
                self._load(bad)
        for key in ("unrecognized", "pse"):
            bad = dict(self.values, **{key: True})
            with self.subTest(extra=key), self.assertRaises(ValueError):
                self._load(bad)

    def test_instance_validation_and_heldout_test_firewall(self):
        invalid = [dict(stream="test", sample_id="test/fixture.png@x0_y0"),
                   dict(sample_id="train/fixture.png@x0_y0"), dict(sample_id=123),
                   dict(kind="rotate"), dict(direction="unknown"), dict(target_date="both"),
                   dict(spatial="label"), dict(level=0), dict(level=6), dict(level=True),
                   dict(level=1.0), dict(epoch=-1), dict(epoch=True), dict(epoch=1), dict(view=1)]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self._instance(**settings)
        for ids in ([], [self.ids[0], self.ids[0]], ["train/x"], ["test/x"], [123]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                list(validation_instances(self.protocol, ids))
        for sample_id in ("val/x", "test/x"):
            with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                sample_training_instance(self.protocol, sample_id, epoch=0)
        for epoch, view in ((-1, 0), (0, -1), (True, 0), (0, 0.5)):
            with self.subTest(epoch=epoch, view=view), self.assertRaises(ValueError):
                sample_training_instance(self.protocol, "train/x", epoch=epoch, view=view)

    def test_protocol_rejects_base_config_with_test_access_enabled(self):
        base = yaml.safe_load(self.base_path.read_text(encoding="utf-8"))
        base["evaluation"]["run_test_after_training"] = True
        self.base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run_test_after_training: false"):
            self._load(self.values)


if __name__ == "__main__":
    unittest.main()
