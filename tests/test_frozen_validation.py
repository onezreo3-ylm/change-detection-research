"""Frozen-bundle CPU engineering checks; never research-data evaluation.

One explicitly synthetic CPU checkpoint is trained for this class. Each test
then owns a temporary bundle and validation-data copy. There are no held-out
test files, no GPU model work, and no modifications to frozen workspace code.
"""

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from pcdc.config import load_config, protocol_signature
from pcdc.corruption_protocol import canonical, load_protocol, make_instance, sample_ids
from pcdc.data import LEVIRDataset, file_sha256, seed_worker
from pcdc.engine import source_files, train
from pcdc.evaluate_corruptions import evaluate as score_frozen_validation
from pcdc.frozen_validation import (
    EvaluationConfig, FrozenInstanceDataset, ReviewRequired,
    build_validation_dataset, load_bundle, load_evaluation_config,
)
from pcdc.perturbations import make_injection_mask
from pcdc.prepare_corruptions import prepare
from pcdc.smoke import create_fixture


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CORE = tuple(f"pcdc/{name}.py" for name in (
    "__init__", "__main__", "benchmark", "config", "corruption_protocol", "data",
    "engine", "inspect_labels", "losses", "metrics", "model", "perturbations",
    "prepare_corruptions", "runtime", "smoke",
))
BINDINGS = ("protocol_sha256", "manifest_sha256", "instances_sha256", "checkpoint_sha256")


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class FrozenValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.fixture_root = Path(cls.fixture.name)
        cls.addClassCleanup(random.setstate, random.getstate())
        cls.addClassCleanup(np.random.set_state, np.random.get_state())
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.addClassCleanup(torch.use_deterministic_algorithms,
                            torch.are_deterministic_algorithms_enabled())
        base = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml").values)
        base["runtime"].update(device="cpu", num_threads=2)
        base["data"].update(root=str(cls.fixture_root / "data"), tile_size=32,
                            expected_image_size=64,
                            expected_pairs={"train": 2, "val": 1, "test": 1})
        base["train"].update(epochs=1, batch_size=2, eval_batch_size=2, num_workers=0, amp=False)
        base["experiment"].update(name="frozen_validation_synthetic_fixture",
                                  output_dir=str(cls.fixture_root / "baseline"))
        base["evaluation"]["run_test_after_training"] = False
        base_path = cls.fixture_root / "base.yaml"
        write_yaml(base_path, base)
        base_config = load_config(base_path)
        create_fixture(base_config)
        # CPU training must not collect otherwise-available CUDA RNG states.
        with patch("torch.cuda.is_available", return_value=False), \
                patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("GPU RNG access")), \
                redirect_stdout(io.StringIO()):
            train(base_config)
        cls.checkpoint_path = base_config.output_dir / "best.pt"
        cls.checkpoint_sha256 = file_sha256(cls.checkpoint_path)
        protocol = yaml.safe_load((ROOT / "configs/corruption_protocol_v1.yaml").read_text(encoding="utf-8"))
        protocol.update(name="frozen_validation_synthetic_protocol", base_config=str(base_path),
                        reference_checkpoint=str(cls.checkpoint_path),
                        reference_checkpoint_sha256=cls.checkpoint_sha256,
                        reference_audit=str(base_config.output_dir / "data_audit.json"),
                        output_dir=str(cls.fixture_root / "prepared"))
        protocol["calibration"].update(source_pairs=2, preview_levels=[1, 5], preview_tile_pixels=32)
        protocol_path = cls.fixture_root / "protocol.yaml"
        write_yaml(protocol_path, protocol)
        # Accurately simulate the legacy preparation graph: its original 15
        # files are snapshotted, while newer evaluation modules exist locally.
        installed = source_files()
        frozen = {name: installed[name] for name in ORIGINAL_CORE}
        with patch("pcdc.prepare_corruptions.source_files", return_value=frozen), \
                patch("pcdc.prepare_corruptions.source_hashes", side_effect=lambda: {
                    name: file_sha256(path) for name, path in frozen.items()}), \
                redirect_stdout(io.StringIO()):
            prepare(load_protocol(protocol_path))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        shutil.copytree(self.fixture_root / "prepared", self.prepared)
        self.data_root = self.root / "data"
        # Validation replay needs neither training files nor held-out test files.
        shutil.copytree(self.fixture_root / "data" / "val", self.data_root / "val")
        shutil.copyfile(self.fixture_root / "data" / "SYNTHETIC_ONLY.json",
                        self.data_root / "SYNTHETIC_ONLY.json")
        self.base_path = self.prepared / "base_config.yaml"
        base = yaml.safe_load(self.base_path.read_text(encoding="utf-8"))
        base["data"]["root"] = str(self.data_root)
        write_yaml(self.base_path, base)
        self.protocol_path = self.prepared / "candidate_resolved.yaml"
        protocol = yaml.safe_load(self.protocol_path.read_text(encoding="utf-8"))
        protocol.update(base_config=str(self.base_path), output_dir=str(self.prepared))
        write_yaml(self.protocol_path, protocol)
        self.manifest_path = self.prepared / "val_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.instance_path = self.prepared / self.manifest["instance_file"]
        self.environment_path = self.prepared / "environment.json"
        self.acceptance_path = self.root / "acceptance.yaml"
        self.job_path = self.root / "evaluation.yaml"
        self.job_values = {
            "schema_version": 1, "name": "synthetic_frozen_validation_check",
            "protocol_config": "prepared/candidate_resolved.yaml",
            "manifest": "prepared/val_manifest.json", "acceptance": "acceptance.yaml",
            "output_dir": "unused_scoring_output",
        }
        self._rebind_expected_hashes()

    def _rebind_expected_hashes(self):
        """Rebind checksum expectations only; never repair rows/counters/inventory."""
        protocol = load_protocol(self.protocol_path)
        self.manifest["archived_base_config_sha256"] = file_sha256(self.base_path)
        self.manifest["protocol_sha256"] = protocol.fingerprint
        self.manifest["base_protocol_signature"] = protocol_signature(protocol.base.values)
        self.manifest["instance_file_sha256"] = file_sha256(self.instance_path)
        write_json(self.manifest_path, self.manifest)
        self.job_values.update(
            protocol_sha256=protocol.fingerprint, manifest_sha256=file_sha256(self.manifest_path),
            instances_sha256=file_sha256(self.instance_path), checkpoint_sha256=self.checkpoint_sha256,
            preparation_environment_sha256=file_sha256(self.environment_path),
        )
        write_yaml(self.job_path, self.job_values)
        self._review()
        return load_evaluation_config(self.job_path)

    def _review(self, **changes):
        value = {"schema_version": 1, "status": "engineering_only",
                 "approved_for": "validation_only", "visual_review": "not_applicable_synthetic",
                 "reviewer": "automated-unit-test-fixture",
                 "notes": "Explicit synthetic CPU engineering check; not approval of research severities."}
        value.update({key: self.job_values[key] for key in BINDINGS})
        value.update(changes)
        write_yaml(self.acceptance_path, value)
        return value

    def _job(self):
        return load_evaluation_config(self.job_path)

    def _bundle(self, require_review=True):
        return load_bundle(self._job(), require_review=require_review)

    def _no_model_or_data_reads(self):
        stack = ExitStack()
        real_open = Path.open

        def checked_open(path, *args, **kwargs):
            resolved = path.resolve()
            if resolved == self.checkpoint_path or self.data_root in resolved.parents:
                raise AssertionError(f"Preflight accessed model/data: {resolved}")
            return real_open(path, *args, **kwargs)

        stack.enter_context(patch.object(Path, "open", checked_open))
        stack.enter_context(patch("pcdc.frozen_validation.read_checkpoint",
                                  side_effect=AssertionError("Checkpoint deserialization")))
        stack.enter_context(patch.object(LEVIRDataset, "__init__",
                                        side_effect=AssertionError("Dataset construction")))
        stack.enter_context(patch("torch.cuda.is_available", side_effect=AssertionError("Device access")))
        return stack

    def test_evaluation_schema_relative_paths_and_duplicate_yaml(self):
        job = self._job()
        self.assertIsInstance(job, EvaluationConfig)
        self.assertEqual(job.path("protocol_config"), self.protocol_path)
        self.assertEqual(job.path("manifest"), self.manifest_path)
        self.assertEqual(job.path("output_dir"), self.root / "unused_scoring_output")
        bad_settings = [("schema_version", True), ("schema_version", 2), ("name", " "),
                        ("acceptance", None), ("manifest_sha256", "bad"),
                        ("checkpoint_sha256", "A" * 64), ("instances_sha256", 4),
                        ("preparation_environment_sha256", "0" * 63)]
        for key, value in bad_settings:
            bad = dict(self.job_values, **{key: value})
            write_yaml(self.job_path, bad)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self._job()
        write_yaml(self.job_path, dict(self.job_values, unrecognized=True))
        with self.assertRaises(ValueError):
            self._job()
        self.job_path.write_text(yaml.safe_dump(self.job_values) + "\nname: duplicate\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML"):
            self._job()

    def test_preflight_reads_no_model_data_or_device_and_allows_new_evaluation_files(self):
        self._review(status="pending", visual_review="not_completed", reviewer=None)
        with self._no_model_or_data_reads():
            bundle = self._bundle(require_review=False)
        self.assertEqual(bundle.job.source, self.job_path)
        self.assertEqual(bundle.manifest_path, self.manifest_path)
        self.assertEqual(bundle.approval["status"], "pending")
        self.assertEqual(len(bundle.instances), 45 * 4)
        self.assertEqual(set(bundle.manifest["source_sha256"]), set(ORIGINAL_CORE))
        self.assertIn("pcdc/frozen_validation.py", source_files())
        self.assertNotIn("pcdc/frozen_validation.py", bundle.manifest["source_sha256"])
        self.assertFalse((self.prepared / "source" / "pcdc" / "frozen_validation.py").exists())
        for key in BINDINGS:
            self.assertEqual(bundle.fingerprints[key], self.job_values[key])
        self.assertFalse((self.root / "unused_scoring_output").exists())

    def test_missing_and_pending_review_reject_before_any_hash_or_data_work(self):
        job = self._job()
        missing_path = self.root / "missing_review_job.yaml"
        write_yaml(missing_path, dict(job.values, acceptance="absent.yaml"))
        missing = load_evaluation_config(missing_path)
        with patch("pcdc.frozen_validation.file_sha256", side_effect=AssertionError("Hash work before gate")):
            with self.assertRaises(ReviewRequired):
                load_bundle(missing, require_review=True)
            self._review(status="pending", visual_review="not_completed", reviewer=None)
            with self.assertRaises(ReviewRequired):
                load_bundle(job, require_review=True)
        with self._no_model_or_data_reads():
            self.assertIsNone(load_bundle(missing, require_review=False).approval)

    def test_acceptance_bindings_and_schema_are_strict(self):
        invalid = [(key, "f" * 64) for key in BINDINGS]
        invalid += [("schema_version", True), ("status", "accepted"), ("approved_for", "test"),
                    ("visual_review", "unknown"), ("reviewer", True), ("notes", None)]
        for key, value in invalid:
            self._review(**{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._bundle(require_review=False)
        review = self._review()
        review["allow_test"] = True
        write_yaml(self.acceptance_path, review)
        with self.assertRaises(ValueError):
            self._bundle()

    def test_approval_requires_evidence_identity_and_nonempty_notes(self):
        invalid = [dict(status="approved", visual_review="not_completed"),
                   dict(status="approved", visual_review="passed", reviewer=None),
                   dict(status="approved", visual_review="passed", reviewer=" "),
                   dict(status="approved", visual_review="passed", notes=" "),
                   dict(status="engineering_only", visual_review="passed")]
        for changes in invalid:
            self._review(**changes)
            with self.subTest(changes=changes), self.assertRaises(ReviewRequired):
                self._bundle()
        self._review()
        self.assertEqual(self._bundle().approval["status"], "engineering_only")

    def test_engineering_gate_and_checkpoint_cannot_be_relabelled_as_research(self):
        base = yaml.safe_load(self.base_path.read_text(encoding="utf-8"))
        base["experiment"]["kind"] = "research"  # Intentional invalid fixture, never real research.
        write_yaml(self.base_path, base)
        self._rebind_expected_hashes()
        with self.assertRaisesRegex(ReviewRequired, "Engineering-only"):
            self._bundle()
        self._review(status="approved", visual_review="passed",
                     notes="Synthetic negative test: a smoke checkpoint must never be relabelled.")
        bundle = self._bundle()
        with patch.object(LEVIRDataset, "__init__", side_effect=AssertionError("Dataset before kind check")):
            with self.assertRaisesRegex(ValueError, "Cannot relabel"):
                build_validation_dataset(bundle)

    def test_bound_artifact_byte_tampering_is_rejected(self):
        paths = [self.manifest_path, self.instance_path, self.base_path, self.environment_path,
                 self.prepared / "candidate_source.yaml",
                 self.prepared / "source" / "pcdc" / "data.py"]
        for path in paths:
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    self._bundle(require_review=False)
            finally:
                path.write_bytes(original)
        # The index is bound as a parsed mapping, not by its whitespace. Change
        # an actual digest rather than treating equivalent JSON as a mutation.
        index_path = self.prepared / "source_sha256.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["pcdc/data.py"] = "f" * 64
        write_json(index_path, index)
        with self.assertRaisesRegex(ValueError, "source hash index changed"):
            self._bundle(require_review=False)

    def test_current_frozen_source_change_rejected_without_editing_workspace(self):
        original = source_files()
        altered = self.root / "altered_data_fixture.py"
        altered.write_bytes(original["pcdc/data.py"].read_bytes() + b"\n# test-only mutation\n")
        changed = dict(original, **{"pcdc/data.py": altered})
        with patch("pcdc.frozen_validation.source_files", return_value=changed):
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                self._bundle(require_review=False)

    def test_environment_versions_rejected_even_after_rebinding_hash(self):
        original = json.loads(self.environment_path.read_text(encoding="utf-8"))
        for key in ("numpy", "pillow", "jpeg_codec", "libjpeg_turbo"):
            changed = dict(original, **{key: "not-the-installed-version"})
            write_json(self.environment_path, changed)
            self._rebind_expected_hashes()
            with self.subTest(version=key), self.assertRaisesRegex(ValueError, "versions differ"):
                self._bundle(require_review=False)

    def test_manifest_counts_and_inventory_structure_rejected_after_rebinding_hash(self):
        original = copy.deepcopy(self.manifest)
        changes = []
        for key in ("condition_counts", "stratum_counts"):
            changed = copy.deepcopy(original)
            first = next(iter(changed[key]))
            changed[key][first] += 1
            changes.append(changed)
        changed = copy.deepcopy(original)
        changed["instance_count"] += 1
        changes.append(changed)
        for key, value in (("split", "test"), ("tiles", 5), ("tile_size", 31)):
            changed = copy.deepcopy(original)
            changed["inventory"][key] = value
            changes.append(changed)
        changed = copy.deepcopy(original)
        changed["inventory"]["entries"][0]["size"][0] = 63
        changes.append(changed)
        changed = copy.deepcopy(original)
        changed["inventory"]["entries"].append(copy.deepcopy(changed["inventory"]["entries"][0]))
        changes.append(changed)
        changed = copy.deepcopy(original)
        changed["instance_file"] = "../val_instances.jsonl"
        changes.append(changed)
        for index, changed in enumerate(changes):
            self.manifest = changed
            self._rebind_expected_hashes()
            with self.subTest(case=index), self.assertRaises(ValueError):
                self._bundle(require_review=False)

    def test_instance_duplicates_bounds_seeds_and_exact_order_rejected_after_rebinding_hash(self):
        lines = self.instance_path.read_text(encoding="utf-8").splitlines()
        mutations = [lines[:-1], lines + [lines[-1]], [lines[0], lines[0], *lines[2:]],
                     [lines[1], lines[0], *lines[2:]], ["", *lines[1:]]]
        for key, value in (("level", 6), ("sample_id", "val/val_0.png@x64_y0"),
                           ("seed", -1), ("target_date", "both")):
            first = json.loads(lines[0])
            first[key] = value
            mutations.append([canonical(first), *lines[1:]])
        first = json.loads(lines[0])
        first["seed"] += 1  # Still a valid seed, but not this frozen realization.
        mutations.append([canonical(first), *lines[1:]])
        for index, changed in enumerate(mutations):
            self.instance_path.write_text("\n".join(changed) + "\n", encoding="utf-8")
            self._rebind_expected_hashes()
            with self.subTest(case=index), self.assertRaises(ValueError):
                self._bundle(require_review=False)

    def test_build_uses_only_validation_and_requires_no_train_or_test_directory(self):
        bundle = self._bundle()
        self.assertFalse((self.data_root / "train").exists())
        self.assertFalse((self.data_root / "test").exists())
        dataset, checkpoint = build_validation_dataset(bundle)
        self.assertEqual(dataset.split, "val")
        self.assertEqual(len(dataset), 4)
        self.assertEqual(checkpoint["kind"], "smoke")
        self.assertNotIn("test", checkpoint["manifests"])
        self.assertEqual(dataset.manifest(), bundle.manifest["inventory"])

    def test_changed_checkpoint_bytes_rejected_before_deserialization(self):
        different = self.root / "not_a_checkpoint.pt"
        different.write_bytes(b"explicit synthetic corrupt checkpoint fixture")
        protocol = yaml.safe_load(self.protocol_path.read_text(encoding="utf-8"))
        protocol["reference_checkpoint"] = str(different)
        write_yaml(self.protocol_path, protocol)
        self._rebind_expected_hashes()
        bundle = self._bundle()
        with patch("pcdc.frozen_validation.read_checkpoint", side_effect=AssertionError("Unbound checkpoint load")):
            with self.assertRaisesRegex(ValueError, "checkpoint content changed"):
                build_validation_dataset(bundle)

    def test_validation_inventory_and_same_size_content_tampering_are_distinguished(self):
        path = self.data_root / "val" / "A" / "val_0.png"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            build_validation_dataset(self._bundle())
        path.write_bytes(original)
        altered = bytearray(original)
        altered[altered.index(b"IDAT") + 12] ^= 1
        path.write_bytes(altered)
        self.assertEqual(path.stat().st_size, len(original))
        # PNG headers remain readable; the source hash must catch the byte edit
        # even though the inventory has unchanged filenames/dimensions/lengths.
        with self.assertRaisesRegex(ValueError, "source content changed"):
            build_validation_dataset(self._bundle())

    def test_manifest_validation_content_must_match_checkpoint_audit(self):
        source = next(iter(self.manifest["file_sha256"]))
        self.manifest["file_sha256"][source]["image_a"] = "f" * 64
        self._rebind_expected_hashes()
        with self.assertRaisesRegex(ValueError, "does not match checkpoint audit"):
            build_validation_dataset(self._bundle())

    def test_actual_clean_only_cpu_evaluation_allows_missing_acceptance_without_approving_it(self):
        self.job_values["acceptance"] = "missing_acceptance.yaml"
        write_yaml(self.job_path, self.job_values)
        job = self._job()
        output = job.path("output_dir")
        with patch("torch.cuda.is_available", return_value=False), redirect_stdout(io.StringIO()):
            summary = score_frozen_validation(job, clean_only=True)
        self.assertEqual(summary["status"], "completed_clean_only")
        self.assertEqual(summary["kind"], "smoke")
        self.assertEqual(summary["mode"], "clean_validation_only")
        self.assertTrue(summary["clean_matches_reference"])
        self.assertFalse(summary["corruption_scoring_run"])
        self.assertFalse(summary["test_evaluated"])
        self.assertFalse(summary["training_started"])
        self.assertIsNone(summary["approval_status"])
        self.assertFalse(job.path("acceptance").exists())
        self.assertFalse((output / "conditions.json").exists())
        self.assertFalse((output / "conditions").exists())
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, summary)
        clean = json.loads((output / "clean_metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(clean["matches_reference"])
        for key in ("precision", "recall", "f1", "iou", "oa"):
            self.assertEqual(clean["metrics"][key], clean["reference_metrics"][key])
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        self.assertIsNone(provenance["approval"])
        self.assertNotIn(str(job.path("acceptance")), provenance["input_file_sha256"])
        self.assertFalse((self.data_root / "test").exists())
        # Successful clean verification neither grants corruption approval nor
        # makes an existing evaluation directory safe to overwrite.
        with self.assertRaises(ReviewRequired):
            score_frozen_validation(job)
        with self.assertRaises(FileExistsError):
            score_frozen_validation(job, clean_only=True)
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8")), summary)

    def test_actual_full_45_cell_cpu_smoke_reports_equal_mean_deltas_and_preserves_sources(self):
        job = self._job()
        prepared_before = {str(path): file_sha256(path)
                           for path in self.prepared.rglob("*") if path.is_file()}
        installed_before = {name: file_sha256(path) for name, path in source_files().items()}
        validation_before = {str(path): file_sha256(path)
                             for path in (self.data_root / "val").rglob("*") if path.is_file()}
        with patch("torch.cuda.is_available", return_value=False), redirect_stdout(io.StringIO()):
            summary = score_frozen_validation(job)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["kind"], "smoke")
        self.assertEqual(summary["approval_status"], "engineering_only")
        self.assertEqual(summary["condition_count"], 45)
        self.assertTrue(summary["clean_matches_reference"])
        self.assertTrue(summary["corruption_scoring_run"])
        self.assertFalse(summary["test_evaluated"])
        self.assertFalse(summary["training_started"])
        output = job.path("output_dir")
        conditions = json.loads((output / "conditions.json").read_text(encoding="utf-8"))
        expected_ids = list(self.manifest["condition_counts"])
        self.assertEqual(set(conditions), set(expected_ids))
        clean_f1 = summary["clean_f1"]
        for condition_id, result in conditions.items():
            with self.subTest(condition=condition_id):
                self.assertEqual(result["samples"], 4)
                self.assertEqual(result["pixels"], 4 * 32 * 32)
                self.assertEqual(sum(result["counts"].values()), result["pixels"])
                self.assertEqual(result["clean_reference_f1"], clean_f1)
                self.assertEqual(result["clean_minus_f1"], clean_f1 - result["metrics"]["f1"])
                saved = json.loads((output / "conditions" / f"{condition_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, result)
        expected_mean = sum(conditions[key]["metrics"]["f1"] for key in expected_ids) / 45
        self.assertEqual(summary["corruption_mf1"], expected_mean)
        self.assertEqual(summary["clean_minus_corruption_mf1"], clean_f1 - expected_mean)
        for family, value in summary["family_mf1"].items():
            expected = [conditions[key]["metrics"]["f1"] for key in expected_ids
                        if conditions[key]["kind"] == family]
            self.assertEqual(len(expected), 5)
            self.assertEqual(value, sum(expected) / 5)
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8")), summary)
        environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["device"], "cpu")
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        for path, checksum in validation_before.items():
            self.assertEqual(provenance["input_file_sha256"][path], checksum)
        self.assertEqual(prepared_before, {path: file_sha256(Path(path)) for path in prepared_before})
        self.assertEqual(installed_before, {name: file_sha256(path) for name, path in source_files().items()})
        self.assertEqual(validation_before, {path: file_sha256(Path(path)) for path in validation_before})
        self.assertEqual(file_sha256(self.checkpoint_path), self.checkpoint_sha256)
        self.assertFalse((self.data_root / "train").exists())
        self.assertFalse((self.data_root / "test").exists())

    def test_wrapper_noop_normalization_matches_baseline_bitwise(self):
        bundle = self._bundle()
        dataset, _ = build_validation_dataset(bundle)
        instance = copy.deepcopy(bundle.instances[0])
        instance.update(kind="none", parameters={}, condition_id="engineering_none_l0", level=0,
                        injection_mask={"mode": "global"})
        wrapped = FrozenInstanceDataset(dataset, [instance])
        original = dataset[0]
        replayed = wrapped[0]
        self.assertEqual(wrapped.split, "val")
        for key in ("image_a", "image_b", "mask"):
            self.assertTrue(torch.equal(replayed[key], original[key]), key)
        self.assertEqual(replayed["sample_id"], original["sample_id"])
        self.assertEqual(replayed["instance_json"], canonical(instance))
        self.assertEqual(replayed["injection_support"].dtype, torch.bool)
        self.assertEqual(tuple(replayed["injection_support"].shape), (1, 32, 32))
        self.assertTrue(replayed["injection_support"].all())

    def test_wrapper_large_seed_and_heterogeneous_masks_collate_safely(self):
        bundle = self._bundle()
        dataset, _ = build_validation_dataset(bundle)
        ids = sample_ids(dataset)
        instances = [make_instance(bundle.protocol, ids[index], "noise", 3, stream="val",
                                   direction="neutral", target_date="A", spatial=spatial)
                     for index, spatial in enumerate(("global", "local"))]
        for instance in instances:
            instance["seed"] = 2**64 - 1  # Explicit custom wrapper-only engineering fixture.
        wrapper = FrozenInstanceDataset(dataset, instances)
        batch = next(iter(DataLoader(wrapper, batch_size=2, num_workers=0)))
        self.assertEqual(batch["image_a"].shape, (2, 3, 32, 32))
        self.assertEqual(batch["injection_support"].shape, (2, 1, 32, 32))
        self.assertEqual(batch["injection_support"].dtype, torch.bool)
        self.assertEqual([json.loads(item)["seed"] for item in batch["instance_json"]], [2**64 - 1] * 2)
        self.assertTrue(batch["injection_support"][0].all())
        self.assertFalse(batch["injection_support"][1].all())

    def test_wrapper_two_spawn_workers_match_single_process_uint64_mask_collation(self):
        protocol = load_protocol(self.protocol_path)
        dataset = LEVIRDataset(protocol.base, "val")
        ids = sample_ids(dataset)
        instances = [make_instance(protocol, sample_id, "noise", 3, stream="val",
                                   direction="neutral", target_date=("A", "B")[index % 2],
                                   spatial=("global", "local")[index % 2])
                     for index, sample_id in enumerate(ids)]
        for index, instance in enumerate(instances):
            instance["seed"] = 2**63 + index  # Deliberately above signed-int64 range.
        wrapper = FrozenInstanceDataset(dataset, instances)
        serial = list(DataLoader(wrapper, batch_size=2, num_workers=0,
                                 generator=torch.Generator().manual_seed(73)))
        parallel = list(DataLoader(wrapper, batch_size=2, num_workers=2,
                                   multiprocessing_context="spawn", worker_init_fn=seed_worker,
                                   generator=torch.Generator().manual_seed(73), timeout=30))
        self.assertEqual(len(parallel), 2)
        for reference, batch in zip(serial, parallel):
            self.assertEqual(set(batch), set(reference))
            for key, expected in reference.items():
                with self.subTest(field=key):
                    if isinstance(expected, torch.Tensor):
                        self.assertTrue(torch.equal(batch[key], expected))
                        self.assertEqual(batch[key].device.type, "cpu")
                    else:
                        self.assertEqual(batch[key], expected)
            self.assertTrue(all(json.loads(item)["seed"] >= 2**63 for item in batch["instance_json"]))
            self.assertEqual(batch["injection_support"].dtype, torch.bool)
            self.assertTrue(batch["injection_support"][0].all())
            self.assertFalse(batch["injection_support"][1].all())

    def test_wrapper_preserves_semantic_labels_other_date_and_exterior(self):
        bundle = self._bundle()
        dataset, _ = build_validation_dataset(bundle)
        sample_id = sample_ids(dataset)[0]
        before = dataset[0]
        source_file_hashes = {str(path): file_sha256(path) for path in (
            dataset.records[0].image_a, dataset.records[0].image_b, dataset.records[0].mask)}
        for target in ("A", "B"):
            instance = make_instance(bundle.protocol, sample_id, "brightness", 5, stream="val",
                                     direction="darken", target_date=target, spatial="local")
            output = FrozenInstanceDataset(dataset, [instance])[0]
            target_key, other_key = ("image_a", "image_b") if target == "A" else ("image_b", "image_a")
            self.assertTrue(torch.equal(output["mask"], before["mask"]))
            self.assertTrue(torch.equal(output[other_key], before[other_key]))
            support = output["injection_support"][0]
            self.assertTrue(torch.equal(output[target_key][:, ~support], before[target_key][:, ~support]))
            self.assertFalse(torch.equal(output[target_key], before[target_key]))
            expected = make_injection_mask((32, 32), instance["injection_mask"]) > 0
            np.testing.assert_array_equal(support.numpy(), expected)
        self.assertEqual(source_file_hashes, {path: file_sha256(Path(path)) for path in source_file_hashes})

    def test_wrapper_rejects_unknown_duplicate_empty_and_pretransformed_samples(self):
        bundle = self._bundle()
        dataset, _ = build_validation_dataset(bundle)
        instance = bundle.instances[0]
        invalid = [[], [instance, copy.deepcopy(instance)],
                   [dict(instance, sample_id="val/val_0.png@x64_y0")],
                   [dict(instance, sample_id="test/val_0.png@x0_y0")]]
        for instances in invalid:
            with self.subTest(instances=instances), self.assertRaises(ValueError):
                FrozenInstanceDataset(dataset, instances)
        dataset.pair_transform = lambda pair: pair
        with self.assertRaisesRegex(ValueError, "untransformed"):
            FrozenInstanceDataset(dataset, [instance])
        dataset.pair_transform = None
        dataset.split = "train"
        with self.assertRaises(ValueError):
            FrozenInstanceDataset(dataset, [instance])


if __name__ == "__main__":
    unittest.main()
