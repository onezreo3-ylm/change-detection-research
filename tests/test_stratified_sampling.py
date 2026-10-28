"""Sampling/resume engineering checks; all source names and tensors are synthetic."""
from __future__ import annotations

import copy
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from pcdc.config import protocol_signature
from pcdc.robust_baselines import _restore_rng, _rng_state
from scripts.stratified_sampling import ObservedLoader, StratifiedBatchSampler, digest
from scripts.train_stratified_robust import FLAVOR, ROOT, signature, validate_resume


class StratifiedTests(unittest.TestCase):
    def setUp(self):
        proposal = yaml.safe_load((ROOT / "configs/valais_b0es_candidate_proposal_20261004.yaml").read_text(encoding="utf-8"))
        self.settings = proposal["sampling"]
        self.names = [f"{i:05d}.tif" for i in range(11025)]
        self.positive = self.names[:1046]

    def sampler(self, epoch=1):
        return StratifiedBatchSampler(self.names, self.positive, self.settings, epoch=epoch)

    def test_exact_full_batches_and_positive_tail(self):
        plan = self.sampler().plan()
        self.assertEqual(len(plan.batches), 2757)
        for row in plan.batches[:-1]:
            self.assertEqual(len(row), 4)
            self.assertEqual(sum(i < 1046 for i in row), 1)
            self.assertEqual(len(set(row)), 4)
        self.assertEqual(len(plan.batches[-1]), 1)
        self.assertLess(plan.batches[-1][0], 1046)

    def test_all_positive_sources_repeat_two_or_three_times(self):
        from collections import Counter
        counts = Counter(i for row in self.sampler().plan().batches for i in row if i < 1046)
        self.assertEqual(len(counts), 1046)
        self.assertEqual(set(counts.values()), {2, 3})
        self.assertEqual(sum(counts.values()), 2757)

    def test_empty_draws_have_no_replacement(self):
        values = [i for row in self.sampler().plan().batches for i in row if i >= 1046]
        self.assertEqual(len(values), 8268)
        self.assertEqual(len(set(values)), 8268)

    def test_epoch_repeatability_and_change(self):
        first, other = self.sampler().plan(), self.sampler(2).plan()
        self.assertEqual(first, self.sampler().plan())
        self.assertNotEqual(first.summary["source_batch_sha256"], other.summary["source_batch_sha256"])

    def test_sampling_does_not_change_runtime_rngs(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        self.sampler().plan()
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_rebuilt_sampler_matches_resumed_epoch(self):
        current = self.sampler()
        current.set_epoch(37)
        resumed = self.sampler(37)
        self.assertEqual(list(current), list(resumed))

    def test_partition_rejects_unknown_duplicate_and_wrong_counts(self):
        for positives in (self.positive + ["unknown.tif"], self.positive + [self.positive[0]], self.positive[:-1]):
            with self.subTest(count=len(positives)), self.assertRaises(ValueError):
                StratifiedBatchSampler(self.names, positives, self.settings)

    def test_epoch_rejects_bool_zero_negative(self):
        for epoch in (True, 0, -1):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                self.sampler(epoch)

    def batches(self, sampler):
        return [{"sample_id": [f"train/{self.names[i]}@x0_y0" for i in row]} for row in sampler.plan().batches]

    def observed(self, rows, sampler):
        class Loader:
            generator = torch.Generator().manual_seed(42)
            def __iter__(self):
                return iter(rows)
            def __len__(self):
                return len(rows)
        return ObservedLoader(Loader(), sampler)

    def test_actual_stream_matches_plan(self):
        sampler = self.sampler()
        observed = self.observed(self.batches(sampler), sampler)
        self.assertEqual(len(list(observed)), 2757)
        self.assertEqual(observed.observed, dict(sampler.plan().summary, actual_source_stream_verified=True))

    def test_actual_stream_rejects_changed_order_missing_and_extra(self):
        sampler = self.sampler()
        original = self.batches(sampler)
        changed = copy.deepcopy(original)
        changed[0]["sample_id"].reverse()
        for rows in (changed, original[:-1], original + [original[-1]]):
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                list(self.observed(rows, sampler))

    def test_rng_checkpoint_restores_global_and_loader_streams(self):
        loader = SimpleNamespace(generator=torch.Generator().manual_seed(42))
        before = _rng_state(loader)
        expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=loader.generator))
        _restore_rng(before, loader)
        actual = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=loader.generator))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))
        _restore_rng(before, loader)

    def resume_fixture(self):
        base = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text(encoding="utf-8"))
        config = SimpleNamespace(fingerprint="parent", base=SimpleNamespace(values=base))
        identity, sources, new_sources = {"proposal": "fixed"}, {"core": "fixed"}, {"trainer": "fixed"}
        checkpoint = {
            "training_flavor": FLAVOR, "training_signature": signature(config, identity),
            "sampling_identity": identity, "source_sha256": sources,
            "stratified_trainer_source_sha256": new_sources, "audit_sha256": "data",
            "initial_model_tensor_sha256": "initial", "protocol_signature": protocol_signature(base),
            "epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
        }
        return checkpoint, config, identity, sources, new_sources

    def test_resume_accepts_exact_identity(self):
        checkpoint, config, identity, sources, new_sources = self.resume_fixture()
        validate_resume(checkpoint, config, identity, sources, new_sources, "data", "initial")

    def test_resume_rejects_changed_protocol_sources_initialization(self):
        checkpoint, config, identity, sources, new_sources = self.resume_fixture()
        for key in ("training_flavor", "training_signature", "sampling_identity", "source_sha256",
                    "stratified_trainer_source_sha256", "audit_sha256", "initial_model_tensor_sha256"):
            changed = copy.deepcopy(checkpoint)
            changed[key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(changed, config, identity, sources, new_sources, "data", "initial")

    def test_resume_rejects_missing_epoch(self):
        checkpoint, config, identity, sources, new_sources = self.resume_fixture()
        checkpoint["history"] = [{"epoch": 2}]
        with self.assertRaises(ValueError):
            validate_resume(checkpoint, config, identity, sources, new_sources, "data", "initial")


if __name__ == "__main__":
    unittest.main()
