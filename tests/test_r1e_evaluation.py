"""Fail-closed checks for the separately frozen R1E seed-compatibility gate."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from pcdc.robust_baselines import load_robust_config
from scripts.r1e_evaluation import check_declared_replication_delta


ROOT = Path(__file__).resolve().parents[1]


def _pair(mode: str = "b1", seed: int = 43):
    training = load_robust_config(
        ROOT / "configs" / f"r1_{mode}_seed{seed}.yaml")
    reference = load_robust_config(
        ROOT / "configs" / f"robust_{mode}_seed42.yaml")
    return training, reference


def _mutable(training):
    base = SimpleNamespace(values=copy.deepcopy(dict(training.base.values)),
                           run_test_after_training=False)
    protocol = SimpleNamespace(values=copy.deepcopy(dict(training.protocol.values)),
                               path=training.protocol.path)
    return SimpleNamespace(mode=training.mode, base=base, protocol=protocol,
                           values=copy.deepcopy(dict(training.values)))


class R1EReplicationDeltaTests(unittest.TestCase):
    def test_all_declared_seed_method_pairs_pass(self):
        for seed, stream in ((43, 310945), (44, 310946)):
            for mode in ("b1", "b2"):
                with self.subTest(seed=seed, mode=mode):
                    check_declared_replication_delta(
                        *_pair(mode, seed), seed=seed, train_stream_seed=stream)

    def test_wrong_train_seed_or_stream_fails(self):
        training, reference = _pair()
        for seed, stream in ((44, 310945), (43, 310946)):
            with self.subTest(seed=seed, stream=stream):
                with self.assertRaisesRegex(ValueError, "seed or split"):
                    check_declared_replication_delta(
                        training, reference, seed=seed, train_stream_seed=stream)

    def test_decoder_change_fails(self):
        training, reference = _pair()
        modified = _mutable(training)
        modified.base.values["model"]["decoder_channels"] = 128
        with self.assertRaisesRegex(ValueError, "base training/inference"):
            check_declared_replication_delta(
                modified, reference, seed=43, train_stream_seed=310945)

    def test_reserved_validation_seed_change_fails(self):
        training, reference = _pair()
        modified = _mutable(training)
        modified.protocol.values["seeds"]["val"] = 99
        with self.assertRaisesRegex(ValueError, "seed or split"):
            check_declared_replication_delta(
                modified, reference, seed=43, train_stream_seed=310945)

    def test_corruption_severity_change_fails(self):
        training, reference = _pair()
        modified = _mutable(training)
        modified.protocol.values["corruptions"]["blur"]["neutral"][0]["radius"] = 0.4
        with self.assertRaisesRegex(ValueError, "corruption operator"):
            check_declared_replication_delta(
                modified, reference, seed=43, train_stream_seed=310945)

    def test_consistency_objective_change_fails(self):
        training, reference = _pair("b2")
        modified = _mutable(training)
        modified.values["consistency"]["weight"] = 2.0
        with self.assertRaisesRegex(ValueError, "objective or exposure"):
            check_declared_replication_delta(
                modified, reference, seed=43, train_stream_seed=310945)

    def test_test_access_change_fails(self):
        training, reference = _pair()
        modified = _mutable(training)
        modified.base.run_test_after_training = True
        with self.assertRaisesRegex(ValueError, "seed or split"):
            check_declared_replication_delta(
                modified, reference, seed=43, train_stream_seed=310945)


if __name__ == "__main__":
    unittest.main()
