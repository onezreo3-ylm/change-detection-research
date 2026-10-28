"""Checks for the predeclared R1 paired-source estimator and selection gate."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_r1 import METRICS, SEEDS, paired_bootstrap, selection_gate


class R1AnalysisTests(unittest.TestCase):
    def test_bootstrap_uses_a_common_source_draw_and_is_reproducible(self) -> None:
        counts = np.zeros((2, 3, 2, 2, 4), dtype=np.int64)
        counts[..., 0] = 100
        counts[..., 3] = 10
        counts[..., 2] = 5
        counts[:, :, 1, :, 3] += np.array([1, 2])[:, None, None]
        counts[:, :, 1, :, 2] -= np.array([1, 2])[:, None, None]
        first = paired_bootstrap(counts, replicates=1000, seed=17)
        self.assertEqual(first, paired_bootstrap(counts, replicates=1000, seed=17))
        self.assertGreater(first[0], 0)
        self.assertLessEqual(first[0], first[1])
        self.assertLessEqual(first[1], first[2])

    def test_gate_requires_all_five_conditions_and_stability(self) -> None:
        by_seed = {}
        for seed in SEEDS:
            b1 = {key: 0.0 for key in METRICS}
            b1.update(clean_f1=0.91, corruption_mf1=0.90, pooled_recall=0.89)
            b2 = dict(b1, corruption_mf1=0.901)
            by_seed[seed] = {"b1": b1, "b2": b2}
        self.assertEqual(selection_gate(by_seed, [0.0001, 0.001, 0.002])["decision"],
                         "select_b2")
        by_seed[44]["b2"]["clean_f1"] = 0.905
        self.assertEqual(selection_gate(by_seed, [0.0001, 0.001, 0.002])["decision"],
                         "retain_b1")
        by_seed[44]["b2"]["clean_f1"] = 0.91
        by_seed[44]["b1"]["corruption_mf1"] = 0.906
        self.assertEqual(selection_gate(by_seed, [0.0001, 0.001, 0.002])["decision"],
                         "stop_for_instability_review")

    def test_bootstrap_rejects_unmatched_seed_dimension(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap(np.zeros((2, 2, 2, 45, 4), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
