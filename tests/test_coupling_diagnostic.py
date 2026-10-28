"""Synthetic fixtures are engineering checks, never research results."""
import unittest
import numpy as np
import torch

from scripts.coupling_diagnostic import (area_stratum, select_train_samples,
    four_cell_diagnostic, metrics_from_counts, pool_counts, image_sha256,
    validate_marginal_hashes)


class CouplingDiagnosticTests(unittest.TestCase):
    def records(self):
        return [{"id": f"train/{a}_{i}", "split": "train", "foreground_pixels": a}
                for a in (0, 10, 100, 500, 2000, 5000) for i in range(60)]

    def test_selection_order_independent_exact_quotas(self):
        rows = self.records()
        a = select_train_samples(rows, seed=42)
        b = select_train_samples(reversed(rows), seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a["samples"]), 64)
        self.assertEqual(a["strata"], dict(empty=48, positive_1=4, positive_2=3,
                                        positive_3=3, positive_4=3, positive_5=3))
        self.assertNotEqual(a["selection_sha256"], select_train_samples(rows, seed=43)["selection_sha256"])

    def test_shortfall_fills_positives_only(self):
        rows = [r for r in self.records() if r["foreground_pixels"] != 10]
        result = select_train_samples(rows, seed=42)
        self.assertEqual(result["initial_deficits"]["positive_1"], 4)
        self.assertEqual(len(result["fill_ids"]), 4)
        self.assertEqual(result["strata"]["empty"], 48)

    def test_selection_rejects_leaks_duplicates_shortage(self):
        for rows in ([dict(id="val/x", split="val", foreground_pixels=1)],
                     self.records() + [self.records()[0]], self.records()[:3]):
            with self.assertRaises(ValueError):
                select_train_samples(rows, seed=42)

    def test_area_boundaries(self):
        self.assertEqual([area_stratum(n) for n in (0, 64, 65, 256, 257, 4096, 4097)],
                         ["empty", "positive_1", "positive_2", "positive_2", "positive_3", "positive_4", "positive_5"])
        with self.assertRaises(ValueError):
            area_stratum(1, [2, 1])

    def test_counts_transitions_and_pooled_metrics(self):
        target = torch.tensor([1., 1., 0., 0.])
        clean = torch.tensor([2., -2., -2., 2.])
        changed = -clean
        result = four_cell_diagnostic(clean, dict(uu=clean, uv=changed, vu=changed, vv=clean), target)
        self.assertEqual(result["cells"]["uv"]["transitions"], dict(
            tp_to_fn=1, fn_to_tp=1, tn_to_fp=1, fp_to_tn=1,
            clean_tp=1, clean_fn=1, clean_tn=1, clean_fp=1))
        self.assertEqual(result["groups"]["D"]["counts"], dict(tn=2, fp=2, fn=2, tp=2))
        self.assertEqual(result["groups"]["O"]["transitions"]["clean_tp"], 2)
        self.assertEqual(result["logit_effects"]["i"]["all"]["mean_abs"], 2)
        c = pool_counts([dict(tn=90, fp=0, fn=9, tp=1), dict(tn=0, fp=0, fn=0, tp=1)])
        self.assertAlmostEqual(metrics_from_counts(c)["f1"], 4/13)

    def test_additive_response_zero_interaction_not_zero_main_effect(self):
        target = torch.tensor([1., 0.])
        values = {k: torch.full((2,), v) for k, v in dict(uu=0., uv=4., vu=2., vv=6.).items()}
        result = four_cell_diagnostic(values["uu"], values, target)
        self.assertEqual(result["logit_effects"]["i"]["all"]["rms"], 0)
        self.assertEqual(result["logit_effects"]["a"]["all"]["mean"], -1)
        self.assertEqual(result["logit_effects"]["b"]["all"]["mean"], -2)

    def test_same_instance_degenerate_and_zero_denominator(self):
        z, y = torch.full((4,), -2.), torch.zeros(4)
        result = four_cell_diagnostic(z, dict.fromkeys(("uu", "uv", "vu", "vv"), z), y)
        self.assertEqual(result["groups"]["D"]["metrics"]["f1"], 0)
        self.assertEqual(result["groups"]["D"]["metrics"]["oa"], 1)
        self.assertEqual(result["logit_effects"]["i"]["foreground"]["pixels"], 0)
        self.assertEqual(result["logit_effects"]["i"]["all"]["rms"], 0)

    def test_rejects_nonfinite_and_shape_mismatch(self):
        z, y = torch.ones(2), torch.zeros(2)
        for bad in (torch.tensor([float("nan"), 0]), torch.ones(3)):
            with self.assertRaises(ValueError):
                four_cell_diagnostic(z, dict(uu=bad, uv=z, vu=z, vv=z), y)

    def test_marginal_hashes_and_duplicate_instances(self):
        hashes = [image_sha256(np.full((2, 3, 3), v, dtype=np.uint8)) for v in range(4)]
        a, b, c, d = hashes
        cells = dict(uu=[a,c], uv=[a,d], vu=[b,c], vv=[b,d])
        self.assertTrue(validate_marginal_hashes(cells)["matched"])
        self.assertFalse(validate_marginal_hashes(cells)["fully_degenerate"])
        cells["uv"] = [b, d]
        with self.assertRaises(ValueError):
            validate_marginal_hashes(cells)
        self.assertTrue(validate_marginal_hashes(dict.fromkeys(("uu", "uv", "vu", "vv"), [a,c]))["fully_degenerate"])


if __name__ == "__main__":
    unittest.main()
