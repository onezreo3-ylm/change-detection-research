import copy
import itertools
import random
import unittest

from scripts.screen_spatial_action_feasibility import add, checked_pair, counts, metrics, optimal_whole_tile, pixel_bound


def fixture(y, a, b):
    def confusion(p):
        return dict(tn=sum(t == 0 and q == 0 for t, q in zip(y, p)),
            fp=sum(t == 0 and q == 1 for t, q in zip(y, p)),
            fn=sum(t == 1 and q == 0 for t, q in zip(y, p)),
            tp=sum(t == 1 and q == 1 for t, q in zip(y, p)))
    return dict(a0_counts=confusion(a), a1_counts=confusion(b), transitions_a0_to_a1=dict(
        fp_to_tn=sum(t == 0 and p == 1 and q == 0 for t, p, q in zip(y, a, b)),
        fn_to_tp=sum(t == 1 and p == 0 and q == 1 for t, p, q in zip(y, a, b)),
        tp_to_fn=sum(t == 1 and p == 1 and q == 0 for t, p, q in zip(y, a, b)),
        tn_to_fp=sum(t == 0 and p == 0 and q == 1 for t, p, q in zip(y, a, b))))


class SpatialActionBoundTests(unittest.TestCase):
    def test_pixel_formula_matches_exhaustive_all_pixel_action_choices(self):
        rng = random.Random(20261027)
        for _ in range(30):
            y, a, b = [[rng.randrange(2) for _ in range(8)] for _ in range(3)]
            r = fixture(y, a, b)
            brute = max(metrics(counts(fixture(y, [b[i] if s[i] else a[i] for i in range(8)], a)['a0_counts']))['f1']
                        for s in itertools.product((0, 1), repeat=8))
            self.assertEqual(metrics(pixel_bound(r))['f1'], brute)

    def test_whole_tile_fractional_solution_matches_exhaustive_pooled_f1(self):
        rng = random.Random(20261028)
        for _ in range(60):
            pairs = []
            for _ in range(7):
                y, a, b = [[rng.randrange(2) for _ in range(6)] for _ in range(3)]
                pairs.append(checked_pair(fixture(y, a, b)))
            brute = max(metrics(add(p[s] for p, s in zip(pairs, choice)))['f1']
                        for choice in itertools.product((0, 1), repeat=len(pairs)))
            self.assertEqual(metrics(optimal_whole_tile(pairs)['counts'])['f1'], brute)

    def test_all_negative_labels_keep_original_zero_f1_convention(self):
        r = fixture([0, 0], [1, 1], [0, 0])
        self.assertEqual(metrics(pixel_bound(r))['f1'], 0)
        self.assertEqual(metrics(optimal_whole_tile([checked_pair(r)])['counts'])['f1'], 0)

    def test_identical_actions_are_deterministic_and_select_a0_on_ties(self):
        r = fixture([0, 1, 1], [0, 1, 0], [0, 1, 0])
        result = optimal_whole_tile([checked_pair(r)] * 4)
        self.assertEqual(result['selected_a1'], 0)
        self.assertEqual(result, optimal_whole_tile([checked_pair(r)] * 4))

    def test_impossible_transitions_and_changed_support_are_rejected(self):
        row = fixture([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0])
        for key, change in [('transitions_a0_to_a1', 'fp_to_tn'), ('a1_counts', 'tn')]:
            r = copy.deepcopy(row); r[key][change] += 1
            with self.assertRaises(ValueError): checked_pair(r)
        with self.assertRaises(ValueError): checked_pair(row, pixels=5)

    def test_non_integer_or_negative_counts_are_never_coerced(self):
        for x in (-1, 1.1, True, '1'):
            with self.assertRaises(ValueError): counts(dict(tn=x, fp=0, fn=1, tp=1))

    def test_integer_solver_is_stable_for_counts_above_float_integer_precision(self):
        k = 10**20
        pairs = [((k, 1, 1, k), (k + 1, 0, 2, k - 1)), ((k, 3, 4, k), (k + 2, 1, 0, k + 4))]
        brute = max(metrics(add(p[s] for p, s in zip(pairs, choice)))['f1']
            for choice in itertools.product((0, 1), repeat=2))
        result = optimal_whole_tile(pairs)
        self.assertEqual(metrics(result['counts'])['f1'], brute)
        n, d = 2 * result['counts'][3], 2 * result['counts'][3] + result['counts'][1] + result['counts'][2]
        for choice in itertools.product((0, 1), repeat=2):
            other = add(p[s] for p, s in zip(pairs, choice))
            self.assertGreaterEqual(n * (2 * other[3] + other[1] + other[2]), 2 * other[3] * d)
        # This improvement is invisible after float division; integer decisions
        # must still retain it, rather than treating the outputs as an exact tie.
        pair = ((k, 10, 10, k), (k + 1, 9, 10, k))
        exact = optimal_whole_tile([pair])
        self.assertEqual(exact['counts'], pair[1]); self.assertEqual(exact['selected_a1'], 1)

    def test_empty_unmatched_actions_or_uncertified_iteration_cap_fail_closed(self):
        with self.assertRaises(ValueError): optimal_whole_tile([])
        with self.assertRaises(ValueError): optimal_whole_tile([((1, 0, 1, 0), (1, 0, 0, 0))])
        r = fixture([0, 1], [1, 0], [0, 1])
        with self.assertRaises(RuntimeError): optimal_whole_tile([checked_pair(r)], limit=0)


if __name__ == '__main__': unittest.main()
