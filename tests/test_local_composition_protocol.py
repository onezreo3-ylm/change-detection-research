import itertools
import unittest

from pcdc.local_composition_protocol import f1_fraction, optimal_finite_actions, select_original_batches


class CompositionProtocolTests(unittest.TestCase):
    def select(self, ids=None, **kw):
        default=dict(batch_size=4,count=3,calibration_first=1,salt='synthetic-fixed',minimum_fit=1,minimum_calibration=1)
        default.update(kw)
        if ids is None: ids=[f'train/s{i}.png@x{x}_y0' for i in range(8) for x in range(4)]
        return select_original_batches(ids,[f's{i}.png' for i in range(6)],['s6.png','s7.png'],**default)

    def test_selection_preserves_complete_original_order_without_outcomes(self):
        ids=[f'train/s{i}.png@x{x}_y0' for i in range(8) for x in range(4)]
        a=self.select(ids);b=self.select(ids)
        self.assertEqual(a,b)
        self.assertFalse(a['selection_uses_outcomes'])
        for r in a['batches']:
            start=r['original_start']
            self.assertEqual(start%4,0)
            self.assertEqual(r['sample_ids'],ids[start:start+4])

    def test_failed_coverage_no_resampling_or_incomplete_batch_substitution(self):
        with self.assertRaises(ValueError): self.select(minimum_calibration=3)
        with self.assertRaises(ValueError): self.select(calibration_first=3)
        with self.assertRaises(ValueError): self.select(count=9)

    def test_wrong_split_duplicates_and_partition_overlap_are_rejected(self):
        with self.assertRaises(ValueError): self.select(['val/s0.png@x0_y0']*4)
        with self.assertRaises(ValueError): self.select(['train/s0.png@x0_y0']*4)
        with self.assertRaises(ValueError): select_original_batches(['train/a@x0_y0'],['a'],['a'],batch_size=1,count=1,calibration_first=0,salt='s',minimum_fit=0,minimum_calibration=0)

    def test_exact_fractional_solution_matches_exhaustive_multiaction_product(self):
        cases=[[[8,2,3,7],[9,1,4,6],[7,3,1,9]], [[6,4,2,8],[8,2,3,7]], [[10,0,1,9],[9,1,0,10]]]
        def pooled(choice): return [sum(case[i][k] for case,i in zip(cases,choice)) for k in range(4)]
        brute=max(f1_fraction(pooled(c)) for c in itertools.product(*[range(len(x)) for x in cases]))
        result=optimal_finite_actions(cases)
        self.assertEqual(result['exact_f1'],[brute.numerator,brute.denominator])
        self.assertEqual(result['counts'],pooled(result['choices']))

    def test_empty_positive_case_and_deterministic_tie(self):
        result=optimal_finite_actions([[[10,0,0,0],[9,1,0,0]]])
        self.assertEqual(result['choices'],[0]);self.assertEqual(result['f1'],0)
        result=optimal_finite_actions([[[8,2,1,9],[8,2,1,9]]])
        self.assertEqual(result['choices'],[0])

    def test_label_support_changes_invalid_counts_and_empty_cases_fail(self):
        for cases in ([],[[]],[[[1,0,0,0],[0,0,0,1]]],[[[0,0,-1,1]]],[[[0.,1,0,0]]]):
            with self.assertRaises(ValueError): optimal_finite_actions(cases)


if __name__=='__main__': unittest.main()
