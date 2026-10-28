import unittest

import torch

from scripts.screen_task_projection import load,null_projection,witnesses


class TaskProjectionScreenTests(unittest.TestCase):
    def test_full_linear_nullspace_preserves_outputs_but_not_every_feature(self):
        r=witnesses(load())['full_vector']
        self.assertTrue(r['nonzero_feature_residual']);self.assertTrue(r['all_linear_logits_unchanged'])

    def test_mean_protection_can_destroy_two_correct_pixel_decisions(self):
        r=witnesses(load())['mean_only']
        self.assertTrue(r['mean_linear_logit_unchanged'])
        self.assertEqual(r['before_counts'],[1,0,0,1]);self.assertEqual(r['after_counts'],[0,1,1,0])

    def test_predicted_positive_protection_keeps_false_positive_and_GT_mask_is_privileged(self):
        r=witnesses(load())['positive_mask']
        self.assertNotEqual(r['predicted_positive_mask'],r['truth_dependent_TP_mask'])
        self.assertEqual(r['predicted_positive_protected_counts'][1],1)
        self.assertEqual(r['GT_TP_protected_counts'][1],0)
        self.assertFalse(r['GT_mask_is_deployment_input'])

    def test_clean_teacher_and_first_order_protection_do_not_transfer_automatically(self):
        r=witnesses(load())
        self.assertTrue(r['teacher_shift']['clean_linear_change_zero'])
        self.assertEqual(r['teacher_shift']['observed_linear_change'],[1.])
        self.assertTrue(r['finite_step']['first_order_change_zero'])
        self.assertTrue(r['finite_step']['finite_step_changed_hard_prediction'])

    def test_invalid_numeric_inputs_and_dense_size_scope(self):
        j=torch.eye(2,dtype=torch.float64);r=torch.ones(2,dtype=torch.float64)
        for jj,rr,tol in [(j.float(),r,1e-12),(j,torch.ones(3,dtype=torch.float64),1e-12),
                          (j,r,float('nan')),(j,r,True)]:
            with self.assertRaises(ValueError):null_projection(jj,rr,rtol=tol)
        m=witnesses(load())['dense_size']
        self.assertEqual(m['explicit_dense_gram_bytes'],17179869184)
        self.assertTrue(m['does_not_rule_out_matrix_free_or_subset_methods'])
        self.assertFalse(m['actual_measured_GPU_memory'])

if __name__=='__main__':unittest.main()
