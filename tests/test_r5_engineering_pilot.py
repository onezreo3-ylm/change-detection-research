"""The API correction must not grant a second real training step."""
import unittest
from unittest.mock import patch

from scripts.valais_r5_engineering_pilot import check_unused_budget, cuda_index


class R5StartupCorrectionTests(unittest.TestCase):
    def fixture(self):
        return {'status':'incomplete','error':'ValueError: Expected a torch.device with a specified index or an integer, but got:cuda',
            'optimizer_steps':0,'pair_views':0,'initial_model_tensor_sha256':'sha',
            'test_accessed':False,'new_val_inference':False,'elapsed_seconds':.969}

    def test_explicit_integer_index_is_used_with_implicit_or_explicit_cuda_device(self):
        with patch('torch.cuda.current_device',return_value=0):
            self.assertEqual(cuda_index('cuda'),0)
        self.assertEqual(cuda_index('cuda:0'),0)
        with self.assertRaises(ValueError): cuda_index('cpu')

    def test_exact_pre_forward_failure_leaves_only_original_remaining_budget(self):
        self.assertAlmostEqual(check_unused_budget(self.fixture(),'sha'),599.031)

    def test_real_forward_step_numerical_or_oom_failure_cannot_be_retried(self):
        for key,value in [('optimizer_steps',1),('pair_views',20),('initial_model_tensor_sha256','other'),
            ('status','completed'),('error','CUDA out of memory'),('error','FloatingPointError'),
            ('elapsed_seconds',601),('test_accessed',True)]:
            row=self.fixture(); row[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): check_unused_budget(row,'sha')


if __name__=='__main__': unittest.main()
