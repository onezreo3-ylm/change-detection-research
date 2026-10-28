import unittest
from pcdc.config import protocol_signature
from scripts import five_view_baselines as f
from scripts import c5r5_runtime_identity as ident
from scripts import five_view_40ep_runtime as worker
import test_c5r5_40ep as original


class RuntimeIdentityTests(unittest.TestCase):
    def test_bf16_projection_carries_all_prior_engineering_and_40ep_margin(self):
        profile={'roles':{'c5':{'timings_seconds':[.2]*32,'budget_projection':{'engineering_seconds_debited':400}}}}
        result=ident.project(profile,'c5','bf16_training_fp32_validation')
        self.assertEqual(result['engineering_seconds_debited'],400)
        self.assertAlmostEqual(result['projected_total_role_seconds'],.2*110280*1.25+2400+400)
        self.assertTrue(result['within_frozen_budget'])

    def test_checkpoint_precision_is_explicit_and_cannot_be_forged_as_fp32(self):
        args=list(original.FortyEpochTests().fixture());cp,cfg,identity,plan=args
        cp['training_flavor']=ident.FLAVOR
        identity['optimization_mode']='bf16_training_fp32_validation'
        cp['training_signature']=f.digest(identity)
        with self.assertRaises(ValueError):worker.validate_checkpoint(*args)
        cfg.values['train']['amp']=True
        cp['protocol_signature']=protocol_signature(cfg.values)
        worker.validate_checkpoint(*args)
