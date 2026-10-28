"""Fail-closed identity, selection, gate, execution and independent residual audits."""
import copy
from pathlib import Path
import unittest

import torch

from pcdc.config import Config
from pcdc.frozen_validation import load_evaluation_config
from scripts import c5r5_fixed_evaluation as evaluation
from scripts import c5r5_fixed_residual as residual
from scripts.coupling_diagnostic import four_cell_diagnostic
from test_coupling_p0_runner import manifest_fixture


class FixedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,_,cls.paths,cls.settings,_,cls.runs=evaluation.load(
            evaluation.ROOT/'configs/valais_c5r5_fixed_eval_v1_20261022.yaml')
        cls.benchmark=load_evaluation_config(cls.paths['benchmark_config'])
        from pcdc.corruption_protocol import load_protocol
        cls.fp32=load_protocol(cls.benchmark.path('protocol_config')).base

    def selection(self):
        plan=[{'epoch':e,'source_batch_sha256':str(e)} for e in range(1,41)]
        history=[{'epoch':e,'val':{'metrics':{'f1':.4}}} for e in range(1,41)]
        last=dict(epoch=40,best_epoch=1,best_val_f1=.4,history=history,
            training_flavor=evaluation.FLAVOR,file_sha256={'train':{},'val':{}})
        best=copy.deepcopy(last);best.update(epoch=1,history=history[:1])
        return last,best,copy.deepcopy(plan),plan

    def scores(self):
        base=dict(clean_f1=.5,clean_recall=.5,mf1=.5,corruption_recall=.5,
            clean_empty_fpr=.001,corruption_empty_fpr=.001)
        return {'b0e':base,'c5':dict(base),'r5':dict(base,mf1=.505)}

    def test_explicit_real_precision_horizon_mapping_preserves_both_original_configs(self):
        before=copy.deepcopy(self.runs['c5'].base.values)
        fp32=copy.deepcopy(self.fp32.values)
        mapping=evaluation.inference_mapping(self.runs['c5'].base,self.fp32)
        self.assertFalse(mapping['checkpoint_fields_modified'])
        self.assertNotEqual(mapping['training_protocol_signature'],mapping['fp32_evaluation_protocol_signature'])
        self.assertEqual(self.runs['c5'].base.values,before);self.assertEqual(self.fp32.values,fp32)

    def test_undeclared_model_data_threshold_or_precision_changes_cannot_use_adapter(self):
        base=self.runs['c5'].base
        for section,key,value in [('evaluation','threshold',.4),('model','decoder_channels',99),
                ('data','mean',[0.,0.,0.]),('train','amp',False),('runtime','device','cpu')]:
            values=copy.deepcopy(base.values);values[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                evaluation.inference_mapping(Config(values,base.source),self.fp32)

    def test_first_clean_tie_and_exact_history_prefix_are_required(self):
        args=self.selection();evaluation.validate_selection(*args)
        changed=list(self.selection());changed[0]['best_epoch']=2
        changed[1].update(epoch=2,best_epoch=2,history=changed[0]['history'][:2])
        with self.assertRaises(ValueError):evaluation.validate_selection(*changed)

    def test_wrong_flavor_partial_epochs_test_scope_or_actual_stream_are_rejected(self):
        for key,value in [('epoch',39),('training_flavor','matched_robust_baseline_v1'),
                ('file_sha256',{'train':{},'val':{},'test':{}})]:
            args=list(self.selection());args[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):evaluation.validate_selection(*args)
        args=list(self.selection());args[2][7]['source_batch_sha256']='changed'
        with self.assertRaises(ValueError):evaluation.validate_selection(*args)

    def test_frozen_six_metric_gate_passes_only_at_declared_bounds(self):
        scores=self.scores();result=evaluation.gate_comparison(scores,self.settings['gates'])
        self.assertEqual(result['passed'],{'c5_vs_b0e':True,'r5_vs_c5':True})
        scores['r5']['mf1']=.50499
        self.assertFalse(evaluation.gate_comparison(scores,self.settings['gates'])['checks']['r5_vs_c5']['mf1'])

    def test_large_corruption_gain_cannot_override_clean_failure(self):
        scores=self.scores();scores['r5'].update(clean_f1=.45656,mf1=.8)
        result=evaluation.gate_comparison(scores,self.settings['gates'])
        self.assertTrue(result['checks']['r5_vs_c5']['mf1'])
        self.assertFalse(result['passed']['r5_vs_c5']);self.assertFalse(result['p2_training_authorized'])

    def test_clean_and_corruption_empty_tile_fpr_each_protect_both_comparisons(self):
        for role,comparison in [('c5','c5_vs_b0e'),('r5','r5_vs_c5')]:
            for key in ('clean_empty_fpr','corruption_empty_fpr'):
                scores=self.scores();scores[role][key]+=.00021
                with self.subTest(role=role,key=key):
                    self.assertFalse(evaluation.gate_comparison(scores,self.settings['gates'])['checks'][comparison][key])

    def test_incomplete_or_nonfinite_saved_comparison_cannot_be_declared_passed(self):
        for value in (float('nan'),float('inf'),-1.,1.001):
            scores=self.scores();scores['r5']['clean_f1']=value
            with self.subTest(value=value),self.assertRaises(ValueError):evaluation.gate_comparison(scores,self.settings['gates'])
        scores=self.scores();del scores['r5']['clean_empty_fpr']
        with self.assertRaises(ValueError):evaluation.gate_comparison(scores,self.settings['gates'])

    def test_real_smoke_requires_exact_coverage_frozen_models_scope_temperature_and_budget(self):
        report=dict(status='passed_train_only_c5r5_cuda_smoke',smoke_only=True,research_result=False,
            predictions=102,main_predictions=98,records=0,training=False,gradients_enabled=False,amp=False,
            test_accessed=False,val_accessed=False,elapsed_seconds=20.,maximum_observed_temperature_c=60,
            models={r:{'model_state_unchanged':True} for r in evaluation.ROLES},
            self_checks=[dict(model=r,repeat_bitwise_equal=True,temporal_swap_allclose=True,max_swap_abs_error=0.)
                for r in evaluation.ROLES])
        evaluation.validate_smoke(report,[50,60])
        for key,value in [('predictions',101),('test_accessed',True),('amp',True),('elapsed_seconds',601)]:
            with self.subTest(key=key),self.assertRaises(ValueError):evaluation.validate_smoke(dict(report,**{key:value}),[50,60])
        with self.assertRaises(ValueError):evaluation.validate_smoke(report,[50,85])

    def test_new_roles_preserve_independent_original_four_cell_cartesian_audit(self):
        manifest=manifest_fixture(Path('synthetic-c5r5-audit').resolve());rows=[];cached={}
        logits=torch.full((1,1,256,256),-1.)
        for foreground in (0,1):
            target=torch.zeros_like(logits);target.flatten()[0]=foreground
            cached[foreground]=four_cell_diagnostic(logits,dict.fromkeys(('uu','uv','vu','vv'),logits),target)
        marginal=dict(matched=True,a_instances_identical=True,b_instances_identical=True,fully_degenerate=True)
        for role in evaluation.ROLES:
            for sample in manifest['samples']:
                for condition in sample['conditions']:
                    rows.append(dict(model=role,source_id=sample['id'],stratum=sample['stratum'],
                        foreground_pixels=sample['foreground_pixels'],family=condition['family'],level=condition['level'],
                        marginal_check=marginal,diagnostic=cached[sample['foreground_pixels']]))
        report=residual.summarize(rows)
        self.assertEqual(residual.audit_records(manifest,rows,report)['records'],1536)
        with self.assertRaises(ValueError):residual.audit_records(manifest,rows[:-1],report)
        changed=copy.deepcopy(report);changed['aggregates']['r5']['all']['groups']['D']['counts']['fp']+=1
        with self.assertRaises(ValueError):residual.audit_records(manifest,rows,changed)


if __name__=='__main__':unittest.main()
