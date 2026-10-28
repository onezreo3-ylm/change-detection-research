"""Synthetic fail-closed checks for the explicit P1 flavor and saved-count audits."""
import copy
from pathlib import Path
import unittest

import torch
import yaml

from pcdc.evaluate_corruptions import counts_report
from scripts.coupling_diagnostic import four_cell_diagnostic
from scripts.valais_p1_posttraining import FLAVOR, ROOT, gate_comparison, measure_reports, validate_history
from scripts.valais_p1_residual import audit_records, summarize, validate_execution
from test_coupling_p0_runner import manifest_fixture


class P1IdentityTests(unittest.TestCase):
    def fixture(self):
        planned = [{"epoch": epoch, "source_batch_sha256": str(epoch)} for epoch in range(1,101)]
        actual = [dict(row, actual_source_stream_verified=True) for row in planned]
        history = [{"epoch": epoch, "train": {"sampling": actual[epoch-1], "optimizer_steps":2757,
            "amp_skipped_steps":0,"pair_views":22050},"val":{"metrics":{"f1":.4}}} for epoch in range(1,101)]
        last = {"epoch":100,"history":history,"best_epoch":1,"best_val_f1":.4,
            "file_sha256":{"train":{},"val":{}},"training_flavor":FLAVOR,
            "model":{"w":torch.ones(1)},"optimizer":{"state":{}}}
        best = copy.deepcopy(last)
        best.update(epoch=1,history=copy.deepcopy(history[:1]))
        return last,best,actual,planned

    def test_exact_completed_history_first_tie_and_original_flavor_remain_unchanged(self):
        args=self.fixture()
        validate_history(*args)
        self.assertEqual(args[0]['training_flavor'],FLAVOR)

    def test_wrong_flavor_test_scope_or_incomplete_budget_is_rejected(self):
        for key,value in [('training_flavor','matched_robust_baseline_v1'),('file_sha256',{'train':{},'val':{},'test':{}}),('epoch',99)]:
            args=list(self.fixture()); args[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_history(*args)

    def test_later_tie_or_wrong_actual_source_stream_is_rejected(self):
        args=list(self.fixture()); args[0]['best_epoch']=2
        args[1].update(epoch=2,best_epoch=2,history=copy.deepcopy(args[0]['history'][:2]))
        with self.assertRaises(ValueError): validate_history(*args)
        args=list(self.fixture()); args[2][3]['source_batch_sha256']='changed'
        with self.assertRaises(ValueError): validate_history(*args)

    def test_skipped_update_or_nonfinite_completed_state_is_rejected(self):
        args=list(self.fixture()); args[0]['history'][9]['train']['amp_skipped_steps']=1
        with self.assertRaises(ValueError): validate_history(*args)
        args=list(self.fixture()); args[1]['model']['w'][0]=float('nan')
        with self.assertRaises(FloatingPointError): validate_history(*args)


class P1SavedMetricsTests(unittest.TestCase):
    def report(self,empty_fp):
        empty=counts_report([100-empty_fp,empty_fp,0,0])
        positive=counts_report([80,5,5,10])
        pooled=counts_report([180-empty_fp,empty_fp+5,5,10])
        return dict(pooled,by_source={'val/empty':empty,'val/positive':positive})

    def test_empty_source_fpr_is_separate_from_overall_background_fpr(self):
        clean=self.report(10); conditions={'brightness_l1':self.report(20)}
        values=measure_reports(clean,conditions)
        self.assertEqual(values['clean_empty_fpr'],.1)
        self.assertEqual(values['corruption_empty_fpr'],.2)
        self.assertNotEqual(values['clean_empty_fpr'],clean['metrics']['background_fpr'])

    def test_aggregate_counts_or_source_scope_tampering_is_rejected(self):
        clean=self.report(10); clean['counts']['fp']+=1
        with self.assertRaises(ValueError): measure_reports(clean,{'x':self.report(20)})
        clean=self.report(10); clean['by_source']['test/leak']=clean['by_source'].pop('val/empty')
        with self.assertRaises(ValueError): measure_reports(clean,{'x':self.report(20)})

    def test_predeclared_gate_preserves_both_empty_fpr_limits(self):
        gates=yaml.safe_load((ROOT/'configs/valais_p1_posttraining_v1_20261019.yaml').read_text(encoding='utf-8'))['gates']
        base=dict(clean_f1=.5,clean_recall=.5,mf1=.5,corruption_recall=.5,clean_empty_fpr=.001,corruption_empty_fpr=.001)
        cs=dict(base,clean_f1=.52,clean_recall=.53)
        augmented=dict(cs,mf1=.51)
        scores={'b0e':base,'cs':cs,'as':augmented}
        self.assertEqual(gate_comparison(scores,gates)['passed'],{'cs_vs_b0e':True,'as_vs_cs':True})
        for key in ('clean_empty_fpr','corruption_empty_fpr'):
            changed=copy.deepcopy(scores); changed['as'][key]+=.0003
            with self.subTest(key=key): self.assertFalse(gate_comparison(changed,gates)['passed']['as_vs_cs'])
        changed=copy.deepcopy(scores); changed['as']['clean_f1']=cs['clean_f1']-.012
        self.assertFalse(gate_comparison(changed,gates)['passed']['as_vs_cs'])

    def test_nonfinite_comparison_metrics_are_rejected(self):
        with self.assertRaises(ValueError): gate_comparison({'cs':{'f1':float('nan')}},{})


class P1ResidualAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=manifest_fixture(Path('synthetic-p1-fixture').resolve())
        cls.rows=[]
        logits=torch.full((1,1,256,256),-1.)
        cached={}
        for foreground in (0,1):
            target=torch.zeros_like(logits); target.flatten()[0]=foreground
            cached[foreground]=four_cell_diagnostic(logits,dict.fromkeys(('uu','uv','vu','vv'),logits),target)
        marginal=dict(matched=True,a_instances_identical=True,b_instances_identical=True,fully_degenerate=True)
        for role in ('cs','as'):
            for sample in cls.manifest['samples']:
                for condition in sample['conditions']:
                    cls.rows.append(dict(model=role,source_id=sample['id'],stratum=sample['stratum'],
                        foreground_pixels=sample['foreground_pixels'],family=condition['family'],level=condition['level'],
                        marginal_check=marginal,diagnostic=cached[sample['foreground_pixels']]))
        cls.report=summarize(cls.rows)

    def test_both_frozen_models_and_all_original_conditions_are_audited(self):
        self.assertEqual(audit_records(self.manifest,self.rows,self.report)['records'],1536)
        self.assertFalse(self.report['gate']['passed'])

    def test_missing_or_duplicate_residual_row_is_rejected(self):
        for rows in (self.rows[:-1],self.rows[:-1]+self.rows[:1]):
            with self.assertRaises(ValueError): audit_records(self.manifest,rows,self.report)

    def test_fabricated_group_summary_is_rejected(self):
        report=copy.deepcopy(self.report)
        report['aggregates']['cs']['all']['groups']['D']['counts']['fp']+=1
        with self.assertRaises(ValueError): audit_records(self.manifest,self.rows,report)

    def test_partial_execution_or_invalid_guard_cannot_be_accepted(self):
        report=dict(status='completed_p1_train_residual_diagnostic',smoke_only=False,records=1536,
            main_predictions=6272,predictions=6276,training=False,gradients_enabled=False,amp=False,
            test_accessed=False,val_accessed=False,elapsed_seconds=20.,budget_seconds=21600,
            maximum_observed_temperature_c=60,self_checks=[dict(model=r,repeat_bitwise_equal=True,
                temporal_swap_allclose=True,max_swap_abs_error=0.) for r in ('cs','as')],
            models={r:{'model_state_unchanged':True} for r in ('cs','as')})
        validate_execution(report,[50,60])
        for key,value in [('predictions',6275),('test_accessed',True),('elapsed_seconds',21601)]:
            changed=dict(report);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_execution(changed,[50,60])
        with self.assertRaises(ValueError): validate_execution(report,[50,85])


if __name__=='__main__': unittest.main()
