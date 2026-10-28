"""Fail-closed scope, BN-only intervention and exact R5 accounting tests."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from pcdc.losses import ChangeDetectionLoss
from scripts.coupling_diagnostic import four_cell_diagnostic
from scripts.valais_feasibility_screen import (calibration_selection, probe_measurements,
    recalibrating_bn, report_counts, r5_loss, unchanged_except_bn, validate_selection)


class ScreenSelectionTests(unittest.TestCase):
    def test_hash_selection_is_order_independent_disjoint_and_exact_batch_composition(self):
        names=[f'{i}.tif' for i in range(400)]
        foreground={n:100 for n in names[:100]}
        probes=names[390:]
        a=calibration_selection(names,foreground,probes,123)
        b=calibration_selection(list(reversed(names)),foreground,probes,123)
        self.assertEqual(a,b)
        for rows in a.values():
            self.assertEqual(len(rows),256)
            self.assertEqual(len(set(rows)),256)
            self.assertFalse(set(rows)&set(probes))
        for i in range(0,256,4):
            self.assertEqual(sum(n in foreground for n in a['stratified_clean'][i:i+4]),1)

    def test_invalid_names_overlap_or_insufficient_real_positive_data_are_rejected(self):
        names=[f'{i}.tif' for i in range(400)]
        for inventory,foreground,probes in [(names+names[:1],{},[]),
            (names,{},[]),(names,{},['absent.tif']),(['test/x.tif',*names],{},[])]:
            with self.subTest(inventory=inventory[:1]),self.assertRaises(ValueError):
                calibration_selection(inventory,foreground,probes,123)

    def manifest(self,root):
        def sample(i,positive=False):
            name=f'{i}.tif'
            return {'name':name,'foreground_pixels':int(positive),
                'files':{k:{'path':str(root/'train'/folder/name)} for k,folder in
                         [('image_a','A'),('image_b','B'),('mask','label')]}}
        probes=[sample(i,i<16) for i in range(64)]
        uniform=[sample(i) for i in range(64,320)]
        stratified=[sample(i,i%4==0) for i in range(320,576)]
        return {'data_root':str(root),'probe':probes,
                'calibration':{'uniform_clean':uniform,'stratified_clean':stratified}}

    def test_alias_to_test_or_calibration_probe_overlap_is_rejected_before_image_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.manifest(Path(tmp))
            validate_selection(m)
            bad=copy.deepcopy(m)
            bad['calibration']['uniform_clean'][0]=bad['probe'][0]
            with self.assertRaises(ValueError): validate_selection(bad)
            bad=copy.deepcopy(m)
            bad['probe'][0]['files']['image_a']['path']=str(Path(tmp)/'test'/'A'/'0.tif')
            with self.assertRaises(ValueError): validate_selection(bad)


class ScreenBNTests(unittest.TestCase):
    def test_only_bn_statistics_update_and_batch_count_and_momentum_restore(self):
        torch.manual_seed(7)
        model=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4),torch.nn.Dropout(.5))
        original=copy.deepcopy(model.state_dict())
        momentum=model[1].momentum
        model.requires_grad_(False)
        with recalibrating_bn(model):
            self.assertTrue(model[1].training)
            self.assertFalse(model[0].training)
            self.assertFalse(model[2].training)
            self.assertFalse(torch.is_grad_enabled())
            model(torch.full((8,3,4,4),2.))
            model(torch.full((8,3,4,4),4.))
        changed=unchanged_except_bn(original,model.state_dict())
        self.assertIn('1.running_mean',changed)
        self.assertEqual(int(model[1].num_batches_tracked),2)
        self.assertEqual(model[1].momentum,momentum)
        self.assertFalse(model[1].training)
        model.load_state_dict(original)
        self.assertEqual(unchanged_except_bn(original,model.state_dict()),[])

    def test_parameter_mutation_is_not_accepted_as_bn_calibration(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4))
        original=copy.deepcopy(model.state_dict())
        with torch.no_grad(): model[0].weight.add_(1)
        with self.assertRaises(ValueError): unchanged_except_bn(original,model.state_dict())

    def test_error_during_calibration_restores_eval_mode_and_momentum(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm2d(2))
        old=model[0].momentum
        with self.assertRaises(RuntimeError):
            with recalibrating_bn(model): raise RuntimeError('sensor failure')
        self.assertFalse(model[0].training)
        self.assertEqual(model[0].momentum,old)


class ScreenAccountingTests(unittest.TestCase):
    def test_probe_counts_and_errors_match_original_diagnostic_and_conserve(self):
        target=torch.tensor([0,0,1,1.]).reshape(1,1,2,2)
        clean=torch.tensor([-2,2,-2,2.]).reshape_as(target)
        changed=-clean
        counts,transitions=probe_measurements(clean,changed,target)
        old=four_cell_diagnostic(clean,{c:changed for c in ('uu','uv','vu','vv')},target)
        self.assertEqual(counts,old['cells']['uu']['counts'])
        self.assertEqual(transitions,{k:old['cells']['uu']['transitions'][k] for k in transitions})
        old_counts=old['clean']['counts']
        self.assertEqual(counts['fn']-old_counts['fn'],transitions['tp_to_fn']-transitions['fn_to_tp'])
        self.assertEqual(counts['fp']-old_counts['fp'],transitions['tn_to_fp']-transitions['fp_to_tn'])

    def test_nonfinite_probe_or_nonbinary_labels_fail_closed(self):
        x=torch.zeros(1,1,2,2)
        for z,y in [(x+float('nan'),x),(x,x+.5)]:
            with self.assertRaises(ValueError): probe_measurements(x,z,y)

    def test_pool_empty_fpr_uses_empty_tiles_and_area_groups_recover_counts(self):
        rows=[{'foreground_pixels':0,'counts':{'tn':90,'fp':10,'fn':0,'tp':0}},
              {'foreground_pixels':10,'counts':{'tn':85,'fp':5,'fn':4,'tp':6}}]
        report=report_counts(rows)
        self.assertEqual(report['empty_fpr'],.1)
        self.assertEqual(report['counts'],{'tn':175,'fp':15,'fn':4,'tp':6})
        for k in report['counts']:
            self.assertEqual(sum(v['counts'][k] for v in report['by_area'].values()),report['counts'][k])

    def test_r5_exact_weights_preserve_clean_half_and_all_four_cell_gradients(self):
        values=torch.tensor([0.,1.,2.,3.,4.],requires_grad=True)
        logits=values.reshape(5,1,1,1)
        target=torch.ones(1,1,1,1)
        loss_fn=ChangeDetectionLoss(1,1,1)
        actual=r5_loss(logits,target,loss_fn)
        expected=sum(weight*loss_fn(logits[i:i+1],target)['total'] for i,weight in enumerate([.5,.125,.125,.125,.125]))
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        expected_grad=torch.autograd.grad(expected,values,retain_graph=True)[0]
        actual.backward()
        torch.testing.assert_close(values.grad,expected_grad,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue((values.grad.abs()>0).all())
        with self.assertRaises(ValueError): r5_loss(logits[:4],target,loss_fn)


if __name__=='__main__': unittest.main()
