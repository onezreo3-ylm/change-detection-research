import copy
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pcdc.bound_local_actions import ObservedBatchIdentity
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector
from scripts import run_local_composition_probe as runner
from scripts import local_composition_probe_supervision as supervisor


class LocalCompositionProbeTests(unittest.TestCase):
    def test_selected_train_subset_is_picklable_for_windows_loader_workers(self):
        from types import SimpleNamespace
        # Use an actual map-style object with a global class, as required by Windows spawn.
        from torch.utils.data import TensorDataset
        actual=TensorDataset(torch.arange(20));actual.split='train'
        subset=runner.SelectedTrainTiles(actual,[8,9,10,11])
        restored=pickle.loads(pickle.dumps(subset))
        self.assertEqual(restored.split,'train')
        self.assertEqual([int(restored[i][0]) for i in range(len(restored))],[8,9,10,11])
        actual.split='val'
        with self.assertRaises(ValueError): runner.SelectedTrainTiles(actual,[0])

    def test_frozen_runtime_reuses_actual_shared_lease_and_protection(self):
        path,cfg,v=supervisor.load_runtime()
        self.assertEqual(Path(cfg['global_lease']).name,'gpu_study.lock')
        self.assertEqual(cfg['confirmed_pause_c'],85)
        self.assertEqual(cfg['immediate_pause_c'],86)
        self.assertFalse(v['new_model_fitting'])
        self.assertFalse(v['test_or_martigny_access'])
        self.assertEqual(supervisor.session_path(cfg,'probe').name,'probe_session')
        with self.assertRaises(ValueError): supervisor.session_path(cfg,'evaluate')

    def test_full_suite_requires_new_modules_and_terminal_success(self):
        names=['test_bound_local_actions.','test_local_composition_protocol.','test_local_composition_probe.','test_b0e_fixed_evaluation.']
        log='\n'.join(names)+'\nRan 604 tests in 30.000s\n\nOK\n'
        self.assertEqual(supervisor.passed_suite_count(log),604)
        with self.assertRaises(ValueError): supervisor.passed_suite_count(log.replace('test_local_composition_probe.','old'))
        with self.assertRaises(ValueError): supervisor.passed_suite_count(log.replace('604','595'))
        with self.assertRaises(ValueError): supervisor.passed_suite_count(log.replace('OK','FAILED'))

    def test_budget_cannot_reset_active_or_existing_owner_reservation(self):
        _,cfg,v=supervisor.load_runtime()
        with patch.object(supervisor,'read',return_value={'entries':[{'status':'running'}]}):
            with self.assertRaises(RuntimeError): supervisor.reserve(cfg,v,'probe')
        with patch.object(supervisor,'read',return_value={'entries':[{'id':'local_composition_probe_owner_20261027','status':'completed'}]}):
            with self.assertRaises(RuntimeError): supervisor.reserve(cfg,v,'probe')

    def test_budget_reserves_engineering_and_finalization_and_keeps_original_cap(self):
        _,cfg,v=supervisor.load_runtime()
        ledger=dict(entries=[],batch_upper_bound_seconds=172800,total_charged_seconds=172000)
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic') as save:
            identifier,limit=supervisor.reserve(cfg,v,'probe')
        self.assertEqual(limit,680)
        self.assertEqual(save.call_args.args[1]['entries'][0]['maximum_seconds'],680)
        self.assertFalse(save.call_args.args[1]['entries'][0]['research_training'])

    def test_full_all_subset_original_architecture_mechanism_on_explicit_cpu(self):
        v,_=runner.load();torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            model=AdapterCorrectedDetector(SiameseChangeDetector()).eval().requires_grad_(False)
            torch.nn.init.normal_(model.adapter.conv3.weight,std=.01)
            a,b=torch.rand(1,3,256,256),torch.rand(1,3,256,256)
            target=(torch.rand(1,1,256,256)<.25).float()
        identity=ObservedBatchIdentity(('train/explicit_synthetic@x0_y0',),(('a'*64,'b'*64),),('c'*64,),'d'*64,'e'*64,'f'*64)
        r=runner.diagnose_batch(model,a,b,target,identity,v,lambda:None,engineering_only=True)
        self.assertTrue(r['exact_joint_composition']);self.assertTrue(r['exact_per_sample_counts'])
        self.assertEqual(r['actual_forwards'],{'encoder':1,'adapter':2,'decoder':33})
        self.assertEqual(len(r['rows']),1)
        self.assertTrue(all(len(menu)==16 for menu in r['rows'][0]['menus'].values()))
        self.assertEqual(len(r['rows'][0]['singles']),4)

    def test_truth_informed_probe_upper_bound_does_not_establish_method_success(self):
        v,_=runner.load();rows=[]
        for sid in ['train/fit','train/cal']:
            for cid in ['clean','brightness_l1']:
                base=[8,2,3,7];correct=[10,0,0,10]
                rows.append(dict(sample_id=sid+'@x0_y0',source_id=sid,condition_id=cid,
                    a0_counts=base,full_adapter_counts=base,
                    menus={arm:[base]+[correct]*15 for arm in v['arms']},
                    singles=[dict(loss_utility=1.),dict(loss_utility=-1.)]))
        r=runner.summarize(rows,v,dict(fit_names=['fit'],calibration_names=['cal']),['clean','brightness_l1'])
        self.assertFalse(r['actual_method_effectiveness'])
        self.assertFalse(r['method_novelty_verified'])
        self.assertFalse(r['next_fitting_resource_gate']) # Only one corruption family, below the frozen gate.
        self.assertGreater(r['groups']['fit']['reference_oracle_gain_over_better_fixed'],0)

    def test_cost_settlement_rejects_active_shared_ledger_without_mutation(self):
        _,cfg,_=supervisor.load_runtime()
        with patch.object(supervisor,'read',side_effect=[{'entries':[]},{'entries':[{'status':'running'}]}]),patch.object(supervisor,'atomic') as save:
            with self.assertRaises(RuntimeError): supervisor.sync_costs(cfg)
            save.assert_not_called()


if __name__=='__main__':unittest.main()
