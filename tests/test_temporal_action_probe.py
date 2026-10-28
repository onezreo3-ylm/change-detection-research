import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from pcdc.temporal_action_protocol import (ASSIGNMENTS,assignment_counts,expanded_menu,
    old_two_action_menu,require_old_menu_replay,joint_resource_gates)
from scripts import run_temporal_action_probe as runner
from scripts import temporal_action_probe_supervision as supervisor


def fixture():
    return dict(a0_counts=[100,10,10,100],full_both_counts=[100,10,10,100],
        regional_deltas=[[[1,-1,0,0],[0,0,-1,1],[1,-1,-1,1]] for _ in range(4)])


class TemporalActionProbeTests(unittest.TestCase):
    def test_all256_assignments_and_old16_preserve_exact_order(self):
        row=fixture();menu=expanded_menu(row)
        self.assertEqual(len(set(ASSIGNMENTS)),256)
        self.assertEqual(menu,[assignment_counts(row['a0_counts'],row['regional_deltas'],a) for a in ASSIGNMENTS])
        self.assertEqual(old_two_action_menu(row),[menu[ASSIGNMENTS.index(tuple(3 if i&(1<<r) else 0 for r in range(4)))] for i in range(16)])

    def test_old_subspace_replay_rejects_single_cell_or_full_action_change(self):
        row=fixture();old=dict(a0_counts=row['a0_counts'],full_adapter_counts=row['full_both_counts'],
            menus={'reference_replay_joint':old_two_action_menu(row)})
        require_old_menu_replay(row,old)
        for which in ('full_adapter_counts','menu'):
            bad=copy.deepcopy(old)
            (bad['menus']['reference_replay_joint'][7] if which=='menu' else bad[which])[0]+=1
            with self.assertRaises(ArithmeticError):require_old_menu_replay(row,bad)

    def test_label_changes_invalid_actions_and_impossible_counts_rejected(self):
        row=fixture()
        with self.assertRaises(ValueError):assignment_counts(row['a0_counts'],row['regional_deltas'],(0,1,4,0))
        bad=copy.deepcopy(row);bad['regional_deltas'][0][0]=[1,0,0,0]
        with self.assertRaises(ArithmeticError):expanded_menu(bad)
        bad=copy.deepcopy(row);bad['regional_deltas'][0][0]=[100,-100,0,0]
        with self.assertRaises(ArithmeticError):expanded_menu(bad)

    def test_strong_constant_and_extra_temporal_increment_are_both_required(self):
        v,_=runner.load();families={str(i):{'positive','negative'} for i in range(3)}
        signs=dict(positive=.5,negative=.5);constants={str(i):.90 for i in range(7)}
        self.assertTrue(all(joint_resource_gates(.91,.905,constants,signs,families,v).values()))
        self.assertFalse(joint_resource_gates(.91,.905,dict(constants,**{'6':.909}),signs,families,v)['oracle_margin'])
        self.assertFalse(joint_resource_gates(.91,.909,constants,signs,families,v)['temporal_increment'])
        with self.assertRaises(ValueError):joint_resource_gates(.91,.905,{'a':.9},signs,families,v)

    def test_real_mode_requires_original_references_before_model_execution(self):
        v,_=runner.load()
        with self.assertRaises(ValueError):runner.diagnose_batch(None,None,None,None,None,v,lambda:None)

    def test_original_architecture_actual_mixed_choices_and_compact_counts(self):
        import torch
        from pcdc.bound_local_actions import ObservedBatchIdentity
        from pcdc.model import SiameseChangeDetector
        from pcdc.phase5_models import AdapterCorrectedDetector
        v,_=runner.load();torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(v['smoke']['seed'])
            model=AdapterCorrectedDetector(SiameseChangeDetector()).eval().requires_grad_(False)
            torch.nn.init.normal_(model.adapter.conv3.weight,std=v['smoke']['nonzero_adapter_final_std'])
            a,b=torch.rand(1,3,256,256),torch.rand(1,3,256,256)
            y=(torch.rand(1,1,256,256)<v['smoke']['synthetic_foreground_probability']).float()
        identity=ObservedBatchIdentity(('train/synthetic@x0_y0',),(('a'*64,'b'*64),),('c'*64,),'d'*64,'e'*64,'f'*64)
        r=runner.diagnose_batch(model,a,b,y,identity,v,lambda:None,engineering_only=True,loss_kwargs=v['smoke']['loss'])
        self.assertEqual(r['actual_forwards'],dict(encoder=1,adapter=2,decoder=12))
        self.assertEqual(r['actual_assignments_checked'],4)
        self.assertTrue(r['full_final_identity_checks'])
        self.assertEqual(len(r['rows'][0]['singles']),12)
        self.assertEqual(len(expanded_menu(r['rows'][0])),256)
        self.assertEqual(set(runner.CONSTANTS)&set(r['rows'][0]),set(runner.CONSTANTS))

    def test_runtime_and_data_protocol_do_not_substitute_resources(self):
        _,cfg,v=supervisor.load_runtime()
        self.assertEqual(Path(cfg['global_lease']).name,'gpu_study.lock')
        self.assertEqual((cfg['confirmed_pause_c'],cfg['immediate_pause_c']),(85,86))
        self.assertFalse(v['new_model_fitting']);self.assertFalse(v['test_or_martigny_access'])
        self.assertEqual(v['amp_dtype'],'float16');self.assertEqual(v['original_eval_batch_size'],32)

    def test_temporal_subcap_includes_previous_prototypes_and_original_shared_cap(self):
        _,cfg,v=supervisor.load_runtime()
        ledger=dict(entries=[dict(kind='temporal_alternatives_CPU_engineering',status='completed',charged_seconds=34.),
            dict(kind='temporal_actions_engineering',status='completed',charged_seconds=137.)],
            total_charged_seconds=172000.,batch_upper_bound_seconds=172800.)
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic'):
            _,limit=supervisor.reserve(cfg,v,'probe')
        self.assertEqual(limit,680.)
        ledger['entries'][-1]['status']='running'
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic') as save:
            with self.assertRaises(RuntimeError):supervisor.reserve(cfg,v,'probe')
            save.assert_not_called()

    def test_prototypes_cannot_evade_engineering_reserve(self):
        _,cfg,v=supervisor.load_runtime()
        ledger=dict(entries=[dict(kind='temporal_actions_engineering',status='completed',charged_seconds=481.)],
            total_charged_seconds=150000.,batch_upper_bound_seconds=172800.)
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic') as save:
            with self.assertRaises(RuntimeError):supervisor.reserve(cfg,v,'probe')
            save.assert_not_called()

    def test_full_suite_must_include_new_temporal_and_binding_tests(self):
        modules=('test_bound_local_actions.','test_local_composition_protocol.','test_local_composition_probe.',
            'test_b0e_fixed_evaluation.','test_local_composition_resource_v2.','test_bound_temporal_actions.','test_temporal_action_probe.')
        log='\n'.join(modules)+'\nRan 636 tests in 90.000s\n\nOK\n'
        self.assertEqual(supervisor.passed_suite_count(log),636)
        with self.assertRaises(ValueError):supervisor.passed_suite_count(log.replace('test_temporal_action_probe.','old'))

if __name__=='__main__':unittest.main()
