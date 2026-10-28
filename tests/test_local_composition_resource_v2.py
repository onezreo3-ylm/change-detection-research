import unittest
from unittest.mock import patch

from pcdc.local_composition_resource import amortized_condition_prediction,require_resource_only_revision


class ResourceRevisionTests(unittest.TestCase):
    def test_v2_reuses_exact_model_action_functions_and_cumulative_family_cap(self):
        from scripts import run_local_composition_probe as v1
        from scripts import run_local_composition_probe_v2 as v2
        from scripts import local_composition_probe_supervision_v2 as supervisor
        self.assertIs(v2.diagnose_batch,v1.diagnose_batch)
        self.assertIs(v2.summarize,v1.summarize)
        _,cfg,v=supervisor.load_runtime()
        ledger={'entries':[{'kind':'local_composition_probe_owner_wall','status':'completed','charged_seconds':44.5},
                           {'kind':'local_composition_probe_engineering','status':'completed','charged_seconds':190.}],
                'total_charged_seconds':150000.,'batch_upper_bound_seconds':172800.}
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic') as save:
            _,limit=supervisor.reserve(cfg,v,'probe')
        self.assertEqual(limit,3600.-44.5-300.-120.)
        self.assertEqual(save.call_args.args[1]['entries'][-1]['kind'],'local_composition_probe_owner_wall')

    def test_cumulative_engineering_cannot_reset_at_v2(self):
        from scripts import local_composition_probe_supervision_v2 as supervisor
        _,cfg,v=supervisor.load_runtime()
        ledger={'entries':[{'kind':'local_composition_probe_engineering','status':'completed','charged_seconds':301.}],
                'total_charged_seconds':150000.,'batch_upper_bound_seconds':172800.}
        with patch.object(supervisor,'read',return_value=ledger),patch.object(supervisor,'atomic') as save:
            with self.assertRaises(RuntimeError):supervisor.reserve(cfg,v,'probe')
            save.assert_not_called()

    def test_scientific_inputs_and_gates_cannot_change_in_resource_revision(self):
        old=dict(pilot_original_batches=1,whole_probe_limit_seconds=3600,threshold=.5,selected_original_batches=8)
        new=dict(old,pilot_original_batches=2,output_dir='new_identity')
        require_resource_only_revision(new,old)
        for key,value in [('threshold',.51),('selected_original_batches',4),('whole_probe_limit_seconds',7200)]:
            with self.assertRaises(ValueError):require_resource_only_revision(dict(new,**{key:value}),old)

    def test_startup_is_paid_once_per_condition_and_steady_every_batch(self):
        r=amortized_condition_prediction([[14.,3.],[15.,3.5]],conditions=46,batches_per_condition=8,multiplier=1.5,fixed_overhead=180.)
        self.assertEqual(r['per_condition_fixed_seconds'],11.5)
        self.assertEqual(r['maximum_steady_batch_seconds'],3.5)
        self.assertEqual(r['predicted_main_seconds'],1.5*46*(11.5+8*3.5)+180.)
        self.assertFalse(r['uses_performance_scores'])

    def test_slower_steady_batch_is_not_hidden_by_negative_startup(self):
        r=amortized_condition_prediction([[2.,5.],[3.,4.]],conditions=2,batches_per_condition=8,multiplier=1.5,fixed_overhead=0.)
        self.assertEqual(r['per_condition_fixed_seconds'],0.)
        self.assertEqual(r['predicted_main_seconds'],1.5*2*8*5)

    def test_missing_nonfinite_or_optimistic_parameters_are_rejected(self):
        for data in ([],[[3.]],[[3.,float('nan')]],[[3.,0.]],[[True,2.]]):
            with self.assertRaises(ValueError):amortized_condition_prediction(data,conditions=46,batches_per_condition=8,multiplier=1.5,fixed_overhead=180.)
        with self.assertRaises(ValueError):amortized_condition_prediction([[3.,2.]],conditions=46,batches_per_condition=8,multiplier=.5,fixed_overhead=180.)


if __name__=='__main__':unittest.main()
