"""Research-identity, resource and process-protection regression checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import complete_b0e_replication as runner
from scripts import b0e_replication_supervision as supervisor


class MissingControlTests(unittest.TestCase):
    def test_frozen_configs_are_clean_same_seed_r1_controls(self):
        v,refs,out=runner.load_study()
        self.assertEqual(list(refs),[43,44])
        for seed,ref in refs.items():
            self.assertEqual(ref.mode,'b0e')
            self.assertEqual(ref.secondary,'clean')
            self.assertFalse(ref.base.run_test_after_training)
            self.assertEqual(ref.base.values['experiment']['seed'],seed)

    def test_runtime_retains_identity_and_independent_thermal_policy(self):
        _,cfg,v=supervisor.load_runtime()
        self.assertEqual(cfg['immediate_pause_c'],86)
        self.assertEqual(cfg['poll_seconds'],2)
        self.assertEqual(cfg['gpu_uuid'],'GPU-8ada1895-7628-bb88-1dda-c57679117fa2')
        self.assertFalse(cfg['automatic_retry'])

    def test_cpu_fp32_calls_archived_runner_with_disabled_scaler(self):
        import torch
        _,refs,_=runner.load_study()
        with patch.object(runner,'run_two_view_epoch',return_value={'fixture':'only'}) as call:
            runner.epoch_run(refs[43],None,None,torch.device('cpu'),None,None,lambda:None,amp=False)
        self.assertFalse(call.call_args.kwargs['amp'])
        self.assertFalse(call.call_args.kwargs['scaler'].is_enabled())

    def test_budget_uses_slowest_repeated_timing_and_reserves_evaluation(self):
        rows=[dict(seed=43,seconds_per_step_after_first=.1),
              dict(seed=44,seconds_per_step_after_first=.11),dict(seed=43,seconds_per_step_after_first=.12)]
        result=runner.budget_estimate(rows,80000)
        self.assertEqual(result['predicted_training_seconds']['43'],(.12*178000+6000)*1.25)
        total=sum(result['predicted_training_seconds'].values())
        self.assertTrue(runner.budget_estimate(rows,total+3600)['passed'])
        self.assertFalse(runner.budget_estimate(rows,total+3599)['passed'])

    def test_invalid_or_incomplete_pilot_cannot_admit_training(self):
        for rows in ([],[dict(seed=43,seconds_per_step_after_first=.1)]*3,
                     [dict(seed=s,seconds_per_step_after_first=float('nan')) for s in (43,44,43)]):
            with self.assertRaises(ValueError):runner.budget_estimate(rows,100000)

    def completed_fixture(self):
        history=[];plan=dict(epochs=[])
        for epoch in range(1,101):
            history.append(dict(epoch=epoch,train=dict(source_samples=7120,pair_views=14240,
                optimizer_steps=1780,amp_skipped_steps=0,exposure_stream_sha256=f'h{epoch}'),
                val=dict(metrics=dict(f1=.9 if epoch in (7,8) else .8))))
            plan['epochs'].append(dict(clean_exposure_sha256=f'h{epoch}'))
        status=dict(status='completed',test_evaluated=False,initial_model_tensor_sha256='initial',
                    best_epoch=7,best_val_f1=.9)
        return history,plan,status

    def test_complete_order_and_first_maximum_are_required(self):
        history,plan,status=self.completed_fixture()
        result=runner.check_history(history,plan,status,'initial')
        self.assertEqual(result['attempts'],178000)
        status['best_epoch']=8
        with self.assertRaisesRegex(ValueError,'selection'):runner.check_history(history,plan,status,'initial')

    def test_changed_stream_and_exposure_are_rejected(self):
        for key,value in [('source_samples',7119),('pair_views',14239),
                          ('optimizer_steps',1779),('exposure_stream_sha256','other')]:
            history,plan,status=self.completed_fixture();history[52]['train'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.check_history(history,plan,status,'initial')

    def test_completed_test_or_wrong_initialization_are_rejected(self):
        history,plan,status=self.completed_fixture();status['test_evaluated']=True
        with self.assertRaises(ValueError):runner.check_history(history,plan,status,'initial')
        status['test_evaluated']=False
        with self.assertRaisesRegex(ValueError,'initialization'):runner.check_history(history,plan,status,'other')

    def test_ledger_preserves_historical_charge_and_never_decreases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.json'
            path.write_text(json.dumps(dict(entries=[dict(id='old',charged_seconds=9),
                dict(id='new',charged_seconds=2,research_training=False)],total_charged_seconds=11)),encoding='utf-8')
            cfg=dict(ledger_path=str(path),study_output=tmp)
            supervisor.ledger_update(cfg,'new',4,'running')
            current=json.loads(path.read_text())
            self.assertEqual(current['total_charged_seconds'],13)
            self.assertEqual(current['entries'][0]['charged_seconds'],9)
            with self.assertRaises(ValueError):supervisor.ledger_update(cfg,'new',3,'running')

    def test_preparation_tamper_is_detected_without_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'derived.json';p.write_text('{}')
            bound={str(p):runner.sha(p)}
            runner.verify(bound);p.write_text('{"edited":true}')
            with self.assertRaises(ValueError):runner.verify(bound)

    def test_guard_requires_creation_bound_attachment_and_peers(self):
        # Prevent regressions in the pre-CUDA attachment contract.
        with tempfile.TemporaryDirectory() as tmp:
            session=Path(tmp)
            with patch.object(supervisor.common,'wait_attachment',return_value={
                'runtime_sha256':'wrong','adapter_sha256':'wrong'}):
                with self.assertRaises(ValueError):supervisor.attached({},session)


if __name__=='__main__':unittest.main()
