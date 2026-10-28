import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from pcdc.losses import ChangeDetectionLoss
from scripts.levir_tail_risk_control import EmpiricalTailLoss, estimate_budget, load_study, state_digest, ROOT, FLAVOR
from scripts.levir_tail_risk_supervision import age_seconds, ledger_update, require_fresh
from scripts import levir_tail_risk_supervision as supervision
from scripts.windows_shared_json import atomic_json, read_json, shared_text_reader, sharing_operation


class LevirTailRiskTests(unittest.TestCase):
    def test_q1_matches_original_values_and_full_gradient_exactly(self):
        torch.manual_seed(19)
        x = torch.randn(8,1,9,7, requires_grad=True)
        y = torch.randint(0,2,x.shape).float()
        args = dict(bce_weight=0.7, dice_weight=1.3, dice_smooth=1.)
        a, b = ChangeDetectionLoss(**args)(x,y), EmpiricalTailLoss(**args,q=1)(x,y)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        ga, = torch.autograd.grad(a['total'], x)
        gb, = torch.autograd.grad(b['total'], x)
        self.assertTrue(torch.equal(ga,gb))

    def test_selects_by_weighted_bce_and_dice_and_clean_always_retained(self):
        x = torch.tensor([-2.,0.,1.,-5.,1.,3.]).reshape(6,1,1,1).requires_grad_()
        y = torch.zeros_like(x)
        fn = EmpiricalTailLoss(1.,1.,1.)
        bce, dice = fn.components(x,y)
        expected = 0.5*(bce[:3]+dice[:3]).mean() + 0.5*(bce[4:]+dice[4:]).mean()
        out = fn(x,y)['total']
        self.assertTrue(torch.allclose(out,expected))
        out.backward()
        self.assertTrue((x.grad[:3] != 0).all())
        self.assertEqual(float(x.grad[3]),0.)
        self.assertTrue((x.grad[4:] != 0).all())

    def test_ties_choose_original_secondary_index_and_ceil_odd(self):
        x = torch.zeros(6,1,2,2, requires_grad=True)
        EmpiricalTailLoss(1.,1.,1.)(x,torch.zeros_like(x))['total'].backward()
        self.assertTrue((x.grad[3:5] != 0).all())
        self.assertTrue((x.grad[5] == 0).all())

    def test_single_source_tail_equals_original(self):
        x = torch.tensor([-3.,2.]).reshape(2,1,1,1)
        y = torch.tensor([0.,1.]).reshape_as(x)
        a = EmpiricalTailLoss(1.,1.,1.)(x,y)
        b = ChangeDetectionLoss(1.,1.,1.)(x,y)
        self.assertTrue(torch.allclose(a['total'], b['total']))

    def test_rejects_odd_or_misaligned_view_batches(self):
        fn = EmpiricalTailLoss(1.,1.,1.)
        for shape, other in [((3,1,2,2),(3,1,2,2)), ((4,2,2,2),(4,2,2,2)), ((4,1,2,2),(4,1,2,3))]:
            with self.assertRaises(ValueError):
                fn(torch.zeros(shape),torch.zeros(other))

    def test_rejects_invalid_q(self):
        for q in [0,-.5,1.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                EmpiricalTailLoss(1.,1.,1.,q=q)

    def test_fp16_logits_accumulate_in_fp32(self):
        x = torch.zeros(4,1,2,2,dtype=torch.float16, requires_grad=True)
        out = EmpiricalTailLoss(1.,1.,1.)(x,torch.zeros_like(x))
        self.assertTrue(all(v.dtype == torch.float32 for v in out.values()))
        out['total'].backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_exact_state_digest_detects_optimizer_bn_rng_changes(self):
        state = {'model':{'bn.running_mean':torch.tensor([1.])},'optimizer':{'step':torch.tensor(2.)},'rng':[3,4]}
        baseline = state_digest(state)
        self.assertEqual(baseline,state_digest(copy.deepcopy(state)))
        for key in ['model','optimizer','rng']:
            changed = copy.deepcopy(state)
            if key == 'rng':
                changed[key][0] += 1
            else:
                changed[key][next(iter(changed[key]))] += 1
            self.assertNotEqual(baseline,state_digest(changed))

    def test_budget_margin_and_prior_spend_are_both_binding(self):
        values = {'engineering':{'allowance_fraction':.25,'per_epoch_validation_save_allowance_seconds':60},
                  'candidate':{'total_attempted_steps':178000,'epochs':100},
                  'budgets':{'single_training_upper_bound_hours':24,'batch_upper_bound_gpu_hours':48}}
        self.assertTrue(estimate_budget(.2,values,300)['passed'])
        self.assertFalse(estimate_budget(.4,values,300)['passed'])
        self.assertFalse(estimate_budget(.2,values,160000)['passed'])
        self.assertFalse(estimate_budget(.2,values,300,prior_training_seconds=80000)['passed'])

    @unittest.skipUnless(os.name == 'nt','Windows file-sharing semantics')
    def test_windows_writer_waits_for_reader_then_publishes_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state.json'
            atomic_json(p,{'step':1})
            completed=threading.Event()
            failures=[]
            def publish():
                try: atomic_json(p,{'step':2})
                except BaseException as error: failures.append(error)
                finally: completed.set()
            with shared_text_reader(p) as old:
                writer=threading.Thread(target=publish)
                writer.start()
                self.assertFalse(completed.wait(.05))
                self.assertEqual(json.load(old),{'step':1})
            writer.join(timeout=3)
            self.assertTrue(completed.is_set())
            self.assertEqual(failures,[])
            self.assertEqual(read_json(p),{'step':2})

    def test_invalid_json_is_never_treated_as_a_fresh_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state.json'
            p.write_text('{broken',encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                read_json(p)

    def test_unrecoverable_access_error_is_not_hidden_or_retried_forever(self):
        def denied(): raise PermissionError('persistent access denied')
        with self.assertRaises(PermissionError):
            sharing_operation(denied,timeout=0)

    def test_failed_prefix_stays_charged_against_same_24_hour_candidate_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'ledger.json'
            p.write_text(json.dumps({'entries':[{'kind':'tail_risk_train_whole_worker_process',
                'charged_seconds':47.906,'status':'failed_preserve'}], 'batch_upper_bound_seconds':172800,
                'total_charged_seconds':458.602}))
            cfg={'ledger_path':str(p),'pilot_wall_limit_seconds':1800,'single_training_limit_seconds':86400}
            with patch.object(supervision,'ROOT',Path(directory)):
                _,limit=supervision.reserve_charge(cfg,'train',Path(directory)/'new_session')
            self.assertAlmostEqual(limit,86400-47.906)

    def test_stale_or_inactive_heartbeat_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'state.json'
            for status, age in [('running',60),('failed_preserve',0)]:
                p.write_text(json.dumps({'status':status,'updated_utc':(datetime.now(timezone.utc)-timedelta(seconds=age)).isoformat()}))
                with self.assertRaises(RuntimeError):
                    require_fresh(p,30)

    def test_future_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            age_seconds((datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat())

    def test_ledger_never_reduces_or_resets_prior_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'ledger.json'
            p.write_text(json.dumps({'entries':[{'id':'prior','charged_seconds':280}, {'id':'new','charged_seconds':5}]}))
            cfg = {'ledger_path':str(p)}
            ledger_update(cfg,'new',9,'completed',0)
            value = json.loads(p.read_text())
            self.assertEqual(value['total_charged_seconds'],289)
            self.assertEqual(value['entries'][0]['charged_seconds'],280)
            with self.assertRaises(ValueError):
                ledger_update(cfg,'new',8,'completed',0)

    def test_real_study_preserves_original_scope_and_explicit_flavor(self):
        _, values, ref, _ = load_study(ROOT/'configs/levir_tail_risk_control_20261024.yaml')
        self.assertEqual(values['candidate']['loss']['upper_tail_fraction'],.5)
        self.assertFalse(ref.base.run_test_after_training)
        self.assertNotEqual(FLAVOR,'matched_robust_baseline_v1')

    def exercise_guard_failure(self, role, *, dead_owner=False, temperature=40, wall_seconds=100):
        class Handle:
            def __init__(self, alive=True):
                self.alive, self.stopped = alive, False
            def is_alive(self): return self.alive
            def stop(self): self.stopped, self.alive = True, False
            def close(self): pass
        worker, owner, peer = Handle(), Handle(not dead_owner), Handle()
        cfg = {'study_output':'X:/unused','startup_limit_seconds':120, 'heartbeat_max_age_seconds':30,
               'progress_stall_seconds':900,'confirmed_pause_c':85,'immediate_pause_c':86,
               'confirmation_samples':3,'max_sensor_failures':3,'confirmation_seconds':1,'poll_seconds':2}
        proof = {'runtime_sha256':'same','worker':{},'owner':{},'guard':{'process_id':12,'creation_time_ticks':34},
                 'watchdog':{'process_id':12,'creation_time_ticks':34}, 'allowed_wall_seconds':wall_seconds,
                 'created_utc':datetime.now(timezone.utc).isoformat()}
        with patch.object(supervision,'load_runtime',return_value=(Path('runtime'),cfg)), \
             patch.object(supervision,'wait_attachment',return_value=proof), \
             patch.object(supervision,'sha',return_value='same'), \
             patch.object(supervision,'WindowsProcess',side_effect=[worker,owner,peer]), \
             patch.object(supervision,'creation_ticks',return_value=34), \
             patch.object(supervision.os,'getpid',return_value=12), \
             patch.object(supervision,'require_fresh',return_value={}), \
             patch.object(supervision,'atomic'), \
             patch.object(supervision,'gpu_state',return_value={'temperature_c':temperature}), \
             patch.object(supervision.time,'monotonic',side_effect=[0,1]), \
             patch.object(supervision.traceback,'print_exc'):
            result = supervision.guard_loop(Path('runtime'),'pilot',role)
        self.assertTrue(worker.stopped)
        self.assertEqual(result['status'],'stopped_worker_preserve')
        return result

    def test_independent_guard_stops_bound_worker_if_owner_dies(self):
        result = self.exercise_guard_failure('guard',dead_owner=True)
        self.assertIn('owner',result['error'])

    def test_independent_guard_stops_at_86_degrees(self):
        # This route reaches peer readiness before its first temperature query.
        with patch.object(Path,'exists',return_value=True), patch.object(supervision,'read',return_value={'updated_utc':datetime.now(timezone.utc).isoformat()}):
            result = self.exercise_guard_failure('guard',temperature=86)
        self.assertIn('temperature_pause',result['error'])

    def test_independent_watchdog_stops_at_original_wall_limit(self):
        result = self.exercise_guard_failure('watchdog',wall_seconds=.5)
        self.assertIn('budget',result['error'])


if __name__ == '__main__':
    unittest.main()
