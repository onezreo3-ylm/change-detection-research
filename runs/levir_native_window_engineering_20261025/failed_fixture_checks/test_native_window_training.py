import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
import yaml

from pcdc.losses import ChangeDetectionLoss
from scripts import native_window_training as t
from scripts import native_window_supervision as s


class FakeBase:
    split="train"
    tile_size=512
    records=[SimpleNamespace(name="train_x.png")]
    tiles=[(0,0,0),(0,512,0),(0,0,512),(0,512,512)]
    def __len__(self):return len(self.tiles)
    def __getitem__(self,i):
        _,x,y=self.tiles[i]
        a=torch.arange(512*512,dtype=torch.float32).reshape(1,512,512).expand(3,-1,-1).clone()+i
        return {"image_a":a,"image_b":a+3,"mask":(a[:1]%3==0).float(),"sample_id":f"train/train_x.png@x{x}_y{y}"}


class NativeWindowTests(unittest.TestCase):
    def test_quadrants_preserve_absolute_pixels_and_order(self):
        x=torch.arange(2*3*64*64).reshape(2,3,64,64)
        q=t.quadrants(x,32)
        expected=torch.stack([x[n,:,y:y+32,z:z+32] for n in range(2) for y in (0,32) for z in (0,32)])
        self.assertTrue(torch.equal(q,expected))
        with self.assertRaises(ValueError):t.quadrants(x[:,:,:63],32)

    def test_per256_loss_values_and_gradient_exact(self):
        torch.manual_seed(9);x=torch.randn(2,1,64,64,requires_grad=True);target=(torch.rand_like(x)>.7).float()
        original=ChangeDetectionLoss();a=t.PerTileLoss(original,32)(x,target)
        b=original(t.quadrants(x,32),t.quadrants(target,32))
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
        ga,=torch.autograd.grad(a['total'],x,retain_graph=True);gb,=torch.autograd.grad(b['total'],x)
        self.assertTrue(torch.equal(ga,gb))
        with self.assertRaises(ValueError):t.PerTileLoss(original,32)(x[:,:,:60],target[:,:,:60])

    def test_control_keeps_original_loss_and_rejects_target_mismatch(self):
        x=torch.zeros(4,1,32,32);target=torch.ones_like(x);original=ChangeDetectionLoss()
        self.assertTrue(torch.equal(t.PerTileLoss(original,32)(x,target)['total'],original(x,target)['total']))
        with self.assertRaises(ValueError):t.PerTileLoss(original,32)(x,target[:3])

    def test_both_dates_and_mask_identical_between_arms(self):
        base=FakeBase();native=t.WindowGroups(base,'native512')[1];control=t.WindowGroups(base,'grouped256')[1]
        for key in ('clean_a','clean_b','mask'):
            self.assertTrue(torch.equal(t.quadrants(native[key],256),control[key]))
        for batch in (native,control):
            self.assertTrue(torch.equal(batch['clean_a'],batch['secondary_a']))
            self.assertTrue(torch.equal(batch['clean_b'],batch['secondary_b']))
            self.assertEqual(batch['clean_a'].numel(),3*512*512)
        self.assertEqual(control['sample_id'],['train/train_x.png@x512_y0','train/train_x.png@x768_y0','train/train_x.png@x512_y256','train/train_x.png@x768_y256'])

    def test_identical_group_order_across_arms_and_metadata_prediction(self):
        base=FakeBase();loaders=[]
        for arm in t.ARMS:
            loaders.append(DataLoader(t.WindowGroups(base,arm),batch_size=1,shuffle=True,num_workers=0,collate_fn=t.one_group,generator=torch.Generator().manual_seed(42)))
        metadata=DataLoader([base[i]['sample_id'] for i in range(len(base))],batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
        for _ in range(3):
            actual=[[b['group_id'] for b in l] for l in loaders];expected=[b[0] for b in metadata]
            self.assertEqual(actual[0],actual[1]);self.assertEqual(actual[0],expected)

    def test_no_validation_or_unplanned_arm_optimization(self):
        base=FakeBase();base.split='val'
        with self.assertRaises(ValueError):t.WindowGroups(base,'native512')
        with self.assertRaises(ValueError):t.WindowGroups(FakeBase(),'new_arm')
        with self.assertRaises(ValueError):t.one_group([{},{}])

    def test_study_rejects_changed_loss_unit_or_heldout(self):
        v=yaml.safe_load((t.ROOT/'configs/levir_native_window_training_20261025.yaml').read_text(encoding='utf-8'))
        t.validate_study(v)
        for key,value in [('loss_unit',512),('epochs',101),('test_or_martigny_access',True),('new_normalization',True)]:
            changed=copy.deepcopy(v);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):t.validate_study(changed)

    def test_budget_gate_requires_both_arms_and_shared_remainder(self):
        rows=[{'mean_seconds_per_step_after_first':.1},{'mean_seconds_per_step_after_first':.2}]
        b=t.budget_estimate(rows,150000,120000,86000);self.assertTrue(b['passed'])
        self.assertEqual(b['predicted_total_seconds'],81750)
        self.assertFalse(t.budget_estimate(rows,80000,120000,86000)['passed'])
        self.assertFalse(t.budget_estimate(rows,150000,120000,40000)['passed'])

    def test_supervisor_import_does_not_import_torch(self):
        r=subprocess.run([sys.executable,'-c',"import sys; import scripts.native_window_supervision; assert 'torch' not in sys.modules"],cwd=s.ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def ledger(self,d,entries=None,total=None):
        entries=entries or [];p=Path(d)/'ledger.json';p.write_text(json.dumps({'entries':entries,'total_charged_seconds':sum(e['charged_seconds'] for e in entries) if total is None else total,'batch_upper_bound_seconds':172800}),encoding='utf-8')
        return {'study_output':d,'ledger_path':str(p),'pilot_wall_limit_seconds':1800,'training_wall_limit_seconds':86400,'whole_study_limit_seconds':129600}

    def test_no_shared_or_study_or_candidate_budget_reset(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.ledger(d,[{'kind':'native_window_gpu_pilot','mode':'pilot','status':'completed','charged_seconds':100}])
            _,limit=s.reserve(c,'native512');self.assertEqual(limit,86300)
            with self.assertRaises(RuntimeError):s.reserve(c,'native512')
        with tempfile.TemporaryDirectory() as d:
            c=self.ledger(d,total=172780);_,limit=s.reserve(c,'pilot');self.assertEqual(limit,20)
        with tempfile.TemporaryDirectory() as d:
            c=self.ledger(d,[{'kind':'native_window_gpu_native512','mode':'native512','status':'completed','charged_seconds':129590}]);_,limit=s.reserve(c,'grouped256');self.assertEqual(limit,10)
        with tempfile.TemporaryDirectory() as d:
            c=self.ledger(d,[{'id':'native_window_pilot_20261025','charged_seconds':1,'status':'completed'}])
            with self.assertRaises(RuntimeError):s.reserve(c,'pilot')

    def test_runtime_rejects_weakened_safety_and_extra_mode(self):
        p=s.ROOT/'configs/levir_native_window_runtime_20261025.yaml';s.load_runtime(p)
        original=yaml.safe_load(p.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'runtime.yaml'
            for key,value in [('poll_seconds',10),('training_wall_limit_seconds',90000),('immediate_pause_c',90),('automatic_retry',True)]:
                cfg=copy.deepcopy(original);cfg['study_config']=str(s.ROOT/'configs/levir_native_window_training_20261025.yaml');cfg[key]=value;target.write_text(yaml.safe_dump(cfg))
                with self.subTest(key=key),self.assertRaises(ValueError):s.load_runtime(target)
        with self.assertRaises(ValueError):s.session_path({'study_output':'.'},'new_arm')

    def test_dead_supervisor_prevents_cuda_import_or_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'study_output':d,'startup_limit_seconds':1};session=Path(d)/'native512_session';session.mkdir()
            proof={'worker':{'process_id':os.getpid(),'creation_time_ticks':1},'owner':{},'guard':{},'watchdog':{}}
            class Dead:
                def __init__(self,settings):pass
                def is_alive(self):return False
                def close(self):pass
            with patch.object(s,'load_runtime',return_value=(Path(d)/'runtime.yaml',cfg)),patch.object(s,'attachment',return_value=proof),patch.object(s.shared,'creation_ticks',return_value=1),patch.object(s,'WindowsProcess',Dead):
                with self.assertRaises(RuntimeError):s.worker('unused','native512')
            self.assertEqual(s.read(session/'worker_result.json')['status'],'failed_preserve')

    def test_ledger_successful_updates_use_current_arm_and_skips(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.ledger(d,[{'id':'run','kind':'native_window_gpu_native512','mode':'native512','status':'running','charged_seconds':10,'research_training':True}])
            arm=Path(d)/'native512';arm.mkdir();s.atomic(arm/'status.json',{'epoch':1,'optimizer_steps':1778})
            session=Path(d)/'native512_session';session.mkdir();s.atomic(session/'progress.json',{'stage':'native512_train','epoch':2,'step':25,'amp_skips_in_stage':1})
            s.ledger_update(cfg,'run',11,'running');ledger=s.read(cfg['ledger_path'])
            self.assertEqual(ledger['new_research_optimizer_updates'],1802)
            self.assertEqual(ledger['total_charged_seconds'],11)

    def test_independent_guard_stops_worker_if_bound_owner_dies(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d)/'native512_session';session.mkdir()
            _,cfg=s.load_runtime(s.ROOT/'configs/levir_native_window_runtime_20261025.yaml');cfg['study_output']=d
            proof={r:{'role':r} for r in ('owner','guard','watchdog','worker')};proof['guard'].update(process_id=os.getpid(),creation_time_ticks=1)
            handles={}
            class Handle:
                def __init__(self,settings):self.role=settings['role'];self.active=self.role!='owner';self.stopped=False;handles[self.role]=self
                def is_alive(self):return self.active
                def stop(self):self.stopped=True;self.active=False
                def close(self):pass
            with patch.object(s,'load_runtime',return_value=(Path(d)/'runtime.yaml',cfg)),patch.object(s,'attachment',return_value=proof),patch.object(s.shared,'creation_ticks',return_value=1),patch.object(s,'WindowsProcess',Handle):
                state=s.guard_loop('unused','native512','guard')
            self.assertTrue(handles['worker'].stopped);self.assertEqual(state['status'],'stopped_worker_preserve')


if __name__=='__main__':unittest.main()
