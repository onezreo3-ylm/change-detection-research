"""The approved shorter horizon must have a new identity and retain finite gates."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from pcdc.config import Config, load_config, protocol_signature
from scripts import five_view_baselines as f
from scripts import c5r5_40ep_identity as identity
from scripts import five_view_40ep as worker
from scripts.valais_c5r5_40ep import passing_test_count

CONFIG = f.ROOT/'configs/valais_c5r5_40ep_seed42_v1_20261020.yaml'


class FortyEpochTests(unittest.TestCase):
    def fixture(self):
        values=copy.deepcopy(load_config(f.ROOT/'configs/valais_clean_full_fp32_seed42_20261002.yaml').values)
        values['train']['epochs']=40
        cfg=Config(values,Path('base.yaml'))
        files={'train':{},'val':{}};manifests={'train':{},'val':{}}
        ident={'role':'c5','implementation_sha256':{},'initial_model_tensor_sha256':'initial',
            'data_files_sha256':f.digest(files),'manifests_sha256':f.digest(manifests)}
        plan=[{'epoch':1,'source_draws':11025,'pair_views':55125}]
        history=[{'epoch':1,'val':{'metrics':{'f1':.2}},
            'train':dict(plan[0],optimizer_steps=2757,amp_skipped_steps=0,actual_streams_verified=True)}]
        cp={'format_version':1,'kind':'research','training_flavor':identity.FLAVOR,'role':'c5','identity':ident,
            'protocol_signature':protocol_signature(cfg.values),'config':cfg.values,'training_signature':f.digest(ident),
            'source_sha256':{},'initial_model_tensor_sha256':'initial','file_sha256':files,'manifests':manifests,
            'history':history,'epoch':1,'best_epoch':1,'best_val_f1':.2,'model':{'w':torch.ones(1)},
            'optimizer':{'state':{}},'scheduler':{'T_max':40,'last_epoch':1}}
        return cp,cfg,ident,plan

    def test_approved_horizon_is_independent_and_original_100_is_unchanged(self):
        _,v,paths,_=identity.load_study(CONFIG)
        _,parent,_,_=f.load_study(paths['parent_config'])
        self.assertEqual((v['epochs'],v['cosine_tmax']),(40,40))
        self.assertEqual(parent['epochs'],100);self.assertNotEqual(identity.FLAVOR,f.FLAVOR)
        self.assertEqual(v['gates'],parent['gates']);self.assertNotEqual(paths['output_dir'],paths['parent_output_dir'])

    def test_unreviewed_scope_budget_schedule_or_sampling_changes_are_rejected(self):
        values=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
        for key,value in [('epochs',39),('cosine_tmax',100),('maximum_training_seconds_per_role',172800),
                ('source_batch_size',2),('pair_seed',1),('carry_parent_pilot_debit',False),('training_flavor',f.FLAVOR)]:
            with tempfile.NamedTemporaryFile('w',suffix='.yaml',dir=CONFIG.parent,delete=False,encoding='utf-8') as stream:
                changed=dict(values);changed[key]=value;yaml.safe_dump(changed,stream);path=Path(stream.name)
            try:
                with self.subTest(key=key),self.assertRaises(ValueError):identity.load_study(path)
            finally:path.unlink()

    def test_projection_uses_40_epochs_and_includes_all_engineering_debits(self):
        _,v,_,_=identity.load_study(CONFIG)
        result=identity.estimate_budget([.56820833333445]*24,v,80.)
        self.assertAlmostEqual(result['projected_total_role_seconds'],.56820833333445*110280*1.25+2400+80)
        self.assertTrue(result['within_frozen_budget'])
        self.assertFalse(identity.estimate_budget([.7]*24,v,80)['within_frozen_budget'])

    def test_invalid_or_nonfinite_timing_and_debit_are_rejected(self):
        _,v,_,_=identity.load_study(CONFIG)
        for timings,debit in [([],0),([float('nan')],0),([float('inf')],0),([0],0),([.5],-1),([.5],1800)]:
            with self.subTest(timings=timings,debit=debit),self.assertRaises(ValueError):identity.estimate_budget(timings,v,debit)

    def test_role_budget_does_not_reset_the_original_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'authorization.json').write_text('{"engineering_seconds_debited":{"c5":123.5,"r5":120}}')
            self.assertEqual(identity.role_debit({'output_dir':root},'c5'),123.5)
            self.assertEqual(86400-identity.role_debit({'output_dir':root},'r5'),86280)

    def test_checkpoint_rejects_old_flavor_and_out_of_horizon_epoch(self):
        worker.validate_checkpoint(*self.fixture())
        for key,value in [('training_flavor',f.FLAVOR),('epoch',41),('role','r5')]:
            args=list(self.fixture());args[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_checkpoint(*args)

    def test_checkpoint_rejects_missing_successful_updates_and_changed_stream(self):
        for key,value in [('optimizer_steps',2756),('actual_streams_verified',False),('source_draws',11024)]:
            args=list(self.fixture());args[0]['history'][0]['train'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_checkpoint(*args)

    def test_checkpoint_selection_requires_first_clean_f1_maximum(self):
        args=list(self.fixture());cp=args[0]
        second=copy.deepcopy(cp['history'][0]);second['epoch']=2;second['train']['epoch']=2
        cp['history'].append(second);cp['epoch']=2;cp['scheduler']['last_epoch']=2
        args[3].append(dict(args[3][0],epoch=2))
        worker.validate_checkpoint(*args)
        cp['best_epoch']=2
        with self.assertRaises(ValueError):worker.validate_checkpoint(*args)

    def test_cosine_checkpoint_requires_shorter_schedule_and_exact_epoch(self):
        for schedule in [{'T_max':100,'last_epoch':1},{'T_max':40,'last_epoch':0}]:
            args=list(self.fixture());args[0]['scheduler']=schedule
            with self.subTest(schedule=schedule),self.assertRaises(ValueError):worker.validate_checkpoint(*args)
        model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=1e-6)
        for _ in range(40):opt.step();scheduler.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
        self.assertEqual(scheduler.state_dict()['T_max'],40)

    def test_admission_requires_the_complete_passing_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tests.log'
            for text in ['Ran 387 tests in 1.0s\n\nOK','Ran 397 tests in 1.0s\n\nFAILED (failures=1)']:
                path.write_text(text,encoding='utf-8')
                with self.assertRaises(ValueError):passing_test_count(path,397)
            path.write_text('Ran 397 tests in 1.0s\n\nOK\n',encoding='utf-8')
            self.assertEqual(passing_test_count(path,397),397)


if __name__=='__main__':unittest.main()
