"""Five-view identity, actual-input, independent-RNG and admission controls."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from pcdc.config import Config,load_config,protocol_signature
from pcdc.data import file_sha256
from pcdc.losses import ChangeDetectionLoss
from scripts import five_view_baselines as f
from scripts.valais_c5r5_study import estimate_budget


def base_values():
    return copy.deepcopy(load_config(f.ROOT/'configs/valais_clean_full_fp32_seed42_20261002.yaml').values)


def settings():
    _,v,_,_=f.load_study(f.ROOT/'configs/valais_c5r5_study_v1_20261020.yaml')
    return v


def corruptions():
    path=f.ROOT/'configs/valais_corruption_protocol_v1_20261002.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))['corruptions']


class FiveViewStreamTests(unittest.TestCase):
    def test_uniform_epoch_uses_each_source_once_preserves_tail_and_global_rng(self):
        names=[f'{i}.tif' for i in range(13)]
        random.seed(81);torch.manual_seed(81);np.random.seed(81)
        py,tr,npstate=random.getstate(),torch.get_rng_state().clone(),np.random.get_state()
        sampler=f.UniformEpochSampler(names,settings()['source_seed_template'])
        a=sampler.batches();b=sampler.batches()
        self.assertEqual(a,b);self.assertEqual(len(a[-1]),1)
        self.assertEqual(sorted(i for batch in a for i in batch),list(range(13)))
        sampler.set_epoch(2);self.assertNotEqual(a,sampler.batches())
        instance=f.instance_for('train/1.tif@x0_y0',2,settings(),corruptions())
        self.assertEqual(instance,f.instance_for('train/1.tif@x0_y0',2,settings(),corruptions()))
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))

    def test_pair_distribution_stays_in_existing_grid_and_retains_duplicate_draws(self):
        s,c=settings(),corruptions();duplicates=0
        for i in range(64):
            x=f.instance_for(f'train/{i}.tif@x0_y0',1,s,c)
            self.assertIn(x['family'],f.FAMILIES);self.assertIn(x['level'],f.LEVELS)
            for v in ('u','v'):
                self.assertEqual(x[v]['parameters'],c[x['family']][x[v]['direction']][x['level']-1])
            duplicates+=x['u']['direction']==x['v']['direction']
        self.assertGreater(duplicates,0);self.assertLess(duplicates,64)

    def test_nontrain_or_out_of_budget_instances_and_duplicate_sources_are_rejected(self):
        for sid,epoch in [('val/1.tif',1),('test/1.tif',1),('train/1.tif',0),('train/1.tif',101)]:
            with self.assertRaises(ValueError):f.instance_for(sid,epoch,settings(),corruptions())
        with self.assertRaises(ValueError):f.UniformEpochSampler(['a','a'],'{epoch}')


class FiveViewActualInputTests(unittest.TestCase):
    def fixture(self,root):
        values=base_values();values['data']['root']=str(root);values['experiment']['kind']='smoke'
        for folder in ('A','B','label'):(root/'train'/folder).mkdir(parents=True)
        raw=np.stack(np.meshgrid(np.arange(256,dtype=np.uint8),np.arange(256,dtype=np.uint8)),axis=-1)
        a=np.concatenate([raw,raw[:,:,:1]],axis=-1);b=255-a
        mask=np.zeros((256,256),dtype=np.uint8);mask[60:90,70:100]=1
        for folder,image in [('A',a),('B',b),('label',mask)]:Image.fromarray(image).save(root/'train'/folder/'1.tif')
        hashes={k:file_sha256(root/'train'/folder/'1.tif') for k,folder in [('image_a','A'),('image_b','B'),('mask','label')]}
        return {'base_values':values,'base_config_source':str(root/'base.yaml'),'data_root':str(root),
            'audit':{'splits':{'train':{'file_sha256':{'1.tif':hashes}}}},'positive_pixels':{'1.tif':900}}

    def test_c5_and_r5_share_candidate_pixels_but_have_correct_distinct_actual_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.fixture(Path(tmp));c=f.FiveViewDataset(m,settings(),corruptions(),'c5')[0]
            r=f.FiveViewDataset(m,settings(),corruptions(),'r5')[0]
            self.assertEqual(c['candidate_pixels'],r['candidate_pixels'])
            self.assertEqual(c['instance_json'],r['instance_json'])
            self.assertTrue(torch.equal(c['mask'],r['mask']))
            self.assertTrue(all(torch.equal(c['a'][0],x) for x in c['a']))
            self.assertTrue(torch.equal(c['a'][0],r['a'][0]))
            self.assertNotEqual(c['used_pixels'],r['used_pixels'])
            used=json.loads(r['used_pixels']);candidate=json.loads(r['candidate_pixels'])
            self.assertEqual(used[1],[candidate['au'],candidate['bu']])
            self.assertEqual(used[2],[candidate['au'],candidate['bv']])
            self.assertEqual(used[3],[candidate['av'],candidate['bu']])
            self.assertEqual(used[4],[candidate['av'],candidate['bv']])

    def test_changed_encoded_image_label_count_or_nontrain_path_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            bad=copy.deepcopy(m);bad['positive_pixels']['1.tif']=1
            with self.assertRaises(ValueError):f.FiveViewDataset(bad,settings(),corruptions(),'r5')[0]
            bad=copy.deepcopy(m);entry=bad['audit']['splits']['train']['file_sha256'].pop('1.tif')
            bad['audit']['splits']['train']['file_sha256']['../test/1.tif']=entry
            with self.assertRaises(ValueError):f.FiveViewDataset(bad,settings(),corruptions(),'c5')
            (root/'train'/'A'/'1.tif').write_bytes(b'changed')
            with self.assertRaises(ValueError):f.FiveViewDataset(m,settings(),corruptions(),'r5')[0]

    def test_synthetic_marker_is_explicitly_rejected_for_a_research_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root);(root/'SYNTHETIC_ONLY.json').write_text('{}')
            m['base_values']['experiment']['kind']='research'
            with self.assertRaises(ValueError):f.FiveViewDataset(m,settings(),corruptions(),'r5')


class FiveViewObjectiveAndBudgetTests(unittest.TestCase):
    def test_exact_view_weighting_and_gradients_and_wrong_view_count(self):
        torch.manual_seed(9)
        z=torch.randn(10,1,2,2,requires_grad=True);target=torch.randint(0,2,(2,1,2,2)).float()
        loss=ChangeDetectionLoss(1,1,1)
        actual=f.weighted_loss(z,target,loss)
        for key in ('total','bce','dice'):
            expected=sum(w*loss(z[2*i:2*i+2],target)[key] for i,w in enumerate([.5,.125,.125,.125,.125]))
            torch.testing.assert_close(actual[key],expected,rtol=1e-6,atol=1e-7)
        actual['total'].backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue((z.grad.flatten(1).abs().sum(1)>0).all())
        with self.assertRaises(ValueError):f.weighted_loss(z[:8],target,loss)

    def test_projection_respects_user_24_hour_cap_and_does_not_reduce_epochs(self):
        s=settings();self.assertEqual(s['maximum_training_seconds_per_role'],86400)
        self.assertTrue(estimate_budget([.2]*24,s)['within_frozen_budget'])
        self.assertFalse(estimate_budget([.4]*24,s)['within_frozen_budget'])
        self.assertEqual(s['epochs'],100)
        with self.assertRaises(ValueError):estimate_budget([float('nan')],s)
        close=(86400-6000-10)/(275700*1.25)
        self.assertTrue(estimate_budget([close]*24,s)['within_frozen_budget'])
        self.assertFalse(estimate_budget([close]*24,s,pilot_seconds=20)['within_frozen_budget'])

    def test_single_source_tail_still_has_five_views_and_one_joint_bn_update(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.head=torch.nn.Conv2d(3,1,1)
            def forward(self,a,b):
                both=self.bn(torch.cat([a,b]));n=len(a)
                return {'logits':self.head((both[:n]-both[n:]).abs())}
        model=Toy().train();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        batch={'a':torch.rand(1,5,3,256,256),'b':torch.rand(1,5,3,256,256),'mask':torch.zeros(1,1,256,256)}
        losses,norm,logits,target=f.update(model,opt,ChangeDetectionLoss(1,1,1),batch,torch.device('cpu'),1.)
        self.assertEqual(logits.shape,(5,1,256,256));self.assertEqual(int(model.bn.num_batches_tracked),1)
        self.assertTrue(np.isfinite(norm));self.assertTrue(np.isfinite(losses['total']))


class FiveViewCheckpointTests(unittest.TestCase):
    def fixture(self):
        cfg=Config(base_values(),Path('base.yaml'))
        files={'train':{},'val':{}};manifests={'train':{},'val':{}}
        identity={'role':'c5','implementation_sha256':{},'initial_model_tensor_sha256':'initial',
            'data_files_sha256':f.digest(files),'manifests_sha256':f.digest(manifests)}
        plan=[{'epoch':1,'source_draws':11025,'pair_views':55125}]
        history=[{'epoch':1,'val':{'metrics':{'f1':.2}},'train':dict(plan[0],optimizer_steps=2757,amp_skipped_steps=0,actual_streams_verified=True)}]
        cp={'format_version':1,'kind':'research','training_flavor':f.FLAVOR,'role':'c5','identity':identity,
            'protocol_signature':protocol_signature(cfg.values),'config':cfg.values,'training_signature':f.digest(identity),
            'source_sha256':{},'initial_model_tensor_sha256':'initial','file_sha256':files,'manifests':manifests,
            'history':history,'epoch':1,'best_epoch':1,'best_val_f1':.2,'model':{'w':torch.ones(1)},'optimizer':{'state':{}}}
        return cp,cfg,identity,plan

    def test_original_flavor_source_selection_and_successful_budget_are_required(self):
        args=self.fixture();f.validate_checkpoint(*args)
        for field,value in [('training_flavor','stratified_two_view_baseline_v1'),('role','r5'),
            ('source_sha256',{'changed':'sha'}),('initial_model_tensor_sha256','other'),('file_sha256',{'train':{},'test':{}}),('best_epoch',2)]:
            a=list(self.fixture());a[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):f.validate_checkpoint(*a)
        a=list(self.fixture());a[0]['history'][0]['train']['optimizer_steps']=2756
        with self.assertRaises(ValueError):f.validate_checkpoint(*a)


if __name__=='__main__':unittest.main()
