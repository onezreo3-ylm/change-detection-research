import itertools
import unittest
from unittest.mock import patch

import torch

from pcdc.bound_local_actions import ObservedBatchIdentity,ReferenceBinding
from pcdc.bound_temporal_actions import BoundTemporalActionSession
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector


class BoundTemporalActionTests(unittest.TestCase):
    def test_failure_after_parent_capture_keeps_original_error_and_removes_hooks(self):
        m,a,b,i=self.fixture()
        with patch.object(BoundTemporalActionSession,'_variant_value_hashes',side_effect=RuntimeError('injected variant hash failure')):
            with self.assertRaisesRegex(RuntimeError,'injected variant hash failure'):
                BoundTemporalActionSession(m,a,b,i,amp=False,engineering_only=True)
        self.assertTrue(all(not mod._forward_hooks for mod in m.modules()))

    def fixture(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            model=AdapterCorrectedDetector(SiameseChangeDetector(32,8)).eval().requires_grad_(False)
            torch.nn.init.normal_(model.adapter.conv3.weight,std=.01)
            a,b=torch.rand(1,3,128,128),torch.rand(1,3,128,128)
        identity=ObservedBatchIdentity(('train/explicit_synthetic@x0_y0',),(('a'*64,'b'*64),),('c'*64,),'d'*64,'e'*64,'f'*64)
        return model,a,b,identity

    def session(self):
        m,a,b,i=self.fixture()
        return BoundTemporalActionSession(m,a,b,i,amp=False,engineering_only=True)

    def register_two_regions(self,s):
        for region,(r0,r1) in enumerate([(1,4),(12,15)]):
            mask=torch.zeros(1,1,16,16,dtype=torch.bool);mask[:,:,r0:r1,r0:r1]=True
            for kind in s.KINDS:s.register(f'{region}:{kind}',kind,mask,binding=s.binding)

    def test_both_full_and_noop_keep_the_original_outputs_and_forward_counts(self):
        m,a,b,i=self.fixture()
        with torch.no_grad():old0,old1=m.action_logits(a,b)
        s=BoundTemporalActionSession(m,a,b,i,amp=False,engineering_only=True)
        with s:
            s.register('full','both',torch.ones(1,1,16,16,dtype=torch.bool),binding=s.binding)
            self.assertTrue(torch.equal(old0,s.baseline(binding=s.binding)))
            self.assertTrue(torch.equal(old1,s.decode(('full',),binding=s.binding,arm='ordinary_GN')))
            self.assertTrue(torch.equal(old0,s.decode((),binding=s.binding,arm='reference_replay')))
            self.assertEqual(s._calls,dict(encoder=1,adapter=2,decoder=3))

    def test_all_sixteen_mixed_choices_match_three_actual_all_placement_outputs(self):
        s=self.session()
        with s:
            self.register_two_regions(s);base=s.baseline(binding=s.binding)
            alls={k:s.decode((f'0:{k}',f'1:{k}'),binding=s.binding,arm='reference_replay') for k in s.KINDS}
            supports=[s.support(f'{r}:both',binding=s.binding) for r in range(2)]
            for choices in itertools.product((None,*s.KINDS),repeat=2):
                names=tuple(f'{r}:{k}' for r,k in enumerate(choices) if k)
                actual=s.decode(names,binding=s.binding,arm='reference_replay');expected=base.clone()
                for support,k in zip(supports,choices):
                    if k:expected=torch.where(support,alls[k],expected)
                self.assertTrue(torch.equal(actual,expected))
            self.assertFalse(torch.equal(alls['A_only'],alls['both']))

    def test_A_B_exchange_correspondence_uses_separate_actual_reference_sessions(self):
        m,a,b,i=self.fixture();left=BoundTemporalActionSession(m,a,b,i,amp=False,engineering_only=True)
        with left:
            for k in left.KINDS:left.register(k,k,torch.ones(1,1,16,16,dtype=torch.bool),binding=left.binding)
            values={k:left.decode((k,),binding=left.binding,arm='reference_replay') for k in left.KINDS}
            base=left.baseline(binding=left.binding)
        right=BoundTemporalActionSession(m,b,a,i,amp=False,engineering_only=True)
        with right:
            for k in right.KINDS:right.register(k,k,torch.ones(1,1,16,16,dtype=torch.bool),binding=right.binding)
            swap={k:right.decode((k,),binding=right.binding,arm='reference_replay') for k in right.KINDS}
            self.assertTrue(torch.equal(base,right.baseline(binding=right.binding)))
            self.assertTrue(torch.equal(values['A_only'],swap['B_only']))
            self.assertTrue(torch.equal(values['B_only'],swap['A_only']))
            self.assertTrue(torch.equal(values['both'],swap['both']))

    def test_conflicting_placements_unknown_kind_and_foreign_binding_are_rejected(self):
        s=self.session()
        with s:
            self.register_two_regions(s)
            with self.assertRaises(ValueError):s.decode(('0:A_only','0:B_only'),binding=s.binding,arm='reference_replay')
            with self.assertRaises(ValueError):s.register('bad','unknown',torch.ones(1,1,16,16,dtype=torch.bool),binding=s.binding)
            with self.assertRaises(ValueError):s.baseline(binding=ReferenceBinding('wrong',s.binding.evidence_sha256))

    def test_data_write_bypassing_version_is_caught_and_all_caches_are_freed(self):
        s=self.session();s._variants['A_only'].data.add_(1)
        with self.assertRaisesRegex(ValueError,'temporal-variant values'):s.close()
        self.assertTrue(s._closed);self.assertFalse(s._variants);self.assertFalse(s._base_features)
        self.assertFalse(s._replay.frames)

    def test_placement_map_and_variant_versions_cannot_be_mutated(self):
        s=self.session();self.register_two_regions(s)
        s._which['0:A_only']='B_only'
        with self.assertRaises(ValueError):s.decode(('0:A_only',),binding=s.binding,arm='reference_replay')
        with self.assertRaises(ValueError):s.close()
        s=self.session();s._variants['B_only'].add_(1)
        with self.assertRaises(ValueError):s.baseline(binding=s.binding)
        with self.assertRaises(ValueError):s.close()

    def test_external_hooks_or_invalid_constructor_do_not_leak_capture_hooks(self):
        m,a,b,i=self.fixture();h=m.adapter.register_forward_hook(lambda mod,args,out:out)
        with self.assertRaises(ValueError):BoundTemporalActionSession(m,a,b,i,amp=False,engineering_only=True)
        h.remove()
        with self.assertRaises(ValueError):BoundTemporalActionSession(m,a,b,i,amp=True,engineering_only=True)
        self.assertFalse(m.adapter._forward_hooks)


if __name__=='__main__':unittest.main()
