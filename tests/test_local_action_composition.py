import unittest

import torch

from pcdc.local_action_composition import compare_joint, compose_disjoint_outputs, confusion_vector


class LocalCompositionTests(unittest.TestCase):
    def fixture(self):
        base = -torch.ones(1,1,5,5)
        supports = [torch.zeros_like(base,dtype=torch.bool) for _ in range(2)]
        supports[0][:,:,0:2,0:2]=True; supports[1][:,:,3:5,3:5]=True
        singles=[torch.where(s,torch.ones_like(base),base) for s in supports]
        labels=torch.zeros_like(base); labels[:,:,0:2,0:2]=1
        return base,singles,supports,labels

    def test_disjoint_actual_joint_outputs_have_exact_counterfactual_count_sum(self):
        base,singles,supports,labels=self.fixture()
        joint=compose_disjoint_outputs(base,singles,supports)
        result=compare_joint(base,singles,joint,supports,labels)
        self.assertTrue(result['joint_equals_pasted_individuals'])
        self.assertTrue(result['count_increment_additivity'])
        self.assertEqual(result['joint_counts'],[17,4,0,4])
        self.assertEqual(result['hard_prediction_changes'],[4,4])

    def test_overlap_is_rejected_even_when_output_values_happen_to_match(self):
        base,singles,supports,_=self.fixture()
        supports[1]=supports[0].clone()
        with self.assertRaises(ValueError):compose_disjoint_outputs(base,singles,supports)

    def test_outside_preservation_alone_does_not_certify_composition(self):
        base,singles,supports,labels=self.fixture()
        joint=compose_disjoint_outputs(base,singles,supports);joint[:,:,0,0]=-2
        result=compare_joint(base,singles,joint,supports,labels)
        self.assertEqual(result['outside_joint_changed_positions'],0)
        self.assertFalse(result['joint_equals_pasted_individuals'])
        self.assertFalse(result['count_increment_additivity'])

    def test_empty_mismatched_nonfinite_or_nonbinary_inputs_fail_closed(self):
        base,singles,supports,labels=self.fixture()
        with self.assertRaises(ValueError):compose_disjoint_outputs(base,[],[])
        with self.assertRaises(ValueError):compose_disjoint_outputs(base,singles,[supports[0]])
        bad=singles[0].clone();bad[0,0,0,0]=float('inf')
        with self.assertRaises(ValueError):compose_disjoint_outputs(base,[bad,singles[1]],supports)
        with self.assertRaises(ValueError):confusion_vector(base,labels+2)

    def test_zero_action_and_action_order_keep_identical_outputs_and_counts(self):
        base,singles,supports,labels=self.fixture()
        normal=compose_disjoint_outputs(base,singles,supports)
        reversed_order=compose_disjoint_outputs(base,singles[::-1],supports[::-1])
        self.assertTrue(torch.equal(normal,reversed_order))
        self.assertTrue(torch.equal(compose_disjoint_outputs(base,[base,base],supports),base))
        self.assertEqual(int(confusion_vector(normal,labels).sum()),25)


if __name__=='__main__':unittest.main()
