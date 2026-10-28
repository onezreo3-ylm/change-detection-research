"""CPU-only checks of scientific effect gates and paired-bootstrap semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_b0e_replication as analysis


class CompleteControlAnalysisTests(unittest.TestCase):
    def fixture(self):
        rows={s:{m:dict(corruption_mf1=.8,clean_f1=.9,pooled_recall=.85,background_fpr=.004)
                 for m in analysis.MODES} for s in analysis.SEEDS}
        for seed in rows:rows[seed]['b1']['corruption_mf1']+=.01
        return rows,analysis.load_config()['rules']

    def test_positive_mean_cannot_override_clean_or_recall_harm(self):
        rows,rules=self.fixture()
        self.assertTrue(analysis.primary_gate(rows,[.001,.01,.02],rules)['all_passed'])
        for metric in ('clean_f1','pooled_recall'):
            damaged=copy.deepcopy(rows)
            for seed in damaged:damaged[seed]['b1'][metric]-=.01
            result=analysis.primary_gate(damaged,[.001,.01,.02],rules)
            self.assertTrue(result['checks']['mean_mf1_gain'])
            self.assertFalse(result['all_passed'])

    def test_positive_average_does_not_override_insufficient_seed_wins(self):
        rows,rules=self.fixture()
        for seed in (43,44):rows[seed]['b1']['corruption_mf1']=.799
        rows[42]['b1']['corruption_mf1']=.84
        result=analysis.primary_gate(rows,[.001,.01,.02],rules)
        self.assertTrue(result['checks']['mean_mf1_gain'])
        self.assertFalse(result['checks']['positive_seed_count'])
        self.assertFalse(result['all_passed'])

    def test_zero_interval_lower_is_not_positive(self):
        rows,rules=self.fixture()
        result=analysis.primary_gate(rows,[0,.01,.02],rules)
        self.assertFalse(result['checks']['source_paired_lower_positive'])

    def test_background_protection_and_finite_interval(self):
        rows,rules=self.fixture()
        for seed in rows:rows[seed]['b1']['background_fpr']+=.0002
        self.assertFalse(analysis.primary_gate(rows,[.001,.01,.02],rules)['checks']['background_fpr_protection'])
        for bad in ([.1,0,.2],[0,float('nan'),.2],[0,.1]):
            with self.assertRaises(ValueError):analysis.primary_gate(rows,bad,rules)

    def test_missing_seed_cannot_become_a_partial_primary_result(self):
        rows,rules=self.fixture();del rows[44]
        with self.assertRaises(ValueError):analysis.primary_gate(rows,[.001,.01,.02],rules)
        with self.assertRaisesRegex(ValueError,'Missing required'):
            analysis.load_records({'runs':{}},complete=True)

    def test_pixel_and_semantic_support_cannot_change_between_conditions(self):
        counts=np.zeros((64,45,4),dtype=np.int64);counts[:,:,0]=1024**2
        counts[:,:,0]-=10;counts[:,:,2]=10
        reference=analysis.validate_source_support({'by_source':counts})
        self.assertEqual(reference.tolist(),[10]*64)
        invalid=counts.copy();invalid[5,17,0]-=1
        with self.assertRaises(ValueError):analysis.validate_source_support({'by_source':invalid})
        invalid=counts.copy();invalid[5,17,0]-=1;invalid[5,17,2]+=1
        with self.assertRaisesRegex(ValueError,'label support'):
            analysis.validate_source_support({'by_source':invalid})

    def test_methods_must_share_actual_source_label_support(self):
        counts=np.zeros((64,45,4),dtype=np.int64);counts[:,:,0]=1024**2
        reference=analysis.validate_source_support({'by_source':counts})
        counts[5,:,0]-=1;counts[5,:,2]+=1
        with self.assertRaisesRegex(ValueError,'same source label support'):
            analysis.validate_source_support({'by_source':counts},reference)

    def test_original_bootstrap_uses_shared_sources_and_nonlinear_pooled_f1(self):
        # Same source counts across methods must produce exactly zero even when
        # source prevalence and sample F1 differ substantially.
        counts=np.zeros((4,3,2,2,4),dtype=np.int64)
        base=np.array([[100,0,0,0],[80,0,4,16],[0,10,0,10],[20,1,7,2]])
        for source in range(4):counts[source,:,:,:,:]=base[source]
        self.assertEqual(analysis.original.paired_bootstrap(counts,replicates=257,seed=1),[0.,0.,0.])
        # Candidate with additional TP and fewer FN is monotonically better.
        counts[:,:,1,:,3]+=1;counts[:,:,1,:,2]=np.maximum(0,counts[:,:,1,:,2]-1)
        first=analysis.original.paired_bootstrap(counts,replicates=257,seed=1)
        self.assertGreater(first[0],0)
        self.assertEqual(first,analysis.original.paired_bootstrap(counts,replicates=257,seed=1))

    def test_certification_header_cannot_replace_actual_successful_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);allowed=root/'runs/eval';session=allowed/'evaluation_session';session.mkdir(parents=True)
            outputs=allowed/'seed43';outputs.mkdir()
            roles={}
            for role,state in [('owner',{'status':'completed','worker_exit_code':0}),
                               ('worker',{'status':'completed','test_accessed':False}),
                               ('guard',{'status':'worker_exited','maximum_observed_temperature_c':75}),
                               ('watchdog',{'status':'worker_exited'})]:
                p=session/(role+'_state.json' if role!='worker' else 'worker_result.json')
                p.write_text(json.dumps(state),encoding='utf-8');roles[role]=str(p)
            for n in ('summary.json','conditions.json','clean_metrics.json'):(outputs/n).write_text('{}',encoding='utf-8')
            paths=[*map(Path,roles.values()),*outputs.glob('*.json')]
            cert=dict(status='completed_independent_external_supervision_audit',seed=43,test_accessed=False,
                      roles=roles,bound_sha256={str(p):analysis.sha(p) for p in paths})
            p=outputs/'evaluation_completion.json';p.write_text(json.dumps(cert),encoding='utf-8')
            with patch.object(analysis,'ROOT',root),patch.object(analysis,'load_config',return_value={'new_evaluation_root':'runs/eval'}):
                analysis.check_new_completion(p,analysis.sha(p),43)
                # Even a freshly hash-consistent certificate must reject failed
                # actual owner evidence, rather than trusting its success header.
                owner=Path(roles['owner']);owner.write_text(json.dumps({'status':'failed','worker_exit_code':1}),encoding='utf-8')
                cert['bound_sha256'][str(owner)]=analysis.sha(owner);p.write_text(json.dumps(cert),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'did not complete'):
                    analysis.check_new_completion(p,analysis.sha(p),43)


if __name__=='__main__':unittest.main()
