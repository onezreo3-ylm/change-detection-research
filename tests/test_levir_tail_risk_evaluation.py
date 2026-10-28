import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pcdc.config import protocol_signature
from pcdc.evaluate_corruptions import COUNT_KEYS
from scripts.levir_tail_risk_control import load_study,FLAVOR
from scripts.levir_tail_risk_evaluation import (ROOT,bootstrap_tail,check_finished_history,check_risk_identity,
    f1,gate_results,grouped_counts,model_scores)
from scripts.levir_tail_risk_eval_owner import reserve


class TailRiskEvaluationTests(unittest.TestCase):
    def identity(self):
        _,study,ref,_=load_study(ROOT/'configs/levir_tail_risk_control_20261025_iofix.yaml')
        authority={'risk_training_signature':'unique_risk_signature'}
        cp={'format_version':1,'training_flavor':FLAVOR,'kind':'research','training_signature':authority['risk_training_signature'],
            'config':copy.deepcopy(ref.base.values),'risk_config':copy.deepcopy(study),'reference_robust_config':dict(ref.values),
            'protocol_signature':protocol_signature(ref.base.values),'initial_model_tensor_sha256':study['candidate']['initialization_sha256'],
            'corruption_protocol_sha256':ref.protocol.fingerprint,'file_sha256':{'train':{},'val':{}}}
        return cp,authority,ref,study

    def test_accepts_explicit_actual_risk_identity(self):
        check_risk_identity(*self.identity())

    def test_original_matched_flavor_cannot_be_relabelled_through_adapter(self):
        cp,a,r,s=self.identity();cp['training_flavor']='matched_robust_baseline_v1'
        with self.assertRaises(ValueError):check_risk_identity(cp,a,r,s)

    def test_changed_q_or_precision_is_rejected(self):
        cp,a,r,s=self.identity();cp['risk_config']['candidate']['loss']['upper_tail_fraction']=.75
        with self.assertRaises(ValueError):check_risk_identity(cp,a,r,s)
        cp,a,r,s=self.identity();cp['config']['train']['amp']=False
        with self.assertRaises(ValueError):check_risk_identity(cp,a,r,s)

    def test_held_out_split_or_wrong_signature_is_rejected(self):
        for change in ['test','signature']:
            cp,a,r,s=self.identity()
            if change=='test':cp['file_sha256']['test']={}
            else:cp['training_signature']='wrong'
            with self.assertRaises(ValueError):check_risk_identity(cp,a,r,s)

    def histories(self):
        rows=[{'epoch':e,'learning_rate':.001,'train':{'optimizer_steps':1780,'amp_skipped_steps':0,
            'source_samples':7120,'pair_views':14240,'exposure_stream_sha256':str(e),'exposure_counts':{'noise':7120}},
            'val':{'metrics':{'f1':.9 if e==65 else .8}}} for e in range(1,101)]
        last={'epoch':100,'history':rows,'best_epoch':65,'best_val_f1':.9}
        best={'epoch':65,'history':copy.deepcopy(rows[:65]),'best_epoch':65,'best_val_f1':.9}
        status={'status':'completed_training_requires_explicit_validation','epoch':100,'best_epoch':65,'best_val_f1':.9,
                'optimizer_steps':178000,'amp_skipped_steps':0,'pair_views':1424000,'test_evaluated':False}
        return best,last,status,copy.deepcopy(rows)

    def test_completed_history_and_original_exposures_required(self):
        data=self.histories();self.assertEqual(check_finished_history(*data)['attempts'],178000)
        data=self.histories();data[3][17]['train']['exposure_stream_sha256']='changed'
        with self.assertRaises(ValueError):check_finished_history(*data)

    def test_incomplete_reference_or_nonselected_best_rejected(self):
        best,last,status,ref=self.histories()
        with self.assertRaises(ValueError):check_finished_history(best,last,status,ref[:-1])
        best['best_epoch']=64
        with self.assertRaises(ValueError):check_finished_history(best,last,status,ref)

    def toy(self):
        rng=np.random.default_rng(120)
        tp=rng.integers(10,100,size=(2,7,4));fp=rng.integers(0,100,size=(2,7,4))
        return np.stack((900-fp,fp,100-tp,tp),axis=-1)

    def test_bootstrap_matches_direct_shared_source_draws_and_resorts_tails(self):
        counts=self.toy();seed=19;n=40;k=2
        actual=bootstrap_tail(counts,replicates=n,seed=seed,k=k,chunk=11)
        indices=np.random.default_rng(seed).integers(0,4,size=(n,4))
        expected=[];wrong_fixed=[]
        observed_order=np.argsort(f1(counts.sum(2)),axis=1)[:,:k]
        for draw in indices:
            scores=f1(counts[:,:,draw,:].sum(2))
            tails=np.sort(scores,axis=1)[:,:k].mean(1)
            expected.append(tails[1]-tails[0])
            fixed=[scores[m,observed_order[m]].mean() for m in range(2)]
            wrong_fixed.append(fixed[1]-fixed[0])
        np.testing.assert_array_equal(actual,expected)
        self.assertFalse(np.allclose(actual,wrong_fixed))

    def test_bootstrap_chunk_does_not_change_rng_draws(self):
        a=bootstrap_tail(self.toy(),replicates=61,seed=13,k=5,chunk=1)
        b=bootstrap_tail(self.toy(),replicates=61,seed=13,k=5,chunk=50)
        np.testing.assert_array_equal(a,b)

    def cells(self,array):
        result={}
        for i,values in enumerate(array):
            result[str(i)]={'counts':dict(zip(COUNT_KEYS,values.sum(0).tolist())),
                'by_source':{f'val/{s}.png':{'counts':dict(zip(COUNT_KEYS,row.tolist()))} for s,row in enumerate(values)}}
        return result

    def test_pair_coverage_pooled_counts_and_label_support_are_checked(self):
        a,b=self.toy();candidate,baseline=self.cells(a),self.cells(b);ids=list(candidate)
        pooled,sources=grouped_counts(candidate,baseline,ids,4)
        self.assertEqual(pooled.shape,(2,7,4,4));self.assertEqual(len(sources),4)
        candidate['0']['by_source'].pop('val/0.png')
        with self.assertRaises(ValueError):grouped_counts(candidate,baseline,ids,4)
        candidate=self.cells(a);candidate['0']['counts']['tp']+=1
        with self.assertRaises(ValueError):grouped_counts(candidate,baseline,ids,4)
        candidate=self.cells(a);candidate['0']['by_source']['val/0.png']['counts']['tp']+=1;candidate['0']['counts']['tp']+=1
        with self.assertRaises(ValueError):grouped_counts(candidate,baseline,ids,4)

    def test_zero_foreground_is_finite_and_tail_is_computed_per_model(self):
        np.testing.assert_array_equal(f1(np.array([[10,0,0,0]])),[0.])
        cells={str(i):{'counts':{'tn':10,'fp':0,'fn':0,'tp':0}} for i in range(5)}
        scores=model_scores(cells,list(cells),{'metrics':{'f1':0.}})
        self.assertEqual(scores['pooled_recall'],0.)
        self.assertEqual(scores['worst_five_f1'],0.)

    def test_any_guard_failure_rejects_even_if_bootstrap_positive(self):
        base=dict(clean_f1=.9,mean_f1=.89,worst_five_f1=.87,worst_single_f1=.86,pooled_recall=.9,pooled_background_fpr=.003,worst_five_conditions=[])
        candidate=dict(base);candidate.update(clean_f1=.899,worst_five_f1=.875,worst_single_f1=.865)
        rules={'clean_f1_min_delta':-.0005,'mean_corruption_f1_min_delta':-.0005,'pooled_corruption_recall_min_delta':-.001,
               'pooled_background_fpr_max_delta':.0001,'worst_five_mean_f1_min_delta':.002,'worst_single_condition_f1_min_delta':.002}
        result=gate_results(candidate,base,rules,.001)
        self.assertFalse(result['all_passed']);self.assertFalse(result['gates']['clean_f1'])
        candidate['clean_f1']=.9
        self.assertFalse(gate_results(candidate,base,rules,0.)['all_passed'])

    def test_evaluation_keeps_existing_training_charge_in_same_candidate_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'ledger.json'
            p.write_text(json.dumps({'entries':[{'kind':'tail_risk_train_whole_worker_process','charged_seconds':29063.078}],
                'total_charged_seconds':29602.4456,'batch_upper_bound_seconds':172800}))
            _,limit=reserve({'ledger_path':str(p)},{'evaluation_wall_limit_seconds':100000})
            self.assertAlmostEqual(limit,86400-29063.078)
            with self.assertRaises(RuntimeError):reserve({'ledger_path':str(p)},{'evaluation_wall_limit_seconds':7200})


if __name__=='__main__':unittest.main()
