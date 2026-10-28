"""CPU-only identity, support and protection checks for the pending val adapter."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_b0e_replications as evaluator
from scripts import fixed_development_evaluation_supervision as supervisor
from pcdc.evaluate_corruptions import counts_report


class FixedReplicationEvaluationTests(unittest.TestCase):
    def condition_fixture(self):
        counts=dict(tn=1024**2-6,fp=2,fn=3,tp=1)
        sources={f'source{i}.png':dict(samples=16,counts=copy.deepcopy(counts)) for i in range(64)}
        pooled={k:64*v for k,v in counts.items()};metrics=counts_report([pooled[k] for k in evaluator.COUNT_KEYS])['metrics']
        cid='brightness_l1';row=dict(condition_id=cid,samples=1024,pixels=1024*256**2,
            counts=pooled,metrics=metrics,by_source=sources,joint_strata={'increase|A|global':{'samples':1024}})
        clean=copy.deepcopy(row)
        manifest={'stratum_counts':{cid+'|increase|A|global':1024}}
        return cid,row,clean,manifest

    def test_runtime_remains_one_fixed_original_gpu_validation_job(self):
        _,cfg,v=supervisor.load_runtime()
        self.assertEqual(v['whole_evaluation_limit_seconds'],3600)
        self.assertEqual(cfg['immediate_pause_c'],86)
        self.assertEqual(cfg['poll_seconds'],2)
        self.assertFalse(v['test_or_martigny_access'])
        self.assertEqual(supervisor.session_path(cfg,'evaluate').name,'evaluation_session')
        with self.assertRaises(ValueError):supervisor.session_path(cfg,'train')

    def test_condition_conserves_pixels_labels_strata_and_source_axis(self):
        cid,row,clean,manifest=self.condition_fixture()
        evaluator.check_condition(row,cid,clean,manifest)
        for mutate in ('source','strata','pixels','labels','arithmetic'):
            changed=copy.deepcopy(row)
            if mutate=='source':del changed['by_source']['source0.png']
            elif mutate=='strata':changed['joint_strata']['increase|A|global']['samples']-=1
            elif mutate=='pixels':changed['by_source']['source0.png']['counts']['tn']-=1
            elif mutate=='labels':
                changed['by_source']['source0.png']['counts']['tn']-=1
                changed['by_source']['source0.png']['counts']['fn']+=1
            else:changed['metrics']['f1']+=.01
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                evaluator.check_condition(changed,cid,clean,manifest)

    def test_clean_replay_requires_counts_and_all_five_metrics(self):
        _,row,_,_=self.condition_fixture()
        expected=dict(samples=row['samples'],metrics={**row['metrics'],**row['counts']})
        evaluator.check_replay(row,expected)
        changed=copy.deepcopy(row);changed['counts']['fp']+=1
        with self.assertRaises(ValueError):evaluator.check_replay(changed,expected)
        changed=copy.deepcopy(row);changed['metrics']['oa']-=1e-9
        with self.assertRaises(ValueError):evaluator.check_replay(changed,expected)

    def make_process_fixture(self,root):
        values=dict(owner=dict(status='completed',worker_exit_code=0),worker=dict(status='completed',test_accessed=False),
            guard=dict(status='worker_exited',maximum_observed_temperature_c=78),watchdog=dict(status='worker_exited'))
        for role,value in values.items():
            p=root/(role+'_state.json' if role!='worker' else 'worker_result.json');p.write_text(json.dumps(value),encoding='utf-8')
        proof=dict(mode='train',runtime_sha256=evaluator.file_sha256(evaluator.ROOT/'configs/b0e_replication_runtime_20261026.yaml'),
            adapter_sha256=evaluator.file_sha256(evaluator.ROOT/'scripts/b0e_replication_supervision.py'),
            source_sha256=evaluator.file_sha256(evaluator.common.__file__))
        for i,role in enumerate(values):
            proof[role]=dict(process_id=100+i,creation_time_ticks=10000+i,
                executable='E:/Download/anaconda3/envs/geosa-basa/python.exe',
                command_line=f'python -m scripts.b0e_replication_supervision {role} --mode train')
        (root/'attachment.json').write_text(json.dumps(proof),encoding='utf-8')
        return values

    def test_preparation_requires_real_finished_protection_and_inactive_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values=self.make_process_fixture(root)
            with patch.object(evaluator.common,'snapshot_identity',return_value={'active':False}):
                evaluator.verify_training_session(root)
            with patch.object(evaluator.common,'snapshot_identity',return_value={'active':True}):
                with self.assertRaisesRegex(RuntimeError,'still active'):evaluator.verify_training_session(root)
            values['owner']['status']='running';(root/'owner_state.json').write_text(json.dumps(values['owner']),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'must finish'):evaluator.verify_training_session(root,require_inactive=False)

    def test_success_header_does_not_override_guard_error_or_missing_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values=self.make_process_fixture(root)
            values['guard']['error']='sensor stop'
            (root/'guard_state.json').write_text(json.dumps(values['guard']),encoding='utf-8')
            with self.assertRaises(ValueError):evaluator.verify_training_session(root,require_inactive=False)
            del values['guard']['error'];del values['guard']['maximum_observed_temperature_c']
            (root/'guard_state.json').write_text(json.dumps(values['guard']),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'thermal'):evaluator.verify_training_session(root,require_inactive=False)

    def test_completed_execution_must_bind_actual_code_and_distinct_creation_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_process_fixture(root);p=root/'attachment.json';proof=json.loads(p.read_text())
            proof['runtime_sha256']='0'*64;p.write_text(json.dumps(proof),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'binding differs'):evaluator.verify_training_session(root,require_inactive=False)
            self.make_process_fixture(root);proof=json.loads(p.read_text());proof['guard']['process_id']=proof['worker']['process_id']
            p.write_text(json.dumps(proof),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'distinct'):evaluator.verify_training_session(root,require_inactive=False)

    def test_evaluation_reserve_subtracts_engineering_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=root/'ledger.json'
            ledger.write_text(json.dumps(dict(entries=[dict(id='historical',kind='other',status='completed',charged_seconds=90000),
                dict(id='cpu',kind='b0e_fixed_validation_engineering',status='completed',charged_seconds=120)],
                total_charged_seconds=90120,batch_upper_bound_seconds=172800)),encoding='utf-8')
            cfg=dict(ledger_path=str(ledger),study_output=tmp)
            identifier,limit=supervisor.reserve(cfg,{'whole_evaluation_limit_seconds':3600},'evaluate')
            self.assertEqual(limit,3360)
            supervisor.ledger_update(cfg,identifier,100,'completed',0)
            value=json.loads(ledger.read_text())
            self.assertEqual(value['entries'][0]['charged_seconds'],90000)
            self.assertEqual(value['total_charged_seconds'],90220)
            with self.assertRaises(RuntimeError):supervisor.reserve(cfg,{'whole_evaluation_limit_seconds':3600},'evaluate')

    def test_cpu_cost_settlement_does_not_reset_shared_or_reserve_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=root/'ledger.json'
            ledger.write_text(json.dumps(dict(entries=[dict(id='old',charged_seconds=50,status='completed')],total_charged_seconds=50)),encoding='utf-8')
            cfg=dict(study_output=tmp,ledger_path=str(ledger))
            with patch.object(supervisor,'load_runtime',return_value=(None,cfg,{})):
                supervisor.record_cost('prepare',3,'fixture measured CPU wall')
                supervisor.sync_costs(cfg);supervisor.sync_costs(cfg)
                self.assertEqual(json.loads(ledger.read_text())['total_charged_seconds'],53)
                with self.assertRaises(ValueError):supervisor.record_cost('prepare',3,'duplicate')

    def test_seed_checkpoint_must_keep_exact_config_initialization_and_both_data_identities(self):
        tv,refs,_=evaluator.training.load_study();ref=refs[43]
        audit={'splits':{s:dict(file_sha256={'fixture':'a'*64},samples=1) for s in ('train','val')}}
        initial=tv['runs']['43']['initialization_sha256']
        cp=dict(training_flavor='matched_robust_baseline_v1',robust_mode='b0e',config=copy.deepcopy(ref.base.values),
            robust_config=copy.deepcopy(dict(ref.values)),training_signature=ref.fingerprint,
            severity_review_sha256=evaluator.file_sha256(ref.paths['severity_review']),
            source_sha256=evaluator.robust_source_hashes(),initial_model_tensor_sha256=initial,
            file_sha256={s:audit['splits'][s]['file_sha256'] for s in ('train','val')},
            manifests={s:{'samples':1} for s in ('train','val')})
        evaluator.check_checkpoint_identity(cp,ref,initial,audit)
        for key in ('training_signature','initial_model_tensor_sha256','source_sha256','file_sha256'):
            changed=copy.deepcopy(cp);changed[key]={} if isinstance(cp[key],dict) else 'changed'
            with self.subTest(key=key),self.assertRaises(ValueError):evaluator.check_checkpoint_identity(changed,ref,initial,audit)

    def test_stopped_supervisor_aborts_worker_before_evaluation_is_called(self):
        class InactiveHandle:
            def __init__(self,*args):pass
            def is_alive(self):return False
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'evaluation_session';session.mkdir()
            for role in ('guard','watchdog'):(session/(role+'_state.json')).write_text('{}',encoding='utf-8')
            cfg=dict(study_output=tmp,startup_limit_seconds=1,heartbeat_max_age_seconds=30)
            proof={r:{} for r in ('owner','guard','watchdog')}
            proof['worker']=dict(process_id=os.getpid(),creation_time_ticks=111)
            with patch.object(supervisor,'load_runtime',return_value=(None,cfg,{})),patch.object(supervisor,'attached',return_value=proof),\
                 patch.object(supervisor.common,'creation_ticks',return_value=111),patch.object(supervisor,'WindowsProcess',InactiveHandle),\
                 patch.object(evaluator,'evaluate') as evaluate:
                with self.assertRaisesRegex(RuntimeError,'supervisor exited'):supervisor.worker('evaluate')
                evaluate.assert_not_called()
            self.assertEqual(json.loads((session/'worker_result.json').read_text())['status'],'failed_preserve')

    def test_original_green_full_suite_cannot_validate_new_analysis_and_evaluator(self):
        old=(evaluator.ROOT/'runs/b0e_replication_unittest_cpu_r1_20261026.log').read_text(encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'old526test'):supervisor.passed_suite_count(old)
        fixture='test_b0e_fixed_evaluation.Fixture ... ok\ntest_b0e_replication_analysis.Fixture ... ok\nRan 546 tests in 80.0s\nOK\n'
        self.assertEqual(supervisor.passed_suite_count(fixture),546)
        with self.assertRaises(ValueError):supervisor.passed_suite_count(fixture.replace('OK','FAILED'))


if __name__=='__main__':unittest.main()
