"""Explicit synthetic CPU smoke of the original-architecture bound action seam."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from pcdc.bound_local_actions import BoundLocalActionSession, ObservedBatchIdentity
from pcdc.local_action_composition import compare_joint
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/bound_local_actions_cpu_20261027.yaml'


def main():
    started = time.perf_counter()
    v = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    out = ROOT/v['output_dir']
    if out.exists(): raise FileExistsError('Preserve existing bound-session smoke evidence')
    if (v['device'] != 'cpu' or v['amp'] is not False or v['dtype'] != 'float32'
            or not v['engineering_only'] or v['real_data_or_checkpoint'] is not False
            or v['optimizer_updates'] != 0 or v['threshold'] != .5
            or v['decoder_channels'] != 64 or v['normalization_groups'] != 8):
        raise ValueError('Only the explicitly synthetic original-architecture CPU FP32 smoke is admitted')
    torch.set_num_threads(v['num_threads'])
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(v['seed'])
        model = AdapterCorrectedDetector(SiameseChangeDetector(v['decoder_channels'], v['normalization_groups']))
        torch.nn.init.normal_(model.adapter.conv3.weight, std=v['synthetic_adapter_final_weight_std'])
        model.eval().requires_grad_(False)
        shape = (v['batch_size'], 3, v['input_size'], v['input_size'])
        a, b = torch.rand(shape), torch.rand(shape)
        labels = (torch.rand(v['batch_size'],1,v['input_size'],v['input_size']) < .25).float()
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()
    identity = ObservedBatchIdentity(
        tuple(f'train/engineering_synthetic_{i}.png@x0_y0' for i in range(v['batch_size'])),
        tuple((h(f'synthetic-original-A-{i}'),h(f'synthetic-original-B-{i}')) for i in range(v['batch_size'])),
        tuple(h(f'synthetic-instance-{i}') for i in range(v['batch_size'])),
        h('explicit random synthetic actor, no checkpoint'),h('explicit random synthetic adapter, no checkpoint'),
        hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    with torch.no_grad():
        original_a0, original_a1 = model.action_logits(a, b)
    session = BoundLocalActionSession(model,a,b,identity,amp=False,engineering_only=True)
    evidence = session.evidence
    arms = {}
    with session:
        baseline = session.baseline(binding=session.binding)
        if not torch.equal(baseline,original_a0): raise ArithmeticError('Original no-action replay differs')
        stage2_size = v['input_size']//8
        for i,(r0,r1,c0,c1) in enumerate(v['action_boxes_stage2']):
            mask = torch.zeros(v['batch_size'],1,stage2_size,stage2_size,dtype=torch.bool)
            mask[:,:,r0:r1,c0:c1] = True
            session.register(str(i),mask,binding=session.binding)
        supports = [session.support(str(i),binding=session.binding) for i in range(2)]
        if not torch.equal(baseline,session.decode((),binding=session.binding,arm='reference_replay')):
            raise ArithmeticError('No-op replay differs from the actual original baseline')
        for arm in v['arms']:
            singles = [session.decode((str(i),),binding=session.binding,arm=arm) for i in range(2)]
            joint = session.decode(('0','1'),binding=session.binding,arm=arm)
            arms[arm] = compare_joint(baseline,singles,joint,supports,labels)
        replay = arms['reference_replay']
        if not replay['joint_equals_pasted_individuals'] or not replay['count_increment_additivity']:
            raise ArithmeticError('Actual encoded-batch composition failed')
        if not all(replay['individual_locality']): raise ArithmeticError('Replay action is not local')
        actual_session_forwards = dict(session._calls)
    result = dict(status='passed_synthetic_CPU_original_architecture_bound_actions',
        engineering_only=True, actual_device='cpu', actual_dtype='torch.float32',amp=False,
        explicit_random_weights=True,real_data_or_checkpoint=False,optimizer_updates=0,
        original_a0_bit_exact=True, reference_noop_bit_exact=True,
        original_whole_adapter_output_computed_for_separate_identity_unit_test=True,
        ordered_identity=asdict(identity), actual_evidence=evidence, arms=arms,
        session_actual_forwards=actual_session_forwards,
        total_actual_forwards={k:actual_session_forwards[k]+n for k,n in [('encoder',1),('adapter',2),('decoder',2)]},
        final_full_model_and_cache_identity_check_passed=True, hooks_removed=True,
        elapsed_seconds=time.perf_counter()-started, full_project_tests_pending_GPU_idle=True,
        GPU_FP16_validated=False, task_effectiveness=False, novelty_verified=False)
    out.mkdir()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    paths=[CONFIG,ROOT/v['protocol'],Path(__file__).resolve(),ROOT/'pcdc/bound_local_actions.py',
           ROOT/'pcdc/local_feature_replay.py',ROOT/'pcdc/local_action_composition.py',
           ROOT/'pcdc/model.py',ROOT/'pcdc/phase5_models.py']
    verification=dict(input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                      result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'], actual_forwards=result['total_actual_forwards'],
                         exact_composition=replay['joint_equals_pasted_individuals'],
                         elapsed_seconds=result['elapsed_seconds'],real_data_or_checkpoint=False)))


if __name__ == '__main__': main()
