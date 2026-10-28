"""Explicit CPU/random-weight engineering of the sample-bound alternative seam."""
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
import yaml

from pcdc.bound_local_actions import ObservedBatchIdentity
from pcdc.bound_temporal_actions import BoundTemporalActionSession
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/bound_temporal_actions_cpu_20261027.yaml'


def main():
    started=time.perf_counter();v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Preserve bound alternative smoke evidence')
    if (v['device']!='cpu' or v['dtype']!='float32' or v['amp'] is not False or
        not v['engineering_only'] or v['real_data_or_checkpoint'] is not False or
        v['optimizer_updates']!=0 or v['decoder_channels']!=64 or v['normalization_groups']!=8):
        raise ValueError('Explicit synthetic original-architecture CPU scope required')
    torch.set_num_threads(v['num_threads']);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(v['seed'])
        model=AdapterCorrectedDetector(SiameseChangeDetector(v['decoder_channels'],v['normalization_groups'])).cpu().eval().requires_grad_(False)
        torch.nn.init.normal_(model.adapter.conv3.weight,std=v['synthetic_adapter_final_weight_std'])
        shape=(v['batch_size'],3,v['input_size'],v['input_size']);a,b=torch.rand(shape),torch.rand(shape)
    n=v['batch_size'];identity=ObservedBatchIdentity(tuple(f'train/explicit_synthetic_{i}' for i in range(n)),
        tuple(('a'*64,'b'*64) for _ in range(n)),tuple('c'*64 for _ in range(n)),'d'*64,'e'*64,hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    def register(s):
        size=v['input_size']//8
        for region,(r0,r1,c0,c1) in enumerate(v['action_boxes']):
            mask=torch.zeros(n,1,size,size,dtype=torch.bool);mask[:,:,r0:r1,c0:c1]=True
            for kind in s.KINDS:s.register(f'{region}:{kind}',kind,mask,binding=s.binding)
    calls=dict(encoder=0,adapter=0,decoder=0)
    s=BoundTemporalActionSession(model,a,b,identity,amp=False,engineering_only=True)
    with s:
        register(s);baseline=s.baseline(binding=s.binding)
        all_variants={k:s.decode(tuple(f'{i}:{k}' for i in range(4)),binding=s.binding,arm='reference_replay') for k in s.KINDS}
        supports=[s.support(f'{i}:both',binding=s.binding) for i in range(4)]
        count=0
        for choice in itertools.product((None,*s.KINDS),repeat=4):
            names=tuple(f'{i}:{kind}' for i,kind in enumerate(choice) if kind)
            actual=s.decode(names,binding=s.binding,arm='reference_replay');pasted=baseline.clone()
            for support,kind in zip(supports,choice):
                if kind:pasted=torch.where(support,all_variants[kind],pasted)
            if not torch.equal(actual,pasted):raise ArithmeticError('Bound alternative composition differs')
            count+=1
        for name in calls:calls[name]+=s._calls[name]
    swap_identity=replace(identity,source_pair_sha256=tuple(('b'*64,'a'*64) for _ in range(n)))
    swapped=BoundTemporalActionSession(model,b,a,swap_identity,amp=False,engineering_only=True)
    with swapped:
        register(swapped)
        swap_all={k:swapped.decode(tuple(f'{i}:{k}' for i in range(4)),binding=swapped.binding,arm='reference_replay') for k in swapped.KINDS}
        symmetry=dict(noop=torch.equal(baseline,swapped.baseline(binding=swapped.binding)),
            A_matches_swapped_B=torch.equal(all_variants['A_only'],swap_all['B_only']),
            B_matches_swapped_A=torch.equal(all_variants['B_only'],swap_all['A_only']),
            both=torch.equal(all_variants['both'],swap_all['both']))
        if not all(symmetry.values()):raise ArithmeticError('Bound temporal exchange differs')
        for name in calls:calls[name]+=swapped._calls[name]
    result=dict(status='passed_bound_temporal_action_CPU_smoke',all256_actual_combinations_exact=count==256,
        temporal_exchange=symmetry,actual_forwards=calls,full_model_variant_identity_checks_passed=True,
        real_data_or_checkpoint=False,explicit_random_weights=True,actual_device='cpu',actual_dtype='torch.float32',
        optimizer_updates=0,method_effectiveness=False,novelty_verified=False,CUDA_FP16_validated=False,
        full_project_tests_pending_gpu_idle=True,elapsed_seconds=time.perf_counter()-started)
    out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    paths=[CONFIG,ROOT/v['protocol'],Path(__file__).resolve(),ROOT/'pcdc/bound_temporal_actions.py',
        ROOT/'pcdc/bound_local_actions.py',ROOT/'pcdc/local_feature_replay.py',ROOT/'pcdc/model.py',ROOT/'pcdc/phase5_models.py',
        ROOT/'tests/test_bound_temporal_actions.py']
    (out/'verification.json').write_text(json.dumps(dict(input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],actual_combinations=count,actual_forwards=calls,
        temporal_exchange=symmetry,elapsed_seconds=result['elapsed_seconds'],real_data_or_checkpoint=False)))


if __name__=='__main__':main()
