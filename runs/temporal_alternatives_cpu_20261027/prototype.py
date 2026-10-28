"""Synthetic CPU falsification only; never a real-data or learned-policy runner."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from pcdc.bound_local_actions import model_tensor_sha256
from pcdc.local_feature_replay import ReferenceGroupNormReplay,fpn_convolution_support
from pcdc.model import SiameseChangeDetector
from pcdc.phase5_models import AdapterCorrectedDetector

OUT=Path(__file__).resolve().parent


def main():
    if (OUT/'result.json').exists(): raise FileExistsError('Preserve the first CPU mechanism check')
    started=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261027)
        model=AdapterCorrectedDetector(SiameseChangeDetector()).cpu().eval().requires_grad_(False)
        torch.nn.init.normal_(model.adapter.conv3.weight,std=.01)
        image_a,image_b=torch.rand(1,3,256,256),torch.rand(1,3,256,256)
    state=model_tensor_sha256(model);calls=dict(encoder=0,adapter=0,decoder=0)
    boxes=[(2,8,2,8),(2,8,24,30),(24,30,2,8),(24,30,24,30)]

    def observed(a,b):
        fa,fb=model._encode(a,b);calls['encoder']+=1
        ra,rb=model.adapter(fa[1]),model.adapter(fb[1]);calls['adapter']+=2
        base=model.detector.difference_features(fa,fb)
        variants={1:torch.abs((fa[1]+ra)-fb[1]),
                  2:torch.abs(fa[1]-(fb[1]+rb)),
                  3:torch.abs((fa[1]+ra)-(fb[1]+rb))}
        return base,variants

    def decode(features):
        calls['decoder']+=1
        value=model.detector.decoder(features,(256,256))
        if not torch.isfinite(value).all():raise ArithmeticError('Nonfinite synthetic output')
        return value

    with torch.no_grad():
        base,variants=observed(image_a,image_b)
        masks=[];supports=[]
        for r0,r1,c0,c1 in boxes:
            ms=[torch.zeros(1,1,*x.shape[-2:],dtype=torch.bool) for x in base]
            ms[1][:,:,r0:r1,c0:c1]=True
            masks.append(ms[1]);supports.append(fpn_convolution_support(ms,(256,256)))
        union=torch.zeros_like(supports[0])
        for support in supports:
            if (union&support).any():raise ValueError('Fixed propagated supports overlap')
            union|=support
        if union.all():raise ValueError('No outside region remains')
        replay=ReferenceGroupNormReplay(model.detector.decoder)
        with replay.capture():z0=decode(base)
        def action(choice,features,vs,reference):
            if len(choice)!=4 or any(type(x) is not int or not 0<=x<=3 for x in choice):
                raise ValueError('Exactly one admissible variant per each fixed region')
            changed=list(features)
            for mask,variant in zip(masks,choice):
                if variant:changed[1]=torch.where(mask,vs[variant],changed[1])
            with reference.replay():return decode(changed)
        all_variants={v:action((v,v,v,v),base,variants,replay) for v in (1,2,3)}
        matches=[];outside=[];single_changes={};wrong_both_recovery=[]
        for choice in itertools.product(range(4),repeat=4):
            actual=action(choice,base,variants,replay)
            pasted=z0.clone();selected_union=torch.zeros_like(union)
            for support,variant in zip(supports,choice):
                if variant:
                    pasted=torch.where(support,all_variants[variant],pasted)
                    selected_union|=support
            exact=bool(torch.equal(actual,pasted))
            matches.append(exact)
            outside.append(int(torch.count_nonzero((actual-z0)[~selected_union])))
            if not exact or outside[-1]:raise ArithmeticError('Actual alternative composition differs')
            if sum(v!=0 for v in choice)==1:
                region=next(i for i,v in enumerate(choice) if v)
                variant=choice[region]
                single_changes[f'{region}:{variant}']=int(torch.count_nonzero(actual-z0))
                if variant in (1,2):
                    wrong=torch.where(supports[region],all_variants[3],z0)
                    wrong_both_recovery.append(float((actual-wrong).abs().max()))
        swapped,swapped_variants=observed(image_b,image_a)
        swapped_replay=ReferenceGroupNormReplay(model.detector.decoder)
        with swapped_replay.capture():swapped_z0=decode(swapped)
        swapped_all={v:action((v,v,v,v),swapped,swapped_variants,swapped_replay) for v in (1,2,3)}
        symmetry=dict(noop=bool(torch.equal(z0,swapped_z0)),
            A_matches_swapped_B=bool(torch.equal(all_variants[1],swapped_all[2])),
            B_matches_swapped_A=bool(torch.equal(all_variants[2],swapped_all[1])),
            both=bool(torch.equal(all_variants[3],swapped_all[3])))
    if not all(symmetry.values()):raise ArithmeticError('Synthetic temporal identity correspondence failed')
    if not all(single_changes.values()) or not all(x>0 for x in wrong_both_recovery):
        raise ArithmeticError('Fixture did not distinguish the mutually exclusive actions')
    if model_tensor_sha256(model)!=state:raise ArithmeticError('Original fixed model state changed')
    if any(m._forward_hooks for m in model.modules()):raise ArithmeticError('A hook remains')
    result=dict(status='passed_synthetic_CPU_temporal_alternative_composition',engineering_only=True,
        device='cpu',dtype='torch.float32',seed=20261027,original_decoder_channels=64,original_GN_groups=8,
        actual_forwards=calls,actual_assignments_checked=len(matches),all256_exact=all(matches),
        maximum_outside_changed_positions=max(outside),nonzero_single_changed_positions=single_changes,
        both_output_cannot_recover_A_B_by_support_pasting_max_abs=wrong_both_recovery,
        temporal_exchange_correspondence=symmetry,model_state_unchanged=True,hooks_removed=True,
        real_data_or_checkpoint=False,optimizer_updates=0,method_novelty_verified=False,method_effectiveness=False,
        GPU_FP16_validated=False,speed_advantage_measured=False,elapsed_seconds=time.perf_counter()-started,
        comparison_decoder_views_per_observed_pair=dict(all_three_variants_after_baseline=4,selected_variant_after_baseline=2),
        cost_caveat='Variants may be batched or share sparse computation. Calls are not a latency/FLOP lower bound.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    paths=[Path(__file__).resolve(),ROOT/'docs/TEMPORAL_ALTERNATIVES_CPU_SCOPE_2026-10-27.md',
        ROOT/'docs/METHOD_TEMPORAL_ALTERNATIVES_PROPOSAL_2026-10-27.md',
        ROOT/'pcdc/local_feature_replay.py',ROOT/'pcdc/bound_local_actions.py',
        ROOT/'pcdc/model.py',ROOT/'pcdc/phase5_models.py']
    verification=dict(input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                      result_sha256=hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],all256_exact=result['all256_exact'],actual_forwards=calls,
        temporal_exchange_correspondence=symmetry,minimum_wrong_both_recovery_max_abs=min(wrong_both_recovery),
        elapsed_seconds=result['elapsed_seconds'],real_data_or_checkpoint=False)))


if __name__=='__main__':main()
