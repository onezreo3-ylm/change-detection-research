"""CPU synthetic actual joint-versus-single feature interventions."""
from pathlib import Path
import hashlib
import json

import torch
import yaml

from pcdc.local_action_composition import compare_joint
from pcdc.local_feature_replay import ReferenceGroupNormReplay, fpn_convolution_support
from pcdc.model import FPNDifferenceDecoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/local_action_composition_cpu_20261027.yaml'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    v = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); out = ROOT / v['output_dir']
    if out.exists(): raise FileExistsError('Preserve the existing joint-action prototype')
    if v['device'] != 'cpu' or v['dtype'] != 'float32' or not v['engineering_only'] or v['threshold'] != .5:
        raise ValueError('Only explicit CPU FP32 synthetic scope is admitted')
    torch.set_num_threads(v['num_threads']); g = torch.Generator(device='cpu').manual_seed(v['seed'])
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(g.get_state())
        model = FPNDifferenceDecoder(v['decoder_channels'], v['normalization_groups']).cpu().eval().requires_grad_(False)
    features = [torch.rand(1, c, s, s, generator=g) for c, s in zip(v['feature_channels'], v['feature_sizes'])]
    size = tuple(v['output_size']); stage = v['edited_feature_stage']; edits = []; supports = []; residuals = []
    for r0, r1, c0, c1 in v['action_boxes']:
        masks = [torch.zeros(1, 1, s, s, dtype=torch.bool) for s in v['feature_sizes']]
        masks[stage][:, :, r0:r1, c0:c1] = True
        supports.append(fpn_convolution_support(masks, size))
        residual = torch.zeros_like(features[stage])
        residual[:, :, r0:r1, c0:c1] = v['edit_amplitude'] * torch.rand(1, v['feature_channels'][stage], r1-r0, c1-c0, generator=g)
        residuals.append(residual)
        action = [x.clone() for x in features]; action[stage] += residual; edits.append(action)
    if len(edits) != 2 or (supports[0] & supports[1]).any(): raise ValueError('Frozen two-action supports must be disjoint')
    joint_features = [x.clone() for x in features]
    for residual in residuals: joint_features[stage] += residual
    labels = (torch.rand(1, 1, *size, generator=g) < v['synthetic_foreground_probability']).float()
    with torch.no_grad():
        initial = model(features, size)
        bias_shift = float(initial.median()) if v['center_synthetic_classifier_bias_before_all_arms'] else 0.
        model.classifier.bias.sub_(bias_shift)
        original_state = {k: t.clone() for k, t in model.state_dict().items()}
        replay = ReferenceGroupNormReplay(model)
        with replay.capture(): base = model(features, size)
        ordinary_singles = [model(x, size) for x in edits]; ordinary_joint = model(joint_features, size)
        replay_singles = []
        for action in edits:
            with replay.replay(): replay_singles.append(model(action, size))
        with replay.replay(): replay_joint = model(joint_features, size)
        union = supports[0] | supports[1]
        masked_singles = [torch.where(s, x, base) for x, s in zip(ordinary_singles, supports)]
        masked_joint = torch.where(union, ordinary_joint, base)
    arms = dict(ordinary_GN=compare_joint(base, ordinary_singles, ordinary_joint, supports, labels),
        ordinary_GN_plus_output_support_mask=compare_joint(base, masked_singles, masked_joint, supports, labels),
        reference_statistics_all_six_GN=compare_joint(base, replay_singles, replay_joint, supports, labels))
    full = arms['reference_statistics_all_six_GN']; negative = arms['ordinary_GN_plus_output_support_mask']
    if not full['joint_equals_pasted_individuals'] or not full['count_increment_additivity'] or not all(full['individual_locality']):
        raise ArithmeticError('Reference replay did not certify exact composition')
    if not all(n > 0 for n in full['hard_prediction_changes']): raise ArithmeticError('Fixture did not exercise hard prediction changes')
    if negative['outside_joint_changed_positions'] != 0: raise ArithmeticError('Output-mask control failed outside support')
    if negative['joint_equals_pasted_individuals']: raise ArithmeticError('Frozen fixture did not distinguish ordinary output masking')
    if any(not torch.equal(t, original_state[k]) for k, t in model.state_dict().items()): raise ArithmeticError('Fixed model state changed')
    if any(m._forward_hooks for m in replay.modules.values()): raise ArithmeticError('Replay hook leaked')
    result = dict(status='passed_actual_joint_CPU_composition_engineering_only',
        seed=v['seed'], explicit_synthetic_bias_shift_before_all_arms=bias_shift,
        shared_reference=True, influence_supports_disjoint=True, support_positions=[int(s.sum()) for s in supports],
        output_positions=base.numel(), actual_decoder_forwards=8, original_decoder_GN_count=len(replay.modules),
        arms=arms, actual_device=str(base.device), actual_dtype=str(base.dtype),
        original_model_state_unchanged_after_fixture=True, hooks_removed=True, optimizer_updates=0,
        real_data_or_checkpoint=False, method_effectiveness=False, method_novelty_verified=False,
        GPU_FP16_validated=False, full_project_tests_pending_gpu_idle=True,
        scope='Same reference and weights; fixed residuals; disjoint propagated supports only; no overlapping-action guarantee.')
    out.mkdir()
    with (out/'result.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    paths=[CONFIG,ROOT/v['protocol'],ROOT/'pcdc/local_action_composition.py',ROOT/'pcdc/local_feature_replay.py',
        ROOT/'pcdc/model.py',Path(__file__).resolve()]
    with (out/'verification.json').open('x',encoding='utf-8') as f:
        json.dump(dict(input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},result_sha256=sha(out/'result.json'),
            no_new_research_GPU_job=True),f,indent=2)
    print(json.dumps(dict(status=result['status'],replay_exact_composition=full['joint_equals_pasted_individuals'],
        replay_exact_count_additivity=full['count_increment_additivity'],
        output_mask_joint_max_difference=negative['joint_minus_pasted_max_abs'],
        output_mask_count_additivity=negative['count_increment_additivity'],
        hard_prediction_changes=full['hard_prediction_changes'],real_data_or_checkpoint=False)))


if __name__=='__main__': main()
