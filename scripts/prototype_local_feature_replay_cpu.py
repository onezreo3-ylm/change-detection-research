"""Synthetic CPU mechanism check; never load research data or checkpoints."""
from pathlib import Path
import hashlib
import json

import torch
import yaml

from pcdc.local_feature_replay import ReferenceGroupNormReplay, fpn_convolution_support
from pcdc.model import FPNDifferenceDecoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/local_feature_replay_cpu_20261027.yaml'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    v = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); out = ROOT / v['output_dir']
    if out.exists(): raise FileExistsError('Preserve the original mechanism prototype')
    if v['device'] != 'cpu' or v['dtype'] != 'float32' or not v['engineering_only']:
        raise ValueError('Only the frozen explicit CPU synthetic scope is admitted')
    torch.set_num_threads(v['num_threads'])
    generator = torch.Generator(device='cpu').manual_seed(v['seed'])
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        model = FPNDifferenceDecoder(v['decoder_channels'], v['normalization_groups']).cpu().eval()
    model.requires_grad_(False)
    snapshot = {k: x.clone() for k, x in model.state_dict().items()}
    features = [torch.rand(1, c, s, s, generator=generator) for c, s in zip(v['feature_channels'], v['feature_sizes'])]
    masks = [torch.zeros(1, 1, s, s, dtype=torch.bool) for s in v['feature_sizes']]
    stage = v['edited_feature_stage']; r0, r1 = v['edit_rows']; c0, c1 = v['edit_columns']
    masks[stage][:, :, r0:r1, c0:c1] = True
    edited = [x.clone() for x in features]
    edited[stage][:, :, r0:r1, c0:c1] += v['edit_amplitude'] * torch.rand(1, v['feature_channels'][stage], r1-r0, c1-c0, generator=generator)
    size = tuple(v['output_size']); support = fpn_convolution_support(masks, size)
    replay = ReferenceGroupNormReplay(model)
    names = tuple(replay.modules)
    if len(names) != 6 or support.all() or not support.any(): raise ValueError('Original six-GN graph/support required')
    with torch.no_grad():
        with replay.capture(): reference = model(features, size)
        with replay.replay(): identity = model(features, size)
        ordinary = model(edited, size)
        with replay.replay(): acted = model(edited, size)
        last = ReferenceGroupNormReplay(model, (names[-1],))
        with last.capture(): model(features, size)
        with last.replay(): last_only = model(edited, size)
        output_mask_control = torch.where(support, ordinary, reference)
    if not torch.equal(reference, identity): raise ArithmeticError('Zero action is not bit-exact')
    report = {}
    for name, value in [('ordinary_GN', ordinary), ('last_GN_only', last_only),
                        ('all_six_reference_replay', acted), ('ordinary_plus_output_support_mask', output_mask_control)]:
        delta = value - reference
        report[name] = dict(outside_max_abs=float(delta[~support].abs().max()),
            outside_changed_positions=int(torch.count_nonzero(delta[~support])),
            inside_max_abs=float(delta[support].abs().max()), all_finite=bool(torch.isfinite(value).all()))
    if report['ordinary_GN']['outside_max_abs'] <= 1e-7: raise ArithmeticError('Synthetic ordinary-GN leakage not observed')
    if report['all_six_reference_replay']['outside_changed_positions'] != 0 or report['all_six_reference_replay']['inside_max_abs'] <= 1e-7:
        raise ArithmeticError('Full replay failed locality or suppressed the entire action')
    if report['ordinary_plus_output_support_mask']['outside_changed_positions'] != 0:
        raise ArithmeticError('Mandatory simple output-mask control failed')
    if any(not torch.equal(x, snapshot[k]) for k, x in model.state_dict().items()): raise ArithmeticError('Model state changed')
    if any(m._forward_hooks for m in replay.modules.values()): raise ArithmeticError('A replay hook leaked')
    result = dict(status='passed_synthetic_cpu_mechanism_only', original_decoder_width=v['decoder_channels'],
        original_normalization_groups=v['normalization_groups'], covered_GN_names=names,
        zero_action_bit_exact=True, output_positions=reference.numel(), convolution_support_positions=int(support.sum()),
        comparisons=report, model_state_unchanged=True, hooks_removed=True, actual_device=str(reference.device),
        actual_dtype=str(reference.dtype), raw_data_or_checkpoint_loaded=False, optimization_updates=0,
        research_effectiveness=False, method_novelty_verified=False, GPU_FP16_validated=False,
        full_project_tests_pending_gpu_idle=True,
        limitation='The output-mask control also preserves outside support; locality alone is not method superiority.')
    out.mkdir()
    with (out / 'result.json').open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    inputs = [CONFIG, ROOT/v['protocol'], ROOT/'pcdc/local_feature_replay.py', Path(__file__).resolve(), ROOT/'pcdc/model.py']
    with (out / 'verification.json').open('x', encoding='utf-8') as f:
        json.dump(dict(input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
            result_sha256=sha(out/'result.json'), actual_scope='synthetic CPU FP32 original decoder graph only'), f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
