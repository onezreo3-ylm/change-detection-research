"""Prepare a hash-sealed TRAIN-only P0 manifest and original-resolution QA.

No model forward, training, validation/test data access or automatic acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from pcdc.config import UniqueKeyLoader, load_config
from pcdc.data import _read_rgb, _read_mask, file_sha256
from pcdc.engine import check_checkpoint, read_checkpoint
from pcdc.perturbations import corrupt_rgb
from pcdc.runtime import write_json
from scripts.coupling_diagnostic import (select_train_samples, image_sha256,
                                         validate_marginal_hashes)

ROOT = Path(__file__).resolve().parents[1]
CELLS = {'uu': ('au', 'bu'), 'uv': ('au', 'bv'),
         'vu': ('av', 'bu'), 'vv': ('av', 'bv')}


def resolve(config, value):
    path = Path(value)
    return (path if path.is_absolute() else config.parent / path).resolve()


def checked_path(config, values, key):
    path = resolve(config, values[key])
    if file_sha256(path) != values[key + '_sha256']:
        raise ValueError(f'Frozen input changed: {key}')
    return path


def train_path(root, folder, name):
    if folder not in ('A', 'B', 'label') or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Invalid train source identity')
    base = (Path(root).resolve() / 'train' / folder).resolve()
    path = (base / name).resolve()
    # Resolve symlinks/junctions and forbid train folder escaping the declared root.
    if not base.is_relative_to(Path(root).resolve()) or path.parent != base:
        raise ValueError('Train path escaped its root')
    if any(part.lower() in ('val', 'validation', 'test') for part in path.parts):
        raise ValueError('Non-train path forbidden')
    return path


def draw_instance(corruptions, family, level, seed, sid, variant):
    """Independent with-replacement draws from EXISTING discrete direction table.

    Same-direction duplicates are retained, not forced to be opposite or redrawn.
    """
    if family not in ('brightness', 'contrast', 'gamma', 'color_bias') or level not in (1, 3, 5):
        raise ValueError('Outside P0 family/level scope')
    payload = json.dumps(['coupling-p0-instance-v1', seed, sid, family, level, variant],
                         separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    local_seed = int(digest[:16], 16)
    directions = sorted(corruptions[family])
    rng = np.random.Generator(np.random.PCG64(local_seed))
    direction = directions[int(rng.integers(len(directions)))]
    return {'direction': direction, 'parameters': corruptions[family][direction][level - 1],
            'seed': local_seed}


def source_bindings(config_path, paths):
    dependencies = list((ROOT / 'pcdc').glob('*.py')) + [
        ROOT / 'scripts/coupling_diagnostic.py', ROOT / 'scripts/prepare_coupling_p0.py',
        ROOT / 'scripts/evaluate_coupling_p0.py',
        ROOT / 'docs/COUPLING_P0_PRIOR_ART_2026-10-07.md',
        ROOT / 'docs/EXPERIMENT_REPLAN_2026-10-07.md', config_path, *paths]
    return {str(p.resolve()): file_sha256(p) for p in sorted(set(dependencies))}


def qa_sheet(raws, mask, family, directions, corruptions, level, path, title):
    # No scaling, color changes or overlays on the image tiles themselves.
    if (len(raws) != 2 or any(x.shape != (256, 256, 3) or x.dtype != np.uint8 for x in raws)
            or mask.shape != (256, 256) or not np.isin(mask, (0, 1)).all()):
        raise ValueError('QA requires original uint8 256px RGB tiles and a binary aligned mask')
    columns = [('clean', None)] + [(d, corruptions[family][d][level - 1]) for d in directions]
    width, height = 256 * len(columns), 2 * (256 + 22) + 256 + 22
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (date, raw) in enumerate(zip(('A', 'B'), raws)):
        for col, (direction, parameters) in enumerate(columns):
            changed = raw if parameters is None else corrupt_rgb(raw, family, parameters, seed=0)
            x, y = col * 256, row * 278
            canvas.paste(Image.fromarray(changed), (x, y + 22))
            draw.text((x + 3, y + 3), f'{date} {family} L{level} {direction}', fill='black')
    y = 556
    canvas.paste(Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB'), (0, y + 22))
    draw.text((3, y + 3), 'binary change label (not an edit mask)', fill='black')
    draw.text((270, y + 30), title, fill='black')
    canvas.save(path, optimize=True)
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'pixels': list(canvas.size), 'original_resolution': True, 'resized': False,
            'family': family, 'level': level, 'directions': directions, 'title': title}


def prepare(config_path):
    config_path = Path(config_path).resolve()
    v = yaml.load(config_path.read_bytes(), Loader=UniqueKeyLoader)
    if v['schema_version'] != 1 or v['name'] != 'valais_coupling_p0_v1':
        raise ValueError('Unexpected P0 configuration')
    if (v['families'] != ['brightness', 'contrast', 'gamma', 'color_bias']
            or v['levels'] != [1, 3, 5] or v['selection']['positive_quotas'] != [4, 3, 3, 3, 3]
            or v['selection']['empty_count'] != 48 or v['threshold'] != .5
            or v['max_seconds'] != 21600 or v['qa']['level'] != 5
            or v['qa']['original_tile_pixels'] != 256):
        raise ValueError('P0 scope differs from the frozen execution settings')
    output = resolve(config_path, v['output_dir'])
    if output.exists():
        raise FileExistsError(output)
    base_path = resolve(config_path, v['base_config'])
    base = load_config(base_path)
    if base.run_test_after_training or base.values['data']['splits']['train'] != 'train':
        raise ValueError('P0 requires disabled test and explicit train folder')
    inputs = [checked_path(config_path, v, key) for key in ('label_report', 'data_audit', 'corruption_config')]
    balance, audit = [json.loads(p.read_text(encoding='utf-8')) for p in inputs[:2]]
    corruptions = yaml.load(inputs[2].read_bytes(), Loader=UniqueKeyLoader)['corruptions']
    if balance['test_accessed'] is not False or balance['status'] != 'development_label_balance_only':
        raise ValueError('Unrecognized historical label audit')
    inventory = audit['splits']['train']
    if inventory['source_pairs'] != 11025 or balance['train']['pairs'] != 11025:
        raise ValueError('Wrong historical training inventory')
    foreground = {row['name']: row['positive_pixels'] for row in balance['train']['positive_manifest']}
    names = [row['name'] for row in inventory['entries']]
    if len(names) != len(set(names)) or len(foreground) != 1046 or not set(foreground) <= set(names):
        raise ValueError('Historical positive manifest does not match inventory')
    records = [{'id': f'train/{name}@x0_y0', 'name': name, 'split': 'train',
                'foreground_pixels': foreground.get(name, 0)} for name in names]
    selection = select_train_samples(records, **v['selection'])
    models = {}
    for key, model in v['models'].items():
        config = resolve(config_path, model['config'])
        cp_path = checked_path(config_path, model, 'checkpoint')
        cp = read_checkpoint(cp_path)
        cfg = load_config(config)
        check_checkpoint(cfg, cp)
        if cfg.run_test_after_training or cfg.data_root != base.data_root:
            raise ValueError('Model inference data/protocol mismatch')
        # Verify every original model source, without adding new diagnostic code to old identity.
        for relative, sha in cp['source_sha256'].items():
            if file_sha256(ROOT / relative) != sha:
                raise ValueError(f'Original checkpoint implementation changed: {relative}')
        if cp['file_sha256']['train'] != inventory['file_sha256']:
            raise ValueError('Models/audit do not bind the same training data')
        models[key] = {'config': str(config), 'config_sha256': file_sha256(config),
                       'checkpoint': str(cp_path), 'checkpoint_sha256': model['checkpoint_sha256'],
                       'epoch': cp['epoch']}
        inputs.append(config)
        del cp
    samples, raw_cache = [], {}
    for row in selection['samples']:
        files = {}
        for key, folder in [('image_a', 'A'), ('image_b', 'B'), ('mask', 'label')]:
            path = train_path(base.data_root, folder, row['name'])
            sha = file_sha256(path)
            if sha != inventory['file_sha256'][row['name']][key]:
                raise ValueError(f'Original training file changed: {path}')
            files[key] = {'path': str(path), 'sha256': sha}
        a, b = [_read_rgb(Path(files[k]['path'])) for k in ('image_a', 'image_b')]
        d = base.values['data']
        mask = _read_mask(Path(files['mask']['path']), d['label_values'], d['label_decode'])
        if a.shape != (256, 256, 3) or b.shape != a.shape or int(mask.sum()) != row['foreground_pixels']:
            raise ValueError('Training tile or mask differs from frozen audit')
        conditions = []
        for family in v['families']:
            for level in v['levels']:
                instances = {variant: draw_instance(corruptions, family, level, v['instance_seed'], row['id'], variant)
                             for variant in ('u', 'v')}
                images = {date + variant: image_sha256(corrupt_rgb(raw, family, inst['parameters'], seed=inst['seed']))
                          for date, raw in [('a', a), ('b', b)] for variant, inst in instances.items()}
                marginal = {cell: [images[x], images[y]] for cell, (x, y) in CELLS.items()}
                validate_marginal_hashes(marginal)
                conditions.append({'family': family, 'level': level, **instances, 'images': images,
                                   'marginal_hashes': marginal,
                                   'degenerate_parameters': instances['u']['parameters'] == instances['v']['parameters']})
        samples.append({**row, 'stratum': row['area_stratum'], 'files': files, 'conditions': conditions})
        raw_cache[row['id']] = (a, b, mask)
    output.mkdir(parents=True, exist_ok=False)
    qa_dir = output / 'previews'
    qa_dir.mkdir()
    # Six anchors: all five positive-area strata and one empty tile, chosen before inference.
    anchor_strata = ['positive_1', 'positive_2', 'positive_3', 'positive_4', 'positive_5', 'empty']
    absent_qa_strata = [s for s in anchor_strata if not any(x['stratum'] == s for x in samples)]
    anchors = [next(x for x in samples if x['stratum'] == s)
               for s in anchor_strata if s not in absent_qa_strata]
    qa_images = []
    for index, anchor in enumerate(anchors):
        family = v['families'][index % len(v['families'])]
        a, b, mask = raw_cache[anchor['id']]
        qa_images.append(qa_sheet((a, b), mask, family, sorted(corruptions[family]), corruptions, 5,
                                  qa_dir / f'anchor_{index + 1}_{family}.png', anchor['id'] + ' ' + anchor['stratum']))
    # Additional color-bias extremes for large positive anchor (all four directions).
    anchor = max(anchors, key=lambda x: x['foreground_pixels'])
    a, b, mask = raw_cache[anchor['id']]
    qa_images.append(qa_sheet((a, b), mask, 'color_bias', sorted(corruptions['color_bias']), corruptions, 5,
                              qa_dir / 'anchor_large_color_bias.png', anchor['id'] + ' ' + anchor['stratum']))
    transport_bytes = sum(4 * ((x['bytes'] + 2) // 3) for x in qa_images)
    settings = {key: v[key] for key in ('families', 'levels', 'threshold', 'max_seconds',
                'gpu_uuid', 'temperature_stop_c', 'nvidia_smi', 'selection')}
    manifest = {'schema_version': 1, 'name': v['name'], 'split': 'train',
                'base_config': str(base_path), 'data_root': str(base.data_root), 'output_dir': str(output),
                'config_path': str(config_path), 'config_sha256': file_sha256(config_path),
                'source_hashes': source_bindings(config_path, [base_path, *inputs]),
                'models': models, 'settings': settings, 'samples': samples, 'qa_images': qa_images,
                'selection': {k: value for k, value in selection.items() if k != 'samples'},
                'expected_predictions': 9408, 'model_inference_performed': False,
                'val_test_image_or_label_accessed': False,
                'qa_status': 'pending_visual_review', 'qa_transport_base64_bytes': transport_bytes,
                'absent_qa_strata': absent_qa_strata,
                'qa_transport_within_budget': transport_bytes < v['qa']['maximum_chat_base64_bytes'],
                'parameter_sampling': 'independent_with_replacement_existing_direction_entries_no_redraw',
                'limitations': 'Enriched train sample; sampled visibility QA; no spatial independence or research novelty claim.'}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'status': 'prepared_pending_visual_review', 'samples': len(samples),
                      'qa_images': len(qa_images), 'qa_base64_bytes': transport_bytes,
                      'manifest_sha256': file_sha256(output / 'manifest.json')}))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config)


if __name__ == '__main__':
    main()
