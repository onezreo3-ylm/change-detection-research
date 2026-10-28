"""Strict identity for the separately approved 40-epoch ordinary controls."""
from __future__ import annotations

from pathlib import Path
import statistics

import yaml

from pcdc.config import UniqueKeyLoader
from scripts import five_view_baselines as f

FLAVOR = 'matched_five_view_baseline_40ep_v1'
EPOCHS = 40
UPDATES = 110280
PAIR_VIEWS = 2205000


def load_study(path):
    source = Path(path).resolve()
    v = yaml.load(source.read_bytes(), Loader=UniqueKeyLoader)
    parent_source = (source.parent / v['parent_config']).resolve()
    _, parent, parent_paths, base = f.load_study(parent_source)
    expected = dict(parent, name='valais_uniform_c5r5_40ep_seed42_v1_20261020',
        output_dir='../runs/valais_c5r5_40ep_seed42_v1_20261020', epochs=EPOCHS,
        protocol_document='../docs/C5R5_40EP_PROTOCOL_2026-10-20.md', minimum_unittests=397,
        parent_config='valais_c5r5_study_v1_20261020.yaml', training_flavor=FLAVOR,
        cosine_tmax=EPOCHS, carry_parent_pilot_debit=True, smoke_updates_per_role=3)
    if v != expected:
        raise ValueError('Unreviewed 40-epoch protocol, budget, gate or inheritance change')
    paths = dict(parent_paths)
    for key in ('output_dir', 'protocol_document'):
        paths[key] = (source.parent / v[key]).resolve()
    paths.update(parent_config=parent_source, parent_output_dir=parent_paths['output_dir'])
    paths['output_dir'].relative_to(f.ROOT / 'runs')
    return source, v, paths, base


def implementation_files():
    files = f.implementation_files()
    for name in ('scripts/c5r5_40ep_identity.py', 'scripts/five_view_40ep.py',
                 'scripts/valais_c5r5_40ep.py'):
        files[name] = f.ROOT / name
    return files


def estimate_budget(timings, settings, debit):
    if not timings or any(not 0 < x < 1800 for x in timings):
        raise ValueError('Invalid inherited real CUDA timings')
    if not 0 <= debit < 1800 or settings['epochs'] != EPOCHS:
        raise ValueError('Invalid engineering debit or horizon')
    worker = statistics.mean(timings) * UPDATES * settings['pilot_training_time_multiplier']
    worker += EPOCHS * settings['pilot_extra_seconds_per_epoch']
    return {'steady_mean_seconds_per_step': statistics.mean(timings),
        'projected_worker_seconds_including_margin': worker,
        'engineering_seconds_debited': debit, 'projected_total_role_seconds': worker + debit,
        'maximum_role_seconds': 86400, 'within_frozen_budget': worker + debit <= 86400}


def role_debit(paths, role):
    proof = f.read(paths['output_dir'] / 'authorization.json')
    return proof['engineering_seconds_debited'][role]
