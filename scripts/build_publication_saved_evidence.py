"""Make bounded paper plots and a retrospective register from saved evidence.

No model, checkpoints, raw datasets, fresh bootstrap or held-out result files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/publication_evidence_audit_20261027'
REGISTER = ROOT / 'configs/publication_experiment_register_20261027.yaml'
FAMILIES = ('brightness', 'contrast', 'gamma', 'color_bias', 'haze', 'shadow', 'noise', 'blur', 'jpeg')
POLICIES = ('C1-U', 'C1-H', 'G1-U', 'G1-H', 'G2-U', 'G2-H', 'G3-U', 'G3-H')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def write_json(path: Path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def write_csv(path: Path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='runs/publication_evidence_audit_20261027')
    args = parser.parse_args()
    OUT = (ROOT / args.output_dir).resolve()
    if OUT.parent != (ROOT / 'runs').resolve() or not OUT.name.startswith('publication_evidence_audit_20261027'):
        raise ValueError('Use a new, direct runs/publication_evidence_audit_20261027* generation')
    # This directory is one immutable generation. Do not silently refresh it.
    if OUT.exists():
        raise FileExistsError('Preserve the existing saved-evidence generation')
    registry = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    assert registry['complete_invocation_census'] is False
    assert registry['pre_registration_claim'] is False
    ids = [r['id'] for r in registry['families']]
    assert len(ids) == len(set(ids))
    for r in registry['families']:
        p = (ROOT / r['report']).resolve()
        assert p.parent == (ROOT / 'docs').resolve() and p.is_file()

    p6_path = 'runs/levir_phase6_action_utility_seed42/summary.json'
    p7_path = 'runs/levir_phase7_val_seed42/summary.json'
    b7_path = 'runs/levir_phase7_val_seed42/bootstrap.json'
    d7_path = 'runs/levir_phase7_val_seed42/decision.json'
    p6, p7, b7, d7 = [read(p) for p in (p6_path, p7_path, b7_path, d7_path)]
    table_verification = read('runs/manuscript_working_draft_20261026/tables/verification.json')
    for p in (p6_path, p7_path, b7_path, d7_path):
        assert sha(ROOT / p) == table_verification['input_sha256'][p]
    assert p6['status'] == p7['status'] == d7['status'] == 'completed'
    assert p6['split'] == 'train_gate_only'
    assert p6['gate_sources'] == 89 and p6['perturbed_instances'] == 64080
    assert p7['test_evaluated'] is False and d7['better_constant'] == 'G0'
    assert b7['unit'] == 'original_source_pair'
    assert b7['interval_order'] == ['lower', 'median', 'upper']
    assert set(b7['learned_mf1_minus_better_constant']) == set(POLICIES)
    assert set(p6['utility']['by_family_sign']) == set(FAMILIES)

    family_rows = []
    for family in FAMILIES:
        counts = p6['utility']['by_family_sign'][family]
        n = sum(counts[k] for k in ('positive', 'negative', 'zero'))
        assert n == 1424 * 5
        family_rows.append(dict(family=family, denominator=n, **counts,
            **{f'{k}_fraction': counts[k] / n for k in ('positive', 'negative', 'zero')}))
    for sign in ('positive', 'negative', 'zero'):
        assert sum(r[sign] for r in family_rows) == p6['utility'][sign]
        assert abs(mean(r[f'{sign}_fraction'] for r in family_rows)
                   - p6['utility'][sign] / 64080) < 1e-15

    policy_rows = []
    for name in (*POLICIES, 'O'):
        delta = p7['actions'][name]['corruption_mf1'] - p7['actions']['G0']['corruption_mf1']
        ci = b7['learned_mf1_minus_better_constant'][name] if name != 'O' else b7['O_mf1_minus_better_constant']
        lo, median, hi = ci
        assert lo <= median <= hi and lo <= delta <= hi
        if name != 'O':
            entry = d7['selector_feasibility']['policies'][name]
            assert entry['passes'] is False and entry['mf1_gain_over_better_constant'] == delta
            assert ci == entry['mf1_gain_bootstrap'] and delta < 0
        policy_rows.append(dict(policy=name, delta_fraction=delta,
            delta_percentage_points=100 * delta, lower_fraction=lo, median_fraction=median,
            upper_fraction=hi, interval_role='nominal_saved_source_paired_not_simultaneous',
            deployment_role='label_informed_diagnostic_only' if name == 'O' else 'learned_policy'))

    OUT.mkdir(); figdir = OUT / 'figures'; figdir.mkdir()
    write_csv(OUT / 'action_family_signs.csv', family_rows)
    write_csv(OUT / 'policy_mf1_differences.csv', policy_rows)
    write_json(OUT / 'plot_data.json', dict(families=family_rows, policies=policy_rows))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
    positions = np.arange(len(FAMILIES))
    left = np.zeros(len(FAMILIES))
    for sign, color, hatch, label in (
        ('positive', '#367FA3', '', 'Positive task-loss utility'),
        ('negative', '#D3784B', '///', 'Negative task-loss utility'),
        ('zero', '#C4C4C4', 'xx', 'Zero utility')):
        values = np.array([r[f'{sign}_fraction'] * 100 for r in family_rows])
        ax.barh(positions, values, left=left, label=label, color=color, hatch=hatch, linewidth=.4, edgecolor='white')
        left += values
    assert np.allclose(left, 100)
    ax.set(yticks=positions, yticklabels=[f.replace('_', ' ') for f in FAMILIES], xlim=(0, 100),
           xlabel='Fraction of task-loss utility signs (%)')
    ax.set_title('Fixed action utility on train-gate sources', pad=28)
    ax.invert_yaxis(); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=1, frameon=False)
    ax.text(0, 1.012, '89 sources; 7,120 repeated tile-condition instances / family',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8)
    for suffix in ('svg', 'png'): fig.savefig(figdir / f'figure1_action_signs.{suffix}', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.1), layout='constrained')
    for i, r in enumerate(policy_rows):
        diagnostic = r['policy'] == 'O'; color = '#5C5858' if diagnostic else '#367FA3'
        lo, hi, delta = [100 * r[k] for k in ('lower_fraction', 'upper_fraction', 'delta_fraction')]
        ax.hlines(i, lo, hi, color=color, linewidth=1.3)
        ax.plot([lo, hi], [i, i], '|', color=color, markersize=6)
        ax.plot(delta, i, 'D' if diagnostic else 'o', color=color,
                markerfacecolor='white' if diagnostic else color, markersize=5)
    ax.axvline(0, color='black', linewidth=.8)
    ax.axhline(7.5, color='#999999', linestyle=':', linewidth=.8)
    ax.set(yticks=np.arange(9), yticklabels=[*POLICIES, 'O (label-informed)'],
        xlabel='Corruption mF1 difference from G0 (percentage points)',
        title='Whole-tile policies on development validation')
    ax.invert_yaxis(); ax.margins(x=.12)
    ax.text(.5, -.17, 'Points: saved estimates; bars: nominal 95% source-paired intervals\n'
            '64 sources; one actor seed; correlated policies; O uses ground truth',
            transform=ax.transAxes, ha='center', va='top', fontsize=9)
    for suffix in ('svg', 'png'): fig.savefig(figdir / f'figure2_policy_differences.{suffix}', dpi=200)
    plt.close(fig)

    write_json(OUT / 'experiment_register.json', registry)
    write_csv(OUT / 'experiment_register.csv', registry['families'])
    lines = ['# Documented experiment families', '',
        'Retrospectively compiled 2026-10-27; not a complete invocation census or a preregistration.', '',
        '| Family | Data role | Disposition | Interpretation limit | Source |',
        '| --- | --- | --- | --- | --- |']
    for r in registry['families']:
        lines.append(f"| {r['id']} | {r['data_role']} | {r['disposition']} | {r['limit']} | "
            f"[{Path(r['report']).name}](<{(ROOT / r['report']).as_posix()}>) |")
    (OUT / 'experiment_register.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    inputs = [p6_path, p7_path, b7_path, d7_path,
        'runs/manuscript_working_draft_20261026/tables/verification.json',
        'configs/publication_experiment_register_20261027.yaml',
        'docs/PUBLICATION_SAVED_EVIDENCE_SCOPE_2026-10-27.md',
        *[r['report'] for r in registry['families']]]
    result = dict(status='saved_evidence_arithmetic_verified_render_review_pending',
        source_sha256=sha(Path(__file__)), input_sha256={p: sha(ROOT / p) for p in inputs},
        output_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()},
        figure_count=2, all_nine_families_included=True, all_eight_learned_policies_included=True,
        diagnostic_label_informed_O_separate=True, register_families=len(ids),
        invocation_count_unknown=True, nominal_intervals_not_multiplicity_corrected=True,
        no_new_inference=True, no_new_resampling=True, no_test_result_directory_read=True,
        no_missing_b0e_result_imputed=True, full_project_tests_pending_gpu_idle=True,
        publication_ready=False)
    write_json(OUT / 'verification.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'figure_count', 'register_families', 'publication_ready')}))


if __name__ == '__main__':
    main()
