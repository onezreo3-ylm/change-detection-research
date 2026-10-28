"""Exact hard-action feasibility diagnostics from saved train-gate counts only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/spatial_action_feasibility_20261027.yaml'
KEYS = ('tn', 'fp', 'fn', 'tp')
TRANSITIONS = ('fp_to_tn', 'fn_to_tp', 'tp_to_fn', 'tn_to_fp')
NAMES = ('a0_no_correction', 'a1_shared_stage2_adapter', 'label_informed_choice')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')


def counts(value):
    if set(value) != set(KEYS) or any(type(value[k]) is not int or value[k] < 0 for k in KEYS):
        raise ValueError('Require four exact nonnegative integer counts')
    return tuple(value[k] for k in KEYS)


def add(rows):
    total = [0] * 4
    for row in rows:
        for i, x in enumerate(row): total[i] += x
    return tuple(total)


def metrics(row):
    tn, fp, fn, tp = row
    total = sum(row)
    if total <= 0: raise ValueError('Empty action support')
    def divide(a, b): return a / b if b else 0.0
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
        f1=divide(2 * tp, 2 * tp + fp + fn), iou=divide(tp, tp + fp + fn),
        oa=(tp + tn) / total, **dict(zip(KEYS, row)), pixels=total)


def checked_pair(row, pixels=None):
    a, b = counts(row['a0_counts']), counts(row['a1_counts'])
    t = row['transitions_a0_to_a1']
    if set(t) != set(TRANSITIONS) or any(type(t[k]) is not int or t[k] < 0 for k in TRANSITIONS):
        raise ValueError('Require four nonnegative integer error transitions')
    if sum(a) != sum(b) or (pixels is not None and sum(a) != pixels):
        raise ValueError('Action pixel support differs')
    if a[2] + a[3] != b[2] + b[3]: raise ValueError('Label support differs')
    if any(t[k] > a[i] for k, i in (('fp_to_tn', 1), ('fn_to_tp', 2), ('tp_to_fn', 3), ('tn_to_fp', 0))):
        raise ValueError('Error transition exceeds its source class')
    expected = (a[0] + t['fp_to_tn'] - t['tn_to_fp'],
                a[1] - t['fp_to_tn'] + t['tn_to_fp'],
                a[2] + t['tp_to_fn'] - t['fn_to_tp'],
                a[3] - t['tp_to_fn'] + t['fn_to_tp'])
    if expected != b: raise ValueError('Error transition conservation differs from observed actions')
    return a, b


def pixel_bound(row):
    """Label-informed F1 bound for selection between these two hard outputs."""
    a, _ = checked_pair(row)
    t = row['transitions_a0_to_a1']
    return (a[0] + t['fp_to_tn'], a[1] - t['fp_to_tn'],
            a[2] - t['fn_to_tp'], a[3] + t['fn_to_tp'])


def ratio(row):
    return 2 * row[3], 2 * row[3] + row[1] + row[2]


def optimal_whole_tile(pairs, limit=64):
    """Maximize pooled F1 with integer fractional updates; exact label-informed.

    Each tile has two fixed confusion vectors with the same label support.
    For q=n/d, maximize 2*d*TP - n*(TP+FP+P) by independent tile choices.
    Zero score ties choose a0. A zero residual proves the global ratio optimum.
    """
    pairs = list(pairs)
    if not pairs: raise ValueError('No action pairs')
    for a, b in pairs:
        if len(a) != 4 or len(b) != 4 or any(type(x) is not int or x < 0 for x in (*a, *b)):
            raise ValueError('Invalid confusion vector')
        if sum(a) <= 0 or sum(a) != sum(b) or a[2] + a[3] != b[2] + b[3]:
            raise ValueError('Unmatched tile or label support')
    totals = [add(p[i] for p in pairs) for i in (0, 1)]
    n0, d0 = ratio(totals[0]); n1, d1 = ratio(totals[1])
    if d0 == d1 == 0: return dict(counts=totals[0], selected_a1=0, iterations=0)
    best = totals[1] if n1 * max(d0, 1) > n0 * max(d1, 1) else totals[0]
    n, d = ratio(best)
    # An all-negative label set has F1=0 under the original convention.
    if n == 0 and best[2] == 0: return dict(counts=totals[0], selected_a1=0, iterations=0)
    for iteration in range(1, limit + 1):
        selected = [int(2 * d * (b[3] - a[3]) - n * ((b[3] + b[1]) - (a[3] + a[1])) > 0)
                    for a, b in pairs]
        pooled = add(p[s] for p, s in zip(pairs, selected))
        nn, dd = ratio(pooled)
        residual = nn * d - n * dd
        if residual < 0: raise ArithmeticError('Fractional objective decreased')
        if residual == 0:
            return dict(counts=pooled, selected_a1=sum(selected), iterations=iteration)
        n, d = nn, dd
    raise RuntimeError('Exact fractional solver did not certify convergence')


def jsonl(path, expected_sha):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for line in f:
            h.update(line)
            yield json.loads(line)
    if h.hexdigest() != expected_sha: raise ValueError('Saved action record hash differs from original summary')


def run():
    started = time.monotonic()
    v = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    base, out = ROOT / v['phase6_dir'], ROOT / v['output_dir']
    if out.exists(): raise FileExistsError('Preserve existing feasibility diagnosis')
    summary = read(base / 'summary.json'); split = read(ROOT / v['source_partition'])
    if summary['status'] != 'completed' or summary['split'] != 'train_gate_only' or summary['test_evaluated'] is not False:
        raise ValueError('Require completed original train-gate evidence')
    if v['threshold'] != .5 or v['test_or_martigny_access'] is not False:
        raise ValueError('Original fixed hard-output protocol required')
    fit = {'train/' + n for n in split['fit_names']}
    calibration = {'train/' + n for n in split['calibration_names']}
    groups = {'all': fit | calibration, 'fit': fit, 'calibration': calibration}
    if fit & calibration or any(len(groups[k]) != n for k, n in v['source_counts'].items()):
        raise ValueError('Original source partition changed')
    sample_source = {f'{s}@x{x}_y{y}': s for s in groups['all'] for x in (0, 256, 512, 768) for y in (0, 256, 512, 768)}
    clean = {}
    for r in jsonl(base / 'clean_actions.jsonl', summary['clean_actions_sha256']):
        key = r['sample_id']
        if key not in sample_source or key in clean or r['source_id'] != sample_source[key]:
            raise ValueError('Duplicate or unexpected clean sample')
        checked_pair(r, v['tile_pixels']); clean[key] = r
    if set(clean) != set(sample_source): raise ValueError('Incomplete original clean source/tile coverage')
    condition_order = list(summary['actions'][NAMES[0]]['conditions'])
    expected_ids = {f'{f}_l{l}' for f in v['families'] for l in range(1, v['levels_per_family'] + 1)}
    if len(condition_order) != v['condition_count'] or set(condition_order) != expected_ids:
        raise ValueError('Original condition axis changed')
    cells = {cid: {} for cid in condition_order}
    for r in jsonl(base / 'action_utility.jsonl', summary['raw_action_utility_sha256']):
        cid, sid = r['condition_id'], r['sample_id']
        if cid not in cells or sid not in clean or sid in cells[cid] or r['source_id'] != sample_source[sid]:
            raise ValueError('Duplicate or unexpected corrupted sample')
        a, _ = checked_pair(r, v['tile_pixels']); reference = counts(clean[sid]['a0_counts'])
        if a[2] + a[3] != reference[2] + reference[3]: raise ValueError('Corruption label support changed')
        if r['label_informed_action'] != ('a1' if r['uT'] > 0 else 'a0'):
            raise ValueError('Original loss-informed action decision differs')
        if r['foreground_pixels'] != a[2] + a[3]: raise ValueError('Foreground metadata differs')
        cells[cid][sid] = r
    if any(set(rs) != set(clean) for rs in cells.values()): raise ValueError('Incomplete original45-condition coverage')

    output = {}; rows_csv = []; maximum_iterations = 0
    for group, sources in groups.items():
        group_conditions = {}
        for cid in (*condition_order, 'clean'):
            records = list(clean.values()) if cid == 'clean' else list(cells[cid].values())
            records = [r for r in records if r['source_id'] in sources]
            pairs = [checked_pair(r) for r in records]
            actions = [add(p[i] for p in pairs) for i in (0, 1)]
            if cid != 'clean':
                actions.append(add(p[int(r['label_informed_action'] == 'a1')] for p, r in zip(pairs, records)))
            optimum = optimal_whole_tile(pairs, v['fractional_iteration_limit'])
            maximum_iterations = max(maximum_iterations, optimum['iterations'])
            pixel = add(pixel_bound(r) for r in records)
            action_metrics = {f'a{i}': metrics(a) for i, a in enumerate(actions[:2])}
            if len(actions) == 3: action_metrics['original_loss_informed_O'] = metrics(actions[2])
            action_metrics.update(optimal_whole_tile=metrics(optimum['counts']), optimal_pixel=metrics(pixel))
            best_f1 = max(action_metrics['a0']['f1'], action_metrics['a1']['f1'])
            if action_metrics['optimal_whole_tile']['f1'] + 1e-15 < best_f1 or action_metrics['optimal_pixel']['f1'] + 1e-15 < action_metrics['optimal_whole_tile']['f1']:
                raise ArithmeticError('Exact action upper-bound ordering failed')
            if len(actions) == 3 and action_metrics['optimal_whole_tile']['f1'] + 1e-15 < action_metrics['original_loss_informed_O']['f1']:
                raise ArithmeticError('Optimal tile result below original loss-informed choice')
            if group == 'all' and cid != 'clean':
                for i, action_name in enumerate(NAMES):
                    if metrics(actions[i]) != summary['actions'][action_name]['conditions'][cid]:
                        raise ValueError('Original condition metric/count replay failed')
            group_conditions[cid] = dict(actions=action_metrics, instances=len(records),
                selected_a1_whole_tile=optimum['selected_a1'], fractional_iterations=optimum['iterations'])
            for name, m in action_metrics.items():
                rows_csv.append(dict(partition=group, condition=cid, action=name, **m))
        method_names = ('a0', 'a1', 'original_loss_informed_O', 'optimal_whole_tile', 'optimal_pixel')
        averages = {n: float(np.mean([group_conditions[c]['actions'][n]['f1'] for c in condition_order])) for n in method_names}
        if group == 'all':
            for n, old in zip(method_names[:3], NAMES):
                if averages[n] != summary['actions'][old]['corruption_mf1']: raise ValueError('Original mF1 replay failed')
        gain = averages['optimal_pixel'] - max(averages['a0'], averages['a1'])
        granularity = averages['optimal_pixel'] - averages['optimal_whole_tile']
        gates = dict(pixel_gain=gain >= v['minimum_pixel_bound_gain_over_better_constant'],
            spatial_granularity=granularity >= v['minimum_pixel_minus_optimal_whole_tile_bound'])
        output[group] = dict(source_count=len(sources), mf1=averages, pixel_gain_over_better_constant=gain,
            pixel_minus_optimal_tile=granularity, resource_gates=gates, conditions=group_conditions)
    passed = all(all(output[g]['resource_gates'].values()) for g in v['required_partitions'])
    out.mkdir()
    with (out / 'conditions.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    result = dict(status='completed_exploratory_saved_train_action_space_diagnosis',
        groups=output, resource_screen_passed=passed,
        decision='prepare_observable_information_and_prior_art_comparison_only' if passed else 'stop_this_fixed_spatial_selection_candidate',
        learned_method_success=False, new_method_admitted=False, new_gpu_job_started=False,
        historical_71_18_split_is_not_new_independent_validation=True,
        upper_bound_scope='Two fixed0.5-threshold hard output masks; truth-informed choices, not deployable gains',
        original_loss_O_not_renamed_as_F1_bound=True, fractional_maximum_iterations=maximum_iterations,
        new_inference=False, new_resampling=False, raw_test_access=False,
        seconds=time.monotonic() - started, full_project_tests_pending_gpu_idle=True)
    write(out / 'analysis.json', result)
    inputs = [CONFIG, ROOT / v['scope'], ROOT / v['source_partition'], base / 'summary.json',
        base / 'action_utility.jsonl', base / 'clean_actions.jsonl']
    write(out / 'verification.json', dict(status='exact_counts_transitions_original45_replayed_and_bounds_checked',
        input_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in inputs},
        source_sha256=sha(__file__), output_sha256={'analysis.json': sha(out / 'analysis.json'), 'conditions.csv': sha(out / 'conditions.csv')},
        original_action_record_hashes_matched=True, rows=45 * 1424, clean_rows=1424,
        observed_data_scope='saved train-gate only', full_project_tests_pending_gpu_idle=True))
    return {k: result[k] for k in ('status', 'resource_screen_passed', 'decision', 'fractional_maximum_iterations', 'seconds')}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['analyze', 'smoke'])
    action = p.parse_args().action
    if action == 'smoke':
        fixture = dict(a0_counts=dict(tn=1, fp=1, fn=1, tp=1),
            a1_counts=dict(tn=1, fp=1, fn=1, tp=1), transitions_a0_to_a1=dict(
                fp_to_tn=1, fn_to_tp=1, tp_to_fn=1, tn_to_fp=1))
        bound = pixel_bound(fixture); tile = optimal_whole_tile([checked_pair(fixture)])
        if metrics(bound)['f1'] != 1 or metrics(tile['counts'])['f1'] != .5:
            raise ArithmeticError('Synthetic upper-bound smoke failed')
        result = dict(status='passed_synthetic_count_only_smoke', pixels=4,
            no_research_data_read=True, no_model_or_gpu=True, method_effectiveness=False)
    else:
        result = run()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
