"""Independent exact breakpoint sweep, not the fractional implementation."""
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/spatial_action_feasibility_20261027'
KEYS = ('tn', 'fp', 'fn', 'tp')


def total(rows):
    return tuple(sum(row[i] for row in rows) for i in range(4))


def f1(row):
    n, d = 2 * row[3], 2 * row[3] + row[1] + row[2]
    return Fraction(n, d) if d else Fraction(0)


def solve(pairs):
    # At infinitesimal positive q, zero-intercept scores favor lower FP.
    choices = []
    events = {}
    for i, (a, b) in enumerate(pairs):
        dt, dz = b[3] - a[3], (b[3] + b[1]) - (a[3] + a[1])
        choices.append(int(dt > 0 or (dt == 0 and dz < 0)))
        if dz:
            breakpoint = Fraction(2 * dt, dz)
            if 0 < breakpoint < 1:
                events.setdefault(breakpoint, []).append(i)
    pooled = list(total([p[c] for p, c in zip(pairs, choices)]))
    best = f1(pooled)
    for q in sorted(events):
        for i in events[q]:
            old, new = choices[i], 1 - choices[i]
            for k in range(4): pooled[k] += pairs[i][new][k] - pairs[i][old][k]
            choices[i] = new
        best = max(best, f1(pooled))
    return best


def synthetic_check():
    fixtures = [
        [((1, 1, 1, 1), (1, 1, 1, 1))],
        [((2, 0, 0, 0), (0, 2, 0, 0))],
        [((1, 3, 4, 2), (3, 1, 2, 4)), ((2, 2, 3, 1), (3, 1, 2, 2))],
        [((10**20, 10, 10, 10**20), (10**20 + 1, 9, 10, 10**20))],
    ]
    for pairs in fixtures:
        brute = max(f1(total([p[c] for p, c in zip(pairs, choice)]))
                    for choice in itertools.product((0, 1), repeat=len(pairs)))
        assert solve(pairs) == brute


def main():
    synthetic_check()
    result = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8'))
    split = json.loads((ROOT / 'runs/phase7_descriptor_cache_seed42/selector_split_manifest.json').read_text(encoding='utf-8'))
    groups = dict(fit={'train/' + n for n in split['fit_names']},
        calibration={'train/' + n for n in split['calibration_names']})
    groups['all'] = groups['fit'] | groups['calibration']
    rows = {}
    base = ROOT / 'runs/levir_phase6_action_utility_seed42'
    summary = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
    for name, sha_key in [('action_utility.jsonl', 'raw_action_utility_sha256'), ('clean_actions.jsonl', 'clean_actions_sha256')]:
        h = hashlib.sha256()
        with (base / name).open('rb') as f:
            for line in f:
                h.update(line); row = json.loads(line)
                rows.setdefault(row.get('condition_id', 'clean'), []).append(row)
        assert h.hexdigest() == summary[sha_key]
    checks = 0
    for group, sources in groups.items():
        for cid, records in rows.items():
            selected = [r for r in records if r['source_id'] in sources]
            pairs = [(tuple(r['a0_counts'][k] for k in KEYS), tuple(r['a1_counts'][k] for k in KEYS)) for r in selected]
            optimal = solve(pairs)
            expected = result['groups'][group]['conditions'][cid]['actions']
            assert optimal == f1(tuple(expected['optimal_whole_tile'][k] for k in KEYS)), (group, cid)
            # Independently form the eight truth x a0 x a1 cells. The only
            # unavoidable FN/FP cells are y1/a00 and y0/a11, respectively.
            pooled = [0, 0, 0, 0]
            for r in selected:
                a, t = r['a0_counts'], r['transitions_a0_to_a1']
                true0 = [a['tn'] - t['tn_to_fp'], t['tn_to_fp'], t['fp_to_tn'], a['fp'] - t['fp_to_tn']]
                true1 = [a['fn'] - t['fn_to_tp'], t['fn_to_tp'], t['tp_to_fn'], a['tp'] - t['tp_to_fn']]
                assert min(true0 + true1) >= 0
                pooled[0] += sum(true0[:3]); pooled[1] += true0[3]
                pooled[2] += true1[0]; pooled[3] += sum(true1[1:])
            assert tuple(pooled) == tuple(expected['optimal_pixel'][k] for k in KEYS)
            checks += 1
    assert checks == 138
    guard = json.loads((base / 'guard_completion.json').read_text(encoding='utf-8'))
    for key, name in [('summary_sha256', 'summary.json'), ('guard_state_sha256', 'temperature_guard_state.json')]:
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == guard[key]
    evidence = dict(status='passed_independent_rational_event_sweep_and_eight_cell_reconstruction',
        exact_group_condition_checks=checks, synthetic_bruteforce_cases=4,
        original_phase6_guard_summary_bindings_exact=True,
        original_O_is_not_the_new_F1_bound=True, no_new_model_or_resampling=True,
        no_new_effectiveness_claim=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256((OUT / 'analysis.json').read_bytes()).hexdigest())
    with (OUT / 'independent_verification.json').open('x', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence))


if __name__ == '__main__': main()
