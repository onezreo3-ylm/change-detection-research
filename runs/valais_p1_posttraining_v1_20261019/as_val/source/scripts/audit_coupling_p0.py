"""Independent saved-result audit of P0. Standard library only; no inference.

Reads only explicitly listed train imagery to hash its bytes, never enumerate
datasets or open validation/test images. Does not establish scientific validity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

COUNTS = ('tn', 'fp', 'fn', 'tp')
CELLS = ('uu', 'uv', 'vu', 'vv')
TRANSITIONS = ('tp_to_fn', 'fn_to_tp', 'tn_to_fp', 'fp_to_tn',
               'clean_tp', 'clean_fn', 'clean_tn', 'clean_fp')
MODELS = ('b0', 'b1', 'b0e')
FAMILIES = ('brightness', 'contrast', 'gamma', 'color_bias')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def integers(row, keys):
    require(set(row) == set(keys), 'Unexpected count/transition keys')
    require(all(type(row[k]) is int and row[k] >= 0 for k in keys),
            'Counts and transitions must be nonnegative integers')


def sum_fields(rows, keys):
    output = dict.fromkeys(keys, 0)
    for row in rows:
        integers(row, keys)
        for key in keys:
            output[key] += row[key]
    return output


def metrics(c):
    tn, fp, fn, tp = (c[k] for k in COUNTS)
    div = lambda a, b: a / b if b else 0.0
    return dict(precision=div(tp, tp+fp), recall=div(tp, tp+fn),
                f1=div(2*tp, 2*tp+fp+fn), iou=div(tp, tp+fp+fn),
                oa=div(tp+tn, tp+tn+fp+fn), fpr=div(fp, tn+fp),
                pixels=tp+tn+fp+fn)


def equal_metrics(actual, expected):
    require(set(actual) == set(expected), 'Metric schema mismatch')
    for key, value in expected.items():
        require(isinstance(actual[key], (int, float)) and
                math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12),
                f'Metric mismatch: {key}')


def check_record(row, sample):
    require(row['foreground_pixels'] == sample['foreground_pixels'] and
            row['stratum'] == sample['stratum'], 'Sample stratum/area changed')
    diagnostic = row['diagnostic']
    conditions = [c for c in sample['conditions'] if
                  c['family'] == row['family'] and c['level'] == row['level']]
    require(len(conditions) == 1, 'Missing/duplicate sealed condition')
    condition = conditions[0]
    images = condition['images']
    require(set(images) == {'au', 'av', 'bu', 'bv'} and
            all(isinstance(h, str) and len(h) == 64 and set(h) <= set('0123456789abcdef')
                for h in images.values()), 'Invalid sealed pixel hashes')
    require(condition['marginal_hashes'] == {
        'uu': [images['au'], images['bu']], 'uv': [images['au'], images['bv']],
        'vu': [images['av'], images['bu']], 'vv': [images['av'], images['bv']]},
        'Sealed four-cell marginal reuse mismatch')
    a_same, b_same = images['au'] == images['av'], images['bu'] == images['bv']
    require(row['marginal_check'] == dict(matched=True, a_instances_identical=a_same,
                b_instances_identical=b_same, fully_degenerate=a_same and b_same),
            'Saved marginal report differs from sealed pixel hashes')
    require(set(diagnostic['cells']) == set(CELLS), 'Four cells required')
    clean = diagnostic['clean']['counts']
    integers(clean, COUNTS)
    equal_metrics(diagnostic['clean']['metrics'], metrics(clean))
    fg = sample['foreground_pixels']
    require(clean['tp'] + clean['fn'] == fg and
            clean['tn'] + clean['fp'] == 65536-fg, 'Clean pixel conservation failed')
    for cell in CELLS:
        item = diagnostic['cells'][cell]
        c, t = item['counts'], item['transitions']
        integers(c, COUNTS)
        integers(t, TRANSITIONS)
        require(c['tp']+c['fn'] == fg and c['tn']+c['fp'] == 65536-fg,
                'Cell foreground/background pixel conservation failed')
        for key in COUNTS:
            require(t['clean_'+key] == clean[key], 'Transition clean denominator changed')
        for transition, denominator in [('tp_to_fn', 'tp'), ('fn_to_tp', 'fn'),
                                         ('tn_to_fp', 'tn'), ('fp_to_tn', 'fp')]:
            require(t[transition] <= clean[denominator], 'Transition exceeds clean denominator')
        require(c['tp'] == clean['tp']-t['tp_to_fn']+t['fn_to_tp'] and
                c['fn'] == clean['fn']+t['tp_to_fn']-t['fn_to_tp'] and
                c['tn'] == clean['tn']-t['tn_to_fp']+t['fp_to_tn'] and
                c['fp'] == clean['fp']+t['tn_to_fp']-t['fp_to_tn'],
                'Clean-to-cell transition identities failed')
        equal_metrics(item['metrics'], metrics(c))
    if a_same and b_same:
        require(all(diagnostic['cells'][k] == diagnostic['cells']['uu'] for k in CELLS),
                'Fully degenerate cells must have identical counts/transitions')
    for group, cells in [('D', ('uu', 'vv')), ('O', ('uv', 'vu'))]:
        expected_c = sum_fields((diagnostic['cells'][k]['counts'] for k in cells), COUNTS)
        expected_t = sum_fields((diagnostic['cells'][k]['transitions'] for k in cells), TRANSITIONS)
        g = diagnostic['groups'][group]
        require(g['counts'] == expected_c and g['transitions'] == expected_t,
                'D/O does not equal pooled four-cell counts/transitions')
        equal_metrics(g['metrics'], metrics(expected_c))
    for key in ('f1', 'recall', 'fpr'):
        delta = diagnostic['groups']['D']['metrics'][key]-diagnostic['groups']['O']['metrics'][key]
        require(math.isclose(diagnostic['D_minus_O'][key], delta, abs_tol=1e-12), 'D/O delta mismatch')
    for effect in ('m', 'a', 'b', 'i'):
        stats = diagnostic['logit_effects'][effect]
        for area, expected_n in [('all', 65536), ('foreground', fg), ('background', 65536-fg)]:
            s = stats[area]
            require(s['pixels'] == expected_n, 'Logit effect pixel conservation failed')
            require(all(math.isfinite(s[k]) for k in ('sum', 'abs_sum', 'square_sum')) and
                    s['abs_sum'] >= 0 and s['square_sum'] >= 0, 'Invalid logit summary')
            if expected_n:
                equal_metrics({k: s[k] for k in ('mean', 'mean_abs', 'rms')},
                    dict(mean=s['sum']/expected_n, mean_abs=s['abs_sum']/expected_n,
                         rms=math.sqrt(s['square_sum']/expected_n)))
            else:
                require(all(s[k] is None for k in ('mean', 'mean_abs', 'rms')) and
                        all(s[k] == 0 for k in ('sum', 'abs_sum', 'square_sum')), 'Empty-area summary invalid')
        for key in ('sum', 'abs_sum', 'square_sum'):
            require(math.isclose(stats['all'][key], stats['foreground'][key]+stats['background'][key],
                                 rel_tol=1e-10, abs_tol=1e-8), 'Logit subgroup sums do not conserve')
    require(row['marginal_check']['matched'] is True, 'Marginal identity check failed')


def audit_records(manifest, report, rows):
    require(report['status'] == 'completed_train_p0_diagnostic' and report['smoke_only'] is False,
            'Only fully completed research diagnostic can be audited')
    require(report['train_only'] is True and report['val_accessed'] is False and
            report['test_accessed'] is False and report['training'] is False and
            report['gradients_enabled'] is False and report['amp'] is False, 'Execution scope changed')
    require(report['records'] == 2304 and report['main_predictions'] == 9408 and
            report['predictions'] == 9414, 'Full inference coverage not reached')
    require(report['budget_seconds'] == manifest['settings']['max_seconds'] and
            0 <= report['elapsed_seconds'] <= report['budget_seconds'] <= 21600 and
            0 <= report['maximum_observed_temperature_c'] < 85, 'Execution guard failed')
    checks = report['self_checks']
    require(len(checks) == 3 and {c['model'] for c in checks} == set(MODELS) and
            all(c['repeat_bitwise_equal'] is True and c['temporal_swap_allclose'] is True and
                math.isfinite(c['max_swap_abs_error']) and c['max_swap_abs_error'] >= 0 for c in checks),
            'Numerical self checks incomplete')
    samples = {s['id']: s for s in manifest['samples']}
    require(len(samples) == 64 and len(manifest['samples']) == 64 and
            sum(s['foreground_pixels'] > 0 for s in samples.values()) == 16,
            'Expected unique 16 positive / 48 empty train sources')
    expected = {(m, sid, f, level) for m in MODELS for sid in samples
                for f in FAMILIES for level in (1, 3, 5)}
    observed, clean_by_model_source = set(), {}
    buckets = defaultdict(list)
    for row in rows:
        identity = (row['model'], row['source_id'], row['family'], row['level'])
        require(identity in expected and identity not in observed, 'Unexpected or duplicate record identity')
        observed.add(identity)
        check_record(row, samples[row['source_id']])
        clean_key = identity[:2]
        clean = row['diagnostic']['clean']['counts']
        require(clean_by_model_source.setdefault(clean_key, clean) == clean,
                'Clean prediction counts changed across conditions')
        for label in ('all', 'family:'+row['family'], 'stratum:'+row['stratum'],
                      f"condition:{row['family']}:{row['level']}"):
            buckets[(row['model'], label)].append(row)
    require(observed == expected and len(rows) == 2304, 'Incomplete Cartesian record coverage')
    independent = {}
    for (model, label), selected in buckets.items():
        reported = report['aggregates'][model][label]
        require(reported['records'] == len(selected), 'Aggregate record count mismatch')
        groups = {}
        for group, cells in [('D', ('uu', 'vv')), ('O', ('uv', 'vu'))]:
            # Independently reconstruct from original cells, never trust saved groups.
            c = sum_fields((r['diagnostic']['cells'][cell]['counts'] for r in selected for cell in cells), COUNTS)
            t = sum_fields((r['diagnostic']['cells'][cell]['transitions'] for r in selected for cell in cells), TRANSITIONS)
            g = reported['groups'][group]
            require(g['counts'] == c and g['transitions'] == t, 'Report aggregate counts/transitions mismatch')
            equal_metrics(g['metrics'], metrics(c))
            groups[group] = {'metrics': metrics(c), 'transitions': t}
        independent[(model, label)] = groups
    gate_models = {}
    for model in ('b1', 'b0e'):
        passing = []
        for family in sorted(FAMILIES):
            groups = independent[(model, 'family:'+family)]
            harmful = sum(groups[g]['transitions'][t] for g in ('D', 'O') for t in ('tp_to_fn', 'tn_to_fp'))
            if abs(groups['D']['metrics']['f1']-groups['O']['metrics']['f1']) >= .005 and harmful > 0:
                passing.append(family)
        gate_models[model] = dict(qualifying_families=passing, passed=len(passing) >= 2)
    require(report['gate']['models'] == gate_models and
            report['gate']['passed'] == all(g['passed'] for g in gate_models.values()), 'Independent gate mismatch')
    for model in MODELS:
        clean = sum_fields((c for (m, _), c in clean_by_model_source.items() if m == model), COUNTS)
        r = report['aggregates'][model]['clean_unique_sources']
        require(r['sources'] == 64 and r['counts'] == clean, 'Unique clean aggregate mismatch')
        equal_metrics(r['metrics'], metrics(clean))
    return {'records': len(rows), 'sources': 64, 'models': 3,
            'independent_gate': {'models': gate_models, 'passed': all(g['passed'] for g in gate_models.values())}}


def verify_manifest_files(manifest):
    require(manifest['split'] == 'train', 'Manifest is not train-only')
    root = Path(manifest['data_root']).resolve()
    forbidden = {'val', 'validation', 'test'}
    checked_images = []
    for sample in manifest['samples']:
        require(sample['id'] == f"train/{sample['name']}@x0_y0", 'Non-train source identity')
        for key, folder in [('image_a', 'A'), ('image_b', 'B'), ('mask', 'label')]:
            item = sample['files'][key]
            path = Path(item['path']).resolve()
            require(path.parent == root / 'train' / folder and path.name == sample['name'] and
                    not forbidden.intersection(p.lower() for p in path.parts), 'Non-train image path forbidden')
            checked_images.append((path, item['sha256']))
    # Validate ALL image scopes before opening the first image byte.
    bindings = [(Path(p), h) for p, h in manifest['source_hashes'].items()]
    bindings.append((Path(manifest['config_path']), manifest['config_sha256']))
    for model in manifest['models'].values():
        bindings.extend([(Path(model['checkpoint']), model['checkpoint_sha256']),
                         (Path(model['config']), model['config_sha256'])])
    for path, expected in checked_images + bindings:
        require(sha256(path) == expected, f'Frozen file SHA256 changed: {path}')
    return {'train_image_hashes_verified': len(checked_images), 'source_model_config_hashes_verified': len(bindings)}


def verify_review(manifest, report, manifest_path, review_path):
    """Verify accepted review and every exact, locally derived train QA artifact."""
    manifest_path, review_path = Path(manifest_path).resolve(), Path(review_path).resolve()
    require(report['review_sha256'] == sha256(review_path), 'Report review SHA256 mismatch')
    review = json.loads(review_path.read_text(encoding='utf-8'))
    require(review['manifest_sha256'] == sha256(manifest_path) and
            review['status'] == 'accepted_train_only_sampled_visibility' and
            bool(review['reviewer']) and bool(review['limitations']), 'Review acceptance binding invalid')
    expected = [(str(Path(x['path']).resolve()), x['sha256']) for x in manifest['qa_images']]
    observed = [(str(Path(x['path']).resolve()), x['sha256']) for x in review['reviewed_images']]
    require(expected and len(set(expected)) == len(expected) and len(set(observed)) == len(observed) and
            sorted(expected) == sorted(observed) and all(x.get('findings') for x in review['reviewed_images']),
            'Review must cover exactly every sealed QA image once')
    qa_root = manifest_path.parent / 'previews'
    for path, _ in expected:
        p = Path(path)
        require(p.parent == qa_root and p.suffix.lower() == '.png' and
                not {'val', 'validation', 'test'}.intersection(x.lower() for x in p.parts),
                'QA path is outside the derived train preview folder')
    for path, digest in expected:
        require(sha256(path) == digest, 'Reviewed QA image SHA256 changed')
    return {'qa_review_sha256': sha256(review_path), 'reviewed_qa_hashes_verified': len(expected)}


def audit(manifest_path, report_path, records_path, smoke_report_path, output_path, review_path=None):
    output_path = Path(output_path)
    require(not output_path.exists(), 'Audit output already exists; refusing overwrite')
    paths = dict(manifest=Path(manifest_path), report=Path(report_path), records=Path(records_path),
                 smoke_report=Path(smoke_report_path))
    paths['review'] = Path(review_path) if review_path else paths['manifest'].parent / 'visual_review.json'
    manifest, report, smoke = [json.loads(paths[k].read_text(encoding='utf-8'))
                               for k in ('manifest', 'report', 'smoke_report')]
    manifest_hash = sha256(paths['manifest'])
    require(report['manifest_sha256'] == manifest_hash and smoke['manifest_sha256'] == manifest_hash,
            'Reports do not bind this manifest')
    require(smoke['status'] == 'engineering_clean_cuda_smoke_passed' and smoke['smoke_only'] is True and
            smoke['val_accessed'] is False and smoke['test_accessed'] is False and smoke['predictions'] == 9,
            'Completed clean CUDA smoke required')
    rows = [json.loads(line) for line in paths['records'].read_text(encoding='utf-8').splitlines() if line.strip()]
    result = audit_records(manifest, report, rows)
    result.update(verify_review(manifest, report, paths['manifest'], paths['review']))
    result.update(verify_manifest_files(manifest))
    result.update(status='passed_saved_result_audit', test_accessed=False, val_accessed=False,
                  model_inference_performed=False,
                  inputs={k: {'path': str(p.resolve()), 'sha256': sha256(p)} for k, p in paths.items()},
                  limitations='Saved counts and hashes verified; no independent prediction rerun, visibility or novelty acceptance.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('manifest', 'report', 'records', 'smoke-report', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--review', type=Path, help='Defaults to visual_review.json beside the manifest')
    args = parser.parse_args()
    result = audit(args.manifest, args.report, args.records, args.smoke_report, args.output, args.review)
    print(json.dumps({'status': result['status'], 'records': result['records']}))


if __name__ == '__main__':
    main()
