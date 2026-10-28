"""CPU-only identity selection and exact finite-action diagnostics."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def select_original_batches(sample_ids, fit_names, calibration_names, *, batch_size,
                            count, calibration_first, salt, minimum_fit, minimum_calibration):
    if (type(batch_size) is not int or batch_size < 1 or type(count) is not int or count < 1
            or type(calibration_first) is not int or not 0 <= calibration_first <= count
            or not isinstance(salt, str) or not salt):
        raise ValueError('Invalid fixed batch selection parameters')
    if (not sample_ids or len(set(sample_ids)) != len(sample_ids)
            or set(fit_names) & set(calibration_names)):
        raise ValueError('Duplicate sample identities or overlapping source partitions')
    source = lambda sid: sid.split('@',1)[0].removeprefix('train/')
    allowed = set(fit_names) | set(calibration_names)
    if any(not isinstance(sid,str) or not sid.startswith('train/') or source(sid) not in allowed for sid in sample_ids):
        raise ValueError('Samples must belong to the original train fit/calibration partition')
    candidates=[]
    for start in range(0,len(sample_ids)-batch_size+1,batch_size):
        ids=sample_ids[start:start+batch_size]
        sources=sorted({source(sid) for sid in ids})
        candidates.append(dict(original_start=start,sample_ids=ids,source_names=sources,
            selection_sha256=canonical_sha(dict(salt=salt,sample_ids=ids)),
            contains_calibration=bool(set(sources)&set(calibration_names))))
    ordered=sorted(candidates,key=lambda r:(r['selection_sha256'],r['original_start']))
    calibration=[r for r in ordered if r['contains_calibration']]
    if len(calibration)<calibration_first: raise ValueError('Insufficient original calibration-containing batches')
    selected=calibration[:calibration_first]
    starts={r['original_start'] for r in selected}
    selected += [r for r in ordered if r['original_start'] not in starts][:count-len(selected)]
    if len(selected)!=count: raise ValueError('Insufficient complete original batches')
    selected.sort(key=lambda r:r['original_start'])
    selected_sources={s for r in selected for s in r['source_names']}
    coverage=dict(fit=len(selected_sources&set(fit_names)),calibration=len(selected_sources&set(calibration_names)))
    if coverage['fit']<minimum_fit or coverage['calibration']<minimum_calibration:
        raise ValueError('Fixed identity-only source coverage gate failed')
    return dict(batches=selected,source_coverage=coverage,selection_uses_outcomes=False,
                complete_original_batch_size=batch_size,selected_samples=count*batch_size)


def _counts(row):
    if len(row)!=4 or any(type(x) is not int or x<0 for x in row) or sum(row)<=0:
        raise ValueError('Require nonempty nonnegative integer TN/FP/FN/TP counts')
    return tuple(row)


def f1_fraction(row):
    tn,fp,fn,tp=_counts(row)
    return Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)


def optimal_finite_actions(cases, *, maximum_iterations=128):
    """Truth-informed pooled-F1 optimum over a fixed finite menu per sample.

    Uses exact rational comparisons, not a sum of per-sample F1 or guessed
    threshold. This is not a deployable action selector.
    """
    if not cases or type(maximum_iterations) is not int or maximum_iterations<1:
        raise ValueError('Nonempty finite-action cases and a positive limit are required')
    menus=[]
    for case in cases:
        if not case: raise ValueError('Each observed sample needs at least one admissible action')
        menu=[_counts(row) for row in case]
        label_axes={(r[0]+r[1],r[2]+r[3]) for r in menu}
        if len(label_axes)!=1: raise ValueError('An intervention changed the observed label/pixel support')
        menus.append(menu)
    def total(choices): return tuple(sum(menu[i][k] for menu,i in zip(menus,choices)) for k in range(4))
    choices=[0]*len(menus)
    ratio=f1_fraction(total(choices))
    for iteration in range(1,maximum_iterations+1):
        chosen=[]
        for menu in menus:
            objective=[2*r[3]*ratio.denominator-(2*r[3]+r[1]+r[2])*ratio.numerator for r in menu]
            chosen.append(max(range(len(menu)),key=lambda i:(objective[i],-i)))
        counts=total(chosen)
        new=f1_fraction(counts)
        if new<ratio: raise ArithmeticError('Fractional action optimization decreased its objective')
        if new==ratio:
            return dict(counts=list(counts),choices=chosen,f1=float(new),
                        exact_f1=[new.numerator,new.denominator],iterations=iteration,truth_informed=True)
        choices,ratio=chosen,new
    raise RuntimeError('Exact finite-action optimization did not converge within the frozen limit')
