"""CPU-only amortized resource estimate; no scientific score or gate changes."""
import math


def require_resource_only_revision(new,old):
    allowed={'kind','protocol','output_dir','pilot_original_batches','pilot_estimator',
             'prior_probe_metadata','prior_probe_completion'}
    changed={k for k in set(new)|set(old) if new.get(k)!=old.get(k)}
    if not changed<=allowed:raise ValueError('Resource revision changed scientific inputs or rules: '+str(sorted(changed-allowed)))
    if new.get('pilot_original_batches')!=2 or new.get('whole_probe_limit_seconds')!=3600:
        raise ValueError('Keep the cumulative original cap and fixed two-batch pilot')
    return changed


def amortized_condition_prediction(condition_batch_seconds, *, conditions, batches_per_condition,
                                   multiplier, fixed_overhead):
    if (not condition_batch_seconds or type(conditions) is not int or conditions<1
            or type(batches_per_condition) is not int or batches_per_condition<2
            or not math.isfinite(multiplier) or multiplier<1
            or not math.isfinite(fixed_overhead) or fixed_overhead<0):
        raise ValueError('Require explicit conservative resource parameters')
    for row in condition_batch_seconds:
        if len(row)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in row):
            raise ValueError('Each fixed pilot condition requires two positive full-wall batch times')
    startup=max(max(first-steady,0.) for first,steady in condition_batch_seconds)
    steady=max(row[1] for row in condition_batch_seconds)
    return dict(per_condition_fixed_seconds=startup,maximum_steady_batch_seconds=steady,
        predicted_main_seconds=multiplier*conditions*(startup+batches_per_condition*steady)+fixed_overhead,
        includes_startup_IO_and_all_batch_processing=True,uses_performance_scores=False)
