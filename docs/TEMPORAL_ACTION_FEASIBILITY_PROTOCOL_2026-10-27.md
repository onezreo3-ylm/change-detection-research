# Frozen exploratory temporal-placement feasibility probe

2026-10-27. Frozen before reading any real A-only or B-only outcome. The question
was proposed before the old four-region real probe; that completed candidate
failed its original utility gate and remains closed. This is a distinct action
space, not a resource revision or a rescued claim about that failed candidate.

Use exactly the old eight complete original32-tile batches,256 tiles from the
same16 train-gate sources (12fit/4calibration), original clean plus45 conditions,
instances, actor/adapter checkpoints, stage2 four boxes, input ordering,
channels-last CUDA FP16 and threshold0.5. All are development data already used
in prior work. There is no fresh confirmatory holdout, raw validation/test,
Martigny, geometric transform, new model parameter, optimizer or training.

Each region chooses none/A-only/B-only/both. Residuals are computed once from
the observed A/B features by the same frozen shared adapter. The bound wrapper
owns actual inputs, model, variants and reference statistics. Same-region
alternatives and intersecting propagated supports cannot execute jointly.
Three all-region reference-replay outputs plus baseline reconstruct256 legal
assignments. This is the strong postselection information control. It is not
a deployable selector: labels are used only for diagnostic upper bounds.

Before any local result is accepted, a0 and full-both must exactly replay the
original saved counts. Every one of the old16 none/both menus must exactly
match the completed probe for every measured source/condition. A pilot actually
decodes all256 assignments on the first selected complete batch for clean and
brightness_l1. Main execution actually decodes four fixed mixed assignments
per batch and compares bitwise logits and per-sample additive confusion counts
with the reconstruction; all other assignments are explicitly derived, not
misreported as actual decodes. All three reference outputs must preserve the
outside support bitwise. Full identity checks run on session close.

Strong constant policies are none; whole-image A-only/B-only/both under original
GN; all-four-region A-only/B-only/both under reference replay. Pooled pixel F1
is computed per condition, then equally averaged across45 corruptions. Exact
rational finite-menu optimization gives the label-informed pooled-F1 upper
bound separately for old16 and new256 options. Do not average tile F1 or use
F1(expected counts) as expected F1. Report clean separately, all/fit/calibration
separately, all constants and both upper bounds, including negative results.

Before fitting can even be reconsidered, BOTH fit and calibration must pass:

- New256-action oracle mF1 minus best of all seven constant-policy mF1 >=0.005.
- New256-action oracle mF1 minus old16-action oracle mF1 >=0.002.
- At least5% positive and5% negative single-action BCE+Dice utilities, each
  sign present in at least3 corruption families.

The0.005 gate retains the earlier minimum actionable room, now against stronger
constants. The additional0.002 gate demands material extra room specifically
from temporal alternatives, rather than repackaging old regional choice. These
are exploratory resource gates, not significance levels or publication claims.
Failure closes this fixed temporal action space: no enlarged masks, alternative
source selection, threshold change or head fitting to rescue it. Passing still
does not authorize automatic fitting or establish novelty/learnability.

Shared cumulative172800s cap and all prior charges remain. This question has a
3600s subcap INCLUDING its earlier temporal prototypes/bound-interface checks,
480s engineering reserve and120s postprocessing reserve. Two pilot batches per
condition measure startup and steady cost; forecast the full main run with the
existing conservative1.5 multiplier plus180s reserve, then add pilot exhaustive
check allowance measured independently. Refuse main if it does not fit the
actual protected remaining wall. One worker, independent CPU owner/temperature
guard/watchdog, actual PID/creation/source bindings,85C confirmed/86C immediate
stop, fresh heartbeat, no automatic retry or service-quota budget reset.
