# Phase 7 whole-tile action selector protocol v1 — draft for review

Date: 2026-09-22. Status: **approved by the user and frozen for Phase 7 v1**.

Approval record: after reviewing the completed draft with SHA256
`f1909323547bd0fef02415b312d367bbd6c751f3fead72689bb1a8ea161923f0`,
the user replied `批准，请继续` in the current Codex task. The approval covers only
the bounded Phase 7 workflow and stop conditions in this document.

This document defines the next bounded experiment after the completed Phase 5-6
diagnostic. The approval authorizes its implementation, train-only descriptor
generation, selector training, pre-val seal, and one official-validation scoring
run. It does not authorize Phase 8-10 or held-out test access.

## 1. Evidence and bounded research question

The frozen Phase 6 diagnostic found a genuine but small whole-tile action-selection
space on the 89 train-only gate sources:

- `uT > 0`: 43,949 / 64,080 instances (`0.6858458177`);
- `uT < 0`: 19,848 / 64,080 instances (`0.3097378277`);
- both signs occur in all nine accepted corruption families;
- `a0` corruption mF1: `0.8930599229`;
- `a1` corruption mF1: `0.8939128296`;
- label-informed actual-action mF1: `0.8960960698`;
- label-informed margin over the better constant action: `0.0021832402`, with
  paired-source 95% interval `[0.0014987865, 0.0030834679]`.

The Phase 7 question is therefore deliberately narrow:

> Can a small inference-time selector, using only information computable from the
> observed image pair and the frozen actor/adapter, choose between the already
> executable whole-tile actions `a0` and `a1` better than both constant actions,
> without materially harming clean change detection or change recall?

Phase 7 is not a test of local actions, expert banks, perturbation recognition,
joint training, or held-out generalization. It does not establish novelty. In
particular, G2 and G3 contain equivalent target information and are treated as a
parameterization/control comparison.

## 2. Immutable upstream evidence

Implementation must stop before loading data if any required hash differs.

| Artifact | SHA256 |
| --- | --- |
| `docs/PHASE5_6_PROTOCOL_FOR_REVIEW_2026-09-15.md` | `63acfd4815360fe6c732086c5d80dfebd6e85c136cbe43dc9555c6dd00ea11fa` |
| `configs/phase5_6_v1_seed42.yaml` | `12820dc7753884e54eda461e718fd3aad31f914dbf60333cfbabf413f28ebf40` |
| Phase 5-6 actor/gate manifest | `778f0a87200760f7c9102388d59bef96f9633e51b6de3334caf0c503447819d1` |
| Phase 6 gate-action manifest | `ac7daa655fa9b535748d8628942a832d2ea7a8e23eb3d7768a6dc40065e38796` |
| actor `best.pt` | `20e53ddc95a6e510c8ed988621112965f51669abdcb517bba858b5b8bb095de1` |
| actor `status.json` | `470bfd2d4661632a75adbee6db2ce6263b64bc546391a41d9e73a3fbe3778023` |
| actor `guard_completion.json` | `0caf4fee738e104eab9173ff3a49f06c31ec6f5c96c35173af6f35eaeeee7c4a` |
| adapter `best.pt` | `7447184bb1852cb03dd1bebe5e681ea2165d2fae2425348457ad820fe5e103dd` |
| adapter `status.json` | `040bad7b42a53fe5df6195f9973ec1d9d92b8b8c68a5d64466ae4e6d1078ada8` |
| adapter `guard_completion.json` | `71cfc27e38d6f4768fce41eae73d93f7ba130a74c64d9906e103269b3ec5242b` |
| Phase 6 `action_utility.jsonl` | `0d6753a8e0fb7a85642f5614f642841a6d797e8fa6382856605e81c7a97fa3ce` |
| Phase 6 `clean_actions.jsonl` | `fdcf43dced94e9e3afd5dd5ed825b37fc6c86492de5a0c083c1304650d5caadb` |
| Phase 6 `status.json` | `1d708d03fd0432397add9274692a2a4fbfa5084482235b4da13bf120acab890e` |
| Phase 6 `gate_decision.json` | `dc4307d215e5b7d513d98a922eee5cbaa2219f1715227fb76cecb800cee573fe` |
| accepted official-val corruption manifest | `951e1498f2d538d55be91ea2629ddde7edf1af6651bf97184db18ce2b7bc0039` |

The actor, adapter, decoder, and every BatchNorm state remain frozen in `eval()`.
No selector gradient may reach them. The actor checkpoint remains epoch 45 and the
adapter checkpoint remains epoch 11.

## 3. Data roles and leakage boundary

| Sources | Role in Phase 7 | Permitted use |
| --- | --- | --- |
| 356 actor train sources | closed to selector fitting | frozen actor/adapter provenance only |
| 71 gate-fit train sources | selector fitting | descriptor normalization and weight fitting |
| 18 gate-calibration train sources | model/threshold calibration | checkpoint selection and policy threshold only |
| 64 official val sources | one-shot Phase 7 development comparison | scoring after every selector choice is sealed |
| 128 official test sources | closed | no enumeration, loading, hashing, scoring, or selection |

All partitions are by original 1024x1024 source pair. Its 16 tiles, clean view,
all 45 perturbations, cached descriptors, labels, and action outputs inherit the
same partition. No related tile or transformed copy may cross a boundary.

### 3.1 Deterministic 71/18 split of the existing gate sources

The split uses only the already recorded source-level foreground count and filename;
it does not use Phase 6 utility, corruption outcome, or selector result.

1. Take the 89 sources whose frozen partition is `gate`.
2. Sort by `(foreground_pixel_count, canonical_filename)`.
3. Form consecutive groups of five; the last group contains four sources.
4. In each of the 18 groups, assign the source with the smallest
   `SHA256("phase7_gate_fit_cal_v1_seed42\\0" + canonical_filename)` to
   gate-calibration; assign the others to gate-fit.
5. Here `\\0` is the two ASCII characters backslash and zero, matching the explicit
   spelling convention used by the approved Phase 5-6 split. It is not a NUL byte.

A read-only pre-review reproduction gives:

| Partition | Sources | Tiles | Foreground pixels | Foreground fraction | Zero-FG sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate-fit | 71 | 1,136 | 3,387,918 | 0.0455066251 | 7 |
| gate-calibration | 18 | 288 | 879,331 | 0.0465886328 | 2 |

The derived calibration source names are:

```text
train_5.png, train_9.png, train_33.png, train_34.png, train_52.png,
train_65.png, train_70.png, train_103.png, train_114.png, train_122.png,
train_152.png, train_201.png, train_212.png, train_283.png, train_306.png,
train_347.png, train_375.png, train_443.png
```

The Phase 6 utilities were not used to select these names. A post-split audit found
positive/negative `uT` fractions of `0.6818/0.3131` in gate-fit and
`0.7018/0.2965` in gate-calibration. These are audit results, not balancing inputs.

### 3.2 Official validation firewall

The official val actor/adapter clean checkpoint selection has already occurred, so
val is a development split rather than a final independent test. Nevertheless,
Phase 7 selector weights, target normalization, checkpoint epoch, policy threshold,
and model ranking must all be sealed before any Phase 7 selector output is computed
on val. After the one-shot val run, no Phase 7 hyperparameter may be changed and
rescored under v1. A change requires a versioned v2 protocol and must label the v1
val result as already observed.

## 4. Exact examples and targets

For an observed pair `z`, define its actual correction utility

```text
u_obs(z) = L(a0, z) - L(a1, z)
```

using the same per-sample BCE plus soft-Dice task loss as Phase 6. Positive utility
means that the actually executed `a1` action has lower task loss.

For every Phase 6 perturbed record `z = Tx`:

```text
u_obs   = uT
u_clean = u0
a_obs   = uT - u0
```

For the corresponding clean observation `z = x`:

```text
u_obs   = u0
u_clean = u0
a_obs   = 0
```

This extension preserves `a_obs = u_obs - u_clean` exactly and gives the selector
an explicit clean-input training case without inventing a new action. Clean and
perturbed targets remain labels only; they are never selector inputs.

Gate-fit contains 51,120 perturbed records and 1,136 unique clean records.
Gate-calibration contains 12,960 perturbed records and 288 unique clean records.

## 5. Inference-available selector descriptor

The selector receives neither pixels nor action predictions. It uses a fixed,
date-swap-invariant descriptor computed before the decoder, allowing exactly one
decoder branch to execute after selection.

Let `FA, FB` be the frozen stage-2 encoder tensors and let
`RA = R(FA), RB = R(FB)` be the outputs of the frozen shared adapter residual branch
before residual addition. Define

```text
S = (FA + FB) / 2
D = abs(FA - FB)
C = (RA + RB) / 2
E = abs(RA - RB)
```

For each of `S,D,C,E`, concatenate its 128 channel-wise spatial means and 128
channel-wise population standard deviations (`unbiased=False`). The result is a
fixed 1,024-dimensional float32 vector:

```text
d(z) = [mean(S), std(S), mean(D), std(D),
        mean(C), std(C), mean(E), std(E)]
```

Consequences and boundaries:

- exchanging dates A/B must produce bitwise-identical descriptors under the fixed
  deterministic runtime;
- the adapter residual is computed for every input, but only the selected decoder
  path is run; all reported latency must include encoder, adapter, selector, and
  selected decoder costs;
- no corruption family, severity, direction, target date, spatial mask, overlap,
  filename, source identity, paired clean descriptor, label, `uT`, `u0`, or `a`
  enters the selector input;
- a0/a1 logits and masks are excluded from v1 input, avoiding a mandatory double
  decoder pass before the decision;
- descriptors are cached only after source-partition checks and carry source ID,
  sample ID, observation kind, manifest record ID, and file/content hashes.

Each descriptor dimension is standardized with a weighted mean and population
standard deviation computed on gate-fit only. Standard deviations have a `1e-6`
floor. Gate-calibration and val reuse the frozen gate-fit statistics.

## 6. Selector architecture and initialization

Every learned control uses the same two-output network, including variants that
use only the first output:

```text
1024
 -> Linear(1024,128,bias=True)
 -> LayerNorm(128)
 -> GELU
 -> Linear(128,64,bias=True)
 -> LayerNorm(64)
 -> GELU
 -> Linear(64,2,bias=True)
```

There is no dropout, BatchNorm, attention, convolution, residual block, or metadata
embedding. The network has 139,970 parameters. All linear weights use Xavier
uniform initialization, linear biases are zero, and LayerNorm scale/bias are one/
zero. Seed 42, the initialized tensor hash, and the exact shuffled record order are
shared across learned controls. An unused second output remains present so G1 and
the direct classifier have exactly the same parameter count as G2/G3.

## 7. Required controls

| ID | Training target or policy | Purpose |
| --- | --- | --- |
| A0 | always `a0` | no-correction constant |
| G0 | always `a1` | always-correction constant |
| C1 | direct `1[u_obs>0]` sign classifier | ordinary class-weighted action-selection control |
| G1 | regress `u_obs`; second output masked | ordinary direct-utility gate |
| G2 | regress `(u_obs, u_clean)` | paired clean-utility auxiliary target |
| G3 | regress `(u_obs, a_obs)` | equivalent paired-difference parameterization |
| O | choose the lower-loss actually executed action with labels | diagnostic reference only |

`O` is not deployable, is not used for training or threshold selection, and is not
called a theoretical upper bound.

For every unconstrained policy, the de-standardized first output is the action
score. For the harm-constrained paired policies, G2 uses its predicted
`u_clean`, while G3 first derives the information-equivalent
`u_clean_hat = u_obs_hat - a_obs_hat`. Their constrained score is identically

```text
s_H = min(u_obs_hat, u_clean_hat)
```

so G2 and G3 have the same policy class and differ only in target
parameterization/optimization. C1 and G1 have no clean-utility head, so their
constrained score remains the primary benefit score. They nevertheless receive
the identical empirical clean-safety budget and threshold search below. This is
the strong ordinary-selector control for whether explicitly predicting paired
clean risk adds value.

Each learned control produces two policies from the same selected checkpoint:

- `-U` (unconstrained): choose `a1` iff the primary score is strictly above zero;
- `-H` (harm-constrained): use the deterministic gate-calibration threshold in
  Section 10.

The same `-H` feasibility limits, threshold grid, objective, and tie rules are
applied to C1, G1, G2, and G3. For C1, the raw benefit logit is the score and zero
is the unconstrained threshold.

## 8. Target scaling and losses

### 8.1 Equal clean/perturbed contribution

An epoch visits each of the 51,120 gate-fit perturbed records exactly once. Each
record is paired with its source tile's cached clean descriptor. Its contribution is

```text
0.5 * loss(perturbed observation) + 0.5 * loss(clean observation)
```

Thus clean and perturbed observations each contribute half of the total objective;
the repeated clean descriptor does not create a new independent sample. Metrics and
bootstrap resampling continue to use original source pairs, never these repetitions.

Target means and population standard deviations use these same weights and gate-fit
only. Each regression component is independently standardized with a `1e-6` scale
floor. Predictions are converted back to original task-loss units before applying
the zero threshold or reporting errors.

### 8.2 Regression controls

G1 uses Smooth L1 loss with `beta=1.0` in standardized target space. G2 and G3 use
the arithmetic mean of the primary and auxiliary Smooth L1 losses, with the same
`beta`. There is no sign loss, ranking loss, oracle-F1 loss, or family reweighting.

The robust loss is predeclared because the observed train-only Phase 6 utilities
are strongly concentrated near zero but have legitimate heavy tails: `uT` standard
deviation is about `0.0849`, while its observed range is approximately
`[-2.0856, 1.0499]`. No record is clipped or winsorized in v1.

### 8.3 Direct classification control

C1 uses binary cross-entropy with logits on `1[u_obs>0]`. Exact zero utilities use
target zero (`a0` wins ties). Positive and non-positive class weights are inverse
to their weighted gate-fit frequencies and normalized so the expected per-record
weight is one. No utility magnitude enters C1.

This is an action-selection class-weight control; it does not retrain the actor or
adapter with a different foreground-weighted segmentation loss. Such retraining
would change the fixed action set and therefore requires a separate protocol.

## 9. Optimization and checkpoint selection

- device: real CUDA only; no CPU or synthetic substitution;
- dtype: float32; AMP disabled for the small selector;
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`;
- schedule: cosine decay to `1e-6`;
- batch: 256 perturbed records, each paired with its clean observation;
- epochs: 100; `drop_last=false` (200 attempted updates per epoch);
- total attempted updates per learned control: 20,000;
- gradient norm clipping: `1.0`;
- deterministic algorithms and data order: strict, seed 42;
- checkpoints: every complete epoch, including optimizer/scheduler/RNG state;
- resume: complete-epoch exact resume only;
- temperature guard: the already frozen guarded-execution mechanism is mandatory.

For G1-G3, checkpoint selection uses gate-calibration primary-target Smooth L1
only, not the auxiliary loss, sign accuracy, F1, or official val. For C1 it uses
unweighted gate-calibration binary log loss. Ties choose the earliest epoch. All
100 epochs are still executed so variants have identical optimization budgets.

No corruption family, level, or source is reweighted. No hyperparameter search is
performed. A numerical or hash failure stops the run rather than substituting a
different setting.

## 10. Identical harm-constrained threshold calibration

The `-H` policy is a deterministic post-checkpoint threshold calibration, not a
separately trained model. It uses only the 18 gate-calibration sources.

For C1/G1 the constrained score is the primary score. For G2/G3 it is the matched
`s_H` defined in Section 7. Candidate thresholds are `+infinity`, `-infinity`, and
the 1,001 empirical constrained-score quantiles at probabilities
`0,0.001,...,1.000`, deduplicated after float64 quantile calculation. For each
candidate, execute the corresponding whole-tile a0/a1 choices using cached actual
action outputs.

A threshold is feasible only if its clean gate-calibration result satisfies both:

```text
clean F1(policy)     >= clean F1(a0)     - 0.0005
clean recall(policy) >= clean recall(a0) - 0.0010
```

Among feasible thresholds, choose the one with maximum equal-weight 45-condition
corruption mF1 on gate-calibration. Ties are resolved by, in order:

1. higher clean recall;
2. higher clean F1;
3. lower selected negative-utility mass;
4. larger threshold, hence fewer corrections.

The resulting scalar threshold is sealed with the checkpoint and normalization
hashes. The official val split cannot change it. The same feasibility limits,
candidate grid, objective, and tie rules apply to C1 and G1-G3.

The clean limits are motivated by the completed train-only diagnostic: on all 89
gate sources, always-correction improved clean F1 from `0.8973447` to `0.8983447`
but reduced clean recall from `0.8924771` to `0.8871842`. Phase 7 must not turn that
tradeoff into an unrecorded source of apparent robustness.

## 11. Guarded execution order

Approval of this document would authorize only the following ordered Phase 7 v1
workflow:

1. Implement the split, descriptor, selector, losses, cache format, threshold
   calibration, evaluator, and source-pair bootstrap as small testable modules.
2. Run the full unit suite and synthetic engineering checks.
3. Run a real-CUDA, train-only guarded smoke that exercises descriptor extraction,
   one optimizer step for each learned control, checkpoint resume, and both actions.
4. Materialize and hash the 71/18 source manifest. Abort on any cross-source leak.
5. Generate only gate-fit/gate-calibration descriptors and actual-action caches;
   verify finite values, A/B swap invariance, record counts, and provenance hashes.
6. Train C1 and G1-G3 for the complete matched budget; select checkpoints and
   calibrate all `-H` thresholds without touching official val.
7. Write a sealed pre-val manifest containing every source/config/code/cache/
   checkpoint/threshold hash and the complete list of evaluation variants.
8. Only after that seal, run one guarded official-val evaluation of all variants,
   constants, and the diagnostic label-informed reference.
9. Produce the Phase 7 report, rerun the full unit suite and a final consistency
   smoke, then stop. Do not proceed automatically to Phase 8.

A technical crash may resume from an exact checkpoint or restart an incomplete
cache shard under the recorded idempotent rules. It may not justify changing the
protocol after any val selector result is visible.

## 12. Official-val evaluation

The accepted val manifest contains 64 original pairs, 1,024 tiles, 45 conditions,
and 46,080 perturbed instances. Each learned policy must actually select and decode
one whole-tile action. Pixelwise mixing of a0/a1 predictions is forbidden.

### 12.1 Required metrics

Report clean and corruption results for A0, G0, C1-U/H, G1-U/H, G2-U/H, G3-U/H,
and O:

- pooled precision, recall, F1, IoU, OA, and background FPR;
- equal-weight 45-condition corruption mF1;
- action-a1 selection rate overall and by family, level, source, target date,
  spatial mode, direction, and injection overlap;
- selected positive-utility fraction and negative-utility fraction;
- captured positive utility mass and selected negative utility mass;
- mean, median, 95th, 99th, and maximum decision regret, where
  `regret=max(u_obs,0)-1[choose a1]*u_obs`;
- `FP->TN`, `FN->TP`, `TP->FN`, and `TN->FP` counts with correct denominators,
  overall and for clean inputs;
- per-family/level/source action and metric tables;
- selector regression MAE, Smooth L1, sign accuracy, balanced accuracy, AUROC and
  calibration curves as diagnostic metrics only;
- wall time, peak GPU memory, parameter count, descriptor/cache size, selected
  decoder counts, and amortized per-tile latency. No compute-saving claim is allowed
  because the adapter and selector run for every input.

### 12.2 Confidence intervals

Use 20,000 percentile-bootstrap replicates with seed 20260922. The resampling unit
is the original official-val source pair; all of its clean tiles, conditions, and
actions receive the same replicate multiplicity. Report intervals for each primary
  mF1 difference, clean F1/recall difference, label-informed diagnostic margin,
  and selector regret.
Tiles or perturbation instances are never treated as independent replicates.

## 13. Predeclared Phase 7 decisions

Let `B*` be the better of A0 and G0 on official-val corruption mF1. For a learned
policy `P`, let `delta_P = mF1(P) - mF1(B*)`. The loss-selected reference O and
`mF1(O)-mF1(B*)` must be reported, but O is not an F1 oracle or upper bound and is
not used as the denominator of a claimed "captured headroom" ratio.

### 13.1 Selector-feasibility gate

At least one learned policy must satisfy all of the following to justify a separate
Phase 8 review:

1. `delta_P >= 0.0005`;
2. the paired-source 95% bootstrap lower bound for `delta_P` is strictly positive;
3. clean F1 is at least `clean F1(A0) - 0.0005`;
4. clean recall is at least `clean recall(A0) - 0.0010`;
5. corruption recall is at least `corruption recall(B*) - 0.0010`;
6. all four actual error transitions and the utility/regret tails are finite and
   fully reported.

If no learned policy passes, the whole-tile selector route stops. A positive sign
accuracy, lower regression loss, or isolated family gain cannot override failure.

### 13.2 Simpler-control rule

Paired auxiliary supervision is supported only if the best passing G2/G3 policy
beats the best passing C1/G1 policy by at least `0.0005` corruption mF1 and its
paired-source 95% interval is strictly above zero under the same clean/recall
limits. Otherwise the conclusion is that the ordinary selector is sufficient;
the simpler passing control is retained and no paired-target mechanism claim is
made.

### 13.3 G2/G3 equivalence rule

A G3 parameterization advantage is supported only if matched G3 beats matched G2
by at least `0.0005` corruption mF1 with a strictly positive paired-source 95%
interval, while satisfying the same harm limits. Otherwise G2 and G3 are treated
as empirically equivalent target parameterizations. G3 is not credited with new
target information in either case.

If an apparent gain requires relaxing the harm limits for only one method, it is a
failed matched comparison, not evidence for that method.

## 14. Scope exclusions and stop conditions

Phase 7 v1 does not authorize:

- local/quadrant/pixel actions or spatial masks;
- multiple adapters, experts, insertion stages, or learned perturbation signatures;
- actor/adapter fine-tuning, joint training, or updated utility labels;
- full-445 training, cross-fitting for a final method, independent actor seeds, or
  public-method reproduction;
- unseen/compound/geometric perturbations;
- held-out test access;
- Phase 8-10 execution after a pass without another reviewed protocol.

Stop and report before interpretation if any frozen hash changes, a source crosses
a split, val is touched before the pre-val seal, test is enumerated, a cache/action
record is missing or non-finite, deterministic replay differs, the temperature
guard fails, or the fixed actions do not reproduce their bound checkpoints.

Because action-benefit prediction now becomes the central mechanism question, a
targeted primary-literature update on learning-to-defer, selective prediction,
cost-sensitive selection, and conditional action value is required before making
any novelty claim. It is not required to report this bounded diagnostic honestly.

## 15. Planned artifacts after approval

No files below are created by this draft. An approved implementation should use:

```text
configs/phase7_v1_seed42.yaml
runs/phase7_protocol_v1_seed42/
runs/levir_phase7_c1_seed42/
runs/levir_phase7_g1_seed42/
runs/levir_phase7_g2_seed42/
runs/levir_phase7_g3_seed42/
runs/levir_phase7_val_seed42/
docs/PHASE7_RESULTS_<date>.md
```

Every run must save resolved configuration, source hashes, environment, manifests,
normalization statistics, initialized tensor hash, descriptor/cache hashes,
checkpoint and resume state, successful/attempted updates, guard completion,
threshold calibration trace, per-instance decisions, actual confusion counts,
  bootstrap parameters, `test_evaluated=false`, and whether the artifact is research
or engineering-only.

## 16. Decisions requested from the reviewer

Approval of v1 would freeze all of the following together:

1. the deterministic 71/18 source-level gate-fit/calibration split;
2. the 1,024-dimensional symmetric stage-2 feature/residual descriptor and the
   one-decoder action timing;
3. the exact 139,970-parameter two-output MLP and seed-42 initialization;
4. A0/G0/C1/G1/G2/G3/O controls, the primary-score unconstrained policy, and the
   information-equivalent G2/G3 clean-risk constrained score;
5. equal clean/perturbed weighting, standardized Smooth L1 targets, C1 weighted
   binary loss, and the matched 100-epoch optimization budget;
6. the common zero-threshold and harm-calibrated policies, including the clean F1
   and recall budgets;
7. the pre-val seal, one-shot official-val evaluation, source-pair bootstrap, and
   selector/paired-target/G2-G3 decision gates;
8. the explicit stop after the Phase 7 report, with Phase 8-10 and held-out test
   remaining closed.

These decisions are now frozen for Phase 7 v1. Any substantive change requires a
versioned protocol and a new review before execution.
