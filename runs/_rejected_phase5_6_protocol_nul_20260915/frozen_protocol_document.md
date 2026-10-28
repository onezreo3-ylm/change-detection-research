# Phase 5-6 mechanism diagnostic protocol v1 — approved and frozen

Date: 2026-09-15. Status: **approved and frozen by the user; implementation not yet started**.

Approval record: in the current Codex task, the user replied
`批准按草案执行` after the six decisions at the end of this document were
presented for review. The approval authorizes only the bounded Phase 5-6
diagnostic described here. It does not authorize Phase 7-10 or held-out test
access.

This draft follows the Phase 4 gate in
[`STAGE4_FROZEN_BASELINE_RESULTS_2026-09-15.md`](<E:/new research/detection/docs/STAGE4_FROZEN_BASELINE_RESULTS_2026-09-15.md>).
It was deliberately reviewed before implementation or long training. This
approval authorizes only the actor detector, one shared stage-2 residual
adapter, and the Phase 6 action-utility diagnostic. It does not authorize a
learned gate, local actions, Phase 7-10, or held-out test access.

## Why this diagnostic is still warranted

B1 is the strongest absolute seed-42 baseline (`0.9023764` corruption mF1).
B2 reaches `0.9007421`; its output consistency does not improve over B1, but its
paired-source clean-to-corruption gap remains positive with 95% CI
`[0.0024115, 0.0038991]`. Its net FP and FN residual intervals are also strictly
positive. B0-E is worse than B0, excluding clean double-view exposure as the
source of B1/B2's improvement.

The residual is modest. The purpose of Phase 5-6 is therefore a bounded
feasibility test: determine whether one real correction action produces enough
beneficial and harmful cases to justify learning a selector. It is not a claim
that another model component must be added.

## Proposed source-pair split

Only the 445 original official training pairs participate. The original val is
used only for clean-val checkpoint selection and later frozen comparisons; test
is never enumerated or opened.

Recommended deterministic split algorithm:

1. Decode each training mask with the already adopted
   `author_uint8_floor255` rule and count foreground pixels over the original
   1024×1024 source image.
2. Sort source pairs by `(foreground_pixel_count, canonical_filename)`.
3. Partition that list into 89 adjacent groups of five.
4. Within each group, select the smallest
   `SHA256("phase5_actor_gate_v1_seed42\0" + canonical_filename)` as the gate
   source; the other four are actor sources.
5. Freeze filenames plus A/B/label SHA256 values in a manifest. Tiles inherit
   their source membership; no tile, transformed copy, or cached feature may
   cross the boundary.

This yields exactly 356 actor and 89 gate pairs while deterministically
stratifying by change prevalence. A read-only train-label audit gave:

| Partition | Sources | Foreground pixels | Foreground fraction | Median source fraction | Zero-FG sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor | 356 | 17,145,092 | 0.0459293 | 0.0258546 | 38 |
| Gate | 89 | 4,267,249 | 0.0457255 | 0.0259533 | 9 |

The actor/gate empirical CDF distance for foreground fraction is `0.011236`.
No val/test label was used to construct or rebalance this proposal.

## Proposed actor detector

Use the simpler B1 training rule because B1 was stronger than B2 in 43/45
conditions and B2 did not establish an absolute gain.

- Fresh actor-only detector; do not load the full-445 B1/B2 checkpoints.
- Same shared ResNet-18/FPN architecture and random initialization seed 42.
- Same accepted corruption protocol, clean+perturbed two-view batching, BN
  ordering, loss, optimizer, AMP, clipping, cosine schedule, and 100 epochs.
- Expected actor tiles: 5,696; expected batches/epoch: 1,424; attempted updates:
  142,400; pair views: 1,139,200.
- Checkpoint selection: clean validation F1 only. No corruption score or test
  score selects a checkpoint.
- Exact epoch-boundary resume and the existing temperature guard are mandatory.

## Proposed correction action

The detector is frozen in `eval()` before adapter training. Its parameters and
BatchNorm state never update during adapter optimization.

One shared stage-2 residual adapter is applied independently to both temporal
feature tensors, so the action requires no oracle knowledge of which date was
perturbed:

```text
R(F) = Conv1x1(128→32) → GroupNorm(8) → ReLU
       → Conv3x3(32→32) → GroupNorm(8) → ReLU
       → Conv1x1(32→128, zero-initialized)
correct(F) = F + R(F)
```

Frozen implementation details: the first two convolutions use no bias because
they are immediately followed by GroupNorm; the final convolution uses a bias,
and both its weights and bias are initialized to exact zero. The 3x3 convolution
uses stride 1 and padding 1. No dropout, activation after the last convolution,
or additional normalization is present.

The two executable actions are exactly:

- `a0`: bypass the adapter for both dates;
- `a1`: apply the same adapter to both dates at stage 2, then run the unchanged
  absolute-difference decoder.

There is no type label, perturbation metadata, PSE, expert bank, spatial mask,
or target-date oracle at inference.

Adapter training proposal:

- Actor sources only; detector frozen; adapter is the only trainable module.
- One deterministic accepted perturbation per actor tile/epoch, with the same
  clean+perturbed source exposure and one optimizer step per source batch.
- Objective is the equal mean of the existing supervised change-detection loss
  on corrected clean and corrected perturbed views. No consistency term and no
  post-hoc reward objective are added at this stage.
- AdamW/cosine/AMP/clipping values match B1; 100 epochs; clean corrected-val F1
  selects the adapter checkpoint.
- Save the no-action detector hash, adapter hash, exposure stream, successful
  updates, AMP skips, guard record, and `test_evaluated=false`.

## Phase 6 fixed action-utility diagnostic

After selection, freeze detector, adapter, decoder, and all BN state. Generate a
new train-only gate manifest covering every gate tile under all 45 accepted
family×level cells. Direction, date, and global/local choices are derived from a
versioned stateless hash; labels do not affect perturbation placement.

There are 1,424 gate tiles and 64,080 fixed perturbed instances. Clean actions
may be cached once per tile, but every reported perturbed action must actually
run the decoder. For each tile and perturbation, compute the existing per-sample
BCE+soft-Dice task loss for all four combinations:

```text
uT = L(a0, Tx) - L(a1, Tx)
u0 = L(a0, x)  - L(a1, x)
a  = uT - u0
```

No gradients flow while labels are generated. Save raw losses, utilities,
action predictions/confusion counts, source IDs, perturbation metadata, and
transition counts `FP→TN`, `FN→TP`, `TP→FN`, `TN→FP`. Report overall and by
family, level, source pair, target date, spatial mode, direction, and injection
overlap. Confidence intervals resample original source pairs, never tiles.
The frozen implementation uses 20,000 percentile-bootstrap replicates and RNG
seed 20260915. A source pair is the resampling unit; all of its tiles and 45
conditions receive the same replicate multiplicity.

Diagnostic references:

- always no correction (`a0`);
- always correction (`a1`);
- label-informed per-tile choice between the two **actually executed** actions.

The last item is diagnostic only, is not available at inference, and is not
called a theoretical F1 upper bound.

## Proposed pre-data continuation thresholds

These thresholds are intentionally stated before Phase 6 outputs exist:

1. `uT>0` and `uT<0` cases must each constitute at least 5% of gate instances,
   and each sign must occur in at least three corruption families.
2. The label-informed actual-action choice must improve corruption mF1 over the
   better constant action by at least `0.002` absolute, with the 95% paired-source
   bootstrap lower bound above zero.
3. Raw transition counts must show that the gain is not solely an increase in
   foreground bias: report both removed errors (`FP→TN`, `FN→TP`) and newly
   introduced errors (`TP→FN`, `TN→FP`) with their correct denominators.
4. If either sign-distribution or actionable-margin condition fails, stop the
   mechanism route. Do not train G1/G2/G3 and do not reinterpret a failed action
   as evidence for a selector.

If all conditions pass, Phase 7 still requires a new reviewed protocol freezing
gate inputs, capacity, splits, losses, harm constraints, G0-G3 controls, and
evaluation thresholds.

## Approved decisions

The user approved the following as protocol version 1 before implementation:

1. foreground-stratified 356/89 deterministic source split;
2. actor-only B1 detector rather than B2;
3. the single 128→32→32→128 shared stage-2 adapter applied to both dates;
4. frozen-detector, supervised clean+perturbed adapter training for 100 epochs;
5. exhaustive 45-cell gate manifest and actual four-action execution;
6. the 5% sign prevalence and 0.002 actionable-margin continuation thresholds.

This approval permits implementation and execution of items 1-6 only. A learned
gate, Phase 7-10, and held-out test evaluation remain closed unless a later
protocol is separately reviewed and frozen.
