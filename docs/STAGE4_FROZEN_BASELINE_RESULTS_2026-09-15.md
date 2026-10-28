# Phase 4 frozen baseline results and gate decision

Date: 2026-09-15. Scope: seed 42, original LEVIR-CD validation split only.
The held-out test split was not read. All four scored runs report
`test_evaluated=false`, `training_started=false`, and completed temperature guards.

## Outcome

The ordinary augmentation baseline B1 is the strongest absolute model in this
run. B2's output-consistency term slightly reduces the clean-to-corruption gap,
but it lowers both clean F1 and corruption mF1 relative to B1. B0-E performs no
better than B0 under corruption, so B1/B2's robust gain is not an artifact of a
second clean view, doubled exposure, or the shared two-view batch shape.

The Phase 4 continuation gate is **passed only for a separately frozen and
reviewed Phase 5-6 mechanism diagnostic**. B2 still has statistically stable net
FP and FN degradation, especially at accepted levels 4-5. This result does not
authorize a gate model, local actions, Phase 7-10, or held-out test evaluation.

## Training comparability

| Run | Best epoch | Clean-val F1 used for selection | Successful steps | AMP skips | Attempted steps | Pair views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 68 | 0.9060729 | 177,936 | 64 | 178,000 | 1,424,000 |
| B2 | 63 | 0.9038268 | 177,933 | 67 | 178,000 | 1,424,000 |
| B0-E | 66 | 0.9039767 | 177,938 | 62 | 178,000 | 1,424,000 |

All three runs have the same initial model tensor SHA256
`eb4fa2550e47d732c5b21acb6899ba5539856bee35b532da58006f56108001a8`.
B1 and B2 also have the identical sampled corruption stream SHA256
`51f98534b6e01a4ff7a7ee0b6b308b70014739ff3edc00ff76348360eda91187`.

## Fixed validation comparison

`corruption mF1` is the equal mean of 45 condition F1 values after pooling all
pixels within each condition. Recall and background FPR below pool all 45
condition confusion counts; they are not tile averages.

| Run | Clean F1 | Corruption mF1 | Clean−corruption | Pooled recall | Background FPR | Net FP vs clean×45 | Net FN vs clean×45 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.9059755 | 0.8794148 | 0.0265607 | 0.8630714 | 0.0042898 | +1,638,981 | +4,490,151 |
| B1 | 0.9060729 | **0.9023764** | 0.0036964 | **0.8963591** | 0.0039554 | +413,962 | +507,167 |
| B2 | 0.9038268 | 0.9007421 | **0.0030847** | 0.8923001 | **0.0038961** | +252,889 | +497,009 |
| B0-E | 0.9039767 | 0.8767160 | 0.0272607 | 0.8524891 | 0.0039452 | +1,384,490 | +4,760,494 |

B1 beats B2 in 43 of 45 condition F1 values and in every family mean. The
source-paired 95% interval for B1−B2 corruption mF1 is
`[-0.0008166, 0.0042656]`, so the seed-42 difference is not treated as a stable
superiority claim. B2 exceeds B0 by 0.0213273 mF1; its source-paired interval is
`[0.0162710, 0.0272039]`.

## B2 residual

| Level | Mean F1 | Clean−mean F1 | Net FP | Net FN |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9036535 | 0.0001733 | +5,961 | +3,033 |
| 2 | 0.9029181 | 0.0009087 | +22,889 | +22,793 |
| 3 | 0.9015167 | 0.0023101 | +44,328 | +69,315 |
| 4 | 0.8994561 | 0.0043707 | +75,504 | +137,840 |
| 5 | 0.8961660 | 0.0076608 | +104,207 | +264,028 |

The five largest condition gaps are color-bias L5 (0.0137857), Gamma L5
(0.0120058), color-bias L4 (0.0095125), blur L5 (0.0088522), and shadow L5
(0.0083716). Across the 64 original validation pairs, 42 have a positive net FP
increase, 49 a positive net FN increase, and 38 both when the 45 corruptions are
compared with the same model's clean counts. The exhaustive per-source records
in each run's `paired_by_source.json` retain the original FP/FN counts, avoiding
a cherry-picked tile-only conclusion.

The deterministic 20,000-replicate paired bootstrap resampled original image
pairs, pooled pixels inside each resampled condition, and then averaged the 45
condition F1 values. Its B2 95% intervals are:

- clean−corruption mF1: `[0.0024115, 0.0038991]`;
- net FP versus repeated clean: `[149,902.5, 363,414.7]`;
- net FN versus repeated clean: `[356,250.5, 652,981.8]`;
- levels 1-3 clean gap: `[0.0007772, 0.0015399]`;
- levels 4-5 clean gap: `[0.0047183, 0.0075983]`.

All lower bounds remain positive. Since the visual review accepted all five
levels for this validation-only benchmark, this is clear residual degradation
under the Phase 4 wording, while the effect size is explicitly modest after B1/B2.

## Gate and boundary

Phase 4 therefore supports moving to a **small Phase 5-6 diagnostic**: freeze the
356/89 actor/gate source-pair split, train detector and one stage-2 zero-init
residual adapter only on actor sources, freeze them, and measure whether the real
no-correction/correction actions contain both beneficial and harmful gate-source
examples. The exact split, training budget, action/loss definition, and acceptance
thresholds must be documented and reviewed before that new long training starts.

If useful and harmful actions do not coexist, or the actionable margin is too
small, the route stops there. Phase 7 comparisons, local actions, independent
seeds, public-method reproductions, and final test remain closed until their own
conditions and freeze points are met.

Machine-readable evidence, bootstrap parameters, exact input hashes, source-pair
spread, and the bounded gate decision are in
[`gate_analysis.json`](<E:/new research/detection/runs/stage4_frozen_comparison_seed42/gate_analysis.json>).
