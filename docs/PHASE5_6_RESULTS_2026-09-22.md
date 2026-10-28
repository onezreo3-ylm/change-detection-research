# Phase 5-6 action-feasibility results — frozen v1

Date: 2026-09-22. Status: **completed**.

This report records the completed, user-approved Phase 5-6 diagnostic defined
in `PHASE5_6_PROTOCOL_FOR_REVIEW_2026-09-15.md`. It follows the accepted Phase
4 comparison in `STAGE4_FROZEN_BASELINE_RESULTS_2026-09-15.md`. All research
runs used real CUDA with the frozen temperature guard. The held-out test split
was not enumerated, loaded, evaluated, or used for selection.

## Frozen actor detector

- Output: `runs/levir_phase5_actor_b1_seed42`
- Status: completed, temperature guard completed
- Training: 100 epochs, 1,139,200 pair views, 142,400 attempted updates
- Successful optimizer steps: 142,349; AMP overflow skips: 51
- Selection: clean validation F1 only
- Best checkpoint: epoch 45, clean-val F1 `0.8997169052660828`
- `test_evaluated=false`

Artifact SHA256:

- `best.pt`: `20e53ddc95a6e510c8ed988621112965f51669abdcb517bba858b5b8bb095de1`
- `status.json`: `470bfd2d4661632a75adbee6db2ce6263b64bc546391a41d9e73a3fbe3778023`
- `guard_completion.json`: `0caf4fee738e104eab9173ff3a49f06c31ec6f5c96c35173af6f35eaeeee7c4a`

## Frozen stage-2 adapter

- Output: `runs/levir_phase5_stage2_adapter_seed42`
- Status: completed, temperature guard completed
- Detector and BatchNorm state remained frozen; only the shared adapter was
  trainable
- Training: 100 epochs, 1,139,200 pair views, 142,400 attempted updates
- Successful optimizer steps: 142,350; AMP overflow skips: 50
- Selection: clean corrected validation F1 only
- Best checkpoint: epoch 11, clean corrected-val F1
  `0.9000564658819357`
- `test_evaluated=false`

Artifact SHA256:

- `best.pt`: `7447184bb1852cb03dd1bebe5e681ea2165d2fae2425348457ad820fe5e103dd`
- `status.json`: `040bad7b42a53fe5df6195f9973ec1d9d92b8b8c68a5d64466ae4e6d1078ada8`
- `guard_completion.json`: `71cfc27e38d6f4768fce41eae73d93f7ba130a74c64d9906e103269b3ec5242b`

## Phase 6 fixed action-utility diagnostic

- Output: `runs/levir_phase6_action_utility_seed42`
- Status: completed, temperature guard completed
- Scope: 89 train-only gate sources, 1,424 tiles, all 45 frozen conditions,
  64,080 perturbed instances
- Actor checkpoint: epoch 45; adapter checkpoint: epoch 11
- Raw action utility SHA256:
  `0d6753a8e0fb7a85642f5614f642841a6d797e8fa6382856605e81c7a97fa3ce`
- `test_evaluated=false`

### Utility sign distribution

| Utility sign | Count | Fraction |
| --- | ---: | ---: |
| `uT > 0` | 43,949 | 0.6858458177 |
| `uT < 0` | 19,848 | 0.3097378277 |
| `uT = 0` | 283 | 0.0044163546 |

Both nonzero signs occur in all nine corruption families. The frozen minimum
was 5% for each sign in at least three families, so the sign-distribution gate
passes. Mean perturbed utility is `0.002265845392209937`.

### Constant actions and diagnostic label-informed choice

| Action | Corruption mF1 |
| --- | ---: |
| `a0` no correction | 0.8930599229 |
| `a1` shared stage-2 adapter | 0.8939128296 |
| Label-informed actual-action choice | 0.8960960698 |

The better constant action is `a1`. The label-informed choice improves mF1 by
`0.0021832401970008908`. Its paired-source percentile-bootstrap interval
(20,000 replicates, seed 20260915) is
`[0.001498786540712313, 0.0030834678876288926]`, with bootstrap median
`0.002184636133696183`. This exceeds the frozen `0.002` margin and has a
strictly positive lower bound, so the actionable-margin gate passes.

The label-informed choice is diagnostic only and is not an inference-time
selector or a theoretical upper bound.

### Error transitions from `a0` to `a1`

| Transition | Count | Denominator | Rate |
| --- | ---: | ---: | ---: |
| `FP -> TN` | 2,460,732 | 19,360,133 | 0.1271030524 |
| `FN -> TP` | 591,950 | 21,481,728 | 0.0275559769 |
| `TP -> FN` | 1,783,847 | 170,544,477 | 0.0104597172 |
| `TN -> FP` | 619,172 | 3,988,160,542 | 0.0001552525 |

Both removed-error types and both introduced-error types are present. The
adapter's benefit is therefore not reported as a one-sided foreground-bias
effect.

## Frozen gate decision

Both predeclared conditions pass:

1. utility sign prevalence/family coverage: pass;
2. actionable mF1 margin and paired-source confidence interval: pass.

The frozen decision is
`pass_to_separately_reviewed_phase7_protocol`. This result supports preparing a
separate Phase 7 protocol; it does **not** authorize a learned gate, local
actions, Phase 7-10 execution, or held-out test access by itself.

Gate artifact SHA256:

- `gate_decision.json`: `dc4307d215e5b7d513d98a922eee5cbaa2219f1715227fb76cecb800cee573fe`
- `status.json`: `1d708d03fd0432397add9274692a2a4fbfa5084482235b4da13bf120acab890e`
- `guard_completion.json`: `a5b11556d818a353bfc70b4e0cf694d50021922b9af0cd670a192a98c5cdcf77`

## Final verification

- `python -m unittest discover -s tests -v`: 168 tests, all passed
  (`62.344s`)
- Fresh real-CUDA, train-only engineering smoke:
  `runs/smoke_phase5_6_cuda_final_20260922`
- Smoke device: NVIDIA GeForce RTX 3060 Laptop GPU
- Actor, adapter, and Phase 6 smoke paths completed with finite utilities;
  detector gradients after adapter step were zero
- Smoke temperature guard completed and `test_evaluated=false`
- Smoke `status.json` SHA256:
  `fe24d4673d5da9611b04020f8244b12750f67925034c45bcec6b8521c8850612`

## Scope conclusion

The entire approved Phase 5-6 v1 batch is complete and reproducibly bound to
its frozen protocol, checkpoints, manifests, raw utilities, guard records, and
train-only split. The mechanism route passes its diagnostic gate, but the next
executable research batch remains a separately frozen Phase 7 protocol. The
held-out test firewall remains closed.
