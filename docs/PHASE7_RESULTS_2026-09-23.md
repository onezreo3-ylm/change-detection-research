# Phase 7 whole-tile selector results — frozen v1

Date: 2026-09-23. Status: **completed; whole-tile selector route stopped**.

This report records the completed, user-approved Phase 7 v1 experiment defined
in `PHASE7_PROTOCOL_FOR_REVIEW_2026-09-22.md`. The run followed the frozen
Phase 5-6 actions: `A0` uses the actor without correction and `G0` always uses
the shared stage-2 adapter. Four small selectors (`C1`, `G1`, `G2`, and `G3`)
were trained only on the train-gate split, sealed before official validation,
and evaluated once on the accepted development-validation manifest. All
research runs used real CUDA and the frozen temperature guard. The held-out
test split was not enumerated, loaded, evaluated, or used for selection.

## Frozen identity and data firewall

- Frozen protocol SHA256:
  `4a41c265600d69a5d5f785873b7b4941a57947efbb43e6a39af46b18231f2c1f`
- Phase 7 config SHA256:
  `962f9ea966b2ee94d96a3af9741c74e5282642505677ff68457b6af71e036a2b`
- Phase 7 fingerprint:
  `41a152f4dbbf59a152c9bdf2fd93dc9e7543ead351c9e69bf54dfbc50954e0f6`
- Approval SHA256:
  `ccdeae044f5682d049ec74b5dc364b4c6fd7cee10d2db0ab1173206ae830d1e9`
- Train-gate selector split SHA256:
  `35e5a10d273bb8dbb5bb7eb9e3fb1623985ac1e4e8a39d35115888588d924c91`
- Pre-validation seal SHA256:
  `753961326a4b788144fa3d2ecc44ba8e737d6ecbdb9e51e3c22b85c04c047607`
- Official validation contained 64 original source pairs, 1,024 clean tiles,
  45 frozen corruption conditions, and 46,080 perturbed instances.
- `test_evaluated=false` in every cache, training, smoke, seal, and official
  validation completion artifact.

## Train-gate descriptor cache

The guarded cache in `runs/phase7_descriptor_cache_seed42` completed all 45
conditions on 89 train-gate sources: 1,424 clean rows and 64,080 perturbed rows,
each with a 1,024-dimensional descriptor. The source-disjoint fit/calibration
partition contained 71/18 sources, 1,136/288 clean rows, and 51,120/12,960
perturbed rows. No validation or test data were used.

Artifact SHA256:

- `status.json`: `eb9724d8989933195f566dd4df2b89b934bc5e22a9e89f4bf80be1fdf35b0918`
- `guard_completion.json`: `030ab6ee756a1dbf8a7097e05f0584a261fcf64c3ebf04be6ce2b124aa7b73e9`
- `normalization.json`: `bbf2a49fc1f4f160175ced5242fc47c305052080a4a68dfa6beaff34027e1e98`

## Matched selector training

All four selectors used the same initialization tensor hash
`f53f6aa1616cc6349035970554c220376e12aaaff330c19b4523c831ee01b25f`,
the same deterministic training-order hash
`bc923e8d709f11ff5a0b661aa2b921d30315709aafa687e01b1ff91599a5ef8f`,
and the complete matched budget of 100 epochs / 20,000 successful updates.
Checkpoint selection and threshold calibration used only the frozen train-gate
fit/calibration partition.

| Variant | Frozen target | Best epoch | Calibration selection metric | Harm-constrained threshold | Best checkpoint SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| C1 | direct `1[u_obs>0]` classifier | 2 | 0.8153319792 | `+inf` | `41decd08f25e674323ac9ee6c1cf807e80669caa74e40aa884042584bee619ec` |
| G1 | regress `u_obs` | 6 | 0.1127315561 | 0.0159837261 | `5675ae28a741b82e8959b73c5856a94979ddb32e2f90f8a522fd9d5d5b1c4df5` |
| G2 | regress `(u_obs, u_clean)` | 6 | 0.1092939794 | 0.0130912598 | `1e975d24e043ceeeb0fc09482e22cac32ee936ca08f98deb07bff32fb7b78a05` |
| G3 | regress `(u_obs, a_obs)` | 17 | 0.1127993616 | 0.0330653069 | `8e94a3fa0e0d470e1857ef7a1cdc1fedae6d083aa029a577473cc9c4f3ba73b0` |

Every selector run and temperature guard completed, used the NVIDIA GeForce
RTX 3060 Laptop GPU, and recorded `validation_evaluated=false` and
`test_evaluated=false` during training.

## One-shot official-validation result

`G0` is the frozen better constant because its corruption mF1
`0.8959143825` exceeds `A0`'s `0.8955731297`. The table reports each learned
policy's mF1 difference relative to `G0`; intervals are the predeclared
source-pair percentile bootstrap with 20,000 replicates and seed 20260922.
`-U` is the unconstrained calibrated threshold and `-H` is the common
harm-constrained calibration.

| Policy | Corruption mF1 | Delta vs G0 | 95% interval for delta | Clean F1 | Clean recall | Corruption recall | a1 rate | Mean regret | Gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | 0.8955731297 | -0.0003412528 | — | 0.8997169053 | 0.8956342068 | 0.8907151026 | 0.000000 | 0.0081994776 | constant |
| G0 | 0.8959143825 | 0 | — | 0.9000564659 | 0.8903072801 | 0.8849423597 | 1.000000 | 0.0050899043 | better constant |
| C1-U | 0.8956065074 | -0.0003078751 | [-0.0007971431, 0.0001539828] | 0.8998123459 | 0.8920954685 | 0.8872203233 | 0.674609 | 0.0059251695 | fail |
| C1-H | 0.8955731297 | -0.0003412528 | [-0.0012336039, 0.0004512378] | 0.8997169053 | 0.8956342068 | 0.8907151026 | 0.000000 | 0.0081994776 | fail |
| G1-U | 0.8957707550 | -0.0001436274 | [-0.0008129869, 0.0004037445] | 0.8999622893 | 0.8931582263 | 0.8878273858 | 0.529883 | 0.0064430593 | fail |
| G1-H | 0.8955970112 | -0.0003173713 | [-0.0011273289, 0.0004134311] | 0.8997481524 | 0.8954122811 | 0.8903437745 | 0.048698 | 0.0081719385 | fail |
| G2-U | 0.8956900959 | -0.0002242866 | [-0.0009664897, 0.0003613453] | 0.8998526561 | 0.8930034109 | 0.8880016754 | 0.435612 | 0.0056295029 | fail |
| G2-H | 0.8955368033 | -0.0003775792 | [-0.0012424403, 0.0003699726] | 0.8997172323 | 0.8951452601 | 0.8901765471 | 0.036003 | 0.0074730386 | fail |
| G3-U | 0.8953650786 | -0.0005493039 | [-0.0013365173, 0.0000479619] | 0.8994265196 | 0.8929838814 | 0.8877098934 | 0.563867 | 0.0078771714 | fail |
| G3-H | 0.8955269919 | -0.0003873906 | [-0.0012671325, 0.0003975564] | 0.8996698037 | 0.8954438833 | 0.8904696784 | 0.009852 | 0.0082604009 | fail |
| O | 0.8978832000 | +0.0019688175 | [0.0014614855, 0.0026480439] | 0.9018433488 | 0.8949907998 | 0.8899396212 | 0.647548 | 0 | diagnostic only |

`O` is the label-informed actual-action diagnostic. It is not an
inference-time selector, an F1 oracle, or an upper bound.

## Frozen gate decision

The selector-feasibility gate required at least one learned policy to achieve
all of the following: mF1 gain of at least `0.0005`, a strictly positive
paired-source interval lower bound, and the predeclared clean-F1, clean-recall,
corruption-recall, transition, and regret safeguards.

No learned policy passes. Every learned policy has a negative point estimate
against `G0`, and no interval has a positive lower bound. The unconstrained
policies also violate the clean-recall floor; the harm-constrained policies
preserve the clean safeguards but still lose corruption mF1. Diagnostic
corruption AUROC is weak (`C1=0.5666`, `G1=0.4780`, `G2=0.4877`,
`G3=0.4730`) and cannot override the primary gate.

The frozen decision is therefore:

`stop_whole_tile_selector_route`

Consequences under the protocol:

- the paired-auxiliary claim is unsupported because there is no passing policy;
- G3 has no supported advantage over G2. The matched unconstrained difference
  is `-0.0003250173` with interval
  `[-0.0006655692, -0.0000523975]`; the harm-constrained difference is
  `-0.0000098114` with interval `[-0.0002101933, 0.0002000146]`;
- G2 and G3 are treated as empirically equivalent target parameterizations by
  the frozen rule;
- no Phase 8 review or execution is automatically authorized, and no local
  action, learned signature, Phase 8-10, or held-out test experiment was run.

## Compute and interpretation boundary

The official evaluator completed in 1,032.625 wall-clock seconds. Accounted
model compute was 128.482 seconds, peak GPU allocation was 1,091,890,688 bytes,
the descriptor cache occupied 342,688,435 bytes, and amortized accounted model
compute was 2.727628 ms per tile. The matched evaluator computes both frozen
actions once for every tile so all policies and `O` can be compared exactly;
the result makes no compute-saving claim.

The Phase 6 label-informed margin remains real on official validation, but these
particular whole-tile descriptor/selector families do not convert it into an
inference-time gain over the stronger constant action. This is a negative
mechanism result for the frozen route, not evidence that all conditional or
spatial correction mechanisms are impossible.

## Final verification and primary artifacts

- Official status/summary SHA256:
  `c2dccdf169856309df2874fc7af8bca87069c2b146ba9b3f84581bd47da54613`
- Official guard-completion file SHA256:
  `ce1b44baf834fd19faf4745279d47b275e05b87d42b4fef9f2b8a7e1315ae910`
- Raw corruption decisions SHA256:
  `6bce230b84f2d81459c1753cbbc1a5acd9eeec0d85f3287638de17360864b4f9`
- Raw clean decisions SHA256:
  `02f9f4f3222be9309916fe1a50f2ea5e8f4bc5c3f5d335d34753faa15e2a706e`
- `decision.json` SHA256:
  `a9ec3674475f30a20020eb57c0adfb3e0adb0531069e6840bbae2ce023cb432c`
- `bootstrap.json` SHA256:
  `928b04198da3b554bbe5fd2e3fbcb5fff32f959ac18138928b6371b2308ea457`
- `selector_diagnostics.json` SHA256:
  `caf1736bb71d838936ab300618e4b159bdc9d65ae121eb3ac6c62612c1279a1e`
- Full unit suite: 179 tests passed (`60.208s`).
- Fresh final real-CUDA engineering smoke:
  `runs/smoke_phase7_final_cuda_20260923`; all four variants completed finite
  optimization and exact checkpoint-resume checks, both fixed actions executed,
  the temperature guard completed, and validation/test remained closed.
- Final smoke `status.json` SHA256:
  `3cf9063870f047e37cbc43d029c431a2efab6b3198dd6d1235dc73d3bfabb316`
- Final smoke `guard_completion.json` SHA256:
  `e57d3675c7e945643a83c5df42ba6897f9dfb92d829edf29b3397cbb699b5295`

The approved Phase 7 v1 batch is complete. Its stopping rule has fired, so the
whole-tile selector mechanism route ends here and the held-out test firewall
remains closed.
