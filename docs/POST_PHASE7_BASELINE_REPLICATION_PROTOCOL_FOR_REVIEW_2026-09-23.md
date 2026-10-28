# Post-Phase 7 baseline replication and route-closure protocol — draft for review

Date: 2026-09-23. Batch name: **R1**. Status: **approved in principle and
frozen for exact configuration binding before execution**.

This document defines the scientifically justified next batch after the frozen
Phase 7 v1 selector gate failed. It does not reopen the rejected whole-tile
selector route, authorize local actions, or authorize held-out test access. Its
purpose is to determine whether the strongest simple robust baselines are
reproducible across training seeds and to select one terminal development model
before any separately reviewed final-test protocol.

## 1. Why the next step is replication rather than Phase 8 local actions

Phase 7 evaluated matched `C1/G1/G2/G3` whole-tile selectors. The better
constant was `G0` with corruption mF1 `0.8959143825`; the best learned policy,
`G1-U`, reached `0.8957707550`, a delta of `-0.0001436274` with paired-source
95% interval `[-0.0008129869, 0.0004037445]`. No learned policy passed the
predeclared selector-feasibility gate, so the frozen decision was
`stop_whole_tile_selector_route`.

The failure does not specifically implicate whole-tile action size. The
label-informed diagnostic retains headroom in both spatial strata:

| Official-val stratum | A0 F1 | G0 F1 | Label-informed O F1 | O margin over better constant | Best learned F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global corruptions | 0.8907420842 | 0.8910962831 | 0.8933396679 | 0.0022433848 | 0.8910257497 |
| Local corruptions | 0.9001348674 | 0.9004637530 | 0.9021728110 | 0.0017090580 | 0.9002509301 |

The diagnostic margin is also positive in all nine corruption families, while
the learned selector fails overall and has corruption AUROC only
`0.4729986–0.5665956`. This points first to an action-utility predictability
failure, not evidence that quadrant or pixel actions will solve the problem.
Starting local actions now would therefore bypass the failed gate and add a new
action space without an evidence-backed selector.

The practical simple baseline is stronger. On the same accepted official-val
corruption benchmark, the full-train seed-42 B1 model achieved clean F1
`0.9060728686` and corruption mF1 `0.9023764361`. Relative to the Phase 7
split-trained `G0`, these are higher by `0.0060164027` and `0.0064620536`,
respectively. This is not a matched mechanism comparison because B1 uses all
445 train sources, but it is the relevant operational comparison for choosing
what to reproduce and eventually freeze.

Seed 42 alone does not settle B1 versus B2. B1 exceeded B2 by
`0.0016343587` corruption mF1 and won 43/45 conditions, but the frozen
source-paired interval for B1 minus B2 was
`[-0.0008166181, 0.0042655612]`. The next unresolved question is therefore the
seed stability of the two already-defined simple baselines.

## 2. Research question and hypotheses

Primary question:

> Across matched seeds 42, 43, and 44, does ordinary supervised corruption
> augmentation (B1) remain the appropriate terminal development baseline, or
> does probability-consistency training (B2) produce a reproducible improvement
> large enough to justify its additional objective?

The experiment is a replication and model-selection batch, not a new mechanism
claim. B2 bears the burden of showing an improvement over the simpler B1. A
reduced clean-to-corruption gap without better absolute corruption mF1 is not by
itself a win.

## 3. Immutable starting evidence

- Phase 7 result report SHA256:
  `f6666b26b1b7a5c4bafba2fbea3bde50c4a51deaa89e46c15059c72adffdda64`
- Phase 7 decision artifact SHA256:
  `a9ec3674475f30a20020eb57c0adfb3e0adb0531069e6840bbae2ce023cb432c`
- Stage 4 gate analysis SHA256:
  `79605cdf46e41f2bff5cd6577184e5cb4689aee07beceb87f62a0a060105d986`
- Seed-42 B1 source config SHA256:
  `b2ee6636080f171b3d67cfa5867237a7c79d82b37f4a4a80c1021879ffe17c71`
- Seed-42 B2 source config SHA256:
  `7a1fedb28026e2fdc085e222364e78323bf2ff051851bbc36cedaecf12ace363`
- Seed-42 B1 best checkpoint SHA256:
  `c4c0b849081bc5ec5de501b5e5e6c36b07c1c6b7b8ffb403bbeec49d1b1d746d`
- Seed-42 B2 best checkpoint SHA256:
  `fab488db39030f61d21fb2743ff8ae13b5da3a5d489cc5b625a67dcd64fe2173`
- Seed-42 B1 official-val summary SHA256:
  `9c1b97ba4bd57d374fa9138a87e7ed9ff6c96a76d5cae1a18c9264462a03da5c`
- Seed-42 B2 official-val summary SHA256:
  `531a212217968f18a16ccb7e02ed5d8f2196f1e579c5e14e9779d2d0e4cf4c81`

The seed-42 artifacts are reused as completed evidence and are never retrained
or overwritten.

## 4. Experimental matrix

Four new research runs are proposed:

| Run | Training seed | Secondary view | Consistency objective |
| --- | ---: | --- | --- |
| R1-B1-43 | 43 | frozen v1 perturbed view | none |
| R1-B2-43 | 43 | exactly matched to R1-B1-43 | symmetric probability MSE, weight 1.0 |
| R1-B1-44 | 44 | frozen v1 perturbed view | none |
| R1-B2-44 | 44 | exactly matched to R1-B1-44 | symmetric probability MSE, weight 1.0 |

The completed seed-42 B1/B2 pair supplies the third seed. B0, B0-E, actor,
adapter, selector, local-action, and public-method runs are out of scope.

## 5. Frozen factors

Except for the declared training seed and deterministic train-stream seed, every
factor remains identical to the accepted Stage 4 protocol:

- original LEVIR-CD train/val split; label decoding
  `author_uint8_floor255`; non-overlapping 256×256 tiles;
- shared randomly initialized ResNet-18 encoder and the existing FPN-style
  absolute-difference decoder, width 64;
- no PSE, PCDC, CPC, expert bank, adapter, gate, local action, foundation model,
  geometric perturbation, or extra prediction head;
- 100 epochs, batch size 4, AdamW, learning rate `0.001`, weight decay
  `0.0001`, cosine schedule to `1e-6`, AMP, and gradient clip norm 1.0;
- one clean and one secondary view per source batch, concatenated in the frozen
  order and processed in a single forward/BatchNorm update;
- BCE plus per-sample soft Dice task loss on both views;
- B2 alone adds symmetric probability-space MSE with weight 1.0;
- checkpoint selection by clean validation F1 only at threshold 0.5;
- the accepted nine-family, five-level, non-geometric corruption operators and
  severity values;
- official-val evaluation by the existing fixed 64-source, 1,024-tile,
  45-condition manifest; no validation-driven severity changes.

Within each seed, B1 and B2 must have identical initial model tensors, source
order, corruption instances, attempted optimizer steps, view order, and
BatchNorm behavior. The deterministic corruption train-stream seed is paired
within method pair and differs across replication seeds. The frozen mapping is
`310945` for training seed 43 and `310946` for training seed 44; these remain
distinct from the reserved validation/test seeds `310943/310944`. Validation uses
the already frozen manifest rather than newly generated per-seed instances.

## 6. Configuration and approval preflight

No training may start directly from this prose draft. After approval, a freeze
step must create exact per-seed base, corruption-training, B1, and B2 YAML files,
then record their SHA256 values in one approval artifact. That preparation must:

1. change only the experiment name, output directory, training seed, and declared
   train-stream seed relative to the seed-42 templates;
2. derive new severity-review bindings without changing any operator or severity;
3. prove that the official-val manifest, instances, evaluation configuration,
   threshold, and aggregation rule are byte-identical to Stage 4;
4. snapshot and hash all training/evaluation implementation sources;
5. reject existing output directories, missing CUDA/data/checkpoints, hash drift,
   test-enabled configuration, or an unapproved field difference;
6. run unit tests and a fresh train-only real-CUDA smoke for both B1 and B2;
7. seal the four run configurations before loading research data.

If the current implementation cannot express these seed changes without a source
change, the source change and its tests must be reviewed and included in the
freeze. It cannot be introduced after the run starts.

## 7. Training execution and recovery

- Interpreter: `E:/Download/anaconda3/envs/geosa-basa/python.exe`.
- Device: real CUDA only; no CPU, synthetic-data, or random-checkpoint fallback.
- Runs execute sequentially on the single GPU under the existing frozen
  temperature guard.
- Expected budget per run: 100 complete epochs, 178,000 attempted updates, and
  1,424,000 pair views. Actual successful steps and AMP skips are reported.
- A temperature or quota interruption occurs only at a complete-epoch boundary
  when possible. Resume must use the matching run's `last.pt`, restore optimizer,
  scheduler, scaler, RNG, epoch, best metric, and exposure-stream state, and prove
  deterministic continuation.
- An active matching process is monitored rather than started again.
- No run may reuse or overwrite a seed-42 directory.

## 8. Pre-validation seal and one-shot evaluation

After all four trainings complete, but before corruption validation is opened, a
pre-validation seal binds:

- protocol and approval hashes;
- all four configs, status files, guards, histories, exposure summaries,
  `best.pt`, and `last.pt` artifacts;
- initial tensor and within-seed exposure/order equality checks;
- evaluation code, fixed manifest, instances, acceptance, and environment;
- `test_evaluated=false` for every input.

Each new best checkpoint is then evaluated once on the frozen official-val
benchmark. Evaluation must produce pooled clean metrics, equal-weight
45-condition corruption mF1, pooled corruption recall and background FPR,
per-family/per-level tables, net FP/FN versus repeated clean, per-source records,
compute accounting, and guarded completion. A numerical, hash, isolation, or
temperature-guard failure stops the batch without substituting a resource.

## 9. Statistical report

The final report combines the completed and immutable seed-42 results with seeds
43 and 44. It reports, for B1 and B2:

- each seed separately;
- three-seed mean and sample standard deviation for clean F1, corruption mF1,
  clean-to-corruption gap, pooled recall, background FPR, net FP, and net FN;
- B2-minus-B1 differences per seed and their mean/standard deviation;
- condition win counts per seed;
- source-paired percentile bootstrap intervals with 20,000 replicates and frozen
  seed `20260923`. Each replicate resamples the same original validation source
  pairs for both methods and all three seeds, computes each seed's 45-condition
  mF1, and then averages the three paired deltas;
- full training cost, successful/attempted updates, AMP skips, wall time, and
  peak GPU memory.

The three seeds are not treated as enough to estimate a reliable population
confidence interval over all possible training randomness. Source-pair bootstrap
and between-seed variation are reported separately.

## 10. Predeclared selection rule

B2 replaces the simpler B1 as the terminal development baseline only if all five
conditions hold across the frozen three-seed aggregate:

1. mean `mF1(B2)-mF1(B1) >= 0.0005`;
2. the source-paired 95% bootstrap lower bound for the three-seed mean delta is
   strictly positive;
3. mean clean F1(B2) is at least mean clean F1(B1) minus `0.0005`;
4. mean corruption recall(B2) is at least mean corruption recall(B1) minus
   `0.0010`;
5. B2's corruption mF1 exceeds B1's in at least two of the three seeds.

If any condition fails, retain B1. A smaller clean-to-corruption gap, lower FPR,
or isolated condition win cannot override the rule. If either method has a
cross-seed corruption-mF1 range greater than `0.005`, stop for instability review
instead of freezing a final model.

Passing this selection rule freezes a **simple development baseline**, not a new
mechanism or novelty claim. It authorizes preparation of a separate final-test
protocol for review; it does not itself authorize test access.

## 11. Explicit exclusions and stop conditions

This batch does not authorize:

- Phase 7 selector retraining or alternate selector architectures;
- quadrant, region, pixel, mask, or other local correction actions;
- actor/adapter reuse, joint training, utility-label refresh, or full-train
  action-gate training;
- hyperparameter, consistency-weight, severity, threshold, or architecture search;
- public-method reproduction, unseen/compound perturbations, or geometric shifts;
- held-out test enumeration, hashing, loading, inference, or model selection.

Stop and report if any frozen input changes, within-seed B1/B2 initial tensors or
corruption streams differ, validation is accessed before the seal, a run cannot
resume exactly, the temperature guard fails, or any artifact reports
`test_evaluated=true`.

## 12. Approval boundary

The user's 2026-09-23 instruction delegates reasonable downstream research
judgment and asks execution to continue until explicitly interrupted. That
standing instruction authorizes R1 because this batch follows the failed Phase 7
gate, narrows rather than expands the mechanism scope, and keeps held-out test
closed. A bound approval record must still capture the reviewed-draft hash, this
frozen document hash, and every generated YAML hash before long training starts.

After R1 completes, the required terminal action is its report and selection
decision. No final test or new mechanism starts automatically; either requires a
new evidence-backed protocol and freeze under the same delegated-judgment rule.
