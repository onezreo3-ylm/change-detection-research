# PCDC research specification and reviewed execution boundaries

## 2026-10-27 method-first preference

The user answered the pending route question with "优先提出并验证新方法".
Method discovery and falsifiable validation take priority over polishing a
negative-results paper. The active B0-E43/44 matched controls continue because
they are needed to quantify any later increment. This preference does not
reopen failed candidates, the consumed test, Martigny, foundation models or
geometric perturbations, and does not extend the original shared48hour budget.
First screen a distinct spatial action/error-transition hypothesis using saved
train-gate evidence and primary literature, under
`NEW_METHOD_FEASIBILITY_SCOPE_2026-10-27.md`. It is a new exploratory feasibility
screen, not a successful method, a reinstatement of the failed whole-tile
selector or permission to run concurrent GPU jobs. New model fitting requires
an explicit versioned protocol, matched controls, source identity, tests,
remaining-budget admission and safe serial execution after the current queue.

## 2026-10-26 publication-evidence continuation

The user explicitly requested continued improvement until the experiments have
a defensible publication evidence chain ("持续改进并完善实验，直到证据链完善可达到能发表的论文").
This authorizes evidence-driven preparation, implementation, bounded matched
controls, analysis and manuscript improvement without repeated approval of
ordinary choices. It does not establish novelty, effectiveness or acceptance.
The completed negative candidates and their stopping rules remain closed.
The next concrete study fills the missing clean double-view B0-E controls at
seeds43/44 against the already completed B1/B2 replications; it does not search
for another correction module. See
`PUBLICATION_EVIDENCE_CONTINUATION_2026-10-26.md` and
`B0E_REPLICATION_PROTOCOL_2026-10-26.md`. The original shared48hour ledger is
retained, with no automatic extension. Test/Martigny, foundation models and
geometric perturbations remain outside this batch.

## 2026-10-25 explicit request to start new training

After the completed native-window probe reported no consistent accuracy gain,
the user explicitly requested new training ("有提升吗，开启新的训练"). The current
batch is now the two serial, ordinary native-window training controls frozen in
`LEVIR_NATIVE_WINDOW_TRAINING_PROTOCOL_2026-10-25.md`: native512 and the same
source windows split into grouped256 inputs, with identical initialization,
group ordering, per256 BCE/Dice loss units, update attempts and pixel exposure.
This explicitly admits the original train and clean development-val used for
checkpoint selection, after engineering, identity, budget and independent
runtime checks. It does not reopen any failed candidate, held-out test or
Martigny, or admit new architecture, normalization or geometric image transforms.
Earlier dated no-training restrictions below describe their completed batches.

## 2026-10-25 current-window continuation and new-question screen

The user requested continuation in the current window and explicitly chose
screening a research question that does not depend on the missing parent-scene
crosswalk. This replaces waiting for that table as the only next action. The
completed tail-risk candidate and source-group admission decision remain closed.
The new screen uses primary literature, static code, and already-saved fixed
LEVIR train16 maps only; no new raw validation, test, or Martigny access and no
new training. See `NEW_QUESTION_SCREEN_2026-10-25.md`.

The selected next diagnostic concerns native-resolution window context and
prediction stability. It admits engineering preparation and, after identity,
isolation, budget, and independent temperature checks, one bounded train-only
inference probe under `LEVIR_TILING_CONTEXT_PROBE_PROTOCOL_2026-10-25.md`.
No new architecture, normalization replacement, geometric image transform, or
long training is admitted by that screen. Window-reading/fusion changes must be
explicitly recorded as a separate diagnostic protocol; historical scores, splits,
thresholds, failed gates, and artifacts must not be overwritten.

## 2026-10-22 reviewed transition: screen a replacement research question

After the completed C5/R5 negative comparison, the user asked for an objective
continue-or-pivot decision and then instructed execution of the recommended next
step. This authorizes a literature/evidence screen of replacement questions,
read-only analysis of saved development counts, and bounded train-only image and
label review. It does not authorize another long training run, a new architecture,
foundation models, geometric perturbations, or reopening either held-out test.
Current raw review is restricted to the original fixed Sion train probe; no new
validation images/labels are read. All original protocols and artifacts remain.
The new screen is exploratory after earlier results were observed, not a new
preregistered effectiveness test or confirmation of novelty. See
`DIRECTION_SCREEN_2026-10-22.md` for the resulting comparison and next checkpoint.

The dated sections below retain the historical scope of their original batches.

Updated: 2026-09-13. The original scope came from the user-supplied
`PCDC_Codex_Handoff_All_In_One.md`; the latest reviewed roadmap is
`EXPERIMENT_PLAN_2026-09-12.md`. Neither a handoff nor a proposed mechanism
establishes novelty or overrides explicit user requests.

## Reviewed transition after the completed baseline

Phase 1 completed 100 epochs and independent validation/test verification on
2026-09-12. The user then reviewed a revised roadmap in the task
“后续实验流程详细一点” and requested its next step in this task.

The revised research candidate is **supervision of the benefits and harms of
actual correction actions**, not three pre-declared PSE/PCDC/CPC innovations.
PSE is a possible diagnostic control; MoE is not being added. G2 `(uT,u0)` and
G3 `(uT,uT-u0)` have equivalent target information and require matched controls.
See `PUBLIC_PRIOR_ART_2026-09-12.md` for the public-paper comparison.

This batch is limited to the public comparison, frozen baseline protocol,
explicit train/val-only development mode, deterministic non-geometric corruption
operators, a **candidate** validation manifest, and train-only severity previews.
It does not start new long training, implement correction/gate models, or read
held-out test image/label files. Candidate severities must be reviewed before
scoring the benchmark. Original baseline data, YAMLs and checkpoints remain intact.

The follow-up request to continue on 2026-09-13 also advances the protected
validation evaluator and clean-val compatibility check. It does not establish
that previously unread severity images passed visual review. Real corruption
scoring requires a separate review record bound to the unchanged protocol,
manifest, instance and checkpoint hashes. Engineering-only authorization is
restricted to explicitly synthetic smoke checkpoints; held-out test stays closed.

## Research question

在非语义成像扰动下，如何校正双时相特征，抑制伪变化而保留真实变化？
目标是 `CD(I1, I2) ≈ CD(I1, T(I2))`，其中 T 保持标签语义且不涉及几何失配。
这只是需要实验验证的目标，不保证严重信息损失下仍可识别变化。

范围：亮度、对比度、Gamma、色偏、雾霾、阴影、噪声、模糊、JPEG。
暂不研究配准、平移、旋转、视角变化、跨区域域泛化、GIS/OSM、DSM、
不确定性、基础模型或以频域方法为主要贡献。

## Frozen model and clean-input baseline protocol

- Dataset: original LEVIR-CD, not LEVIR-CD+. Retain the supplied train/val/test split.
- Backbone: one torchvision ResNet-18; stage channels 64/128/256/512.
- Both dates pass through the same encoder in one concatenated temporal batch.
  This avoids sequential A-then-B BatchNorm running-statistics updates.
- Interaction: absolute feature differences at all four stages.
- Decoder: 1x1 lateral projections, top-down nearest-neighbor fusion, 3x3
  Conv/GroupNorm/ReLU refinement, two learned upsampling/refinement stages, one logit.
- Default width: 64. Encoder BatchNorm remains trainable. No auxiliary heads.
- Default initialization: explicitly random (`encoder_weights: null`). Optional
  ImageNet initialization is a separate, logged protocol setting; no silent fallback.
- Default input protocol: on-the-fly, non-overlapping 256x256 tiles of original
  1024x1024 images. No resize, dropped border, random crop, flip, or photometric augmentation.
- RGB uint8 becomes float32 in [0,1], then uses the YAML mean/std for both dates.
- Masks: explicitly configured background/foreground values, by default 0/255.
  RGB masks are accepted only when all channels agree. No thresholding unknown labels.
- Loss: weighted BCEWithLogits plus per-sample soft Dice, then batch mean.
- Metrics: pooled pixel confusion counts over the whole split, not mean tile F1.
  Report precision/recall/F1/IoU/OA in [0,1], at fixed threshold 0.5.
  A zero denominator returns 0; all-negative perfect predictions have OA=1, F1=0.
- Checkpoint selection uses validation F1 only. Test never selects a checkpoint.
- Default optimizer: AdamW; cosine learning-rate schedule. YAML holds all settings.
- Deterministic algorithms are strict; unsupported operations must raise.

The original image split has 445/64/128 pairs. Non-overlapping 256 tiles yield
7120/1024/2048 tiles. See the [dataset author's project page](https://justchenhao.github.io/LEVIR/)
and [author's published protocol](https://levir.buaa.edu.cn/static/pdfs/2022_hao_chen_adversarial.pdf).
The counts alone cannot prove split identity; inventory and optional content audits
are recorded separately. This loader requires split directories, not a flat
directory plus split text files; do not infer a new split automatically.

## Original handoff components: NOT implemented, no longer the default roadmap

1. PSE: continuous appearance/atmospheric/detail attributes, severity, and spatial
   response maps, supervised by synthetic perturbation metadata during training.
2. PCDC: light residual experts conditioned on predicted perturbation signatures
   and masks. Candidate insertion points are stages 1-3; stage 4 initially unchanged.
3. CPC: preserve clean-vs-perturbed change evidence without forcing the two dates'
   features to be equal. Both clean and perturbed pairs retain change supervision.

These remain historical hypotheses, not three verified innovations. The original
rationale is in `NOVELTY_REVIEW_2026-09-11.md`; the newer action-benefit roadmap
and public comparison above take precedence for subsequent research decisions.

### Issues to resolve before Phase 4-6

- In `F_corrected = F + M * sum(w_i E_i(F))`, a no-op residual expert must satisfy
  `E0(F)=0`. An `nn.Identity` expert inside this sum is not a no-op.
- Three group scores can confuse opposing brightness shifts and heterogeneous
  detail degradations. Signature sufficiency requires a controlled test.
- A generated injection mask is not a ground-truth causal map of feature damage;
  all-global masks also make localization evaluation trivial.
- LOGO does not automatically train a useful expert for a never-seen group.
- Test inference must use predicted signatures, not the generator's true type,
  severity, or mask. Oracle conditioning can only be a separately labeled upper bound.
- Severity ranges must preserve target observability. Label invariance alone
  does not make erased or fully occluded buildings recoverable.

## Staged execution and acceptance

1. Clean baseline, real-data audit, training and independent evaluation: completed.
2. Public-neighbor comparison, explicit development-only mode, deterministic
   candidate perturbations and train-only visibility QA: current batch.
3. After accepting the candidate protocol, measure the existing clean baseline's
   degradation on validation only; freeze severity independently of model scores.
4. B1 augmentation and B2 ordinary consistency with matched initialization,
   exposure, updates and BN behavior; add B0-E to isolate exposure when needed.
5. Only if a meaningful residual problem remains: separately reviewed actor/gate
   source-pair split, shared frozen detector/corrector, and actual action diagnostics.
6. Matched G2/G3 and harm-constraint controls, then actual local actions if justified.
7. Independent seeds, public comparisons, unseen/compound tests and final held-out
   evaluation only after method/hyperparameter/action/protocol freezing.

Phase 1 engineering acceptance includes loader, forward/backward, train/val/test,
all five metrics, best checkpoint, standalone evaluation, tests, and instructions.
Synthetic tests do not close a real-LEVIR research gate. The earlier instruction
to stop before Phase 2 was the pre-review boundary; the dated reviewed transition
above authorizes only the next preparation batch, not all later model experiments.

### Observed data gate (2026-09-11; explicit protocol adopted)

The local candidate at `E:/thesis/datasets/LEVIR-CD` has the expected source
counts, but 32 masks contain 709 pixels with values 156 or 254. Strict 0/255
validation therefore blocks that strict configuration. On 2026-09-11 the user
directed the experiment to continue with the existing LEVIR-CD copy. The real run
uses the separately named `author_uint8_floor255` protocol, matching the dataset
author's public `label // 255` loader: 255 is foreground and lower uint8 values are
background. Raw files and the strict default configuration remain unchanged.
See `EXPERIMENT_STATUS_2026-09-11.md` for the audit and run state.
