# Final held-out test: single B1 seed-42 checkpoint, v1

Date: 2026-09-26. This is a separate final-evaluation protocol under the
user's standing instruction to continue with evidence-based judgment. It does
not amend the validation-only corruption acceptance or any R1/R1E artifact.
It authorizes test access **only after** the exact evaluator, YAML, approval,
source hashes, CUDA/temperature preflight, synthetic smoke, full unit tests,
and immutable pretest seal have all been completed. Before that seal, no test
directory enumeration, file hashing, decoding, visualization, or inference is
permitted.

## Decision already made; no test selection

R1/R1E retained ordinary supervised corruption augmentation B1 under its
three-seed validation gate. The operational incumbent is the already existing
seed-42 B1 `runs/levir_b1_augmented_seed42/best.pt`, SHA256
`c4c0b849081bc5ec5de501b5e5e6c36b07c1c6b7b8ffb403bbeec49d1b1d746d`.
It was selected at epoch 68 by clean validation F1 (0.9060728685624768),
before R1 replication. The later corruption scores are not used to choose a
seed. No B2, other seed, ensemble, alternate epoch, or revised threshold may
be examined on test. The source R1 report SHA256 is
`4cc6b8204a1b9a4dae2ea74c89910f04f36277fe54a8582aa1a44786d2136798`;
the completion audit SHA256 is
`a7e20f05b841a67c7103fb571eb31237e24067d1b1c8e821b8606a64890b6dea`.

## Locked model, data, and metric semantics

Use the seed-42 B1 checkpoint exactly, strict state loading and archived
architecture: shared random-initialized ResNet-18 encoder, four absolute
feature-difference stages, width-64 FPN-style decoder. Inference uses the
archived 256×256 non-overlapping grid, RGB uint8/255, the archived RGB
mean/std, `author_uint8_floor255` label decoding, deterministic algorithms,
AMP and channels-last on the bound real CUDA GPU, batch size 32, and sigmoid
threshold **0.5**. No TTA, resizing, overlap, calibration, oracle corruption
metadata, or validation-driven adjustment is permitted. The separate test
loader may change only the development lifecycle flag needed to open test;
its `protocol_signature` must equal the archived checkpoint signature and all
other inference/data fields must match the archived config. The archived
checkpoint intentionally contains train/val audits but no test audit; the
new evaluator creates its own test-only inventory/content audit after sealing
instead of pretending test was present at training time.

The local original LEVIR-CD split is expected to contain 128 test source
pairs, 1024×1024 each, producing 2048 tiles. First access must validate
matched A/B/label names, dimensions, count, tile coverage, readable binary
decoded labels, file SHA256 values, and path/content separation from archived
train/val. Record the test manifest and file digests before model inference.
Counts or filenames alone do not establish geographic independence. If the
audit fails, stop without scoring or substituting data.

## One-shot evaluation suite

The sole model is evaluated once on all 2048 clean test tiles and then on one
fixed set of 45 non-geometric corruption conditions (nine families × five
severity levels), each covering all 2048 tiles: 92,160 perturbed instances.
This is a *new, explicitly test-scoped* acceptance of the previously
train-visually-reviewed operator/level values, not an extension of the
`validation_only` acceptance. Keep placement after tiling and before
normalization; perturb exactly one date; use the frozen global/local ellipse
rules and exact operator parameters in the archived Phase-2 corruption YAML.
No test visual QA, adaptive severity choice, new perturbation, or model
selection is allowed. These are controlled synthetic interventions, not a
claim of physical atmosphere or natural robustness.

Test instances are generated from tile IDs only, never image pixels, labels,
or predictions. For each family, order test IDs by
`(keyed_seed(310944, family, sample_id), sample_id)`; cyclically assign the
Cartesian strata of sorted directions × `(A,B)` × `(global,local)`. Reuse the
same assignment at levels 1–5. Each instance's uint64 seed is
`keyed_seed("levir_tile_corruptions_v1", 310944, "test", sample_id,
family, direction, date, spatial_mode, 0, 0)`; ellipse parameters derive
from `keyed_seed(instance_seed, "mask")` with NumPy PCG64 and the archived
uniform ranges. Thus the test stream is distinct from train 310942 and val
310943. Produce a test manifest and canonical instance JSONL only after the
pretest seal, hash and verify them before scoring. Expected total is 45 ×
2048, balanced strata to within one tile. No generated field enters the
detector input except the transformed RGB pair.

## Fixed outputs and uncertainty

For clean and each condition, compute pooled integer TN/FP/FN/TP over tiles,
then precision, recall, F1, IoU, OA, foreground recall and background FPR at
threshold 0.5. Report per-original-source-pair counts. The primary robustness
descriptor is the unweighted mean of the 45 pooled-condition F1 values;
report clean F1, clean-minus-corruption-mF1, per-family/per-level F1, pooled
corruption recall/FPR, and net FP/FN relative to clean repeated 45 times.
Do not turn these into a pass/fail gate or alter the model based on them.

Descriptive uncertainty is a 5,000-replicate percentile bootstrap over the
128 original source pairs, seed 20260926 (NumPy PCG64). Each replicate draws
128 pair indices with replacement; the **same** draw is used for clean and all
45 conditions, recomputing pooled F1 and the equal-condition mF1. Report 2.5%
and 97.5% quantiles for clean F1, corruption mF1 and their gap. This conditions
on this one trained checkpoint and source-pair collection; it is not an
interval over training seeds or a geographic-population guarantee. Also report
runtime, maximum observed guard temperature and measured peak inference GPU
memory, explicitly distinguishing them from the missing R1 training peak.

## Freeze, execution, and terminal rule

The exact test YAML must bind this document, R1 report/audit, archived B1
config/status/guard/checkpoint, archived base and corruption YAMLs, separate
test acceptance, test seed/counts, and fixed output directory by SHA256. The
approval records code and YAML hashes and the independent test scope. The
pretest seal binds approval, source, checkpoint, guard policy, synthetic-smoke
and full-unit-test evidence, plus `test_accessed=false`. Metadata preflight,
real CUDA identity and temperature-sensor preflight, and synthetic-only smoke
must not touch the real held-out test directory. The evaluation CLI must refuse
to run without a valid pretest seal, on any hash drift, with an existing output
directory, wrong GPU, missing guard, or a different checkpoint.

Run on `E:/Download/anaconda3/envs/geosa-basa/python.exe` with the frozen
single-GPU temperature guard. Audit test data first, then complete clean and
all 45 conditions without changing frozen inputs. The durable completion
marker is valid only when all counts, condition/stratum/source coverage,
clean/corrupt numerical consistency, input/source hashes, and guard completion
pass. A partial, guard-stopped, hash-drifted, or numerical-failure output is
**not** a final result and must not be silently restarted, mixed, or used to
revise the protocol. Report such failure for an explicit scientific recovery
assessment. After a valid result, do not rerun test for another model or use
the scores to tune this method. Subsequent work may synthesize and write up
the now-frozen research findings, not re-open model selection on this split.
