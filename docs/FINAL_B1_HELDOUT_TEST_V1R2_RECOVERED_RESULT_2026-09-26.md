# Final B1 held-out test: V1R2 audit-recovered result

Date: 2026-09-26. The single preselected B1 seed-42 checkpoint's existing
one-shot held-out outputs are **certified by a separate read-only V1R2 audit**.
The original V1 run is **not** retrospectively valid or complete: its
`status.json` remains `failed_incomplete_no_automatic_retry`, the guard state
remains `failed_guard`, and `guard_completion.json` is absent. No original V1
file was changed, no test image/label was reopened during recovery, and no
second test inference or model selection occurred.

## Chain of evidence and disposition

R1/R1E's validation-only three-seed gate retained B1 before held-out test
access. The operational incumbent was already the seed-42 epoch-68 `best.pt`
(SHA256 `c4c0b849081bc5ec5de501b5e5e6c36b07c1c6b7b8ffb403bbeec49d1b1d746d`),
selected on clean validation F1; no later corruption or test result chose it.
The one-shot V1 pretest seal SHA256 is
`33db6693bf3868bc056d68970c488f3ea2685cebf7f3368db0bb79cbd51ede67`.

V1 completed clean scoring and all 45 fixed non-geometric conditions, then
failed while constructing its final guard-completion marker: its local
`_sha(text)` validator shadowed a byte-digest helper. The exception happened
**after** its summary and status write, and the exception handler marked
status/guard failed. The original [failure audit](../runs/final_test_b1_v1_protocol/failure_audit.json)
records this; it is not a valid guarded V1 completion. A sealed V1R auditor
then stopped on its own one-shot-generator summation defect without issuing a
certificate or reporting scores. Its [failure audit](../runs/final_test_b1_v1r_protocol/failure_audit.json)
is preserved. The separately frozen [V1R2 correction protocol](FINAL_TEST_B1_V1R2_AUDIT_CORRECTION_PROTOCOL_2026-09-26.md)
changed only that auditor summation behavior, added a generator regression,
and passed 206 unittest cases plus synthetic smoke before its second
score-record pass.

V1R2 independently replayed the reserved-seed 310944 ID-only JSONL recipe,
verified 128 original source pairs, 2048 non-overlapping tiles, 45 conditions
and 92,160 corruption instances; reconciled every pooled, source, stratum and
injection-region integer confusion count; recomputed all metrics and the
5,000-replicate paired source bootstrap; and checked source/checkpoint/input
hashes, original traceback and 143 temperature observations. The original
input byte snapshot was identical before and after certification. The
[separate certificate](../runs/final_test_b1_v1r2_audit/recovery_certificate.json)
SHA256 is `68b8a52415f6af50a856dc7c080f9355139d1c642325d25c69b1b273bfd5c5ed`;
the [V1R2 addendum](../runs/final_test_b1_v1r2_audit/v1r2_correction_addendum.json)
SHA256 is `e8abf42dd43c1d7aa6c3fa7217ebd25a855bb91b514b8e880814693cc5fd8427`.
V1R2's preaudit seal SHA256 is
`556a6ef17fb78b99af2b3a679e3f0d9d321ba676403c6df096be8db02034d29c`.

## Single-checkpoint test descriptors

All F1 values are fractions, not percentages. The primary robustness
descriptor is the equal mean of 45 condition-level pooled-pixel F1 values,
not F1 from all corruption pixels pooled together.

| Descriptor | Audit-recovered value |
| --- | ---: |
| Clean F1 | 0.898428 |
| Corruption mF1, 45 conditions | 0.894760 |
| Clean minus corruption mF1 | 0.003669 |
| Pooled corruption recall | 0.879080 |
| Pooled corruption background FPR | 0.004609 |
| Net false positives vs clean repeated 45× | +870,944 |
| Net false negatives vs clean repeated 45× | +1,300,284 |

The fixed 128-source-pair, 5,000-replicate percentile bootstrap gives 95%
intervals: clean F1 `[0.886898, 0.907585]`; corruption mF1
`[0.883173, 0.904030]`; clean-minus-corruption gap `[0.003140, 0.004254]`.
The lowest of the nine family means is color bias (0.887826); level 5 has
the lowest level mean (0.889260). These are descriptive test outcomes, **not**
an invitation to retune severity, threshold, model or training seed.

Original V1 scoring elapsed 1700.906 s and recorded peak allocated inference
GPU memory of 1,058,620,928 bytes. Its 143 guard observations peaked at 64°C
and ended at 58°C; the policy pause threshold was 85°C. The embedded summary
guard record says `completed`, but the persisted guard state became
`failed_guard` during post-scoring exception handling. This audit supports a
separately recovered score record, not a claim that the original V1 guarded
completion marker exists. R1 training peak GPU memory was not instrumented
and is not inferred from the inference peak.

## Scientific boundary

The validation-based decision remains **retain ordinary supervised B1**;
neither B2 nor another seed was evaluated on test. These scores describe one
checkpoint on one held-out split with controlled synthetic interventions.
The source-pair intervals condition on that checkpoint and collection; they
do not quantify training-seed or geographic-population uncertainty. Original
train/val/test pair audits check exact encoded/decoded duplicates but do not
establish geographic independence. The test is now spent for method
development. Do not rerun it to compare models, reopen the rejected whole-tile
selector route, or launch unsupported local actions. The defensible next
activity is reporting/reproducibility packaging or a separately designed
study on fresh development data, not test-driven tuning.
