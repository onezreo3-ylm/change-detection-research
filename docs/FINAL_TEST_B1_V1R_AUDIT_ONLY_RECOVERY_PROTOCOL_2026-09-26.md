# V1R: audit-only recovery of the failed one-shot B1 final test

Date: 2026-09-26. This is a new, separately approved forensic evaluation of
the **existing** one-shot output. It does not amend the frozen V1 final-test
protocol, evaluator, approval, pretest seal, test data, or output files. The V1
run remains `failed_incomplete_no_automatic_retry`; its missing
`guard_completion.json` must never be fabricated or described as present.

## Why recovery is scientifically preferable to another test inference

The V1 process evaluated the preselected B1 seed-42 checkpoint on clean test
and all 45 fixed corruption conditions. Its progress and 45 condition files
exist. It reached the write of `summary.json` and `status.json`, then failed
when the local `_sha(text)` validation helper, which shadows an imported
`_sha(bytes)` digest helper, received `summary.json` bytes while constructing
the final guard-completion marker. The error is at
`scripts/final_test_b1_v1.py:808`; the original source SHA256 is
`fc48e169ef45e57af9b2058c20f4b4d165a3f787410cf62bee6472c9d5c070cd`.
The exception handler then recorded `failed_guard`. The preserved guard state
still says `evaluation_status=completed`, and the embedded summary guard
record says `completed`; this is evidence of a finished final temperature
check, **not** a substitute for the missing V1 completion marker. The observed
maximum temperature was 64°C, below the fixed pause threshold. The failure is
documented in `runs/final_test_b1_v1_protocol/failure_audit.json`, SHA256
`83f052b71ee66f2b04ae29498a153cf4543dddd6b92ddcdbba86680bd30ff214`.

A second test inference would expose the held-out split again, risk selective
repetition, consume GPU time and contradict the one-shot design. V1R instead
checks whether the already generated evidence supports a **separately
certified recovered result**. It cannot retroactively turn V1 into a valid
guarded run. If any independent check fails, V1R stops without reporting test
scores as valid. There is no model, threshold, seed, severity, checkpoint,
preprocessing, dataset, or statistical-method choice in V1R.

## Immutable input and no-new-access boundary

V1R reads only the frozen configuration, checkpoint/provenance metadata,
original logs, guard state, test-data audit/manifest/instance JSONL, clean and
condition **score records**, and original summary. It must not enumerate,
hash, decode, or infer on raw held-out image or label files; it must not load
the model on CUDA. It never writes into or deletes the V1 output directory.

Before parsing numerical score records, create exact V1R code/tests and YAML,
run synthetic-only unit/smoke checks and the full unittest suite, record source
hashes and an approval, and seal the complete **byte hashes** of all V1 input
artifacts. Byte hashing for sealing is not score interpretation. The seal
records `scores_interpreted=false` and `new_test_inference=false`. The recovery
output uses a new directory and is created only after the seal passes. The
original V1 status and failed guard record remain immutable.

## Independent verification required for certification

1. Recheck V1 protocol/config/approval/pretest-seal, checkpoint, original
   evaluator/guard source and failure-audit hashes. Require the exact
   post-scoring traceback and failed status, no original guard-completion
   marker, no active matching process, and 45 condition files. Bind every V1
   input byte hash before calculating a metric.
2. Recheck the audit/manifest shape: 128 original test pairs, 2048 exact
   non-overlapping tile IDs, 45 conditions, 92,160 instances and expected
   condition/stratum coverage. Independently replay the ID-only reserved-seed
   `310944` instance recipe and compare every canonical JSONL row, without
   opening any image or label. Check encoded file SHA strings are well-formed;
   V1 control flow reached its post-scoring file rehash before summary write.
3. Independently validate each clean/condition report from nonnegative integer
   TN/FP/FN/TP: sample and pixel totals, 128 source-pair groups with 16 tiles
   each, source/stratum/regional count reconciliation, and all pooled metrics at
   the predeclared zero-denominator convention. Require `conditions.json` to
   equal the 45 individual files and to match the original summary's hashes.
4. Independently recompute the unweighted 45-condition mF1, per-family and
   per-level F1, pooled corruption recall/FPR, net FP/FN against clean repeated
   45 times, and the 5,000-replicate common-source-pair percentile bootstrap
   with seed `20260926`. Compare with the unchanged V1 summary to tight
   floating tolerance (`1e-12`); reject NaN, infinity or missing fields.
5. Validate all 143 original guard observations, GPU identity, source/config
   hashes, monotonic times, sensor decisions and recorded final check. Require
   an exact bookkeeping-only failure trace after the summary/status write;
   neither a temperature stop nor an earlier scoring exception is recoverable.
   Preserve and disclose `temperature_guard_state.status=failed_guard` and
   `guard_completion.json` absent even if certification succeeds.

Only when **all** checks pass may V1R write a separate recovery certificate
that binds the unchanged V1 summary and every input hash. The certificate
must state `original_v1_valid=false`, `new_test_inference=false`,
`recovery_type=audit_only_post_scoring_bookkeeping_failure`, and
`recovered_scores_certified=true`. It must not create a V1 guard completion
file. The scientific report may then quote the single checkpoint's clean and
corruption test descriptors, bootstrap intervals, compute and limitations,
clearly labeled as V1R-certified recovery. The R1 development decision remains
unchanged; no B2/other-seed test comparison or test-driven model changes follow.

If certification fails, retain both V1 and V1R as failed and report exactly
which invariant failed. No second test inference is authorized by this
protocol. This recovery is a reproducibility/audit remedy, not an opportunity
to revise the method after seeing held-out scores.
