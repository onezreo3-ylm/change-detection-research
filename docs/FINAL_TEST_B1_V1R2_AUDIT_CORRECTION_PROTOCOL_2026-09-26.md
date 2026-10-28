# V1R2: one-defect correction to the sealed audit-only final-test recovery

Date: 2026-09-26. This is a separate, pre-score-pass correction protocol.
The original V1 test is still invalid/incomplete, and the sealed V1R audit
attempt is preserved as a failed auditor implementation. Neither source,
approval, seal, scores, nor output of V1 or V1R may be edited.

V1R's first score-record pass stopped at `Source counts do not sum: clean`.
Inspection of its **auditor code**, not of the test numbers, identifies an
unambiguous Python iterator defect: `add_counts` runs four independent sums
over the same one-shot generator. The first sum exhausts it; the other three
are zero. Its prior synthetic test supplied a list, missing the case. The
frozen V1R source SHA256 is
`d001ed2697c26a183a035b43836bbf9260350a15b2268332c0657ace7e9b69f7`;
its seal SHA256 is
`544bf807c53ae5b3a69310636805d9025299e970f8a416f4e65c4e44a73e8153`.
The failed attempt is documented in
`runs/final_test_b1_v1r_protocol/failure_audit.json`. No test score was
reported or used for model selection.

V1R2 retains every scientific rule, artifact, fixed seed, formula and check
in the V1R protocol. The **sole** correction is to materialize the iterable
once before summing TN/FP/FN/TP, with a new synthetic generator-regression
test. V1R2's wrapper code, regression test, YAML, approval, the V1R source,
and all preexisting V1 input bytes must be sealed before the second
score-record pass. The wrapper must bind the failed V1R audit and its seal.
It may not inspect raw held-out images/labels, load a checkpoint/model, invoke
CUDA inference, change V1 score records, change the B1 seed-42 choice, or
revise statistical thresholds. Full unittest and synthetic-only smoke are
required again. The V1R2 certificate, if any, must be written in a new output
directory and identify V1R2's seal; V1/V1R remain unchanged.

Certification requires all V1R independent checks to pass after this one
materialization fix. If any further check fails, retain V1R2 as failed and
report the exact cause; this protocol does not authorize a further test
inference or an unreviewed succession of auditor revisions. Successful V1R2
results may be reported only as **audit-recovered existing one-shot scores**,
with the original V1 missing guard-completion marker and failed guard-state
bookkeeping explicitly disclosed. No test-driven model changes follow.
