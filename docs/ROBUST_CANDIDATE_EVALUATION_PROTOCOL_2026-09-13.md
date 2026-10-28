# Robust candidate frozen-validation protocol

This evaluator compares a completed B0-E/B1/B2 run with the already completed
B0 corruption evaluation. It does not authorize held-out test access.

## Fixed inputs

- The existing B0 evaluation config, accepted visual-severity record, validation
  manifest, and 45-condition instance table are reused without regeneration.
- The B0 summary and condition records are bound by SHA256 and checked against
  the frozen benchmark fingerprints and condition/stratum coverage.
- A candidate job binds `best.pt`, `last.pt`, `status.json`,
  `guard_completion.json`, and `training_exposure_summary.json` by SHA256.
- `best.pt` must be the clean-validation-selected checkpoint from the same
  completed run as `last.pt`. The last checkpoint must contain the complete
  development-only history and reproduce the saved exposure summary.
- Research and synthetic-smoke artifacts have an explicit `expected_kind`; a
  smoke checkpoint cannot be relabeled as research evidence.

## Scoring and comparison

Clean validation is evaluated first and must reproduce the candidate
checkpoint's integer confusion counts and derived metrics exactly. Each of the
45 corruption conditions is then evaluated separately. Pixels are pooled
within a condition before metrics are calculated, and corruption mF1 is the
equal mean of condition F1 values.

The result records:

- clean F1 and corruption mF1, with candidate-minus-B0 deltas;
- all condition and family F1 values;
- pooled error counts, foreground recall, and background false-positive rate;
- per-condition count/metric deltas;
- paired records grouped by the 64 original validation image pairs, not by
  treating correlated tiles as independent samples.

All model execution is wrapped by the same in-process GPU temperature guard as
the B0 evaluation. A result is valid only when `guard_completion.json` is the
last successful commit marker. Every output continues to record
`test_evaluated: false` and `training_started: false`.

## Commands

```powershell
python -m scripts.evaluate_candidate_guarded `
  --config configs/<candidate-evaluation>.yaml `
  --guard-config configs/evaluation_temperature_guard.yaml `
  --preflight-only

python -m scripts.evaluate_candidate_guarded `
  --config configs/<candidate-evaluation>.yaml `
  --guard-config configs/evaluation_temperature_guard.yaml
```

The config is created only after a guarded training run completes, so every
candidate artifact hash is known and frozen before scoring begins.
