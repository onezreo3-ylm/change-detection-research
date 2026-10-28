# Temporal-placement feasibility result

2026-10-27. Fixed train-only exploratory upper-bound probe; no learned method or held-out evaluation.

| Group | Old16 oracle mF1 | New256 oracle mF1 | Best fixed mF1 | Gain vs best fixed (pp) | Extra vs old16 (pp) | Joint gates |
| --- | --- | --- | --- | --- | --- | --- |
| all | 0.904790492 | 0.905352499 | 0.906386511 (full_both_counts) | -0.103401 | +0.056201 | FAIL |
| fit | 0.905128901 | 0.905684318 | 0.907377281 (full_both_counts) | -0.169296 | +0.055542 | FAIL |
| calibration | 0.903987029 | 0.904564684 | 0.904031224 (full_both_counts) | +0.053346 | +0.057765 | FAIL |

Fit and calibration each required +0.5pp over the strongest of seven constants and +0.2pp over the old16 oracle, plus original utility-sign/family gates. No rule changed after outcomes.

The fixed temporal-placement candidate is CLOSED. Even label-informed selection did not satisfy the frozen joint utility gates. Do not fit its predictor, enlarge regions, change thresholds or reselect sources to rescue it.

The original a0/full-both counts and every old16 menu replayed exactly for all measured train cells. Pilot clean/brightness_l1 each actually checked all256 assignments on one original32 batch; main checked four fixed mixed assignments per batch and derived the remaining252. CUDA FP16 composition and final identity checks passed.

All256 choices are diagnostic and use labels; this is not deployment accuracy, novelty certification or a universal impossibility result. Twelve fit and four calibration sources were previously used development data. Reference normalization/cache and generic routing/finite optimization remain known components. Decoder view counts do not establish a latency advantage.

Protected worker and independent owner/guard/watchdog exited normally; maximum recorded GPU temperature 66C. No optimizer updates, raw test or Martigny access. Original cumulative48h ledger retained.

Evidence: runs/temporal_action_probe_20261027/result.json, completion.json, probe_rows.jsonl, authorization.json. Protocol: docs/TEMPORAL_ACTION_FEASIBILITY_PROTOCOL_2026-10-27.md.
