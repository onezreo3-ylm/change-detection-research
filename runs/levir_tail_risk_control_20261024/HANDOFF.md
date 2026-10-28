# Next delegated ordinary training control

This is an ordinary sample upper-tail loss control, not a novel method.
The user authorized continued training with hourly checks on2026-10-24.
Current protocol and YAML are fixed; implementation, real-CUDA pilot and
formal single-worker launch authorization are NOT complete. Training has
NOT started. Do not say it is running or wait for another routine permission.

Read docs/LEVIR_TAIL_RISK_CONTROL_PROTOCOL_2026-10-24.md,
configs/levir_tail_risk_control_20261024.yaml,
feasibility_from_saved_counts.json, and the mutable managed status/ledger.
The mean-risk B1 baseline is already strong and successful; all nine core
files still match its original source hashes. Keep model, optimizer, seed,
100epochs, FP16 AMP, clean selection, accepted corruption stream/exposure and
threshold unchanged. Use a new explicitly identified risk-training flavor.

Implementation can reuse unchanged run_two_view_epoch with a separate loss
object that receives ordered clean+secondary logits. For N source samples:
0.5*mean(original per-sample clean BCE+Dice) + 0.5*mean(largest ceil(N/2)
secondary per-sample BCE+Dice). Fixed deterministic tie order. Original
mean validation criterion and clean-F1 checkpoint selection remain.
Do not monkeypatch old module globals, modify its archived source, or claim
equivalent effective supervision gradients; all physical views still run.

First validate mean/q1 loss+gradient identity, original tensor initialization,
repeat/next-step replay, real tail batch and actual source/instance streams.
Run each declared loss's32-step real-CUDA engineering check, record all
physical compute and AMP skips/peak memory/timing, and test the24h/48h budget
with25% margin. Preserve failed fields and never silently change precision.
Only after data/code/config/flavor/source identities and bounded runtime pass
should a hidden CPU owner and independent PID/creation-bound seconds-level
temperature guard launch the single candidate. Budget watchdog is external;
hourly App follow-up is not the runtime or thermal guard.

Explicitly adapt fixed45-condition development evaluation to the new true
checkpoint flavor; old readers stay strict. Never use old LEVIR test or
Martigny. Gates are frozen in YAML; no q/weight/strength grid or extra modules
after failures. A seed42 pass only supports considering43/44 replication.
