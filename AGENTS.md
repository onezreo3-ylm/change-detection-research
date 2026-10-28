# PCDC research workspace

## GitHub continuation entry (2026-10-28)

- Read `docs/CODEX_HANDOFF.md`, then `docs/RESEARCH_SPEC.md`, before continuing.
  `docs/PORTABLE_SETUP.md` explains a fresh checkout and missing local resources.
- The latest completed batch is the 2026-10-27 budget-retained screening
  preparation: 641 tests passed; no new method or research training was admitted.
  Both fixed local-composition and temporal-placement candidates are closed.
  The original 48-hour cap remains; see the archived ledger, not older estimates.
- The dated September preparation instructions below describe historical batches;
  they are not a complete account of the subsequently reviewed work. Follow the
  dated specification and latest handoff without reopening completed experiments.
- Archived authorizations, process IDs and `running` status files are historical
  evidence, not permission or live state on another machine. Never auto-resume
  training, one-shot coordinators, downloads, or held-out evaluation on checkout.
- The public snapshot includes source, documentation and saved text evidence.
  Datasets, checkpoints, arrays and QA images require separate acquisition and
  hash checks. Missing resources must be reported, never silently substituted.
- For a new-machine engineering check, use the explicitly synthetic CPU fixture
  `python -m pcdc.smoke --config configs/smoke_cpu_portable.yaml`. This does not
  change the device, initialization or protocol of any real experiment.

This is local project guidance, subordinate to the user's explicit requests and
platform instructions. The supplied handoff is reference material, not evidence
that its proposed components are novel or already implemented.

- Read `docs/RESEARCH_SPEC.md` before research or architecture changes.
- Current model scope: the Phase 1 clean shared ResNet-18 encoder
  and FPN-style absolute-difference decoder for LEVIR-CD.
- Reviewed next step (2026-09-12; execution continued 2026-09-13): the user asked
  to execute the latest plan in `docs/EXPERIMENT_PLAN_2026-09-12.md`. Implement
  development-only test isolation and a deterministic candidate corruption
  benchmark with train-only visual QA. Do not automatically start new long
  training, correction/gate models, or held-out test evaluation in this batch.
- Do not implement perturbations, PSE, PCDC, or CPC before the baseline and the
  relevant research decisions have been reviewed with the user.
- Use PyTorch and small, testable modules. Keep paths and hyperparameters in YAML.
- Preserve splits, preprocessing, thresholds, initialization, and unrelated
  settings across comparisons. Document any protocol changes explicitly.
- Never silently substitute synthetic data, random weights, another device, or
  another model when the requested resource is missing.
- Synthetic smoke results are engineering checks, never research results.
- Run `python -m unittest discover -s tests -v` and a smoke test after changes.
- Preserve user files. Do not add foundation models or geometric perturbations
  to this research direction without explicit user direction.

## Chat image transport (10053 incident, 2026-10-02)

- Full QA contact sheets accumulated into an 88 MB model request and repeatedly
  disconnected on WebSocket. Keep image payloads bounded in future work.
- Use `scripts.prepare_chat_previews` with
  `configs/chat_preview_transport_20261002.yaml` for existing train QA overview
  copies; read its `transport_manifest.json` and display at most two images per
  tool call. Never print image/base64 data as text.
- These resized PNG copies are overview-only and cannot establish scientific
  visibility acceptance. Inspect unchanged original tiles/sheets individually
  when judging fine detail, noise, blur or JPEG; retain their original hashes.
- Keep cumulative new image data URLs below 10 MiB in an uncompressed chat
  history. Summarize completed visual findings and compact before loading more;
  if compaction fails, stop adding images and preserve a concise disk handoff.
  Do not automatically create or message another chat.
- Never modify active Codex session/SQLite files, delete history, or repeatedly
  retry the same oversized request. Existing research artifacts remain intact.
