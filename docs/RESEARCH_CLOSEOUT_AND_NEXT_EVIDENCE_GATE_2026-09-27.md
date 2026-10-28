# LEVIR-CD research closeout and next evidence gate

Date: 2026-09-27. This is an evidence/claim boundary and reproducibility
handoff, not a newly trained or newly tested model. The currently approved
experimental line has reached its terminal decisions. The user has delegated
ongoing judgment, but that does not make the already spent held-out test a
development set or authorize an unsupported new mechanism.

## What the evidence supports

| Question | Frozen evidence | Defensible conclusion |
| --- | --- | --- |
| Can the Phase 7 whole-tile action selector beat its stronger constant action? | [Phase 7 result](PHASE7_RESULTS_2026-09-23.md): all eight learned policies have negative point mF1 differences versus G0; none passes the paired-source or harm safeguards. | `stop_whole_tile_selector_route`. An oracle-like label-informed diagnostic is not a deployable selector or proof of a learnable gain. |
| Does ordinary B2 consistency improve over matched B1 augmentation? | [R1/R1E three-seed replication](R1_R1E_REPLICATION_RESULT_2026-09-26.md): mean B2−B1 corruption mF1 −0.001018; 20k paired-source interval [−0.003249, +0.000829]; all five preregistered B2 gates fail. | `retain_b1`. B2's smaller FPR/gap does not rescue the primary score and clean/recall failures. |
| What does the preselected B1 seed-42 checkpoint do on held-out LEVIR-CD? | [V1R2 recovered result](FINAL_B1_HELDOUT_TEST_V1R2_RECOVERED_RESULT_2026-09-26.md): clean F1 0.898428; 45-condition corruption mF1 0.894760. | A one-checkpoint descriptive result, separately audit-certified. Original V1 is still incomplete and must not be called a valid guarded completion. |

The supported contribution is a disciplined, reproducible **negative
mechanism finding** plus a strong ordinary-augmentation baseline, not a new
adaptive correction method or a general claim that local/conditional methods
cannot work. The corruption suite is synthetic and non-geometric; these
scores do not establish natural-weather, misregistration, geographic transfer,
or field deployment robustness. Six-neighbor public-prior-art review in
[the prior-art note](PUBLIC_PRIOR_ART_2026-09-12.md) is not a novelty proof.

## Reproducibility package and failure disclosure

The [closeout manifest](../configs/research_closeout_20260927.yaml) binds the
Phase 7 decision, R1/R1E analysis/audit, final report, original V1 failure,
V1R2 seal, certificate and audit by SHA256. Run the metadata-only verifier:

```powershell
& 'E:\Download\anaconda3\envs\geosa-basa\python.exe' -m scripts.verify_research_closeout check
```

It checks the 14 named artifacts and all 70 V1 inputs already bound by the
V1R2 certificate, plus the decision and failure-state chain. It hashes
derived artifacts and the preselected checkpoint as bytes; it does **not**
open raw test images/labels, deserialize weights, run inference or recalculate
test metrics. The verifier is an integrity check, not a replication of the
model's predictions. Run `synthetic-smoke` for its fixture-only tamper test.

The failure path is part of the result: V1 scored clean and all 45 conditions
but failed during final guard-completion bookkeeping; original status and
guard state remain failed, and the completion marker does not exist. V1R's
first auditor had a generator-exhaustion bug and did not certify anything.
The separately sealed V1R2 corrected only that audit bug and certified the
unchanged existing score records. Never replace this chain with a claim that
V1 completed normally, or conceal the absent guard marker.

## Next experiment is conditional on genuinely new development evidence

1. **No model experiment on the spent test.** Do not re-evaluate B2, other
   seeds, thresholds, severities, selectors or local actions on the current
   LEVIR-CD test split. Do not tune on the recovered result, including its
   worst family/level. The Phase 7 whole-tile route stays closed.
2. A new study first needs an independently sourced, legally usable image-pair
   collection with documented labels, identity and spatial provenance. Before
   modeling, define source/geographic units, detect exact encoded/decoded
   duplicates against existing splits, and assess geographic overlap where
   provenance permits. These checks are prerequisites, not promises that
   count-only manifests establish independence. No new dataset has been
   designated by the present protocol.
3. Freeze a **new** development/holdout partition, corruption acceptance,
   candidate action, negative controls, primary endpoint, harm constraints,
   source-level uncertainty, compute budget and stopping gates *before* any
   new holdout inference. A local-action or signature model is warranted only
   if train-only diagnostics and a separate validation gate show a stable
   actionable margin over no-op and always-act controls. If this gate fails,
   stop that route as a normal finding. Keep the new holdout sealed until the
   method and hyperparameters are fixed.
4. If such independent data are unavailable, the next useful work is
   manuscript/reproducibility preparation and independent code review, not
   another GPU run. Any publication must label the V1R2 outcome as an
   audit-recovered one-shot score and distinguish validation-based selection
   from held-out description.

This sequence allows autonomous preparation, integrity monitoring and
protocol drafting while preventing a scientifically invalid test-driven
feedback loop. New external data acquisition or a materially different
research question requires its own documented scope and provenance before
execution; nothing here silently substitutes a resource.
