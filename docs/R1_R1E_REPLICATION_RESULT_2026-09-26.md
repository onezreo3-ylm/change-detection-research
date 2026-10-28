# R1 training / R1E evaluation: three-seed baseline replication result

Date: 2026-09-26. Decision: **retain B1** (ordinary supervised corruption
augmentation). This is a development-validation method decision, not a new
mechanism claim or a held-out-test result.

## Provenance and protocol disposition

R1 completed four real-CUDA 100-epoch trainings (B1/B2 at seeds 43/44), with
matched within-seed initial tensors, source/exposure streams, attempted updates,
and guarded completion. The immutable seed-42 pair supplies the third seed.
Original R1 official-val evaluation failed for B1-43 **before any inference or
corruption score** because the old evaluator incorrectly required the changed
training seed/name/train-stream fingerprint to equal the seed-42 benchmark.
Its output directory was never created, stdout was empty, and no other original
R1 evaluation ran. The original R1 evaluation batch was closed, not patched.

A distinct [R1E recovery protocol](R1E_REPLICATION_EVALUATION_RECOVERY_PROTOCOL_2026-09-26.md)
was approved and hash-frozen before scoring. Its separate evaluator checks each
checkpoint against its own sealed training config, then proves that only the
R1-declared training identity and train-stream seed differ from the seed-42
reference. All original model, validation manifest/instances, label decoding,
threshold, operators/severities, and aggregation rules remain fixed. The old
evaluator and original pre-validation seal remain unchanged. R1E's four
one-shot official-val runs completed with clean-checkpoint replay, 45/45
conditions, matching source/condition coverage, input/source hashes, and
temperature-guard completion. Held-out test was not read or scored.

- Original R1 pre-validation seal SHA256:
  `55a4454f2c34162871f4f7082e7282c57a27eaae347ef79547e4eae21743edbf`.
- Original pre-scoring failure stderr SHA256:
  `0e5491547b84208ad8703eccfdcd8c8d30fa7f17dd698b16004f325b07d1f25e`.
- R1E pre-validation seal SHA256:
  `78632392842ccf4298db7fac28e27797cac5b7dddcf5b651f2fe705f2510b9b0`.
- [Six-run analysis manifest](../runs/r1e_protocol_20260926/analysis_manifest.yaml)
  SHA256: `104a414e6ba3ec9c1394d9125cb1629cb94a6414b44a97bde7f52680fb264390`.
- [Machine-readable analysis](../runs/r1e_protocol_20260926/analysis.json)
  SHA256: `71909cac24279f5803c30d11494d61ef823e9539a99cd5a00fc77a8ed1a8df09`.
- Frozen analysis source `scripts/analyze_r1.py` SHA256:
  `bbebdbda1acf647dfabb5ee718b21e24672380e742e47c2b38639177f6a8d81c`.

## Official validation results

Corruption mF1 is pooled-pixel F1 per condition, averaged equally across the
45 accepted non-geometric conditions. All F1 values are fractions, not percent.

| Seed | B1 clean F1 | B1 corruption mF1 | B2 clean F1 | B2 corruption mF1 | B2−B1 mF1 | Condition wins B1:B2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.906073 | 0.902376 | 0.903827 | 0.900742 | −0.001634 | 43:2 |
| 43 | 0.903419 | 0.899568 | 0.904095 | 0.900387 | +0.000818 | 17:28 |
| 44 | 0.904068 | 0.899653 | 0.900667 | 0.897415 | −0.002239 | 40:5 |

Three-seed mean ± sample standard deviation:

| Metric | B1 | B2 | Mean B2−B1 |
| --- | ---: | ---: | ---: |
| Clean F1 | 0.904520 ± 0.001384 | 0.902863 ± 0.001907 | −0.001657 |
| Corruption mF1 | 0.900533 ± 0.001597 | 0.899514 ± 0.001827 | −0.001018 |
| Clean−corruption gap | 0.003987 ± 0.000378 | 0.003349 ± 0.000323 | −0.000638 |
| Pooled corruption recall | 0.891717 ± 0.004495 | 0.888315 ± 0.004308 | −0.003402 |
| Background FPR | 0.003885 ± 0.000128 | 0.003801 ± 0.000083 | −0.000084 |
| Net FP vs repeated clean | 396,099 ± 42,539 | 323,076 ± 71,138 | −73,023 |
| Net FN vs repeated clean | 585,158 ± 116,132 | 497,219 ± 22,853 | −87,939 |

The source-paired 20,000-replicate bootstrap (seed 20260923; common original
validation source-pair draw across all three seeds/methods/conditions) gives
the 95% percentile interval for mean B2−B1 corruption mF1:
**[−0.003249, +0.000829]** (median −0.000968). This is a source-pair interval,
not a population interval over all possible training seeds; three seeds are
too few for the latter.

The frozen R1 selection rule requires five simultaneous B2 wins. All five fail:
the mean mF1 delta is below +0.0005; bootstrap lower bound is not positive;
B2's mean clean F1 is more than 0.0005 below B1; its pooled corruption recall
is more than 0.001 below B1; and it wins only one of three seeds. The smaller
B2 clean-to-corruption gap and FPR do not override absolute mF1. Cross-seed
mF1 ranges are 0.002808 (B1) and 0.003327 (B2), both below the 0.005
instability stop threshold. Hence the predeclared decision is **retain B1**,
not an instability stop.

## Compute and reporting limits

Each run attempted 178,000 optimizer updates and processed 1,424,000 paired
views. `Measured hours` below sums the frozen per-epoch train and validation
phase timers; it excludes setup, checkpoint I/O and pauses, so it is not a
whole-process elapsed wall clock. The number of successful updates equals
attempts minus AMP skips.

| Run | Successful / attempted updates | AMP skips | Measured train+val hours |
| --- | ---: | ---: | ---: |
| B1-42 | 177,936 / 178,000 | 64 | 8.13 |
| B2-42 | 177,933 / 178,000 | 67 | 8.49 |
| B1-43 | 177,933 / 178,000 | 67 | 8.17 |
| B2-43 | 177,934 / 178,000 | 66 | 8.33 |
| B1-44 | 177,935 / 178,000 | 65 | 7.80 |
| B2-44 | 177,936 / 178,000 | 64 | 7.82 |

Peak GPU memory was **not instrumented** by the frozen R1 training code and
cannot be reconstructed reliably after the runs. This is an explicit reporting
gap, not a zero or substituted benchmark value. No retraining is performed to
fill it. The RTX 3060's capacity and spot `nvidia-smi` readings are not peak
training memory measurements.

## Route closure and next boundary

B1 is retained as the simple development baseline. R1/R1E do not support
reopening Phase 7 whole-tile selectors or launching local actions. The original
seed-42 B1 checkpoint remains the incumbent operational candidate; no
post-hoc choice among seed-42/43/44 checkpoints is claimed from corruption
validation. Before any held-out test access, a separate final-test protocol
must explicitly freeze the single checkpoint, model/threshold/data settings,
evaluation suite and reporting plan, with no test-driven revisions. The
held-out test is still closed.
