# R1E: evaluation-only recovery of the R1 seed-replication batch

Date: 2026-09-26. Status: approved by the user's standing delegated judgment,
subject to exact approval/hash freeze before any scoring. This is a **new**
evaluation-only batch. It does not amend or retroactively pass R1's frozen
evaluation implementation.

## Failure and scientific boundary

All four R1 training runs completed 100 epochs with guarded completion, paired
initial tensors and within-seed exposure equality. The original R1 pre-validation
seal has SHA256
`55a4454f2c34162871f4f7082e7282c57a27eaae347ef79547e4eae21743edbf`.
The first official-val attempt (B1, seed 43) failed in
`pcdc.evaluate_candidate._validate_checkpoints` before an output directory,
model inference, corruption scoring, or a guard attachment. The stdout is empty;
stderr SHA256 is
`0e5491547b84208ad8703eccfdcd8c8d30fa7f17dd698b16004f325b07d1f25e`.
No other R1 official-val attempt ran. The held-out test remained closed.

The original evaluator requires the checkpoint's whole training config and
train-corruption protocol fingerprint to equal the seed-42 benchmark. This is
incompatible with the R1-approved seed-43/44 changes even though inference,
validation files and validation instances are fixed. R1 section 6 prohibits
introducing a source change after its run starts. Therefore the old R1 evaluation
is closed as a pre-scoring protocol failure, not silently patched or retried.

## R1E scope and immutable inputs

R1E reuses the four hash-bound R1 `best.pt`/`last.pt` pairs, completed training
status/guards/exposure records, seed-42 official-val artifacts, original R1 seal,
and the same accepted 64-source / 1,024-tile / 45-condition validation manifest
and 46,080 deterministic instances. No model training, checkpoint reselection,
threshold change, new severity, hyperparameter choice, selector, local action,
or held-out test access is permitted. R1E outputs use new directories and do not
overwrite any old R1 or seed-42 artifact.

## Exact compatibility correction

The new R1E evaluator must validate each checkpoint against its own frozen R1
training YAML and corruption protocol, including exact configuration,
training signature, robust configuration, source hashes, status, guard,
exposure, best/last relationship, and file inventories. It separately compares
that training protocol against the immutable seed-42 training/validation
protocol. Only the R1-approved experiment name, output path, training seed,
train-stream seed, and paths/fingerprints derived solely from those differences
may differ. The training seeds are 43/44, train-stream seeds 310945/310946;
validation/test reserved seeds remain 310943/310944. All model, data,
preprocessing, split, loss, inference, threshold, optimizer, operator/severity,
validation manifest/instance and aggregation settings must remain identical.
The evaluator must fail closed on any other drift. It must not edit the frozen
`pcdc/evaluate_candidate.py` or the original R1 seal.

## Freeze, execution, and decision

Before validation inference, create and hash the separate R1E evaluator and
guard wrapper, four exact evaluation YAMLs, an approval mapping each output to
its own R1 training config and original R1 seal, and a new R1E pre-validation
seal. Unit tests must cover each allowed difference and reject disallowed drift;
run the complete unittest suite and a non-research smoke. Then run real CUDA,
guarded one-shot official-val evaluation sequentially B1-43, B2-43, B1-44,
B2-44. There is no partial-result reuse after a failed evaluation. Validate
each result, completion marker, clean-checkpoint replay, complete 45-condition
coverage and input/source hashes before starting the next. Abort on any hash,
numeric, isolation, guard or hardware failure.

Combine these four results with immutable seed-42 evidence using the already
sealed `scripts/analyze_r1.py`, 20,000 common-source bootstrap replicates,
seed 20260923, and exactly R1 section 10's predeclared B2-versus-B1 rule and
instability threshold. Report the original R1 failure and R1E correction
openly. Do not access held-out test without a separately frozen final-test
protocol.
