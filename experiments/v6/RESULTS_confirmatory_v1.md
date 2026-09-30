# V6 Confirmatory Experiment — Results

*Report date: 2026-09-30.*

- **Record:** `experiments/v6/runs/confirmatory_v1/record.json` (sha256 `18d11300d599e75cad29a3aa5e4584b7280adce71657832fcd6032da6114e3d9`).
- **Protocol:** `experiments/v6/AMENDMENT3.md` at tag `v6-amendment3-lock` (commit
  `116035c575a770700432241bb7acc23c65876b34`).
- **Unchanged:** this report does not modify the protocol, the pre-registration, the runner or
  the record.
- **Verification:** every number below was read from the record and **independently recomputed
  bitwise** from the stored raw data (§8).

## 1. Protocol

- **Estimand.** D\*(A, B) = min over within-layer head permutations of the squared distance
  between unit-normalised per-prompt attribution tensors: last-position Taylor attribution,
  name-swap corruption, 200 locked IOI prompts. That is, the **distance between attribution
  profiles, modulo within-layer head permutation** (AMENDMENT3.md §1).
- **Only confirmatory hypothesis (H1, run level).**
  - Δ = mean over distinct primary runs of D\*(r@143000, r′@143000) − mean over primary runs
    and checkpoint pairs {(123000,133000), (123000,143000), (133000,143000)} of D\*(r@s, r@t).
  - H0: Δ ≤ 0.
  - Test: the Gate 5 two-way (runs × prompts) bootstrap with **B = 200**, inherited from the
    validated Gate 5 procedure. One-sided 5th-percentile lower bound.
  - **Decision rule:** SUPPORTED_AT_RUN_LEVEL iff lower bound > 0.
- **H2–H4:** not tested in this experiment; they remain UNRESOLVED.
- **Gate 1** (pair-level intervals) failed and is closed, with no third attempt. Per-pair D\*
  values are descriptive point estimates only.

## 2. Confirmatory dataset

- **Runs.** 10 candidate runs: `EleutherAI/pythia-410m` (reference) and `pythia-410m-seed1` …
  `seed9` (PolyPythias).
- **Checkpoints.** 123000, 133000 and 143000 per run, each loaded by pinned commit SHA and
  verified against the Hub LFS sha256 of its weights file.
- **Prompts** (locked files, disjoint, unique):

  | Set | n | sha256 |
  |---|---|---|
  | attribution | 200 | `aa3e32c8ab51d4aabe96e83b51d30a947e120b68bfcbf0da1dff87868abee0d2` |
  | matching | 1000 | `0849499b8117dac405743c2b33f500545f42aa1f4faff85461feecdef475d296` |

- **Seeds.**

  | Component | Seed |
  |---|---|
  | Prompt resampling | 20260929 |
  | Run resampling | 20260930 |
  | S3 cross-fit | 20260931 |
  | Model inference | deterministic (no RNG) |

Checkpoint provenance (full weight hashes are in the record):

| Run | Step | Pinned = resolved commit | Status | Weights sha256 (verified) | Load attempts |
|---|---|---|---|---|---|
| reference | 123000 | `51cabaff46863a0421bf3b44194431cc94d3466e` | ok | `e2e14ed434a44812…` | 1 |
| reference | 133000 | `45c291dfb81b96ad2522eabf89eb953045cd537a` | ok | `dd788b3f14ab853a…` | 1 |
| reference | 143000 | `bba6a464f54bbf08fc174cfb351d9794d58af21d` | ok | `1c88b0bf18293fae…` | 1 |
| seed1 | 123000 | `3a55097f9efd407b7e7f780c1860d7514ec7bd86` | ok | `253697a98e6e7224…` | 1 |
| seed1 | 133000 | `2c44de039f189452fc0a3c90a17916b223c1b1b4` | ok | `2902845ff7274e45…` | 1 |
| seed1 | 143000 | `a77e7d87906f1c77138c009121f9d6439144663c` | ok | `b40bd3670e128927…` | 1 |
| seed2 | 123000 | `319875fb868a5cd6636423c0b5ff95f786c5cd18` | ok | `6d927033194c4ad1…` | 1 |
| seed2 | 133000 | `59534882c2c34c7d7b8a41bb9e251164477de8a1` | ok | `70e7c29a8e70e570…` | 2 |
| seed2 | 143000 | `8f8a4fabfcd5f2504ee550cb33771760d6cec419` | ok | `6d4a8361e3f4160c…` | 1 |
| seed3 | 123000 | `1cb2ac3dd2b729f33a208c0c9ed4da16ebd066b8` | ok | `64e553fef8139197…` | 1 |
| seed3 | 133000 | `a2a1327db3a383c0f77c4b53ea483ddd4c6d4ae5` | ok | `3a1a89ce3cd72e78…` | 1 |
| seed3 | 143000 | `83438f87adb388fb205a83ad516e3b349b5cfcba` | ok | `77e2110771b16b10…` | 1 |
| seed4 | 123000 | `b884f27285abb230c4503db0a0c7f0a108651fae` | ok | `aa916580029e8b15…` | 2 |
| seed4 | 133000 | `f1dee9051011ae3b7bd191a82db07bb8c71cb702` | ok | `39234f222d7665ec…` | 1 |
| seed4 | 143000 | `b122bdb9065e19909d34653da26e06e49fd4ded2` | ok | `ee1cf6382571265d…` | 1 |
| seed5 | 123000 | `e48b7a7fe656ffc0886b83de9af62a877ccb68bc` | ok | `13c77f62804f1a36…` | 1 |
| seed5 | 133000 | `8b6f55ae443381e65968483290be8d0c2557db43` | ok | `5e3b4d8c8dead4a2…` | 1 |
| seed5 | 143000 | `b9156c2847b0e50f298706af140656bbf0b7cf06` | ok | `8714fb7a9c08688b…` | 1 |
| seed6 | 123000 | `ce6bdbca13fe93cb6b142721a0801fb7b9ae8611` | ok | `53297801cb15129e…` | 1 |
| seed6 | 133000 | `eba0d2f483d7163dfeac5e89e258af476a1afee2` | ok | `047a817aee5f38d2…` | 1 |
| seed6 | 143000 | `36a33e1487d89429085aec228a9ccfe831271243` | ok | `0fd62a1f8f5e2f0e…` | 1 |
| seed7 | 123000 | `a12420095e08c46806901446f76a01927973fa70` | ok | `6ceb9be3d2d2e387…` | 1 |
| seed7 | 133000 | `fbad49e4aced1398fab8d94b531c63cbfa60c70a` | ok | `a8a7928f37cb1287…` | 1 |
| seed7 | 143000 | `4c83aca7bf06be6061797566759d4cf370307b7b` | ok | `756fcba3b987f0d2…` | 1 |
| seed8 | 123000 | `52449162281639c3cf5000d31ac5959fa4ec32a8` | ok | `123d2a3c697a1fb1…` | 1 |
| seed8 | 133000 | `453deb67190a41b3f4a65080df130480fb62b5a4` | ok | `e804e5d63dfcde10…` | 1 |
| seed8 | 143000 | `b3d0ef9e3205c5f235e44c7b8e8791ce04301452` | failed | — | 2 |
| seed9 | 123000 | `1933c5b650555211652330d6864ecf50ae7a8bc5` | ok | `bd6c2c676672d521…` | 2 |
| seed9 | 133000 | `582bc5e55e7535ebb395c0fccefe7c87aa16db88` | ok | `b912b7ff3aa3e584…` | 1 |
| seed9 | 143000 | `6155dcb7e1a9080a5cb61d0ef1f8041e91b5cc7a` | ok | `9b146525f3af954e…` | 1 |

## 3. Eligibility and exclusions

All decisions come from the frozen rules (AMENDMENT3.md §4):
- **Inclusion:** IOI accuracy on the 1,000 matching prompts at step 143000; one-sided binomial
  test against 0.5, α = 0.05.
- **Matching (star design, against the reference):** exact one-sided 95 % Clopper–Pearson bound
  on the discordance rate < 0.02.
- **Completeness:** all three checkpoints loaded, verified and measured; at most one identical
  retry per checkpoint.

| Run | IOI accuracy | Matching statistic | Result | Frozen rule responsible |
|---|---|---|---|---|
| reference | 999/1000 | — | **primary** | §4.1–4.2 complete + included (reference) |
| seed1 | 991/1000 | 8 disc., bound 0.0144 | **primary** | §4.1–4.3 complete, included, matched |
| seed2 | 995/1000 | 6 disc., bound 0.0118 | **primary** | §4.1–4.3 complete, included, matched |
| seed3 | 997/1000 | 4 disc., bound 0.0091 | **primary** | §4.1–4.3 complete, included, matched |
| seed4 | 473/1000 | 528 disc., bound 0.5544 | excluded | §4.2 fails inclusion (p = 0.96) |
| seed5 | 1000/1000 | 1 disc., bound 0.0047 | **primary** | §4.1–4.3 complete, included, matched |
| seed6 | 1000/1000 | 1 disc., bound 0.0047 | **primary** | §4.1–4.3 complete, included, matched |
| seed7 | 983/1000 | 16 disc., bound 0.0242 | excluded | §4.3 not matched (bound ≥ 0.02) |
| seed8 | — | — | excluded | §4.1 incomplete (step 143000 download failed on both permitted attempts) |
| seed9 | 994/1000 | 5 disc., bound 0.0105 | **primary** | §4.1–4.3 complete, included, matched |

**Primary run set (R = 7):** reference, seed1, seed2, seed3, seed5, seed6, seed9.

- **No dependence on D\*.**
  - Eligibility uses only checkpoint completeness and per-prompt correctness on the matching
    set. It never reads attribution tensors or distances.
  - In the runner it is computed (`eligibility`, line 533) **before** any D\* is calculated
    (`pairwise`, line 535).
  - An independent re-implementation of the binomial and Clopper–Pearson rules from the cached
    correctness vectors reproduces every decision and the primary list exactly.
- **seed4:** fails inclusion (473/1000, p = 0.96), consistent with its documented broad
  failure (PolyPythias loss spikes; pilot diagnostics). Retained in every table and in S1.
- **seed7:** complete and included (983/1000) but not matched. Its 16 disagreements give a
  bound of 0.0242 ≥ 0.02.
- **seed8:** steps 123000 and 133000 loaded and verified. Step 143000 failed on both permitted
  attempts with a network error (`ChunkedEncodingError: Connection broken: IncompleteRead`).
  - Under §4.1 an incomplete run is excluded and never replaced, so seed8 is excluded.
  - It was not recovered, and the primary analysis was not recomputed with it.
  - The exclusion reflects a download failure, not a property of the model.

## 4. Primary estimand

Per-pair D\* point estimates (descriptive only; no pair-level confidence intervals, Gate 1).
D\* is bounded in [0, 4] for unit-normalised tensors.

Within-lineage pairs (7 runs × 3 = 21 pairs):

| Run | 123k–133k | 123k–143k | 133k–143k |
|---|---|---|---|
| reference | 0.0171 | 0.0270 | 0.0250 |
| seed1 | 0.0375 | 0.0456 | 0.0346 |
| seed2 | 0.0535 | 0.0510 | 0.0544 |
| seed3 | 0.0183 | 0.0209 | 0.0168 |
| seed5 | 0.0445 | 0.0538 | 0.0327 |
| seed6 | 0.0967 | 0.2221 | 0.2209 |
| seed9 | 0.0669 | 0.1041 | 0.1123 |

Cross-run pairs at step 143000 (21 distinct pairs; self-pairs excluded):

| | reference | seed1 | seed2 | seed3 | seed5 | seed6 | seed9 |
|---|---|---|---|---|---|---|---|
| reference | — | 1.8199 | 1.5938 | 1.7160 | 1.6749 | 1.6693 | 1.4885 |
| seed1 | 1.8199 | — | 1.6644 | 1.8934 | 0.5365 | 1.6940 | 1.3744 |
| seed2 | 1.5938 | 1.6644 | — | 1.7463 | 1.5314 | 1.6136 | 1.2564 |
| seed3 | 1.7160 | 1.8934 | 1.7463 | — | 1.9350 | 1.4222 | 1.8004 |
| seed5 | 1.6749 | 0.5365 | 1.5314 | 1.9350 | — | 1.5184 | 1.3201 |
| seed6 | 1.6693 | 1.6940 | 1.6136 | 1.4222 | 1.5184 | — | 1.4958 |
| seed9 | 1.4885 | 1.3744 | 1.2564 | 1.8004 | 1.3201 | 1.4958 | — |

Descriptive summaries (not pre-registered endpoints):

| Pair set | min | median | max | mean |
|---|---|---|---|---|
| Within-lineage (21 pairs) | 0.0168 | 0.0456 | 0.2221 | 0.0646 |
| Cross-run (21 pairs) | 0.5365 | 1.6136 | 1.9350 | 1.5602 |

The smallest cross-run value (seed1 vs seed5, 0.5365) exceeds the largest within-lineage value
(0.2221).

## 5. Primary H1 result

| Quantity | Value |
|---|---|
| R (primary runs) | 7 |
| B | 200 |
| Valid bootstrap replicates | 200 of 200 (none dropped) |
| Δ | 1.4956684432868828 |
| One-sided 95 % lower bound | 1.269157942978685 |
| Decision rule | SUPPORTED_AT_RUN_LEVEL iff lower bound > 0 |
| **H1 status** | **SUPPORTED_AT_RUN_LEVEL** |

**Independent reconstruction.**
- Recomputing D\* from the stored attribution tensors, the prompt resamples from seed
  20260929 and `lineage.two_way_bootstrap(seed=20260930)` reproduced Δ, the lower bound and
  n_valid **exactly**, and lower bound > 0.
- Δ equals mean(cross) − mean(within) computed by hand.
- No bootstrap replicate was selected or discarded manually.

**Calibration note: R = 7 was not a separately simulated point.**
- The run-level procedure was validated by synthetic type-I simulation at R = 6 and R = 10
  (Gate 5) and at R = 4 and R = 5 (the pre-declared small-R extension, gate criteria notes 5
  and 6).
- R = 7 lies inside that range, but calibration at R = 7 itself is assumed, not separately
  simulated. The record carries this warning.
- This is a disclosed limitation, not a failure. The analysis was not rerun to obtain R = 7
  calibration.

## 6. Sensitivity analyses

**DESCRIPTIVE / SENSITIVITY ANALYSES — NOT USED TO DECIDE H1.** All were pre-specified
(AMENDMENT3.md §9), computed by the locked runner in the same execution, and reproduced exactly
from the stored data.

| Analysis | Runs | Δ | Lower bound (descriptive) |
|---|---|---|---|
| S1: all complete runs (no inclusion/matching; includes seed4 and seed7) | 9 | 1.3378 | 1.0286 |
| S2: primary set without seed3 | 6 | 1.4112 | 1.1682 |
| S3: cross-fitted D̂_cf (20 splits, seed 20260931) | 7 | 1.5009 | point estimate only |

- **Jackknife (pre-declared diagnostic only; never determines H1):** leave-one-run-out
  SE 0.1135, t = 13.18 on 6 df.
- **Not used for selection:** none of S1–S3 or the jackknife was used to select or replace
  the primary result. The primary H1 decision is the bootstrap result in §5 alone.

## 7. Integrity and leakage audit

**Integrity** (all verified 2026-09-30):

| Item | Value | Check |
|---|---|---|
| Commit that produced the record | `116035c575a770700432241bb7acc23c65876b34` | recorded in `git` field |
| Lock tag | `v6-amendment3-lock` → `116035c` | `lock_tag_at_head: true` |
| Tree at execution | clean (`dirty: false`) | recorded |
| Protocol hash | `b3d55e106aaf584c3a2dda3b9ced43607f96b76aa98abf4339ba7ca56c5bd706` | equals `EXPECTED_PROTOCOL_SHA256`; the protocol stored in the record equals the locked code protocol |
| Runner hash | `477953a430ba3e2aa758d3fd8baaccebe8f8f81cd1e6c4b7ab066de58faf3d5b` | equals the runner file at the lock |
| Record sha256 | `18d11300d599e75cad29a3aa5e4584b7280adce71657832fcd6032da6114e3d9` | `record.sha256` valid |
| Prompt-set hashes | as §2 | recomputed from the files |
| Model commits and weights | 29/30 checkpoints: pinned = resolved; local sha256 = Hub sha256 | the 30th (seed8@143000) failed to download |
| Execution window | 2026-09-29T21:03:58Z – 2026-09-30T01:33:56Z | recorded |

**The second execution attempt did not overwrite anything.**
- It stopped with `ProtocolError: … record.json exists; records are never overwritten`.
- That check runs before any file is written.
- `record.json`, `record.sha256` and the cache carry the first run's timestamps (03:33 local),
  and the record hash is unchanged.

**Leakage audit.** Result: **NO EVIDENCE OF POST-HOC SELECTION.** What was actually checked:

1. **Code and protocol after the lock.**
   - `git diff v6-amendment3-lock` shows no change to `AMENDMENT3.md`, `PREREGISTRATION.md`,
     the runner, the prompt files, `glassbox/` or `tests/`.
   - The only commit after the lock (`95c7bd2`) adds the run output.
2. **What the record used.**
   - The protocol hash and runner hash equal the locked values.
   - The runner exposes no option for the run set, checkpoints, prompts, thresholds, metric,
     estimator, bootstrap or inference method, and reads no environment variables.
3. **Exclusions.**
   - Reproduced independently from raw correctness data, with no use of D\*.
   - Computed before D\* in the code path.
   - seed8's exclusion follows the pre-written incompleteness rule, and it was not replaced or
     re-added.
4. **One execution only.**
   - `runs/` contains a single confirmatory output directory.
   - All 29 cache entries are keyed to the locked protocol and runner hashes.
   - The owner's terminal log shows one completed execution and one refused re-execution.
5. **Bootstrap.**
   - The stored result equals an independent recomputation with the pinned seeds.
   - All 200 replicates are valid; none was removed.
6. **Sensitivity and method.**
   - S1–S3 and the jackknife are exactly the pre-declared set, and all were produced in the
     same execution.
   - No alternative analysis was run on confirmatory data.

**Limits of this audit.** It cannot exclude executions performed outside this repository or
machine and not reported. It relies on the git history, the record, the cache and the owner's
terminal log.

## 8. Reproducibility

- **Raw data preserved.** `experiments/v6/runs/confirmatory_v1/_cache/` holds 29 `.json` +
  29 `.npz` files, 17.9 MB for the whole run directory:
  - per checkpoint, the attribution tensor (float64, [200, 24, 16]) and, at step 143000, the
    correctness vector (bool, [1000]);
  - no model weights, credentials or temporary files (checked).
- **Re-derivable results.** From these files, every eligibility decision, D\* value, Δ, the
  lower bound, S1–S3 and the jackknife reproduce bitwise, with no model downloads.
- **Full re-execution.**
  - Check out tag `v6-amendment3-lock` and run
    `python experiments/v6/confirmatory.py --out <new dir> --free-disk`. It requires about 29 GB
    of downloads.
  - Attribution is deterministic: bitwise identical on repeat measurement and across loading
    paths in pipeline validation.

## 9. Limitations

1. Gate 1 failed. Per-pair D\* values are point estimates only; no pair-level confidence
   intervals exist.
2. R = 7 was not itself a separately simulated calibration point. R = 4, 5, 6 and 10 were
   validated.
3. seed8 was excluded because its step-143000 download failed, under the frozen rule. This is
   not a model property, and it reduces the number of runs.
4. The within-lineage comparison conflates training time with lineage: within-run pairs
   differ in step, cross-run pairs do not.
5. Initialisation and data-order effects are coupled in the PolyPythias seed design.
6. One model size: Pythia-410M.
7. One task: IOI (templated prompts from one generator).
8. One attribution procedure: last-position, first-order Taylor, name-swap corruption.
9. D\* is not established to be importance-weighted. On the 410M controlled perturbation,
   scaling the attribution profile produced a monotonic increase in D\*, while perturbing ten
   low-attribution heads produced a substantially smaller change. This supports sensitivity to
   the measured attribution profile, but does not establish that D\* is importance-weighted.
10. Attribution-profile divergence is not equivalent to mechanistic or circuit divergence.
11. Generalisation to other architectures, tasks, scales and attribution methods is untested.
12. TransformerLens reads the base architecture config from the model's default branch. This
    was guarded only indirectly, by the vocabulary hash and attribution-shape checks.

## 10. Claim boundary

**Primary result wording.** "Under the preregistered estimand, independently trained,
performance-matched Pythia-410M runs show greater attribution-profile distance (last-position
Taylor attribution on IOI, modulo within-layer head permutation) than late checkpoints within
the same training lineage (Δ = 1.50, run-level one-sided 95% lower bound 1.27, 7 runs)."

This result is evidence for attribution-profile divergence under the preregistered estimand.
It is NOT evidence, by itself, for:
- mechanistic divergence;
- different causal mechanisms;
- different circuits;
- causal effects of training;
- causal explanations of model behavior.

The experiment establishes an empirical divergence result. It does not identify the causal
source of that divergence.

## 11. Conclusion

**H1 is SUPPORTED_AT_RUN_LEVEL under the preregistered estimand.**

The strongest currently justified claim: performance-matched, independently trained
Pythia-410M runs exhibited substantially greater attribution-profile divergence than late
checkpoints within the same training lineage, under the preregistered D\* estimand. This is not
upgraded to a mechanistic or causal claim.
