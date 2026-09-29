# V6 Amendment 3: confirmatory protocol (FINAL TEXT, pending lock)

*Final text 2026-09-29. It becomes locked with the commit tagged `v6-amendment3-lock`,
created only after the final re-audit and the owner's explicit approval. It supersedes
`AMENDMENT3_DRAFT.md` (kept in git history) and, where they conflict,
`PREREGISTRATION.md` (§10 there). Gate criteria and results:
`audits/amendment3_gate_criteria.md` (notes 1–6). H1–H4 are **UNRESOLVED** until confirmatory
data exist.*

## 1. Estimand and terminology

- **D\*(A, B) = min_{π∈G} ‖f̃_A − π·f̃_B‖².**
  - G: within-layer head permutations.
  - f̃_m: the unit-Frobenius-normalised per-prompt attribution tensor of model m, over the
    locked attribution prompt set (§5), under procedure 𝒜 (§6).
- **Name.** D\* is a **distance between attribution profiles, modulo within-layer head
  permutation**. Differences are **attribution-profile divergence**.
- **Interpretation limit.** D\* is never interpreted as a distance between mechanisms or
  circuits.
- **Implementation.** `glassbox.v6.identifiability.profile_orbit_distance` (plug-in
  estimator). The positional D_M (`operational_mechanistic_distance`) is **not used**.

## 2. The only confirmatory hypothesis

- **H1 (run level).** Among performance-matched Pythia-410M runs, attribution-profile
  distance at step 143000 is greater between independently trained runs than between late
  checkpoints within the same training lineage.
  - Δ = mean_{r<r′} D\*(r@143000, r′@143000) − mean_r mean_{(s,t)} D\*(r@s, r@t),
    with (s, t) ∈ {(123000,133000), (123000,143000), (133000,143000)}.
  - Implementation: `glassbox.v6.lineage.delta`.
  - H0: Δ ≤ 0. H1: Δ > 0.
  - The within term is **within-training-lineage variation**, not "same mechanism".
- **H2–H4** are **not tested** in this experiment and remain UNRESOLVED. Their earlier
  definitions use the superseded positional D_M and are not confirmatory endpoints here.
- **Multiplicity.** There is a single confirmatory test, so there is no multiplicity
  correction. The sensitivity analyses (§9) are descriptive and never change the H1 decision.

## 3. Runs and checkpoints

- **Runs:** 10, in this fixed order: `EleutherAI/pythia-410m` (the **reference**), then
  `pythia-410m-seed1` … `seed9`.
- **Checkpoints per run:** 123000, 133000 and 143000.
- **Pinning.** Every checkpoint is loaded **by its full commit SHA**, never by branch name.
  The full SHAs are in `experiments/v6/confirmatory.py` (`PROTOCOL`). First 10 characters:

  | Run | 123000 | 133000 | 143000 |
  |---|---|---|---|
  | pythia-410m | 51cabaff46 | 45c291dfb8 | bba6a464f5 |
  | seed1 | 3a55097f9e | 2c44de039f | a77e7d8790 |
  | seed2 | 319875fb86 | 59534882c2 | 8f8a4fabfc |
  | seed3 | 1cb2ac3dd2 | a2a1327db3 | 83438f87ad |
  | seed4 | b884f27285 | f1dee90510 | b122bdb906 |
  | seed5 | e48b7a7fe6 | 8b6f55ae44 | b9156c2847 |
  | seed6 | ce6bdbca13 | eba0d2f483 | 36a33e1487 |
  | seed7 | a12420095e | fbad49e4ac | 4c83aca7bf |
  | seed8 | 5244916228 | 453deb6719 | b3d0ef9e32 |
  | seed9 | 1933c5b650 | 582bc5e55e | 6155dcb7e1 |

- **Verification at load.** The resolved commit must equal the pinned SHA, and the local
  weights file's sha256 must equal the Hub's LFS sha256 for that commit. **Any mismatch stops
  the run.**
- **File format.** Seed repositories are loaded with `use_safetensors=False`, i.e. their
  official `.bin` files, never a conversion pull request.

## 4. Eligibility, the primary run set, and failures (all final)

1. **Completeness.** A run is complete iff all three of its checkpoints load, pass
   verification (§3) and are measured without error.
   - An incomplete run is **excluded**, with the checkpoint and error recorded.
   - No retry may change the model, revision, prompts or settings.
   - A retry after a transient failure is allowed only for the identical pinned
     configuration, and it is recorded.
2. **Inclusion.** At step 143000, IOI accuracy on the locked **matching set** (1,000 prompts)
   is above chance: one-sided exact binomial test against 0.5, p < 0.05. Correct means
   logit(target) > logit(distractor) at the last position.
3. **Matching (star design).** A run is matched iff `exact_paired_equivalence` against the
   reference's per-prompt correctness on the matching set (step 143000) gives an exact
   one-sided 95 % Clopper–Pearson discordance bound < 0.02.
4. **Primary run set.** The reference plus every other run that is complete, included and
   matched, in the §3 order.
   - Deterministic. No other subsets are searched.
   - An excluded run is **never replaced** by another run.
5. **H1 is UNTESTED and remains UNRESOLVED if:**
   - the reference is incomplete or fails inclusion; or
   - the primary set has fewer than **R_min = 4** runs.
   - R_min = 4 rests on the validated range: R = 4 and 5 (note 6) and R = 6 and 10 (Gate 5).
     R = 7–9 are assumed, not simulated.
6. **Seed 4 and seed 3.**
   - Seed 4 stays in the complete dataset and in every table. Its documented broad failure
     is reported: 84/200 IOI in the pilot, name-probability mass 0.047, sentence NLL 3.86 vs
     3.44–3.66 for the other runs, and loss spikes documented in PolyPythias. Its primary
     status is decided **only** by rules 1–3.
   - Seed 3 (a PolyPythias outlier) is treated the same way.
7. **Recording.** Every exclusion is recorded with the run, the rule and the reason.

## 5. Prompt sets (locked files, never regenerated during execution)

| Set | File | n | sha256 |
|---|---|---|---|
| Attribution | `experiments/v6/prompts/ioi_attribution_v1.json` | 200 | `aa3e32c8ab51d4aabe96e83b51d30a947e120b68bfcbf0da1dff87868abee0d2` |
| Matching | `experiments/v6/prompts/ioi_matching_v1.json` | 1000 | `0849499b8117dac405743c2b33f500545f42aa1f4faff85461feecdef475d296` |

- **Construction** (`prompts/build_prompt_sets.py`; deterministic, model-independent):
  - The attribution set is the first 200 prompts of the IOI generator stream with seed 0.
  - The matching set takes the seed-2 stream in order, skipping prompts already in the
    attribution set (1 skipped) or already taken (2 skipped), until 1,000 prompts are
    collected (1,003 consumed).
- **Invariants.** The sets are disjoint, each is unique, and their order is fixed. The runner
  stops if any count, hash, uniqueness or disjointness check fails.
- This corrects the draft's untrue claim that the sets were disjoint.

## 6. Measurement procedure 𝒜

- **Loading.** Explicit, the same for every run:
  1. the config at the pinned SHA;
  2. exactly the pinned weights file (`model.safetensors` for the original run,
     `pytorch_model.bin` for seeds), loaded with safetensors or `torch.load(weights_only=True)`;
  3. `from_config` plus `load_state_dict` in float32 on CPU. Any missing parameter, or any
     extra key other than the legacy buffers `attention.bias`, `attention.masked_bias` and
     `rotary_emb.inv_freq`, stops the run;
  4. conversion by TransformerLens as `pythia-410m` (`hf_model=`).
  - `from_pretrained` is not used: for `.bin`-only repositories it starts a background
    download of an unmerged conversion PR, which was found during pipeline validation.
  - Verified on pythia-70m@143000: attributions are **bitwise identical** to the
    TransformerLens checkpoint path used in Gates 2 and 7, and a repeated measurement of the
    same checkpoint is bitwise identical.
  - The TransformerLens tokenizer/config come from the base model's default branch. The
    runner stops unless the tokenizer vocabulary sha256 equals `9f23fcef…` (the pinned Pythia
    vocabulary) and all 38 names are single tokens.
- **Attribution.** Per prompt: `GlassboxV2.attribution_patching` (Taylor, three passes), last
  position, bidirectional name-swap corruption, target vs distractor logit difference.
  Result: a [200, 24, 16] tensor per checkpoint.
- **Determinism.** Attribution is deterministic; Control A was bitwise identical in every
  pilot. No randomness enters measurement, and this is recorded.

## 7. Inference (Gate 5 procedure, unchanged)

- **Unit.** The training run.
- **Point estimate.** Δ̂ from plug-in D\*.
- **Primary procedure.** `lineage.two_way_bootstrap` with **B = 200**. B = 200 is **inherited
  from the validated Gate 5 procedure** and is not changed.
  - **Prompts:** one multinomial resample of the 200 attribution prompts per replicate,
    drawn by `numpy.random.default_rng(20260929)` and **shared by every pair distance**. Each
    pair distance is recomputed with `orbit_distance_weighted`.
  - **Runs:** resampled with replacement by `two_way_bootstrap(seed=20260930)`. Self-pairs are
    dropped. Replicates with fewer than two distinct runs are NaN, excluded, and counted.
  - **Decision:** the one-sided lower bound is the 5th percentile of the valid replicates.
    **H1 is supported at the run level iff the lower bound > 0.**
- **Jackknife.** The jackknife is a diagnostic/backup analysis only. It never substitutes
  for the primary bootstrap result and never determines the confirmatory H1 decision.
- **Per-pair values.** Every per-pair D\* is a **descriptive point estimate only**.
  - Gate 1 failed and is closed (notes 1 and 4), with no third attempt, so **no pair-level
    confidence interval is reported or claimed**.
  - Uncertainty is stated only for Δ, at the run level.

## 8. Seeds

| Component | Seed / status |
|---|---|
| Attribution prompts | generator seed 0 (locked file) |
| Matching prompts | generator seed 2 (locked file) |
| Prompt resampling | `default_rng(20260929)` |
| Run resampling | `two_way_bootstrap(seed=20260930)` |
| S3 cross-fit splits | `seed=20260931` |
| Model inference / attribution | deterministic, no RNG |

## 9. Sensitivity analyses (pre-specified, descriptive, never change the H1 decision)

- **S1.** All complete runs, ignoring inclusion and matching, so seed 4 is included if
  complete. Same Δ and bootstrap procedure. The lower bound is labelled descriptive.
- **S2.** The primary set without seed 3, if seed 3 is in it and ≥ 4 runs remain. Same
  procedure, labelled descriptive.
- **S3.** Δ from the cross-fitted D̂_cf (`crossfit_orbit_distance`, 20 splits,
  seed 20260931). Point estimate only.
- **J.** Jackknife on the primary set: diagnostic only (§7).

## 10. No post-hoc changes

> After confirmatory execution begins, the primary estimand, distance definition, prompt
> sets, run set, checkpoint set, inclusion/exclusion rules, thresholds, estimator,
> resampling procedure, and hypothesis definitions cannot be changed based on observed
> confirmatory results.

- **Runner constraints.** The runner exposes no option for any of these.
- **Reporting.** Every result is reported whatever its direction: primary, sensitivity
  analyses, exclusions and failures.

## 11. Interpretation boundaries

- **Controlled perturbation (Gate 7).** "On the 410M controlled perturbation, scaling the
  attribution profile produced a monotonic increase in D\*, while perturbing ten
  low-attribution heads produced a substantially smaller change. This supports sensitivity
  to the measured attribution profile, but does not establish that D\* is
  importance-weighted."
- **Claim boundary.**
  > The confirmatory experiment can provide evidence for attribution-profile divergence
  > under the preregistered estimand. It does not by itself establish mechanistic
  > divergence, different causal mechanisms, different circuits, or a causal explanation of
  > training-induced differences. Such stronger claims require additional intervention or
  > causal evidence.

## 12. Gate record

| Gate | Result |
|---|---|
| 1 Per-pair intervals | **FAILED, closed.** No third attempt. Per-pair D\* is point estimates only (§7). |
| 2 Invariance under weight-level head permutation | PASS |
| 3 Synthetic discrimination | PASS |
| 4 Within-lineage pairs | PASS |
| 5 Run-level inference validity | PASS, R = 6 and 10; extension R = 4 and 5 PASS (note 6) |
| 6 Seed 4 policy | §4.6 |
| 7 Controlled perturbation sensitivity | PASS (wording §11) |
| 8 Terminology | PASS |
| 9 Lock | After the re-audit and the owner's explicit approval |

## 13. Execution (`experiments/v6/confirmatory.py`)

- **Code path.** The runner implements this document only.
  - `PROTOCOL` holds every fixed choice and is checked against `EXPECTED_PROTOCOL_SHA256`,
    together with explicit invariants (metric, estimator, inference, B = 200, α, R_min,
    checkpoints, pairs, matching and inclusion rules, seeds, 40-hex commits).
  - The CLI accepts only `--out`, `--pipeline-validation` and `--free-disk`.
- **Lock guard.** A CONFIRMATORY run refuses to start unless the git working tree is clean
  and HEAD carries the tag `v6-amendment3-lock`.
- **Resume after interruption.** Each verified, successfully measured checkpoint is cached
  under `<out>/_cache`.
  - The cache key is the protocol hash + runner hash + pinned commit.
  - Entries are written atomically; failures are never cached.
  - Reuse is flagged `from_cache` in the record. A resumed run is identical to an
    uninterrupted one (tested).
- **Load retries.** At most one retry of the *identical* pinned configuration. Every attempt
  is recorded.
- **Output.** `record.json` and `record.sha256`, never overwritten.
- **Command** (only after the lock tag and the owner's approval):
  `python experiments/v6/confirmatory.py --out experiments/v6/runs/confirmatory_v1 --free-disk`
- **Residual risks (disclosed).**
  - R = 7–9 calibration is assumed.
  - TransformerLens fetches the architecture config from the base model's default branch
    (guarded only indirectly, by the vocabulary check and attribution-shape checks).
