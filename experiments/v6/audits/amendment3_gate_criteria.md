# Amendment 3 gates: pre-declared criteria

*Written 2026-09-29, **before** any gate simulation or real-model gate run. These criteria are
not edited after results are seen; changes are appended as dated notes with reasons. The
existing positional D_M results and the earlier sorted-head results are not inputs to any
decision below.*

**Estimand (approved).** D\*(A, B) = min over within-layer head permutations π of
‖f̃_A − π·f̃_B‖² over the IOI prompt distribution P, under attribution procedure 𝒜 (last
position, Taylor, name-swap corruption). This is a **distance between attribution profiles,
modulo within-layer head permutation**. It is never interpreted as a distance between
mechanisms or circuits.

## Gate 1: prompt-level interval validity

- **Candidates.**
  - (i) Percentile bootstrap.
  - (ii) Basic (bias-reflected) bootstrap.
  - (iii) Bracketed: [q_{2.5%}(boot), D̂_cf + (q_{97.5%}(boot) − D̂)].
  - All use a paired prompt bootstrap (prompts resampled jointly for both models).
  - D̂_cf is the cross-fitted orbit distance (10 splits).
- **Conditions.** 3 regimes × 3 cases = 9 conditions.
  - Regimes:
    - dense;
    - sparse (60 % of heads scaled by 0.01);
    - unstable (heads come in near-duplicate twin pairs within each layer, so the optimal
      matching is non-unique up to small perturbations).
  - Cases:
    - similar: same structure, arbitrary relabelling;
    - perturbed: same structure plus a small perturbation;
    - different: independent structure.
- **Settings.**
  - Generator: `identifiability_synthetic.py`. Defaults unchanged; the twin option is added
    for the unstable regime.
  - L = 12, H = 16, N = 200 prompts.
  - D\* approximated with N_POP = 20,000 fresh prompts.
  - B = 200 bootstrap draws; R = 200 replicates per condition.
- **Monte Carlo SE** at nominal 0.95: √(0.95·0.05/200) = 0.0154.
- **Pass (per procedure).** Empirical coverage ≥ 0.95 − 2·0.0154 = **0.919** in all 9
  conditions.
- **Selection.** Among passing procedures, the one with the smallest maximum mean width
  across conditions. If none passes, **Gate 1 fails** and no per-pair interval is reported
  as a CI.

## Gate 3: synthetic discrimination (same simulation family, R = 200 per regime)

On the plug-in orbit distance, per regime (dense, sparse, unstable):

1. **A (same structure + relabelling) ≈ F (same structure, no relabelling).**
   |mean(A − F)| ≤ 3 × SE of the paired difference.
2. **B (perturbation) > A** in ≥ 95 % of replicates.
3. **C (independent structure) > B** in ≥ 95 % of replicates.
4. **D (C relabelled) = C:** |D − C| ≤ 1e-9 in every replicate.
5. **E (same magnitude multiset, new per-prompt behaviour) > B** in ≥ 95 % of replicates.

## Gate 2: invariance under genuine function-preserving head permutations

- On a real model: pythia-70m@143000 here; pythia-410m-deduped@143000 on the Mac.
- Permute head parameter slices (W_Q, W_K, W_V, b_Q, b_K, b_V, W_O) with a random π.
- Recompute per-prompt attribution tensors (same prompts).
- **Pass.**
  - D̂(X, X_π) ≤ **1e-4**;
  - |D̂(X_π, Y) − D̂(X, Y)| ≤ **1e-4**, with Y a different checkpoint.
  - Tolerance rationale: float32 summation-order effects produced relative attribution
    deviations of about 1e-3 on pythia-70m; the squared distance of unit tensors scales as
    their square. 1e-4 is 0.0025 % of the metric's range [0, 4].

## Gate 5: run-level inference for Δ

- **Estimand.**
  Δ = E_{r≠r′} D\*(r@143k, r′@143k) − E_r mean_{s<t∈S} D\*(r@s, r@t),
  with S = {123000, 133000, 143000}.
- **Estimator.** Δ̂ with the plug-in D̂.
  - Rationale: the plug-in's downward bias grows with the true distance, so it can only
    shrink a positive Δ (conservative for H1).
  - The cross-fitted Δ̂_cf is reported as a sensitivity analysis only.
- **Unit.** The training run.
  - All pair distances involving a run are dependent, and all pairs share the same prompts.
- **Primary procedure.** Two-way bootstrap:
  - each replicate resamples **runs** with replacement (pairs of a run with itself are
    dropped) **and** prompts with replacement, jointly for all models;
  - recompute Δ̂*;
  - the one-sided 95 % lower bound is the 5th percentile;
  - reject H0 (Δ ≤ 0) iff the lower bound > 0.
- **Fallback, used only if the primary fails validity.** Leave-one-run-out jackknife SE with a
  one-sided t_{R−1} test.
- **Validity criterion.** Simulated type-I error at the H0 boundary (Δ = 0 exactly by
  construction) ≤ 0.05 + 2·0.0154 = **0.081**, at R = 6 and R = 10 runs, with ≥ 200
  simulated experiments each.
- **Reporting.** Power is reported under H1 worlds but is **not** a selection criterion.

## Gate 7: controlled perturbation sensitivity (real model; not mechanism validation)

- **Heads chosen on held-out data.** Target heads = top-10 by mean |attribution| in
  pythia-410m-deduped@123000 on held-out prompts (IOI seed 1).
- **Intervention.** Scale those heads' W_O by c ∈ {1, 0.75, 0.5, 0.25, 0} in
  pythia-410m-deduped@143000.
- **Evaluation.** D̂(m, m_c) on the evaluation prompts (IOI seed 0, N = 100).
- **Pass.**
  - D̂(m, m_1) ≤ 1e-12;
  - D̂ strictly increases as c decreases (Spearman(c, D̂) = −1 over the 5 doses).
- **Reported, not gated.** The same doses applied to 10 low-attribution heads (ranks
  > 100 in the held-out model).
- Neither model is in the confirmatory run set.

## Gate 4: within-lineage reference pairs

- **Exact pairs, per included run:** (123000, 133000), (123000, 143000), (133000, 143000).
- They are described as **within-training-lineage variation**.
- Branch existence is verified on the Hub before the lock.

## Gate 6: seed 4

- The complete dataset retains seed4.
- The primary analysis applies the pre-specified inclusion rule (IOI accuracy above chance,
  one-sided binomial α = 0.05) and the Amendment 2 matching rule.
- Seed4 enters every descriptive table and a pre-specified sensitivity analysis. Its broad
  failure is reported:
  - 84/200 IOI;
  - name mass 0.047;
  - sentence NLL 3.86 vs 3.44–3.66 for the other runs;
  - documented loss spikes (PolyPythias).

## Gate 8: terminology

- "Attribution-profile divergence" is used throughout V6 code comments, records and
  documents.
- "Mechanistic divergence" appears nowhere except in quoted history or negations.

## Gate 9: freeze

Amendment 3 is locked (dated commit plus tag) only after Gates 1–8 pass. Only then is the
real confirmatory experiment run.

---

## Dated note 1 (2026-09-29, after the Gate 1 run; criteria above unchanged)

**Gate 1 result: FAIL.** No procedure reached ≥ 0.919 coverage in all 9 conditions.

- The bracketed interval failed only sparse/similar (0.915).
- A diagnosis on the first 120 replicates of that cell found every miss on the **upper**
  side: D\* exceeded D̂_cf by 1.0–2.5 × the sampling half-width.
- Cause: the upper margin (q_{97.5%} − D̂) is the spread of the *in-sample* estimator. The
  cross-fitted estimator uses half-samples and is more variable, so the margin understates
  its uncertainty.

**One revision is allowed, and only one.** It is declared before being run:

- **Bracketed-v2** = [q_{2.5%}(bootstrap of D̂), q_{97.5%}(bootstrap of D̂_cf)].
  - Each bootstrap replicate recomputes the cross-fitted estimator (K = 10 splits) on the
    resampled prompts.
- Same 9 conditions and same R = 200, **with fresh seeds** (seed offset 10,000) so that no
  replicate used for the failing run is reused.
- Same pass threshold of 0.919 in all 9 conditions.
- **If v2 fails,** Gate 1 is closed as failed. Per-pair intervals are then **not** reported
  as confidence intervals: only point estimates, labelled as such. Confirmatory inference
  relies solely on the run-level procedure (Gate 5), and the lock proceeds only if the
  pre-registration states this explicitly.

## Dated note 2 (2026-09-29, before any full Gate 5 run)

The H1 power worlds were redefined as λ ∈ {0.15, 0.3} (plus an independent-structure
world). A 2-experiment smoke test showed that the original λ = 0.5 gives Δ ≈ 0.6, which is
uninformative about power. Power is reported only, never used to select a procedure. The
H0 validity criterion and the H0 worlds are unchanged.

## Dated note 3 (2026-09-29, before the Gate 2 run on the pre-declared 410M model)

- **What happened.** A sandbox *validation* run of the Gate 2 script on pythia-70m (not the
  pre-declared gate model) gave:
  - D(X, X_π) = 7.2e-6, within tolerance;
  - |D(X_π, Y) − D(X, Y)| = 1.4e-4, above the pre-declared 1e-4.
- **Why the second condition was ill-posed.**
  - √D is a metric on orbits, so |D(X_π,Y) − D(X,Y)| ≤ √D(X,X_π)·(√D(X,Y) + √D(X_π,Y)).
  - Float noise in the self-distance therefore propagates, scaled by √D(X,Y). A fixed
    absolute tolerance is the wrong test.
- **Corrected Gate 2** (applies to the 410M gate run, which has not been run):
  1. D(X, X_π) ≤ 1e-4. Unchanged; this is the invariance test.
  2. |D(X_π,Y) − D(X,Y)| ≤ √D(X,X_π)·(√D(X,Y) + √D(X_π,Y)) + 1e-12. This checks that the
     implementation respects the metric bound.
- **Also reported** (not gated): the Spearman correlation between π·X and X_π.

## Dated note 4 (2026-09-29): gate results from the Mac runs

- **Gate 1 revision (bracketed-v2; fresh seeds, offset 10,000): FAIL.**
  - sparse/similar 0.915 and sparse/perturbed 0.915 (17 misses each, all above the upper
    bound), against the 0.919 threshold.
  - All other cells 0.95–1.00.
  - Per dated note 1, **Gate 1 is closed as failed. No further revision will be tried.**
  - Consequence: per-pair intervals are reported only as point estimates, never as
    confidence intervals, and confirmatory inference relies solely on the run-level
    procedure.
- **Gate 2: PASS** (pythia-410m-deduped@143000 vs @133000, 50 prompts).
  - D(X, X_π) = 1.7e-7.
  - |D(X_π,Y) − D(X,Y)| = 1.2e-5 ≤ triangle bound 2.7e-4.
  - Equivariance Spearman 0.99999.
- **Gate 5: PASS** for the primary two-way bootstrap. Type-I error at the H0 boundary:

  | Regime | R = 6 | R = 10 |
  |---|---|---|
  | sparse | 0.055 | 0.050 |
  | dense | 0.045 | 0.065 |

  - All are ≤ 0.081. The jackknife gave 0.035–0.050.
  - Power was 1.0 in every H1 world, including λ = 0.15 (mean Δ ≈ 0.037).
- **Gate 7: PASS** (pythia-410m-deduped@143000; heads chosen in @123000 on held-out
  prompts).
  - D = 6e-17 at c = 1, then 0.028, 0.155, 0.521 and 1.161 at c = 0.75, 0.5, 0.25 and 0.
    Strictly increasing.
  - Low-attribution heads (report only): 0.0019, 0.0079, 0.0196, 0.0385.
- **Earlier results:** Gates 3, 4 and 8 PASS. Gate 6 is specified; the star-design choice
  awaits approval.
- **Lock status.** The approved rule ("lock only after Gates 1–8 pass") is not met, because
  Gate 1 failed. Locking requires an explicit owner decision to accept the pre-declared
  Gate 1 fallback. Until then Amendment 3 remains unlocked.

## Dated note 5 (2026-09-29): R = 4 / R = 5 run-count validation, written BEFORE running

- **Purpose.** Only to establish whether the already validated run-level procedure (Gate 5)
  remains calibrated when fewer than 6 runs are eligible. Nothing about the estimator, the
  bootstrap, D\* or H1 may change as a result.
- **Design.** Identical to Gate 5 (`gate5_simulation.py`), with only the run count changed:
  - **World:** H0 boundary only. Every checkpoint of every run is an independent
    `perturb()` of one shared structure S0, and each run has its own random within-layer
    relabelling. So Δ = 0 exactly.
  - **Regimes:** sparse and dense.
  - **Size:** 3 checkpoints per run, N = 200 prompts, L = 12, H = 16, generator
    `identifiability_synthetic.py` (unchanged).
  - **R** ∈ {4, 5}.
- **Estimator.** Plug-in `profile_orbit_distance`. Within-run pairs (0,1), (0,2), (1,2);
  cross-run pairs at the last checkpoint. Δ̂ = `lineage.delta`.
- **Inference.** `lineage.two_way_bootstrap`, B = 200. Prompts are resampled by one multinomial
  draw shared by all pairs; runs are resampled with replacement; self-pairs are dropped;
  replicates with fewer than 2 distinct runs are NaN and excluded (validated behaviour). The
  one-sided lower bound is the 5th percentile; reject iff > 0.
- **Seeds (same formula as Gate 5).**
  - Experiment `sim` uses `np.random.default_rng([sim, R, sum(map(ord, "H0" + regime))])`.
  - `two_way_bootstrap(seed=sim)`.
  - sim = 0 … 199.
- **Replicates.** 200 simulated experiments per cell. 4 cells: {sparse, dense} × {4, 5}.
- **Criterion (inherited unchanged from Gate 5).** Type-I error = rejection rate at H0
  ≤ 0.05 + 2·√(0.95·0.05/200) = **0.0808**.
- **Pass rule.**
  - R = 4 passes iff both regimes pass. R = 5 passes iff both regimes pass.
  - If both R = 4 and R = 5 pass, the locked minimum number of eligible runs becomes
    **R_min = 4**.
  - If either fails, **R_min stays 6**, the failure is reported, and the owner decides. No
    repair or tuning.
- **Reported, not gated.** Jackknife rejection rate; the number of NaN bootstrap replicates.
- **Assumption, stated.** Run counts 7–9 are not simulated. Calibration between the validated
  R = 6 and R = 10 is assumed, not shown.

## Dated note 6 (2026-09-29): R = 4 / R = 5 results (criterion from note 5, unchanged)

200 experiments per cell. Raw per-experiment log: `gate5_small_r.jsonl`; summary:
`gate5_small_r_results.json`.

| Cell | Bootstrap type-I (MC SE) | Jackknife (report only) | Pass (≤ 0.0808) |
|---|---|---|---|
| H0 sparse, R = 4 | 0.065 (0.017) | 0.050 | yes |
| H0 dense, R = 4 | 0.030 (0.012) | 0.010 | yes |
| H0 sparse, R = 5 | 0.045 (0.015) | 0.030 | yes |
| H0 dense, R = 5 | 0.025 (0.011) | 0.015 | yes |

- **Result.** R = 4 and R = 5 both pass, so by the pre-declared rule **R_min = 4**.
- **Omission.** Note 5 listed the number of NaN bootstrap replicates as "reported". The reused
  Gate 5 `_experiment` does not return it, and that code was deliberately not modified, so
  it is not reported.
- **Assumption still in force.** R = 7–9 remain assumed, not simulated.
