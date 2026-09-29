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
