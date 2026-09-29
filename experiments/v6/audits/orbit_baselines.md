# Baselines for `profile_orbit_distance` (draft for Amendment 3)

*Written 2026-09-29. No real-model confirmatory analysis has been run. `profile_orbit_distance`
has not been applied to any real pilot data, seeds included. The only empirical work here
is a synthetic coverage study (`orbit_baseline_simulation.py` / `.json`) using the generator
fixed in `identifiability_synthetic.py`. H1–H4 remain UNRESOLVED.*

**Disclosure.** From the positional D_M I already know that same-run late checkpoints have
highly corresponding heads (relabelling-null percentile 0.00). That prior knowledge bears on
option B and is flagged there.

---

## 0. The estimand

**Setting.**

- Fix a prompt distribution P: the IOI generator `v6.tasks.ioi/1.0.0`, including its name,
  place and object distributions.
- Fix an attribution procedure 𝒜: last-position first-order Taylor attribution patching of
  the target−distractor logit difference, with bidirectional name-swap corruption.
- For model m, 𝒜 defines a function f_m : x ↦ a_m(x) ∈ ℝ^{L×H}.
- Norm: ‖f‖² = E_{x∼P} Σ_{l,h} f(x)_{l,h}². Normalised function: f̃ = f / ‖f‖.

**Definition.**

  **D\*(A, B) = min_{π∈G} ‖ f̃_A − π·f̃_B ‖²**, where G = S_H^L (within-layer head permutations).

- The plug-in estimator replaces E_P by the mean over the N evaluation prompts.
- √D\* is a metric on G-orbits. D\* lies in [0, 4].
- **D\* = 0 iff the two models have identical attribution profiles over P, up to
  within-layer head relabelling and a positive global scale.**

**Answer to the conceptual question: yes.** The estimand is *the distance between
(scale-normalised) attribution profiles over P, modulo within-layer head permutation*. It is
**not** a distance between mechanisms.

**Why "mechanism" would overclaim, in both directions.**

- **Small D\* does not imply the same circuit.**
  - 𝒜 sees only each head's first-order effect through its last-position output.
  - Differences in paths through other positions, in MLPs, or beyond first order can leave
    the profile unchanged.
- **Large D\* does not imply different circuits.**
  - Matching is one-to-one within a layer. A role split across two heads, merged into one, or
    moved to an adjacent layer registers as a difference.
  - First-order approximation error can also differ between models: Taylor vs exact patching
    agree in magnitude (Pearson 0.93–0.96) but less in rank (Spearman 0.68–0.75).
- **The estimand depends on P and 𝒜.** Change the prompt distribution, corruption or
  attribution method and D\* changes.

V6 therefore uses **"attribution-profile distance (modulo head permutation)"** throughout.
"Mechanism" is reserved for claims backed by a later intervention experiment.

---

## 1. Candidate baselines

### A. Prompt resampling (paired prompt bootstrap)

1. **Definition.** Draw N prompt indices with replacement, *jointly for both models*.
   Recompute D̂ (re-solving the matching). Repeat B times.
2. **Null represented.** None. It is not a null distribution; there is no hypothesis D\* = 0
   or D\* ≤ c behind it.
3. **Source of variation.** Sampling variability of D̂ as an estimate of D\*, due to the
   finite prompt sample. This is valid only insofar as the N prompts are i.i.d. from P; the
   generator draws items i.i.d. given its seed. It does **not** estimate measurement noise,
   training stochasticity, or run-to-run variation.
4. **Valid for deterministic profiles?** Yes. Uncertainty comes from which prompts were
   sampled, not from noise in the attribution itself.
5. **Head-permutation invariant?** Yes. Each replicate re-solves the minimum (tested).
6. **Head independence assumed?** No. Whole [L, H] slices are resampled per prompt, so all
   head dependence is preserved.
7. **Confidence intervals.** The **percentile bootstrap is invalid** here (§2). The only
   construction that held up in simulation is the **bracketed interval**
   [q_{α/2}(boot), D̂_cf + (q_{1−α/2}(boot) − D̂)], where D̂_cf is the cross-fitted orbit
   distance (matching fitted on half the prompts, evaluated on the other half).
8. **Scientific meaning.** "How precisely have 200 prompts pinned down this pair's
   attribution-profile distance on P?" This is a *standard error*, not a *noise floor*, and
   the doc does not call it one.
9. **Failure modes.**
   - The plug-in minimum is biased toward similarity, and the bias is largest when the
     models differ (no good matching exists).
   - The minimum is non-smooth near near-tied matchings, which breaks bootstrap consistency.
   - Duplicated prompts in resamples.
   - The templated prompts are not a natural distribution; P is the generator's distribution.
10. **Supports a formal H1 test?** Not alone, since it has no null. It is the right
    *within-pair* uncertainty component of a test.

### B. Same-run checkpoint reference

1. **Definition.**
   - For each included run r, fix late checkpoints S = {123000, 133000, 143000} (to be
     pre-registered; no other steps may be added later).
   - Reference set R = {D\*(r@s, r@t) : s < t ∈ S}.
   - The comparison statistic is Δ = mean over cross-run pairs at step 143000 of D\* minus
     mean over R.
2. **Null represented.** H0: attribution profiles of independent runs differ from each other
   no more than late checkpoints of one run differ from each other (Δ ≤ 0).
3. **Source of variation.** Late-training drift within a lineage: shared initialisation,
   shared data order, continued training.
4. **Valid for deterministic profiles?** Yes. It is a comparison of real distances, not a
   noise model.
5. **Head-permutation invariant?** Yes, as long as both sets use D\*.
   - Within-run pairs *also* have known head correspondence, which enables a validity check:
     the optimal matching should recover the identity for non-negligible heads
     (`alignment_recovery`).
6. **Head independence assumed?** No.
7. **Confidence intervals.**
   - Two-stage: runs are the outer unit (resample runs; all within-run and cross-run pairs
     among the resampled runs are recomputed), with the bracketed prompt interval (A) nested
     inside.
   - With 5–10 runs, run-level resampling is coarse. Report the run count prominently and
     prefer a permutation-free exact comparison (e.g. every cross-run pair vs the maximum
     within-run value) as a secondary.
8. **Scientific meaning.** "Do independently seeded runs end up further apart in attribution
   profile than one run moves over its last ~20k steps?" This measures **same training
   lineage**, *not* "same mechanism".
   - Late checkpoints can themselves differ in mechanism (training continues and the learning
     rate is still nonzero).
   - The reference confounds lineage with training-time distance: within-run pairs differ in
     step, cross-run pairs do not.
9. **Failure modes.**
   - **Choice of steps is arbitrary.** Larger gaps give a larger reference. It must be
     pre-registered.
   - **Learning-rate schedule.** Late checkpoints may be close only because updates are small,
     which makes the reference optimistic.
   - **Prior knowledge (disclosure).** I already know within-run positional D_M is small.
   - **Not a mechanism null.**
   - Requires downloading 2 extra checkpoints per included run.
10. **Supports a formal H1 test?** Yes. It is the only option with a real-data comparator and
    a stated H0. The test is then about lineage-relative attribution-profile divergence,
    which the revised H1 below states explicitly.

### C. Head-permutation invariance check

1. **Definition.** For a model X and random π ∈ G: D(X, π·X) = 0, and
   D(X, π·Y) = D(X, Y) for any Y.
   - At weight level: permute head parameter slices, re-run attribution, recompute D.
2. **Null represented.** None. For an invariant statistic the "null distribution" under
   relabelling is a point mass at the observed value, so it cannot test anything.
3. **Source of variation.** None. It checks a property.
4. **Valid for deterministic profiles?** Yes.
5. **Head-permutation invariant?** It *verifies* invariance. Tests pass exactly (|D − C| <
   1e-14 on synthetic data). The weight-level check on pythia-70m showed the attribution side
   is equivariant.
6. **Head independence assumed?** Not applicable.
7. **Confidence intervals.** Not applicable.
8. **Scientific meaning.** The metric cannot be moved by arbitrary parameter labels.
9. **Failure modes.** Rank ties or near-ties can make the Hungarian solution non-unique, but
   the minimum value is still invariant. A float-noise tolerance must be pre-specified for
   the weight-level check (observed 0.12 % of max |attr| on pythia-70m).
10. **Supports a formal H1 test?** No. It is a **validation gate**: it must pass before any
    test.

### D. Synthetic calibrated reference

1. **Definition.** Place the observed D̂ relative to the synthetic distributions for F/A
   (known-equivalent), B (small perturbation) and C/E (known-different), under a generator
   matched to the real data's L, H, N and sparsity.
2. **Null represented.** Only "a model drawn from the generator with the same structure".
   That is a statement about the generator, not about Pythia.
3. **Source of variation.** Generator randomness: structure draws and idiosyncratic per-prompt
   components.
4. **Valid for deterministic profiles?** Only after mapping. Real profiles have no separate
   noise term, so the generator's noise must be read as "idiosyncratic per-prompt variation",
   and its scale is an assumption.
5. **Head-permutation invariant?** Yes, with an invariant metric.
6. **Head independence assumed?** The *generator* assumes independent heads given the
   factors. Real heads are coupled through the residual stream.
7. **Confidence intervals.** Simulation quantiles. They are only as good as the generator.
8. **Scientific meaning.** Whether the observed distance is "of the size" a structural
   difference produces *in a model family we defined*.
9. **Failure modes.**
   - **Transfer.** Transferring the calibration to Pythia requires the generator to match
     real attribution-profile statistics: rank K of profiles across prompts, sparsity pattern,
     layer profile, idiosyncratic-to-shared variance ratio, and head coupling.
   - **The "small perturbation" level (EPS) is arbitrary**, and it decides what counts as
     equivalent.
   - **High circularity risk.** Tuning the generator after seeing real distances can place
     any observation in any region.
10. **Supports a formal H1 test?** No. Use it for **method validation**: type-I error, power
    and interval coverage (§2), with generator parameters fixed before real data are used.

### E1. Cross-fitted estimation (estimator correction, used by A and B)

1. **Definition.** D̂_cf: fit π̂ on half S1, evaluate ‖X̃_A,S2 − π̂·X̃_B,S2‖² on half S2;
   average over K splits.
2. Not a null. It is a de-biasing device. By construction the in-sample minimum on S2 is at
   most D̂_cf.
3. **Source of variation.** Removes selection-on-evaluation-data bias.
4. Valid for deterministic profiles.
5. G-invariant (tested).
6. No head independence assumed.
7. Enters the bracketed interval (A).
8. **Meaning.** An honest, slightly conservative estimate of D\*.
9. **Failure modes.** The half-size fit (N/2) makes the matching noisier, hence the upward
   bias.
10. Supports inference as part of A/B.

### E2. Real-architecture known-different control (intervention calibration)

1. **Definition.** Modify one included model by a known intervention with a graded dose:
   - e.g. mean-ablate, or scale by c ∈ {0.75, 0.5, 0}, the output of a pre-registered set of
     heads (chosen by rank in a *held-out* model, not the model being modified);
   - measure D\*(m, m_c).
2. **Null / positive control.** A positive control: a known, dose-controlled change in the
   network must produce monotonically increasing D\*. Also a within-architecture analogue of
   Control D.
3. **Source of variation.** Intervention strength.
4. Valid for deterministic profiles.
5. Invariant.
6. No head independence assumed.
7. **CIs.** Prompt bootstrap (bracketed).
8. **Meaning.** Tests that the metric is *sensitive* on the real architecture, without
   generator assumptions. Absence of monotonic response means the metric is unfit.
9. **Failure modes.** Ablation changes behaviour as well as attribution. The chosen heads
   could be ones the metric happens to weight lightly.
10. **Supports H1?** No. It is a **sensitivity gate** (like Gate 1 / Control D).

### E3. Why no "split-half noise floor"

- The per-prompt profiles are indexed by prompt. Two disjoint halves of one model's prompts
  cannot be compared head-profile to head-profile.
- The attributions are deterministic, so a model vs itself on the same prompts gives exactly
  0.
- Unlike the positional D_M, the orbit distance therefore has no within-model measurement
  floor. Any "floor" must come from A (precision) or B (reference).

---

## 2. Synthetic coverage study (bootstrap validity)

**Setup.**

- Generator from `identifiability_synthetic.py`, unchanged.
- D\* is approximated with 20,000 fresh prompts.
- 40 replicates per cell, N = 200 prompts, 200 bootstrap draws.
- Coverage is known only to about ±0.04 at 40 replicates.
- **Nominal coverage is 0.95.**

| Interval | dense A | dense B | dense C | sparse A | sparse B | sparse C |
|---|---|---|---|---|---|---|
| Percentile bootstrap | 0.93 | 0.88 | **0.25** | **0.38** | **0.45** | **0.28** |
| Basic (bias-reflected) bootstrap | 0.90 | 0.90 | 0.93 | 0.72 | 0.88 | 0.93 |
| [in-sample, cross-fit] bracket | 0.25 | 0.20 | 0.88 | 0.57 | 0.60 | 0.85 |
| **Bracket + sampling error** | **0.95** | **0.88** | **1.00** | **0.90** | **0.90** | **0.97** |

| Bias vs D\* | dense A | dense B | dense C | sparse A | sparse B | sparse C |
|---|---|---|---|---|---|---|
| In-sample plug-in | −0.0001 | −0.0001 | **−0.0145** | −0.0049 | −0.0043 | **−0.0206** |
| Cross-fitted | +0.0006 | +0.0004 | +0.0243 | +0.0014 | +0.0019 | +0.0295 |
| Mean percentile-CI width | 0.0048 | 0.0062 | 0.0410 | 0.0112 | 0.0136 | 0.0610 |

**Conclusions.**

- The plug-in estimate is biased toward similarity, most strongly when the models truly
  differ. That is exactly the case H1 is about.
- The naive bootstrap inherits this bias and under-covers badly (0.25–0.45).
- The bracketed interval is near nominal (0.88–0.95) where the models are similar and
  conservative (0.97–1.00) where they differ.
- **Before the lock**, rerun the study at ≥ 200 replicates, and with the generator's rank
  and sparsity parameters set from *single-model* real statistics, never from cross-model
  distances.

## 3. Assessment

| Option | Construct validity | Statistical validity | Interpretability | Compute | Circularity risk | Confirmatory? |
|---|---|---|---|---|---|---|
| A prompt bootstrap (bracketed) | Right target: precision of D̂ for D\* on P | Percentile invalid; bracketed ≈ nominal/conservative in simulation | High ("±") | B × Hungarian; seconds | Low | Component only |
| B same-run reference | Lineage-relative; not mechanism | Valid comparison; few runs, so coarse run-level inference | High, if the wording is careful | +2 checkpoints per run (~1.6 GB each) | Medium: step choice must be pre-registered; prior knowledge disclosed | **Yes** (with A nested) |
| C invariance check | Validates the metric's key property | Exact | High | Trivial | None | No (gate) |
| D synthetic calibration | Weak transfer to real models | Only as good as the generator | Medium | Moderate | **High** if tuned post hoc | No (method validation) |
| E1 cross-fitting | Removes selection bias | Supports A's interval | Medium | ×K splits | Low | Component |
| E2 intervention control | Strong for *sensitivity* on the real architecture | Dose–response check | High | Needs model runs | Low if heads are chosen on a held-out model | No (gate) |

## 4. Recommendation (derived from the criteria, not from seed results)

A single baseline cannot carry H1. Each option answers a different question. The
recommendation assigns roles:

1. **Gates, which must pass before any test.**
   - **C:** exact invariance, plus the weight-level equivariance check with a pre-specified
     tolerance.
   - **E2:** monotone dose–response on one held-out real model.
   - **D, restricted to method validation:** bracketed-interval coverage ≥ 0.90 in all cells
     at ≥ 200 replicates, with generator parameters fixed from single-model statistics.
2. **Uncertainty.**
   - **A with E1:** the bracketed paired prompt-bootstrap interval for every pair distance.
   - The percentile bootstrap is not used.
3. **Confirmatory comparator.**
   - **B:** within-run late-checkpoint pairs at pre-registered steps {123000, 133000, 143000}.
   - Inference is at the run level, with prompt uncertainty nested.
   - It is described as *lineage-relative attribution-profile divergence*.

Why this follows from the criteria:

- B is the only option with both a real-data comparator and a stated H0.
- A is the only one that validly quantifies precision for deterministic, prompt-indexed
  profiles, and only in its bracketed form.
- C and D cannot serve as real-data nulls: C is degenerate, and D transfers poorly and is
  circularity-prone.
- E2 is the only real-architecture sensitivity check.

None of these choices depends on seed-pair orbit distances, which have not been computed.

**Limits of the recommendation.**

- **The run count is the binding constraint.** With seed4 excluded, the current pilot has 5
  runs. PolyPythias offers 9 seeds plus the original at 410M. Some may fail inclusion or
  matching, and seeds 3 and 4 are documented outliers.
- **Matching needs more prompts.** Amendment 2's exact rule needs n ≥ 500–1000 prompts to be
  satisfiable.

## 5. Proposed revised H1 (replaces the mechanism wording; pending approval)

> **H1 (revised).** Let 𝓡 be the set of Pythia-410M training runs (original + PolyPythias
> seeds) that pass the inclusion rule and are pairwise performance-matched on the IOI
> distribution P under Amendment 2. For independent runs r ≠ r′ ∈ 𝓡 at step 143000, the
> **attribution-profile distance modulo within-layer head permutation**
> D\*(r, r′) (§0, procedure 𝒜, distribution P) exceeds, on average, the same distance
> between late checkpoints of a single run (pairs of steps {123000, 133000, 143000}):
>
>   **Δ = E_{r≠r′}[D\*(r@143k, r′@143k)] − E_{r, s<t}[D\*(r@s, r@t)] > 0.**
>
> H0: Δ ≤ 0. Test: run-level resampling of Δ with the bracketed prompt interval nested.
> Pre-registered α and minimum effect size to be set by the power analysis.

**What H1 does not claim.** It does not claim that performance-matched runs use different
*mechanisms* or *circuits*. It claims their last-position attribution profiles (under 𝒜, on
P), after optimal within-layer head matching, differ more than a run's own late-training
drift. Any mechanism-level statement requires an intervention experiment (e.g. transferring
aligned head activations between runs and measuring the decision change, i.e. the
pre-registered but unimplemented intervention-transfer D_M).

## 6. Decisions needed before the Amendment 3 lock

1. Approve the estimand wording (§0) and the terminology change across V6.
2. Approve the role assignment (§4) and the revised H1 (§5), or amend them.
3. Pre-register the checkpoint set for B, the E2 intervention (heads, doses, held-out
   model), the coverage-study parameters, the run set, and the prompt count (with
   Amendment 2).
