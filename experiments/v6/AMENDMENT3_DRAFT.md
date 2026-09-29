> **SUPERSEDED by `AMENDMENT3.md` (final text, 2026-09-29). Kept unchanged below for the audit
> trail. Known errors in this draft, corrected in the final text: the 'disjoint' prompt claim
> was untrue (1 overlap, 2 duplicates) and B = 2,000 conflicted with the validated B = 200.**

# Amendment 3 (DRAFT, NOT LOCKED)

*Drafted 2026-09-29. It is locked by a dated commit and a tag `v6-amendment3-lock` only
after Gates 1–8 pass (criteria: `audits/amendment3_gate_criteria.md`, written before any
gate was run, plus dated notes 1–3). Until then **no confirmatory real-model experiment is
run**, and H1–H4 remain UNRESOLVED. The positional D_M results and the earlier sorted-head
results were not used to choose anything below.*

## 1. Estimand and terminology (approved)

- **Definition.** D\*(A, B) = min_{π∈G} ‖f̃_A − π·f̃_B‖².
  - G: within-layer head permutations.
  - f̃_m: scale-normalised last-position Taylor attribution map of model m over the IOI
    prompt distribution P (`v6.tasks.ioi/1.0.0`, name-swap corruption).
- **Name.** It is called the **distance between attribution profiles, modulo within-layer
  head permutation**. Differences are **attribution-profile divergence**.
- **Interpretation limit.** D\* is **not** interpreted as a distance between mechanisms or
  circuits. That would require a later intervention experiment.
- **Implementation.** `glassbox.v6.identifiability.profile_orbit_distance`.

## 2. Revised H1 (approved in principle)

- **Statement.** Among performance-matched Pythia-410M runs, attribution-profile distance at
  step 143000 is greater between independently trained runs than between late checkpoints
  within the same training lineage (distance modulo within-layer head permutation).
- **Estimand.**
  Δ = E_{r≠r′} D\*(r@143000, r′@143000) − E_r mean_{(s,t)} D\*(r@s, r@t),
  with (s, t) ∈ {(123000, 133000), (123000, 143000), (133000, 143000)}.
- **Hypotheses.** H0: Δ ≤ 0. H1: Δ > 0. One-sided α = 0.05.
- **Wording.** The within term is **within-training-lineage variation**, not "same mechanism".

## 3. Runs and checkpoints (Gate 4, verified on the Hub 2026-09-29)

**Candidate runs.** EleutherAI `pythia-410m` (original) and `pythia-410m-seed1` … `seed9`
(PolyPythias, Apache-2.0). The seed jointly controls initialisation and data order. Branch
commits:

| Run | step123000 | step133000 | step143000 |
|---|---|---|---|
| pythia-410m | `51cabaff46` | `45c291dfb8` | `bba6a464f5` |
| seed1 | `3a55097f9e` | `2c44de039f` | `a77e7d8790` |
| seed2 | `319875fb86` | `59534882c2` | `8f8a4fabfc` |
| seed3 | `1cb2ac3dd2` | `a2a1327db3` | `83438f87ad` |
| seed4 | `b884f27285` | `f1dee90510` | `b122bdb906` |
| seed5 | `e48b7a7fe6` | `8b6f55ae44` | `b9156c2847` |
| seed6 | `ce6bdbca13` | `eba0d2f483` | `36a33e1487` |
| seed7 | `a12420095e` | `fbad49e4ac` | `4c83aca7bf` |
| seed8 | `5244916228` | `453deb6719` | `b3d0ef9e32` |
| seed9 | `1933c5b650` | `582bc5e55e` | `6155dcb7e1` |

- The confirmatory run must record the resolved commit per checkpoint (`hub_provenance`) and
  stop if it differs from this table.
- Seed weights are loaded with `use_safetensors=False`, i.e. the official `.bin` files.

## 4. Inclusion, matching and seed 4 (Gate 6)

- **Inclusion** (unchanged, pre-specified): IOI accuracy above chance at step 143000
  (one-sided binomial, α = 0.05).
- **Matching** (Amendment 2, exact rule): exact Clopper–Pearson discordance bound < 2 pp on a
  **matching set of 1,000 IOI prompts** (seed 2; disjoint from the attribution prompts).
  - At n = 1,000 a pair matches iff ≤ 12 prompts disagree.
- **Primary run set — proposed operationalisation, needs your approval.** Runs passing
  inclusion **and** matched to the original `pythia-410m` run (a star design; deterministic,
  no search over subsets). The full pairwise matching matrix is reported.
- **Seed 4.**
  - Retained in the complete dataset and in every table.
  - Its broad failure is reported: 84/200 IOI, name mass 0.047, sentence NLL 3.86 (others
    3.44–3.66), and loss spikes documented in PolyPythias.
  - Its expected exclusion comes from the inclusion rule, not from its distances.
- **Seed 3** is flagged as a PolyPythias outlier. It is kept if it passes inclusion and
  matching.
- **Pre-specified sensitivity analyses.**
  - (S1) All 10 runs, no inclusion or matching.
  - (S2) Primary set without seed 3.
  - (S3) Cross-fitted D̂_cf in place of the plug-in.

## 5. Measurement

- **Attribution prompts.** N = 200 (IOI seed 0), the same for all 30 checkpoints.
- **Estimator.** Plug-in D̂.
  - Its bias toward similarity grows with the true distance, so it can only shrink a
    positive Δ (conservative for H1).
- **Prompt resampling.** B = 2,000 multinomial prompt resamples, shared by all models.

## 6. Inference (Gate 5)

- **Unit.** The training run.
- **Primary procedure.** Two-way bootstrap:
  - each replicate resamples runs (self-pairs dropped) and prompts jointly;
  - the one-sided 95 % lower bound is the 5th percentile of Δ̂*;
  - reject H0 iff it is > 0.
  - Implementation: `glassbox.v6.lineage.two_way_bootstrap`.
- **Fallback, used only if the primary fails Gate 5 validity.** Leave-one-run-out jackknife
  with t_{R−1} (`lineage.jackknife`).
- **Per-pair values.** Gate 1 failed, so every per-pair D̂ is reported as a **point estimate
  only**, never with a confidence interval. Uncertainty statements are made only for Δ, at
  the run level (Gate 5 validated this procedure's type-I error).

## 7. Gate status (as of this draft)

| Gate | Criterion | Status |
|---|---|---|
| 1 Interval validity (R = 200, 9 conditions) | coverage ≥ 0.919 everywhere | **FAILED, closed.** Original procedures and the single pre-declared revision (v2: 0.915 in sparse/similar and sparse/perturbed) both failed. Fallback: per-pair values are point estimates only |
| 2 Invariance under weight-level head permutation | D(X, X_π) ≤ 1e-4; cross check within the triangle bound | **PASS** (410M: 1.7e-7; 1.2e-5 ≤ 2.7e-4) |
| 3 Synthetic discrimination A–E | as pre-declared | **PASS** (dense, sparse, unstable) |
| 4 Within-lineage reference pairs | exact pairs, branches exist | **PASS** (§3) |
| 5 Run-level inference validity | type-I ≤ 0.081 at R = 6, 10 | **PASS**: bootstrap 0.045–0.065, jackknife 0.035–0.050; power 1.0 down to Δ ≈ 0.037 |
| 6 Seed 4 policy | inclusion primary; retained descriptively | **Specified**; star design awaits approval |
| 7 Controlled perturbation sensitivity | D(c = 1) = 0; strictly increasing | **PASS** (410M: 0 → 0.028 → 0.155 → 0.521 → 1.161) |
| 8 Terminology | "attribution-profile divergence" throughout | **PASS** |
| 9 Freeze | after 1–8 | **Blocked, pending owner decisions:** accept the Gate 1 fallback? approve the star design? |

## 8. Cost of the confirmatory run (estimates, not measured)

- **Checkpoints.** 30 in total.
  - The original run's three are 1.6 GB each; three are already cached.
  - The 27 seed checkpoints are 0.91 GB each, about 25 GB in total.
  - Download, measure, cache and delete: the pilot cache is per model and resumable.
- **Attribution compute.** About 2 s per prompt per checkpoint on the Mac CPU. That is
  30 × 200 × 2 s ≈ 3.3 h, plus the 1,000-prompt matching forward passes.
