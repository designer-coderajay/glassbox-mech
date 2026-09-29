# Glassbox V6 pre-registration (DRAFT, not yet locked)

| Field | Value |
|---|---|
| Status | **DRAFT.** Items marked `PENDING` are not decided. The document is locked by a dated commit (tag `v6-prereg-lock`) after the pilot and before any confirmatory run. Edits after the lock are appended as dated amendments, never silent changes. |
| Written | 2026-09-28 |
| Author | Ajay Pravin Mahale |
| Code | `glassbox/v6/` (metrics `v6.distances/1.0.0`, task `v6.tasks.ioi/1.0.0`, schema `v6.claims/1.0.0`) |
| Disclosure | This draft was written after one smoke run on one pair (§9). That run validated the pipeline only; its numbers are not used to set any margin, threshold or sample size. |

This document is self-contained: it restates every definition the analysis depends on.

## 1. Question

Across pairs of models with the same architecture that perform equally well on a task
(performance distance ≈ 0), do their mechanisms differ (mechanistic distance > noise), and
does a black-box behavioral distance carry information about that mechanistic distance?

Identifiability limit, stated up front: different mechanisms can produce identical
behavior. Black-box similarity cannot prove mechanistic similarity, and black-box difference
cannot identify a specific mechanism. V6 tests only whether behavioral distance *predicts an
independently measured* mechanistic distance.

## 2. Arms, tasks and models

- **Arm A (natural models):** Pythia checkpoints and deduplicated twins (EleutherAI,
  Apache-2.0), loaded via TransformerLens `checkpoint_value`. Task: IOI.
  Model list: `PENDING` (pilot). Proposed scale: 8–10 models (up to 45 pairs).
- **Arm B (constructed models with ground truth):** synthetic-credit models including the
  designed known-positive (§5, Control D). Task: synthetic credit
  (`experiments/decision_audit/credit_rule.py`). Not built yet (milestone 2).
- No other tasks in v6.0.

**Model inclusion (`PENDING`, proposal):** a model enters an arm only if its task accuracy is
above chance under a one-sided binomial test at α = 0.05 on the pilot items. Reason: the
attribution vector of a model that cannot do the task does not describe a task mechanism.
The smoke run (§9) showed pythia-70m near chance on IOI, so this criterion matters in
practice.

## 3. Distances (definitions fixed here)

For models A, B on task T:

### 3.1 Performance distance D_P

Per item i, `correct_i = [LD_i > 0]` where `LD_i = logit(target) − logit(distractor)` at the
last position. `D_P` reports the accuracy difference `acc_A − acc_B` and the mean LD
difference.

A pair is **performance-matched** iff a paired TOST on `d_i = correct_A,i − correct_B,i`
rejects both `H0: mean(d) ≤ −δ` and `H0: mean(d) ≥ +δ`, each at α = 0.05 (one-sided t-tests).
When every `d_i` is identical (SD = 0) the t statistic is undefined; then no p-value is
reported and the pair is matched iff the exact one-sided (1 − α) Clopper–Pearson upper
bound on the discordance rate is below δ (valid because |accuracy gap| ≤ discordance
rate). With 0 discordant items this bound is `1 − α^(1/n)`: 1.49 pp at n = 200,
5.8 pp at n = 50. (Amendment 1, §10.)

- Margin δ = **±0.02 (2 accuracy points), PROVISIONAL.** Final δ is set once, from the pilot,
  before the lock, and is never changed after confirmatory results are seen.
- Power note (my estimate, to be replaced by the pilot's power analysis): with binary
  correctness and ~10 % discordant items, the SD of `d_i` is ≈ 0.32, so passing TOST at
  δ = 0.02 needs roughly 700+ items per pair. The confirmatory item count must be set
  accordingly; a ±2pp margin with tens of items cannot declare equivalence.

### 3.2 Mechanistic distance D_M (operational)

D_M is an *operational* measurement of mechanistic difference under one attribution
procedure. It is not a ground-truth measure of "the mechanism".

- Attribution vector: per attention head (l, h), Taylor attribution patching
  `attr(l,h) = ∇_{z_lh} LD · (z_clean − z_corrupt)` (`GlassboxV2.attribution_patching`,
  3 passes) with IOI name-swap corruption, **averaged over items**.
- **Primary:** `D_M = 1 − ρ_s(attr_A, attr_B)`, Spearman over the identical head set. Range
  [0, 2]. Undefined (constant vector) → recorded as `UNRESOLVED` with reason, never imputed.
- **Secondary 1:** `1 − Jaccard(top-k_A, top-k_B)`, heads ranked by |attr|, ties broken by
  head index. k = 10 (`PENDING` confirmation at the pilot).
- **Secondary 2:** intervention-transfer loss (patch A's circuit activations into B, measure
  the decision change). Not implemented yet.

### 3.3 Behavioral distance D_B

`D_B = mean over probes q of JSD₂(p_A(·|q), p_B(·|q))`, base-2 Jensen–Shannon divergence
between full next-token distributions (bounded in [0, 1]), outputs only. Requires a shared
vocabulary.

Probe set (IOI), one of each per item, generated deterministically from the seed:

| Kind | Construction | Expected answer |
|---|---|---|
| counterfactual | second-clause subject changed to the IO name | flips to the other name |
| feature_swap | first-clause name order swapped | unchanged |
| perturbation | place and object replaced (seeded) | unchanged |

The probe set is fixed before the confirmatory run and is never tuned on held-out models.

## 4. Hypotheses (each tested separately)

| | Claim | Estimand (primary endpoint) | Null | Test | Effect size + CI | Control |
|---|---|---|---|---|---|---|
| H1 | Among matched pairs, D_M exceeds measurement noise | median D_M (matched pairs) − median D_M (Control B) | difference ≤ 0 | permutation over model labels | difference; model-bootstrap 95 % CI | B (noise floor), A, E |
| H2 | D_B separates divergent pairs from null pairs | AUROC of D_B | AUROC ≤ threshold (`PENDING`) | AUROC | AUROC; model-bootstrap CI | B, C |
| H3 | D_B and D_M are positively associated | Mantel r (D_B, D_M matrices) | r ≤ 0 | Mantel permutation | r; model-bootstrap CI | sensitivity set, §6 |
| H4 | H3 holds on held-out models | Spearman ρ(D_B, D_M) on held-out models | ρ ≤ 0 | held-out by model | ρ; model-bootstrap CI | — |

A pair is **divergent** iff its D_M exceeds the larger of the two models' Control B 95th
percentiles (draft rule, implemented in `glassbox/v6/controls.py::is_divergent`). The
percentile values come from data; the rule does not change after data are seen.

Interpretation rules: a hypothesis that is not tested, or that holds under only one
multiverse choice, is `UNRESOLVED`, not negative. A failed H1 is written up as a negative
result (Gate 2).

## 5. Controls (kept conceptually separate)

| | Construction | Tests | Pass rule |
|---|---|---|---|
| A Pipeline null | same model loaded and measured twice | does the pipeline invent differences? | every distance ≤ ε_det. ε_det = `PENDING`, set from repeated identical runs; exactly 0 only where bitwise determinism is shown (it was, on CPU, in §9) |
| B Resampling null | same model, disjoint item halves: D_M between the two half-mean attribution vectors, 200 random splits (`PENDING` confirmation), seeded | measurement noise floor | defines the null distribution for "divergent". Halves use n/2 items, so the null overstates noise at n (conservative) |
| C Task null | same pair, permuted/irrelevant task | metric reacts to task structure | no divergence signal |
| D Known-positive | Arm B X-only vs Y-only models | detects a known difference | exceeds threshold (`PENDING`) in the correct direction; valid only after the correlation-breaking test (X-model follows X and ignores Y when decorrelated, and the reverse) |
| E Random-circuit baseline | same-size random head sets | attribution beats chance | real D_M differs from random |
| F Experimental | performance-matched pairs | the question | measured |

Implemented so far: Controls A and B, and the inclusion check (§2, reported, not yet
enforced). C–E: not implemented.

## 6. Statistics

- Unit of analysis: the model pair. Pairs sharing a model are not independent.
- CIs: bootstrap over **models** (resample models, rebuild all pairs among them), 10 000
  resamples (`PENDING` confirmation), percentile intervals.
- Two nulls, never combined: Control B defines when one pair is divergent; the permutation
  null over model labels tests the H1 and H3 statistics.
- Permutations: 10 000 (`PENDING` confirmation), seed recorded.
- H3 sensitivity set: partial Mantel controlling D_P; leave-one-model-out Mantel; Spearman
  and Pearson variants; top-k Jaccard D_M; per-probe-kind D_B. H3 passes only if the sign and
  threshold hold across all of them.
- Multiple comparisons: primary endpoints are the four above; Benjamini–Hochberg FDR at
  q = 0.05 across secondary metrics.
- Multiverse dimensions: corruption strategy (name swap vs `corruption.py` alternatives),
  attribution method (Taylor vs integrated gradients), D_M variant (Spearman vs top-k), k.
  Full list `PENDING` at lock.
- Held-out split: by model, fixed before the confirmatory run; held-out models never touch
  thresholds or probe design.

## 7. Pilot boundary

The pilot estimates variance and sets: δ, item count, model list, ε_det, the Control D
threshold, the H2 AUROC threshold, k. Pilot data (runs labelled `pilot`) are excluded from
the confirmatory analysis. Runs labelled `smoke` are pipeline checks only. The code refuses
to label a single-pair run `confirmatory`.

## 8. Gates

- **Gate 0:** literature check. If H1 is already established, it becomes a replication and
  the paper centres on H2–H4.
- **Gate 1:** Control A within ε_det and Control D above its threshold in the correct
  direction. If Control D fails, stop and fix the instrument.
- **Gate 2:** H1 primary endpoint passes. If not, publish the negative result.
- **Gate 3:** a claim reaches `REPRODUCED` only if (1) the primary endpoint passes, (2) the
  effect threshold is met, (3) the CI excludes the null, (4) all negative controls behave,
  (5) robustness checks survive, (6) the result holds across the multiverse, (7) no open
  implementation, leakage or determinism issue.
- **Gate 4:** blind reproduction by an outside person, not told the expected result, matches
  in under 30 minutes of setup.

Nothing is described as validated, on the website or elsewhere, until the relevant gate has
passed.

## 9. Runs so far

410M pilot (label `pilot`, n = 200 IOI items, seed 0, commit `61b7c8c`). Audited in
`experiments/v6/audits/audit_pair.py`; audit output committed next to each run.

| Pair | Acc. A / B | D_M | D_M, exact patching (20 items) | Control B threshold | Cross-model split-half D_M (min) vs within-model (max) | D_B |
|---|---|---|---|---|---|---|
| 410m @143k vs 410m @71k | 200/200, 200/200 | 0.170 | 0.268 | 0.169 | 0.181 vs 0.177 | 0.030 |
| 410m @143k vs 410m-deduped @143k | 200/200, 200/200 | 0.765 | 0.782 | 0.141 | 0.736 vs 0.168 | 0.150 |

Pair B record reproduced exactly on rerun (all non-volatile fields). These are pilot
observations; they do not test H1–H4 and are excluded from confirmatory analysis.

6-model pilot (`runs/pilot_410m_6models`, commit `aa8afe5`, metrics 1.1.0): pythia-410m and
pythia-410m-deduped at steps 123000/133000/143000, 200 items, 15 pairs. Models shared with
the earlier runs reproduced bit-for-bit.

| Pair type | n pairs | D_M | D_M (\|attr\| > 0.01 heads) | top-k distance | D_B | Divergent (draft rule) |
|---|---|---|---|---|---|---|
| Same training run, different checkpoint | 6 | 0.049–0.125 | 0.013–0.095 | 0.18–0.33 | 0.011–0.027 | 0 / 6 |
| Different run (standard vs deduped) | 9 | 0.750–0.785 | 0.773–0.813 | 0.75–0.82 | 0.107–0.158 | 9 / 9 |

Interpretation limits: the 15 pairs come from only **two training runs**, so they contain
one independent between-run contrast, not 9. D_P matching of the 8 pairs with one
discordant item relies on the t-based TOST (p = 0.0015–0.0026); the exact discordance
bound for 1/200 is 2.35 pp, above the 2 pp margin, so these matching decisions are
method-sensitive (open decision, §10). D_B tracks the logit-difference gap as well as D_M
within the 9 cross-run pairs (descriptive Spearman 0.62 vs 0.42, n = 9).


| Run | Label | Models | Items | Result |
|---|---|---|---|---|
| `runs/smoke_pythia70m_step143000_vs_step71000` | smoke | pythia-70m @ step 143000 vs @ step 71000 | 40 IOI items, 120 probes, seed 0 | Pipeline works end to end; Control A bitwise identical on CPU; reruns reproduced every distance exactly. Control B threshold 0.281 vs D_M 0.290 (flagged divergent under the draft rule). **Both checkpoints fail the inclusion check** (accuracy 0.425, p = 0.87; 0.500, p = 0.56), so the divergence flag has no scientific meaning here. Not a pilot result. |

## 10. Amendment log (pre-lock changes, all disclosed)

| # | Date | Change | Trigger | Effect on existing results |
|---|---|---|---|---|
| 1 | 2026-09-28 | D_P: SD = 0 case no longer reports p = 0.0; uses the exact discordance bound in §3.1. Metrics version 1.0.0 → 1.1.0. | Audit of the 410M pilot found `p_tost = 0.0` reported where the t statistic is undefined (reporting defect). | None on decisions: both 410M pairs (0/200 discordant) remain matched (bound 1.49 pp < 2 pp). Stored records keep their 1.0.0 values. |

Open decisions raised by the 410M pilot (not yet made; to be decided on grounds that do
not depend on which answer favours H1, and recorded here when decided):

- Whether matching also requires a logit-difference criterion (all 410M models are at
  100 % accuracy, so accuracy-only matching is trivially satisfied at ceiling).
- Whether to add pre-specified D_M sensitivity analyses that down-weight negligible heads
  (magnitude floor, Pearson). Motivation, disclosed: in the same-lineage pair, the primary
  all-head D_M (0.170) drops to 0.04–0.06 when heads with |attr| ≤ 0.01 are excluded.
- The attribution instrument is last-position-only; Taylor vs exact patching agree in
  magnitude (Pearson 0.93–0.96) but less in rank (Spearman 0.68–0.75).
- D_P method for sparse discordance: with 1 discordant item in 200 the t-based TOST
  declares equivalence (p ≈ 0.002) while the exact discordance bound does not (2.35 pp >
  2 pp). One exact method for all cases must be chosen before the lock.
- Independence: the design needs several independent training runs per arm. EleutherAI
  publishes `pythia-410m-seed1` … `seed9` (Apache-2.0, 154 step branches each, same
  architecture and vocabulary as pythia-410m; weights as `pytorch_model.bin`; not in the
  TransformerLens model list). Checked 2026-09-29 on the Hub.
