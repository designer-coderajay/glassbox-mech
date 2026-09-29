# Head identifiability of V6 attribution-profile distances

*Terminology note: "mechanism" in the synthetic scenarios below means *generator
structure*. For real models, all distances here are attribution-profile distances, not
mechanism distances (see `orbit_baselines.md` §0).*

*Written 2026-09-29, before Amendment 3. Code: `glassbox/v6/identifiability.py`; tests:
`tests/test_v6_identifiability.py`; synthetic study:
`experiments/v6/audits/identifiability_synthetic.py` (+ `.json`); permutation nulls:
`experiments/v6/audits/positional_permutation_null.json`. H1–H4 remain **UNRESOLVED**. None of
the candidate invariant metrics below has been applied to any seed (or other pilot) data.*

## 1. What the earlier diagnostic did

`controls.relabel_within_layers` (seed audit, §5) permutes **entries of the saved attribution
vector** within each layer. It does not touch model weights. On its own, it is therefore a
**measurement relabelling null**, not a transformer-symmetry result.

This document adds the weight-level result that connects the two:

- `identifiability.permute_heads_` applies a function-preserving permutation to a real
  HookedTransformer. It permutes each head's W_Q, W_K, W_V, b_Q, b_K, b_V, and its W_O slice,
  along the head axis; b_O is shared.
- The tests verify two properties:
  1. **the function is unchanged**;
  2. **attribution patching on the permuted model returns exactly the relabelled vector**.
- On a random 3-layer rotary model the match holds to 1e-6.
- On **pythia-70m @ step 143000** (IOI prompt, name-swap corruption):
  - the max logit difference is 3.7e-3 on logits of size ~800 (relative 4.5e-6, float32
    summation order);
  - the permuted model's attribution matches π·(original) with Spearman **0.99989** (max
    deviation 0.12 % of max |attr|);
  - the positional D_M between the model and its functionally identical twin is **1.04**
    (Spearman −0.04).

**Conclusion.** For this attribution procedure, relabelling the attribution vector is
equivalent (up to float error) to a genuine function-preserving symmetry. The pre-registered
positional D_M rates a model and an exact functional copy of itself as maximally different. So
it is not a function of the model, only of its parameterisation.

## 2. Formal setting

- **Attribution tensor.** Model m has X ∈ ℝ^{N×L×H} (items × layers × heads). Mean vector:
  a = mean_i X[i] ∈ ℝ^{L×H}.
- **Symmetry group.** G = S_H^L: an independent permutation π_l of the H heads in each layer.
  - Action on attributions: (π·X)[i,l,h] = X[i,l,π_l(h)].
  - Equivariance (verified in §1): X(π·θ) = π·X(θ).
- **Identifiability requirement.** A distance between models must satisfy
  d(A, B) = d(A, π·B) for all π ∈ G. A metric violating this measures parameter labels, not
  function.

## 3. Candidate distances

Definitions are fixed in code. Each is summarised below with what it can and cannot claim.

| Name | Definition | G-invariant | What it measures |
|---|---|---|---|
| `positional_dm` (pre-registered D_M) | 1 − ρ_s(a_A, a_B) | **No** | Agreement of head-*labelled* attributions. Valid only when head correspondence is known (checkpoints of one run). |
| `scalar_quotient_dm` | min_{π∈G} [1 − ρ_s(a_A, π·a_B)] = sort each layer (rearrangement inequality) | Yes | Per-layer *distributions of scalar attribution magnitudes* (1-D optimal transport). **Not a mechanism distance.** |
| `profile_orbit_distance` | D = min_{π∈G} ‖X̃_A − π·X̃_B‖_F², with X̃ = X/‖X‖_F; the minimum splits into one exact assignment problem per layer (Hungarian) | Yes | Distance between multisets of per-head **causal-effect profiles across inputs**, up to relabelling and overall scale. √D is a metric on G-orbits (G finite, acting by isometries); range [0, 4]. |
| `crossfit_aligned_dm` | Per split: align heads on half the items (per-layer Hungarian on 1 − Pearson of profiles), then 1 − ρ_s of the means on the *other* half; averaged over K splits | Yes (exactly, up to ties) | Head-labelled agreement after data-driven alignment, not fitted on the evaluation items. |

**Caveats on the profile metrics.**

- **Optimistic bias.** Both minimise or align using data, so both are biased toward
  similarity. `crossfit` reduces this by fitting and evaluating on different items; `orbit`
  does not.
- **Still attribution distances.** They inherit the instrument's limits (last-position,
  first-order Taylor). A small distance means similar attribution profiles. It does not prove
  identical circuits.
- **Matching is not independent of the construct.** Alignment by attribution profiles is not
  independent of attribution. Alignment by an independent signal (attention patterns, OV/QK
  similarity) would be stronger, and is listed in §7.

## 4. Positional D_M under the relabelling null

Null: D_M(a_A, π·a_B) for 10,000 uniformly random π ∈ G per pair (seeded). The percentile is
where the observed D_M falls in that null. **Low** means the two models' head labels line up
better than arbitrary labels.

| Pair type | n | Observed D_M | Null mean / median | Null p95 / p99 | Observed percentile |
|---|---|---|---|---|---|
| Same run, different checkpoint | 6 | 0.049–0.125 | 0.85–0.90 / 0.85–0.90 | 0.94–0.99 / 0.97–1.03 | **0.00** (all) |
| Standard vs deduped | 9 | 0.750–0.785 | 0.91–0.93 / 0.91–0.93 | 1.00–1.02 / 1.03–1.06 | **0.05–0.32** |
| Independent seeds | 15 | 0.804–1.112 | 0.89–1.00 / 0.89–1.00 | 0.97–1.09 / 1.00–1.13 | **3.9–99.6** (spread across the range) |

Per-pair values are in `positional_permutation_null.json`.

**Reading:**

- **Within a run,** head labels correspond almost perfectly, and positional D_M is
  interpretable.
- **Standard vs deduped** shows far-better-than-chance correspondence. This is consistent with
  shared initialisation or strong convergence; I did not verify which. But 0.75 is not far
  below the null (median ~0.92), so correspondence is only partial.
- **Across seeds** there is no evidence of any correspondence. Positional D_M there is
  indistinguishable from arbitrary relabelling and carries no information about attribution-profile similarity.
- One seed pair sits at the 99.6th percentile. With 15 dependent pairs this is not
  interpretable.

## 5. Synthetic validation (scenarios fixed before use on real data)

**Generator:**

- X[i,l,h] = m[l,h] + s_l · F[i]·W[l,h] + noise, with shared item factors F.
- L = 12, H = 16, N = 200, K = 8; 30 replicates.
- **Sparse regime:** 60 % of heads scaled by 0.01, matching the ~60 % near-zero heads in
  Pythia-410M.

**Scenarios (B_model vs A_model):**

- **F:** same structure, fresh noise (the floor).
- **A:** arbitrary relabelling.
- **B:** small perturbation.
- **C:** independent structure from the same distribution.
- **D:** C relabelled.
- **E:** same multiset of *mean* effects (relabelled) but new per-input behaviour.

**Desired:** A ≈ F, B low to moderate, C high, D = C, E high.

Mean ± sd over 30 replicates:

| Regime | Metric | F | A | B | C | D | E |
|---|---|---|---|---|---|---|---|
| dense | positional | 0.006 ±0.001 | **0.937 ±0.082** ✗ | 0.052 ±0.014 | 0.996 ±0.071 | 1.021 ±0.071 (≠ C) | 0.961 ±0.090 |
| dense | crossfit_aligned | 0.015 ±0.002 | 0.015 ±0.002 | 0.061 ±0.012 | 0.957 ±0.080 | = C | 0.910 ±0.072 |
| dense | profile_orbit | 0.037 ±0.002 | 0.037 ±0.002 | 0.094 ±0.004 | 1.040 ±0.032 | = C | 1.020 ±0.031 |
| dense | scalar_quotient | 0.002 ±0.001 | 0.002 ±0.001 | 0.021 ±0.007 | 0.125 ±0.042 | = C | **0.023 ±0.007** ✗ |
| sparse | positional | 0.252 ±0.045 | **0.972 ±0.087** ✗ | 0.417 ±0.058 | 1.010 ±0.070 | 0.993 ±0.063 | 0.953 ±0.088 |
| sparse | crossfit_aligned | 0.314 ±0.037 | 0.316 ±0.043 | 0.429 ±0.040 | 0.978 ±0.053 | = C | 0.931 ±0.050 |
| sparse | profile_orbit | 0.085 ±0.011 | 0.085 ±0.011 | 0.149 ±0.014 | 1.330 ±0.064 | = C | 1.239 ±0.067 |
| sparse | scalar_quotient | 0.044 ±0.014 | 0.045 ±0.009 | 0.092 ±0.017 | 0.112 ±0.031 | = C | **0.057 ±0.018** ✗ |

"= C" means |D − C| ≤ 5.6e-15 over all replicates.

- **positional:** fails A (reads a pure relabelling as maximally different) and is not
  invariant.
- **scalar_quotient:** fails E by construction. Different structures with the same magnitude
  histogram look like a small perturbation. In the sparse regime it barely separates C (0.11)
  from B (0.09).
- **crossfit_aligned** and **profile_orbit** meet every desired property. Separation of C,
  (C − F)/sd_C:

| Metric | Dense | Sparse |
|---|---|---|
| crossfit_aligned | 11.7 | 12.6 |
| profile_orbit | 31.1 | 19.3 |

- crossfit's floor rises from 0.015 to 0.31 in the sparse regime: rank noise from
  near-zero heads, the same weakness as positional D_M. orbit's floor stays low (0.085),
  because a Frobenius distance weights heads by magnitude.

## 6. Recommendation (on validity grounds; seed data not consulted)

Selection criteria, in order:

1. exact G-invariance;
2. detects different structures, including E (same magnitudes);
3. robust to the observed sparsity;
4. magnitude-aware, so negligible heads cannot dominate;
5. interpretable range.

| Role | Metric | Reason |
|---|---|---|
| **Primary cross-run D_M (proposed)** | `profile_orbit_distance` | Meets 1–5; best separation in both regimes; √D is a proper metric on orbits. |
| Pre-specified sensitivity | `crossfit_aligned_dm` | Invariant and not fitted on the evaluation items; rank-based like the original D_M, but sensitive to near-zero heads. |
| Within-run only | `positional_dm` | Valid only when the relabelling-null percentile shows correspondence (e.g. < 1 %). Always report that percentile. |
| Descriptive only | `scalar_quotient_dm` | Reported as "attribution-magnitude distribution distance", never as mechanism. |

**Disclosure.** I had already seen seed results under the scalar (sorted) metric before this
analysis (seed audit §5). That metric is **demoted** here on the E scenario. The recommended
primary has not been run on any real pilot data.

## 7. Open before Amendment 3 can be locked

1. **Null / reference class for the orbit distance on real data.**
   - Real attributions are deterministic given the items (Control A: bitwise identical), so
     there is no "fresh measurement noise" floor like scenario F.
   - Split-half is not possible either: profiles are indexed by item, so two halves are not
     comparable head-profile to head-profile.
   - Candidates:
     - an item bootstrap (resample items jointly for both models) for CIs;
     - within-run checkpoint pairs as an empirical "same-lineage" reference class;
     - a relabelling-invariant permutation test at the model level.
   - This must be chosen and written down before any H1 test.
2. **Independent alignment signal.** Optionally, align heads by attention patterns or OV/QK
   weights (independent of attribution) as a further sensitivity analysis.
3. **Construct wording.** Even the invariant metrics measure *last-position Taylor attribution
   profiles*. H1's wording ("matched performance does not imply matched mechanism") should
   become "… does not imply matched attribution profiles (up to head relabelling)", unless a
   circuit-level criterion is added.
4. **Seeds share data *and* init differences** (coupled in PolyPythias). Whether standard and
   deduped share init remains unverified; the §4 evidence is suggestive only.

## 8. Proposed Amendment 3 (not yet approved)

> D_M for pairs without verified head correspondence is `profile_orbit_distance` on
> unit-Frobenius-normalised per-item attribution tensors, with `crossfit_aligned_dm` as a
> pre-specified sensitivity analysis. The positional D_M is reported only for pairs whose
> relabelling-null percentile is below 1 %, and always alongside that percentile.
> `scalar_quotient_dm` is descriptive only. The null/reference class (§7.1) and H1's wording
> (§7.3) are fixed in the same amendment, before any confirmatory run. H1–H4 stay UNRESOLVED
> until then.
