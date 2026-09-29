# Post-pilot audit: Pythia-410M seed pilot

*Written 2026-09-29. Run audited: `experiments/v6/runs/pilot_410m_seeds` (label `pilot`,
commit `d0cae27`, 200 IOI items, seed 0; models pythia-410m and pythia-410m-seed1…seed5, all
at step 143000). This is an audit of a pilot. Nothing here is evidence for H1–H4, and no
confirmatory analysis has been run.*

**Summary.**

- **Provenance defect (fixed).** The five seed models were not loaded from the revision we
  requested. transformers silently loaded an unmerged bot-conversion PR instead.
- **Seed4 is a genuine outlier.** It is a documented unstable training run, not a loading
  error.
- **Matching rule.** The implemented t-based matching rule gives inconsistent decisions, so
  one exact rule is proposed.
- **Most important: the primary D_M is invalid for independently initialised models.** A
  function-preserving relabelling of one model's heads already produces D_M ≈ 0.76–1.05, the
  same range as the seed pairs.
- **What survives.** A label-free diagnostic still separates seed pairs from the noise
  floor. That is a weaker, different claim.

---

## 1. Provenance

**What we requested:**

| Repo | Revision | Commit | Weights file | Weights sha256 (Hub LFS) | config.json blob | Tokenizer file |
|---|---|---|---|---|---|---|
| EleutherAI/pythia-410m | step143000 | `bba6a464f5` | model.safetensors | `1c88b0bf1829` | `0425fa13` | tokenizer.json |
| EleutherAI/pythia-410m-seed1 | step143000 | `a77e7d8790` | pytorch_model.bin | `b40bd3670e12` | `aed8cbb7` | tokenizer.json |
| EleutherAI/pythia-410m-seed2 | step143000 | `8f8a4fabfc` | pytorch_model.bin | `6d4a8361e3f4` | `aed8cbb7` | tokenizer.json |
| EleutherAI/pythia-410m-seed3 | step143000 | `83438f87ad` | pytorch_model.bin | `77e2110771b1` | `aed8cbb7` | tokenizer.json |
| EleutherAI/pythia-410m-seed4 | step143000 | `b122bdb906` | pytorch_model.bin | `ee1cf6382571` | `aed8cbb7` | tokenizer.json |
| EleutherAI/pythia-410m-seed5 | step143000 | `b9156c2847` | pytorch_model.bin | `8714fb7a9c08` | `aed8cbb7` | tokenizer.json |

- **Config.** The seed config differs from the standard config only in `classifier_dropout`
  (None vs 0.1; unused by the causal LM head) and `transformers_version`. Architecture:
  24 layers × 16 heads, d_model 1024, vocab 50304, rotary 0.25, parallel residual.
- **Vocabulary.** The seed1 vocabulary is identical to pythia-410m's (50,277 entries), and the
  IOI names tokenise identically. All models were run with the pythia-410m tokenizer, because
  TransformerLens converts the seeds as that architecture.
- **Parameter count.** Seed4: 405,334,016 parameters over 364 tensors, excluding buffers.
  This equals the analytic count for pythia-410m.

**What was actually loaded (defect, now fixed).**

- The pilot log shows the seed weights came from `resolve/refs%2Fpr%2F1/model.safetensors`.
- The requested revisions contain only `pytorch_model.bin`. In that situation transformers
  4.57 loads a `model.safetensors` from an open conversion PR if one exists.
- These PRs were opened by **SFconvertbot**, not EleutherAI, and are unmerged:
  - seeds 1–4 each have one PR (`refs/pr/1`) branched from the current `main`;
  - seed5 has two (`refs/pr/1` from an older `main` commit `38934e3`, and `refs/pr/2`).
- **Metadata check.** In every seed repo, `main`, `step143000` and each PR's parent hold the
  byte-identical `pytorch_model.bin`. Seed5's two PRs contain the identical safetensors file
  (`ec3df36fe077`).
- **Tensor check (seed4, done here).** Both files match the Hub sha256. All 364 tensors of the
  PR safetensors are bit-identical to the official `step143000` `.bin`.
- **Tensor check (seeds 1, 2, 3, 5).** Not yet checked. `seed_diagnostics.py` does this.
- **Fix.** `load_model` now passes `use_safetensors=False` for seed variants. It was
  confirmed to load commit `b122bdb` (step143000) for seed4.
- **Provenance recording.** Every pilot record now stores each model's repo, revision,
  resolved commit and weights sha256 (`measure.hub_provenance`).
- **Consequence.** If the tensor check passes for all seeds, the stored pilot numbers were
  computed from the correct weights and stand. If any seed fails, that seed's numbers are
  void and must be re-measured.

## 2. Seed independence

Source: van der Wal et al., *PolyPythias: Stability and Outliers across Fifty Language Model
Pre-Training Runs*, ICLR 2025, arXiv 2503.09543 (abstract and §5, read 2026-09-29).

- The PolyPythias add 9 seeds per size (14M–410M) on top of the original Pythia run.
- The seed jointly controls **parameter initialisation and training-data order**. These are
  coupled; a later release with decoupled seeds is announced in the OpenReview discussion.
- Hyperparameters, data and architecture are shared with the original Pythia.
- So the six models are six training runs that differ in init and data order. Seed effects
  cannot be split into init vs data order.
- Whether standard and deduped Pythia share an initialisation seed was **not verified**.
  This matters for §5.

## 3. Seed4 diagnosis

| Question | Finding | Status |
|---|---|---|
| Correct weights loaded? | Loaded file tensor-identical to the official step143000 `.bin`; sha256 matches the Hub | Verified |
| Actually step 143000? | Commit `b122bdb906` = branch `step143000`; `main` holds the same weights | Verified |
| Tokenizer/config identical? | Same config (bar dropout/version), identical vocabulary | Verified |
| IOI evaluation working? | The other 5 models score 198–200/200 on the same items with the same code | Supported |
| Genuinely poor checkpoint? | PolyPythias §5: 410M seeds 3 and 4 are the only runs with loss spikes; they enter a training state early, fail to reach the final state, and score worse downstream | Supported by source |
| Broader problem or only IOI? | 84/200 (below chance, mean logit difference −0.14), D_B to every other seed 0.27–0.37 vs 0.08–0.13 among the others; broad LM check pending | Partly verified |

The paper's HTML renders digits twice ("410M410", "seed 33 and 44"). I read this as 410M
seeds 3 and 4; check the PDF to confirm.

**Conclusion.**

- Seed4 is excluded by the inclusion rule drafted in PREREGISTRATION §2 **before** this run
  (accuracy must beat chance, one-sided binomial, α = 0.05). Its p-value is ≈ 0.99.
- It is excluded for that rule, not for being inconvenient. It stays in the record, and it
  is reported in every table below.
- Seed3 is also flagged as an outlier by PolyPythias, but passes inclusion (200/200). It is
  kept, with its higher noise floor noted (Control B p95 0.243 vs 0.12–0.17).

**Pending (run on the Mac).** `seed_diagnostics.py` covers:

- tensor equality for seeds 1, 2, 3 and 5;
- reproducing the pilot's IOI results through the fixed loader;
- name-probability mass and top-1-is-a-name rate;
- counterfactual-probe accuracy;
- mean next-token loss on 20 self-written sentences (a task-independent sanity check).

## 4. Matching-rule audit

Candidates. The decision is made on statistical grounds only; D_M does not enter it.

| Rule | Error control | Behaviour on these data | Verdict |
|---|---|---|---|
| Point estimate \|Δacc\| ≤ 2 pp | none | 10 of 15 pairs matched | Rejected: no sampling error control |
| Paired t-TOST (as pre-registered) | asymptotic | **Non-monotone.** 2 discordant items (std\|seed1): not matched, p = 0.079. 3 discordant (seed1\|seed2): matched, p = 0.043 | Rejected: invalid approximation for sparse binary data |
| Exact discordance bound (Amendment 1 fallback) | exact, conservative | 3 of 15 matched (only 0-discordant pairs) at n = 200 | **Proposed for all cases** |

- **Why the exact rule.** |acc_A − acc_B| ≤ discordance rate, and the one-sided
  Clopper–Pearson bound gives a valid level-α test at any n. It is monotone, and it is
  already the pre-registered fallback, so this removes the t/exact split rather than adding a
  new method.
- **Cost.** It is conservative. The confirmatory item count must be raised: matched iff
  discordant ≤ 4 at n = 500, ≤ 12 at n = 1000, ≤ 29 at n = 2000.
- **Proposed Amendment 2** (to be recorded before any confirmatory analysis):
  - D_P matching uses `exact_paired_equivalence` for every pair.
  - δ = 2 pp is unchanged.
  - Confirmatory n is set by power analysis.
  - The function exists and is tested but is not yet wired into `performance_distance`.

## 5. D_M audit

**Definition.** D_M = 1 − Spearman ρ between two 384-vectors of mean per-head Taylor
attribution. It is computed at the last position, with name-swap corruption, averaged over
items. There is no normalisation; ranks make it scale-free. 1 means uncorrelated; values
above 1 mean negatively correlated.

**Construct-validity failure: head labels are arbitrary across independent runs.**

- Heads within a layer are interchangeable. Permuting them leaves the model's function
  unchanged.
- The primary D_M compares head (l, h) with head (l, h). That is only meaningful when models
  share head correspondence (checkpoints of one run).
- Relabelling null (primary D_M between a model and itself with heads shuffled within
  layers; 200 shuffles, 5th–95th percentile):

| Model | Split-half noise floor p95 (primary) | Relabelling null p05–p95 (primary) | Split-half floor p95 (label-free) |
|---|---|---|---|
| std | 0.141 | 0.762–0.929 | 0.041 |
| seed1 | 0.154 | 0.859–1.049 | 0.043 |
| seed2 | 0.118 | 0.819–1.011 | 0.038 |
| seed3 | 0.243 | 0.727–0.872 | 0.061 |
| seed4 | 0.289 | 0.814–0.971 | 0.069 |
| seed5 | 0.174 | 0.805–0.995 | 0.054 |

- Seed pairs have primary D_M 0.80–1.11, **inside the relabelling null**.
- So the primary D_M cannot distinguish "different mechanism" from "same model with
  relabelled heads" for these pairs.
- The earlier "divergent" flags for seed pairs are uninformative. Control B (prompt
  resampling) was the wrong null for cross-run comparisons.
- This also qualifies the standard-vs-deduped result (0.75–0.79). That sits at or just below
  the standard model's relabelling-null 5th percentile. Its interpretation depends on whether
  the two runs share an initialisation (§2, unverified).

**Label-free diagnostic.**

- `sorted_within_layer_dm` sorts each layer's head values before the rank correlation.
- It is invariant to within-layer relabelling. It approximately lower-bounds any
  label-invariant version of D_M, because sorting aligns heads optimistically.
- It discards which head does what.

| Pair type | n | Primary D_M | Label-free D_M | Its noise floor (p95) |
|---|---|---|---|---|
| Same run, different checkpoint (6-model pilot) | 6 | 0.049–0.125 | 0.016–0.040 | — |
| Standard vs deduped (6-model pilot) | 9 | 0.750–0.785 | 0.131–0.145 | — |
| Seed pairs, excluding seed4 | 10 | 0.936–1.112 | 0.092–0.262 | 0.038–0.061 |
| Seed pairs with seed4 | 5 | 0.804–0.983 | 0.116–0.145 | ≤ 0.069 |

- Every seed pair's label-free D_M exceeds the larger of the two models' label-free
  split-half floors.
- It is stable on held-out prompts: between-model distance on disjoint halves matched the
  full-data value within ±0.01 for every pair.
- Descriptive reading: independently trained runs differ in how attribution magnitude is
  distributed within and across layers. That is weaker than "different mechanisms", and is
  not tested inferentially here.

**Other checks.**

- **Negligible heads.** About 300 of 384 heads have |attr| < 0.01. Within a run they are
  stable (low split-half floor). The label-free diagnostic is also affected by their ranks.
- **Confidence confounds** (non-seed4 pairs, n = 10, non-independent, descriptive Spearman):
  - D_B vs |Δ mean LD| 0.01; D_B vs label-free D_M 0.36;
  - label-free D_M vs |Δ mean LD| −0.33;
  - primary D_M vs |Δ mean LD| 0.28.
  - No strong confidence confound is visible; n is too small to rule one out.
- **Attribution scale differs across runs** (L1 norm 9.1–17.8, seed4 4.0). Scale does not
  enter Spearman, but it does affect which heads clear any magnitude threshold.

## 6. Dependence structure

- **Experimental unit:** the training run (model). The 15 pairs from 6 runs are not 15
  replications; every pair shares a model with 8 others.
- **Inference must resample or permute models.** For example, H1 by permutation of run
  labels between "matched" and null sets, with CIs from a model-level bootstrap, as
  pre-registered.
- **Three questions, kept separate:**
  - **(A)** Do independently trained, performance-matched runs differ in attribution
    patterns? Not testable with the primary D_M across runs (§5). It needs a label-invariant
    or head-matched D_M, fixed before testing.
  - **(B)** Is the measured distance above the measurement floor? The relevant null for
    cross-run pairs must include label arbitrariness, not only prompt resampling.
  - **(C)** Does D_B relate to D_M? D_B is output-only and label-free, so it is valid.
    With 5 usable runs (10 dependent pairs) no inference is possible.

## 7. Descriptive pilot results

"exact" = exact discordance bound; "t" = implemented t-TOST (—: SD = 0, decided by the exact
bound).

| Pair | Acc A | Acc B | Δacc | Disc. | Matched (exact) | Matched (t) | D_M | Label-free D_M | D_B | Control-B thr. | Divergent (draft) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| std\|seed1 | 200 | 198 | +1.0 pp | 2 | no | no | 1.001 | 0.196 | 0.080 | 0.154 | yes |
| std\|seed2 | 200 | 199 | +0.5 pp | 1 | no | yes | 1.041 | 0.190 | 0.109 | 0.141 | yes |
| std\|seed3 | 200 | 200 | 0.0 | 0 | yes | — | 0.985 | 0.262 | 0.127 | 0.243 | yes |
| std\|seed4 | 200 | 84 | +58.0 pp | 116 | no | no | 0.913 | 0.145 | 0.372 | 0.289 | yes |
| std\|seed5 | 200 | 200 | 0.0 | 0 | yes | — | 0.989 | 0.130 | 0.088 | 0.174 | yes |
| seed1\|seed2 | 198 | 199 | −0.5 pp | 3 | no | yes | 0.936 | 0.092 | 0.107 | 0.154 | yes |
| seed1\|seed3 | 198 | 200 | −1.0 pp | 2 | no | no | 0.963 | 0.178 | 0.121 | 0.243 | yes |
| seed1\|seed4 | 198 | 84 | +57.0 pp | 114 | no | no | 0.983 | 0.141 | 0.338 | 0.289 | yes |
| seed1\|seed5 | 198 | 200 | −1.0 pp | 2 | no | no | 1.087 | 0.155 | 0.082 | 0.174 | yes |
| seed2\|seed3 | 199 | 200 | −0.5 pp | 1 | no | yes | 0.980 | 0.128 | 0.105 | 0.243 | yes |
| seed2\|seed4 | 199 | 84 | +57.5 pp | 115 | no | no | 0.930 | 0.120 | 0.275 | 0.289 | yes |
| seed2\|seed5 | 199 | 200 | −0.5 pp | 1 | no | yes | 1.112 | 0.148 | 0.094 | 0.174 | yes |
| seed3\|seed4 | 200 | 84 | +58.0 pp | 116 | no | no | 0.804 | 0.143 | 0.270 | 0.289 | yes |
| seed3\|seed5 | 200 | 200 | 0.0 | 0 | yes | — | 0.996 | 0.222 | 0.107 | 0.243 | yes |
| seed4\|seed5 | 84 | 200 | −58.0 pp | 116 | no | no | 0.916 | 0.116 | 0.290 | 0.289 | yes |

| Subset | Pairs | Primary D_M | Label-free D_M | D_B |
|---|---|---|---|---|
| All | 15 | 0.804–1.112 | 0.092–0.262 | 0.080–0.372 |
| Matched (exact rule) | 3 | 0.985–0.996 | 0.130–0.262 | 0.088–0.127 |
| Involving seed4 | 5 | 0.804–0.983 | 0.116–0.145 | 0.270–0.372 |
| Not involving seed4 | 10 | 0.936–1.112 | 0.092–0.262 | 0.080–0.127 |

The "divergent (draft)" column uses the primary D_M. Per §5 it carries no information for
these pairs and must not be cited.

## 8. Unresolved

1. **Tensor equality for seeds 1, 2, 3 and 5**, and the broad-LM and behaviour checks for all
   six models. Run `python experiments/v6/audits/seed_diagnostics.py
   experiments/v6/runs/pilot_410m_seeds`.
2. **Amendment 2** (exact matching rule for all pairs; confirmatory n from a power
   analysis). Needs your approval before it is recorded.
3. **Amendment 3: how D_M handles head correspondence.** Options, to be chosen on validity
   grounds before any test:
   - (a) restrict the head-indexed D_M to pairs with shared initialisation, and drop
     cross-seed H1;
   - (b) a head-matched D_M, matching heads on held-out prompts by a feature independent of
     the attribution being compared (e.g. attention-pattern or output similarity), then
     computing D_M on matched heads;
   - (c) a label-invariant summary (layer profile or sorted-within-layer), accepting the loss
     of head identity.
   - Also: define the cross-run null (relabelling null plus prompt resampling).
4. **Whether standard and deduped Pythia share an initialisation**, which decides how the
   6-model pilot is read.
5. **H1 as currently worded** ("matched performance ≠ matched mechanism") cannot be tested
   across seeds with the pre-registered D_M. Either reformulate it around (3) or restrict its
   scope. This decision must precede any confirmatory run.
6. **ε_det, the D_B–confidence relationship, and the Control D known-positive (Arm B)** are
   all still open.

## Tests added for failure modes found in this audit

| Failure mode | Test |
|---|---|
| Seed loader silently used a bot-conversion PR | `test_seed_variant_loads_revision_and_passes_hf_model` (asserts `use_safetensors=False`) |
| No record of the exact commit loaded | `test_hub_provenance_resolves_exact_commit`, `test_hub_provenance_offline_is_recorded_not_fatal`, pilot record assertion |
| Head-indexed D_M ≈ 1 for a relabelled copy of the same model | `test_head_indexed_dm_cannot_distinguish_a_relabelled_model`, `test_relabelling_preserves_each_layer_multiset`, `test_sorted_within_layer_dm_is_zero_for_relabelled_self` |
| t-TOST decisions non-monotone in discordance | `test_t_tost_is_not_monotone_in_discordance_regression`, `test_exact_equivalence_is_monotone_in_discordance`, `test_exact_equivalence_matches_clopper_pearson` |
| Below-chance model must fail inclusion | `test_seed4_score_fails_inclusion_regression` |
