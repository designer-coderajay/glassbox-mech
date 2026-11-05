# V8 phase E1b: directed attribution and a hurt-only fix. PROTOCOL

*Protocol v1.0, written 2026-11-05 before any E1b code, data or run. It is committed
first and not changed afterwards; any deviation goes in a dated addendum at the end.
Pre-registration order: this protocol, then code and tests (with the new corpus,
questions and the sealed test-split hash), then the run, then results committed
unchanged, then RESULTS.md and the ASSURANCE row.*

## 0. Why E1b exists, and what it is not

E1 (`experiments/v8/e1_golden_rag/`, ASSURANCE V3.10) **failed**: the fix was
verified on 9 of 13 questions, against a bar of 0.80. Its RESULTS §7 lists four
post-hoc observations:
1. attribution has no direction, and 3 of 16 large effects *raised* accuracy;
2. the fix reverted every attributed factor, including changes that helped;
3. small effects add up;
4. the two-sided ±0.15 band rejected fixes that beat a weak baseline.

E1b tests a changed method that answers points 1, 2 and 4, on **new data**. Point 3 is
measured, not designed around.

- **E1's verdict stays FAIL.** E1b does not re-score E1, and E1 data is not reused for
  any gated number.
- **Known risk: criteria shaped by a seen failure.** The fix rule and the recovery
  check were chosen after seeing E1's results. Two things limit how much that can
  flatter the result:
  - E1b runs on a new corpus and a new sealed test split, written after this protocol;
  - E1's two-sided check is reported for every question as a descriptive number.
  The risk is still disclosed in RESULTS.

## 1. Question

On a new multi-change RAG regression, does a **directed** investigation (which changes
*hurt*) yield a fix that restores the system? Does it keep changes that helped, clear
the visible and inert changes, reject a wrong fix, and beat the scripted baselines on
the same run budget?

## 2. System (unchanged from E1 except the data)

- The E1 pipeline, models, factors, sampling settings, outcome matching, competence
  ladder and gate are reused as written in E1 PROTOCOL §2–3 and Addendum 1 items 2–3:
  - Qwen2.5-1.5B-Instruct → 7B ladder;
  - MiniLM embedder, top-k 3;
  - temperature 0.7, at most 48 new tokens, blocks of 50.
  The code is copied into the E1b folder, and only data paths and seeds change.
- **New data**, written in the code commit:
  - a new fictional company with about 24 short policy documents, v1 and v2. About 8
    documents change in v2, and none are reused from E1;
  - 32 new questions with accepted and stale answer lists: 8 dev (4 updated,
    4 unchanged) and 24 test (12 + 12).
- The test file's sha256 is sealed in Addendum 1 before any model run. As in E1, the
  test split is not read during development, except by the toy dry run.
- **The author writes the data**, after having seen E1's per-question results. That is
  a source of design bias, and it is disclosed.
- **Factors (as E1):**

  | Factor | Baseline | Candidate |
  |---|---|---|
  | `index` | fresh (v2) | stale (v1) |
  | `chunk_words` | 120 | 40 |
  | `template` | T1 | T2 |
  | `log_level` | info | debug (inert by construction) |

- **Seeds:** `e1b-main`, `e1b-reference`, `e1b-recovery`, `e1b-gate`.

## 3. The method change (library, tests first)

`glassbox.v7.intervene.InterventionReport` gets a direction-aware view. Its existing
outputs (`attributed`, `evidence`, tests) are unchanged. Given a target outcome value
(here `"correct"`) and the rate `p(cell)` of that value in a cell:

- sufficiency delta: `d_suff(f) = p(only_f) − p(baseline)`;
- necessity delta: `d_nec(f) = p(all_but_f) − p(candidate)`.

A factor is **harmful** iff
- its sufficiency test is significant and `d_suff(f) < 0`, **or**
- its necessity test is significant and `d_nec(f) > 0`.

The tests are the existing permutation tests, with Holm within each family, as now.
A factor is **helpful** iff its sufficiency test is significant and `d_suff(f) > 0`, and
it is not harmful.

`investigate` reports the harmful and helpful lists next to `attributed`, and its
plain-language report states the direction. It still never says "caused".

**GB's blame set** in E1b is the harmful list. **GB's fix** reverts every harmful
factor. If the harmful list is empty, there is no fix, and the question counts as **not
verified** wherever it is in G5's denominator.

## 4. Ground truth (reference run)

- Separate seed `e1b-reference`. 200 samples per cell per test question. The cells are
  `baseline`, `candidate` and each `only_<f>` (6 cells; E1 had no `candidate` cell).
- The outcome is binary, so TVD equals the absolute accuracy difference. Its 95%
  percentile bootstrap CI uses 2,000 resamples.
- **Pair labels** (question, factor):
  - **harmful-large:** CI lower bound ≥ 0.20, and `only_f` accuracy < baseline accuracy;
  - **helpful-large:** CI lower bound ≥ 0.20, and `only_f` accuracy > baseline accuracy;
  - **null:** CI upper bound ≤ 0.10. `log_level` is null by construction, as in E1;
  - otherwise **small/uncertain**: reported, not scored.
- **Regressed question:** baseline accuracy minus candidate accuracy has a 95% bootstrap
  CI lower bound ≥ 0.20.
- **Retrieval truth** (exact, no sampling): whether a chunk containing an accepted
  answer is in the top-3 under each configuration. Reported, not scored. E1 omitted
  this; E1b implements and tests it.

## 5. Methods compared (same main-run budget as E1)

The cells of `run_interventions` at **30 runs per cell**, α = 0.05, 999 permutations,
seed `e1b-main`.

| ID | Method | Blame set |
|---|---|---|
| **GB** | `investigate` with the §3 direction view | harmful list |
| **B0** | Blame every change between releases | all four factors |
| **B1** | One run per side plus the trace diff, first divergence (as E1) | retrieval → `index`, `chunk_words`; generation → `template` |
| **B2** | Revert one change at a time, one run each (as E1) | factors whose single reverted run is correct while the candidate run is wrong |

Not tested, as in E1: LLM-judge analysis and commercial tools (E6/E7), and a human
workflow (E4, blind).

## 6. Fix and recovery

- **Recovery rerun:** seed `e1b-recovery`, 60 runs per arm, every test question. The
  arms are:
  - `baseline` and `candidate`;
  - `gb_fix`: candidate with GB's harmful factors reverted;
  - `wrong_fix`: candidate with `template` only reverted (the visible suspect);
  - `b2_fix`: candidate with B2's blamed factors reverted.

  An arm whose configuration equals one already run on that question reuses those runs.
  Arms with nothing to revert are not run.
- **Verified recovery** of a fix on a question means both:
  1. **better than the candidate:** the fix's correct-rate is higher, with a one-sided
     permutation p < 0.05;
  2. **not worse than the baseline:** the lower bound of the 95% bootstrap CI of
     (fix − baseline) accuracy is ≥ −0.15.

  A fix may exceed the baseline. E1's two-sided check is also reported for every
  question, descriptively.
- **Wrong fix rejected** on a question means `wrong_fix` is **not** verified by the same
  rule.

## 7. Criteria (test split; gated)

| ID | Criterion | Pass rule |
|---|---|---|
| G1 | Sensitivity to harm | GB's harmful list contains ≥ 0.75 of the **harmful-large** pairs |
| G2 | Inert never blamed | `log_level` in the harmful list on ≤ 2 of 24 questions |
| G3 | Visible suspect cleared | Among `template` pairs labelled **null**, ≤ 0.10 in the harmful list. Not testable below 4 pairs |
| G4 | Helpful changes kept | Among **helpful-large** pairs, ≤ 0.10 in the harmful list. Not testable below 4 pairs |
| G5 | Fix verified | GB's fix verified on ≥ 0.80 of **regressed** questions |
| G6 | Wrong fix rejected | Wrong fix rejected on ≥ 0.80 of regressed questions where `template` is not harmful-large |
| G7 | Better than today's workflow | F1 over harmful-large (positive) and null (negative) pairs: GB exceeds each of B0, B1 and B2 by ≥ 0.10 |

- **Overall PASS** iff:
  - the generator passed the competence gate;
  - every testable criterion passes;
  - there are at least 6 harmful-large pairs;
  - there are at least 6 regressed questions.
- **INCONCLUSIVE** if either count is below 6 (the case is too weak to judge) or no
  rung passes the gate.
- **FAIL** otherwise, with every failing number reported.
- Testability is decided only by the counts above. It is reported, not chosen.

**Descriptive (not gated):**
- per-method misses and false alarms, with Wilson 95% CIs;
- B2's fix verified rate on regressed questions, next to GB's;
- E1's two-sided recovery check for every fix arm;
- small/uncertain pairs, and how often a fix fails while a small factor is left in place;
- the undirected `attributed` list, scored as in E1, for comparison;
- B2's re-draw flips (5 extra seeds);
- `log_level` sampled TVDs;
- retrieval truth;
- runtime per stage, and the number of generations.

## 8. Compute and execution

- **Where:** Colab T4, the same venv recipe as E1 (`docs/v7/COLAB_GPU.md`), float32,
  checkpointed per question on Drive. A resume is allowed and disclosed.
- **Estimate** (not a commitment): reference 24 × 6 × 200 = 28,800 generations; main
  7,200; recovery at most 24 × 5 × 60 = 7,200. About 43,000 generations, likely 4–5 h
  on a T4 for rung 1. E1 took 3.2 h for 37,000.
- **Order:** gate → per question (reference, main, recovery) → scoring. Main-run
  decisions never read reference labels; labels are joined only in scoring.
- Scoring code and its tests are committed before the run.
- `uv pip freeze` and the per-investigation timings are saved inside `--out`, so
  nothing is lost in temporary directories.

## 9. What a PASS would and would not show

- **Would show:** on a second author-built RAG system, with one generator and one
  embedder, a direction-aware investigation produced a fix that restored the regressed
  questions. It kept changes that helped, cleared the visible and inert changes,
  rejected the wrong fix, and beat scripted versions of common practice on the same run
  budget.
- **Would not show:**
  - anything blind: the author designed the causes and the data (E4);
  - generalisation beyond this pipeline, factor set and model;
  - anything against commercial tools (E6/E7);
  - that engineers would use it (phase G);
  - that E1 passed.
- A FAIL is reported as a FAIL. Only a PASS makes E1b the golden case, and the E1 FAIL
  is always shown next to it.

---
*(Addenda go below this line, with dates.)*
