# V7 Experiment 3: noise-aware group diff. Can it tell a real change from run-to-run noise?

*Protocol v1.0. Written 2026-10-05, BEFORE the group-diff code exists and before any run.
It is committed first and not changed afterwards. Deviations go in dated addenda below
the line, never by editing this text.*

## 1. Question

The existing diff (`glassbox.v7.diff`) compares **one** run with **one** run. In a
system that samples (an LLM at temperature > 0, a flaky tool call), two runs with the
same configuration already differ. The pairwise diff then reports a divergence that is
only noise.

This experiment tests a new **group diff**, which compares *N* baseline runs with *M*
candidate runs. It reports a step only when that step's behaviour differs **beyond the
run-to-run variation**. The questions:

1. With no real change, does it stay silent at the declared error rate (Type I)?
2. With one injected change, does it put the first finding beyond noise at the right
   step?
3. On a real sampling LLM (GPT-2 small), does it stay silent between two groups of good
   runs and localise a known retrieval fault?

It also measures, without gating, two limitations that are expected in advance:
- how often the *pairwise* diff fires on pure noise;
- whether the group diff can tell **one** upstream cause from **two** causes. It is
  expected **not** to, which is why V3 (controlled re-runs) is needed.

## 2. The method under test (fixed before implementation)

Module `glassbox/v7/groupdiff.py`, standard library only (like `trace.py` and
`diff.py`). Input: two lists of trace files, group A (baseline) and group B (candidate).
All traces are integrity-checked on read.

1. **Step keys.** Each non-root span gets the key `<signature>#<k>`.
   - The signature is `glassbox.v7.diff.step_signature`.
   - `k` is the occurrence count of that signature earlier in the same run, starting
     at 0.
2. **Fields per step key.**
   - `present` (True/False).
   - `status` (status code).
   - Each comparable attribute: the same exclusions as `diff._comparable` (volatile
     keys, raw content, `error.type`).
   - Each content hash `glassbox.content.<attr>.sha256`.
   - If a step is absent in a run, its fields have the value `None` in that run.
3. **Classification of each field.** Values are pooled across A ∪ B; *n* is the total
   number of runs.
   - **constant**: one distinct value. Not tested; no divergence.
   - **untestable (high cardinality)**: more than *n*/2 distinct values. Typical case:
     the hash of sampled free text, which is unique per run. Reported, not tested.
     *Why:* with nearly all values unique, a difference between groups cannot be
     separated from noise, whatever the cause.
   - **tested**: everything else.
4. **Statistic and test.**
   - The statistic is the total-variation distance between the A and B categorical
     distributions of the field.
   - The p-value comes from a permutation test: pool A ∪ B and reshuffle the group
     labels `n_perm` times with a fixed seed. p = (1 + #{TVD_perm ≥ TVD_obs}) /
     (1 + n_perm).
5. **Multiple testing.** Holm–Bonferroni across all tested fields of one comparison, at
   α = 0.05.
6. **Noise baseline (reported).** For every field, whether it varies **within** group A.
7. **Output.**
   - The significant (step key, field) pairs, ordered by run position. A step's run
     position is its mean `glassbox.step.index` over the runs where it is present.
   - **`first_beyond_noise`**: the earliest significant pair.
   - **All** significant steps, not just the first. This addresses the concern that a
     first divergence hides a second.
   - The untestable fields.
   - Root-span (environment) differences, which are kept separate as in `diff.py`.
   - A CLI exits 1 if anything is significant.

**Scope of the claim, as for `diff.py`.** A significant step says *where the groups'
behaviour differs beyond noise*. It does **not** say why, or how many causes there are.

## 3. Part A: synthetic pipeline, ground truth known by construction

A script writes real Glassbox traces from a three-step stochastic pipeline (stdlib
`random`, seeded). Each run:

1. **`retrieval`**: retrieves `right` with probability *r*, otherwise `wrong`. Recorded
   as hashed `gen_ai.retrieval.documents`.
2. **`chat`**: answer label `correct` with probability *c_r* if `right` was retrieved,
   *c_w* if `wrong`; otherwise `distractor`.
   - Recorded as the attribute `glassbox.answer.label`.
   - Also records a hashed free-text output containing a per-run random nonce, which
     is unique per run (a stand-in for sampled text).
3. **`execute_tool:send_confirmation`**: present only if the label is `correct`. This
   gives a structural `present` signal downstream.

**Baseline (group A, every world):** r = 0.90, c_r = 0.85, c_w = 0.20.

| World | Group B differs by | Ground truth |
|---|---|---|
| W0 null | nothing (same parameters, new seeds) | no real change |
| W1 chat cause | c_r = 0.35 | change starts at `chat` |
| W2 retrieval cause | r = 0.40 | change starts at `retrieval` (it propagates to `chat` and the tool step) |
| W3 two causes | r = 0.40 **and** c_r = 0.35 | changes at `retrieval` and at `chat` |

- **Group sizes:** N = M = 30 runs.
- **Replications:** R = 200 per world, with independent seeds derived from master seed
  `20261005`.
- **Permutations:** `n_perm` = 999 in Part A.

## 4. Part B: real sampling model (GPT-2 small)

This part reuses Experiment 1's corpus (`experiments/v7/rag_fault/corpus.json`) and its
deterministic bag-of-words retriever with top_k = 1, unchanged.

**Generator.**
- GPT-2 small via TransformerLens, on CPU.
- The prompt is the same as in Experiment 1.
- It **samples** 8 new tokens at temperature 1.0, with no top-k/top-p, seeded per run
  with `torch.manual_seed(seed)`.

**Recorded per run.**
- Hashed retrieval documents.
- Hashed prompt.
- Hashed sampled output (free text).
- The attribute `glassbox.answer.contains_correct`: whether the question's correct
  answer string (trimmed) occurs in the sample, case-insensitive.

**Conditions** (per question, 8 questions):

| Condition | Index | Seeds |
|---|---|---|
| `baseline` | all 8 documents | 0–19 |
| `baseline_more` | all 8 documents | 100–119 |
| `fault` | d1–d4 missing (the stale index from Experiment 1) | 200–219 |

- **Permutations:** `n_perm` = 1999 in Part B.
- **Affected and unaffected questions:** as in Experiment 1. q1–q4 are affected, q5–q8
  unaffected. Experiment 1 showed that retrieval for q5–q8 is unchanged under the
  fault.

## 5. Pre-declared criteria

### Part A (synthetic)

| ID | Criterion | Pass rule |
|---|---|---|
| A1 Type I | W0: fraction of replications with ≥1 significant field | ≤ 0.08 (α = 0.05 plus the Monte-Carlo margin 1.96·√(0.05·0.95/200) ≈ 0.03) |
| A2 Chat localisation | W1: `first_beyond_noise` is at `chat#0` | ≥ 0.80 of replications |
| A2b No false upstream flag | W1: any field of `retrieval#0` significant | ≤ 0.08 of replications |
| A3 Retrieval localisation | W2: `first_beyond_noise` is at `retrieval#0` | ≥ 0.80 of replications |
| A4 Free text untestable | The free-text output hash is classified *untestable* | 100% of replications, all worlds |
| A5 One vs two causes (**descriptive**) | Rate at which `chat#0` is significant in W2 and in W3 | Reported. Expected: high in both, i.e. the group diff **cannot** distinguish one upstream cause from two |
| A6 Pairwise noise (**descriptive**) | W0: fraction of pairs (A_i, B_i) where `diff_traces` reports ≥1 divergence | Reported. Expected: near 1 |

### Part B (GPT-2 sampling)

| ID | Criterion | Pass rule |
|---|---|---|
| B1 Real null | `baseline` vs `baseline_more`: no significant field | ≥ 7/8 questions (with FWER 0.05 per question, P(≥2 false alarms in 8) ≈ 0.06) |
| B2 Real localisation | Affected q1–q4, `baseline` vs `fault`: `first_beyond_noise` at `retrieval#0` | 4/4 |
| B3 Real specificity | Unaffected q5–q8, `baseline` vs `fault`: no field of `retrieval#0` significant | 4/4 |
| B4 Pairwise noise (**descriptive**) | Fraction of pairs (`baseline`_i, `baseline_more`_i) where `diff_traces` reports ≥1 divergence; `contains_correct` rate per condition | Reported |

**Overall result.**
- **PASS** iff A1, A2, A2b, A3, A4, B1, B2 and B3 all pass.
- A5, A6 and B4 are descriptive.
- A FAIL is reported as a FAIL. The method, thresholds, seeds and worlds are not changed
  after any run to obtain a PASS.

## 6. What a PASS would and would not show

**Would show:**
- On these pipelines, the group diff separates real behavioural change from run-to-run
  noise at the declared error rate.
- It localises a single injected change, and stays silent between two groups of good
  runs of a real sampling LLM.

**Would not show:**
- That it finds unknown causes in real systems (V2.4, Hard Gate 3).
- That the first finding beyond noise is the cause. V2.2 stays `NOT_SUPPORTED`; A5 is
  expected to demonstrate exactly this limit.
- Any result for small groups. The power at N < 30 is not measured.
- Anything about numeric attributes, which are treated as categorical in v1. This will
  be stated in the module docstring.

## 7. Execution

1. Commit this protocol.
2. Implement `glassbox/v7/groupdiff.py` test-first, with unit tests on hand-built traces
   only. Part A and Part B are **not** run during development.
3. Commit the code and the experiment scripts.
4. Run on the owner's Mac:
   - Part A: `python3 experiments/v7/noise_baseline/part_a.py --out experiments/v7/noise_baseline/results/part_a`
   - Part B: `python3 experiments/v7/noise_baseline/part_b.py --out experiments/v7/noise_baseline/results/part_b`
5. Commit `results.json` files and Part B traces unchanged. Part A traces are written to
   a temporary directory and not committed: there are 200 × 4 × 60 = 48,000 of them,
   and they are regenerable from the seeds. Their counts and seeds are recorded in
   `results.json`.

---
*(Addenda, if any, go below this line, with dates.)*
