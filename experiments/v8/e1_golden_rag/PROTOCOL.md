# V8 phase E1: golden RAG case. PROTOCOL

*Protocol v1.0, written 2026-11-04 before any code, corpus or run. It is committed
first and not changed afterwards; any deviation goes in a dated addendum at the end.
Pre-registration order: this protocol, then code and tests (with corpus, questions and
the sealed test-split hash), then the run, then results committed unchanged, then
RESULTS.md and the ASSURANCE row.*

## 1. Question

A realistic RAG assistant gets worse after a release in which several things changed
at once. Can the investigation
- name the change or changes that caused the drop,
- clear a visible but harmless edit (the change people blame first),
- never blame an inert change,
- verify the fix by re-running,
- reject a wrong fix,

more reliably than the scripted versions of what engineers do today, on the same run
budget?

**Why this case.**
- It follows the input-V8 P0 list ("stale RAG index"; "prompt/retrieval regression").
- It follows a public scan of real incidents (`private/discovery/DESK_SCAN_RAG.md`,
  n = 6): 5 of 6 involved a retrieval/index-side change, and in at least 2 the cause
  was disputed or bundled with other changes.
- **It is not interview-derived.** No D0 interview had taken place when this was written.

## 2. System (fixed now; built in the code commit)

**Corpus.**
- About 24 short policy documents for a fictional company. They are written for this
  experiment and committed, so there are no licence issues.
- Two versions exist. v2 updates the facts in about 8 documents (for example a refund
  window or a fee); v1 is the old text.

**Questions.**
- 32 questions, each with a short gold answer string. Half ask about facts that changed
  in v2 ("updated"); half ask about facts that did not ("unchanged").
- Split: **dev** = 8 questions (4 updated, 4 unchanged), used to build and debug the
  pipeline and for the competence gate. **Test** = 24 questions (12 + 12), used for all
  gated criteria.
- The test file's sha256 is written in Addendum 1 before any run. The test answers are
  never used while developing.

**Pipeline.** Each step is traced with the Glassbox `Recorder`; hashed content only.
1. Chunk documents into chunks of `chunk_words` words.
2. Embed chunks and query with **sentence-transformers/all-MiniLM-L6-v2**
   (Apache-2.0, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`).
3. Retrieve the `top_k` = 3 chunks by cosine similarity. Retrieval is deterministic.
4. Build the chat prompt from `template` (context plus question).
5. Generate with the generator model:
   - sampling at temperature 0.7, top-p 1.0, at most 48 new tokens;
   - float32 for weights that fit, otherwise the generator rung's stated precision;
   - one seed per run, from `glassbox.v7.intervene.run_seed`.

**Outcome.** `correct` if the normalised gold answer string appears in the normalised
output, otherwise `wrong`. Normalisation is lowercase, collapsed whitespace and
stripped punctuation, defined in code before any run.

**Generator: competence-gated ladder** (licences checked on the Hugging Face API,
2026-11-04):

| Rung | Model | Licence | Revision | Precision |
|---|---|---|---|---|
| 1 | Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | float32 |
| 2 | Qwen/Qwen2.5-7B-Instruct | Apache-2.0 | `a09a35458c702b33eeacc393d103063234e8bc28` | 4-bit (bitsandbytes; licence checked in the code commit) |

- Qwen2.5-3B-Instruct is **excluded**: its licence field is "other", not Apache-2.0.
- **Competence gate:** baseline configuration, dev questions, 10 samples each. The
  first rung with accuracy ≥ 0.80 is used for everything. If neither rung passes, E1 is
  **INCONCLUSIVE** (no gated criterion is scored) and the gate results are reported.
- The gate is run once and its numbers are recorded.

## 3. The incident

| Factor | Baseline (release 1, working) | Candidate (release 2) | Role |
|---|---|---|---|
| `index` | built from current docs (v2) | **stale**: built from v1, not rebuilt after the update | expected real cause on "updated" questions |
| `chunk_words` | 120 | 40 | re-chunking; effect unknown in advance (measured) |
| `template` | T1: "Use the context to answer the question." | T2: T1 plus "Answer concisely in one sentence." | **visible suspect**; effect unknown in advance (measured) |
| `log_level` | info | debug | **inert by construction**: the code never reads it outside logging |

The exact template strings are committed in the code commit, before any run.

## 4. Ground truth (reference run)

With sampling there is no exact answer, so the truth comes from a separate, larger run:
- **Reference run.** Separate master seed `e1-reference`; 200 samples per cell per test
  question. Cells: baseline and each "only factor X switched" (5 cells).
- **True effect** of X on question q: TVD(only_X, baseline), with a 95% percentile
  bootstrap CI (2,000 resamples).
- **Labels:**
  - **large** if the CI lower bound ≥ 0.20;
  - **null** if the CI upper bound ≤ 0.10;
  - otherwise **small/uncertain**, which is reported but not scored.
- `log_level` is labelled **null by construction**, whatever its sampled TVD. Its
  sampled TVD is reported as a noise check.
- **Retrieval truth (exact, no sampling):** whether the gold chunk is in the top-k under
  each configuration. This is reported and used to explain results, not to score.

## 5. Methods compared (same run budget per question)

The budget is the cells of `run_interventions`: baseline, candidate, each factor alone,
each factor reverted alone, at **30 runs per cell**. The baselines are scripted versions
of the "existing workflow" (PLAN §11.1). They use runs drawn from the same budget.

| ID | Method | What it blames |
|---|---|---|
| **GB** | `glassbox.v7.investigate` unchanged (group diff + interventions, α = 0.05, Holm across factors, 999 permutations) | factors in `attributed` |
| **B0** | Config diff + plain logs: what changed between releases | every changed factor |
| **B1** | One run per side + trace diff (first divergence), as in Experiment 10 | factors acting at the first divergent step: retrieval → `index`, `chunk_words`; generation → `template` |
| **B2** | Revert one change at a time, **one run each**, as engineers often do | factors whose single reverted run is correct while the candidate run is wrong |

- **Not tested:** LLM-judge trace analysis and commercial tools such as LangSmith Engine
  and Phoenix. They need accounts, API spend or ToS review. They belong to E6/E7.
- **Human workflow:** PLAN E1 says to compare with "plain logs plus a manual diff under
  the same time budget". The author knows the injected causes, so a human comparison
  here could not be fair. **It is done blind in E4 by a helper.** E1 uses the scripted
  baselines above, and this deviation from the plan text is recorded.

## 6. Fix and recovery

- **Fix.** Take the candidate and revert every factor GB attributed on the question.
  Questions with nothing attributed get no fix and are not scored on recovery.
- **Wrong fix (control).** Take the candidate and revert `template` only: the visible
  suspect.
- **Recovery rerun.** Separate master seed `e1-recovery`; 60 runs each of baseline,
  candidate, fixed and wrong-fix.
- **Verified recovery** on a question means both:
  1. fixed vs candidate: correct-rate higher, permutation p < 0.05 (one-sided);
  2. fixed vs baseline: the 95% bootstrap CI of the accuracy difference lies within
     [−0.15, +0.15].
- **Wrong fix rejected** on a question means it is not verified by the same rule.

## 7. Criteria (test split; gated)

The units are (question, factor) pairs, with labels from §4.

| ID | Criterion | Pass rule |
|---|---|---|
| G1 | Sensitivity | GB attributes ≥ 0.75 of the **large** pairs |
| G2 | Inert never blamed | GB attributes `log_level` on ≤ 2 of 24 questions |
| G3 | Visible suspect cleared | Among `template` pairs labelled **null**, GB attributes ≤ 0.10. If there are fewer than 4 such pairs, G3 is **not testable**, it is reported, and it does not count toward PASS. |
| G4 | Fix verified | Verified recovery on ≥ 0.80 of questions where GB attributed something and at least one large pair exists |
| G5 | Wrong fix rejected | Wrong fix rejected on ≥ 0.80 of questions where `template` is not labelled large and some other factor is large |
| G6 | Better than today's workflow | GB's F1 over large/null pairs (attributed = predicted positive) exceeds each of B0, B1 and B2 by ≥ 0.10 |

- **Overall PASS** iff the generator passed the gate, every testable criterion passes,
  and G1 has at least 6 large pairs.
- **FAIL** otherwise, with every failing number reported.
- **If G1 has fewer than 6 large pairs:** INCONCLUSIVE (the case was too easy or too
  weak).

**Descriptive** (not gated):
- per-method misses and false alarms, with Wilson 95% CIs;
- runtime per method;
- GPU hours, from `timing.json` and the environment record;
- number of generations;
- small/uncertain pairs;
- retrieval truth;
- B2's run-to-run instability (re-run its single-run decision with 5 different seeds
  and count flips);
- `log_level` sampled TVDs.

## 8. Compute and execution

- **Where:** Google Colab GPU, following `docs/v7/COLAB_GPU.md`:
  - isolated Python 3.11 venv and pinned stack;
  - float32 for scored rungs that fit;
  - the environment record written to `results.json`.
- **Resume:** results are checkpointed per question and per stage with
  `glassbox.v7.checkpoint` on Google Drive. A resume after a disconnect is allowed and
  is disclosed in RESULTS.md.
- **Rough cost** (an estimate, not a commitment):
  - reference run: 24 × 5 × 200 = 24,000 generations;
  - main run: 24 × 10 × 30 = 7,200;
  - recovery: 24 × 4 × 60 = 5,760.

  This is likely a few GPU hours on a T4 for rung 1, more for rung 2.
- **Order:**
  1. competence gate;
  2. reference run;
  3. main run (GB, B0–B2);
  4. recovery;
  5. scoring.

  Scoring code is committed before stage 1. Reference labels are computed by code and
  are not inspected by hand before the main run's decisions are saved.

## 9. What a PASS would and would not show

- **Would show:** on this author-built RAG system, with one generator and one
  embedder, the investigation found the causes of a multi-change regression, cleared
  the visible suspect and the inert change, verified the right fix and rejected the
  wrong one, and did better than scripted versions of common manual practice on the
  same run budget.
- **Would not show:**
  - anything blind: the author designed the causes (E4 does that);
  - anything about real customer systems, other frameworks or hosted models;
  - anything against commercial tools (E6/E7);
  - that engineers would use it (phase G).
- This becomes the golden case (Space demo, README case study, a benchmark row and
  the interview demo) **only if** the result is reported as it came out. A FAIL is
  shown as a FAIL.

---
*(Addenda go below this line, with dates.)*

## Addendum 1 (2026-11-04): sealed data, implementation details, toy dry run

Written in the code commit, before any model run.

**1. Sealed files** (sha256). These must match the run's checkout. The test split was not
used while building the code. Only the toy backend has read it, in the dry run of item 6.

| File | sha256 |
|---|---|
| `questions_test.json` (24) | `a6dbd869e45567a02f51dab4dc9b402bf3fa3e253275fdb60c84267657acbb09` |
| `questions_dev.json` (8) | `d080f31cdc427b07d3a6995a188f734fbc00ea0168887d189adb38a0f07b9895` |
| `corpus.json` (24 docs, 8 changed) | `ffe0fe2c0196d83564dee45d1e511e1cf1c8e587094b187bc1b9bd704a04fce4` |

- `make_data.py` regenerates all three byte-identically, and a test checks this.
- Documents are 42–70 words long. So at `chunk_words` 120 every document is one chunk,
  and at 40 most become two.

**2. Sampling seeds (clarifies §2 "one seed per run").**
- Generations are drawn in **blocks of 50** per (configuration, question). Each block has
  its own seed: sha256 of (stage master seed, canonical configuration, question id,
  block number).
- Run *i* uses sample *i* of its block. This keeps a model call batched, so the run fits
  on a T4.
- Each configuration still gets independent samples. That includes the configurations
  that differ only in the inert `log_level`.
- The `seed` argument that `run_interventions` passes to the runner is not used by the
  generator.

**3. Outcome matching (§2).**
- Punctuation becomes spaces, and text is lowercased and split into tokens.
- An answer counts as `correct` if any accepted answer appears as a **contiguous run of
  tokens**. For example, "30 days" matches "30 days" and "thirty days" but not "300
  days".
- Accepted and stale answer lists are in the question files.

**4. Order of stages (§8).** The run checkpoints **per question**: reference, then main,
then recovery for each question before moving to the next. The main-run decisions never
read the reference labels. Labels are only joined to decisions in scoring, after all
questions are done.

**5. Software** (licences checked on PyPI, 2026-11-04). These are **not pinned**; the
versions installed are recorded in `results.json` → `software`.
- sentence-transformers: Apache-2.0.
- bitsandbytes: MIT. Needed for rung 2 only.
- accelerate: Apache. Needed for rung 2's `device_map`.

The rest of the stack uses the CI pins (`docs/v7/COLAB_GPU.md`).

**6. Toy dry run (disclosed; not evidence).**
- The full pipeline ran once on the toy backend: hashed bag-of-words embedder and a
  sentence-picking "generator", **not a language model**. It covered all 24 test
  questions at the protocol's sample sizes, on CPU, in about 42 s.
- The toy failed the competence gate (0.338), so the gate was set to 0 for this dry run
  only, to exercise every stage.
- It completed, and scoring produced all six criteria. The toy numbers are not reported
  as results, and **no threshold, setting or data file was changed after it.**

**7. Known before the run.** With n = 200, a `null` label (CI upper bound ≤ 0.10) is
reachable only when the question's accuracy is high, about 0.85 or more. On a 50%
question with no effect, the CI upper bound is about 0.14. So G3 may turn out to be **not testable**, as §7
allows.
