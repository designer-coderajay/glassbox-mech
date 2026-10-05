# V7 Experiment 1 — Controlled RAG fault: does trace + diff localise a known cause?

*Protocol v1.0. Written 2026-10-05, BEFORE any real-model run. It is committed before
execution and is not changed afterwards. Any deviation is reported in a dated addendum
below the line, never by editing this text.*

## 1. Question

When a RAG system's behaviour degrades because of a **known, injected retrieval fault**,
four things are tested:

1. Does the Glassbox trace diff place the **first divergence at the retrieval step**?
2. Is the diff **specific**, i.e. silent for questions the fault cannot affect?
3. Does **undoing** the fault restore the original behaviour (intervention)?
4. Is the pipeline **deterministic**, so that re-running the baseline gives zero
   divergences (reproduction)?

This tests the instrument (trace + diff) on a system whose cause is known by
construction. It does **not** test whether Glassbox finds unknown causes in real
systems. That is V7 Hard Gate 3 and remains open.

## 2. System under test (fixed)

- **Corpus:** `corpus.json`, 8 short documents about fictional entities (no real
  companies or people), plus 8 questions, one answerable from each document.
  - Each question has a **correct** answer (from its source document).
  - Each question also has a **distractor** answer: the same-type answer from another
    document, so it is present in the corpus.
- **Retriever:** deterministic bag-of-words cosine similarity, lower-cased, top_k = 1.
  - Design note, made before any run: with top_k = 2, every question's runner-up was its
    paired document. The fault would then have changed retrieval for all eight
    questions, and C2 could never test silence. With top_k = 1, the unaffected
    questions retrieve the same single document under both indexes; this was checked
    with the retriever, without any model. The affected questions retrieve their
    paired document instead, and that document contains the distractor.
- **Generator / scorer:** GPT-2 small (`gpt2`, via TransformerLens), CPU. The prompt
  is the retrieved documents, then `Question: …`, then `Answer:`.
  - Behaviour is measured by teacher forcing, not sampling: margin = log p(correct
    answer tokens | prompt) − log p(distractor tokens | prompt).
  - A question is **answered correctly** if margin > 0.
  - No sampling is used, so runs are expected to be deterministic on one machine.
- **Trace:** one Glassbox trace per (condition, question), with content policy
  `hash_only`.
  - The retrieval span records the retrieved document ids and scores as
    `gen_ai.retrieval.documents` (hashed).
  - The chat span records the prompt (`gen_ai.input.messages`) and the preferred answer
    (`gen_ai.output.messages`), both hashed.

## 3. Conditions (fixed)

| Condition | Index | Purpose |
|---|---|---|
| `baseline` | all 8 documents | reference |
| `baseline_repeat` | all 8 documents | determinism / reproduction (C4) |
| `fault` | **stale index**: documents d1–d4 missing (simulates a failed re-index) | injected cause |
| `restored` | all 8 documents again | intervention (C3) |

Questions about d1–d4 are **affected** (their source document is gone). Questions about
d5–d8 are **unaffected** by construction.

## 4. Pre-declared criteria

The traces are compared with `glassbox.v7.diff.diff_traces`, per question, against
`baseline`.

| ID | Criterion | Pass rule |
|---|---|---|
| C1 Localisation | For every **affected** question, `fault` vs `baseline` has a first divergence located at the **retrieval** span | 4/4 |
| C2 Specificity | For an **unaffected** question, if the retrieved document set is unchanged then the diff reports **no divergence**; if it changed (a missing document was a runner-up), the first divergence must still be the retrieval span | 4/4 unaffected questions meet the rule that applies to them |
| C3 Intervention | `restored` vs `baseline` has no behavioural divergence for all 8 questions, and an identical correct/incorrect outcome | 8/8 |
| C4 Determinism | `baseline_repeat` vs `baseline` has no behavioural divergence | 8/8 |
| C5 Effect (descriptive only) | Number of affected questions answered correctly under `baseline` vs `fault` | Reported, not gated |

**Overall result:**
- **PASS** iff C1, C2, C3 and C4 all pass.
- C5 is not a pass criterion. GPT-2 small may fail some questions even with the right
  context; that is a property of the model, not the instrument.
- A FAIL is reported as a FAIL. The diff, retriever and criteria are not changed to make
  this experiment pass.

## 5. What a PASS would and would not show

- **Would show:** in a controlled single-cause RAG fault, trace + diff localises the
  divergence to the faulty component, is silent where the fault cannot act, and
  confirms recovery after the intervention.
- **Would not show:** that Glassbox localises faults in real production systems, with
  multiple interacting causes, with sampling, or across frameworks. It would also not
  show that the first divergence is the cause in general (ASSURANCE row V2.2). Here the
  cause is known only because it was injected.

## 6. Execution

1. Commit this protocol and the code.
2. Run the experiment on the owner's Mac:

   `python3 experiments/v7/rag_fault/run.py --out experiments/v7/rag_fault/results`

3. Commit the traces and `results.json` unchanged. The evaluation is computed by the
   same script and written to `results.json`.

---
*(Addenda, if any, go below this line, with dates.)*
