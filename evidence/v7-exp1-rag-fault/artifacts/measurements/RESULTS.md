# V7 Experiment 1 — Controlled RAG fault: RESULTS

*Protocol `PROTOCOL.md` v1.0, committed in `0e70405` at 2026-10-05 10:18:41 +0200,
before execution. The run wrote its results between 10:19:51 and 10:19:53 the same day,
on the owner's Mac (CPU, GPT-2 small via TransformerLens). The raw traces and
`results.json` in `results/` are committed unchanged.*

## Outcome: **PASS** (all pre-declared gated criteria met)

| Criterion | Rule | Result |
|---|---|---|
| C1 Localisation | First divergence at the retrieval span for 4/4 affected questions | **4/4** |
| C2 Specificity | Unaffected questions: retrieval unchanged ⇒ no divergence | **4/4**. Retrieval was unchanged and there was no divergence in all four |
| C3 Intervention | `restored` vs `baseline`: no divergence and same outcome | **8/8** |
| C4 Determinism | `baseline_repeat` vs `baseline`: no divergence | **8/8** |
| C5 Effect (descriptive) | Affected questions answered correctly | baseline **4/4** → fault **0/4** |

## What the diff showed

- For each affected question (q1–q4), there were two divergences:
  1. **First**, a `content` divergence at `retrieval kb` on
     `gen_ai.retrieval.documents`: the retrieved document changed from the source
     document to its paired document.
  2. **Then**, a `content` divergence at the chat step, because the prompt and the
     preferred answer changed.
- The unaffected questions (q5–q8) showed zero divergences.
- Every divergence was detected from content hashes alone. No document, question or
  answer text was stored in any trace or in `results.json` (checked by string search).

## Per-question margins

Margin = log p(correct) − log p(distractor), from `results.json`:

| Q | baseline (doc) | fault (doc) |
|---|---|---|
| q1 | +10.16 (d1) | −9.80 (d5) |
| q2 | +24.50 (d2) | −11.89 (d6) |
| q3 | +1.95 (d3) | −0.72 (d7) |
| q4 | +7.64 (d4) | −10.70 (d8) |
| q5–q8 | +10.40 / +14.14 / +1.30 / +11.42 | identical |

The day-of-week questions (q3, q7) have small margins: GPT-2's priors over weekday
names compete with the context. Their sign still followed the retrieved document.

## Verification performed

1. The script evaluated the criteria in the same run.
2. **Recomputation:** a separate script re-read all 32 traces after the run (integrity
   check passed for all 32) and recomputed C1–C4 directly from the trace files with
   `diff_traces`. Every value matched.
3. **Not independent reproduction.** The recomputation was done by the same author with
   the same diff code. No one else has reproduced this result (ASSURANCE: no
   `REPRODUCED` rows).

## What this establishes, and what it does not

**Established (under these conditions only):** in a controlled RAG pipeline with one
injected cause (a stale index), the Glassbox trace + diff:
- placed the first divergence at the faulty component;
- stayed silent where the fault could not act;
- confirmed that undoing the fault restored identical behaviour;
- was deterministic.

**Not established:**
- That Glassbox localises **unknown** causes in **real** systems. The cause here was
  known by construction. This is V7 Hard Gate 3, still open.
- Behaviour with sampling or non-deterministic models, multiple interacting causes,
  real retrievers or embeddings, larger corpora, or other frameworks.
- That a first divergence is the cause in general (ASSURANCE row V2.2). Here it
  coincides with the cause only because a single cause was injected.
- Anything about answer quality. GPT-2 small, scored by teacher forcing on a
  two-option choice, is a minimal stand-in for an LLM, not a realistic RAG generator.

**Scale:** 8 questions and 4 conditions. This is a functional test of the instrument,
not a statistical study.
