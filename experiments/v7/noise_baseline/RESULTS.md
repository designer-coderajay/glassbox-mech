# V7 Experiment 3: noise-aware group diff. RESULTS

*Protocol `PROTOCOL.md` v1.0 was committed in `31f80c7` (2026-10-05 20:02:41 +0200)
before any code. The code was committed in `21ceb61` (20:22:14) before any run. The
results were committed unchanged in `3cf24ac` (20:26:04), run on the owner's Mac.
Part B used GPT-2 small via TransformerLens on CPU.*

## Outcome: **FAIL**

One gated criterion missed its pre-declared threshold: A2, 0.79 against ≥ 0.80. Under
the protocol, the experiment therefore FAILS. The threshold, method, seeds and worlds
are not changed after the run.

| Criterion | Rule | Result |
|---|---|---|
| A1 Type I (W0, no change) | ≤ 0.08 | **0.015** (3/200) PASS |
| A2 Chat localisation (W1) | ≥ 0.80 | **0.79** (158/200) **FAIL** |
| A2b No false upstream flag (W1) | ≤ 0.08 | **0.015** (3/200) PASS |
| A3 Retrieval localisation (W2) | ≥ 0.80 | **0.94** (188/200) PASS |
| A4 Free text untestable | 100% | **100%** PASS |
| B1 Real null (GPT-2, baseline vs baseline_more) | ≥ 7/8 | **8/8** PASS |
| B2 Real localisation (q1–q4) | 4/4 | **4/4** PASS |
| B3 Real specificity (q5–q8) | 4/4 | **4/4** PASS |
| A5 One vs two causes (descriptive) | | `chat` flagged in W2 **0.505**, in W3 **0.955** |
| A6 Pairwise diff on pure noise (descriptive) | | **1.0** |
| B4 Pairwise diff on pure noise, GPT-2 (descriptive) | | **1.0** |

## Why A2 failed: a power shortfall, not mislocalisation

This is a post-hoc diagnosis. It is exploratory and does not change the outcome.

**W1 breakdown, 200 replications:**
- 158 had the first finding at `chat#0`, which is correct;
- 39 had **nothing significant**;
- 3 had the first finding at `retrieval#0`, which is wrong. That is the same rate as the
  Type I rate (A1).

Whenever the group diff reported anything, it was at the right step in 158 of 161
replications. The failure is missed detections.

**Check that this is not a bug.** I regenerated four committed replications (three
misses, 8, 12 and 13, plus rep 0) in a sandbox from the protocol seeds. All four
reproduced the committed summary exactly.

**What the misses look like.** In the misses the shift is real but moderate, e.g.
`correct` 20/30 in A vs 10/30 in B. Individual p-values are about 0.013–0.04. Holm
across 5 tested fields needs the smallest p-value to be ≤ 0.01.

**Likely cause: redundant fields inflate the multiple-testing correction.** Three of the
five tested fields carry exactly the same information:
- the tool step's `present`;
- the tool step's `gen_ai.tool.name`;
- the tool step's `status`.

Each is `None` exactly when the step is absent. Holm therefore pays for five tests when
there are about three distinct signals. This was not foreseen in the protocol.

My informal expectation of high power at N = 30 was optimistic; I did not write any
power number into the protocol. The 0.80 threshold is the only commitment, and it was
missed.

**Not done:** re-running with deduplicated fields or larger groups. Either would be a new
method and needs a new pre-registration with fresh seeds (proposed Experiment 3b
below).

## Other findings

1. **False alarms were controlled.**
   - A1: 0.015 on synthetic nulls.
   - B1: 8/8 on real GPT-2 sampling.
   - The method never flagged noise at more than the declared rate.
   - *Caveat on B1:* for each question, only **one** field was testable
     (`glassbox.answer.contains_correct`). The sampled-output hash was classified
     untestable, as designed, and everything else was constant. B1 is therefore a weak
     test of the null on real data.
2. **The pairwise diff fires on every pair of same-configuration runs.**
   - A6 = 1.0 (synthetic) and B4 = 1.0 (GPT-2, 160 pairs).
   - With sampling, the existing `glassbox.v7.diff` reports a divergence for **every**
     pair, even when nothing changed. This was the motivating problem, and it is now
     measured.
3. **A5, one vs two causes.** The pre-declared expectation was "high in both". That was
   only partly right:
   - `chat` was flagged in 0.505 of W2 replications, where retrieval is the only cause;
   - it was flagged in 0.955 of W3 replications, which have two causes.
   - In a single comparison, a `chat` flag therefore cannot tell a second cause apart
     from downstream propagation: half of the W2 replications flag `chat` with no cause
     there. V2.2 (first divergence ≠ cause) stands, and V3 (controlled re-runs) is
     still required.
4. **Real fault on GPT-2.** For q1–q4:
   - The first finding was at retrieval (p = 0.0005, the floor with 1,999
     permutations).
   - `chat` input messages were also flagged; this is expected propagation.
   - For q2 only, `contains_correct` was also significant (0.45 → 0.00).
   - `contains_correct` rates for the affected questions fell from baseline to fault:
     q1 0.10 → 0.00, q2 0.45 → 0.00, q3 0.20 → 0.05, q4 0.05 → 0.00.
   - Unaffected q5–q8: nothing significant.

## What this does and does not show

**Shows** (on these pipelines only, single author, not blind):
- The group diff kept false alarms at or below the declared rate.
- It localised a deterministic retrieval fault in a real sampling LLM.
- Where it detected a single synthetic change, it localised it correctly in 158 of 161
  cases.
- The pairwise diff is unusable on sampled runs without a noise baseline.

**Does not show:**
- Adequate power at N = 30 for moderate effects. A2 failed.
- Any ability to separate one cause from two (A5).
- Anything about unknown causes in real systems (V2.4).

## Proposed next step (not started; needs its own protocol)

**Experiment 3b**, pre-registered with a new master seed:
- Collapse fields that are identical partitions of the runs (e.g. presence-derived
  tool fields) into one test before applying Holm.
- Report a power curve over N ∈ {10, 20, 30, 50}.
- Gate localisation at the N where the declared power is reached.

The A2 result above stays FAIL regardless of what 3b finds.
