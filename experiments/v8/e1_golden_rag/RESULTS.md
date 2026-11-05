# V8 phase E1: golden RAG case. RESULTS

**Overall: FAIL.** G4 (fix verified) was 9 of 13 = 0.69, against a bar of 0.80. The
other five gated criteria passed. Pre-registered in `PROTOCOL.md` v1.0. This is
attempt 1, which was stopped by hand once and resumed in the same folder (§5).

*Provenance, in commit order:*
- Protocol v1.0: `11b3a9d`.
- Code, corpus, questions, sealed test-split hash and Addendum 1: `63167a9`.
- Addendum 2 (stopped by hand after the gate; logging and runbook fixes): `fd07107`.
- The run's scored stages executed at `fd071070ad08f7831a99aefefa45f0b28b04d964`, as
  recorded in `results.json` → `software.git_commit`. The competence gate ran earlier
  at `63167a9` and was loaded from its checkpoint (Addendum 2).
- Results committed unchanged: `4f65fc2`.

*Files, committed unchanged:*

| File | sha256 |
|---|---|
| `attempt1/results.json` | `8f4136e945a2ebc7dcc916407eefc685e02fa946f48e5bf04d06e8cc3371ecfc` |
| `attempt1/run.log` | `2607c9e37e41361f1167302bb30bd58de45416d6a0efab1f08b82b8c570d78f5` |
| `attempt1/checkpoint/gate.json` | `f07393dac22e4f93edff07b8dd17ef4d8619c952f9b7361e8dd4669e6d58320c` |
| `attempt1/pip_freeze.txt` | `c5f5dc66de483ff213fe5418111ae0eaa11d52bd7fc1df2d014997fb9c453322` |

`pip_freeze.txt` was taken after the run, from the same Colab venv (§5, item 4). It is
committed with this file, not with the results.

## 1. Environment (from `results.json`)

- Tesla T4, CUDA 12.8 (torch `2.10.0+cu128`), float32.
- Python 3.11.18 in an isolated `uv` venv, Linux 6.6 x86-64.
- glassbox-mech-interp 4.6.1 (editable, from the checkout), torch 2.10.0,
  transformers 4.57.6, sentence-transformers 5.7.0, numpy 1.26.4.
- Generator: Qwen2.5-1.5B-Instruct, revision `989aa79…`, float32. Embedder:
  all-MiniLM-L6-v2, revision `1110a24…`.

## 2. Competence gate

| Rung | Model | Accuracy (dev, n = 80) | Bar | Result |
|---|---|---|---|---|
| 1 | Qwen2.5-1.5B-Instruct, float32 | 0.900 | ≥ 0.80 | selected |

Rung 2 (7B, 4-bit) was not run.

## 3. Gated results (24 test questions)

Reference labels: 16 large pairs, 70 null pairs (including 24 `log_level` pairs, null
by construction), 10 small/uncertain pairs (not scored).

| ID | Criterion | Observed | Bar | Result |
|---|---|---|---|---|
| G1 | Sensitivity | 15 / 16 large pairs attributed = 0.94 | ≥ 0.75, and ≥ 6 large pairs | **PASS** |
| G2 | Inert never blamed | `log_level` attributed on 0 / 24 questions | ≤ 2 | **PASS** |
| G3 | Visible suspect cleared | 0 / 16 null `template` pairs attributed | ≤ 0.10 (testable: ≥ 4 pairs) | **PASS** |
| G4 | Fix verified | 9 / 13 questions = 0.69 | ≥ 0.80 | **FAIL** |
| G5 | Wrong fix rejected | 11 / 11 | ≥ 0.80 | **PASS** |
| G6 | Better than today's workflow | F1: GB 0.968, B2 0.857, B1 0.467, B0 0.314. Margin over the best baseline (B2): 0.111 | ≥ 0.10 over each | **PASS** |

G6 counts (large = positive, null = negative):

| Method | TP | FN | FP | F1 |
|---|---|---|---|---|
| GB | 15 | 1 | 0 | 0.968 |
| B2 (revert one at a time, one run each) | 12 | 4 | 0 | 0.857 |
| B1 (first divergence) | 14 | 2 | 30 | 0.467 |
| B0 (blame every change) | 16 | 0 | 70 | 0.314 |

### The four questions where the fix was not verified

The pre-registered fix reverts every factor GB **attributed**. A fix counts as verified
only if it beats the candidate (one-sided permutation p < 0.05) **and** the 95% bootstrap
CI of (fixed − baseline) lies inside [−0.15, +0.15]. The band is two-sided, so a fix
that does much better than the baseline also fails.

| Question | GB attributed | Reference labels (large / small) | Recovery, correct of 60 (baseline / candidate / fixed) | Why not verified |
|---|---|---|---|---|
| u_warranty_b | index | large: index; small: template | 60 / 0 / 32 | Correct attribution. The `template` change, labelled small, still costs accuracy once `index` is reverted. |
| u_support_b | chunk_words | large: chunk_words; small: index, template | 15 / 0 / 0 | The large `chunk_words` change **improved** accuracy (only_chunk_words 146/200 vs baseline 44/200). Reverting it did not touch the drop, which came from `index`, labelled small here because the baseline was already poor. |
| u_loyalty_a | index, template | large: index, template, chunk_words | 29 / 0 / 44 | `chunk_words` was missed (the one G1 miss). The fix scored well above the weak baseline, outside the +0.15 band. |
| u_cancel_b | index | large: index; small: chunk_words, template | 43 / 0 / 60 | Correct attribution. The fix exceeded the baseline by more than 0.15. |

## 4. Descriptive results (not gated)

**Misses and false alarms** (Wilson 95% CIs; 16 large, 70 null pairs):

| Method | Misses | Miss CI | False alarms | False-alarm CI |
|---|---|---|---|---|
| GB | 1 | 0.011–0.283 | 0 | 0.000–0.052 |
| B2 | 4 | 0.102–0.495 | 0 | 0.000–0.052 |
| B1 | 2 | 0.035–0.360 | 30 | 0.319–0.545 |
| B0 | 0 | 0.000–0.194 | 70 | 0.948–1.000 |

- **Direction of the large effects.** "Large" is defined by total-variation distance and
  has no direction. 13 of the 16 large pairs lowered accuracy and 3 raised it:
  `chunk_words` and `template` on u_loyalty_a, and `chunk_words` on u_support_b. GB
  attributes a change that matters without saying whether it helped or hurt. On the
  questions above, reverting a change that helped is part of why the fix failed.
- **Small/uncertain pairs:** 10. These are `template` 6, `chunk_words` 3, `index` 1.
  They are not scored. Two of the four G4 misses involve a small-labelled factor that
  the fix left in place.
- **`log_level` sampled TVDs** (noise check): mean 0.015, max 0.105.
- **B2's run-to-run instability:** 4 flips in total across the 24 questions when its
  single-run decision was re-drawn with 5 more seeds per question.
- **Runtime** (sum of per-unit seconds in `results.json`): reference 6,589 s, main
  (all four methods) 2,943 s, recovery 1,906 s. That is 3.18 h on one T4, plus 108 s
  for the gate. The resumed session's log runs from 14:08:48 to 17:19:37 on 2026-11-04.
- **Generations:** 36,960 recorded samples (reference 24,000, main 7,200, recovery
  5,760), plus 80 for the gate and B2's re-draws.

**Not reported, though listed as descriptive in the protocol (§7):**
- Retrieval truth (whether the gold chunk is in the top-k per configuration). The code
  does not compute it. This is an omission in the code commit, not something left out
  after seeing the results.
- GPU hours from `timing.json`. The per-investigation `timing.json` files were written
  to temporary directories and not kept. The per-unit seconds above replace them.

## 5. Deviations and disclosures

1. **Stopped by hand, then resumed** (Addendum 2). Attempt 1 was stopped with Ctrl+C
   after the competence gate and before any test-question unit was saved. It resumed
   in the same `--out` at `fd07107`, with an unchanged settings fingerprint. The gate
   result was loaded from `checkpoint/gate.json`. Every scored unit (reference, main,
   recovery for all 24 questions) was computed in the resumed session.
2. **Log covers the resumed session only.** The first session's `tee` log could not be
   created (Addendum 2, item 1), so `run.log` starts at the resume.
3. **Reused runtime and environment.** The resume ran in the same Colab runtime as the
   first session. The venv was rebuilt in place (`uv venv` prompted to replace it), and
   the 38 E1 and checkpoint tests passed before the run.
4. **`pip freeze` taken late.** The download cell first used `pip`, which does not
   exist in the uv venv, and failed. The freeze was then taken with `uv pip freeze`
   from the same venv after the run. The main package versions it lists match
   `results.json` → `software`.
5. **Duplicate log lines in the Colab display.** From question 18, Colab's cell output
   showed some lines twice. The log file on Drive has each line once (`question 19/24`
   appears once).

## 6. What this shows and what it does not

- **Shows (gated, as pre-registered):** on this author-built RAG system, with one
  generator and one embedder, the investigation named the changes that mattered (15 of
  16), never blamed the inert change, cleared the visible but harmless edit, rejected
  the wrong fix every time, and scored higher F1 than the three scripted baselines on
  the same run budget. **But the fix it implied was verified on only 9 of 13
  questions, so E1 as a whole fails.**
- **Does not show:**
  - that reverting what Glassbox attributes repairs the system;
  - anything blind: the author designed the causes (E4 does that);
  - anything about real customer systems, other frameworks or hosted models;
  - anything against commercial tools (E6/E7);
  - that engineers would use it (phase G);
  - independence: the author ran it.

## 7. Post-hoc observations (not tested; for the next protocol)

These were found after looking at the results. They are hypotheses, and E1 is not
re-scored with them.

1. **Attribution is undirected.** The fix rule reverts every attributed factor,
   including changes that improved accuracy. A next version should report the
   direction of each effect and build the fix only from changes that hurt.
2. **"Reverting matters" is the better fix candidate here.** `investigate` also reports
   which single reverts matter (`GB_reverting_matters`). On u_support_b that was
   `index`, the change behind the drop, while `attributed` was `chunk_words`. No fix
   built from `reverting_matters` was rerun, so whether it would have been verified is
   unknown.
3. **Small effects add up.** On u_warranty_b, a `template` change labelled small left
   the fixed system at 32/60 against 60/60. The main run's `all_but_index` cell (19/30)
   points the same way.
4. **The two-sided band penalised fixes that beat a weak baseline** (u_cancel_b,
   u_loyalty_a). Whether a fix that clearly beats the baseline should count as
   recovered is a protocol choice, to be fixed before the next run, not after.

Any change along these lines needs a new protocol, a new sealed test split and a new
run. E1's verdict stays FAIL.

## 8. Attempts

| Attempt | What happened | Scored |
|---|---|---|
| 1 (first session, `63167a9`) | Competence gate completed (0.900). Stopped by hand at the first test question; no test unit saved (Addendum 2). | Gate only |
| 1 (resumed, `fd07107`) | All 24 questions: reference, main, recovery. Completed. | **Yes: FAIL** |
