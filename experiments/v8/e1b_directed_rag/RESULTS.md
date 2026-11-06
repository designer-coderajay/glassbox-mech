# V8 phase E1b: directed attribution and a hurt-only fix. RESULTS

**Overall: PASS.** Every testable gated criterion passed. G4, which checks that changes
that helped are kept, was **not testable**: there were 0 helpful pairs. Pre-registered in
`PROTOCOL.md` v1.0. This is attempt 1. The Colab runtime was terminated once, and the
run resumed in the same folder (§5).

**Read with §6.** The case turned out all-or-nothing:
- every regressed question went from 200/200 to 0/200;
- every other question stayed at 200/200;
- there were no small and no helpful effects.

So E1b shows the investigation working on a clean, strong-effect case. **It does not
test whether the two changes made after E1 fix E1's failure** (direction, and the
one-sided recovery check). E1 itself stays FAIL (ASSURANCE V3.10).

*Provenance, in commit order:*
- Protocol v1.0: `8044a76`.
- Code, data, sealed test-split hash and Addendum 1: `5175240`.
- The run executed at `517524012d0d5e750f22beb8405929f12f9643db`, as recorded in
  `results.json` → `software.git_commit`.
- Results committed unchanged: `0487aa1`.

*Files, committed unchanged:*

| File | sha256 |
|---|---|
| `attempt1/results.json` | `d627e46d1379635f86ef19ca5a2729d4bae5f344241b4eb731f6f9745052f26a` |
| `attempt1/run.log` | `52aae31bc9611f97fe84c271e4aab283317c8d4a13cbdadedcf478489366bc15` |
| `attempt1/checkpoint/gate.json` | `18a63e552dfde76cbf0434553c1b72a09b69c2a96e1d088d8a98c76b57efddf0` |
| `attempt1/pip_freeze.txt` | `c2794c5c74f0005b3bc706a3a8abcffe2b14611ccca1470eba44067a13a129c6` |

`pip_freeze.txt` was written by `run.py` inside the run folder. It lists 110 packages.

## 1. Environment (from `results.json`)

- Tesla T4, CUDA 12.8 (torch `2.10.0+cu128`).
- Python 3.11.18 in an isolated `uv` venv, Linux 6.6 x86-64.
- glassbox-mech-interp 4.6.1 (editable, from the checkout), torch 2.10.0, transformers
  4.57.6, sentence-transformers 5.7.0, bitsandbytes 0.50.2, accelerate 1.15.0.
- Embedder: all-MiniLM-L6-v2 (revision `1110a24…`).

## 2. Competence gate

| Rung | Model | Accuracy (dev, n = 80) | Bar | Result |
|---|---|---|---|---|
| 1 | Qwen2.5-1.5B-Instruct, float32 | 0.700 | ≥ 0.80 | not selected |
| 2 | Qwen2.5-7B-Instruct, 4-bit (bitsandbytes) | 1.000 | ≥ 0.80 | **selected** |

The ladder worked as pre-registered. **The generator therefore differs from E1's**,
where the 1.5B model passed the gate with 0.900. E1 and E1b do not compare like for like.

## 3. Gated results (24 test questions)

Reference labels:
- 18 harmful-large pairs: 12 `index`, 6 `chunk_words`;
- 78 null pairs: all 24 `template`, all 24 `log_level`, 18 `chunk_words`, 12 `index`;
- 0 helpful, 0 small.

15 questions regressed: the 12 "updated" questions and 3 "unchanged" ones.

| ID | Criterion | Observed | Bar | Result |
|---|---|---|---|---|
| G1 | Sensitivity to harm | 18 / 18 harmful-large pairs in GB's harmful list = 1.00 | ≥ 0.75, ≥ 6 pairs | **PASS** |
| G2 | Inert never blamed | `log_level` harmful on 0 / 24 questions | ≤ 2 | **PASS** |
| G3 | Visible suspect cleared | 0 / 24 null `template` pairs | ≤ 0.10 (testable: ≥ 4 pairs) | **PASS** |
| G4 | Helpful changes kept | 0 helpful pairs | ≤ 0.10 (testable: ≥ 4 pairs) | **not testable** |
| G5 | Fix verified | 15 / 15 regressed questions = 1.00 | ≥ 0.80, ≥ 6 questions | **PASS** |
| G6 | Wrong fix rejected | 15 / 15 = 1.00 | ≥ 0.80 | **PASS** |
| G7 | Better than today's workflow | F1: GB 1.000, B2 0.875, B1 0.545, B0 0.316. Margin over the best baseline (B2): 0.125 | ≥ 0.10 over each | **PASS** |

G7 counts (harmful-large = positive, null = negative):

| Method | TP | FN | FP | F1 |
|---|---|---|---|---|
| GB | 18 | 0 | 0 | 1.000 |
| B2 (revert one at a time, one run each) | 14 | 4 | 0 | 0.875 |
| B1 (first divergence) | 18 | 0 | 30 | 0.545 |
| B0 (blame every change) | 18 | 0 | 78 | 0.316 |

## 4. Descriptive results (not gated)

**Misses and false alarms** (Wilson 95% CIs; 18 harmful-large, 78 null pairs):

| Method | Misses | Miss CI | False alarms | False-alarm CI |
|---|---|---|---|---|
| GB | 0 | 0.000–0.176 | 0 | 0.000–0.047 |
| B2 | 4 | 0.090–0.452 | 0 | 0.000–0.047 |
| B1 | 0 | 0.000–0.176 | 30 | 0.285–0.496 |
| B0 | 0 | 0.000–0.176 | 78 | 0.953–1.000 |

- **Fix verified, GB vs B2:** GB 15 / 15 regressed questions, B2 13 / 15.
  - B2 missed `chunk_words` on the three questions where two changes both hurt
    (u_direct_debit_b, u_solar_export_a, u_emergency_a). On u_emergency_a its single
    reverted runs blamed nothing at all.
  - Its fix on u_emergency_a was therefore not run.
  - On u_solar_export_a, B2's fix reverted `index` only and scored 51/60. Its
    fix − baseline CI was [−0.25, −0.07], below the −0.15 bound, so it was not
    verified.
  - On u_direct_debit_b, reverting `index` alone was enough (verified).
- **E1's two-sided check, for comparison:** GB's fix 15 / 15, B2's 13 / 15, the wrong
  fix 0 / 15.
  - Every GB fix scored 60/60, exactly at the 200/200 baseline, so the one-sided and
    two-sided rules agree on every question.
  - The undirected `attributed` list, scored E1-style, has F1 1.00 (18 TP, 0 FP, 0 FN).
  - **On this data, neither change made after E1 made a difference.**
- **Re-chunking hurt three questions whose facts did not change** (n_boiler_cover,
  n_ev_tariff, n_heat_pump). GB found all three, and so did B2.
  - Retrieval truth shows that on n_boiler_cover and n_heat_pump the answer chunk was
    no longer in the top 3 at `chunk_words` 40.
  - On n_ev_tariff it was still retrieved, so that loss happened at generation.
- **Retrieval truth for `index`:** on all 12 "updated" questions, the stale index left
  no chunk with the current answer in the top 3.
- **B2's run-to-run instability:** 0 flips in total over 5 extra seeds per question.
- **`log_level` sampled TVDs:** 0.000 on every question.
- **Small/uncertain pairs:** 0. No failed fix left a small factor in place.
- **Runtime** (sum of per-unit seconds): reference 7,982 s, main 3,070 s, recovery
  2,107 s. That is 3.66 h on one T4, plus 506 s for the gate (both rungs).
- **Generations:** 41,340 recorded samples (reference 28,800, main 7,200, recovery
  5,340), plus 160 for the gate and B2's re-draws. Recovery reused runs where two arms
  had the same configuration (Addendum 1 item 3).

## 5. Deviations and disclosures

1. **Runtime terminated, then resumed.**
   - The first session's log runs from 11:13 on 2026-11-05 to 13:25, the start of
     question 14 (n_priority_register). It then stops, with no error, traceback,
     `KeyboardInterrupt` or `overall:` line.
   - The second session began at 04:33 on 2026-11-06, on a new machine: the model was
     downloaded again. It loaded the gate result and questions 1–13 from the checkpoint,
     computed questions 14–24 in full (all three of question 14's units carry
     second-session times), and finished at 06:16.
   - The owner did not record why the first session ended. A clean stop with no Python
     message points to the Colab runtime being shut down, most likely a disconnect,
     the idle timeout or the free-tier GPU limit. When the owner returned, Colab at
     first gave a CPU-only runtime.
   - The settings fingerprint matched on resume, and every unit was computed once at
     the same commit.
2. **The pre-run model check (runbook step 4) failed twice on CPU-only runtimes**
   ("Found no NVIDIA driver", then `nvidia-smi: command not found`), before a T4 was
   available. No run step was started on CPU.
3. **No other deviation.** The run ended with `overall: PASS`.

## 6. What this shows and what it does not

- **Shows (gated, as pre-registered):**
  - On a second author-built RAG system (a new corpus and a sealed test split), with
    Qwen2.5-7B-Instruct in 4-bit as the generator, the direction-aware investigation
    named every change that hurt (18 / 18).
  - It never blamed the inert or the visible change.
  - The fix it implied restored every regressed question (15 / 15), and the wrong fix
    was rejected every time.
  - It beat the scripted baselines on the same run budget. Most of the margin came from
    regressions with two causes, where reverting one change at a time missed the
    second.
- **Does not show:**
  - **that direction or the one-sided recovery check fix E1's failure.** This case had
    no helpful or small effects, so G4 could not be tested, and E1's rules give the
    same answers here;
  - anything about mixed-direction or small-effect regressions. That is exactly where
    E1 failed;
  - anything about the 1.5B model used in E1: it failed this case's competence gate;
  - anything blind: the author wrote the data after seeing E1 (E4);
  - generalisation beyond this pipeline, factor set and model;
  - anything against commercial tools (E6/E7);
  - that engineers would use it (phase G);
  - independence: the author ran it.
- **Golden case.** By protocol §9, a PASS lets E1b serve as the golden case, but only
  with the E1 FAIL shown next to it and with the scope above.

## 7. Attempts

| Session | What happened | Scored |
|---|---|---|
| 1 (2026-11-05, `5175240`) | Gate (1.5B 0.70 → 7B 1.00); questions 1–13 completed. Runtime ended during question 14 (cause not recorded). | Units 1–13 and the gate, kept |
| 2 (2026-11-06, `5175240`) | Resumed; questions 14–24 completed. | **Yes: PASS** |
