# V8 phase E2: golden agent case and agent fault taxonomy. RESULTS

**Overall: FAIL.** Nine of the ten gated criteria passed. **G3 failed:** the
"harmless" visible change (`persona`) was in GB's harmful list on 4 of the 19 pairs
that the reference run labelled null (0.21, bar ≤ 0.10). Pre-registered in
`PROTOCOL.md` v1.0 with Addenda 1–3. This is attempt 1; its first session crashed
before computing anything (§5).

**Read with §6 and §7.**
- E2 is **not** a golden case. By protocol §11, only a PASS would make it one.
- Two of the five fault classes were produced (wrong tool, premature finish). E2 makes
  no claim about skipped tool, wrong order or answering without acting.
- §7 lists what I saw after unblinding. Those are observations, not a re-score.

*Provenance, in commit order:*
- Protocol v1.0: `7bf0ce1`.
- Code, sealed data and Addendum 1: `2c67e4c`.
- Pilot 1, the one wording revision and Addendum 2: `8d7a1a7`.
- Pilot 2 and Addendum 3: `b0fa1f5`. The test run executed at
  `b0fa1f5d161ac2c1e7967aa9d1ec4772c97745cb` (`results.json` → `software.git_commit`).
- Results committed unchanged in the commit titled "test(v8): E2 attempt 1 results,
  unchanged (overall FAIL; G3 0.21 > 0.10)".

*Files, committed unchanged:*

| File | sha256 |
|---|---|
| `attempt1/results.json` | `77d9cf6c5d053ea942eaef812c938806f015cd4831d089c00e2d2a476a4aef79` |
| `attempt1/run.log` (`e2_attempt1.log` on the host) | `2ac7f0d896d0d7bbb07e22678f3c77aea3257c00ef3c7d23c3e30dd4ed25461b` |
| `attempt1/checkpoint/gate.json` | `de5d43df8309a4dc85bee6250856c70d603ac32e5fd091d78444e18298caa3c9` |
| `attempt1/pip_freeze.txt` | `ba4ccb9191c77002136eafb60b7427a5bc8d88f0fccda30cca5f14d2ff108ec6` |

`attempt1/checkpoint/` holds the gate and 72 per-question units (reference, main and
recovery for each of the 24 test questions).

## 1. Environment (from `results.json`)

- NVIDIA A100-SXM4-80GB, CUDA 12.8 (torch `2.10.0+cu128`), Colab.
- Python 3.11.18 in an isolated `uv` venv, Linux 6.6 x86-64.
- glassbox-mech-interp 4.7.0 (editable, from the checkout), torch 2.10.0, transformers
  4.57.6, accelerate 1.15.0. `pip_freeze.txt` is identical to both pilots'.
- `environment.dtype` says `float32`. That field is wrong for the model, which ran in
  bf16 (`gate.rungs[].precision`); see Addendum 2 item 7.

## 2. Competence gate

| Rung | Model | Accuracy (dev, n = 120) | Parse rate | Result |
|---|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct, bf16 | 1.000 | 1.000 | **selected** |
| 2 | Qwen2.5-14B-Instruct, bf16 | not run | | |

LICENSE sha256 at the pinned revision:
`832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.

## 3. Gated results (24 test questions, 216 pairs)

Reference labels: 12 harmful-large, 5 helpful-large, 191 null, 8 small/uncertain.
11 questions regressed.

| Factor | Harmful-large | Helpful-large | Small | Null |
|---|---|---|---|---|
| `invoice_tool_desc` | 4 (all B) | 0 | 1 | 19 |
| `max_steps` | 4 (all D) | 1 (a_4531) | 0 | 19 |
| `persona` | 3 (e_4, e_5, e_6) | 0 | 2 | 19 |
| `answer_policy` | 1 (e_6) | 0 | 2 | 21 |
| `refund_fix` | 0 | 4 (all C-def) | 0 | 20 |
| `refund_tool_desc` | 0 | 0 | 0 | 24 |
| `runbook_order` | 0 | 0 | 1 | 23 |
| `retries` | 0 | 0 | 2 | 22 |
| `log_level` | 0 | 0 | 0 | 24 |

| ID | Criterion | Observed | Bar | Result |
|---|---|---|---|---|
| G1 | Sensitivity to harm | 12 / 12 harmful-large pairs in GB's harmful list | ≥ 0.75 | **PASS** |
| G2 | Inert never blamed | `log_level` harmful on 0 / 24 questions | ≤ 2 | **PASS** |
| G3 | Visible suspect cleared | `persona` harmful on 4 / 19 null `persona` pairs = 0.21 | ≤ 0.10 | **FAIL** |
| G4 | Helpful change kept | 0 / 5 helpful-large pairs in the harmful list | ≤ 0.10 (≥ 4 pairs) | **PASS** |
| G5 | Fix verified | 10 / 11 regressed questions = 0.91 | ≥ 0.80 | **PASS** |
| G6 | Wrong fix rejected | 8 / 8 eligible regressed questions | ≥ 0.80 | **PASS** |
| G7 | Better than today's workflow | F1: GB 0.706, B2 0.563, B0 0.112, B1 0.084. Margin over B2: 0.143 | ≥ 0.10 over each | **PASS** |
| G8 | Directed fix keeps the gain | 5 / 5 eligible questions | ≥ 0.75 (≥ 4) | **PASS** |
| G9 | Fault type named | 9 / 9 labelled pairs; 0 unlabelled | ≥ 0.75 (≥ 6) | **PASS** |
| G10 | Negative controls hold | 10 / 191 null pairs in the harmful list = 0.052 | ≤ 0.10 | **PASS** |

- **G3's four pairs** are `persona` on the four C-def questions (cd_7069, cd_7081,
  cd_7104, cd_7138). See §7 item 1.
- **G5's one miss** is e_5: GB's fix (revert `persona`) scored 47/60 against a
  baseline of 60/60; the CI of fix − baseline was [−0.32, −0.12].
- **G7 counts** (harmful-large positive, null negative):

  | Method | TP | FN | FP | F1 |
  |---|---|---|---|---|
  | GB | 12 | 0 | 10 | 0.706 |
  | B2 (revert one at a time, one run each) | 9 | 3 | 11 | 0.563 |
  | B0 (blame every change) | 12 | 0 | 191 | 0.112 |
  | B1 (first divergence) | 5 | 7 | 102 | 0.084 |

- **G8:** on the four C-def questions, GB's fix scored 59–60 of 60 and the undirected
  fix (E1's rule) 0 of 60. The fifth eligible question is a_4531 (§7 item 4): 31/60
  against 3/60.
- **G9:** the nine pairs are `invoice_tool_desc` on the four B questions (all labelled
  `wrong_tool`), `max_steps` on the four D questions (all `premature_finish`) and
  `answer_policy` on e_6 (`no_action`).

**Coverage (§8; reported, not gated).**

| Fault class | Factor | Harmful-large test pairs | Produced (≥ 2) |
|---|---|---|---|
| wrong tool | `invoice_tool_desc` | 4 | **yes** |
| premature finish | `max_steps` | 4 | **yes** |
| answering without acting | `answer_policy` | 1 | no |
| skipped tool | `refund_tool_desc` | 0 | no |
| wrong order | `runbook_order` | 0 | no |

`persona` produced `no_action` failures on e_4, e_5 and e_6 (55, 185 and 200 of 200
reference episodes), but it has no fault class in the protocol.

## 4. Descriptive results (not gated)

**Misses and false alarms** (Wilson 95% CIs; 12 harmful-large, 191 null pairs):

| Method | Misses | Miss CI | False alarms | False-alarm CI |
|---|---|---|---|---|
| GB | 0 | 0.000–0.243 | 10 | 0.029–0.094 |
| B2 | 3 | 0.089–0.532 | 11 | 0.033–0.100 |
| B1 | 7 | 0.320–0.807 | 102 | 0.463–0.603 |
| B0 | 0 | 0.000–0.243 | 191 | 0.980–1.000 |

- **GB's 10 false alarms:** `persona` on 4 C-def questions, `runbook_order` on 3
  (cd_7069, cd_7081, cd_7138), `max_steps` on 2 (cd_7069, cd_7081) and on b_5268.
  Addendum 2 item 6 predicted the `max_steps` ones on C-def.
- **Fix verified on the 11 regressed questions:** GB 10, the undirected fix 10, B2 8.
  E1's two-sided check gives the same counts: GB 10, undirected 10, B2 8, wrong fix 1.
- **The undirected `attributed` list**, scored against harmful and helpful pairs
  together: 17 TP, 0 FP, 0 FN.
- **Small/uncertain pairs:** 8. One failed fix left a small factor in place: e_5,
  where `answer_policy` alone scored 148/200 against a baseline of 197/200.
- **A questions:** the candidate scored 167, 169 and 164 of 200 against baselines of
  194, 197 and 197. None met the "regressed" rule (CI lower bounds 0.085–0.11), and GB
  named nothing harmful on them. `retries` was small on two (178 and 179 of 200).
- **B2's instability:** 32 flips in total over 5 extra seeds per question.
- **`log_level` sampled TVDs:** at most 0.015.
- **Parse failures:** 30 in the reference run's 52,800 episodes; 0 under the baseline.
- **Forced finals under the baseline:** 0.
- **Runtime** (sum of per-unit seconds): reference 10,597 s, main 3,570 s, recovery
  1,106 s. That is 4.24 h on one A100, plus 100 s for the gate. Addendum 3 projected
  4.88 h from pilot 1.
- **Episodes:** 52,800 (reference), 14,400 (main) and 5,580 (recovery; arms with the
  same configuration shared runs).

## 5. Deviations and disclosures

1. **First session crashed before any result.**
   - Its log ends in `torch.AcceleratorError: CUDA error: an illegal memory access was
     encountered`, raised in the model's first forward pass during the gate.
   - Nothing had been saved except the settings fingerprint and `pip_freeze.txt`.
   - The cause was not established. The GPU and commit of that session were not
     recorded, because the crash came before `results.json` is written.
   - The owner started a new runtime and ran the same command with the same `--out`.
     That session (A100 80 GB) ran from the gate at 17:57 to `e_6 done` at 22:11 on
     2026-11-12 (log times), without interruption. `run.log` contains both sessions.
2. **The pilots and the test ran on three different cards:** A100 40 GB (pilot 1), L4
   (pilot 2) and A100 80 GB (test).
3. **Pre-run changes are in the addenda:** required tools for C-inel, sampling by
   block, the outcome tokenizer (Addendum 1); the one wording revision for three
   factors, written after a dev-only diagnostic (Addendum 2).
4. **No other deviation.** The run ended with `overall: FAIL`.

## 6. What this shows and what it does not

- **Overall FAIL.** The claim as pre-registered (every testable criterion passes) is
  not supported.
- **Individually pre-declared criteria that passed.** Because the experiment failed
  overall, these are preliminary:
  - GB's harmful list contained every harmful-large pair (12/12) and never the inert
    change.
  - It kept the change that helped (0/5 blamed). Its fix beat E1's revert-everything
    fix on every eligible question (5/5).
  - Its fix was verified on 10 of 11 regressed questions; the wrong fix was rejected
    on 8 of 8.
  - For the two fault classes produced, the fault label was right on every pair.
  - It beat the scripted baselines by the required margin.
- **Does not show:**
  - that the investigation clears a visible but harmless change: G3 failed;
  - anything about skipped tool, wrong order or answering without acting as fault
    classes: they were not produced;
  - anything blind: the author designed the faults, and revised three wordings after
    seeing dev behaviour (E4);
  - other models, frameworks or real production agents (E3);
  - anything against commercial tools (E6/E7), or that engineers would use it (G);
  - independence: the author's code, the owner's run.

## 7. Observations after unblinding (not a re-score)

1. **Where G3 failed, the reference label could not see the effect.**
   - On C-def the baseline is 0/200 because of the refund bug, so `only_persona` vs
     baseline is 0 vs 0 and the pair is labelled null.
   - In the candidate, the bug is fixed. There, reverting `persona` alone restored the
     answer: the wrong-fix arm scored 55, 55, 59 and 58 of 60 against a candidate of
     0/60.
   - GB found `persona` through its necessity test (revert it alone from the
     candidate).
   - Addendum 2 item 6 named this limit for `max_steps` on C-def. **I did not foresee
     it for `persona`, and G3 was not protected against it.** That is a protocol
     design error. The verdict stays FAIL.
2. **All 10 of GB's false alarms are necessity findings on pairs labelled by
   sufficiency.** Nine are on C-def. The tenth is `max_steps` on b_5268, where the
   main run's `all_but_max_steps` cell scored 9/30 against a candidate of 0/30.
3. **The revised `refund_tool_desc` and `runbook_order` did nothing alone,** as pilot 2
   suggested: `refund_tool_desc` scored 200/200 on every C-inel question, and
   `runbook_order` 199–200 of 200 on every D question.
4. **a_4531 looks like an answer-key problem.**
   - Its baseline scored 20/200 although the agent called the right tool: the
     labeller put 180 of 180 failing baseline episodes in `other`.
   - Final answer texts are not stored, so the cause was not checked.
   - It affects two counts: `max_steps` is "helpful" there (one of G4's 5 pairs), and
     it is one of G8's 5 questions. Without it, G4 would be 0/4 and G8 4/4.
5. **Small effects were missed again.** On e_5 the fix failed with a small factor left
   in place, and on three A questions a drop of about 15 points went unattributed.
   E1 showed the same weakness.

## 8. Attempts

| Session | What happened | Scored |
|---|---|---|
| 1 (2026-11-12, GPU not recorded) | Crashed in the gate with a CUDA illegal-memory-access error. Nothing computed. | No |
| 2 (2026-11-12, A100 80 GB, `b0fa1f5`) | Gate (7B: 1.000 / 1.000); all 24 questions. | **Yes: FAIL** |
