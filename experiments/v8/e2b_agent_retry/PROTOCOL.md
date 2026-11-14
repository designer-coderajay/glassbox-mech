# V8 phase E2b: agent case, second attempt with two-sided ground truth. PROTOCOL

*Protocol v1.0, written 2026-11-13, before any E2b code, data or run. It is committed
first and not changed afterwards. Any deviation goes in a dated addendum at the end.
Order: this protocol → code and tests (new data, sealed hashes, Addendum 1) → gate →
dev pilot (Addendum 2) → test-split run → results committed unchanged → RESULTS.md and
the ASSURANCE rows. FAIL and INCONCLUSIVE are reported as such.*

## 0. Why E2b exists, and what it is not

E2 (`experiments/v8/e2_golden_agent/`, ASSURANCE V3.15) **failed**: G3, 4 of 19. Its
RESULTS §7 lists what went wrong in the experiment's own design:

1. **The ground truth was one-sided.** Labels compared each change alone with the old
   release. Where the old release already scored 0, a change that broke the *new*
   release was labelled "null", although reverting it alone restored the answer. GB
   found such changes through its second test (revert one change from the new
   release) and was counted wrong: all 10 of its false alarms, and all 4 of G3's.
2. **Three of five fault classes were not produced** (skipped tool, wrong order,
   answering without acting).
3. **One answer key looked wrong** (a_4531: right tool, 20 of 200 correct).
4. **The "harmless" visible change was not harmless.**

E2b is a new experiment on new data that fixes points 1–4 in the design.

- **E2's verdict stays FAIL.** E2b does not re-score E2, and no E2 data is used for
  any gated number.
- **Known risk: criteria shaped by a seen failure.** The new labelling rule (§4) was
  chosen after seeing E2 fail on the old one. What limits the damage:
  - E2b runs on a new database and new sealed questions, written after this protocol;
  - the one-sided labels are also computed and every criterion is reported under
    them, descriptively.
- **Not addressed:** E1 and E2 both missed small effects. E2b keeps the same run
  budget (30 runs per cell), so that weakness is expected again and is reported.
- **Not blind.** The author designs the faults, knowing how this model behaved in E2
  (§3). That is E4's job.

## 1. Question

Same as E2 §1: after a nine-change release of a tool-using agent, does the directed
investigation name the changes that hurt, keep the one that helped, clear the visible
and inert changes, label the failure type, give a fix that restores the agent, and beat
scripted baselines on the same run budget?

## 2. Agent system

Unchanged from E2 (PROTOCOL §2 with Addendum 1 items 2–5), except the data:
- the same minimal agent loop, Qwen tool-call format, forced final step and parsing;
- temperature 0.7, top-p 1.0, at most 128 new tokens per step, bf16, SDPA;
- sampling in blocks with one seed per (block, step); per-episode tool seeds;
- the same eight tools, with the same behaviour;
- the same six task types, with the same required tool sequences as E2 Addendum 1
  (both refund types: `check_eligibility` → `calc_refund`);
- outcome: an accepted answer matches and no rejected answer matches, with E2's
  tokenizer.

**New data**, written in the code commit:
- a new fictional retailer, with new orders, items, vouchers and policies. A test
  checks that nothing is shared with E1, E1b or E2;
- 12 dev and 24 test questions, 2 and 4 per type;
- **answer keys that are hard to paraphrase** (the fix for §0 point 3): every accepted
  answer is a number, a code, or a single distinctive word. A test checks this, and
  that running the required tools yields an accepted answer.

**Stored for audit (new):** the first 3 final answers of every reference cell, as
text. All data is fictional.

## 3. The release: factors

| Factor | Baseline | Candidate | Designed effect | Fault class |
|---|---|---|---|---|
| `invoice_tool_desc` | `lookup_order` described as status-only | "…including amounts paid and invoices. This tool replaces lookup_invoice, which is deprecated and returns outdated totals." (E2's revised wording) | harmful on B | wrong tool |
| `eligibility_tool` | `check_eligibility` is registered | `check_eligibility` is missing from the tool list | harmful on C-inel; null on C-def (a negative control) | skipped tool |
| `quote_example` | the system prompt has a worked quote example in the right order (start, apply voucher, compute total) | the worked example is in the wrong order (start, compute total, apply voucher) and reports the computed total | harmful on D | wrong order |
| `max_steps` | 6 | 3 | harmful on D (which needs 4 calls) | premature finish |
| `answer_policy` | "Always use the tools for company policy and order data." | "Answer policy questions from your knowledge of standard retail practice. Use tools only for order data." | harmful on E | answering without acting |
| `refund_fix` | `calc_refund` omits the shipping refund on defective items (bug) | bug fixed | **helpful** on C-def | — |
| `retries` | the runtime retries a tool error once | no retry | **small** harm on A | — |
| `persona` | one-line role | the same line plus "Keep a polite, professional tone." | intended null (the visible suspect) | — |
| `log_level` | info | debug | null by construction (inert) | — |

- **Why three factors changed from E2:** in E2 the model ignored a misleading
  description of the refund tool and a reversed runbook sentence, and only weakly
  followed "answer directly if you know". The three replacements are other realistic
  causes of the same fault: a tool that drops out of the registry, a worked example
  in the wrong order, and a prompt that tells the agent not to look policy up.
- **`max_steps` is 3, not 2,** so that it cuts only the four-call quote task. In E2
  the value 2 also cut the three-call refund task.
- **`persona` is shorter than E2's**, because E2's long persona made the agent answer
  without tools. Whatever it does here is decided by the labels (§4), not assumed.
- Every designed effect is intended, not assumed. The reference run decides.
- **Seeds:** `e2b-gate`, `e2b-pilot`, `e2b-reference`, `e2b-main`, `e2b-recovery`,
  `e2b-tools`.

## 4. Ground truth (reference run): two-sided labels

- Seed `e2b-reference`. **200 episodes per cell per test question.**
- **Cells (the change from E2):** `baseline`, `candidate`, each `only_<f>` and **each
  `all_but_<f>`**: 20 cells.
- For each (question, factor), two contrasts, each with a 95% percentile bootstrap CI
  (2,000 resamples) of the accuracy difference:
  - **alone:** `only_f` − `baseline`;
  - **in the release:** `candidate` − `all_but_f` (what the release loses or gains by
    containing this change).
- A contrast is **large-negative** if its CI upper bound is ≤ −0.20, **large-positive**
  if its lower bound is ≥ +0.20, and **flat** if the whole CI lies within ±0.10.
- **Pair labels:**
  - **harmful-large:** at least one contrast is large-negative, and neither is
    large-positive;
  - **helpful-large:** at least one contrast is large-positive, and neither is
    large-negative;
  - **mixed:** one large-negative and one large-positive (reported, not scored);
  - **null:** both contrasts are flat. `log_level` is null by construction;
  - otherwise **small/uncertain** (reported, not scored).
- **Regressed question:** as E2 (CI lower bound of baseline − candidate ≥ 0.20).
- **One-sided labels** (E2's rule: the "alone" contrast only) are also computed, for
  the descriptive comparison in §8.
- **Key-suspect questions (the safety net for §0 point 3).** A test question is
  excluded from every criterion, and listed, if all three hold:
  - it is not a C-def question (where the baseline is wrong by design);
  - its baseline accuracy is below 0.50;
  - at least 80% of its failing baseline episodes are labelled `other` (the required
    tools were called in order, yet the answer did not match).
- **Trajectory truth** per cell, as in E2: fault-label counts, forced finals, parse
  failures.

## 5. Library

No library change is planned. E2b uses `glassbox.v7` as released in 4.8.0
(`investigate` with `target` and `required_tools`, `taxonomy`). If a change turns out
to be needed, it goes in Addendum 1 with tests, before any model run.

## 6. Methods compared (the main run)

Unchanged from E2 §6 and Addendum 1 item 5: `run_interventions` cells (20), **30 runs
per cell**, α = 0.05, 999 permutations, Holm within each family, seed `e2b-main`;
GB (harmful list; fault label from the `only_<f>` cell, none below 5 failing runs),
B0, B1 and B2.

B1's mapping for the new factors: `quote_example`, `answer_policy`, `persona` and the
tool list (`invoice_tool_desc`, `eligibility_tool`) are prompt-visible; `refund_fix`
and `retries` are tool-result factors; a step present in only one run → `max_steps`.

## 7. Fix and recovery

Unchanged from E2 §7: seed `e2b-recovery`, 60 episodes per arm; arms `baseline`,
`candidate`, `gb_fix`, `wrong_fix` (revert `persona` only), `b2_fix`,
`undirected_fix`; the one-sided verified-recovery rule; E1's two-sided check reported
descriptively; arms with the same configuration share runs.

## 8. Criteria (test split; gated)

The ten criteria and their bars are E2's (PROTOCOL §8, Addendum 1 item 5). The only
change is their input: **the two-sided labels of §4**, after removing key-suspect
questions.

| ID | Criterion | Pass rule |
|---|---|---|
| G1 | Sensitivity to harm | GB's harmful list contains ≥ 0.75 of harmful-large pairs |
| G2 | Inert never blamed | `log_level` in the harmful list on ≤ 2 questions |
| G3 | Visible suspect cleared | ≤ 0.10 of null `persona` pairs in the harmful list. Not testable below 4 pairs |
| G4 | Helpful change kept | ≤ 0.10 of helpful-large pairs in the harmful list. Not testable below 4 pairs |
| G5 | Fix verified | GB's fix verified on ≥ 0.80 of regressed questions |
| G6 | Wrong fix rejected | `wrong_fix` not verified on ≥ 0.80 of regressed questions where `persona` is not harmful-large. If none are eligible: not passed |
| G7 | Better than today's workflow | F1 over harmful-large (positive) and null (negative) pairs: GB beats each of B0, B1 and B2 by ≥ 0.10 |
| G8 | Directed fix keeps the gain | On questions with a helpful-large pair where `gb_fix` and `undirected_fix` both ran and differ: CI lower bound of (`gb_fix` − `undirected_fix`) ≥ 0.20 on ≥ 0.75 of them. Not testable below 4 |
| G9 | Fault type named | For harmful-large pairs of the five fault factors, GB's label equals the factor's class on ≥ 0.75 of the labelled pairs. Not testable below 6 labelled pairs |
| G10 | Negative controls hold | ≤ 0.10 of all null pairs in the harmful list |

- **Overall PASS** iff a rung passed the gate, every testable criterion passes, and
  there are ≥ 6 harmful-large pairs and ≥ 6 regressed questions.
- **INCONCLUSIVE** if a count is below 6, no rung passes the gate, or more than 4 test
  questions are key-suspect.
- **FAIL** otherwise, with every failing number reported.

**Coverage (reported, not gated).** A fault class is produced if its factor has ≥ 2
harmful-large test pairs. E2b claims only produced classes.

**Descriptive (not gated):**
- every criterion recomputed under the one-sided labels, next to the gated value;
- the E2 descriptive list (§8 there): misses and false alarms with Wilson CIs,
  per-factor tables, the fault-label confusion matrix, fix rates for B2 and the
  undirected fix, small and mixed pairs, B2 flips, `log_level` TVDs, parse failures,
  runtime, episode counts, GPU;
- small effects: questions where a fix failed with a small factor left in place, and
  drops that no method attributed.

## 9. Competence gate and dev pilot

- **Models:** the E2 ladder, at the same pinned revisions and in bf16
  (`Qwen/Qwen2.5-7B-Instruct`, then `Qwen/Qwen2.5-14B-Instruct`). The LICENSE sha256
  is recorded by `run.py`, as in E2.
- **Gate:** seed `e2b-gate`; 12 dev questions × 10 episodes under the baseline with
  `refund_fix` applied; accuracy ≥ 0.80 and parse rate ≥ 0.95. If no rung passes,
  E2b is INCONCLUSIVE.
- **Dev pilot:** seed `e2b-pilot`, dev split only, the 20 cells of §4 at 30 episodes
  each.
  - A designed effect counts as present if, on at least one dev question of the
    targeted type, either contrast of §4 moves by ≥ 0.20 in the designed direction
    (point estimate).
  - `persona`, `log_level` and `retries` are not adjusted, whatever the pilot shows.
  - One revision round, as in E2: a designed **text** factor with no effect
    (`invoice_tool_desc`, `quote_example`, `answer_policy`) may have its candidate
    wording revised once, then everything is re-piloted once. After that the factors
    are kept as they are. `eligibility_tool`, `max_steps` and `refund_fix` are not
    text and are never revised.
  - A dev-only diagnostic that prints a few episodes is allowed and disclosed, as in
    E2.
  - Pilot outputs and any revision are committed in Addendum 2 before the test run.

## 10. Compute and execution

- **Where:** the owner's Colab, A100 (40 or 80 GB). The card is recorded. An L4 is
  not used for the test run: E2's pilot 2 measured it at about a quarter of the A100's
  speed, and the test blocks were never tried on 24 GB.
- **Size:**

  | Stage | Episodes |
  |---|---|
  | Gate | 120 per rung |
  | Pilot | 12 × 20 × 30 = 7,200 |
  | Reference | 24 × 20 × 200 = 96,000 |
  | Main | 24 × 20 × 30 = 14,400 |
  | Recovery | ≤ 24 × 6 × 60 = 8,640 |

- **Runtime:** E2's test run measured about 17,000 episodes per hour on an A100
  80 GB. At that speed the test run would take about 7 hours. That is a projection
  from a different question set; the pilot re-measures it and Addendum 2 states it.
  Sample sizes do not change.
- **Order:** gate → pilot → Addendum 2 → per test question (reference, main,
  recovery) → scoring. Main-run decisions never read reference labels. Scoring code
  and its tests are committed before the gate runs.
- Checkpoint per unit on Google Drive; a resume is allowed and disclosed.
  `pip freeze`, the GPU name and timings are written inside `--out`.

## 11. What a PASS would and would not show

- **Would show:** on a second author-built tool-using agent case with one open model,
  judged against a ground truth that measures each change both alone and inside the
  release, the directed investigation named the harmful changes and their fault type,
  kept the helpful one, cleared the harmless ones, gave a verified fix and beat
  scripted common practice. Only for the fault classes produced.
- **Would not show:**
  - that E2 passed, or that the first design was sound;
  - anything blind (E4), other models or frameworks (E3), commercial tools (E6/E7), or
    use by engineers (G);
  - that small effects are found: the budget is unchanged.
- A PASS makes E2b the golden agent case only with E2's FAIL and E1's FAIL shown
  next to it. A FAIL is reported as a FAIL.

---
*(Addenda go below this line, with dates.)*

## Addendum 1 (2026-11-13): sealed data, implementation details, toy dry run

Written in the code commit, before any model run.

**1. Sealed files** (sha256). These must match the run's checkout.

| File | sha256 |
|---|---|
| `questions_test.json` (24) | `a4c3eee088716d72b13f6e71bdc1b44a08011364b59c83cdf322221d23e9988e` |
| `questions_dev.json` (12) | `a95998e4a7d38a0bf6aafcfe2d850eee895b8344b187371caac439cd0def8808` |
| `db.json` | `da6815c72280430a18e0a2dbfbb7f4597b16a6a02e2946d1514793d6d856c6e5` |

- `make_data.py` regenerates all three byte-identically; a test checks this.
- The retailer is the fictional "Harrowgate Cycles". A test checks that no question
  text, order id, catalogue item, voucher or policy text is shared with E2, and no
  question text with E1 or E1b.
- Tests check that every accepted answer is one word, one number or one code; that
  running the required tools yields an accepted answer and no rejected one; and that
  the policy search returns the right passage for every policy question.

**2. One task changed so that its key is a code** (follows from §2 "answer keys that
are hard to paraphrase").
- `check_eligibility` now also returns a `decision_code`. For ineligible orders it is
  a code such as `NR-417`.
- C-inel questions ask for "the refund decision code, and the refund amount if one
  is due". The accepted answer is the code. The rejected answers are the item price
  and the price plus shipping, which an agent reports if it calculates a refund
  without checking.
- The tools' other behaviour is E2's. Shipping is 8.40 in this database.

**3. Implementation details.**
- **`eligibility_tool` = missing:** the tool is left out of the schemas given to the
  model. If the model calls it anyway, the runtime answers "unknown tool".
- **Contrasts are signed.** E2 labelled with the CI of the absolute difference. E2b
  uses the CI of the signed difference for both contrasts, and also for the one-sided
  comparison labels. For a large effect the two agree; near zero the signed interval
  is the stricter test for "flat".
- **Key-suspect questions** are removed before scoring, for the gated criteria and
  for the one-sided comparison alike. `run.py` lists them in `criteria`.
- **Pilot check:** per targeted dev question, both contrasts are reported; the effect
  counts as present if either moves by ≥ 0.20 in the designed direction.
- **Block sizes:** gate 10, pilot 30, reference 50, main 30, recovery 60, as in E2.
- **Stored answers:** the first 3 final answers of each reference cell, cut to 300
  characters.
- Everything else (methods, B1 and B2, recovery, scoring functions) is E2's code,
  copied into this folder with the factor names changed.

**4. Library.** No change. E2b runs on `glassbox` 4.8.0 as released (commit
`ed6819b`).

**5. Toy dry run** (disclosed; not evidence).
- The full pipeline ran once with `ToyAgentModel`, a scripted policy that is **not a
  language model**: the pilot on all 12 dev questions, and the test stage on all 24
  test questions at the protocol's sample sizes, on CPU, in about 2 minutes.
- Every stage completed, and scoring produced G1–G10 under both label sets.
- The toy's verdict was FAIL on G5 (11 of 16): the toy answers "not sure" 5% of the
  time, so its baseline is noisy and several fixes fell just under the −0.15 bound.
  The toy numbers are not results.
- **No data file, threshold, sample size or factor value was changed after the dry
  run.** The tests and the dry run read the test split, as in E1b and E2.

**6. Software.** As in E2: the CI pins plus `accelerate` and `huggingface_hub`, on
the CUDA build of `torch==2.10.0`. Recorded in `results.json` and `pip_freeze.txt`.

## Addendum 2 (2026-11-14): pilot 1 results and the one wording revision

Written after pilot 1 and before any test-split run. Pilot 1 read the dev split only.
A re-pilot follows (pilot 2); its results go in Addendum 3, also before the test run.

**1. Pilot 1, as run** (commit `fba2b4356434f61170486870a938a3d0561d1411`, Colab,
NVIDIA A100-SXM4-80GB). Files committed unchanged in `pilot1/`:

| File | sha256 |
|---|---|
| `pilot1/results.json` | `7512441c5c86e41e01ed340006d22d8c0d6755df42f43ed3082b2288f3a5c5d3` |
| `pilot1/run.log` (`e2b_pilot1.log` on the host) | `57f5ee3dc3833e88b5c344c80eb13e22e7e7f55c3cd24ae36924a6fda5be0900` |
| `pilot1/checkpoint/gate.json` | `d41e3e411a46bf47867ad1b9a05bcb11f0e9c8d944378ba0c0a144c72139121f` |
| `pilot1/checkpoint/pilot.json` | `c60d64adbb8807d25b3a7c12f9f041513e4057e64d75e78028968ac1c0a31491` |
| `pilot1/pip_freeze.txt` | `ba4ccb9191c77002136eafb60b7427a5bc8d88f0fccda30cca5f14d2ff108ec6` (same as E2) |

- **Gate:** rung 1 (Qwen2.5-7B-Instruct, bf16) passed: accuracy **0.825** (bar 0.80),
  parse rate 1.000, n = 120. The margin is small; the C-inel dev questions pulled it
  down (item 3).
- **Speed:** 7,200 episodes in 1,716.6 s, i.e. 15,099 episodes per hour. Projection
  for the 118,440 remaining episodes: 7.88 h on this card, assuming test episodes take
  as long as pilot episodes.
- Later, the owner ran the whole notebook again by mistake on a new runtime (an L4).
  The pilot cell then loaded the saved gate and pilot units and computed nothing; the
  log and zip on Drive were rewritten. The files above were copied before that
  happened. The re-zipped upload was not used.

**2. Effect check** (both contrasts, 30 episodes per cell, dev questions of the
targeted type).

| Factor | Alone (`only_f` − baseline) | In the release (candidate − `all_but_f`) | Present |
|---|---|---|---|
| `invoice_tool_desc` | b_3306 −1.00, b_3329 −0.87 | −0.93, −0.93 | yes |
| `eligibility_tool` | ci_4502 −0.13, ci_4526 −0.17 | −1.00, −0.93 | yes |
| `max_steps` | d_fellgate −1.00, d_marlstone −1.00 | −1.00, −1.00 | yes |
| `refund_fix` (helpful) | cd_5703 +1.00, cd_5727 +1.00 | +1.00, +1.00 | yes |
| `quote_example` | 0.00, 0.00 | 0.00, 0.00 | **no** |
| `answer_policy` | e_1 −0.03, e_2 +0.07 | +0.07, −0.07 | **no** |

**3. What the dev diagnostic showed** (`pilot1/diag.py`, seed `e2b-diag`, run by the
owner on an L4; output pasted in the chat and kept in `pilot1/diag_output.txt`, so it
has no file hash from the host).

- **C-inel questions are mostly answered wrongly by the baseline itself (13% and 17%
  in pilot 1).** In 2 of 3 baseline episodes the agent called `check_eligibility`,
  then also `calc_refund`, said the order was not eligible, left out the decision
  code, and added the refund amount the order "would" get (33.50 EUR). Both omissions
  make the answer wrong under the keys written in Addendum 1.
- **This is the agent, not a broken key.** The question asks for the decision code;
  the answers did not give it, and they volunteered a refund amount that is not due.
  So the keys are left as they are.
- **Consequence, stated now:** on the test split the C-inel questions will probably
  meet the pre-registered key-suspect rule (baseline < 0.50, failures labelled
  `other` because the required tools were called in order), and be excluded from
  every criterion. If all four are excluded, E2b stays scoreable (the limit is 4),
  but `skipped_tool` cannot be produced, so E2b will make no claim for it. Changing
  the questions or the baseline prompt now would change sealed data or the baseline
  system, which §9 does not allow, so that is not done.
- **`persona` is not harmless.** With only the polite line added, C-inel went from
  13–17% to 83–97% (the agent stopped calling `calc_refund` and gave the code). The
  two-sided labels will say what it does on the test split.
- **`quote_example`:** the agent still applied the voucher before computing the
  total, three times of three.
- **`answer_policy`:** the agent still called `search_policy`, three times of three.
  But in the full candidate with only the eligibility tool restored (e_2), it
  answered without any tool call, three of three ("30" or "35" EUR). The policy text
  can work in combination with other changes; alone it was ignored.

**4. The one wording revision** (§9: text factors with no effect, once). Only the
candidate text of two factors changed, in `system.py`.

| Factor | Pilot-1 candidate wording | Revised candidate wording |
|---|---|---|
| `quote_example` | "Worked example of a quote: call start_quote, then compute_total, then apply_voucher, and report the total that compute_total returned." | The same, plus: "apply_voucher finalises the quote: after it, make no more tool calls and give the answer." |
| `answer_policy` | "Answer policy questions from your knowledge of standard retail practice. Use tools only for order data." | "Answer policy questions from your knowledge of standard retail practice. Do not call search_policy: it is reserved for staff and is being removed. Use tools only for order data." |

- Both state false things on purpose (a release whose prompt went wrong). The tools
  are unchanged.
- **Disclosed risk:** written after reading the dev diagnostic, so fitted to this
  model's dev behaviour. Not tried on any test question.
- This is the only revision. After pilot 2 the wordings stay, effect or not.
- Baseline texts, `invoice_tool_desc`, `persona`, all non-text factors, data, keys,
  thresholds, sample sizes and criteria are unchanged.
- One test was added (`test_addendum2_revised_wording`); the COLAB.md test count is
  now 115.

## Addendum 3 (2026-11-14): pilot 2 results; the configuration for the test run

Written after pilot 2 and before any test-split run. Pilot 2 read the dev split only.

**1. Pilot 2, as run** (commit `4a252eb0dd7f31b2f0d855000fa47f4f97fe6bab`, Colab,
**NVIDIA L4**). Files committed unchanged in `pilot2/`:

| File | sha256 |
|---|---|
| `pilot2/results.json` | `07d26f130d278eb51432fb63272629283c5a8f6d0e8e8fe620b513ce0fceb3c7` |
| `pilot2/run.log` (`e2b_pilot2.log` on the host) | `2b2dc8f11ff845d6340be39610f937662fbbdab34164fc502c3ae1e41e5d47c8` |
| `pilot2/checkpoint/gate.json` | `22c503456254d223af93ed0b85e40a7ec77ed09bf4d3a64efa9a840e8358e3a5` |
| `pilot2/checkpoint/pilot.json` | `a0782a9efb814f830b7414ee765e81d4ab1131f6f0da7bfb8ed895336443579b` |
| `pilot2/pip_freeze.txt` | `0f2bc48481a4cff2d631032555f57367e755742a7c172ce9c94e9425a748c853` |

- **Different GPU from pilot 1** (A100 80 GB there, L4 here). Same seeds; sampled
  outputs can differ between cards.
- **One package differs from pilot 1:** `cuda-pathfinder` 1.9.0 instead of 1.8.6 (a
  dependency of `cuda-bindings`, resolved at install time). Everything else in
  `pip_freeze.txt` is identical.
- **Gate:** passed again: accuracy 0.833, parse rate 0.995 (n = 120).
- **Speed:** 7,200 episodes in 6,038 s, i.e. 4,293 per hour on the L4.

**2. Effect check** (both contrasts; 30 episodes per cell).

| Factor | Alone | In the release | Present |
|---|---|---|---|
| `invoice_tool_desc` | −1.00, −0.87 | −0.90, −1.00 | yes |
| `eligibility_tool` | −0.17, −0.03 | −0.97, −0.97 | yes |
| `max_steps` | −1.00, −1.00 | −1.00, −1.00 | yes |
| `refund_fix` (helpful) | +1.00, +0.97 | +1.00, +1.00 | yes |
| `answer_policy` (revised) | e_1 −0.77, e_2 −0.93 | −0.87, −0.97 | **yes** (it was not in pilot 1) |
| `quote_example` (revised) | 0.00, 0.00 | 0.00, 0.00 | **no** |

Not adjustable:
- `persona`: C-inel +0.80 and +0.73 alone (it helps there, as in pilot 1); E +0.10
  and −0.13.
- `retries`: A −0.07 and −0.13.
- `log_level`: at most ±0.17 alone (C-inel ci_4502 +0.17, e_1 +0.10 on 30 episodes).

**3. No further changes.** The one revision round is used (§9).
- `quote_example` stays as revised, although it showed no effect. If the reference
  run confirms that, `wrong_order` is not produced and E2b makes no claim for it.
- C-inel baselines were 0.17 and 0.03 again. Addendum 2 item 3 stands: those
  questions will probably be key-suspect on the test split and excluded, so
  `skipped_tool` is probably not produced, although `eligibility_tool` clearly
  breaks the release (in-release contrast −0.97 on both dev questions).
- The test run uses the code of this commit.

**4. What to expect from the test run, stated in advance.**
- **Likely produced:** `wrong_tool` (B), `premature_finish` (D), `no_action` (E).
- **Likely not produced:** `wrong_order` (`quote_example` had no effect) and
  `skipped_tool` (C-inel questions probably excluded).
- **Likely key-suspect:** the four C-inel test questions. Four is within the limit, so
  E2b stays scoreable; more than four would make it INCONCLUSIVE.
- **`persona` will probably be labelled helpful on C-inel** (where its questions are
  excluded anyway) and is unclear elsewhere. G3 is not testable below 4 null `persona`
  pairs.
- The interactions in E are large: with the revised policy in the candidate, every
  `all_but_f` cell except `all_but_answer_policy` scored 0.00 on both dev questions.
  So in the release, only `answer_policy` is necessary for the drop, which is what the
  two-sided labels are built to see.

**5. Runtime for the remaining 118,440 episodes.**

| Card | Measured speed | Projection |
|---|---|---|
| A100 80 GB (pilot 1) | 15,099 episodes per hour | 7.84 h |
| L4 (pilot 2) | 4,293 episodes per hour | 27.6 h |

The test run is on an A100 (§10). Sample sizes are unchanged.
