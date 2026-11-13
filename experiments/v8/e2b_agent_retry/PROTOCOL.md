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
