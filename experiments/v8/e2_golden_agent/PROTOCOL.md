# V8 phase E2: golden agent case and agent fault taxonomy. PROTOCOL

*Protocol v1.0, written 2026-11-09, before any E2 code, data or run. It is committed
first and not changed afterwards. Any deviation goes in a dated addendum at the end.
Order: this protocol → code and tests (data, sealed test-split hash, Addendum 1) →
gate → dev pilot (Addendum 2) → test-split run → results committed unchanged →
RESULTS.md and the ASSURANCE rows. FAIL and INCONCLUSIVE are reported as such.*

## 0. Why E2 exists

- **PLAN phase E2 asks for three things:**
  - a real LLM tool-using agent;
  - coverage of five agent faults (wrong tool, skipped tool, wrong order, premature
    finish, answering without acting), each with negative controls;
  - the full chain, up to fix verification.
- **E1b left one question open** (RESULTS §6, ASSURANCE V3.13). Its case was
  all-or-nothing, with no helpful and no small effects. So G4 (helpful changes kept)
  could not be tested, and E1's undirected rules gave the same answers. E2 builds in:
  - a change that **helps**: a tool bug fix;
  - a change designed to have a **small** effect: a retry setting;
  - questions where a harmful and a helpful change act **together**, so reverting
    every attributed change (E1's rule) and reverting only harmful changes (E1b's rule)
    should give different results. G8 tests this.
- **E1 stays FAIL. E1b's scope is unchanged.** E2 reuses neither experiment's data.
- **Known risks, disclosed:**
  - the author designs the faults and the data, after seeing E1 and E1b. The
    case is not blind (that is E4);
  - the dev pilot (§9) may revise factor wording once. It never touches the test split.

## 1. Question

A release of a tool-using agent changes several things at once, and some questions get
worse. Does Glassbox's directed investigation:
- name the changes that hurt;
- keep the change that helped;
- clear the visible and the inert changes;
- label each failure with the right fault type;
- give a fix that restores the agent;
- beat scripted versions of common practice on the same run budget?

## 2. Agent system

- **A minimal agent loop written for E2.** No agent framework is used: framework
  portability is E3. At each step the model either emits a tool call or a final answer.
  - Tool calls use the model's own chat-template tool format: Qwen's `<tool_call>`
    JSON. Tool schemas are passed via `tools=`.
  - A step that does not parse as a valid tool call is taken as the final answer. The
    parse-failure rate is reported.
  - When the step budget is reached, the runtime appends one fixed message ("Step
    limit reached. Give your final answer now.") and takes the next output as the
    final answer. That forced step counts within the budget, so `max_steps` is the
    total number of LLM calls.
- **Generation:**
  - temperature 0.7, top-p 1.0, at most 128 new tokens per step;
  - bf16, SDPA attention;
  - episodes are batched in blocks of 50 and advanced step by step;
  - each episode has its own seed, derived from (stage seed, question id, cell, run
    index).
- **Tools.** Deterministic Python functions over a fictional retailer's database
  (orders, invoices, quotes, policies). The company and every record are invented and
  written in the code commit. A test checks that nothing is shared with E1 or E1b.

  | Tool | Returns |
  |---|---|
  | `lookup_order(order_id)` | status, items, ship date. Has a transient error with p = 0.10 per call, seeded from the episode seed (stage seed `e2-tools`) |
  | `lookup_invoice(order_id)` | amounts, shipping fee, currency |
  | `check_eligibility(order_id)` | eligible yes/no, with the reason |
  | `calc_refund(order_id)` | refund amount. It does **not** check eligibility (true in every configuration) |
  | `start_quote(item, qty)`, `apply_voucher(quote_id, code)`, `compute_total(quote_id)` | `compute_total` uses the vouchers applied at the time it is called |
  | `search_policy(query)` | top-1 policy passage (keyword match, deterministic) |

- **Task types** and required tool sequences (written in each question's spec):

  | Type | Asks for | Required tools, in order | LLM calls if done right | Test | Dev |
  |---|---|---|---|---|---|
  | A | order status | `lookup_order` | 2 | 4 | 2 |
  | B | amount paid | `lookup_invoice` | 2 | 4 | 2 |
  | C-inel | refund, ineligible order | `check_eligibility` (then stop) | 2 | 4 | 2 |
  | C-def | refund, eligible defective item | `check_eligibility`, `calc_refund` | 3 | 4 | 2 |
  | D | quote total with a voucher | `start_quote`, `apply_voucher`, `compute_total` | 4 | 4 | 2 |
  | E | company policy fact that differs from common practice | `search_policy` | 2 | 4 | 2 |

  So there are 24 test and 12 dev questions. Each question has:
  - accepted answers;
  - rejected answers: the values a known fault would produce, e.g. the buggy refund,
    the pre-voucher total, or a generic policy value.

  A run is **correct** iff an accepted answer matches and no rejected answer matches.
  Matching is by tokens, as in E1.

## 3. The release: factors

The baseline is the old release, and the candidate changes all nine factors.

| Factor | Baseline | Candidate | Designed effect | Fault class |
|---|---|---|---|---|
| `invoice_tool_desc` | `lookup_order` described as status-only | described as "anything about an order, including amounts" | harmful on B | wrong tool |
| `refund_tool_desc` | `calc_refund` says "check eligibility first" | says "eligibility is checked automatically" (false) | harmful on C-inel; null on C-def (a negative control) | skipped tool |
| `runbook_order` | system prompt: "apply the voucher, then compute the total" | "compute the total, then apply the voucher" | harmful on D | wrong order |
| `max_steps` | 6 | 2 | harmful on C-def and D | premature finish |
| `answer_policy` | "Always use tools for company facts" | "If you know the answer, reply directly without tools" | harmful on E | answering without acting |
| `refund_fix` | `calc_refund` omits the shipping refund owed on defective items (bug) | bug fixed | **helpful** on C-def | — |
| `retries` | the runtime retries a tool error once | no retry | **small** harm on A | — |
| `persona` | one-line role | rewritten persona and tone paragraph (visible in the diff) | null (the visible suspect) | — |
| `log_level` | info | debug | null by construction (inert) | — |

- Every designed effect is **intended**, not assumed. The reference run (§4) decides
  the labels, and the dev pilot (§9) checks the effects on dev data only.
- The question types give every fault factor **negative-control pairs**: types where it
  should not matter. These pairs are scored as null pairs when the reference labels
  them null.
- C-def has a harmful factor (`max_steps`) and a helpful one (`refund_fix`) acting
  together. That is the case E1's revert-everything rule should get wrong (G8).
- **Seeds:** `e2-gate`, `e2-pilot`, `e2-reference`, `e2-main`, `e2-recovery`,
  `e2-tools`.

## 4. Ground truth (reference run)

- Seed `e2-reference`. **200 episodes per cell per test question.** The cells are
  `baseline`, `candidate` and each `only_<f>`: 11 cells.
- **Pair labels** use the E1b rules unchanged (E1b PROTOCOL §4). The 95% percentile
  bootstrap CI of the accuracy difference uses 2,000 resamples. The labels are:
  - harmful-large: lower bound ≥ 0.20, `only_f` < baseline;
  - helpful-large: lower bound ≥ 0.20, `only_f` > baseline;
  - null: upper bound ≤ 0.10. `log_level` is null by construction;
  - otherwise small/uncertain: reported, not scored.
- **Regressed question:** the 95% bootstrap CI lower bound of baseline − candidate
  accuracy is ≥ 0.20.
- **Trajectory truth.** Every episode's tool sequence is stored. It is the input to the
  fault labeller (§5) and is reported per cell.

## 5. Library additions (tests first)

1. **`glassbox/v7/taxonomy.py`: fault labeller.**
   - Input: one episode's trace and the question's required tool sequence.
   - It labels a **failing** episode with exactly one class. The first matching rule
     wins:
     1. `no_action`: zero tool calls;
     2. `premature_finish`: the step budget was reached and at least one required tool
        was never called;
     3. `wrong_tool`: at least one tool outside the required set was called, and at
        least one required tool was never called;
     4. `skipped_tool`: the episode ended on its own, every call was to a required
        tool, and at least one required tool was never called;
     5. `wrong_order`: every required tool was called, but their first calls are out
        of the required order;
     6. `other`: everything else, such as the right tools in the right order with a
        wrong answer, an unhandled tool error, or a parse failure.
   - Correct episodes get no label.
   - Tested on hand-written traces for every rule and every tie.
2. **Agent trace recording.**
   - LLM and tool spans use the existing v7 trace recorder.
   - Tool names, the order of calls, argument keys, error flags and the forced-final
     flag are recorded in clear.
   - Text content is hashed, as in E1.
3. **`investigate` with a fault view.**
   - When the incident gives required tool sequences, each `CaseFinding` also carries
     the label counts of the failing runs in each cell.
   - Without them, output is unchanged (tested).
   - The direction view from E1b (`harmful`, `helpful`) is unchanged.

## 6. Methods compared (the main run)

- `run_interventions` cells: baseline, candidate, `only_f` and `all_but_f`, i.e.
  20 cells.
- **30 runs per cell**, α = 0.05, 999 permutations, Holm within each family, seed
  `e2-main`.

| ID | Method | Blame set |
|---|---|---|
| **GB** | `investigate` with the direction view and `target="correct"` | harmful list |
| **B0** | blame every change | all nine factors |
| **B1** | one run per side plus the trace diff; blame what the first divergence touches | a divergence in the LLM input → every prompt-visible factor (`invoice_tool_desc`, `refund_tool_desc`, `runbook_order`, `answer_policy`, `persona`); in a tool result → `refund_fix`, `retries`; in episode length → `max_steps`. `log_level` is never prompt-visible |
| **B2** | revert one change at a time, one run each (as E1/E1b) | factors whose single reverted run is correct while the candidate run is wrong |

- GB's fault label for a pair = the most common label among the failing runs of its
  `only_f` main-run cell. If that cell has fewer than 5 failing runs, the pair has no
  label, and this is counted.
- Not tested: LLM-judge analysis and commercial tools (E6/E7), and a human workflow
  (E4).

## 7. Fix and recovery

- **Recovery rerun:** seed `e2-recovery`, **60 episodes per arm** on every test
  question. The arms are:
  - `baseline` and `candidate`;
  - `gb_fix`: revert GB's harmful factors;
  - `wrong_fix`: revert `persona` only;
  - `b2_fix`: revert B2's blamed factors;
  - **`undirected_fix`** (new): revert GB's undirected `attributed` list. This is E1's
    rule.

  Arms with the same configuration on a question share runs. Arms with nothing to
  revert are not run.
- **Verified recovery** is E1b's one-sided rule, unchanged:
  - the fix beats the candidate, one-sided permutation p < 0.05;
  - the lower bound of the 95% bootstrap CI of (fix − baseline) is ≥ −0.15.

  If GB's harmful list is empty on a regressed question, it counts as not verified.
  E1's two-sided check is reported descriptively.

## 8. Criteria (test split; gated)

| ID | Criterion | Pass rule |
|---|---|---|
| G1 | Sensitivity to harm | GB's harmful list contains ≥ 0.75 of harmful-large pairs |
| G2 | Inert never blamed | `log_level` in the harmful list on ≤ 2 of 24 questions |
| G3 | Visible suspect cleared | ≤ 0.10 of null `persona` pairs in the harmful list. Not testable below 4 pairs |
| G4 | Helpful change kept | ≤ 0.10 of helpful-large pairs in the harmful list. Not testable below 4 pairs |
| G5 | Fix verified | GB's fix verified on ≥ 0.80 of regressed questions |
| G6 | Wrong fix rejected | `wrong_fix` not verified on ≥ 0.80 of regressed questions where `persona` is not harmful-large. If none are eligible: not passed (as in E1b Addendum 1) |
| G7 | Better than today's workflow | F1 over harmful-large (positive) and null (negative) pairs: GB beats each of B0, B1 and B2 by ≥ 0.10 |
| G8 | Directed fix keeps the gain | See below. ≥ 0.75 of eligible questions. Not testable below 4 |
| G9 | Fault type named | For harmful-large pairs of the five fault factors, GB's label equals the factor's class (§3) on ≥ 0.75 of the labelled pairs. Not testable below 6 labelled pairs |
| G10 | Negative controls hold | ≤ 0.10 of all null pairs in the harmful list |

**G8 in detail:**
- Eligible questions have both a helpful-large pair, and `gb_fix` and `undirected_fix`
  arms that differ.
- A question passes if the 95% bootstrap CI lower bound of (`gb_fix` − `undirected_fix`)
  accuracy is ≥ 0.20.

**Overall:**
- **PASS** iff:
  - a rung passed the gate;
  - every testable criterion passes;
  - there are ≥ 6 harmful-large pairs;
  - there are ≥ 6 regressed questions.
- **INCONCLUSIVE** if a count is below 6, or no rung passes the gate.
- **FAIL** otherwise, with every failing number reported.
- Testability is decided only by the counts. It is reported, not chosen.

**Coverage (reported, not gated).**
- A fault class is **produced** if its factor has ≥ 2 harmful-large test pairs.
- E2 may claim to cover only produced classes. PLAN's coverage item stays open for any
  class not produced.

**Descriptive (not gated):**
- misses and false alarms per method, with Wilson 95% CIs;
- per-factor tables of harmful-large, null and blamed pairs;
- the fault-label confusion matrix;
- fix-verified rates for B2 and the undirected fix;
- E1's two-sided check for every arm;
- small/uncertain pairs, including what happened to `retries`;
- B2 re-draw flips (5 extra seeds);
- `log_level` TVDs;
- parse-failure and forced-final rates per cell;
- runtime per stage, generation counts and GPU type.

## 9. Competence gate and dev pilot

- **Models (licence checked 2026-11-09 on the Hugging Face model API; both list
  `apache-2.0`).** Both rungs run in bf16 with no quantisation. The chat templates of
  both include tool calling.
  - Rung 1: `Qwen/Qwen2.5-7B-Instruct`, revision
    `a09a35458c702b33eeacc393d103063234e8bc28`.
  - Rung 2: `Qwen/Qwen2.5-14B-Instruct`, revision
    `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
  - The LICENSE file at each pinned revision is fetched in the code commit, and its
    sha256 goes in Addendum 1.
- **Gate:**
  - Seed `e2-gate`. 12 dev questions × 10 episodes, under the baseline with
    `refund_fix` applied, so that every question is answerable.
  - A rung passes if accuracy is ≥ 0.80 and the tool-call parse rate is ≥ 0.95.
  - The first passing rung is used. If neither passes, E2 is INCONCLUSIVE.
- **Dev pilot:**
  - Seed `e2-pilot`, gate-selected model, dev split only. Cells `baseline`,
    `candidate` and each `only_f` (11 cells), 30 episodes each.
  - It checks that each designed effect exists: at least one dev question of the
    targeted type has `only_f` accuracy at least 0.20 below the baseline (above it for
    `refund_fix`).
  - `persona`, `log_level` and `retries` are not adjusted, whatever the pilot shows.
  - A designed factor that shows no effect may have its **wording** revised **once**,
    then be re-piloted once. If it still shows no effect, it is kept as revised and
    reported.
  - Nothing else may change: no test data, thresholds, sample sizes, criteria or other
    factors.
  - Pilot outputs and any revision are committed in Addendum 2, **before** the
    test-split run. The test split is not read by the pilot.

## 10. Compute and execution

- **Where:** a rented GPU chosen and paid for by the owner. One 80 GB card (an
  A100/H100 class) is recommended so that either rung fits in bf16 without code
  changes. A smaller card is allowed only if the gate-selected model fits it in bf16;
  it is recorded.
  - The runbook is `RUNBOOK.md`, written in the code commit: a generic Linux GPU host,
    a uv venv with the CI pins, tmux, and a checkpoint per question on local disk
    copied out after each stage. A resume is allowed and disclosed.
- **Size:**

  | Stage | Episodes |
  |---|---|
  | Gate | 120 per rung |
  | Pilot | 3,960 |
  | Reference | 24 × 11 × 200 = 52,800 |
  | Main | 24 × 20 × 30 = 14,400 |
  | Recovery | ≤ 24 × 6 × 60 = 8,640 |

  That is about 80,000 episodes of 1–6 LLM calls each.
- **Runtime:** I do not know it, and it is not estimated here. The pilot measures
  episodes per hour on the rented card. Its projection for the remaining stages is
  written in Addendum 2, so the owner can decide whether to rent for the full run
  before it starts. **Sample sizes do not change** with that projection: the owner
  chose full size.
- **Order:** gate → pilot → Addendum 2 → per test question (reference, main,
  recovery) → scoring.
  - Main-run decisions never read reference labels.
  - Scoring code and its tests are committed before the gate runs.
- `uv pip freeze`, the GPU name and per-stage timings are written inside `--out`.

## 11. What a PASS would and would not show

- **Would show:** on one author-built tool-using agent with one open model, the
  directed investigation:
  - named the changes that hurt and labelled each with its fault type;
  - kept the bug fix that helped, where E1's rule would have reverted it;
  - cleared the visible and inert changes and the negative controls;
  - gave a fix that restored the regressed questions;
  - beat scripted common practice on the same budget.

  It would show this only for the fault classes actually produced.
- **Would not show:**
  - anything blind: the author designed the faults (E4);
  - other frameworks (E3), other models, or real production agents;
  - anything against commercial tools (E6/E7);
  - that engineers would use it (phase G);
  - that E1 passed.
- E2 becomes the golden agent case only on a PASS, and it is shown with E1's FAIL and
  E1b's scope next to it.

---
*(Addenda go below this line, with dates.)*

## Addendum 1 (2026-11-09): sealed data, implementation details, toy dry run

Written in the code commit, before any model run.

**1. Sealed files** (sha256). These must match the run's checkout.

| File | sha256 |
|---|---|
| `questions_test.json` (24) | `da640ddc3d080520d14e4f177f52040d4a592ba4b09fcf15603fe03168b6f897` |
| `questions_dev.json` (12) | `153d4b91d21337b6eb65b373cf2c276081f1aff213b52411d14820b02be4f8f2` |
| `db.json` | `17b92559694ed8dfd5fd69d3c9bda2f5a62ee0f976799767454faeb7786f3bb9` |

- `make_data.py` regenerates all three byte-identically; a test checks this.
- The retailer is the fictional "Quarrystone Outfitters". A test checks that no
  question, policy text or company name is shared with E1 or E1b.
- A test runs the required tools for every question (baseline, `refund_fix`
  applied, retries forced) and checks that the result contains an accepted answer
  and no rejected one.

**2. Clarification found while writing tests: required tools for C-inel.**
- The §2 table lists C-inel's required tools as "`check_eligibility` (then stop)".
- Under the §5.1 rules taken literally, an episode that skips the check and calls
  only `calc_refund` would be labelled `wrong_tool` (rule 3: a tool outside the
  required set). That contradicts §3, which designs `refund_tool_desc` as the
  skipped-tool fault.
- So the required sequence for **both** refund types is the refund workflow,
  `check_eligibility` → `calc_refund`. Correct episodes are never labelled, so a
  correct agent may still stop after the check.
- Known side effect, disclosed: a failing C-inel episode that checks and then
  answers wrongly without calling `calc_refund` is labelled `skipped_tool`, not
  `other`.

**3. Sampling** (replaces "each episode has its own seed" in §2).
- Episodes of one (configuration, question) are generated in blocks, advanced step
  by step in one batch. Each (block, step) has one sampling seed derived from (stage
  seed, canonical configuration, question, block, step). Per-sequence seeds are not
  possible in one batched `generate` call.
- Block sizes per stage: gate 10, pilot 30, reference 50, main 30, recovery 60, so
  that no episodes are generated and thrown away.
- Tool randomness (the transient error of `lookup_order`) uses a per-episode seed
  from (stage seed, "tools", configuration, question, episode index). It does not
  depend on batching.

**4. Outcome matching.**
- §2 says matching is by tokens "as in E1". E1's tokenizer treats only ASCII
  punctuation as separators, so "€59.95" would not match "59.95".
- E2 lowercases and treats every character other than a–z and 0–9 as a separator.
  A test covers the euro sign, rejected answers and contractions ("isn't eligible").

**5. Implementation details.**
- **Permutations.** With 9 factors, Holm's first threshold is 0.05 / 9 ≈ 0.0056.
  999 permutations give a smallest p-value of 0.001, so the protocol's 999 is
  enough. Fewer (for example 99) could never reach significance. This is noted
  because a test with 99 permutations found nothing significant.
- **B1.**
  - A first divergence that is a step present in only one run (`added` or
    `removed`) → `max_steps`.
  - A divergence in a `chat` step → the five prompt-visible factors.
  - A divergence in an `execute_tool` step → `refund_fix`, `retries`.
  - The chat spans hash the system prompt and the tool definitions. So whenever a
    prompt-visible factor differs, B1's first divergence is the first chat step,
    and it blames all five prompt-visible factors.
- **GB's fault label** for a pair is the most common label among the failing runs of
  the main-run `only_<f>` cell. Below 5 failing runs it is `None`. Ties go to the
  label listed first in §5.1.
- **G8 eligibility.** A question is eligible if it has a helpful-large pair and
  `gb_fix` and `undirected_fix` both ran with different reverted sets.
  - If either arm was not run, the question is not eligible.
  - For eligible questions, the (gb − undirected) CI uses the two arms' 60
    recovery episodes.
- **G10** is testable with at least one null pair. **G9** needs at least 6
  labelled pairs, as §8 says.
- **Gate parse rate** = valid tool calls / (valid + failed `<tool_call>` blocks) over
  the gate episodes. With no tool calls at all it is 0, and the rung fails.
- **Recovery** reuses runs between arms with the same configuration, as in E1b.
- **Memory.** Cached episode blocks are cleared after each question and stage.

**6. Library changes (tests first).**
- `glassbox/v7/taxonomy.py` implements §5.1, with 19 tests in
  `tests/test_v7_taxonomy.py`.
- `Incident.required_tools` is optional and needs a target. With it, each
  `CaseFinding` has `fault_counts` per cell, and the report gains a "Failure types"
  table. Without it, output is unchanged (4 new tests in
  `tests/test_v7_investigate.py`).

**7. Model licences** (Hugging Face API, 2026-11-09).
- At both pinned revisions, `LICENSE` is git blob
  `6634c8cc3133b3848ec74b9f275acaaa1ea618ab`, 11,343 bytes. Its text is the Apache
  License 2.0 with "Copyright 2024 Alibaba Cloud".
- Neither repository is gated.
- §9 asked for the LICENSE sha256 here. The fetch tool used returns decoded text,
  not bytes, so I could not hash the exact bytes. Instead, `run.py` downloads
  `LICENSE` at the pinned revision on the GPU host and records its sha256 per rung
  in `results.json` → `gate.rungs[].licence_sha256`. The git blob id above
  identifies the same bytes.

**8. Toy dry run** (disclosed; not evidence).
- The full pipeline ran once with `ToyAgentModel`, a scripted policy that is **not a
  language model**:
  - pilot on all 12 dev questions;
  - test stage on all 24 test questions at the protocol's sample sizes;
  - on CPU, in about 2 minutes.
- The toy passed the gate unchanged (accuracy 0.858, parse rate 1.0).
- Every stage completed, and scoring produced G1–G10. On the toy, G8 had only 1
  eligible question (not testable).
- The toy numbers are not results, and **no data file, threshold, sample size or
  factor wording was changed after the dry run.**
  - One toy-only bug was fixed before the full dry run: the toy re-drew its quote
    order at every step.
  - The tests and the dry run read the test split, as in E1b.

**9. Software.**
- The CI pins plus `accelerate` and `huggingface_hub`, on the CUDA build of
  `torch==2.10.0` on the GPU host. They are recorded in `results.json` → `software`
  and in `pip_freeze.txt`. The GPU name is recorded in `results.json` → `gpu`.
