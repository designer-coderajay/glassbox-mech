# Glassbox — Assurance: what is built, what is evidenced, what is only planned

*Created 2026-10-05 (V7 Step 2). Package version `glassbox-mech-interp` 4.5.1.*

This file prevents claims from running ahead of evidence. Every public statement
about Glassbox should trace to a row here. If a claim has no row, it is not yet a
claim. `tests/test_assurance.py` enforces the table's structure:
- statuses must come from the taxonomy below;
- IDs must be unique;
- cited evidence files must exist;
- no row may claim independent reproduction until one has happened.

Implementation status and scientific evidence are separate axes. A feature can be
`IMPLEMENTED` (the code exists and is tested) while the scientific claim it supports
is only a `HYPOTHESIS`.

**Integrity is not truth.** Hashes (e.g. the V6 freeze manifest) show only that an
artefact has not changed since it was hashed. They do not show that the experiment
was correct.

## Status taxonomy (from the V7 plan, §18)

| Status | Meaning in this file |
|---|---|
| `VERIFIED` | Checked by an automated test or bitwise recomputation inside this repository. |
| `EXPERIMENTALLY_SUPPORTED` | Supported by a reported experiment, only under its stated conditions. Not checked by an independent party. |
| `REPRODUCED` | Reproduced by someone other than the author. **No row has this status yet.** |
| `PRELIMINARY` | Exploratory, pilot, or not re-checked since it was reported. |
| `IMPLEMENTED` | The code exists with tests. Says nothing about scientific validity. |
| `HYPOTHESIS` | Believed or intended, not yet tested. |
| `UNRESOLVED` | Tested or posed, with no conclusion reached. A valid result. |
| `REFUTED` | Tested and failed. |
| `PLANNED` | Not built. |
| `NOT_SUPPORTED` | Must not be claimed with the current evidence. |

## L — Legacy toolkit (v1–v5, pre-V6 standard)

These claims predate V6's stricter discipline and were not re-audited against it (V7
plan §70). A "faithful circuit" means a *candidate* circuit under the stated
procedure and thresholds.

| ID | Claim | Status | Evidence | Scope / limits |
|---|---|---|---|---|
| L1 | `analyze()` ranks attention heads by attribution patching (activation difference × gradient) and returns a candidate circuit | `IMPLEMENTED` | `glassbox/core.py`, `tests/test_core_coverage.py` | First-order Taylor approximation; the circuit is a ranked hypothesis, not the model's "true reason" (`docs/METHODOLOGY_AND_ASSURANCE.md` §1). |
| L2 | Sufficiency / comprehensiveness / F1 faithfulness metrics are computed by re-running the model with ablations | `IMPLEMENTED` | `glassbox/core.py`, `tests/test_compliance.py` | Exact vs approximate sufficiency is flagged (`suff_is_approx`). |
| L3 | IOI, GPT-2 Small, default circuit: suff 1.00, comp 0.543, F1 0.704, Grade B (1 head) | `EXPERIMENTALLY_SUPPORTED` | `BENCHMARKS.md`, `benchmarks/run_ioi.py` | One task, one model. The grade uses Glassbox's own thresholds, not an external standard. |
| L4 | Confidence and explanation faithfulness are essentially uncorrelated (r = 0.009) | `EXPERIMENTALLY_SUPPORTED` | `BENCHMARKS.md` | IOI on GPT-2 Small, as disclosed in the paper's limitations. Not shown to generalise. |
| L5 | r = 0.009 shows that Glassbox explanations are "mechanistically grounded" / "driven by causal circuit structure" | `NOT_SUPPORTED` | `BENCHMARKS.md` | A near-zero correlation shows only that confidence does not predict faithfulness. Says nothing about how explanations are produced. See open issue O1. |
| L6 | `analyze()` takes 3 forward/backward passes; 1.8 s on M1 Pro and 4.2 s on an 8-core CPU (GPT-2 Small); 15–37× faster than ACDC | `EXPERIMENTALLY_SUPPORTED` | `BENCHMARKS.md` | Hardware-specific timings. Not re-run in this audit. |
| L7 | On raw GPT-2 decision prompts (credit etc.) Glassbox reports low faithfulness and Grade C rather than a clean explanation | `EXPERIMENTALLY_SUPPORTED` | `BENCHMARKS.md`, `benchmarks/run_decision_functional.py`, `benchmarks/results/decision_functional_2026-10-05.json` | Re-run on 2026-10-05 (gpt2, taylor, single prompt per task) reproduced all five BENCHMARKS rows to 3 decimals. Single-prompt tiers are underpowered by design. |
| L8 | Generates EU AI Act Annex IV technical-documentation structure (9 sections, §8 human sign-off) | `IMPLEMENTED` | `glassbox/compliance.py`, `tests/test_compliance.py` | Produces documentation; it is not legal advice. |
| L9 | A Glassbox report makes a system legally compliant / conformity-assessed under the EU AI Act | `NOT_SUPPORTED` | `docs/METHODOLOGY_AND_ASSURANCE.md` | Legal conformity needs the provider's own assessment and, where applicable, notified bodies. |
| L10 | Modules with no direct test import found: alignment, causal_scrubbing, circuit_diff, corruption, hessian, hf_integration, large_model, layernorm_correction, mlflow_integration, polysemanticity, sae_attribution | `PRELIMINARY` | `glassbox/circuit_diff.py`, `glassbox/hessian.py`, `glassbox/sae_attribution.py` | Found by grep for direct imports in `tests/` on 2026-10-05; they may be tested indirectly. Not claimed as validated until a test cites them. |
| L11 | Full test suite: 1080 passed, 12 skipped, 2 xpassed; 72.99% line coverage (gate 58%) | `VERIFIED` | `docs/VALIDATION_LOG.md` | Local run on 2026-10-05 at commit b84f599, not CI. Supersedes "932 tests / 71%". Coverage measures lines executed, not correctness. |

## S — V6 scientific study (frozen)

| ID | Claim | Status | Evidence | Scope / limits |
|---|---|---|---|---|
| S1 | Head-indexed (positional) attribution distance is not invariant to function-preserving head relabelling: an exact functional twin of pythia-70m scores positional D_M = 1.04 | `VERIFIED` | `experiments/v6/audits/head_identifiability.md`, `tests/test_v6_identifiability.py` | One model (pythia-70m@143000), one attribution procedure. Shows that the metric is not identifiable. It does not show how large real cross-run differences are. |
| S2 | D\* (distance modulo within-layer head permutation, per-layer Hungarian) is invariant under weight-level head permutation | `VERIFIED` | `experiments/v6/audits/amendment3_gate_criteria.md`, `tests/test_v6_identifiability.py` | Gate 2 ran on pythia-410m-deduped. √D\* is the quantity with metric properties on orbits. |
| S3 | Per-pair D\* confidence intervals are calibrated | `REFUTED` | `experiments/v6/audits/amendment3_gate_criteria.md` | Gate 1 failed and is closed. Per-pair D\* values are point estimates only. |
| S4 | Run-level inference on Δ keeps type-I error ≤ 0.081 at R = 4, 5, 6, 10 (synthetic) | `VERIFIED` | `experiments/v6/audits/gate5_results.json`, `experiments/v6/audits/gate5_small_r_results.json` | R = 7–9 not simulated; calibration at R = 7 is assumed. |
| S5 | H1: independently trained, performance-matched Pythia-410M runs show greater attribution-profile distance than late checkpoints within a lineage (Δ = 1.50, one-sided 95% lower bound 1.27, R = 7) | `EXPERIMENTALLY_SUPPORTED` | `experiments/v6/RESULTS_confirmatory_v1.md`, `experiments/v6/runs/confirmatory_v1/record.json` | Preregistered; reproduced bitwise by the author from the stored cache (not independent). One size, one task, one attribution method. Training steps are confounded with lineage. |
| S6 | The V6 result shows different mechanisms or circuits across runs | `NOT_SUPPORTED` | `experiments/v6/RESULTS_confirmatory_v1.md` | Attribution-profile divergence ≠ mechanistic or circuit divergence (RESULTS §10). |
| S7 | Original hypotheses H2–H4 | `UNRESOLVED` | `experiments/v6/PREREGISTRATION.md` | Not tested. The original positional-D_M H1 was superseded before confirmatory data existed. |
| S8 | V6 artefacts are unchanged since the freeze (119 files) | `VERIFIED` | `experiments/V6_FROZEN.md`, `scripts/check_v6_frozen.py`, `tests/test_v6_frozen.py` | Integrity only, not correctness. |

## V — V7 investigation infrastructure

| ID | Claim | Status | Evidence | Scope / limits |
|---|---|---|---|---|
| V1 | A standard trace format for AI-system runs (OpenTelemetry GenAI-compatible) | `PLANNED` | `docs/PLAN_V7_FINAL_UNIFIED.md` | Not built. |
| V2 | Framework-independent behavioural / trajectory diff with first-divergence detection | `PLANNED` | `docs/PLAN_V7_FINAL_UNIFIED.md` | The existing `glassbox/v6/diff.py` is a V6 model-pair attribution diff, not a system-trace diff. |
| V3 | Hypothesis → controlled experiment → intervention → reproduction loop | `PLANNED` | `docs/PLAN_V7_FINAL_UNIFIED.md` | Not built. |
| V4 | Glassbox certifies causal explanations automatically | `NOT_SUPPORTED` | `docs/PLAN_V7_FINAL_UNIFIED.md` | Causal claims need a specified intervention experiment, case by case. |
| V5 | Glassbox saves skilled engineers meaningful time when an AI system behaves unexpectedly | `HYPOTHESIS` | `docs/PLAN_V7_FINAL_UNIFIED.md` | Zero external users so far. Test: five-user experiment (plan §77). |
| V6 | Investigation Benchmark of known-cause failures | `PLANNED` | `docs/PLAN_V7_FINAL_UNIFIED.md` | Not built. |

## O — Open issues found while writing this file (not fixed here)

- **O1 (over-claim).** `BENCHMARKS.md` "Confidence–Faithfulness Orthogonality" interprets
  r = 0.009 as showing that explanations are "mechanistically grounded" and "driven by
  causal circuit structure". That interpretation is not supported (row L5).
  - **Resolved 2026-10-05:** the BENCHMARKS paragraph was replaced by an explicit
    "does not show" statement. No number changed.
  - A grep found the phrase nowhere else in the README, docs or site.
- **O2 (discrepancy).** For `fraud_flag`, `BENCHMARKS.md` reports comprehensiveness
  0.077 / F1 0.141. The local file `reports/decision_functional.json` has 0.098 / 0.176.
  Sufficiency agrees (0.851 vs 0.8501).
  - That file is gitignored and dated 2026-06-13, two days before the BENCHMARKS
    rewrite (commit d2b0dcb, 2026-06-15). The BENCHMARKS figures probably come from a
    later run that was not kept.
  - That is an inference, not verified. Resolution: re-run and commit the raw output to
    a tracked path.
  - **Resolved 2026-10-05:** a fresh run, committed at
    `benchmarks/results/decision_functional_2026-10-05.json`, reproduces the BENCHMARKS
    values exactly, including fraud_flag 0.851 / 0.077 / 0.141 / 2.26×.
    - The gitignored 2026-06-13 file was therefore an earlier, superseded run. Its mean F1
      was 0.2124; the current run's is 0.2008. No published document quotes a mean F1.
  - (Corrected 2026-10-05: an earlier version of this file cited the gitignored file as
    evidence. `tests/test_assurance.py` now requires evidence files to be git-tracked.)
- **O3 (broken reference).** `BENCHMARKS.md` says to reproduce into
  `reports/credit_current.json`, which does not exist in the repository.
  - **Resolved 2026-10-05:** the reference now points to the committed results file.
- **O4 (unverified number).** "932 tests / 71% coverage" is not re-checked (row L11).
  - **Resolved 2026-10-05:** the full local run gives 1080 passed / 72.99% (VALIDATION_LOG
    Run 13). Public surfaces were updated to "1,080 tests, 73% coverage" without the
    "in CI" wording, because CI was not re-checked.
- **O5 (number mismatch).** The README speed row said "~37× faster than ACDC".
  `BENCHMARKS.md` gives 15–37× depending on circuit size and hardware, so ~37× is the
  upper end.
  - **Resolved 2026-10-05:** the README now says 15–37×.
- **O6 (open).** Two `xfail` tests in `tests/test_engine.py` now pass. Check whether the
  underlying issue was fixed, and remove the markers if so.
  - **Findings 2026-10-05 (static analysis only):**
    - The markers were added in baa821c (2026-06-14) for "L9H9 attribution inverts only
      when test_core_coverage.py runs earlier".
    - In the full run of 2026-10-05, that file did run earlier (alphabetical order), and
      both tests passed.
    - No module-level caches, global RNG use or global torch settings were found in
      `glassbox/core.py`. Each test module loads its own GPT-2 instance.
    - Later commits touched `core.py` (78e2c1d, 0666f6e, 09d3f97, c79ca01). Whether one
      of them fixed this is **not established**.
  - **Resolution rule:**
    - Run the trigger order (`test_core_coverage.py` then `test_engine.py`) several
      times.
    - Remove the markers only if every run passes. Otherwise keep them and investigate.
  - **Resolved 2026-10-05:** 3 of 3 dedicated runs in the trigger order passed, plus the
    full-suite run, so 4 of 4 runs showed no inversion. The two `xfail` markers were
    removed, and the tests are now strict again.
    - Caveat: the original root cause was never identified. It may have been fixed
      incidentally by a later `core.py` commit, or it may be rare and intermittent.
    - If the inversion ever recurs, these tests will now fail loudly, which is the
      intended behaviour.

## Update rule

1. When a capability ships or a result changes, update its row in the same commit, with
   the evidence path.
2. Never promote a status without new evidence. `PLANNED → IMPLEMENTED` needs tests.
   `IMPLEMENTED → EXPERIMENTALLY_SUPPORTED` needs an experiment with stated conditions.
3. Use `REPRODUCED` only when someone other than the author reproduces a result.
4. Marketing, README and website copy may only restate rows with status `VERIFIED`,
   `EXPERIMENTALLY_SUPPORTED` or `IMPLEMENTED`, with their scope.
