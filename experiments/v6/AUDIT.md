# V6 repository audit and gap analysis

*Written 2026-09-28 before any V6 code change. Spec: `docs/PLAN_V6_FINAL.md`, revision 3
(frozen). Note: that file is git-ignored (private strategy), so every scientific definition
V6 depends on is restated in `experiments/v6/PREREGISTRATION.md`, which is committed.*

## A. Baseline

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD | `b80bdee` fix(mcp): pin mcp<2 (preceded by `b22a8ba` security hardening) |
| Test baseline | CI run on `b22a8ba` (GitHub Actions, real torch + TransformerLens): **Test + coverage job passed**, lint passed, MCP job failed (mcp 2.x rename; pinned in `b80bdee`). The local "no tests ran" result was a zsh `#` artifact and is not a baseline. |
| Local sandbox | Python 3.10.12; torch installed separately for this session (CPU wheel). |
| Package pins | `torch>=2.0,<2.11`, `transformer_lens>=1.0,<3.0`, `numpy>=1.24,<2.0`, `scipy` (core dep). |
| CLI entry point | `glassbox-ai` (`glassbox/cli.py`). The plan's `glassbox diff` is implemented as the subcommand `glassbox-ai diff`; no new console script is added. |
| Repo hazard found | Git commands run from the sandbox leave `.git/index.lock` behind (the sandbox cannot unlink it). This caused the stale July lock. The sandbox now uses only `git --no-optional-locks` read commands. |

## B. Seven layers → existing modules

| Layer | Existing modules | V6 status |
|---|---|---|
| 1 Observe | `audit_log.py` (hash-chained JSONL), `telemetry.py` | No experiment record with environment/dataset/config hashes. |
| 2 Represent | none | Out of scope for v6.0. |
| 3 Compare | `circuit_diff.py` (head-set diff, Jaccard), `cross_model.py` (normalised-position Jaccard + Pearson for *different* architectures) | No performance/behavioral/mechanistic distance triple; no TOST; no pair-level comparison of same-architecture models. |
| 4 Investigate | `core.py` `GlassboxV2.attribution_patching` (per-head Taylor/IG attribution), `minimum_faithful_circuit`, `_name_swap`, `_decision_value`; `acdc.py`, `causal_scrubbing.py`, `das.py`, `causal_abstraction.py` | Reused as the D_M instrument. |
| 5 Experiment | `corruption.py` (multi-strategy), `prompt_corruption.py`, `cf_gate.py`, `validation.py` (sample-size gate, held-out split) | Controls A–F not implemented as a hierarchy; no `falsify`. |
| 6 Discover | none | Out of scope for v6.0 (by plan). |
| 7 Prove | `evidence_tier.py` (tiers A–D), `evidence_vault.py`, `compliance.py`, `distributional.py` (`bootstrap_ci`), `fdr.py` (BH) | No Claim/Finding object, no evidence-ladder states incl. `UNRESOLVED`, no model-level bootstrap, no permutation/Mantel. |

## C. Reusable components (exact)

| Need | Reuse | Notes |
|---|---|---|
| Per-head attribution | `GlassboxV2.attribution_patching(clean_tokens, corrupted_tokens, target_token, distractor_token, method)` → `(Dict[(l,h)] -> float, clean_ld)` | The D_M input vector. Averaging across prompts is a new, pre-registered analytic choice. |
| Decision value | `glassbox.core._decision_value` | Logit difference used by D_P. |
| Corruption | `GlassboxV2._name_swap` | IOI counterfactual; also a multiverse dimension via `corruption.py`. |
| Circuit | `GlassboxV2.analyze(...)["circuit"]`, `circuit_diff` Jaccard | Secondary D_M (top-k Jaccard) computed directly from attribution ranks for consistency. |
| Bootstrap | `distributional.bootstrap_ci` | Resamples values, not models; model-level bootstrap is new. |
| FDR | `fdr.apply_fdr_correction` / `BenjaminiHochberg` | For secondary endpoints later. |
| Sample-size gate | `validation.SampleSizeGate` | For the pilot/confirmatory runs. |
| Evidence tiers | `evidence_tier.EvidenceTier` | Different concept from the V6 ladder; kept separate, cross-referenced. |
| Audit chain | `audit_log.AuditLog` | Hook-up of V6 records is a later step. |
| IOI prompts | `benchmarks/run_ioi.py` (5 fixed prompts) | Too few and unversioned; V6 needs a deterministic, hashed generator. |
| Credit task | `experiments/decision_audit/credit_rule.py`, `generate_credit_data.py` | Basis for Arm B; the correlated-feature known-positive does not exist yet. |

## D. Gap analysis

| V6 requirement | Status |
|---|---|
| D_P with paired TOST | MISSING |
| D_M primary (1 − Spearman on per-head attribution) | MISSING (attribution itself EXISTS) |
| D_M secondary: top-k Jaccard | REQUIRES MODIFICATION (exists on circuit sets, recomputed from attribution ranks) |
| D_M secondary: intervention-transfer loss | MISSING |
| D_B mean JSD over fixed probe set | MISSING (`BlackBoxAuditor` probes APIs, not open-weight output distributions) |
| Versioned IOI dataset + probe set | MISSING |
| Arm A Pythia checkpoint loading | EXISTS in TransformerLens (`checkpoint_value`), not wired |
| Arm B correlated-feature known-positive + correlation-breaking test | MISSING |
| Control A pipeline null with tolerance | MISSING |
| Controls B, C, E | MISSING (random baselines exist only for faithfulness) |
| Permutation test over model labels (H1), Mantel + sensitivity (H3), model bootstrap, held-out by model (H4) | MISSING |
| AUROC (H2) | MISSING |
| Multiverse over circuit-selection / corruption | REQUIRES MODIFICATION (Paper 1 methods, not wired) |
| Claim/Finding schema with scope + `UNRESOLVED` | MISSING |
| Experiment record with provenance hashes | MISSING |
| `diff` CLI | MISSING |
| `falsify`, `reproduce` | MISSING (later milestones) |
| Pre-registration | MISSING |

## E. Smallest vertical slice (milestone 1)

One task (IOI), two models (Pythia checkpoints), D_P + D_M + D_B, one control (Control A),
Claim/Finding, experiment record, reproducible artifact, tests. New code lives in
`glassbox/v6/` and a `diff` subcommand; the core engine is not modified.

Milestone 1 cannot test H1–H4: those need many pairs and the Control B null. Its Finding is
therefore recorded as `OBSERVED` for the measurements and `UNRESOLVED` for every hypothesis.
