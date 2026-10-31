# Glassbox documentation

Start with the [project README](../README.md). This page maps every public document by
purpose. **Status of any claim:** [`ASSURANCE.md`](../ASSURANCE.md) is authoritative; if
another document disagrees with it, ASSURANCE wins.

## Using Glassbox

| Document | What it covers |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Full walkthrough, examples and API reference |
| [EU_AI_ACT_COMPLIANCE_GUIDE.md](EU_AI_ACT_COMPLIANCE_GUIDE.md) | Drafting Annex IV technical documentation, step by step |
| [MODEL_SUPPORT_MATRIX.md](MODEL_SUPPORT_MATRIX.md) | Which models can and cannot be analysed |
| [LARGE_MODELS.md](LARGE_MODELS.md) | Running on larger models (above ~13B: experimental) |
| [LARGE_MODEL_VERIFICATION.md](LARGE_MODEL_VERIFICATION.md) | Runbook for verifying a large model live |
| [../ENTERPRISE.md](../ENTERPRISE.md) | Self-hosted deployment, CI, containers and licensing |
| [v7/TRACE_SPEC.md](v7/TRACE_SPEC.md) | Trace format of the investigation layer (research stage) |
| [v7/SECURITY_PRIVACY.md](v7/SECURITY_PRIVACY.md) | Security and privacy controls (and limitations) for traces, each pinned by a test |
| [v7/REPRODUCING.md](v7/REPRODUCING.md) | How someone other than the author can reproduce a V7 result and send a report |

## Evidence and methodology

| Document | What it covers |
|---|---|
| [../ASSURANCE.md](../ASSURANCE.md) | Every claim with its evidence and status |
| [../BENCHMARKS.md](../BENCHMARKS.md) | Canonical benchmark numbers, hardware and method |
| [VALIDATION_LOG.md](VALIDATION_LOG.md) | Dated validation runs (models, tests, coverage) |
| [METHODOLOGY_AND_ASSURANCE.md](METHODOLOGY_AND_ASSURANCE.md) | What is proven, measured or hypothesised |
| [METHODOLOGY.md](METHODOLOGY.md) | Design of the compliance pipeline |
| [../MATH_FOUNDATIONS.md](../MATH_FOUNDATIONS.md) | Mathematical definitions |
| [CITATION-LEDGER.md](CITATION-LEDGER.md) | Verified references (append-only) |
| `../experiments/v7/*/` | Pre-registered protocols, code, results and write-ups |
| `../evidence/` | Verifiable evidence packages for completed experiments |

## Research plans (public, for provenance)

| Document | What it covers |
|---|---|
| [PLAN_V7_FINAL_UNIFIED.md](PLAN_V7_FINAL_UNIFIED.md) | V7 research plan and hard gates |
| [PLAN_DECISION_AUDIT.md](PLAN_DECISION_AUDIT.md) | Decision-audit study plan and data provenance |
| [../ROADMAP_V5_FOUNDATIONS.md](../ROADMAP_V5_FOUNDATIONS.md) | V5 technical design (cited by code comments) |
| [archive/ROADMAP_V4.md](archive/ROADMAP_V4.md) | Superseded roadmap, kept for provenance |
| [v7/user_research/](v7/user_research/) | Interview script and public issue scan. `simulations/` holds **simulated** interviews, which are not evidence. |

## Project policies

| Document | What it covers |
|---|---|
| [../LICENSE](../LICENSE), [../LICENSE-COMMERCIAL](../LICENSE-COMMERCIAL), [../NOTICE](../NOTICE) | Licenses; NOTICE is the file-by-file map |
| [../SECURITY.md](../SECURITY.md) | Supported versions, vulnerability reporting, threat model |
| [SUPPLY_CHAIN.md](SUPPLY_CHAIN.md) | Secret scan, dependency licence audit, trusted publishing |
| [../PATENTS.md](../PATENTS.md) | No patents filed; methods publicly disclosed as prior art |
| [../CONTRIBUTING.md](../CONTRIBUTING.md), [../CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | How to contribute |
| [../CHANGELOG.md](../CHANGELOG.md) | Release history |
| [../CITATION.cff](../CITATION.cff) | How to cite |
| [privacy.html](privacy.html) | Website privacy policy |

## Brand

[LOGO-GUIDELINES.md](LOGO-GUIDELINES.md) · [brand-philosophy.md](brand-philosophy.md)

## Rules for every document

- **Numbers come from `BENCHMARKS.md`.** When a number changes, update BENCHMARKS first,
  then README and the website.
- **Status words come from the ASSURANCE taxonomy.** Use `IMPLEMENTED`,
  `EXPERIMENTALLY_SUPPORTED`, `REFUTED` and so on. Never "proven" or "certified".
- **No legal conclusions.** Glassbox drafts documentation; it does not certify
  compliance or classify systems.
- **No promises.** No prices, dates, service levels or features that do not exist yet.
- **Private material stays out of the repository.** Strategy, competitor analysis,
  customer notes, interview notes and plans with business content live in `private/`,
  which is gitignored.
- **Cite only verified sources.** Every citation is recorded in `CITATION-LEDGER.md`.
