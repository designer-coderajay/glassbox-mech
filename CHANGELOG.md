# Changelog

All notable changes to Glassbox are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added — GPU option for V7 scorers (V8 phase B, 2026-10-30)

- **`glassbox.v7.compute`.** `ComputeConfig(device, dtype)` with `cpu` / `cuda` /
  `auto` and `float32` / `float16` / `bfloat16`.
  - The default stays CPU float32, so published experiments reproduce unchanged.
  - Requesting CUDA without a GPU is a clear error, and half precision on CPU is
    rejected.
  - `environment_record()` records device, dtype, GPU name, CUDA, torch and Python
    versions.
- **`GPT2Scorer(compute=...)` and `make_scorer(kind, compute)`.**
- **`python -m glassbox.v7.investigate demo --device ... --dtype ...`.** The
  environment record is written into the evidence package provenance.
- **GPU sanity check runner** (`experiments/v7/gpu_check/run.py`, pre-registered in
  its `PROTOCOL.md`) and the Colab runbook `docs/v7/COLAB_GPU.md`, which uses an
  isolated Python 3.11 venv.
  - Result: **PASS**. On a Colab T4 in float32 the investigation reached the same
    decisions as on CPU (`experiments/v7/gpu_check/RESULTS.md`, ASSURANCE V3.9).

### Added — agent investigation demo (2026-10-14)

- **New Space tab "Agent trajectory (simulated)"** (`dashboard/v7_agent_demo.py`, BSL 1.1):
  a simulated tool-using booking agent with four possible changes. It runs the same
  one-change-at-a-time investigation as on a real agent, shows each change's measured
  effect next to its exact true effect (known because the agent is simulated), and
  shows one before/after trajectory with its first divergent step.

### Changed — public claims audit (2026-10-14)

- **Removed every legal date, countdown and penalty figure** from the README, docs,
  website, generated reports, MCP server and API description. Readers are pointed to
  Regulation (EU) 2024/1689 and Regulation (EU) 2026/1744 (Digital Omnibus on AI)
  instead.
- **Annex IV wording corrected.**
  - "8 of 9 sections auto-filled" and "content-hashed" are replaced. The draft lays out
    all 9 sections, fills the measured fields and marks provider-only fields
    `[PROVIDER TO COMPLETE]`.
  - A real 4.6.0 run left provider fields in 7 of the 9 sections.
  - The Annex IV draft is not content-hashed.
- **Removed unverifiable or comparative claims**, including:
  - "21 frameworks";
  - "strategic monopoly release" and "first open-source tool";
  - the competitor ✅/❌ table;
  - "no other open-source tool";
  - the faithfulness threshold "derived from Article 15(1)";
  - untested model lists and "any OpenAI-compatible API including Claude/Gemini";
  - "Python 3.8+" (the badge now says 3.11, which CI tests).
- **Fixed the API metadata licence**, which said Apache 2.0 and now reads MIT core
  plus BSL 1.1 components.
- **The PyPI description no longer claims "21/21 frameworks".** This takes effect with
  the next release.
- **Website:** the timeline banner and countdown are removed. "37×" is shown as the
  canonical "15–37×", and the Annex IV example card reflects a real IOI run.
  `docs/index.redesign.html` (an old draft) is removed from the repository.

### Changed — HuggingFace Space redesign (2026-10-13)

- **New look and structure.** The theme follows the visitor's light or dark setting.
  There are six focused tabs: Explain a prediction, Layer by layer, Attention,
  Investigate a change, Annex IV draft, and About & limits. Charts are native HTML, so
  they stay sharp in both themes, and examples are cached after first use.
- **Copy corrected to the canonical facts.** Removed: the stale v4.3.0 header,
  "regulator-ready", "9/9 Annex IV sections automated", "Compliant" status labels and
  the pricing link. The Annex IV tab now shows, section by section, which fields the
  analysis filled and which the provider must complete.
- **Reliability.**
  - Every Space dependency is pinned to the CI stack, with Gradio 5.50.0 and CPU torch.
  - GPT-2 is preloaded at build time.
  - Model access is serialised (TransformerLens hooks are not thread-safe).
  - Demo work folders are cleaned up automatically.
- **Privacy.** The Space no longer writes visitors' prompts to an audit log on disk.
- **Removed** the custom `/analyze` and `/health` REST routes. Use the Gradio API
  endpoints (`/explain`, `/logit_lens`, `/attention`, `/annex_iv_draft`,
  `/investigate`) instead.

## [4.6.0] — 2026-10-13

### Fixed — packaging

- **The wheel now installs only the `glassbox` package.** Up to 4.5.1 it also
  installed top-level `tests`, `scripts`, `benchmarks`, `examples`, `api`,
  `dashboard`, `experiments` and `mcp` packages into site-packages. The `mcp` one
  shadows the official `mcp` SDK. If you installed 4.5.1, upgrading removes them.
  `tests/test_packaging.py` pins this.
- **First PyPI release containing the AI-system investigation layer** (`glassbox.v7`,
  BSL 1.1; see NOTICE). Every entry below in this 4.6.0 section ships in it.

### Changed — licensing (effective 2026-10-11)

- **The AI-system investigation layer is now BSL 1.1:** `glassbox/v7/` and
  `dashboard/v7_investigation.py`. `glassbox/cross_model.py`, which already carried a
  BSL 1.1 header, is now also listed in `LICENSE-COMMERCIAL`. Copies of these files
  obtained before 2026-10-11 remain under the terms they were published with. Of
  these files, only `cross_model.py` was in a PyPI release up to 4.5.1.
- **The BSL Additional Use Grant now also covers** offering a commercial AI-system
  investigation, debugging, tracing or observability service or product to third
  parties. Internal production use stays free.
- **The licence's conversion date is stated accurately.** Each version converts to
  Apache-2.0 four years after its first publication (BSL 1.1 paragraph 2). This
  replaces the misleading "ten years" wording.
- **NOTICE is now the authoritative file map.** Everything not listed under BSL is MIT,
  replacing the incomplete MIT file list. SPDX headers were added to the BSL files, and
  `pyproject.toml` gains the `License :: Other/Proprietary License` classifier.
  CONTRIBUTING.md explains how contributions to BSL files are licensed.

### Changed — documentation audit (2026-10-11)

- **Removed public commercial promises that no longer held:**
  - prices, "launch before August 2026", the SLA and the legal-indemnification offer
    (website, `ENTERPRISE.md`, USER_GUIDE);
  - "signed" vaults: hashing is not a signature;
  - offers to classify systems under Annex III.

  The website's pricing section is replaced by a licensing summary, and the waitlist
  becomes release updates.
- **Toned down overclaims:**
  - "regulator-ready", "works on any LLM", "the only/first tool", "Production-ready";
  - grade labels that implied legal compliance;
  - "peer-reviewed" for an arXiv preprint.

  The PyPI classifier changes from Production/Stable to Beta.
- **`ENTERPRISE.md` rewritten** as a deployment and licensing guide, with:
  - correct container flags;
  - an honest statement of the offline set-up;
  - a data warning: the GitHub Action defaults to the public demo Space (also noted in
    `action.yml`).
- **Paper reference corrected in USER_GUIDE:** title, description and preprint status
  now match arXiv:2603.09988. The unsupported "Submitted to ICML 2026" line is removed.
- **Strategy documents moved out of the public repository** into a gitignored
  `private/` folder: competitor analysis, the V8 business plan and the internal tooling
  map. `.dockerignore` excludes private material, and `.vercelignore` stops superseded
  page drafts from being served.
- **New `docs/README.md`:** a documentation index with writing rules.
- **Smaller fixes:**
  - `SECURITY.md` supported versions are now 4.5.x;
  - `METHODOLOGY.md` no longer calls itself "proprietary, all rights reserved"
    (contradicting its MIT license);
  - `LARGE_MODELS.md` states validation status;
  - the privacy policy purpose now matches the form;
  - the Impressum template is updated.

### Results — V7 Experiment 9b, decision-level reproduction (2026-10-13)

- **PASS: 8/8 packages `DECISIONS_REPRODUCED`, 8/8 verified, 7/8 exact.** The run used
  a previously unused GitHub Actions ubuntu-24.04 runner (AMD EPYC 7763), and
  Experiment 6 differs only in floats (max 0.0040).
- **Hard Gate 5 met at the decision level, for a same-author reproduction only.**
  ASSURANCE V9.2 is `EXPERIMENTALLY_SUPPORTED`; V9.1 (Experiment 9) stays `REFUTED`.
- **Attempt 1 and its re-run stopped after 7 packages** because of a workflow defect:
  GitHub's `bash -e` plus `compare` exiting 1 on `DECISIONS_ONLY`. They are recorded
  with Addenda 1–2, and their decisions match attempt 2.

### Results — V7 Experiment 10, investigation benchmark (2026-10-12)

- **Hard Gate 11 met; V10.1 PASS (K1–K5).** 32 incidents generated from the sealed
  seed (hash matches Addendum 1), 192 units, 28 positive. Glassbox exact diagnosis
  21/28 (0.75) against 0.50 for first-divergence blame and 0.00 for revert-all;
  zero-effect causes blamed 1/299; false alarms 1/48; implied fix verified 20/24.
- **Reported shortfalls:** 4 positive units where co-changes cancelled at the outcome
  were not detected; 3 over-blamed a small real cause; small effects were diagnosed
  exactly in 10/39 units. 26 of the 28 positive units come from the GPT-2 RAG family.
- `results/results.json` committed unchanged; write-up in
  `experiments/v7/benchmark/RESULTS.md`. ASSURANCE V10 and V10.1 are
  `EXPERIMENTALLY_SUPPORTED`, V6 is `IMPLEMENTED`.
- **Evidence package** `evidence/v7-exp10-benchmark` (`experiments/v7/benchmark/package.py`)
  and a reproduction recipe; listed in `docs/v7/REPRODUCING.md`.

### Results — V7 Experiment 9, reproduction in an independent environment (2026-10-12)

- **FAIL as pre-registered (7/8).** Re-run from their evidence packages on Linux x86-64
  (Kaggle CPU) instead of the original macOS arm64:
  - Experiments 1, 2, 3 (including its FAIL), 3b, 4, 5 and 8 reproduced with zero
    mismatches;
  - Experiment 6 did not. Every decision was identical, but one seeded Pythia draw
    flipped after cross-architecture float differences.
- **Hard Gate 5 is not met by this experiment.** ASSURANCE V9.1 is `REFUTED`; V3.6 is
  reconsidered openly and kept; seven rows are annotated as same-author reproduced.

### Added — reproduce from an evidence package (2026-10-12)

- **`glassbox/v7/reproduce.py`** has two commands:
  - `plan` prints the verified steps to re-run a packaged experiment from its recipe;
  - `compare` checks fresh results against the package and writes a reproduction report.

  The recipes for all nine packages are in `experiments/v7/reproduce_recipes.json`.
- **Experiment 9 (Hard Gate 5) pre-registered:** a same-author reproduction of
  Experiments 1, 2, 3, 3b, 4, 5, 6 and 8 on Linux x86-64. The runbook is in
  `experiments/v7/reproduction/KAGGLE_RUN.md`.

### Added — trace security and privacy document (Hard Gate 10, 2026-10-12)

- **`docs/v7/SECURITY_PRIVACY.md`:** the controls (C1–C8) and limitations (L1–L8) for
  trace handling. `tests/test_v7_security_controls.py` pins each control, and also the
  limitations that can be tested: unsalted hashes are pseudonymisation, and redaction is
  best-effort. ASSURANCE V8 is `IMPLEMENTED`.
- **Evidence package for Experiment 8:** `evidence/v7-exp8-cross-stack`.
- **Corrections:** TRACE_SPEC's stale "no code implements it" header is fixed, and
  SECURITY.md and the docs index link the new document.

### Added — OTLP import/export and V7 Experiment 8 (2026-10-12)

- **`glassbox/v7/otlp.py`:** imports OTLP JSON from any OpenTelemetry instrumentation
  into the Glassbox trace format and exports Glassbox traces to OTLP JSON. It uses only
  the standard library and is BSL 1.1.
- **Experiment 8 PASS** (Hard Gate 2). It used a LlamaIndex pipeline traced by
  OpenInference on the OpenTelemetry SDK. The import was lossless (32/32), and the
  unchanged diff engine reproduced Experiment 1's conclusions on the same fault. Glassbox
  traces also pass the official OTLP protobuf schema (32/32).
- **ASSURANCE:** V2.10 and V1.2 are `EXPERIMENTALLY_SUPPORTED`. V1.1 stays a hypothesis
  for live-collector export.

### Added — V7 Experiment 7b and validation run 15 (2026-10-12)

- **Experiment 7b PASS** in two Kaggle CPU runs, and Hard Gate 8 is met. ASSURANCE
  V3.8 is `EXPERIMENTALLY_SUPPORTED` after the pre-registered numeric-explanation check
  (Addendum 3) traced the 6th-decimal differences between runs to floating-point
  summation order. RESULTS.md records the caveats. The evidence package is
  `evidence/v7-exp7b-agent`.
- **Validation Run 15:** 1,288 passed, 12 skipped, 75.83% coverage. It supersedes 1,241 /
  75.80%. README, website, ASSURANCE L11, METHODOLOGY_AND_ASSURANCE and the MCP servers
  are updated.

### Fixed — citations and README

- The citation in NOTICE, README and CITATION.cff now matches the arXiv record
  (arXiv:2603.09988, *Causally Grounded Mechanistic Interpretability for LLMs with
  Faithful Natural-Language Explanations*). The unsupported "ICML 2026" venue was
  removed. CITATION.cff's description of r = 0.009 now matches the paper
  (confidence–faithfulness correlation).
- The README was rewritten: it adds the investigation layer, the status of every V7
  experiment (including the FAIL and INCONCLUSIVE results) and a licence table.

## [4.5.1] — 2026-06-27

### Fixed — benchmark-number reconciliation (single canonical set, measured on current build)

- **IOI faithfulness corrected to the current build's measured result.** Default
  circuit on GPT-2 Small now records suff 1.00 / comp 0.543 / **F1 0.704 / Grade B**
  (1-head circuit; reproduced by `analyze()` and `validate_models_matrix.py`),
  superseding the earlier F1 0.49. The **full 26-head Wang et al. circuit** is
  stated separately as suff 1.00 / comp 0.47 / F1 0.64; the previously-circulated
  "suff 1.00 / comp 0.22 / F1 0.64" triple was arithmetically impossible
  (1.00 & 0.22 → 0.36) and is fixed across README, BENCHMARKS, CITATION.cff,
  PAPER_OUTLINE, MATH_FOUNDATIONS, and the compliance/model-support docs.
- **Credit/decision claims reframed around honest failure.** The previously-cited
  "credit suff 0.73 / F1 0.61 / Grade B (14 heads)" does **not reproduce** on the
  current exact-measurement build: raw GPT-2 has no faithful credit circuit, so
  the decision-functional harness records F1 0.00–0.08, Grade C, NON-COMPLIANT
  (`reports/credit_current.json`). All surfaces (README, BENCHMARKS §4, the
  website hero, one-pager) now state this measured behavior — the tool correctly
  refuses to certify a model that cannot faithfully explain its decision.
- **Grade-threshold docs aligned to the code** (`compliance.py:_compute_grade`):
  A = suff ≥ 0.80 ∧ comp ≥ 0.60 ∧ F1 ≥ 0.80; B = ≥ 0.65/0.40/0.65; C = F1 ≥ 0.50.
- Code-review fixes to `core.py` (cap-exit logging, MFC docstring) and
  `validate_models_matrix.py` (apples-to-apples random-comp baseline); CI
  coverage note refreshed to 932 tests / 71%.

### Added / Fixed — engine

- **`analyze(exact_circuit=True, max_circuit_heads=N)` now published** — scale-aware
  circuit selection that grows the circuit by *measured* exact sufficiency, fixing
  the Taylor-path under-sizing that collapses faithfulness on larger / distributed
  circuits (see `docs/VALIDATION_LOG.md`).
- **Fixed the large-model parameter estimate** used for the VRAM warning. Was
  `n_layers · n_heads · d² · 4` (over-counted by ~n_heads/3× — e.g. it reported
  pythia-1.4b as 6.4B); now the standard `12 · n_layers · d²` (Kaplan et al. 2020),
  accurate to within ~15% across the GPT-2 / Pythia family.

> Note: the 4.5.0 entry below stated the (then-believed) numbers as canonical;
> this 4.5.1 section is the correction of record, now released.

## [4.5.0] — 2026-06-13

### Added — V5 Phase B/C foundations (pure, tested cores)

- **`glassbox/auditable.py`** — Auditable Interface: a 5-capability `AuditableModel` protocol (forward / units / read / patch / optional contributions) plus the architecture-agnostic conformance suite `run_conformance` (determinism, patch-identity, reconstruction). This is the gate every backend must pass — "the conformance suite, not trust, is the gatekeeper." Validated against a real GPT-2 Small via `scripts/validate_auditable_gpt2.py` (determinism + patch-identity, 2/2). (7 tests)
- **`glassbox/scaling.py`** — hierarchical head screening + length-aware batch planning, with an explicit false-negative-risk flag. (9 tests)
- **`glassbox/monitoring.py`** — post-market monitoring (Art. 72): two-sided CUSUM drift detector, Johnson–Lindenstrauss fingerprint projector (distance-preserving sketch), and a task-family circuit cache validated by fingerprint match. (12 tests)
- **`glassbox/moe.py`** — Mixture-of-Experts partition (`moe_units`, incl. router) + expert attribution (weight × contribution). (6 tests)
- **`glassbox/distributional.py`** — distributional faithfulness via percentile bootstrap confidence intervals + stratified mean. (9 tests)
- **`glassbox/causal_abstraction.py`** — causal-abstraction certificate computed from interchange-intervention accuracy (IIA); gates tier-A (causal-certified) eligibility. (4 tests)
- **`glassbox/features.py`** — feature-level (SAE) units + sparsity-aware attribution (active features only). (5 tests)
- **`glassbox/frameworks.py`** — multi-framework report packs: cross-walk Glassbox evidence to NIST AI RMF (Govern/Map/Measure/Manage) and ISO/IEC 42001 (9 Annex A objectives), with an explicit "not certification or legal advice" disclaimer. (5 tests)
- All eight modules are re-exported from the top-level `glassbox` package API.
- **`scripts/validate_auditable_gpt2.py`** — real-model validation harness (run locally; exercises the conformance gate + monitoring/framework cores on GPT-2 Small).

### Notes

- **+57 unit tests** this release across the eight new pure cores (verified). Torch-dependent production backends (native-HF, GPU / KV-sharing, real MoE/SAE hooks, SSM/Mamba adapter) ship as protocol-conforming interfaces and are **flagged in their module docstrings as not yet validated at frontier scale** — that work is design-partner-gated (Roadmap 9.4).
- Canonical benchmark numbers are unchanged from 4.4.0 (r=0.009; credit suff 0.73 / F1 0.61, Grade B; IOI F1 0.49, Grade C; 15–37× vs ACDC; 1.8s M1 Pro / 4.2s 8-core CPU).

---

## [4.4.0] — 2026-06-12

### Regulatory accuracy

- **Patent claims retracted** — no patent application was ever filed; "Patents Pending" badge removed from README, PATENTS.md rewritten as a public-disclosure / defensive-publication notice, source headers and NOTICE corrected. Removes false-marking exposure (35 U.S.C. §292).

- **All public surfaces** (site, README, compliance guide, CLAUDE.md): enforcement-date claims updated to dual framing after verifying the **Digital Omnibus provisional agreement of 7 May 2026** — Annex III high-risk obligations expected to defer from 2 Aug 2026 to **2 Dec 2027** (pending formal adoption; 2 Aug 2026 remains the date under current law). Site countdown now targets the expected date with the caveat shown.
- **`docs/EU_AI_ACT_COMPLIANCE_GUIDE.md`** — corrected penalty citation: documentation failures fall under Art. 99(4) (€15M / 3%), not the Art. 99(3) prohibited-practices ceiling (€35M / 7%).
- **`docs/index.html`** — added GDPR Art. 13 privacy notice for the waitlist form (controller, purpose, processors, rights) + footer link; no tracking cookies.

### Fixed

- **`glassbox/cross_model.py`** — **CrossModelComparison's lightweight path silently returned empty attributions.** The attention-difference heuristic cached `blocks.{l}.attn.hook_attn_weights`, which is not a TransformerLens hook name (the post-softmax pattern hook is `hook_pattern`). Every cache lookup failed, the guard skipped every head, and a bare `except Exception: pass` swallowed the evidence — the comparison "succeeded" with an empty circuit. Renamed to `hook_pattern`, hoisted caching out of the per-head loop (one `run_with_cache` per prompt instead of one per head — 144× fewer forward passes on GPT-2 Small), narrowed the exception handling, and added a warning when attribution comes back empty.
- **`glassbox/polysemanticity.py`** — **Same-layer circuit heads were silently dropped from scoring.** Activation collection keyed a dict by `blocks.{l}.attn.hook_z`, so two heads in the same layer (e.g. IOI's L9H6 and L9H9) collided and only the last survived. Now groups heads per layer and extracts each head from the shared layer cache.
- **`scripts/sync_versions.py`** — **mcp/pyproject.toml never synced.** The `^version` pattern was compiled without `re.MULTILINE`, so it could only match at byte 0 and the MCP package version drifted (4.2.6 vs core 4.3.1). Added the flag, synced the version, and added patterns for the site's version badges and JSON-LD `softwareVersion`.
- **`glassbox/hf_integration.py`, `glassbox/mlflow_integration.py`** — optional-dependency `ImportError` re-raises now use `raise … from err` (B904) so the original import failure isn't masked.
- Removed unused imports/variables and normalised import ordering across the package (`ruff --fix`, rules E/F/I).

### Security

- **`api/main.py`** — CORS allowlist contained a `"*"` entry alongside the explicit origins, which makes the whole list behave as a wildcard. Removed; extra origins now come from `GLASSBOX_CORS_ORIGINS`. Methods/headers narrowed to what the API actually uses.
- **`api/main.py`** — white-box endpoints (`/v1/audit/analyze`, `/v1/attention-patterns`) now validate the requested model against an allowlist (README's tested-models table; extendable via `GLASSBOX_MODEL_ALLOWLIST`) instead of passing arbitrary HuggingFace ids to `from_pretrained` — closes a disk/memory-exhaustion vector on hosted deployments.

### Added

- **V5 Phase A — decision functional, CF gate, evidence tiers (ROADMAP_V5 Parts 2.1, 3.3, 6).** Three new dependency-free modules, built test-first (47 new unit tests, pure logic, no torch required):
  - `glassbox/decision.py` — `DecisionFunctional` / `VerbalizerSet`: generalizes the two-token logit diff to disjoint verbalizer *sets* via `D = logsumexp(logits_A) − logsumexp(logits_B)`; for singleton sets this reduces exactly to the legacy logit diff (softmax normalizer cancels), so backward compatibility is mathematical, not approximate. Multi-token variants supported through injected sequence scores; tokenizer overlap between sets is rejected as ill-defined.
  - `glassbox/cf_gate.py` — `CounterfactualGate`: admits a counterfactual into attribution only if it preserves task shape, aligns with the clean prompt, and moves the decision value above a noise floor; everything rejected is **discarded and reported** (counts by reason go to the technical file). Measurement failures are recorded, never raised.
  - `glassbox/evidence_tier.py` — `TierEngine`: the A/B/C/D degradation ladder with downgrade-with-stated-reason semantics and report disclosure text; physically contradictory capability claims (e.g. exact patching without weights) are rejected outright.
- **V5 Sprint 2 — evidence tiers wired into Annex IV reports.** `AnnexIVReport.add_analysis()` accepts an optional `evidence_tier` (TierAssessment or dict, or embedded in the result); across multiple analyses the **weakest tier governs** (most conservative). The governing tier + disclosure text land in the report JSON (top-level `evidence_tier`) and in §3's `evidence_tier_disclosure`. Reports without tiers are byte-identical to before (8 new integration tests; full legacy compliance suite green).
- **`api/main.py`** — thread-safe LRU model cache (`GLASSBOX_MODEL_CACHE_SIZE`, default 2) for `/v1/audit/analyze`; eliminates the per-request model reload that dominated API latency. The attention-patterns endpoint is deliberately uncached (different preprocessing flags would poison a name-keyed cache; documented inline).

- **`api/waitlist.js`** — waitlist signups now deliver a real email notification via Resend (`RESEND_API_KEY` env var; recipient defaults to the founder, override with `NOTIFY_EMAIL`) with `reply_to` set to the registrant for one-click replies, and persist to Vercel KV automatically when a store is attached (`KV_REST_API_URL`/`KV_REST_API_TOKEN`). Both paths are best-effort: failure never blocks the signup, everything still logs under `[waitlist]`. New optional `name`/`company`/`message` fields accepted and length-capped; HTML-escaped in the notification.

### Changed

- **`glassbox/core.py`** — V5 Sprint 4: **every `analyze()` run now verifies its own counterfactual and self-labels its evidence tier.** The corrupted decision value is captured from attribution's existing Pass 2 (its logits were previously discarded — verification costs **zero extra forward passes**; the 3-pass headline stands). The CF gate checks task shape + effect-above-noise (relative floor: 1% of |clean_ld|, helper `_cf_noise_floor`); failures are logged loudly and reported under `result["counterfactual_validation"]`, never absorbed. Each result also carries `result["evidence_tier"]` — honestly tier C for a bare run (no Hessian certificate, no exact-patching verification; B/A require running those), which `AnnexIVReport` picks up automatically. This closes the V5 Phase A loop: verbalizer sets in → verified counterfactual gating attribution → tier-labeled Annex IV out.
- **`glassbox/core.py`** — V5 Step 2: `analyze()` now accepts **verbalizer sets** — `correct=[" approved", " approve", " yes"]` — pooling evidence via `logsumexp(A) − logsumexp(B)` through the entire pipeline (attribution gradients included), via the set-aware `_decision_value()` internals. Singleton sets reproduce legacy results exactly (logsumexp identity); legacy str/str calls are untouched and produce no new keys. Set runs document themselves under a `"decision"` key for the evidence vault. Input resolution lives in module-level `_resolve_decision_tokens()` (8 offline tests); 6 new real-model tests in test_engine.py assert legacy equivalence and end-to-end set runs.
- **`glassbox/core.py`** — V5 Step 1 of the decision-functional migration: all 8 last-position logit-diff computations centralized into one `_decision_value()` helper with byte-identical semantics (fp32 cast-each-then-subtract preserved on gradient paths; native-dtype subtract preserved on exact-patching paths). No behavior change — this isolates the single point where Step 2 swaps in the set-aware `logsumexp(A) − logsumexp(B)` objective from `glassbox.decision`.

- **`docs/index.html`** — landing page redesign: pricing section (Community / Pro waitlist / Enterprise), working email capture wired to `/api/waitlist`, live countdown to the 2 Aug 2026 enforcement date, engineer/compliance audience fork, JSON-LD structured data. All marketing numbers reconciled to BENCHMARKS.md + CI: 710 tests, 15–37× vs ACDC, credit-task suff 0.73 / F1 0.61 / Grade B, Art. 99(4) €15M / 3% penalty (previously cited the Art. 99(3) prohibited-practices cap). Previous design preserved at `docs/_design-snapshot/index.html`.
- **`ENTERPRISE.md`** — billion-parameter claim now states implementation status honestly (verification scheduled) instead of implying verified Llama-3-70B benchmarks.
- **`.claude/CLAUDE.md`** — canonical metrics list reconciled to the same single source of truth.

---

## [4.3.1] — 2026-06-08

### Fixed

- **`pyproject.toml`** — **Broken on fresh install.** The core dependencies had no upper bounds (`torch>=2.0`, `transformer_lens>=1.0`, bare `numpy`), so a clean `pip install glassbox-mech-interp` pulled NumPy 2.x and a torch nightly. NumPy 2.0 changed `np.bool` semantics, breaking the attribution-stability path with `TypeError: 'numpy.bool_' object does not support item assignment`. Pinned the verified-compatible stack: `numpy>=1.24,<2.0`, `torch>=2.0,<2.11`, `transformer_lens>=1.0,<3.0`.
- **`glassbox/core.py`, `glassbox/steering.py`** — **`attention_patterns()` crashed on Apple Silicon / CUDA.** `core.py:attention_patterns()` (auto-head-select path) and `steering.py:save()` called `.numpy()` directly on tensors that live on the MPS/GPU device, raising `TypeError: can't convert mps:0 device type tensor to numpy`. Added `.cpu()` before `.numpy()` in both places. Worked on plain CPU only before this fix.
- **`tests/test_multi_arch.py`, `tests/test_engine.py`** — **CI was silently testing a mock model.** `test_multi_arch.py`'s module-level `_inject_stubs()` decided whether to stub torch/transformer_lens using `sys.modules` membership ("imported yet?") instead of `importlib.util.find_spec` ("installed?"). At collection time the installed `transformer_lens` often isn't imported yet, so it injected a `MagicMock` over the real package, poisoning `sys.modules` for the whole session — every engine test then ran against a fake `HookedTransformer` (58 errors / 30 failures in CI). Fixed `_inject_stubs()` to gate on `find_spec` (matching `conftest.py`), and made the `test_engine.py` `engine` fixture skip when `transformer_lens` is a stub rather than run against it.
- **`glassbox/compliance.py`** — **`NameError` in Annex IV report generation.** The report template called a bare `_get_version()` that only existed as a method elsewhere. Defined a module-level `_get_version()` helper that returns the installed `glassbox.__version__`.
- **`glassbox/prompt_corruption.py`** — **Antonym table not symmetric.** Three entries (`promoted`, `liable`, `graduating`) mapped to values that were not themselves keys, breaking `test_antonym_table_is_symmetric`. Remapped to `demoted`, `exempt`, and `failed` respectively.
- **`scripts/sync_versions.py`** — **`glassbox/__init__.py.__version__` not synced.** The package's own `__version__` had regressed out of the sync list (last fixed in 4.2.6), so it could drift from the canonical `pyproject.toml` version. Re-added it to the patch list.

### Added

- **`.github/workflows/ci.yml`** — Tiered CI pipeline: a fast `ruff` lint job on every push, and a full `pytest` + coverage job (real torch + transformer_lens stack) on pushes/PRs to `main` and a weekly schedule, with a diagnostic step that verifies the model stack imports. Coverage is enforced at a measured floor of **45%** (real coverage with the engine exercised is **48%**; the previously-claimed 80% was never actually met — target remains 80% as module coverage improves).
- **`scripts/check_engine.sh`** — One-command local reproducer of the CI engine-test environment: builds a throwaway venv with the exact pinned stack, asserts a *real* (non-mock) GPT-2 loads, and runs the full suite with coverage.

---

## [4.2.6] — 2026-04-17

### Fixed

- **`glassbox/steering.py`** — **Steering suppression always 0%** — `test_suppression()` monkey-patched `model.run_with_cache` to inject the steering hook, but `GlassboxV2.analyze()` internally calls `model.run_with_hooks` for its gradient/attribution pass. The patched method was never invoked, so every steered result was identical to baseline (zero suppression ratio). Fixed by switching to `model.add_hook(hook_name, fn, is_permanent=True)` which intercepts both call paths, with `model.reset_hooks(including_permanent=True)` in a `finally` block.
- **`glassbox/sae_attribution.py`** — **`_CustomSAE` W_dec shape mismatch** — `_CustomSAE` stores `W_dec` as `(d_model, n_features)`, while sae-lens hub SAEs store it as `(n_features, d_model)`. The attribution function used `sae.W_dec` directly for both, causing the matmul `(d_model, n_features) @ unembed_dir` to fail with a shape error. Fixed with an `isinstance(_CustomSAE)` check: custom SAEs now transpose `W_dec` before the matmul.
- **`glassbox/sae_attribution.py`** — **`TypeError` when `sae_release=None`** — The Neuronpedia URL builder did `"gpt2-small" in self.sae_release` even when `sae_release` is `None` (custom-SAE mode), raising `TypeError: argument of type 'NoneType' is not iterable`. Added a `None` guard: URL is only constructed when `self.sae_release is not None`.
- **`glassbox/__init__.py`** — **`__version__` frozen at "4.2.4"** — The top-level `__version__` string was hardcoded and never updated by `sync_versions.py`. All downstream tooling that imports `glassbox.__version__` (CLI `--version`, MCP server info, HF Space footer) was reporting the wrong version. Fixed to `"4.2.6"` and added to the sync-versions patch list.
- **`tests/conftest.py`** — **No-model tests collected torch at import time** — Compliance, audit-log, risk-register, and widget tests import from `glassbox`, which transitively imports `torch` at module level. Without stubs, pytest collection failed with `ModuleNotFoundError: No module named 'torch'` in environments without GPU/ML deps. Added module-level `sys.modules` stubs for `torch`, `transformer_lens`, `einops`, `scipy`, and `sae_lens` in `conftest.py`, enabling all 155 offline tests to run without torch installed.
- **`tests/test_compliance.py`** — **`test_grade_a` fixture below Grade A threshold** — The `good_result` fixture used `comp=0.68`, giving `F1 = 2*0.92*0.68/(0.92+0.68) = 0.782`, which falls below the Grade A F1 threshold of 0.80 and produces Grade B. The test asserted `startswith("A")` and always failed. Fixed fixture to `comp=0.72` → `F1 = 0.826 ≥ 0.80` → Grade A.
- **Version sync** — Version string unified at `4.2.6` across `pyproject.toml`, `mcp/pyproject.toml`, `glassbox/__init__.py`, `mcp/glassbox_mcp/__init__.py`, `mcp/glassbox_mcp/server.py`, `dashboard/app.py`, `README.md`, `mcp/requirements.txt`, and `.github/workflows/deploy_hf.yml` via `scripts/sync_versions.py --apply`.

---

## [4.2.4] — 2026-04-04

### Fixed (GPU + Large-Model Correctness)

- **`glassbox/acdc.py`** — **OOM for large models** — `run_with_cache()` had no `names_filter`, caching ALL TransformerLens hook outputs (>50 named hooks per layer). For Llama-3-70B this fills several GB of VRAM with unused activations. Added `names_filter=lambda name: "hook_z" in name or "hook_resid_pre" in name` to both `discover()` cache passes, reducing memory to only what ACDC actually needs.
- **`glassbox/acdc.py`** — **KL divergence float32 precision** — `np.sum(np.exp(clean_logprobs) * ...)` was computed in float32. For models with 128k-token vocabularies (Llama-3), accumulating 128k float32 terms loses ~3 significant digits. Both `_test_edge_kl` and `_circuit_kl` now cast to float64 before summation.
- **`glassbox/acdc.py`** — **Edge-count explosion warning** — For Llama-3-7B, ACDC builds ~540k candidate edges; for Llama-3-70B, ~13.4M. Full ACDC on a 70B model would take hundreds of GPU-days. Added an upfront `logger.warning` when estimated edge count exceeds 100k, directing users to the MFC algorithm instead.
- **`glassbox/core.py`** — **Silent zero gradients in Integrated Gradients** — `analyze(method="integrated_gradients")` could produce all-zero attributions when called inside a `torch.no_grad()` context (common in eval scripts). Added `with torch.enable_grad():` around each IG step so gradient tracking is forced regardless of the outer context.
- **`glassbox/core.py`** — **`GlassboxV2` now accepts `dtype` parameter** — Large models (7B+) require `torch.bfloat16` to fit in VRAM. Added `dtype` kwarg to `__init__`: `GlassboxV2("meta-llama/Llama-3-8B", device="cuda", dtype=torch.bfloat16)`. Passed through to `HookedTransformer.from_pretrained()`.

---

## [4.2.3] — 2026-04-04

### Fixed

- **`glassbox/acdc.py`** — **Critical algorithmic bug in ACDC edge pruning.** `pruned_so_far=all_edges[:i]` was passing ALL previously tested edges (both retained and pruned) to `_test_edge_kl`. The correct ACDC algorithm must patch only previously *pruned* edges plus the current candidate, not retained ones. Retained edges must remain active so the marginal importance of each edge is evaluated in context. Fixed to `pruned_so_far=[e for e in all_edges[:i] if e not in circuit_edges]`. Circuits discovered before this fix were incorrect.
- **`glassbox/multi_arch.py`** — **Crash in `adjust_attributions_for_gqa()`** — `head_scores` is `Dict[int, float]` but the inner loop tried to unpack each key as `(layer_key, head_key)`, which raises `ValueError: not enough values to unpack`. Fixed to `for head_key, score in head_scores.items()` with correct `adjusted[(layer, head_key)] = score` composite key.
- **`glassbox/cross_model.py`** — **Out-of-bounds bin index** — `int(norm / 0.1)` returns 10 when `norm == 1.0` (last layer / last head), which is out-of-range for a 10×10 grid (valid indices 0–9). Fixed by clamping to `min(int(norm / bin_size), max_bin)` in all three binning sites: `_jaccard`, `_attribution_pearsonr`, and `_find_consensus_heads`.
- **`glassbox/cross_model.py`** — **Jaccard empty-union returns 1.0** — When both model circuits are empty, `union == 0` triggered `return 1.0` (identical). An empty circuit means no overlap evidence; the correct default is `0.0`.
- **`glassbox/cross_model.py`** — **`unique_to_a` / `unique_to_b` granularity mismatch** — `unique_a = len(normalised_circuit()) - len(shared)` mixed per-position counts with per-bin shared counts. If two positions in A map to the same bin, they counted as 2 in `len(circuit)` but 1 shared bin. Fixed to compare bin-sets directly: `unique_a = len(bins_a - bins_b)`, `unique_b = len(bins_b - bins_a)`.
- **`glassbox/steering.py`** — **Hardcoded `_VERSION = "3.6.0"`** — All steering vector HTML reports were stamped with the wrong version. Applied the same `importlib.metadata.version("glassbox-mech-interp")` dynamic lookup used in `evidence_vault.py` since v4.2.1.
- **`README.md`** — Version badge and PyPI install example updated from `4.2.0` → `4.2.3`.

---

## [4.2.2] — 2026-04-04

### Fixed

- **`glassbox/multi_arch.py`** — `RMSNormFolding.fold()` dimension mismatch. TransformerLens stores `W_Q` as `(n_heads, d_model, d_head)`, not `(n_heads, d_head, d_model)` as the code assumed. `gamma.unsqueeze(0)` produced shape `(1, d_model)` which failed to broadcast against `(d_model, d_head)`. Fixed to `gamma.unsqueeze(1)` → `(d_model, 1)`. Also fixed identity fallback shape and docstring.
- **`glassbox/core.py`** — `comprehensiveness=0` for all non-IOI prompts. When `_name_swap` couldn't find the target/distractor in the prompt (e.g. "The capital of France is" + correct=" Paris"), it appended the distractor as a fallback token. The corrupted prompt prefix was identical to the clean prompt, so corrupt-patching was a no-op. Added degenerate-corruption detection and a `_comp_zero_ablation()` fallback: when clean and corrupted token prefixes are identical, zero-ablation is used instead (sets circuit head z-values to 0). Gives valid comprehensiveness for factual recall, sentiment, arithmetic.
- **`glassbox/core.py`** — `GlassboxV2` now accepts a model name string (`GlassboxV2('gpt2')`) in addition to a pre-loaded `HookedTransformer`. Automatically calls `HookedTransformer.from_pretrained(name, device=device)`.
- **`glassbox/core.py`** — Added `logger.warning` when `clean_ld ≤ 0`: the model prefers the distractor over the target token, meaning faithfulness metrics are unreliable for that prompt.
- **`glassbox/cross_model.py`** — `_analyse_with_glassbox` was only storing circuit-head attributions in `SingleModelResult.attributions` (typically 1–10 heads), making Pearson attribution correlation always return 0 (fewer than 3 overlapping bins). Now parses and stores the full attribution dict (all `n_layers × n_heads` heads) so cross-model Pearson r is computed over the complete attribution vector.

---

## [4.2.1] — 2026-04-04

### Fixed

- **`glassbox/acdc.py`** — Critical crash: `hook(resid_pre, hook_ctx)` renamed to `hook(resid_pre, hook=None)`. TransformerLens passes the hook point as a keyword argument `hook=`; the positional parameter name `hook_ctx` caused `TypeError: got an unexpected keyword argument 'hook'` on every `AutomatedCircuitDiscovery.discover()` call. ACDC was completely non-functional in v4.2.0.
- **`glassbox/cross_model.py`** — `CrossModelReport.summary` and `CrossModelReport.attribution_table` promoted to `@property`. Previously they were plain methods, so `report.summary` returned a bound-method object instead of the string. All docstring examples updated accordingly.
- **`glassbox/evidence_vault.py`** — Hardcoded `_VERSION = "3.6.0"` replaced with dynamic lookup via `importlib.metadata.version("glassbox-mech-interp")`. The AnnexIV vault was stamping every report with the wrong version (3.6.0) regardless of the installed package version.

---

## [4.2.0] — 2026-04-04

### Added

- **`glassbox/acdc.py`** — `AutomatedCircuitDiscovery`: full ACDC algorithm (Conmy et al., NeurIPS 2023, arXiv:2304.14997). Edge-level circuit pruning via exact KL divergence — for each directed edge (sender u → receiver v), patches u's residual-stream contribution with the corrupted activation and measures KL(p_patched ‖ p_clean). Edges with KL < τ are pruned; retained edges form the minimal faithful circuit. `ACDCEdge`, `ACDCCircuit`, `ACDCResult` (with `faithfulness_grade`: STRONG/PARTIAL/WEAK). Uses TransformerLens `hook_result` for per-head residual-stream contributions.
- **`glassbox/multi_arch.py`** — `MultiArchAdapter`: architecture-aware adapter enabling all Glassbox frameworks on non-GPT-2 models. Handles Grouped Query Attention (GQA — Llama-3 8B/70B, Mistral 7B, Phi-3) and RMSNorm (all Llama, Mistral, Phi-3, Gemma). `ArchitectureConfig.from_transformer_lens()` auto-detects from model config. `GQAAttentionMapper` redistributes KV head attribution scores across sharing query heads (equal split, 1/G). `RMSNormFolding` folds γ scale into W_Q/K/V (no bias term, unlike LayerNorm). Architecture registry: 11 families.
- **`glassbox/cross_model.py`** — `CrossModelComparison`: runs mechanistic interpretability on multiple model families sequentially, then computes pairwise circuit similarity. Jaccard similarity on normalised (layer/n_layers, head/n_heads) circuit positions with 10×10 grid binning. Pearson r on normalised attribution vectors. Consensus head detection (heads in ≥50% of models). `CrossModelReport.attribution_table()` for markdown comparison tables. `compare_models()` one-shot wrapper. Memory-safe: explicit `del model` + `gc.collect()` between model loads.

### Mathematical Foundation
- ACDC edge KL: KL(p ‖ q) = Σ p(x)·(log p(x) − log q(x)); τ = 0.10 (Conmy et al.); faithful if KL(circuit) < 0.80
- GQA redistribution: score[q] += kv_score[kv_head_for(q)] / heads_per_kv_group
- RMSNorm folding: W_Q^folded = diag(γ) · W_Q; bias_ratio = 0.0 (no additive β in RMSNorm)
- Cross-model Jaccard: sim(C1, C2) = |bin(C1) ∩ bin(C2)| / |bin(C1) ∪ bin(C2)|; bin_size = 0.1
- Attribution correlation: Pearson r on normalised attribution vectors aligned by normalised position bins

### EU AI Act Mapping
- Art. 13(1) Transparency: ACDC provides exact causal edge evidence (not approximate node-level Taylor)
- Art. 15(1) Robustness: Cross-model comparison validates circuit stability across model families
- Art. 10 Data Governance: MultiArchAdapter ensures consistent analysis across architecture families

### Framework Count
After v4.2.0: **21 mathematical frameworks implemented**
- 18 from v4.1.0 baseline (attribution patching through DAS)
- v4.2.0 adds: ACDC, GQA multi-arch, cross-model comparison

### Constants
- `ACDC_KL_THRESHOLD = 0.10` (Conmy et al. IOI default)
- `ACDC_FAITHFULNESS_THRESHOLD = 0.80`
- `SUPPORTED_ARCHITECTURES`: 11 families (gpt2, llama-2, llama-3, llama-3-70b, mistral, phi-2, phi-3, gemma, pythia, gpt-j, qwen2)

---

## [4.1.1] — 2026-04-04

### Fixed
- **Missing `scipy` core dependency** — `glassbox/fdr.py` and `glassbox/validation.py` import `scipy.stats` but `scipy>=1.9` was not listed in `pyproject.toml` `dependencies`. HuggingFace Space was crashing at startup with `ModuleNotFoundError: No module named 'scipy'`.
- **PyPI project description stale** — `pyproject.toml` description and README were updated after the v4.1.0 tag was pushed, so PyPI showed the old description. This patch tag bakes the correct description into the published package.
- **HuggingFace Space requirements** — `scipy>=1.9.0` added to Space `requirements.txt` in `deploy_hf.yml`. Pin updated to `glassbox-mech-interp>=4.1.1`.

### No API or behaviour changes
This is a pure packaging fix. All v4.1.0 functionality is unchanged.

---

## [4.1.0] — 2026-04-03

### Added
- **`glassbox/hessian.py`** — `HessianErrorBounds`: second-order Taylor error bounds via Pearlmutter (1994) HVP. For each head h: ε(h) = ½·δzᵀ·H·δz computed via `torch.autograd.grad` double-backprop. Flags `hessian_dominated` when |ε(h)|/|α(h)| > 0.20. Spectral-norm fallback when autograd fails. `HessianBoundsReport` + `HeadHessianBound` dataclasses.
- **`glassbox/causal_scrubbing.py`** — `CausalScrubbing`: Anthropic causal scrubbing (Chan et al. 2022). `CircuitHypothesis` dataclass with `from_wang2022_ioi()` preset. CS(H) = E[LD_scrubbed]/LD_clean; strong≥0.80, partial≥0.50. Monte Carlo sampling over corrupted activations. `CausalScrubbingResult` with interpretation and fraction_explained.
- **`glassbox/das.py`** — `DistributedAlignmentSearch`: finds linear subspace R encoding a concept via PCA on activation difference vectors Δz (Geiger et al. 2023). Interchange interventions to compute DAS score. `search_all_layers()` for layer sweep. `DASResult` with rotation_matrix, concept_dims, explained_variance.
- **`CircuitHypothesis.from_wang2022_ioi()`**: pre-built Wang et al. 2022 IOI circuit with 13 heads and role labels.

### Changed
- Version `4.0.0` → `4.1.0`

### Mathematical Foundation
- Hessian bound: ε(h) = ½·δz_hᵀ·H_h·δz_h via Pearlmutter HVP; error_ratio = |ε|/|α|; threshold 0.20
- Causal scrubbing: CS(H) = E_{x}[LD(x; do(acts~P_H))] / LD_clean ∈ [0, 1+]; H₀ strong iff CS≥0.80
- DAS: R = top-k PCA of {Δz = z_clean − z_CF}; DAS score = 1 − |mean(LD_intervened)/mean(LD_clean)|

### EU AI Act Mapping
- Art. 13(1) Transparency: Hessian bounds certify attribution reliability (first-order not dominated by second-order)
- Art. 9(1) Risk Management: Causal scrubbing provides formal hypothesis testing — not just correlation, but causal account
- Art. 15(1) Robustness: DAS identifies whether model representations are distributed or localised

### Mathematical Completeness Scorecard
After v4.1.0: **18/18 mathematical frameworks implemented** (100%)
- v3.6.0 baseline: 7/18
- v3.7.0 added: multi-corruption, held-out validation, SampleSizeGate → 10/18
- v4.0.0 added: FoldedLayerNorm, BH FDR, polysemanticity → 13/18
- v4.1.0 added: Hessian bounds, causal scrubbing, DAS → 18/18 ✓

---

## [4.0.0] — 2026-04-03

### Added
- **`glassbox/layernorm_correction.py`** — `FoldedLayerNorm`: absorbs LayerNorm scale γ into Q/K/V weight matrices; computes per-head bias Δα(h) = α_folded(h) − α_raw(h); flags heads with |bias_ratio| > 0.15 as `layernorm_biased`. `LayerNormBiasReport` with full per-head bias analysis.
- **`glassbox/fdr.py`** — `BenjaminiHochberg`: Benjamini-Hochberg FDR control (E[FDR]≤α) for head-level attribution significance. Supports standard z-test, bootstrap SE, and permutation-based p-values. Reports BH and Bonferroni side-by-side. `FDRReport` + `HeadSignificance` dataclasses.
- **`glassbox/polysemanticity.py`** — `PolysemanticityScorerSAE`: Shannon entropy H(p(feature|head_h)) via SAE feature activations; PCA participation ratio fallback when sae-lens unavailable. `PolysemanticitySummary` with monosemantic_fraction and per-head scores.

### Changed
- Version `3.7.0` → `4.0.0`

### Mathematical Foundation
- Folded LN: W_Q^folded = diag(γ)·W_Q; bias_ratio = Δα(h)/|α_raw(h)|; threshold 0.15
- BH FDR: t_i = (i/K)·α; reject all H₀_(j) for j ≤ i*; E[FDR] = (m₀/K)·α ≤ α
- Polysemanticity: P(h) = H(p(feature|head)) = -Σ p_f log₂(p_f); P_norm ∈ [0,1]

### EU AI Act Mapping
- Art. 13(1) Transparency: LayerNorm bias correction enables attribution scores not artificially inflated by scale parameters
- Art. 9(1) Risk Management: BH FDR prevents false-positive head identifications in compliance reports
- Art. 15(1) Robustness: Polysemanticity score quantifies interpretability quality of identified circuit heads

---

## [3.7.0] — 2026-04-03

### Added
- **`glassbox/corruption.py`** — `MultiCorruptionPipeline` with 4 corruption strategies (`CorruptionStrategy` enum):
  - `NAME_SWAP`: Bidirectional IO⇔S name swap (Wang et al. 2022 standard)
  - `RANDOM_TOKEN`: Replace IO/S tokens with `Uniform(V)` random vocabulary token
  - `GAUSSIAN_NOISE`: Add `N(0, σ²·I)` noise to token embeddings (σ = std of clean embeddings)
  - `MEAN_ABLATION`: Replace last-position residual stream with dataset mean (zero ablation fallback)
  - `RobustnessReport`: aggregated across all corruptions; flags `perturbation_sensitive` when `max_k |S_k − S̄| ≥ 0.10`
  - `CorruptionResult` dataclass: per-strategy S/Comp/F1/LD metrics
- **`glassbox/validation.py`** — Statistical validation gates:
  - `SampleSizeGate`: raises `SampleSizeError` (n<20, hard block) or `SampleSizeWarning` (n<50, soft warn)
  - `HeldOutValidator`: 50/50 train/test split; flags `overfit` when `|F1_train − F1_test| ≥ 0.10`

### Changed
- Version `3.6.0` → `3.7.0`

### Mathematical Foundation
- Robustness criterion: ∀k : |S_k(C) − S̄| < δ = 0.10
- Power analysis: n_min = ((z_{α/2} + z_β) / atanh(ρ_min))² + 3; n≥50 → 80% power at |ρ|≥0.25
- Generalisation gap: gap = |F1_train − F1_test| < δ_gen = 0.10

---

## [3.6.0] — 2026-04-02

### Added
- **Full-stack interactive website**: Live circuit analyzer embedded directly in landing page — paste a prompt, pick a model, get real-time attribution heatmap + faithfulness metrics + compliance grade in-browser. Vanilla JS, no build step, graceful fallback to demo data when backend unavailable
- **WebSocket streaming** (`/ws/{job_id}`): Real-time analysis progress (stage indicators, percent bars, live messages) instead of blocking long-poll requests
- **CORS + rate limiting** in FastAPI: `CORSMiddleware` for Vercel/localhost; 20 req/min rate limiter middleware so the API is production-safe without nginx
- **Vercel API routing** (`api/index.py` + `vercel.json` rewrites): `/api/*` paths now proxy to the FastAPI serverless function — one domain, no separate backend needed for light loads

### Fixed
- **MCP server critical import** (`mcp/server.py`): `GlassboxAnalyzer` → `GlassboxV2` (was `ImportError` at runtime on every circuit discovery call)
- **MCP server `analyze()` signature** (`mcp/server.py`): `corrupted_prompt` → `correct`/`incorrect` (was `TypeError` on every invocation)
- **Async GPU blocking** (`mcp/server.py`): Wrapped all blocking TransformerLens calls in `asyncio.to_thread()` so MCP server event loop no longer stalls during model inference
- **`analyze()` input validation** (`glassbox/core.py`): Empty prompt/correct/incorrect now raises `ValueError` immediately instead of producing silent garbage output
- **Non-deterministic circuit sort** (`glassbox/core.py`): Secondary sort key `(layer, head)` added — compliance reports now produce identical circuit orderings across runs
- **`__version__` sync** (`glassbox/__init__.py`): Was `3.4.0`, now tracks `pyproject.toml` correctly

### Changed
- Version `3.5.0` → `3.6.0`
- README title and Docker image tags updated to `3.6.0`
- `.claude/CLAUDE.md`: HuggingFace Space URL corrected (`affaan/glassbox` → `designer-coderajay/...`), website URL corrected
- `pyproject.toml`: `notify = []` annotated — webhook-based, no pip deps needed

---

## [3.5.0] — 2026-04-01

### Added
- **Claude Code plugin** (`.claude/`): Full project brain, 6 specialized agents (interpretability-researcher on Opus, compliance-generator, python-reviewer, pytorch-build-resolver, code-reviewer, doc-updater), 6 skills (mechanistic-interpretability v2.0 with SAEs + steering vectors, eu-ai-act-compliance with GPAI Articles 51–55, python-testing with Hypothesis + syrupy, pytorch-transformerlens with Pythia multi-model, security-review with pip-audit, circuit-discovery), and 5 slash commands (`/circuit`, `/compliance`, `/review`, `/audit`, `/test`)
- **FastMCP server** (`mcp/`): Model Context Protocol server with 5 tools — `glassbox_circuit_discovery`, `glassbox_faithfulness_metrics`, `glassbox_compliance_report` (full 9-section Annex IV JSON), `glassbox_attention_patterns`, `glassbox_logit_lens`. Pydantic v2 input validation, model allowlist, graceful degradation when library not installed
- **Brand asset** (`assets/glassbox_brand.png`): 1400×800 circuit-trace design with attribution heatmap, L9H9 gold highlight (attribution=0.584), faithfulness bars and compliance card

### Fixed
- `glassbox/__init__.py` `__version__` corrected to `3.5.0`
- MCP server class reference corrected from `GlassboxAnalyzer` → `GlassboxV2`
- MCP server `analyze()` parameter names corrected (`correct`/`incorrect` not `corrupted_prompt`)
- Non-deterministic circuit sort — added secondary sort key `(layer, head)` for reproducible compliance reports
- Input validation added to `analyze()` — empty strings now raise `ValueError` immediately

### Changed
- `deploy_hf.yml` workflow: added `workflow_dispatch` trigger and `glassbox/**` path filter so library changes auto-sync to HuggingFace Space
- HuggingFace Space requirement bumped to `glassbox-mech-interp>=3.5.0`

---

## [Unreleased] — HuggingFace Space UI Fixes — 2026-03-22

### Fixed
- **HF Space About tab blank** — root cause was a CSS selector (`.etali4b10, .svelte-po8fcl { display:none !important }`) that matched the Gradio 4.43.0 Markdown component wrapper, silently hiding every `gr.Markdown()` and `gr.HTML()` block in the app. Full GB_CSS rewrite replacing 430 lines with 270 lines of clean, version-stable CSS using semantic selectors and `data-testid` attributes only.
- **Compliance Report "Error generating compliance report: D"** — `AnnexIVReport` library class raises an internal exception with message `"D"` on HF Space (version mismatch). Restructured `run_compliance_report` to derive grade A/B/C/D directly from the raw `gb.analyze()` result without depending on `AnnexIVReport`. `AnnexIVReport` is now tried non-blocking for the model card only; a fallback model card is generated if it throws.
- **Compliance Report output invisible** — `cr_report` component was `gr.HTML()` but the function returns mixed Markdown+HTML. Changed to `gr.Markdown(sanitize_html=False)` which renders both Markdown tables/headings and inline HTML div blocks.
- **About tab blank (previous attempts)** — replaced `gr.HTML(ABOUT_HTML)` (unreliable in HF iframe) with `gr.Markdown(ABOUT_MD)` using standard Markdown content. Gradio's Markdown component is unconditionally reliable.
- **HF Space URLs in README** — all 7 occurrences of the wrong Space ID (`Glassbox-ai`) corrected to `Glassbox-AI-2.0-Mechanistic-Interpretability-tool`.
- **pyproject.toml project URLs** — `Homepage`, `Dashboard`, `Documentation` updated from stale Render.com URL to active Vercel site, HF Space, and GitHub README.
- **GitHub Actions workflow** — fixed deploy target from wrong Space name to `Glassbox-AI-2.0-Mechanistic-Interpretability-tool`; added force-push and dynamic commit messages.

---

## [3.4.0] — 2026-03-21

### Added
- **MultiAgentAudit** (`glassbox/multi_agent_audit.py`): First open-source tool to trace bias contamination and semantic drift across multi-agent chains. `MultiAgentAudit().audit_chain([AgentCall(...)])` returns a `ChainAuditReport` with per-agent liability scoring, most-liable-agent identification, and Annex IV Article 9 narrative. Scores bias across 8 EU AI Act Article 10(5) protected categories. `to_html()` generates a self-contained liability dashboard. Maps to Article 9, Article 10(2)(f), Article 10(5), Article 13(1). Exported from top-level: `from glassbox import MultiAgentAudit, AgentCall`.
- **SteeringVectorExporter** (`glassbox/steering.py`): Extract and export steering vectors from the residual stream using Representation Engineering (Zou et al. 2023). `extract_mean_diff()`, `extract_pca()`, and `extract_bias_suite()` methods. `apply()` applies a vector as a runtime hook. `test_suppression()` computes before/after faithfulness comparison and suppression ratio. `export_pt()` / `export_numpy()` for regulatory submission artefacts. Maps to Article 9(2)(b), Article 9(5), Article 15(1). Exported: `SteeringVectorExporter, SteeringVector`.
- **AnnexIVEvidenceVault** (`glassbox/annex_iv_vault.py`): Assembles all interpretability findings (circuit analysis, bias tests, steering vectors, multi-agent audits, SAE features, stability scores) into a single machine-readable, regulation-mapped Annex IV evidence vault. `build_annex_iv_vault()` top-level function. Outputs JSON and self-contained HTML suitable for regulatory submission. Covers Annex IV §1–§7, maps to Articles 9, 10, 11, 13, 15, 72.
- **HuggingFace Space v3.4** interactive dashboard deployed at `designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool` with five tabs: Circuit Analysis, Logit Lens, Attention Patterns, Compliance Report, About.

### Changed
- Version bumped to 3.4.0 in `pyproject.toml`.
- README: Added v3.4.0 "What's New" section with full code examples for all three new modules. Updated live services table. Live demo link updated to correct HF Space.

---

## [3.3.0] — 2026-03-20

### Added
- **NaturalLanguageExplainer** (`glassbox/explain.py`): Rule-based, deterministic converter from raw circuit analysis results to structured plain-English compliance summaries. Zero LLM dependency. `verbosity` levels: `"brief"`, `"standard"`, `"detailed"`. `include_article_refs=True` adds EU AI Act article citations to every sentence. Methods: `explain()`, `explain_sections()`, `to_html()`. Exported: `NaturalLanguageExplainer`.
- **HuggingFace Hub integration** (`glassbox/hf_integration.py`): `load_from_hub()` loads any HookedTransformer-compatible model from HF Hub (29 architecture aliases). `HuggingFaceModelCard` pushes/reads compliance metadata sections to/from model card README.md. `push_compliance_section()` adds grade, F1, circuit summary, and article mapping. Exported: `load_from_hub, HuggingFaceModelCard`. Install: `pip install 'glassbox-mech-interp[hf]'`.
- **MLflow integration** (`glassbox/mlflow_integration.py`): `log_glassbox_run()` logs a full circuit analysis result as an MLflow run — grade, F1, sufficiency, comprehensiveness, circuit heads, and prompt metadata. `GlassboxMLflowCallback` for automatic logging during batch analysis. Install: `pip install 'glassbox-mech-interp[mlflow]'`.
- **Slack/Teams alerting** (`glassbox/notify.py`): `SlackAlerter` and `TeamsAlerter` send formatted alerts when CircuitDiff detects drift (`drift_threshold`) or compliance grade drops (`grade_threshold`). Webhook-based, no SDK required. `AlertConfig` for threshold management.
- **GitHub Action CI hook** (`glassbox/ci_hook.py`): `check_compliance_gate()` function suitable for use in GitHub Actions; exits with code 1 if compliance grade drops below configured threshold. Sample workflow in README.

### Changed
- `GlassboxV2.analyze()` now integrates `NaturalLanguageExplainer` — result dict includes `"explanation"` key with plain-English summary when `explain=True` (default False to preserve backward compatibility).
- README: Added v3.3.0 "What's New" section. Updated live services table.

---

## [3.2.0] — 2026-03-20

### Added
- **Black-box audit mode**: `BlackBoxAuditor` class in `glassbox/black_box.py`. Runs behavioural proxy metrics (token probability, output consistency, bias probes) on any model accessible via a callable — GPT-4, Claude, Gemini, or any proprietary API. No model weights required. `audit_api_model(model_fn, prompt, correct, incorrect)` returns faithfulness proxies and Annex IV draft. Exported: `BlackBoxAuditor`.
- **Stability suite** (`glassbox/stability.py`): `stability_suite(gb, prompt, correct, incorrect, n_bootstrap=100)` runs bootstrap F1 estimation, prompt perturbation robustness, and token-swap sensitivity. Returns `StabilityResult` with confidence intervals and `summary_stats()`. Exported: `stability_suite, StabilityResult`.
- **AnnexIVReport** (`glassbox/compliance.py`): `AnnexIVReport` class generates the full 9-section EU AI Act Annex IV technical documentation package. `to_json()`, `to_markdown()`, `to_model_card()` export formats. `add_analysis()` attaches circuit results. Exported: `AnnexIVReport`.
- **REST API** (`glassbox/api.py`): FastAPI-based REST endpoint. `/analyze`, `/compliance-report`, `/audit-log`, `/health` routes. `pip install 'glassbox-mech-interp[api]'`.
- **DeploymentContext enum** (`glassbox/compliance.py`): `FINANCIAL_SERVICES`, `HEALTHCARE`, `HR_EMPLOYMENT`, `EDUCATION`, `LEGAL`, `OTHER_HIGH_RISK` for context-aware risk classification and Annex IV narrative. Exported: `DeploymentContext`.

### Changed
- `GlassboxV2` now accepts `model_name` kwarg for Annex IV auto-population.
- README: Added black-box audit section. Updated live services table to v3.2.0.

---

## [3.1.0] — 2026-03-20

### Added
- **CircuitDiff** (`glassbox/circuit_diff.py`): Mechanistic diff between two model versions
  or checkpoints. `CircuitDiff(gb_a, gb_b).diff(prompt, correct, incorrect)` returns a
  `CircuitDiffResult` with added/removed/shared heads, Jaccard stability score, per-head
  attribution delta, and F1 delta. `batch_diff()` for multi-prompt stability analysis.
  `summary_stats()` aggregates stability mean/std and most-commonly-added/removed heads.
  `to_markdown()` generates PR-ready audit report. Maps to EU AI Act Article 72 (post-market
  monitoring) and Annex IV Section 6 (lifecycle change documentation).
  Exported from top-level: `from glassbox import CircuitDiff, CircuitDiffResult`.

- **Exact sufficiency in `bootstrap_metrics()`** (`glassbox/core.py`): New `_suff_exact()`
  method computes exact causal sufficiency via positive ablation — keeps only circuit heads
  active, corrupts all other heads, measures preserved logit difference. `bootstrap_metrics()`
  now takes `exact_suff=True` (default) to use this method. Return dict includes
  `meta.exact_suff` and `meta.suff_is_approx` fields. Reproducibility note documented in
  docstring: seed=42, GPT-2 small, Apple M1 Pro, PyTorch 2.2.0, TransformerLens 1.19.0.
  This resolves the discrepancy between Taylor approx (~80%) and exact (~100%) sufficiency.

- **Custom SAE upload** (`glassbox/sae_attribution.py`): `SAEFeatureAttributor` now accepts
  `sae_path` parameter. Pass a single `.pt` file path (applied to all layers) or a dict
  `{layer: path}` for per-layer checkpoints. Expected checkpoint keys: `encoder_weight`,
  `encoder_bias`, `decoder_weight`, `decoder_bias`. New `_CustomSAE` internal class mirrors
  the sae-lens encode/decode interface. sae-lens not required when using custom checkpoints.
  Enables SAE attribution for fine-tuned, custom, or non-public models.

- **OpenTelemetry tracing** (`glassbox/telemetry.py`): `setup_telemetry(service_name, endpoint)`
  initialises OTLP trace export. `instrument_glassbox(gb)` monkey-patches `analyze()` to emit
  a span per call with attributes: `glassbox.model`, `glassbox.grade`, `glassbox.f1`,
  `glassbox.circuit_heads`, `glassbox.duration_ms`. `trace_span()` context manager / decorator
  for custom instrumentation. Supports Jaeger, Honeycomb, Datadog OTLP, Grafana Tempo.
  Falls back to no-op silently if opentelemetry-sdk is not installed.
  Install: `pip install 'glassbox-mech-interp[telemetry]'`.
  Exported: `setup_telemetry`, `teardown_telemetry`, `trace_span`, `instrument_glassbox`,
  `is_telemetry_enabled`, `TelemetryConfig`.

### Changed
- `bootstrap_metrics()` default behaviour changed: sufficiency is now exact (`exact_suff=True`)
  rather than Taylor approximation. Pass `exact_suff=False` to restore prior behaviour.
- README: Added v3.1.0 "What's New" section. Updated live services table to v3.1.0.
  Added grade scale research-defined caveat. Added black-box behavioural proxy caveat.
  Added hosted API / Render free-tier disclaimer. Added scaling roadmap in roadmap section.

### Documentation
- Comprehensive legal hardening: Legal Notices & Regulatory Disclaimer (9 subsections),
  GDPR Project & Privacy Notice with Impressum (§5 TMG), trademark disclaimer.
  Legal NOTICE blocks in all compliance-facing modules (compliance.py, risk_register.py,
  bias.py, audit_log.py). CONTRIBUTING.md legal contribution guidelines.

---

## [3.0.0] — 2026-03-20

### Added
- **Bias Analysis Module** (`glassbox/bias.py`): `BiasAnalyzer` class with three EU AI Act
  Article 10(2)(f)-compliant tests. `counterfactual_fairness_test()` swaps demographic
  attributes in prompt templates to measure probability shift (parity gap). `demographic_parity_test()`
  computes positive outcome rates across groups and flags disparity above threshold.
  `token_bias_probe()` detects stereotypical associations between demographic and role tokens.
  All methods work offline (pre-computed logprobs dicts) or online (live `model_fn`).
  `BiasReport` aggregates results into an Annex IV Section 5 markdown report.
  Exported from top-level: `from glassbox import BiasAnalyzer, BiasReport`.
- **Webhooks** (`api/main.py`): Full webhook registration system. `POST /v1/webhooks`
  registers a callback URL with event filters (`job.completed`, `job.failed`) and optional
  HMAC-SHA256 signing secret. `GET /v1/webhooks` lists registered webhooks. `DELETE /v1/webhooks/{id}`
  and `PATCH /v1/webhooks/{id}` manage them. Payloads include `X-Glassbox-Event` and
  `X-Glassbox-Signature` headers. Delivery tracked per webhook (`delivery_count`,
  `last_delivery_status`).
- **Circuit SVG Export** (`dashboard/compliance_dashboard.html`): "Download SVG" button
  in the D3 circuit graph panel. Exports `glassbox-circuit.svg` with inlined styles
  and dark background — ready for paper figures.
- **Multi-Audit History Panel** (`dashboard/compliance_dashboard.html`): Toggleable
  "Audit History" panel with F1-over-time Chart.js line chart (grade C threshold line),
  grade distribution bar chart, and audit table. "Load from API" button fetches
  `GET /v1/audit/reports`. Demo data shows D→C→C→B→B grade trajectory.
- **Risk Register** (`glassbox/risk_register.py`): `RiskRegister` class persists compliance
  risks across audit sessions to a JSON file. `ingest_annex_report()` auto-extracts risks from
  any `AnnexIVReport`. Deduplication by description+model, occurrence counting, severity
  ordering, status tracking (`open | mitigated | accepted | escalated`), `trend_summary()`
  for dashboards, `to_markdown()` for PR comments and report embedding. Maps to EU AI Act
  Article 9 (risk management system) and Annex IV Section 5.
  Exported from top-level: `from glassbox import RiskRegister, RiskEntry`.
- **Test suites** (`tests/test_audit_log.py`, `tests/test_widget.py`): 76 passing tests.
  Full offline coverage of AuditLog hash chain, CircuitWidget/HeatmapWidget HTML rendering.

### Changed
- `glassbox/__init__.py`: Version 3.0.0. `BiasAnalyzer`, `BiasReport`, result dataclasses
  added to public API and `__all__`.
- `pyproject.toml`: Version 3.0.0. Description updated.
- `api/main.py`: `_WEBHOOK_STORE` added. `_fire_webhooks()` wired into async job completion.

---

## [2.9.0] — 2026-03-20

### Added
- **Tamper-evident Audit Log** (`glassbox/audit_log.py`): `AuditLog` class persists every
  compliance audit to an append-only JSONL file. Each record carries a SHA-256 hash of the
  previous entry, forming a hash chain that `verify_chain()` can validate. Supports
  `export_csv()` and `export_json()` for regulator hand-off. `summary()` returns grade
  distribution, compliance rate, and average F1 over all stored audits.
  Now exported from `glassbox` top-level: `from glassbox import AuditLog`.
- **GitHub Actions Composite Action** (`action.yml`): `glassbox-audit@v1` drops EU AI Act
  compliance gates into any CI/CD pipeline. Inputs: `model_name`, `prompt`, `correct_token`,
  `incorrect_token`, `fail_below_grade` (default: C), `deployment_context`, `method`.
  Outputs: `grade`, `f1_score`, `sufficiency`, `comprehensiveness`, `compliance_status`,
  `report_id`. Exits 1 and emits `::error::` annotation when grade falls below threshold.
  Uses composite run steps — no Docker image needed.
- **TypeScript SDK** (`sdk/glassbox.ts`): Zero-dependency fetch-based client for the
  Glassbox REST API. Works in Node.js ≥18, Deno, Bun, and browsers. Typed request/response
  interfaces (`WhiteBoxRequest`, `BlackBoxRequest`, `AuditReport`, `AsyncJobResponse`,
  `AttentionPatternsResponse`). `GlassboxClient` class with `auditWhiteBox()`,
  `auditBlackBox()`, `startBlackBoxJob()`, `waitForJob()` (polling helper), and
  `attentionPatterns()`. `GlassboxError` carries `statusCode` + `detail`. Default export +
  named `createClient()` factory for CJS/ESM compatibility.
- **Jupyter Notebook Widgets** (`glassbox/widget.py`): `CircuitWidget` wraps `GlassboxV2`
  for one-line notebook usage. `CircuitWidget.from_prompt(gb, prompt, correct, incorrect)`
  runs the analysis and renders an inline attribution heatmap via `_repr_html_()`.
  `HeatmapWidget` accepts any pre-computed result dict (from the Python SDK or REST API).
  Both classes gracefully degrade when ipywidgets is absent. Install with:
  `pip install 'glassbox-mech-interp[jupyter]'`.
  Now exported from `glassbox` top-level: `from glassbox import CircuitWidget, HeatmapWidget`.
- **Attention Patterns API endpoint** (`api/main.py`): `POST /v1/attention-patterns` accepts
  `model_name`, `prompt`, and an optional `heads` list (e.g. `["L9H9", "L9H6"]`). Returns
  raw attention matrices, per-head entropy, last-token attention vector, and head-type
  classifications. Returns HTTP 503 on free-tier RAM exhaustion with self-hosting instructions.

### Changed
- `glassbox/__init__.py`: `AuditLog`, `AuditRecord`, `CircuitWidget`, `HeatmapWidget` added
  to public API and `__all__`. Version bumped to 2.9.0.
- `pyproject.toml`: Version bumped to 2.9.0. Description updated to reflect new features.
- Dashboard (`compliance_dashboard.html`): Full Linear/Vercel/Stripe-grade UI redesign.
  Dark-first design system with CSS custom properties, Inter + JetBrains Mono fonts,
  backdrop-blur nav, noise-texture overlay, 4-tier colour scale. All JS wiring preserved.

---

## [2.8.0] — 2026-03-17

### Added
- **Model card generator** (`glassbox/compliance.py`): `AnnexIVReport.to_model_card()` generates
  a HuggingFace-compatible `MODEL_CARD.md` with YAML frontmatter (tags: eu-ai-act, annex-iv,
  compliance, mechanistic-interpretability), compliance status table, faithfulness metrics,
  risk flags, EU AI Act article references, and citation block.
  `save_model_card(path)` convenience method writes it to disk.
- **D3 circuit graph** (`dashboard/compliance_dashboard.html`): Interactive force-directed
  graph visualising the minimum faithful circuit. Nodes = attention heads, size proportional
  to attribution score, colour mapped by layer, gold border on high-attribution heads.
  Draggable, hover tooltips, layer-adjacency edges. Uses D3.js v7.
- **Attribution heatmap** (`dashboard/compliance_dashboard.html`): 12×12 grid of attention
  head attribution scores. Colour intensity maps to score magnitude; circuit members highlighted
  with a gold border. Demo data uses real IOI/GPT-2 results (L9H9, L9H6, L10H0, L3H0…).
- **Async job endpoint** (`api/main.py`): `POST /v1/audit/black-box/async` returns immediately
  with a `job_id`. Audit runs as a FastAPI `BackgroundTask`. Poll status via
  `GET /v1/jobs/{job_id}` (states: queued → running → completed/failed).
  `GET /v1/jobs` lists all session jobs. Accepts same `X-Provider-Api-Key` header.

### Changed
- Dashboard default API URL updated to the live Render endpoint.
- API privacy notice added to dashboard: key sent as header only, never logged, never stored.

## [2.7.0] — 2026-03-17

### Added
- **EU AI Act Compliance module** (`glassbox/compliance.py`): `AnnexIVReport` class generates
  all 9 Annex IV sections as PDF + JSON. Maps faithfulness metrics to Article 13.
  Explainability grades A–D with exact thresholds. 26/26 tests passing.
- **Black-box auditor** (`glassbox/audit.py`): `BlackBoxAuditor` audits any model via API
  (OpenAI, Anthropic, Together, Groq, Azure, custom endpoint) — no model weights needed.
  Uses counterfactual probing, sensitivity sweeps, consistency testing. Zero extra dependencies.
- **REST API** (`api/main.py`): FastAPI app with `POST /v1/audit/analyze` (white-box),
  `POST /v1/audit/black-box`, `GET /v1/audit/report/{id}`, `GET /v1/audit/pdf/{id}`,
  `GET /dashboard` (serves compliance UI), `GET /docs` (Swagger UI).
- **Compliance dashboard** (`dashboard/compliance_dashboard.html`): Full web UI for compliance
  officers. Demo mode with real IOI/GPT-2 data — works with zero backend.
- **Dockerfile** + **render.yaml**: production-ready container, one-click Render deploy.
- **Live deployment**: API live at `https://glassbox-ai-2-0-mechanistic.onrender.com`.

### Security
- API key moved from request body to `X-Provider-Api-Key` header — never logged or stored.
- `_StripKeyFilter` log handler scrubs any accidental key-shaped strings from all log output.
- `SECURITY.md` added with full key handling documentation, GDPR note, self-hosting guide.

### Fixed
- `api/main.py`: version string was hardcoded `2.6.0`; now reads from `glassbox.__version__`.
- README restructured: TOC added, section order fixed, REST API section added, Dashboard
  section updated to reflect live URL, benchmark numbers marked as preliminary.
- All EU AI Act article references verified against final Regulation (EU) 2024/1689 text.
- Penalty claim updated to include Article 99(4) citation and EUR-Lex link.

---

## [2.6.0] — 2026-03-17

### Fixed

- **Version sync** — `glassbox/__init__.py` `__version__` was hardcoded as `"2.3.0"` while
  `pyproject.toml` was at `2.5.2`. Both are now `2.6.0` and will track together going forward.
- **`publish.yml` cleanup** — removed all diagnostic Macaroon-decoding code introduced while
  debugging PyPI OIDC 403 errors (root cause was a Pending Publisher UUID mismatch, now fixed
  at the PyPI project settings level). Workflow now uses the official `pypa/gh-action-pypi-publish@release/v1`
  action — the simplest and most reliable path.
- **`CITATION.cff`** — title and abstract still referenced `Glassbox 2.3`; updated to `2.6`.
  Both `version:` fields updated from `2.5.2` to `2.6.0`.
- **`cli.py`** — CLI banner and argparse description both hardcoded `2.3`; updated to `2.6`.
- **`requirements.txt`** — removed `scipy` (not imported anywhere; Kendall τ-b is implemented
  without it), `streamlit` and `plotly` (dashboard-only, not part of the core package),
  and `pytest` (dev-only). File now mirrors `pyproject.toml` core and dev deps.
- **`deploy_hf.yml`** — heredoc for HuggingFace Space `requirements.txt` was indented 10 spaces
  inside the YAML `run:` block; those spaces were being written verbatim into the file, making
  every package name invalid for `pip install`. Replaced with explicit `echo` statements.
  Updated `glassbox-mech-interp>=2.1.0` → `>=2.6.0`.
- **`README.md`** — feature table header updated from `v2.3.0` to `v2.6.0`.
- **`dist/` cleanup** — removed stale `v2.2.0` wheel and sdist that were committed to git
  despite `dist/` being listed in `.gitignore`.

---

## [2.3.0] — 2025-07-01

### Added

**SAE Feature Attribution** (`glassbox/sae_attribution.py`) — new module.
Bridges circuit-level (attribution patching, EAP) and feature-level
(SAEs, superposition) interpretability. Two methods:
- `SAEFeatureAttributor.attribute()` — decomposes residual stream at each
  layer into sparse feature activations and scores each feature by its
  logit-difference contribution. Links directly to Neuronpedia for each
  active feature.
- `SAEFeatureAttributor.attribute_circuit_heads()` — head-scoped SAE
  attribution: which sparse features are activated by each circuit head?
  (Linear approximation; see docstring.)
Requires: `pip install sae-lens` (optional dep). Supports GPT-2 small
via Joseph Bloom's pretrained residual-stream SAEs.
References: Bloom et al. (2024), Bricken et al. (2023), Cunningham et al. (2023).

**Head Composition Analysis** (`glassbox/composition.py`) — new module.
Computes Q/K/V composition scores between attention head pairs (Elhage et al. 2021, §3.2).
- `HeadCompositionAnalyzer.q_composition_score(sl, sh, rl, rh)` — Q-composition.
- `HeadCompositionAnalyzer.k_composition_score(...)` — K-composition.
- `HeadCompositionAnalyzer.v_composition_score(...)` — V-composition.
- `HeadCompositionAnalyzer.composition_matrix(senders, receivers, kind)` — full matrix.
- `HeadCompositionAnalyzer.full_circuit_composition(circuit, kind, min_score)` — all pairwise scores within a circuit.
- `HeadCompositionAnalyzer.all_composition_scores(circuit)` — Q+K+V in one call.
No extra dependencies. Always available.

**Token Attribution** (`GlassboxV2.token_attribution()`) — added to `core.py`.
Per-input-token attribution via gradient × embedding (Simonyan et al. 2014).
Scores each token by its signed contribution to logit(target) - logit(distractor).
Returns `token_ids`, `token_strs`, `attributions`, `abs_attributions`, `top_tokens`.
Cost: 1 forward + 1 backward pass.

**Attention Pattern Analysis** (`GlassboxV2.attention_patterns()`) — added to `core.py`.
Returns full attention matrices, per-head entropy, last-token attention row, and
heuristic head-type classification: `induction_candidate`, `previous_token`,
`focused`, `uniform`, `self_attn`, `mixed`.
Cost: 1 forward pass.

**Expanded test suite** — 6 new test classes in `tests/test_engine.py`:
- `TestLogitLens` (8 tests) — logit_lens() correctness and mathematical consistency.
- `TestEdgeAttributionPatching` (8 tests) — EAP structure, score finiteness, positivity.
- `TestAttributionStability` (6 tests) — stability scores bounds, Kendall τ-b range.
- `TestTokenAttribution` (7 tests) — token attribution structure, sorting, finiteness.
- `TestAttentionPatterns` (8 tests) — patterns shape, row sums, entropy, head types.
- `TestHeadCompositionAnalyzer` (11 tests) — score bounds, causal validity, matrix shape.

### Changed
- `glassbox/__init__.py` — exports `SAEFeatureAttributor` and `HeadCompositionAnalyzer`.
- `pyproject.toml` — version 2.3.0; added `sae` optional dep group; added full
  classifiers, `arXiv Paper` and `Changelog` URLs, `ruff` and `mypy` config sections.
- `README.md` — complete rewrite. Added feature comparison table vs. TransformerLens /
  Baukit / Pyvene, full API reference, SAE and composition code examples, updated
  benchmarks section, complete citation block.
- `core.py` module docstring — added Simonyan et al. 2014, Olsson et al. 2022,
  Bloom et al. 2024 references; updated complexity table with new methods.
- `core.py` `GlassboxV2` class docstring — added all new method signatures.

---

## [2.2.0] — 2025-05-15

### Added

**Logit Lens** (`GlassboxV2.logit_lens()`) — implements nostalgebraist (2020) extended
with per-head direct effects (Elhage et al. 2021, §2.3).
- Projects residual stream at each layer through ln_final + unembed to show how
  predictions crystallise layer by layer.
- Per-head direct effects via virtual weights: `direct(l,h) = (W_O[l,h] @ z[l,h,-1]) · unembed_dir`.
- Optional inclusion in `analyze()` via `include_logit_lens=True`.
- 1 forward pass.

**Edge Attribution Patching** (`GlassboxV2.edge_attribution_patching()`) — implements
Syed et al. (2024). Scores every directed edge (sender → receiver) in the computation
graph. Formula: `EAP(u→v) = (∂metric/∂resid_pre_v) · Δh_u`. O(3) cost.
- Strictly more informative than node-level AP: reveals which connections carry the signal.
- Gradient captured via `act.register_hook()` to avoid breaking the computation graph.

**Attribution Stability** (`GlassboxV2.attribution_stability()`) — novel metric.
- Runs attribution over K random corruptions (25% token replacement).
- Per-head stability: `S(l,h) = 1 − std/(|mean| + ε)`.
- Global rank consistency: vectorised Kendall τ-b (Kendall 1938) across all C(K,2) pairs.
- No scipy dependency.

**`analyze()` updated** — `include_logit_lens: bool = False` parameter added.

### Changed
- `__version__` bumped to 2.2.0.
- Module docstring updated with new references (Dar et al. 2023, Syed et al. 2024, Kendall 1938).
- Complexity table updated.

### Infrastructure
- `.github/workflows/deploy_hf.yml` — GitHub Actions auto-sync to HuggingFace Space.
- `.github/workflows/publish.yml` — OIDC Trusted Publisher (no API tokens needed).
- PyPI package published at version 2.2.0.

---

## [2.1.0] — 2025-03-10

### Added

**MLP Attribution** (`GlassboxV2.mlp_attribution()`) — per-layer MLP contribution
via `hook_mlp_out`. Completes the circuit picture beyond attention heads. 3 passes.

**Integrated Gradients** — `attribution_patching(method="integrated_gradients")`.
Path-integral attribution (Sundararajan et al. 2017). Costs 2+n_steps passes.
Set `method="integrated_gradients"` in `analyze()` to propagate through.

**Bootstrap 95% CI** (`GlassboxV2.bootstrap_metrics()`) — nonparametric bootstrap
over N prompt triples. Returns mean, std, ci_lo, ci_hi for Suff/Comp/F1.

---

## [2.0.0] — 2025-01-20

### Added

- `GlassboxV2` class — full rewrite of the interpretability engine.
- Attribution patching (Taylor, O(3)) — Nanda et al. (2023).
- Minimum faithful circuit discovery (greedy forward/backward pruning).
- Faithfulness metrics: sufficiency, comprehensiveness, F1 (ERASER framework).
- Functional Circuit Alignment Score (FCAS) — novel cross-model metric.
- Interactive Streamlit dashboard.
- PyPI package `glassbox-mech-interp`.
- CLI: `glassbox-ai analyze`.

### Removed
- `GlassboxEngine` (v1.x class) — replaced by `GlassboxV2`.
  Shim alias kept in `alignment.py` for back-compat.

---

## [1.0.0] — 2024-09-01

Initial release. Basic attribution patching for GPT-2 small on IOI task.
