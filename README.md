<div align="center">

<img src="docs/assets/og-image.png" alt="Glassbox — see inside the black box" width="820" style="max-width:100%;margin-bottom:8px;border-radius:12px"/>

# Glassbox

**Prove what your model actually did, and find out why your AI system changed.**

Trace the circuit behind a model's decision, measure how *faithful* that explanation is, and draft the EU AI Act **Annex IV** technical documentation from one function call, on open-weight models. A newer investigation layer compares runs of an AI system before and after a change and tests which change actually mattered.

[![PyPI version](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue)](https://pypi.org/project/glassbox-mech-interp/)
[![PyPI downloads](https://img.shields.io/pypi/dm/glassbox-mech-interp?color=blue&label=downloads%2Fmonth)](https://pypistats.org/packages/glassbox-mech-interp)
[![License: MIT core](https://img.shields.io/badge/Core-MIT-green.svg)](LICENSE) [![License: BSL 1.1](https://img.shields.io/badge/Compliance%20%2B%20Investigation-BSL%201.1-orange.svg)](LICENSE-COMMERCIAL)
[![Python 3.11 (tested in CI)](https://img.shields.io/badge/python-3.11%20tested-blue.svg)](.github/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)
[![Website](https://img.shields.io/badge/Website-glassbox--ai-blue)](https://repo-ashen-psi.vercel.app)

[**Website**](https://repo-ashen-psi.vercel.app) · [**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) · [**Paper**](https://arxiv.org/abs/2603.09988) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/) · [**User Guide**](docs/USER_GUIDE.md) · [**What is proven**](ASSURANCE.md)

</div>

---

Most explainability tools describe a model **from the outside** and can't tell you whether the explanation is true. Glassbox reads the **inside**. It ranks every attention head by its causal effect on a decision, keeps the minimal circuit that drives it, and then *measures* how faithful that circuit is by re-running the model with parts ablated.

> **The finding behind the tool:** a model's confidence is essentially *uncorrelated* with how faithful its explanation is (**r = 0.009**). An auditor who relies on a confidence score is relying on something unrelated to faithfulness. Faithfulness has to be measured, not assumed.

**Two parts, two maturity levels:**

| Part | What it does | Status |
|---|---|---|
| **Circuit + faithfulness + Annex IV** | Explains one model decision and drafts documentation | Released on PyPI (`glassbox-mech-interp` 4.6.1) |
| **Investigation layer** (`glassbox/v7/`) | Explains why an AI system's behaviour changed between two versions | Included in the PyPI release from 4.6.0; research stage (see [below](#investigating-why-an-ai-system-changed)) |

Every claim in this README is tracked, with its evidence and status, in [`ASSURANCE.md`](ASSURANCE.md).

---

## Install

```bash
pip install glassbox-mech-interp
```

Extras: `pip install "glassbox-mech-interp[compliance]"` (Annex IV engine) · `[jupyter]` (notebook widget) · `[dev]` (tests + lint).
No install? Try the **[live demo on Hugging Face](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)**.

The investigation layer (`glassbox.v7`) ships in the same package from 4.6.0. To re-run the experiments or reproduce an evidence package, install from source, since the experiments and recipes are not in the wheel:

```bash
git clone https://github.com/designer-coderajay/glassbox-mech.git
cd glassbox-mech
pip install -e ".[dev]"
```

## 60-second quickstart: explain one decision

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")   # any open-weight transformer
gb    = GlassboxV2(model)

result = gb.analyze(
    prompt    = "When Mary and John went to the store, John gave a drink to",
    correct   = " Mary",
    incorrect = " John",
)

print(result["circuit"])         # [(9, 9), (9, 6), ...]  <- (layer, head)
print(result["faithfulness"])    # {'sufficiency': 1.0, 'comprehensiveness': 0.543, 'f1': 0.704, ...}
```

Measured on GPT-2 IOI with the current build: **suff 1.00 · comp 0.543 · F1 0.704 · Grade B**, in about 1.8 s on an Apple M1 Pro.

## What you get back

- **The circuit:** which attention heads causally drive the decision, ranked.
- **Faithfulness metrics and a grade (A–D):** sufficiency, comprehensiveness and their F1, *computed by ablation*, so they are reproducible rather than asserted.
- **A plain-English explanation** and a self-graded evidence tier.
- **An EU AI Act Annex IV draft:** all 9 sections laid out. Fields the analysis can measure (circuit, faithfulness, model details) are filled; fields only the provider can supply, including the §8 declaration of conformity, are marked `[PROVIDER TO COMPLETE]`.

```python
from glassbox import AnnexIVReport
report = AnnexIVReport(model_name="gpt2", system_purpose="...", provider_name="Acme Corp",
                       provider_address="...")
report.add_analysis(result)
report.save_json("annex-iv.json")
```

## Honest by design

When a model can't do a task, Glassbox says so instead of inventing a clean story. Point it at raw GPT-2 for a credit decision and it finds no faithful circuit: **F1 0.00–0.08, Grade C, NON-COMPLIANT**. That refusal is intended; the tool won't certify what it can't explain. *("NON-COMPLIANT" here is Glassbox's own quality gate, not a legal ruling.)*

## Measured results

| | Value |
|---|---|
| **Method** | Attribution patching (activation difference × gradient): **3 forward passes**, not a search |
| **Speed** | About **1.8 s** for GPT-2 Small on an Apple M1 Pro (4.2 s on an 8-core CPU); **15–37× faster** than ACDC across the GPT-2 family, depending on circuit size and hardware (see `BENCHMARKS.md`) |
| **IOI benchmark** (GPT-2 Small, default circuit) | suff 1.00 · comp 0.543 · **F1 0.704 · Grade B** (1-head circuit). Full 26-head Wang et al. circuit: suff 1.00 · comp 0.47 · F1 0.64 |
| **Key finding** | Confidence–faithfulness correlation **r = 0.009** |
| **Validated scale** | **82M → 12B** across **9 architecture families** (10 model series) |
| **Quality** | **1,478 tests** passing, 80% coverage (full local run 2026-10-31 on 4.6.1; see `docs/VALIDATION_LOG.md` Run 17) |

Full methodology and raw data: [`BENCHMARKS.md`](BENCHMARKS.md). Reproduce with `scripts/benchmark.py`.

## How it compares

| Approach | Faithfulness measured? | Needs open weights? | Speed (GPT-2) |
|---|---|---|---|
| **Glassbox** (attribution patching) | ✅ suff / comp / F1 | Yes | **1.8 s** |
| ACDC (Conmy et al. 2023) | ✅ | Yes | ~65 s |
| SHAP / LIME (black-box) | ❌ no guarantee | No (works on APIs) | varies |
| Confidence scores | ❌ (r = 0.009) | No | instant |

## Model coverage

Validated end-to-end on **nine architecture families** (ten model series; Yi uses the Llama architecture) from **82M to 12B**: GPT-2, Pythia/GPT-NeoX, GPT-Neo, OPT, Llama-3, Mistral, Gemma-2, Qwen2/2.5, Yi and Phi-3, with grouped-query attention and RMSNorm handled correctly.
- **Beyond about 13B:** gradient-based attribution needs a multi-GPU cluster. This is implemented but not yet validated live.
- **Closed APIs** (GPT-4, Claude, Gemini): only a weaker black-box tier is available, because faithfulness can't be measured without the weights.

See [`docs/VALIDATION_LOG.md`](docs/VALIDATION_LOG.md).

## EU AI Act: Annex IV

Providers of high-risk AI systems must draw up Annex IV technical documentation (Article 11 of [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)). Glassbox drafts the 9-section structure from one analysis: measured evidence fills what it can, and the rest is marked for the provider. Report packs also map to the NIST AI RMF and ISO/IEC 42001.

Application dates for high-risk obligations were amended by [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj) (the Digital Omnibus on AI). Check the current legal text for the dates that apply to your system; this README does not restate them.

> Glassbox produces **evidence and documentation**, not a conformity declaration. It is **not legal advice** and does not by itself make anyone compliant. The EU AI Act does not require mechanistic interpretability. See [Legal Notices](docs/USER_GUIDE.md#legal-notices--regulatory-disclaimer).

## Investigating why an AI system changed

*Research stage. In the PyPI package from 4.6.0. Licensed under BSL 1.1.*

When a RAG assistant or an agent starts behaving differently after a week of changes (new index, edited prompt, new framework version), the investigation layer:

1. **Records** each run as a local trace: nested spans, tool calls and hashed content. Raw text is not stored by default.
2. **Compares** many runs before and after the change. A noise-aware group diff separates real differences from run-to-run sampling noise.
3. **Tests** each change alone and each change reverted alone, then reports which changes moved the outcome and which did not.
4. **Packages** the result as an evidence bundle with a sha256 manifest that anyone can verify.

Try the built-in demo, a small RAG incident with three simultaneous changes:

```bash
python -m glassbox.v7.investigate demo --out demo_out --scorer gpt2
```

It writes a plain-language `REPORT.md` and an evidence package. The live demo has two worked examples: this RAG incident on GPT-2, and a simulated tool-using agent whose true effects are known and shown next to Glassbox's answer.

**On your own system**, a "change" is any configuration difference you can switch on and off and re-run: model version, prompt, retrieval setting, framework or tool version. You supply a function that runs your system under a given configuration and records a trace, plus one that reads the outcome; `glassbox.v7.intervene.run_interventions` does the rest. Only the change types listed in the experiments below have been tested.

### What has been tested so far

Each experiment was pre-registered: the protocol was committed before the code and the run, and the results were committed unchanged. All were designed and run by the author, with the cause known in advance; none has yet been reproduced independently.

| Experiment | Result | Status in [`ASSURANCE.md`](ASSURANCE.md) |
|---|---|---|
| 1. Controlled RAG fault (stale index) | First divergence at retrieval for 4/4 affected questions; silent on 4/4 unaffected; recovery confirmed 8/8 | Experimentally supported |
| 2. Real third-party bug (llama_index #21033) | Difference localised to retrieval; fix confirmed in a later version | Experimentally supported |
| 3. Group diff, single injected change, N = 30 | Localised in 0.79 of replications against a pre-declared 0.80 bar: **FAIL** | Refuted |
| 3b. Same, with redundant fields merged | 0.875 localised; false alarms ≤ 0.02 at every N from 10 to 50 | Experimentally supported |
| 4. Interventions, synthetic pipeline | Real cause blamed in 0.825 of replications; inert co-change blamed in ≤ 0.025 | Experimentally supported |
| 5. Interventions, GPT-2 RAG (3 co-changes) | Known fault blamed 4/4; zero-effect and inert changes never blamed. True template effects of 0.108 and 0.046 were **not** detected | Experimentally supported |
| 6. Same investigation on Pythia-160m, OPT-125m, Qwen2.5-0.5B | Every large true effect blamed (3/3, 3/3, 4/4); none of 36 zero effects. Detected effects of about 0.41; missed effects of about 0.16 | Experimentally supported |
| 7. Tool-using LLM agent (duplicate-email incident) | The agent almost never acted, so the test could not be scored: **INCONCLUSIVE** | Unresolved |
| 7b. Same incident with an instruction-tuned agent (Qwen2.5-3B-Instruct) | Large real cause blamed 4/4; inert change never blamed; hidden retries traced. **PASS** in two runs. The change was mostly "stopped sending", not duplicates. A mid-sized cause (about 0.34) was found 2/4 | Experimentally supported (second run matched every decision; 6th-decimal differences traced to floating-point summation order by a pre-registered check) |
| 8. Same fault on a second stack (LlamaIndex, traced by OpenInference/OpenTelemetry, imported from OTLP) | Import lossless 32/32; unchanged diff engine localised the fault 4/4, silent 4/4, recovery 8/8, deterministic 8/8; Glassbox traces pass the official OTLP schema 32/32 | Experimentally supported (retrieval only; one framework) |
| 9. Re-run 8 packaged experiments on Linux x86-64 (Kaggle) instead of the original macOS arm64 | 7/8 reproduced exactly (incl. Experiment 3's FAIL). Experiment 6: every decision the same, but one sampled run flipped from tiny float differences, so **FAIL** as pre-registered | Refuted (same-author; Gate 5 not met) |
| 9b. Same 8 packages on a fresh GitHub Actions runner (AMD EPYC), judged on gated decisions | 8/8 same decisions; 7/8 also exact. Experiment 6 differs only in floats (max 0.004). **PASS** | Experimentally supported (same-author; decision level; criterion chosen after Experiment 9, stated openly) |
| 10. Benchmark: 32 sealed-seed known-cause incidents (GPT-2 RAG + synthetic agent), Glassbox vs two automated baselines | Exact diagnosis 21/28 (0.75) vs 0.50 and 0.00; zero-effect causes blamed 1/299; false alarms 1/48; fix verified 20/24. All 5 pre-declared criteria **PASS**. Missed 4 units where co-changes cancelled out; small effects mostly not diagnosed | Experimentally supported (author-designed benchmark; 26/28 positives from one family; not blind) |

**Not shown yet:**
- that it finds **unknown** causes in real production systems;
- that it works across frameworks in general (so far: one second stack, retrieval only);
- that it saves engineers time. There are **no external users so far**.

A first divergence shows *where* runs differ, not *why*. Causal statements need the controlled interventions above, case by case.

## How it works, start to finish

### The whole system in one picture

<p align="center">
  <img src="docs/anatomy/glassbox-bench.gif" alt="An animated isometric pegboard. A board of 144 pegs, one per attention head, rises to each head's causal score. The top few light up as the circuit. Keeping only the circuit reads sufficiency 1.00 on one dial, removing it reads comprehensiveness 0.543 on a second, and a third reads F1 0.704 and sets the grade plate to B." width="900">
</p>

| On the bench | In Glassbox |
|---|---|
| The prompt cards on the tray | The decision to explain: a prompt, the answer it should give, and the wrong one, plus a corrupted copy of the prompt |
| 144 pegs rising to different heights | Attribution patching scores every attention head in 3 forward passes (GPT-2 small: 12 layers by 12 heads) |
| A few pegs lighting up | The circuit: the smallest set of heads that still carries the decision |
| Dial S, only the circuit left | Sufficiency: does the circuit alone reproduce the decision? |
| Dial C, the circuit removed | Comprehensiveness: does removing it break the decision? |
| Dial F1 and the grade plate | F1 of the two, and the letter grade A to D |

Open [`docs/anatomy/glassbox-bench-light.html`](docs/anatomy/glassbox-bench-light.html) in a browser to scrub through it (details in [`docs/anatomy/README.md`](docs/anatomy/README.md)). The three dial readings are the IOI benchmark on GPT-2 small from [Measured results](#measured-results); the per-head scores and which heads are lit are drawn for the picture, not taken from a run.

Two paths, two maturity levels. **Path 1** (explain one decision) is released on PyPI. **Path 2** (find out why an AI system changed) is the investigation layer, research stage; see [above](#investigating-why-an-ai-system-changed).

### Path 1: explain one decision (`GlassboxV2.analyze`)

<p align="center">
  <img src="docs/illustrations/04-find-circuit.png" alt="A creature shines a flashlight at a wall of one hundred small squares, one per attention head; a few of them glow red. The note reads: keep the few that carry the decision." width="860"><br>
  <sub><b>Find the circuit.</b> Score every head in 3 passes, keep the few that carry the decision.</sub>
</p>

<p align="center">
  <img src="docs/illustrations/05-test-by-ablation.png" alt="A creature pulls a few squares out of a board of heads while a dial shows whether the decision breaks. The note reads: ablate, don't assert." width="860"><br>
  <sub><b>Test it by taking it out.</b> Faithfulness is computed by re-running the model with parts ablated, not asserted.</sub>
</p>

| Step | What happens | Why it matters |
|---|---|---|
| **1. Frame the decision** | You pass a prompt plus the token the model *should* output and a distractor (for example ` Approved` vs ` Denied`). | Attribution needs a contrast between two outcomes, so this is the question being asked. |
| **2. Corrupt the prompt** | Glassbox builds a counterfactual prompt (`name_swap`, `antonym`, `semantic_negation` or `random_token`, chosen automatically). | The difference between clean and corrupted runs is what each head gets credit for. |
| **3. Rank every head** | Attribution patching (activation difference × gradient) scores all heads in **3 forward passes**. | Fast, so it can run on every audit instead of searching exhaustively. |
| **4. Shrink to a circuit** | Greedy forward selection adds top heads until approximate sufficiency is reached (default target 0.85), then backward pruning drops heads that are not needed. | The smallest set of heads that still carries the decision. |
| **5. Check the counterfactual** | A gate checks that the corrupted prompt really changes the decision by more than noise. Failures are reported, never hidden. | A bad contrast makes every later number meaningless. |
| **6. Measure faithfulness** | **Sufficiency**: does the circuit alone reproduce the decision? (a Taylor approximation by default) **Comprehensiveness**: does removing it break the decision? (exact, by activation patching) **F1** combines the two. | These are computed by re-running the model, not asserted. |
| **7. Grade and label** | Letter grade A to D, a self-assigned evidence tier, and a category such as `faithful`, `backup_mechanisms` or `incomplete`. | Tells an auditor how far to trust the explanation. |
| **8. Draft Annex IV** | `AnnexIVReport` lays out all 9 sections, fills the fields the analysis can measure, and marks provider-only fields `[PROVIDER TO COMPLETE]`. | Documentation, not a conformity declaration. |

### Path 2: find out why an AI system changed (`glassbox.v7`)

<p align="center">
  <img src="docs/illustrations/06-why-it-changed.png" alt="Three switches labelled new index, edited prompt and new framework, with one switched on alone. The notes read: switch each one on alone, then revert each alone; where runs differ is not why." width="860"><br>
  <sub><b>Which change mattered?</b> Switch each change on alone and revert each alone. Where runs first differ is not why.</sub>
</p>

| Step | What happens | Result |
|---|---|---|
| **1. Record** | Each run becomes a local trace: nested spans, tool calls and hashed content. | Raw text is not stored by default. |
| **2. Compare** | A noise-aware group diff of runs before and after, separating real differences from run-to-run noise. | **Where** the runs first differ. |
| **3. Intervene** | Switch each change on alone, and revert each alone. | **Which** change moved the outcome. Causal claims come only from this step. |
| **4. Package** | `REPORT.md` plus an evidence bundle with a sha256 manifest. | Anyone can verify the manifest. |

Run the built-in demo, a small RAG incident with three simultaneous changes:

```bash
python -m glassbox.v7.investigate demo --out demo_out --scorer gpt2
```

### Which entry point should I use?

| I want to... | Use | Needs weights? |
|---|---|---|
| Explain one decision from Python | `GlassboxV2(model).analyze(...)` | Yes |
| Do the same from a terminal | `glassbox-ai analyze ...` (also `glassbox-ai doctor` to check your install) | Yes |
| Use it from Claude Desktop or another MCP client | `pip install glassbox-mcp`, see [`mcp/README.md`](mcp/README.md) | Yes |
| Just look around | The [live demo](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) | No |
| Investigate a changed RAG or agent system | `python -m glassbox.v7.investigate` | No (it needs traces, not weights) |

## Documentation

| | |
|---|---|
| **[User Guide](docs/USER_GUIDE.md)** | Full walkthrough, all examples, complete API reference |
| **[BENCHMARKS.md](BENCHMARKS.md)** | Methodology, hardware, reproducible numbers |
| **[ASSURANCE.md](ASSURANCE.md)** | Every claim with its evidence and status (supported, refuted, unresolved, planned) |
| **[Methodology & Assurance](docs/METHODOLOGY_AND_ASSURANCE.md)** | What's proven vs measured vs hypothesised |
| **[Trace spec](docs/v7/TRACE_SPEC.md)** | The investigation layer's trace format |
| **[CHANGELOG.md](CHANGELOG.md)** | Release history |
| **[Live docs site](https://repo-ashen-psi.vercel.app)** | Interactive overview |

## Limitations

- **White-box only for circuits:** circuit analysis needs weights, activations and gradients. Open models are fully supported; closed APIs get the black-box tier only.
- **Decisions, not free-form chat:** circuit analysis needs a contrast between two outcomes to attribute against.
- **Offline and sampled:** Glassbox audits representative decisions for documentation and monitoring. It is not a real-time monitor on every request.
- **Faithful ≠ fair:** it measures whether the explanation matches the computation, not whether the decision is correct or unbiased.
- **Investigation layer:**
  - so far validated only on small open-weight models and author-designed incidents;
  - small effects (about 0.16 and below at 30 runs per configuration) can be missed;
  - redaction of trace content is best-effort;
  - a hash manifest proves integrity, not truth.

## Developer tooling: diagram and illustration skills

Two optional [Agent Skills](https://agentskills.io) are set up under `.claude/skills/` for people who work on this repo with Claude Code. `.claude/` is git-ignored here, so they live in your checkout and are not shipped with the package.

| Skill | Use it for | Source |
|---|---|---|
| `anatomy` | Interactive isometric SVG/WebGL figures that explain an idea as a working object. Needs Node 20.10+. | [wheresryan22/anatomy](https://github.com/wheresryan22/anatomy) (MIT) |
| `ian-xiaohei-illustrations` | Hand-drawn 16:9 article illustrations in Ian's "Xiaohei" style, for Chinese-language text. It writes image prompts, so it needs an image-generation model. | [helloianneo/ian-xiaohei-illustrations](https://github.com/helloianneo/ian-xiaohei-illustrations) (MIT; keep attribution per its `NOTICE.md`) |

Re-install either with `git clone <source> .claude/skills/<name>`.

The bench figure in [`docs/anatomy/`](docs/anatomy/) is built with `anatomy`; the three English illustrations in [`docs/illustrations/`](docs/illustrations/) follow the Xiaohei style and are hand-drawn vectors with their generator included, not AI-generated art.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md), and run the suite with `pytest --cov=glassbox -v`.

## License

Glassbox is dual-licensed. [NOTICE](NOTICE) holds the authoritative file-by-file map.

| Component | License | Summary |
|---|---|---|
| Interpretability core (circuit discovery, faithfulness, everything not listed below), tests, examples, docs | **MIT** ([LICENSE](LICENSE)) | Use, modify and redistribute freely, including commercially |
| Compliance engine: `compliance.py`, `circuit_diff.py`, `risk_register.py`, `bias.py`, `audit_log.py`, `cross_model.py` | **BSL 1.1** ([LICENSE-COMMERCIAL](LICENSE-COMMERCIAL)) | See the summary below |
| Investigation layer: `glassbox/v7/`, `dashboard/v7_investigation.py`, `dashboard/v7_agent_demo.py` | **BSL 1.1** (from 2026-10-11) | See the summary below |

**BSL 1.1 summary:**
- **Free** for non-production use, and for production use inside your own organisation (for example, investigating or documenting your own AI systems).
- **Needs a commercial license** to offer a compliance, AI-governance, AI-investigation, debugging or observability service or product to third parties.
- **Converts to Apache-2.0** four years after each version is first published.
- Copies of `cross_model.py` and the investigation layer obtained before 2026-10-11 remain under the terms they were published with.

The circuit-discovery and faithfulness core stays MIT and is never feature-gated. For commercial licensing, contact mahale.ajay01@gmail.com. Academic and non-profit research use may qualify for a no-cost license.

## Citation

```bibtex
@article{mahale2026causally,
  title         = {Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations},
  author        = {Mahale, Ajay Pravin},
  year          = {2026},
  eprint        = {2603.09988},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2603.09988}
}
```

---

<div align="center">
<sub>See inside the black box: causal circuits, measured faithfulness, EU AI Act Annex IV, and evidence for why an AI system changed.</sub>
</div>
