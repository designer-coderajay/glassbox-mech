# Seed brief: <scenario title>

<!--
Fill every <...> with facts the user confirms. Keep each claim sourced or marked as an
assumption: agents treat everything here as true, so an unsourced claim becomes a "fact" in
the simulated world. Delete sections that do not apply. Do not include confidential customer
data or real private individuals.
-->

## 1. The event

<What happens, when, and where. One paragraph. Example: "On <date>, a European bank's model-risk
team learns its credit-scoring LLM is classed as high-risk under the EU AI Act and must produce
Annex IV technical documentation before <deadline>.">

## 2. Regulatory background (verify dates before each run)

- The EU AI Act (Regulation (EU) 2024/1689) requires providers of high-risk AI systems to keep
  technical documentation as set out in Annex IV (Article 11).
- <Applicable dates. The Glassbox README states high-risk obligations apply from 2 August 2026,
  and that the Digital Omnibus (provisionally agreed 7 May 2026, pending adoption) would defer
  Annex III obligations to 2 December 2027. Check the current status and date your source.>
- <Penalty exposure, if relevant, with the article cited.>

## 3. The product

Glassbox: open-source tool (MIT core, BSL compliance engine) that traces which attention heads
drive a model's decision, measures how faithful that explanation is by ablation (sufficiency,
comprehensiveness, F1, grade A-D), and generates Annex IV documentation. It works on open-weight
models; closed APIs get only a weaker black-box tier. It produces evidence, not legal advice or
a conformity declaration.

Key claim to test: <e.g. "model confidence is nearly uncorrelated with explanation faithfulness
(r = 0.009 in Glassbox's study)". Cite the source: arXiv 2603.09988 / BENCHMARKS.md.>

## 4. Alternatives the stakeholders know

<e.g. SHAP/LIME-based explainability, GRC/documentation platforms, in-house validation teams,
other interpretability tools. Describe each neutrally, from public information only.>

## 5. Stakeholder groups to simulate

| Group | What they care about | Starting attitude (assumption) |
|---|---|---|
| Bank compliance officers | audit-proof documentation, deadlines, liability | <...> |
| ML / model-risk engineers | effort, model access, false positives | <...> |
| External auditors | evidence standards, reproducibility | <...> |
| Regulators / supervisors | enforceability, consistency | <...> |
| Interpretability researchers | methodological rigour | <...> |

## 6. The question for the simulation

<One sentence that becomes MiroFish's "simulation requirement", e.g. "Which objections to
faithfulness-based Annex IV evidence emerge among these groups over the first weeks after
the event, and which arguments change minds?">

## Sources

- <URL or document, date accessed>
