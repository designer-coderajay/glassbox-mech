# MiroFish seed brief: "My AI system changed behaviour; how do teams investigate?" (V7)

*For the owner to run on their own machine with the `mirofish-simulation` skill. MiroFish
is AGPL-3.0: run it as a separate app, never import it into Glassbox. Its output is
**simulated hypotheses, not evidence**. Keep it below 40 rounds, because every round
costs LLM API calls. Do not upload confidential material: it is sent to the LLM and Zep
providers.*

## 1. The event

Over a few weeks, several teams running LLM, RAG and agent systems see behaviour change
after routine updates:
- a provider model-version upgrade;
- a prompt-template edit;
- an agent-framework minor upgrade;
- a re-index of the document store with new chunking.

Users or support notice first. Engineers then have to find what changed, why, and prove
it to others.

## 2. Background, from public sources

These sources were accessed 2026-10-05. Re-check them before running.
- A Haystack feature request proposes run recording, deterministic replay and diffing
  "to reproduce LLM bugs" — https://github.com/deepset-ai/haystack/issues/11836
- A real llama_index regression: the sync retrieval path dropped nodes that the async
  path kept; fixed in a later release —
  https://github.com/run-llama/llama_index/issues/21033
- Hosted LLM observability and tracing tools exist (e.g. LangSmith, Langfuse, Arize
  Phoenix) and are widely discussed. Describe them neutrally, without feature claims.

## 3. The product (describe exactly as built; no planned features)

Glassbox V7 is an open-source (MIT) Python toolkit that currently:
- records each step of an AI-system run locally as a trace, storing only content hashes
  by default;
- diffs two runs to show the first step where they diverge;
- packages a pre-registered experiment (traces, protocol, results) into a bundle whose
  file integrity can be verified.

It does **not** yet:
- detect regressions automatically;
- attach to frameworks without manual instrumentation;
- run automated one-factor-at-a-time interventions.

Two pre-registered experiments passed: an injected RAG fault, and the llama_index bug
above. Neither was blind.

## 4. Stakeholder groups to simulate

| Group | What they care about | Starting attitude (assumption) |
|---|---|---|
| ML engineers at SaaS companies (hosted LLM APIs) | time to diagnose, model upgrades they don't control | neutral, busy |
| Solo/startup agent builders | cost, drop-in setup, flaky tool calls | interested but price-sensitive |
| MLOps at regulated firms (banks, insurers) | on-prem, root-cause evidence for model-risk | cautious |
| Staff engineers with mature eval stacks | detection and eval coverage, vendor overlap | skeptical |
| OSS maintainers / research engineers | reproducible bug reports, confounded changes | curious, no budget |

## 5. The question for the simulation

Which investigation steps do these groups find slowest, which objections do they raise to
a local trace-and-diff tool, and what would make them try it on their next incident?

## Report back

Follow the format in the skill (Step 4). Then compare the result with
`SIM_2026-10-05_SYNTHESIS.md`. Agreement between two simulations is still not evidence.
Only the five real interviews count.
