# GLASSBOX V7 — FINAL UNIFIED PLAN
## Scientific Infrastructure for Investigating AI Behavior

**Status:** Strategic + technical master plan  
**Relationship to V6:** V6 remains frozen as a completed scientific foundation. V7 does not rewrite V6 results.  
**Primary product promise:** **Investigate why your AI system behaved differently.**

---

## 0. Executive Thesis

Glassbox should evolve from a mechanistic-interpretability research repository into **model- and framework-agnostic investigation infrastructure for AI systems**.

The central problem is not:

> “Can I evaluate whether my AI passed a benchmark?”

It is:

> **“My AI system behaved differently, failed, or changed. What happened, why might it have happened, how can I test the explanation, and what evidence supports the conclusion?”**

Glassbox should make that investigation reproducible.

The long-term system is:

```text
AI system
   ↓
Trace
   ↓
Behavior + trajectory + environment
   ↓
Diff
   ↓
First divergence / anomaly
   ↓
Hypotheses
   ↓
Controlled experiment
   ↓
Intervention
   ↓
Reproduction
   ↓
Evidence
```

The scientific principle is:

> **AI proposes. Evidence certifies.**

Glassbox should never require users to trust an explanation merely because the software produced it.

---

# 1. North Star

### Product

> **Glassbox — investigate why your AI system behaved differently.**

### Scientific mission

> Build reproducible experimental infrastructure for understanding, comparing, challenging, and explaining AI-system behavior.

### Long-term category

**AI investigation infrastructure / scientific instrumentation for AI behavior.**

Glassbox is not primarily:

- a generic LLM evaluation dashboard,
- a LangSmith clone,
- a mechanistic-interpretability-only toolkit,
- an EU AI Act compliance product,
- an observability dashboard,
- a benchmark leaderboard,
- an autonomous scientist that claims causality without evidence.

Those can be components or applications.

The core is **investigation**.

---

# 2. The User Problem

The recurring user problem is:

> “Something changed in my AI system. I need to know what changed, where it diverged, why it may have happened, and whether the proposed explanation survives testing.”

Examples:

- A model upgrade increased hallucinations.
- A RAG system started retrieving irrelevant documents.
- An agent began selecting the wrong tool.
- The same prompt produces different trajectories.
- A successful agent became unreliable after a framework change.
- A production incident cannot be reproduced.
- An evaluator changed and scores suddenly moved.
- A prompt change caused a downstream behavior regression.
- A model behaves differently despite similar benchmark performance.
- An open-weight model's internal attribution profile changed.
- A regulated organization needs reproducible evidence for an AI incident.

---

# 3. The Killer Workflow

The central Glassbox workflow should eventually be:

```text
existing AI system
      ↓
glassbox trace
      ↓
successful run / failed run
      ↓
glassbox diff
      ↓
first meaningful divergence
      ↓
candidate hypotheses
      ↓
controlled experiment
      ↓
intervention
      ↓
reproduction
      ↓
evidence package
```

This workflow matters more than the number of integrations, dashboards, metrics, or algorithms.

---

# 4. Model-Agnostic Principle

Glassbox should not be architected around Transformer internals.

The core object is:

> **an AI-system execution and the evidence surrounding it.**

Investigation depth depends on what the system exposes.

## Evidence levels

### Level 1 — Behavioral

Works with almost any observable AI system.

Capture:

- input
- output
- model identifier
- model version where available
- configuration
- latency
- token usage
- cost
- errors
- retries
- success/failure
- metadata

Question:

> What happened?

---

### Level 2 — System

Capture the surrounding application:

- prompts
- retrieval
- retrieved documents
- tool calls
- tool results
- memory
- state
- application events
- environment
- configuration
- external services
- failures

Question:

> What happened inside the AI application?

---

### Level 3 — Trajectory

For agents:

- model calls
- planning steps
- tool selection
- tool results
- state transitions
- memory reads/writes
- retrieval paths
- retries
- delegation
- sub-agents
- environment interaction
- terminal outcome

Question:

> How did the AI system get there?

---

### Level 4 — Mechanistic

For accessible/open-weight models:

- activations
- gradients
- attribution
- activation patching
- causal interventions
- circuit hypotheses
- attribution profiles
- representation comparisons

Question:

> What internal computation contributed to the observed behavior?

Mechanistic interpretability is therefore a **depth layer**, not the entire product.

---

# 5. Glassbox Trace Protocol

The foundational V7 infrastructure should be a model/framework-independent **Glassbox Trace Protocol**.

A trace should represent an execution as a structured graph.

## Trace node

A node can represent:

- model invocation
- prompt
- response
- tool call
- tool result
- retrieval
- memory operation
- state update
- planner step
- evaluator step
- environment event
- error
- retry
- human intervention
- sub-agent invocation

## Trace edge

Edges describe:

- causal/temporal ordering where supported
- parent/child relationship
- state transition
- tool dependency
- retrieval dependency
- delegation
- retry relationship

The protocol must distinguish:

- observed facts,
- inferred relationships,
- user-provided metadata,
- system-generated hypotheses.

---

# 6. Connection Architecture

Glassbox should support several ways to connect an AI system.

## A. Python wrapper

```python
with glassbox.trace("customer-support"):
    result = agent.run(ticket)
```

## B. Framework adapter

Adapters for systems such as:

- LangGraph
- LangChain
- OpenAI Agents SDK
- Anthropic-based agents
- CrewAI
- AutoGen
- LlamaIndex
- Semantic Kernel
- Google ADK
- custom Python agents

## C. OpenTelemetry-compatible ingestion

Where organizations already have telemetry, Glassbox should ingest compatible execution information rather than requiring a proprietary runtime.

## D. REST / event ingestion

Any language or application should be able to emit a Glassbox-compatible trace.

### Important wording

Do not claim:

> “Glassbox supports every AI framework.”

Instead:

> **Glassbox provides a common investigation protocol for AI systems regardless of model or framework; investigation depth depends on the observability and access available.**

---

# 7. Privacy-First Architecture

AI traces can contain:

- personal data,
- confidential documents,
- credentials,
- proprietary prompts,
- customer information,
- source code,
- model outputs.

Therefore V7 must be designed local-first.

Required capabilities:

- explicit capture
- configurable redaction
- PII detection
- secret detection
- configurable retention
- local storage
- encrypted storage where appropriate
- access control
- selective field capture
- trace deletion
- export control
- sensitive-content masking

The default architecture should not require users to send sensitive traces to a third-party cloud.

---

# 8. Glassbox Diff

`glassbox diff` is the most important future user interface.

Conceptually:

```bash
glassbox diff run_success.json run_failure.json
```

or:

```bash
glassbox diff run_a run_b
```

The output should answer:

1. Did the outcome change?
2. What changed?
3. Where did the executions first meaningfully diverge?
4. Did retrieval change?
5. Did tool selection change?
6. Did tool results change?
7. Did state change?
8. Did model behavior change?
9. Did latency/cost/error behavior change?
10. What hypotheses explain the difference?
11. Which hypotheses have evidence?
12. What experiment should be run next?

The goal is not:

> 500 metrics.

The goal is:

> **A useful diagnosis.**

---

# 9. First Divergence

A core V7 concept is the **first meaningful divergence**.

Two executions may have:

- the same input,
- the same final output,
- different intermediate trajectories.

Or:

- different retrieval,
- different tool choice,
- different state,
- different model response,
- different final result.

Glassbox should locate the earliest material difference that can plausibly explain downstream divergence.

This is a research problem, not a trivial diff.

The system must distinguish:

- correlation,
- temporal precedence,
- dependency,
- causal evidence.

It must not label a node “the cause” merely because it occurred first.

---

# 10. Investigation Engine

The investigation lifecycle:

```text
Observe
   ↓
Reproduce
   ↓
Compare
   ↓
Detect divergence
   ↓
Generate hypotheses
   ↓
Design controls
   ↓
Intervene
   ↓
Measure
   ↓
Attempt falsification
   ↓
Reproduce
   ↓
Package evidence
```

Each stage should produce explicit artifacts.

---

# 11. Hypothesis Engine

Glassbox may use AI to propose hypotheses.

Example:

> Hypothesis H1: retrieval ranking changed because document X entered the candidate set.

Another:

> H2: model upgrade changed tool-selection behavior.

Another:

> H3: memory state caused the agent to select an alternative branch.

But the system must distinguish:

```text
HYPOTHESIS
     ↓
TEST
     ↓
RESULT
     ↓
EVIDENCE STATUS
```

The AI assistant can propose.

The experiment determines.

The evidence determines the status.

---

# 12. Controlled Experiment Engine

Glassbox should eventually support controlled comparisons such as:

```text
Model A vs Model B
Prompt A vs Prompt B
Retriever A vs Retriever B
Tool A vs Tool B
Memory enabled vs disabled
Temperature A vs B
Framework version A vs B
Environment A vs B
```

The engine should preserve:

- fixed inputs,
- matched conditions,
- random seeds where meaningful,
- repeated trials,
- model versions,
- software revisions,
- environment,
- intervention definition,
- statistical method,
- exclusions,
- failures.

---

# 13. Intervention Engine

A diagnosis becomes stronger when changing the suspected factor changes the behavior.

Example:

```text
Observed:
retrieval document X appears before failure

Hypothesis:
document X caused the wrong answer

Intervention:
remove X

Result:
failure disappears across repeated trials

Reproduction:
independent rerun reproduces result
```

This does not automatically prove every causal claim.

Glassbox should report exactly what the intervention establishes.

---

# 14. Stochastic AI Requires Statistical Discipline

Modern AI systems are stochastic.

Therefore:

> One successful rerun is not sufficient evidence.

V7 must support:

- repeated trials
- confidence intervals
- bootstrap procedures where appropriate
- paired comparisons
- null controls
- randomization/permutation tests
- multiple-comparison correction where appropriate
- effect sizes
- uncertainty
- predefined stopping rules
- exclusion logs
- reproducible seeds where meaningful

The exact statistical method should depend on the estimand and experiment.

---

# 15. Evidence Model

Every important investigation should produce:

```text
Claim
 ↓
Evidence
 ↓
Experiment
 ↓
Controls
 ↓
Measurements
 ↓
Interventions
 ↓
Reproduction
 ↓
Provenance
 ↓
Integrity
```

Example claim:

> “Changing the retriever caused a statistically detectable change in tool-selection behavior under the specified test conditions.”

The evidence package should contain everything needed to challenge that claim.

---

# 16. Evidence Package

Proposed machine-readable structure:

```text
evidence/
├── claim.json
├── trace.json
├── experiment.json
├── measurements.json
├── controls.json
├── hypotheses.json
├── interventions.json
├── reproduction.json
├── provenance.json
├── integrity.json
└── report.html
```

This should eventually become an open **Glassbox Evidence Standard**.

---

# 17. Integrity Is Not Truth

Cryptographic hashing can show:

> “This artifact has not changed since it was hashed.”

It cannot show:

> “This experiment was scientifically correct.”

Therefore Glassbox trust requires multiple layers:

```text
Integrity
+
Provenance
+
Controls
+
Reproducibility
+
Independent verification
```

This distinction should be explicit everywhere.

---

# 18. Evidence Status Taxonomy

Use explicit statuses:

- `VERIFIED`
- `EXPERIMENTALLY_SUPPORTED`
- `REPRODUCED`
- `PRELIMINARY`
- `IMPLEMENTED`
- `HYPOTHESIS`
- `UNRESOLVED`
- `REFUTED`
- `PLANNED`
- `NOT_SUPPORTED`

A finding can remain:

> `UNRESOLVED`

That is a valid scientific result.

---

# 19. Trust Model

Glassbox should never say:

> “Trust our explanation.”

It should say:

> **“Here is the evidence, procedure, control, provenance, uncertainty, and reproduction needed to challenge the explanation.”**

Core principles:

1. Evidence before explanation.
2. Experiment before assertion.
3. Reproduction before confidence.
4. Observation separated from interpretation.
5. Hypotheses remain hypotheses until tested.
6. Unresolved is an acceptable result.

---

# 20. AI Investigation Agent

Eventually Glassbox should include an investigation agent.

User:

> “The agent started failing after yesterday's deployment. Find out why.”

Glassbox should:

1. collect relevant traces,
2. compare successful and failed runs,
3. identify candidate divergence points,
4. generate hypotheses,
5. rank experiments by information value,
6. execute permitted tests,
7. evaluate results,
8. attempt reproduction,
9. produce an evidence package.

It must **not** silently convert a hypothesis into a fact.

The investigation agent is an experimental scientist assistant, not an oracle.

---

# 21. RAG Investigation

Example:

> “Why does the system give Aditya a higher salary estimate than John?”

Glassbox should be able to compare:

```text
query
 ↓
retrieval
 ↓
documents
 ↓
ranking
 ↓
context
 ↓
model response
 ↓
final answer
```

Possible findings:

- different documents retrieved,
- stale document,
- ranking changed,
- chunking changed,
- prompt changed,
- model interpretation changed.

The system should identify the earliest measurable divergence and test candidate causes.

---

# 22. Agent Investigation

Example:

Two runs receive the same task.

### Run A

```text
retrieve
→ tool X
→ verify
→ answer
```

### Run B

```text
retrieve
→ tool Y
→ retry
→ tool Y
→ answer
```

Glassbox should show:

- trajectory difference,
- first meaningful divergence,
- downstream consequences,
- candidate explanations,
- experiment/intervention options.

This is where trajectory becomes a first-class research object.

---

# 23. Model Upgrade Investigation

Example:

```text
System before:
GPT-X + Retriever v3

System after:
GPT-Y + Retriever v3
```

Failure rate changes.

Glassbox should isolate:

```text
retrieval unchanged
tool behavior unchanged
prompt unchanged
model response distribution changed
tool-selection probability changed
```

Then test:

```text
same model + old system
new model + old system
old model + new system
new model + new system
```

This is an example of controlled factor isolation.

---

# 24. Failure Investigation

The user should be able to start from:

> “This failed.”

not:

> “I know which metric I need.”

The interface should guide them from incident → evidence.

This is a major UX principle:

> **Users should not need to understand Glassbox before Glassbox helps them investigate.**

---

# 25. Beginner UX

The first interaction should be simple:

```text
1. Connect your AI system.
2. Record two or more runs.
3. Select a successful and problematic execution.
4. Run Diff.
5. Review the first meaningful divergence.
6. Inspect hypotheses.
7. Run an experiment.
8. Export evidence.
```

Advanced users can access:

- statistical controls,
- attribution methods,
- intervention configuration,
- provenance,
- raw traces,
- experiment definitions.

---

# 26. Mechanistic Interpretability Layer

The existing Glassbox research engine becomes one investigation depth.

For open-weight models:

```text
behavior
  ↓
trajectory
  ↓
model internals
  ↓
attribution
  ↓
intervention
```

Existing V6 work remains important here.

V6 established a methodological foundation around permutation-invariant attribution-profile comparison and preregistered comparative analysis.

Its results must remain frozen and must not be rewritten to support V7 product claims.

---

# 27. V6 Claim Boundary

V7 must preserve the exact scientific boundary established by V6.

The V6 result supports a claim about:

> attribution-profile divergence under the specified estimand, task, attribution procedure, model family, and experimental design.

It does **not** by itself establish:

- different causal mechanisms,
- different circuits,
- causal effects of training,
- causal explanations of behavior,
- generalization to all architectures,
- generalization to all tasks,
- generalization to production AI systems.

This discipline is part of the Glassbox brand.

---

# 28. IOI

IOI should become:

> a research task/plugin,

not:

> the identity of Glassbox.

Future users should not need to know what IOI means.

The product identity is AI investigation.

---

# 29. Identifiability

V7 must inherit the methodological lesson from V6:

> representation and measurement choices can create apparent differences that do not correspond to functional differences.

Therefore every comparative method should ask:

- What is invariant?
- What is identifiable?
- What depends on arbitrary labels?
- What changes under equivalent representations?
- What is the estimand?
- What assumptions are required?

This should become a general Glassbox methodology.

---

# 30. Scientific Claim Taxonomy

Glassbox should distinguish:

### Observation

> “The two executions selected different tools.”

### Statistical finding

> “Tool selection differed across the predefined test set.”

### Association

> “Retrieval change was associated with the observed outcome change.”

### Intervention evidence

> “Removing the candidate document changed the failure rate under the intervention protocol.”

### Reproduction

> “The intervention result reproduced across independent runs.”

### Causal claim

Only use when the experiment actually supports the stated causal estimand.

This taxonomy should appear in the UI and evidence package.

---

# 31. Investigation Benchmark

One of the most important V7 additions should be a **Glassbox Investigation Benchmark**.

This makes the product objectively measurable.

Instead of claiming:

> “Glassbox investigates AI.”

measure whether it actually can.

## Benchmark cases

Create controlled failures with known causes:

- retrieval corruption,
- prompt change,
- model upgrade,
- tool failure,
- tool-selection shift,
- memory contamination,
- state corruption,
- framework change,
- evaluator change,
- environment change,
- stochastic regression,
- context-window effects.

Each case should have:

```text
ground truth
observations
hidden cause
expected divergence
possible confounders
required intervention
reproduction protocol
```

---

# 32. Investigation Benchmark Metrics

Measure:

### Localization

Did Glassbox identify the relevant divergence?

### Hypothesis quality

Did it generate the correct causal candidate?

### Experiment quality

Did it choose a discriminating test?

### Intervention success

Did the intervention affect the expected behavior?

### Reproduction

Could the result be reproduced?

### False investigation rate

How often did Glassbox confidently pursue the wrong explanation?

### Evidence completeness

Could another researcher reconstruct the investigation?

### Time-to-diagnosis

How long did the investigation take?

This turns Glassbox from a claim into a measurable system.

---

# 33. Benchmark as a Moat

The benchmark should eventually contain:

- synthetic controlled failures,
- public real-world failures,
- anonymized enterprise-style failures,
- agent failures,
- RAG failures,
- model-update failures,
- framework failures.

Over time, Glassbox can accumulate a validated corpus of AI-behavior investigations.

Potential strategic asset:

> **a structured corpus of AI failure → evidence → hypothesis → intervention → reproduction.**

This is more defensible than simply having another tracing dashboard.

---

# 34. Three Spectacular Demonstrations

V7 should have three flagship demos.

## Demo A — RAG Failure

Same system.

Only retrieval changes.

Glassbox:

```text
success
 ↓
retrieval differs
 ↓
document X enters context
 ↓
answer changes
 ↓
intervention removes X
 ↓
failure disappears
 ↓
reproduction succeeds
```

---

## Demo B — Agent Failure

Same prompt.

Same model.

Same tools.

Different trajectory.

Glassbox identifies the first divergence and tests the suspected tool-selection mechanism.

---

## Demo C — Model Upgrade

Old model → new model.

Same application.

Same environment.

Glassbox identifies:

- behavioral changes,
- trajectory changes,
- cost/latency changes,
- tool behavior changes,
- candidate explanation,
- controlled experiment,
- reproduction.

These demos should be understandable to an AI engineer in minutes.

---

# 35. External User Validation

Do not validate V7 only with yourself.

Give the repository to people who have never seen it.

Instruction:

> “Your AI system started behaving differently. Use Glassbox to investigate why.”

Observe:

- Can they install it?
- Can they connect their system?
- Can they create a trace?
- Can they compare runs?
- Can they understand the diff?
- Can they identify a divergence?
- Can they formulate a hypothesis?
- Can they run an experiment?
- Can they export evidence?
- Do they use it again?

Target:

> **5 independent users with real AI problems and repeated use.**

Then:

> 20 → 100 → 500.

The important metric is not stars.

It is repeated problem-solving.

---

# 36. Product-Market Validation

The central question is:

> **Does Glassbox save a skilled AI engineer meaningful time when an AI system behaves unexpectedly?**

Track:

- time-to-first-value,
- time-to-diagnosis,
- number of investigations,
- repeated investigations,
- successful reproductions,
- time saved,
- number of systems connected,
- team adoption,
- retention,
- willingness to pay,
- incidents investigated.

A useful early product signal is:

> “I had a problem I could not easily diagnose. Glassbox materially reduced the investigation time.”

---

# 37. Open Source Strategy

Open source is a distribution mechanism, not automatically a business model.

## Open-source core

Keep open:

- trace protocol,
- trace SDK,
- diff engine,
- local investigation,
- evidence schema,
- core adapters,
- reproducibility tools,
- research methods.

## Future commercial infrastructure

Potential paid layers:

- hosted traces,
- historical behavior database,
- continuous model/system diffing,
- AI incident management,
- team collaboration,
- enterprise access control,
- retention policies,
- large-scale experiments,
- private evidence storage,
- enterprise integrations,
- audit workflows,
- organization-wide investigation search.

Principle:

> **Open source creates adoption. Infrastructure creates revenue.**

---

# 38. Glassbox Evidence Standard

The long-term goal should be an interoperable evidence format.

Potential standard objects:

```text
Claim
Trace
Experiment
Measurement
Control
Hypothesis
Intervention
Reproduction
Provenance
Integrity
```

Other tools should be able to generate evidence that Glassbox can inspect.

Glassbox should not need to own every trace.

If the evidence standard becomes useful outside Glassbox, the ecosystem expands.

---

# 39. Trace Protocol as Infrastructure

A strategic objective is:

```text
LangGraph ─┐
LangChain ─┤
OpenAI SDK ┤
CrewAI ────┤
AutoGen ───┤
Custom ────┤
OpenTelemetry
            ↓
   Glassbox Trace Protocol
            ↓
   Diff / Investigation / Evidence
```

If many frameworks emit the same investigation-compatible representation, Glassbox becomes infrastructure rather than another UI.

---

# 40. Security Threat Model

V7 must explicitly address:

- prompt leakage,
- output leakage,
- credentials,
- PII,
- trace poisoning,
- evidence tampering,
- replay manipulation,
- model-version substitution,
- dependency compromise,
- malicious adapters,
- malicious plugins,
- compromised environments.

Security documentation should define:

- threat model,
- trust boundaries,
- signed artifacts where appropriate,
- permissions,
- redaction,
- provenance,
- replay protections,
- dependency pinning.

---

# 41. Evidence Poisoning

A particularly important future problem:

> What if the evidence itself is manipulated?

Glassbox should investigate:

- altered traces,
- missing events,
- substituted model versions,
- inconsistent timestamps,
- modified experiment parameters,
- incomplete tool results,
- replayed executions.

Evidence packages should therefore include provenance and integrity information.

Again:

> integrity ≠ scientific truth.

---

# 42. Reproduction Engine

A finding should ideally be replayable.

Example:

```bash
glassbox reproduce evidence/INV-00042
```

The engine should reconstruct:

- model version,
- application version,
- prompt,
- environment,
- tools,
- retrieval configuration,
- experiment parameters,
- seeds where possible.

If exact reproduction is impossible, Glassbox should state why.

---

# 43. Environment Capture

AI behavior is often environment-dependent.

Capture where possible:

- OS/runtime,
- Python version,
- package versions,
- container image,
- GPU/accelerator,
- model revision,
- tokenizer,
- API version,
- tool versions,
- retrieval index revision,
- data revision,
- framework version.

This is part of reproducibility.

---

# 44. Versioned AI Systems

The unit of comparison should often be:

```text
System A
=
model
+
prompt
+
retrieval
+
tools
+
memory
+
framework
+
environment
```

versus:

```text
System B
=
model
+
prompt
+
retrieval
+
tools
+
memory
+
framework
+
environment
```

Glassbox should identify which factors changed.

---

# 45. Multi-Agent Systems

V7 should represent parent/child execution graphs.

Example:

```text
Supervisor
 ├── Research agent
 │    ├── Search
 │    └── Browser
 │
 ├── Coding agent
 │    ├── IDE
 │    └── Test runner
 │
 └── Reviewer
```

The trace protocol must preserve:

- delegation,
- ownership,
- nested traces,
- shared state,
- tool dependencies,
- inter-agent messages.

---

# 46. Framework Independence

Do not make V7 a collection of integrations without a common abstraction.

The order should be:

```text
stable protocol
      ↓
reference implementation
      ↓
one or two adapters
      ↓
real-user validation
      ↓
more adapters
```

Not:

```text
20 integrations
      ↓
unclear product
```

---

# 47. CLI

Proposed interface:

```bash
glassbox trace
glassbox diff
glassbox investigate
glassbox reproduce
glassbox evidence
```

Example:

```bash
glassbox diff run_a run_b
```

```bash
glassbox investigate incident.json
```

```bash
glassbox reproduce evidence/INV-00042
```

```bash
glassbox evidence investigation.json
```

The CLI should remain usable without a hosted service.

---

# 48. SDK

Conceptual API:

```python
from glassbox import trace, diff, investigate

with trace("agent-run"):
    result = agent.run(task)

finding = diff(run_a, run_b)

investigation = investigate(
    finding,
    hypotheses=["retrieval", "tool-selection", "model-change"]
)

investigation.export_evidence("evidence/")
```

The exact API should be determined during implementation and user testing.

---

# 49. Repository Architecture

Target conceptual structure:

```text
glassbox/
├── trace/
│   ├── schema.py
│   ├── recorder.py
│   ├── streaming.py
│   └── adapters/
│
├── diff/
│   ├── behavioral.py
│   ├── trajectory.py
│   ├── retrieval.py
│   ├── tools.py
│   └── report.py
│
├── investigate/
│   ├── findings.py
│   ├── hypotheses.py
│   ├── controls.py
│   ├── interventions.py
│   └── reproduction.py
│
├── evidence/
│   ├── claims.py
│   ├── provenance.py
│   ├── integrity.py
│   └── packages.py
│
├── mechanistic/
│   ├── attribution.py
│   ├── circuits.py
│   └── interventions.py
│
└── integrations/
    ├── langgraph/
    ├── langchain/
    ├── openai/
    └── custom/
```

This is an architectural target, not a claim that all components already exist.

---

# 50. Assurance Documentation

Create:

```text
ASSURANCE.md
```

with a claim table:

| Claim | Status | Evidence |
|---|---|---|
| Trace protocol exists | IMPLEMENTED | tests |
| Attribution-profile comparison works under defined conditions | EXPERIMENTALLY_SUPPORTED | V6 |
| General AI-system diff works | PLANNED / IMPLEMENTATION-DEPENDENT | V7 |
| Autonomous investigation works | HYPOTHESIS / EXPERIMENTAL | V7 |
| Causal explanation is certified automatically | NOT_SUPPORTED unless experimentally established | evidence required |

The document should prevent marketing language from outrunning evidence.

---

# 51. Scientific Research Program

V7 should continue the research trajectory:

```text
Faithful explanation
        ↓
Explanation multiplicity
        ↓
Causal attribution over agentic trajectories
        ↓
Reproducible comparative measurement
        ↓
AI-system investigation
```

Underlying question:

> **When can we legitimately claim that an AI system did something for a particular reason?**

More general:

> **When is an explanation of AI behavior actually evidence?**

This research identity can unify the papers, Glassbox, and future work.

---

# 52. Research Questions for V7

Potential questions:

### RQ1
Can a common trace representation preserve investigation-relevant information across frameworks?

### RQ2
Can first meaningful behavioral divergence be localized reliably?

### RQ3
Can controlled experiments distinguish competing explanations?

### RQ4
Can intervention improve causal attribution of AI-system failures?

### RQ5
Can automated investigation agents choose informative experiments?

### RQ6
How should evidence strength be quantified?

### RQ7
Can independent researchers reproduce Glassbox findings from evidence packages?

### RQ8
How does investigation reliability change across stochastic systems?

### RQ9
How should causal responsibility be attributed across agent trajectories?

### RQ10
Which AI behaviors remain fundamentally under-identified from black-box traces?

---

# 53. Benchmark-Driven Development

Do not build V7 entirely feature-first.

Build:

```text
Failure
 ↓
Ground truth
 ↓
Measurement
 ↓
Method
 ↓
Benchmark
 ↓
User validation
 ↓
Implementation
```

Every major capability should have a benchmark case.

---

# 54. V7 Hard Gates

V7 should not be declared complete until the following are addressed.

## Gate 1 — Standardized trace

A common trace can represent multiple AI execution types.

## Gate 2 — Framework-independent diff

At least two substantially different execution stacks can be compared through the common representation.

## Gate 3 — Real failure localization

Glassbox identifies a meaningful divergence in a real or realistic AI failure.

## Gate 4 — Controlled intervention

Changing the suspected factor changes the measured behavior under a predefined experiment.

## Gate 5 — Reproduction

Another run or independent environment reproduces the finding.

## Gate 6 — External users

At least five independent users complete real investigations.

## Gate 7 — Cross-model

The investigation layer works across more than one model family/provider.

## Gate 8 — Agent

At least one real agent workflow can be traced and investigated.

## Gate 9 — Evidence package

A complete investigation can be exported and reconstructed.

## Gate 10 — Security/privacy

Sensitive trace handling has documented controls.

## Gate 11 — Investigation benchmark

Glassbox is evaluated against known-failure cases with quantitative metrics.

## Gate 12 — Repeat usage

External users return for multiple investigations.

---

# 55. What Not to Build Yet

Do not prioritize:

- giant dashboard,
- Kubernetes platform,
- distributed infrastructure,
- dozens of integrations before protocol stability,
- every model provider,
- every interpretability algorithm,
- full enterprise compliance suite,
- generic benchmark platform,
- autonomous scientist that claims unverified causality,
- massive UI before CLI/core workflow works.

The priority is:

> **Investigation quality.**

---

# 56. Product Roadmap

## Phase 0 — Freeze and Clean

- freeze V6,
- preserve scientific results,
- clarify README,
- publish claim boundaries,
- create ASSURANCE.md,
- define V7 protocol,
- define evidence schema.

## Phase 1 — Trace

Build:

- trace schema,
- Python recorder,
- local storage,
- replay metadata,
- one framework adapter,
- basic redaction.

## Phase 2 — Diff

Build:

- behavioral diff,
- trajectory diff,
- retrieval diff,
- tool diff,
- first-divergence analysis.

## Phase 3 — Investigation

Build:

- findings,
- hypotheses,
- controls,
- experiment definitions,
- statistical analysis,
- investigation reports.

## Phase 4 — Intervention + Reproduction

Build:

- intervention runner,
- controlled experiments,
- reproduction engine,
- evidence package.

## Phase 5 — Benchmark + External Users

Build:

- Investigation Benchmark,
- controlled failure corpus,
- five-user program,
- measurement dashboard for product metrics,
- integration expansion based on demand.

## Phase 6 — Infrastructure

Only after validation:

- cloud service,
- organization-wide traces,
- continuous diffing,
- collaboration,
- incident management,
- enterprise security,
- large-scale investigation.

---

# 57. Flagship V7 Experiment

The flagship experiment should demonstrate:

```text
same task
same agent
same model
same tools
same nominal configuration
```

but a controlled environmental change creates different behavior.

Glassbox should:

1. capture both traces,
2. compare trajectories,
3. identify first meaningful divergence,
4. propose hypotheses,
5. design a discriminating experiment,
6. intervene,
7. measure outcome,
8. reproduce,
9. produce evidence.

This becomes the equivalent of the “Hello World” for Glassbox investigation.

---

# 58. Commercial Thesis

The commercial thesis is not:

> “We built an AI dashboard.”

It is:

> **As AI systems become more autonomous and consequential, organizations need infrastructure for investigating unexpected AI behavior.**

Potential customers:

- AI engineering teams,
- agent developers,
- AI platform teams,
- evaluation teams,
- AI safety teams,
- AI auditors,
- regulated organizations,
- companies operating large AI applications,
- organizations integrating multiple models.

Potential economic value:

> reduce the time and uncertainty involved in diagnosing AI failures and behavioral changes.

---

# 59. Why Open Source Can Still Support a Large Company

Open source does not itself create a large business.

The potential business layers are:

```text
Open protocol
     ↓
Open-source adoption
     ↓
Trace ecosystem
     ↓
Investigation workflows
     ↓
Evidence infrastructure
     ↓
Organization-wide history
     ↓
Continuous monitoring/diffing
     ↓
Enterprise security
     ↓
Collaboration
```

The commercial moat must therefore extend beyond source code.

---

# 60. Potential Moats

Possible long-term defensibility:

### 1. Trace Protocol

A common representation used by multiple ecosystems.

### 2. Evidence Standard

A recognized way to package AI investigations.

### 3. Investigation Methodology

A tested workflow for diagnosing AI behavior.

### 4. Benchmark

A respected benchmark for AI investigation quality.

### 5. Evidence Corpus

A validated database of AI failures, explanations, interventions, and reproductions.

### 6. Ecosystem

Frameworks and tools emitting Glassbox-compatible traces.

### 7. Reputation

Trust created by transparent evidence and reproducibility.

### 8. Research

Scientific advances in AI behavioral investigation.

The strongest potential moat is not:

> “our code is hard to copy.”

It is:

> **protocol + ecosystem + methodology + evidence corpus + reputation + infrastructure.**

---

# 61. Glassbox Investigation Corpus

Long-term, each investigation can become a structured record:

```text
System
Incident
Trace
Observed divergence
Hypotheses
Controls
Intervention
Result
Reproduction
Evidence strength
```

With permission and privacy protections, aggregated investigations could support:

- benchmark construction,
- failure-pattern discovery,
- regression testing,
- research,
- model comparisons,
- investigation-agent training.

This must be privacy-preserving and governed carefully.

---

# 62. AI Incident Investigator

A future flagship product experience:

> **AI Incident Investigator**

Input:

```text
“Agent success rate dropped after deployment.”
```

Output:

```text
Investigating...

1. 14.2% increase in failed runs.
2. Retrieval distribution unchanged.
3. Tool-selection behavior changed.
4. First divergence occurs after planner response.
5. Candidate hypothesis: model update.
6. Controlled experiment proposed.
7. Intervention reproduced the regression.
8. Evidence package generated.

Status:
EXPERIMENTALLY_SUPPORTED
```

The exact causal language depends on the evidence.

---

# 63. Integration with Existing Ecosystem

Glassbox should complement rather than simply replace:

- tracing tools,
- evaluation tools,
- agent frameworks,
- observability systems,
- model providers.

The conceptual relationship is:

```text
Observability
    ↓
“What happened?”

Evaluation
    ↓
“Did it perform?”

Glassbox
    ↓
“Why did behavior differ,
what evidence supports the explanation,
and can we reproduce it?”
```

This distinction must be validated against real products and users rather than asserted as an absolute market boundary.

---

# 64. Competitive Discipline

Do not compete feature-for-feature with every existing AI tooling company.

Instead own a narrow conceptual job:

> **AI behavior investigation.**

A competitor may already have:

- traces,
- evaluations,
- prompts,
- dashboards,
- experiments.

Glassbox should differentiate through the investigation loop:

```text
trace
→ diff
→ divergence
→ hypothesis
→ experiment
→ intervention
→ reproduction
→ evidence
```

If a competitor provides some of these, that is not automatically a problem. The question is whether Glassbox makes the complete investigation loop substantially easier and more trustworthy.

---

# 65. User-Facing Finding

The core UI object should eventually look like:

```text
Finding #GB-0042

Question
Why did the agent fail?

Observation
Tool B was selected instead of Tool A.

First divergence
Planner step 4.

Hypothesis
Model version change altered tool-selection behavior.

Evidence
12/12 controlled trials reproduced the difference.

Intervention
Forced Tool A selection.

Result
Failure rate returned to baseline.

Reproduction
Independent environment: reproduced.

Status
EXPERIMENTALLY_SUPPORTED

Unresolved
Whether the change is caused by a specific internal
representation is not established.
```

This is much more useful than a dashboard full of charts.

---

# 66. Marketing Language Discipline

Use:

> Investigate why your AI system behaved differently.

Use:

> Reproduce AI behavior with evidence.

Use:

> Compare AI executions, test hypotheses, and produce evidence.

Avoid unsupported claims such as:

> Glassbox knows why AI thinks.

> Glassbox proves the model's true reasoning.

> Glassbox automatically discovers the real cause.

> Glassbox guarantees compliance.

The language should always match evidence.

---

# 67. README Direction

The README should eventually begin approximately:

```text
# Glassbox

## Investigate why your AI system behaved differently.

Glassbox records AI behavior, compares executions,
tests hypotheses, and produces reproducible evidence.

For accessible models, Glassbox can additionally investigate
internal computations.

AI proposes.
Evidence certifies.
```

Then immediately show:

```text
glassbox trace
glassbox diff
glassbox investigate
glassbox reproduce
glassbox evidence
```

Then show one spectacular investigation.

---

# 68. Documentation Architecture

Target:

```text
docs/
├── START_HERE.md
├── concepts/
│   ├── what-is-glassbox.md
│   ├── evidence-model.md
│   ├── trace.md
│   ├── diff.md
│   └── investigation.md
│
├── users/
│   ├── researcher.md
│   ├── ai-engineer.md
│   ├── eval-team.md
│   └── compliance.md
│
├── research/
│   ├── v6.md
│   ├── methodology.md
│   ├── preregistration.md
│   └── results.md
│
├── assurance/
│   ├── claims.md
│   ├── limitations.md
│   └── reproducibility.md
│
└── integrations/
    ├── langgraph.md
    ├── langchain.md
    ├── openai.md
    └── custom-agents.md
```

---

# 69. Compliance Position

EU AI Act and similar regulatory workflows should be treated as:

> **an evidence consumer/application of Glassbox**

not:

> **the identity of Glassbox.**

Glassbox can produce structured evidence useful for governance and documentation.

It must not claim that its mechanistic analysis is itself legally required unless supported by the applicable legal text and interpretation.

Compliance claims require separate legal validation.

---

# 70. Legacy Methodology Claims

Older Glassbox claims should be reviewed against V6's stricter scientific standards.

Avoid turning:

> “faithful circuit”

into an unconditional objective truth.

Prefer language such as:

> “candidate faithful circuit under the specified procedure and validation criteria.”

Likewise, any compliance gate or threshold should be clearly described as:

- a Glassbox methodological quality gate,
- an empirical threshold,
- or a legal requirement,

with these categories never conflated.

---

# 71. Versioning Philosophy

A new Glassbox version should exist because it introduces:

> **a scientifically or operationally validated capability or answers a new research question.**

Not because a calendar says it is time.

Conceptually:

```text
V1–V5
mechanistic interpretability / faithfulness

V6
scientific comparative measurement

V7
AI-system investigation infrastructure
```

Future V8 should only exist if V7 reveals a genuinely new research problem.

---

# 72. V7 Scientific Boundaries

V7 must distinguish:

### What is observed

from:

### What is inferred

from:

### What is experimentally supported

from:

### What is causally established.

This should be enforced in:

- schemas,
- reports,
- CLI,
- documentation,
- AI-generated hypotheses,
- marketing.

---

# 73. The One-Sentence Moat

A useful strategic statement:

> **Glassbox is building the evidence and investigation layer for AI systems: a common protocol for observing behavior, comparing executions, testing explanations, reproducing findings, and producing evidence.**

---

# 74. The One-Sentence User Value

> **When your AI behaves differently, Glassbox helps you find where it diverged, test why, and prove what you found.**

---

# 75. The One-Sentence Long-Term Vision

> **As AI becomes more autonomous, Glassbox becomes the scientific instrumentation and evidence layer organizations use to investigate AI behavior.**

---

# 76. Immediate Execution Priorities

Do these in order.

## Priority 1

Freeze V6.

Do not change the confirmatory result.

## Priority 2

Rewrite public positioning around:

> Investigate why your AI system behaved differently.

## Priority 3

Define and document the Trace Protocol.

## Priority 4

Implement the smallest useful `trace`.

## Priority 5

Implement a compelling `diff`.

## Priority 6

Create one real RAG investigation.

## Priority 7

Create one real agent investigation.

## Priority 8

Create one model-upgrade investigation.

## Priority 9

Create the Investigation Benchmark.

## Priority 10

Recruit five independent users.

## Priority 11

Measure repeated usage and time saved.

## Priority 12

Only then expand integrations and commercial infrastructure.

---

# 77. The Five-User Experiment

Give five people the same starting instruction:

> “Your AI system started behaving differently. Use Glassbox to investigate why.”

Do not teach them the intended workflow in advance.

Observe:

- where they get stuck,
- what they expect,
- what they misunderstand,
- whether they reach a finding,
- whether the evidence is understandable,
- whether they trust the result,
- whether they use Glassbox again.

This is one of the highest-value V7 experiments.

---

# 78. Success Ladder

Do not use “unicorn” as the technical success criterion.

Use:

```text
1 real external investigation
        ↓
5 users
        ↓
20 users
        ↓
100 users
        ↓
repeated investigations
        ↓
teams depend on it
        ↓
companies pay
        ↓
frameworks integrate
        ↓
evidence standard adopted
        ↓
investigation corpus grows
        ↓
Glassbox becomes infrastructure
```

A billion-dollar company, if it ever becomes one, would be an outcome of this progression—not the starting assumption.

---

# 79. The Actual Billion-Dollar Test

The strongest test is not:

> “Can Glassbox become a unicorn?”

It is:

> **“Can Glassbox become indispensable when an organization cannot explain an AI failure?”**

If organizations repeatedly experience:

```text
AI incident
→ Glassbox
→ diagnosis
→ experiment
→ evidence
→ resolution
```

then the business case becomes much stronger.

If users only install Glassbox to look at charts once, the thesis is weak.

---

# 80. Final Strategic Principles

### Principle 1
Do not build a generic AI dashboard.

### Principle 2
Do not build “support everything” before solving one painful investigation problem.

### Principle 3
Make `glassbox diff` exceptionally useful.

### Principle 4
Make evidence the product, not merely visualization.

### Principle 5
Use AI to generate hypotheses, never to certify unsupported explanations.

### Principle 6
Treat trajectories as first-class objects.

### Principle 7
Treat model internals as one optional evidence depth.

### Principle 8
Build the Trace Protocol before dozens of integrations.

### Principle 9
Build the Investigation Benchmark so the product can be measured.

### Principle 10
Build the Evidence Standard so investigations can become interoperable.

### Principle 11
Test with external users before building large infrastructure.

### Principle 12
Open source should maximize adoption; commercial infrastructure should monetize organizational complexity.

### Principle 13
Keep scientific claims narrower than marketing pressure.

### Principle 14
A failed experiment is useful if it falsifies a claim.

### Principle 15
Do not optimize for GitHub stars; optimize for repeated real investigations.

---

# 81. Final V7 Definition

V7 is complete only when a real AI system can move through:

```text
AI SYSTEM
   ↓
TRACE
   ↓
BEHAVIOR
   ↓
TRAJECTORY
   ↓
DIFF
   ↓
FIRST DIVERGENCE
   ↓
HYPOTHESES
   ↓
CONTROLLED EXPERIMENT
   ↓
INTERVENTION
   ↓
REPRODUCTION
   ↓
EVIDENCE
```

and when this workflow:

- works across multiple model types,
- works across multiple execution frameworks,
- handles stochastic behavior statistically,
- preserves provenance and integrity,
- protects sensitive traces,
- supports agents,
- supports RAG,
- supports model upgrades,
- incorporates mechanistic analysis where available,
- is measurable on a known-failure benchmark,
- is independently reproducible,
- and has been used repeatedly by external users.

That is the V7 target.

---

# 82. Final Positioning

**Glassbox**

> **Investigate why your AI system behaved differently.**

**Core workflow**

> Trace → Diff → Investigate → Experiment → Reproduce → Evidence

**Scientific principle**

> AI proposes. Evidence certifies.

**Infrastructure thesis**

> One investigation protocol across models, frameworks, agents, RAG systems, and accessible model internals.

**Research thesis**

> When is an explanation of AI behavior actually evidence?

**Long-term company thesis**

> Build the scientific instrumentation and evidence layer for increasingly autonomous AI systems.

---

# 83. Claude / Implementation Brief

When implementing V7, do not begin by building the entire architecture.

Proceed experimentally:

1. Read this plan.
2. Inspect the frozen V6 repository.
3. Do not alter V6 confirmatory results.
4. Create the V7 Trace Protocol specification.
5. Implement one minimal local trace.
6. Implement one minimal diff.
7. Build one controlled RAG failure.
8. Demonstrate localization.
9. Build one intervention.
10. Reproduce it.
11. Package evidence.
12. Turn the case into a benchmark.
13. Test the workflow with external users.
14. Only then expand the protocol and integrations.
15. Keep an explicit `ASSURANCE.md`.
16. Keep implementation status separate from scientific evidence.
17. Do not mark planned capabilities as implemented.
18. Do not use “causal” unless the experiment supports the stated causal claim.
19. Do not optimize architecture complexity before user validation.
20. Treat the Investigation Benchmark and Evidence Standard as first-class V7 deliverables.

The implementation should follow:

> **Build the smallest system that can produce one undeniable, reproducible AI investigation. Then generalize.**

---

# 84. V6 Preprint Track (executed during V7)

*Added 2026-10-04. Paper writing on the V6 result happens inside V7. It does not reopen V6:
the V6 confirmatory record stays frozen (Priority 1, §27).*

## 84.1 Scope

**Working title.** "Head Labels Are Not Intrinsically Comparable Across Training Runs: A
Permutation-Invariant Attribution-Profile Distance and a Preregistered Test on Pythia-410M"

**Central claim ceiling.** "Under a preregistered estimand, independently trained,
performance-matched Pythia-410M runs show greater attribution-profile distance (last-position
Taylor attribution on IOI, modulo within-layer head permutation) than late checkpoints within
the same training lineage (Δ = 1.50, run-level one-sided 95% lower bound 1.27, 7 runs)."

**Terminology rule.**
- Use "attribution-profile divergence" throughout.
- Never use "mechanistic divergence", "different mechanisms" or "different circuits" as
  conclusions.
- If those concepts must be discussed, state explicitly that the experiment does NOT
  establish them.

## 84.2 Authoritative sources (immutable)

- `experiments/v6/AMENDMENT3.md` (tag `v6-amendment3-lock`)
- `experiments/v6/PREREGISTRATION.md`
- `experiments/v6/RESULTS_confirmatory_v1.md` (commit e81dccc)
- `experiments/v6/runs/confirmatory_v1/record.json` (sha256 18d11300…)
- `experiments/v6/audits/`: head_identifiability.md, amendment3_gate_criteria.md (notes 1–6),
  orbit_baselines.md, seed_pilot_audit.md, and the gate JSON files
- `docs/CITATION-LEDGER.md` (append-only)
- `paper/v6_preprint/outline.md` (v0 outline, private, awaiting revision)

## 84.3 Step 1: Claim-to-evidence matrix (before any prose)

Before drafting prose, produce a claim-to-evidence matrix for Sections 1–9.

**Columns for every substantive claim:**
1. Claim ID
2. Proposed claim
3. Exact supporting evidence
4. Source file / record
5. Exact section, line, commit or record location
6. Status: exploratory / validation / preregistered / confirmatory
7. Direct support or contextual support only
8. Important limitation or confound
9. Recommended wording
10. Confidence: HIGH / MEDIUM / INSUFFICIENT

**Issues the matrix must cover:**

- **A. Identifiability.**
  - Functional head permutation on Pythia-70M: relative logit difference ≈ 4.5e-6,
    attribution equivariance Spearman ≈ 0.99989, positional D_M ≈ 1.04.
  - State exactly what this establishes and what it does not.
- **B. Exploratory relabelling pilot.**
  - 10,000 random permutations per pair, 410M pilot.
  - Seeds 1–5 overlap the confirmatory runs.
  - Classify as exploratory. Never present it as independent confirmatory evidence.
- **C. D\*.**
  - Exact definition, within-layer head permutation, Hungarian assignment.
  - Distinguish D\* from √D\*: the metric property applies to √D\*.
  - Verify the range [0, 4] and that √D\* is a metric on orbits.
  - Flag anything that needs a formal proof rather than an assertion.
  - D\* remains an attribution distance, not automatically a mechanistic one.
  - State the optimistic similarity bias.
- **D. Validation gates.**
  - **Gate 1:** give the target, the observed failure, the one permitted revision, the
    revised result and the consequence. Gate 1 failed and is closed; never present it as a
    success.
  - **Gates 2 and 7:** both used `pythia-410m-deduped`.
  - **Gate 5:** validated R = 4, 5, 6 and 10. R = 7–9 were not simulated.
  - **Gate 7 wording:** must not claim importance-weighting.
- **E. Preregistration and amendment.**
  - The original H1 used positional D_M and was superseded before confirmatory data existed;
    say why.
  - H2–H4 were not tested. Do not imply that all hypotheses were confirmed.
- **F. Confirmatory design.** Verify every number against the records:
  - 10 candidate runs × 3 checkpoints, pinned SHAs;
  - 200 attribution prompts and 1,000 matching prompts;
  - star-design eligibility and the exact inclusion rule;
  - B = 200, seeds 20260929/20260930;
  - decision rule: lower bound > 0.
- **G. Results.**
  - Exclusions: seed4 473/1000; seed7 bound 0.0242; seed8 incomplete (download failure).
  - R = 7, Δ = 1.4957, lower bound = 1.2692, 200/200 valid replicates →
    SUPPORTED_AT_RUN_LEVEL.
  - Descriptive ranges, sensitivity analyses and jackknife.
  - Never present the result as evidence for different mechanisms or circuits.
- **H. Limitations, at minimum:**
  - Gate 1 failure, so no pairwise confidence intervals;
  - R = 7 calibration assumed;
  - pilot overlap with the confirmatory runs;
  - Gates 2 and 7 on deduped checkpoints;
  - training steps confounded with lineage;
  - initialisation and data order coupled;
  - one model size, one task, one attribution procedure, last-position attribution only;
  - D\* not established to be importance-weighted;
  - attribution-profile divergence ≠ mechanistic or circuit divergence;
  - generalisation untested;
  - seed8 exclusion was operational, not a model property.
- **I. Reproducibility.**
  - Verify the tag and its commit, the runner hash, the protocol hash, the record SHA-256,
    the 29-checkpoint / 17.9 MB cache and the bitwise-reproduction claim.
  - Our own re-runs are NOT independent reproduction.
- **J. Citations.**
  - Use only VERIFIED rows in `docs/CITATION-LEDGER.md`, or perform a new verification.
  - Do not invent venues or rely on remembered ones.
  - Flag any citation that supports only a narrower claim than the prose suggests.

**Scientific rules:**
1. Separate established fact, exploratory observation, validation result, preregistered
   design, confirmatory result, interpretation and limitation.
2. Never upgrade evidence: attribution-profile divergence ≠ mechanistic divergence ≠
   circuit divergence ≠ causal mechanism difference.
3. Never hide methodological changes. Disclose the positional-D_M H1 and its replacement.
4. Never use the confirmatory result to validate D\* universally.
5. Never silently fix inconsistencies. Report each one and name the authoritative record.
6. No new experiment merely to make the paper cleaner.
7. The R = 7 calibration, if discussed, is labelled as a possible post-hoc analysis, never
   as preregistered.
8. No claim of external or independent reproduction unless another person reproduced the
   result.

**Deliverable for Step 1:**
- Part 1: claim-to-evidence matrix.
- Part 2: scientific risk flags.
- Part 3: required outline changes.
- Part 4: novelty test, answering:
  - What is novel?
  - What is already established?
  - What is an implementation contribution only?
  - What is the strongest defensible contribution statement?
  - What will a skeptical MI reviewer attack first?
- Part 5: GO / REVISE.

No section prose is written until Step 1 returns GO.

## 84.4 Known open items carried in from V6 (already identified 2026-09-30)

- arXiv:1511.07543 is RECALLED (title and authors not confirmed). It is excluded until
  refetched.
- These venues are not on the arXiv records, so they are cited as arXiv only: Wang 2022,
  Pythia, Git Re-Basin, Entezari, Chughtai.
- The v0 outline title differs from §84.1 ("not comparable" vs "not intrinsically
  comparable"). Use §84.1.
- The positional relabelling null came from the 410M pilot (`pilot_410m_6models`,
  `pilot_410m_seeds`), which covered seeds 1–5. Exploratory only.
- Gates 2 and 7 used `pythia-410m-deduped@143000`. The functional-twin experiment used
  `pythia-70m@143000`.

## 84.5 Sequencing within V7

**Owner decision (2026-10-05).** V7 is executed step by step first. The paper is written
afterwards and compiles the results of V6 and V7 together; Step 1 (§84.3) runs at that point.

*Superseded:* "The track runs in parallel with Priorities 2–12."
- It must not alter V6 artefacts.
- Any figure regenerated for the paper is computed from the committed cache and record, and
  is labelled as such.
