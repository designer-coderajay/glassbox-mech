# SIMULATED interviews, 2026-10-05: synthesis

> **SIMULATION, NOT EVIDENCE.** The five "interviewees" are fictional personas played by
> separate LLM agents.
>
> **Do not:**
> - count them toward the five-user decision rule;
> - cite them in ASSURANCE (row V5 stays `HYPOTHESIS`);
> - quote them anywhere as customer or user research.
>
> **Purpose:** rehearse the script, generate hypotheses, and find wording problems before
> real interviews.

## Setup

- **Interviewer:** UX Researcher persona (agency-agents), script v1.1, Q1–Q17.
- **Interviewees:** 5 separate subagents, one per persona, each answering independently.
  They were told to be realistic, not agreeable, and not to invent industry statistics.
- **Personas, written by the assistant (assumptions):**

| ID | Persona | Incident type **seeded in the brief** |
|---|---|---|
| S1 | ML engineer, B2B SaaS support RAG bot, LangSmith | answer quality drop after a provider model-version change |
| S2 | Solo founder-engineer, tool-using booking agent, print logs | wrong or duplicate tool calls after a prompt edit **and** a framework upgrade |
| S3 | MLOps engineer at a bank, self-hosted Langfuse, model-risk sign-off | outdated policy citations after re-chunking the index |
| S4 | Staff engineer, large consumer app, mature evals and APM (**skeptic**) | prompt template slip past evals for one locale |
| S5 | Research engineer and OSS RAG-library maintainer | "scores dropped after upgrading" user report |

Full answers are in `SIM_2026-10-05_TRANSCRIPTS.md`.

## What emerged

Each finding is marked **seeded** (it follows from what I put in the brief) or
**emergent** (the agent added it unprompted).

1. **Detection was the larger cost, not diagnosis (emergent, 5/5).**
   - Time to notice: S1 ~10–12 days, S3 ~8–9 days, S5 ~9 days, S4 ~36 h, S2 ~4–5 days.
   - Every case was noticed by a user or support, not by the team's own tooling. Evals
     passed because coverage of the affected slice was thin (S1, S3, S4, S5).
   - S4: *"you're solving the stage after detection, and detection was my problem."*
2. **Several changes at once, and the hard part was attributing the effect (largely
   emergent, 4–5/5).**
   - S2: prompt and framework (seeded).
   - S3: re-index plus a prompt tweak plus a library upgrade. The last two were added by
     the agent.
   - S4: own template change vs a provider model update (emergent).
   - S5: own splitter change plus the user's embedding-library upgrade (emergent).
   - Several built a one-factor-at-a-time grid by hand (S2, S5, S4).
3. **A step-level diff was wanted, but was "not enough" in a different way for each
   persona (emergent).**
   - S1 wanted to know *why* the model changed behaviour.
   - S2 needed the exact request payload the framework built.
   - S3 needed the **index state and chunk metadata** under the run. The run steps
     themselves were identical.
   - S5 wanted per-stage divergence: chunking vs retrieval vs judge.
   - S4 said their APM vendor already gives them this.
4. **Stochasticity (emergent, S2 and S5).**
   - Flaky behaviour needed 15–20 runs per configuration (S2). LLM-judge noise (S5).
   - Both asked how Glassbox separates a real divergence from sampling noise.
5. **The evidence bundle was valued only in the regulated setting (S3; emergent
   pattern).**
   - Even there it depends on whether model-risk accepts the format.
   - S1, S2, S4 and S5 called it overkill or irrelevant ("a table in a PR is enough").
6. **"Privacy-preserving" was confusing or conditional.**
   - S1: "we already send all of this to LangSmith".
   - S3: it must mean on-prem, nothing leaves the network, and they control what is
     hashed.
   - S2: wants it local or self-hosted, with no per-event pricing.
7. **Blind cases (Q17).**
   - 4 of 5 said yes with conditions: a synthetic rebuild, weeks of elapsed time, and the
     results shared back.
   - S3 needs approvals. S4 would build one only on their own time.
   - S5 was the most willing and framed it as a research collaboration (*two confounded
     causes, public data*).

## Decision rule applied to the simulation (illustration only, does not count)

| Signal | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| Cause guessed or had to be justified to someone | yes (mechanism unproven) | yes (~70%) | yes (model-risk) | no (clean A/B) | yes (moderate) |
| Built or paid for a workaround | yes | yes | yes | yes | yes |

Mechanically, this would read as a **strong** signal (4/5 and 5/5).

**Treat that as meaningless.** The personas were written to have incidents, and LLM
role-play tends to produce coherent, motivated stories. Real interviews may well look
different.

## Hypotheses to test in the real interviews

| # | Hypothesis | What to ask the real interviewees |
|---|---|---|
| H1 | Late detection costs more than slow diagnosis | "How could you have noticed earlier? What would have had to exist?" |
| H2 | The real pain is **attributing an effect across simultaneous changes**, not finding the changed step. This is what Glassbox's controlled intervention aims at (ASSURANCE V3, planned) | "How many things changed in that window? How did you separate them?" |
| H3 | Diffs must reach below steps (request payloads, index or config state) to be useful | "What did you need to see that wasn't in your traces?" (Q12 already) |
| H4 | Noise handling (repeated runs, statistics) is a requirement, not a nice-to-have | "Was the behaviour consistent when re-run?" |
| H5 | Evidence bundles matter only where someone must sign off (regulated) | Q8 already; segment the answers by regulated / unregulated |

## Implications for V7 (to confirm or reject with real interviews, not now)

- The detection gap (H1) is outside V7's current scope. It would be a new direction.
- H2 points at V7 plan §12–13 (controlled experiment and intervention engine) as the
  potential differentiator. It fits the V6 run-level statistics experience (H4).
- If real interviews confirm H3, the recorder should capture the framework request
  payload and data or index state, not only step inputs and outputs. The trace spec §10
  open questions already lists adapters.

## Script problems found (fixed in v1.2)

- Q16's description used "privacy-preserving trace" and "verifiable evidence bundle",
  which confused or put off 4/5. It is reworded to plain behaviour.
- No question probed detection or multiple simultaneous changes. Two short questions
  were added as Q9b and Q9c. The decision rule is unchanged.
