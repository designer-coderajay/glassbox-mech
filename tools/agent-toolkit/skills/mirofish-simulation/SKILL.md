---
name: mirofish-simulation
description: "Run a MiroFish multi-agent social simulation (666ghj/MiroFish) to rehearse how a market, community or stakeholder group might react to a scenario, e.g. how compliance officers, ML engineers, auditors and regulators respond to an EU AI Act enforcement event or a Glassbox launch message. Use when the user mentions MiroFish, swarm/multi-agent simulation, 'simulate how people react', scenario rehearsal, or wants to stress-test positioning before real customer interviews."
---

# MiroFish scenario simulation

MiroFish turns seed documents into a knowledge graph, generates hundreds of LLM-driven personas,
lets them interact on simulated Twitter- and Reddit-style platforms, and writes a report. You can
then interview individual agents. For Glassbox it is a **hypothesis generator**: a cheap way to
see which objections and narratives might come up before talking to real buyers.

## Ground rules (tell the user up front)

1. **License: AGPL-3.0.** Run MiroFish as a separate app. Never copy its code into Glassbox,
   import it from Glassbox, or run it inside a hosted Glassbox service. Its *outputs* (reports you
   generate) are yours to use.
2. **Not evidence.** Simulated agents are LLM outputs. Never present simulation results as
   customer data, market research, survey results or statistics, in pitch decks or anywhere
   else. Phrase findings as "the simulation suggested X; to validate with N real interviews".
3. **It costs money.** Every round calls the LLM for many agents. Start below 40 rounds (MiroFish's
   own advice) and cap with `max_rounds`.

## Setup (once per machine)

Needs Node.js 18+, Python 3.11 with `uv`, an OpenAI-compatible LLM API key, and a Zep Cloud key
(memory graph; free tier at https://app.getzep.com/).

```bash
# Inside the Glassbox repo (pins the reviewed commit):
bash tools/agent-toolkit/install.sh --only MiroFish --setup-mirofish
cd .agent-tools/MiroFish

# Or standalone:
git clone https://github.com/666ghj/MiroFish && cd MiroFish && npm run setup:all && cp .env.example .env
```

Fill in `.env` (the template's comments are in Chinese; the keys are):

| Key | Meaning |
|---|---|
| `LLM_API_KEY` | API key for any OpenAI-SDK-compatible endpoint |
| `LLM_BASE_URL` | that endpoint's base URL (the template defaults to Alibaba DashScope) |
| `LLM_MODEL_NAME` | model name at that endpoint (template default `qwen-plus`) |
| `ZEP_API_KEY` | Zep Cloud key |
| `LLM_BOOST_*` | optional faster model; delete these lines entirely if unused |

Never print or commit `.env`. Then start it:

```bash
npm run dev            # frontend http://localhost:3000, backend API http://localhost:5001
# or: docker compose up -d   (same ports; image ghcr.io/666ghj/mirofish:latest)
```

The UI has English and Chinese locales.

## Step 1: write the seed material

Seed quality decides everything. Copy `templates/glassbox-seed.md`, fill it in with **facts the user
confirms**, and keep claims sourced. Typical Glassbox scenarios:

- "EU AI Act high-risk obligations take effect; a bank's model-risk team must produce Annex IV
  documentation. How do compliance, ML and audit staff react to a tool that measures explanation
  faithfulness?"
- "A competitor announces SHAP-based 'AI Act compliance'. How does the interpretability community react?"
- "Glassbox posts its r = 0.009 confidence-vs-faithfulness finding. Which objections appear?"

Upload 1-3 documents (PDF/MD/TXT): the seed brief plus, optionally, a real public article. Avoid
uploading confidential customer material: it is sent to the LLM and Zep providers.

## Step 2: run it (UI, recommended)

In http://localhost:3000: upload seeds and write the **simulation requirement** (one paragraph:
who, what event, what question you want answered) → review the generated ontology → build the
graph → review generated personas (remove any that impersonate real, named individuals) → set a
small round count → run → generate the report.

## Step 3: pull results through the API (optional)

```bash
API=http://localhost:5001
curl -s "$API/api/simulation/list"                                   # find the simulation_id
curl -s "$API/api/report/list?simulation_id=SIM_ID"                  # find the report_id
curl -s -o report.md "$API/api/report/REPORT_ID/download"            # Markdown report

# Interview one agent (the simulation must have finished its loop and be waiting for commands)
curl -s -X POST "$API/api/simulation/interview" -H 'Content-Type: application/json' \
  -d '{"simulation_id":"SIM_ID","agent_id":0,"prompt":"Would you trust a faithfulness score in an audit? Why or why not?","platform":"reddit"}'
```

Other useful read-only endpoints: `GET /api/simulation/<id>/posts`, `/comments`, `/timeline`,
`/agent-stats`, and `GET /api/report/<id>/sections`.

## Step 4: report back to the user

Structure the summary as:
1. **Scenario and setup:** seeds, requirement, rounds, model used, date.
2. **Recurring objections and arguments**, each with 1-2 quoted agent posts, labelled as simulated.
3. **Narratives that spread vs. died**, from the timeline.
4. **Hypotheses to test with real people:** concrete interview questions for real compliance
   officers or ML leads.
5. **Limits:** LLM persona bias, seed bias, small round count, no ground truth.

Then suggest the next real-world step (for example, five discovery calls using the hypotheses).

## Notes

- Verified from the pinned source (commit 7657031): the endpoints above, `max_rounds` on
  `/api/simulation/start`, and the `.env` keys. MiroFish changes quickly; if an endpoint 404s,
  check `backend/app/api/*.py` in the user's copy.
- This skill was written without a live run (no LLM or Zep keys were available where it was
  built). Treat the first run as a check and fix the instructions if anything differs.
