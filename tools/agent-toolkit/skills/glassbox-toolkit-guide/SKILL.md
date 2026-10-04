---
name: glassbox-toolkit-guide
description: "Router for the tools and skills used to build and sell Glassbox (mechanistic interpretability + EU AI Act Annex IV evidence). Use when the user asks which tool to use, what is installed, how to set the Glassbox toolkit up on a machine, or starts a Glassbox task that spans research, product, compliance evidence, or go-to-market."
---

# Glassbox toolkit guide

## Pick the right tool

| The user wants to... | Use |
|---|---|
| Find which attention heads drive a decision, with faithfulness scores (models up to ~12B) | Glassbox itself: `pip install glassbox-mech-interp`, `GlassboxV2(model).analyze(...)` |
| Do causal analysis on a model too big for local hardware | skill `glassbox-large-model-patching` (nnsight / NDIF) |
| Check whether a circuit-tracer / attribution-graph explanation is faithful | skill `glassbox-grade-attribution-graph` |
| Add accuracy / robustness benchmark evidence to Annex IV | skill `glassbox-eval-evidence` (lm-eval, Inspect AI) |
| Add bias / fairness evidence to Annex IV | skill `glassbox-fairness-evidence` (fairlearn) |
| Hand Annex IV evidence to a GRC / compliance tool | skill `glassbox-oscal-export` (NIST OSCAL) |
| Plan a new feature before coding | skill `glassbox-spec-feature` (spec-kit) |
| Research the web, Reddit, X, YouTube, GitHub, RSS (market, competitors, users) | skill `agent-reach` (if installed) |
| Build local-business lead lists | skill `google-maps-scraper` (needs Docker; check Google's terms and GDPR/anti-spam rules first) |
| Simulate how a market or group might react to a scenario | MiroFish (separate app; AGPL-3.0, keep its code out of Glassbox) |
| Wrap a desktop app as an agent-usable CLI | CLI-Anything (`cli-hub list`, `/cli-anything`) |
| A specialist role (sales engineer, pricing analyst, compliance auditor...) | agency-agents subagents (Claude Code only) |

## Evidence flow (all evidence skills share it)

Every evidence skill writes `{"entries": [...]}`, a list of Glassbox `VaultEntry` dicts. Combine them:

```python
import json
from glassbox import GlassboxV2
from glassbox.evidence_vault import AnnexIVEvidenceVault, VaultEntry

files = ["fairness-evidence.json", "eval-evidence.json"]
entries = [VaultEntry(**e) for f in files for e in json.load(open(f))["entries"]]
# plus vault_entries from patching.json / graded.json if present
vault = AnnexIVEvidenceVault(model_name="...", provider="...").build_vault(
    gb_result=result,  # optional GlassboxV2.analyze() output
    custom_entries=entries,
)
vault.save_json("annex-iv.json")
open("annex-iv.html", "w").write(vault.to_html())
```

Then `glassbox-oscal-export` can turn `annex-iv.json` into OSCAL.

## Set up on a machine with the Glassbox repo

```bash
git clone https://github.com/designer-coderajay/glassbox-mech && cd glassbox-mech
bash tools/agent-toolkit/install.sh            # agents, skills, CLIs for Claude Code
bash tools/agent-toolkit/install.sh --lab      # + Python "lab" env with all research libraries
```

The lab env lives at `.agent-tools/lab` (activate with `source .agent-tools/lab/bin/activate`) and
is pinned in `tools/agent-toolkit/lab.lock.txt`. See `tools/agent-toolkit/README.md` for licenses and caveats.

## Ground rules

- Glassbox output is evidence, not legal advice. Say so whenever compliance status is shown.
- Thresholds (fairness, accuracy) are the user's policy decisions. Ask before choosing them for the user.
- Prefer real runs over claims: if a model download or API key is missing, say what is blocked
  instead of estimating results.
