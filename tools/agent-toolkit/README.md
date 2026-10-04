# Agent toolkit

Third-party agents, skills and CLIs we use while building Glassbox. None of this
ships in the Glassbox package. `install.sh` clones each repo at the commit pinned in
`repos.lock` into `.agent-tools/` (git-ignored) and wires the Claude Code parts into
`~/.claude` (or `$CLAUDE_CONFIG_DIR`).

```bash
bash tools/agent-toolkit/install.sh                       # everything
bash tools/agent-toolkit/install.sh --only Agent-Reach    # one tool
bash tools/agent-toolkit/install.sh --agency-divisions engineering,product,marketing,sales,strategy,testing
bash tools/agent-toolkit/install.sh --only MiroFish --setup-mirofish
bash tools/agent-toolkit/install.sh --lab                 # + Python research env (~8 GB with CUDA torch)
python3 tools/agent-toolkit/package-skills.py             # zip skills for Cowork / claude.ai
```

Cloud sessions are ephemeral: `~/.claude` and `.agent-tools/` disappear with the
container, so re-run the script in each new session that needs the tools. Start a new session
afterwards so Claude Code loads the new agents and skills.

## What gets installed

| Repo | License | Lands as | Use it for in Glassbox |
|---|---|---|---|
| [msitarzewski/agency-agents](https://github.com/msitarzewski/agency-agents) | MIT | ~280 subagents in `~/.claude/agents/` | Role specialists: engineering, product, marketing, sales, research, testing, security |
| [steipete/agent-scripts](https://github.com/steipete/agent-scripts) | MIT | 53 skills symlinked into `~/.claude/skills/` | `create-cli`, `github-deep-review`, `github-project-triage`, `oracle`, `markdown-converter`, `frontend-design` |
| [HKUDS/CLI-Anything](https://github.com/HKUDS/CLI-Anything) | Apache-2.0 | `cli-anything` Claude Code plugin + `cli-hub` CLI | Generating agent-usable CLIs for tools; `cli-hub list` / `cli-hub install <name>` |
| [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) | MIT | `agent-reach` CLI + `agent-reach` skill | Reading web pages, RSS, GitHub, YouTube, Reddit, X for market and user research |
| [Mahanaicoach/google-maps-scraper-kit](https://github.com/Mahanaicoach/google-maps-scraper-kit) | MIT | `google-maps-scraper` skill + `/scrape*` commands | Local-business lead lists (needs Docker, see below) |
| [666ghj/MiroFish](https://github.com/666ghj/MiroFish) | **AGPL-3.0** | Clone only (deps with `--setup-mirofish`) | Multi-agent scenario simulation, run as a separate app |

Also installed:

| Tool | License | Lands as |
|---|---|---|
| [github/spec-kit](https://github.com/github/spec-kit) | MIT | `specify` CLI (pinned commit in `install.sh`) |
| Glassbox skills (this folder, `skills/`) | same as this repo | 7 skills symlinked into `~/.claude/skills/` |

## Lab environment (`--lab`)

A separate virtualenv at `.agent-tools/lab` with Glassbox (editable) plus the research
libraries, pinned in `lab.lock.txt` (compiled from `lab.in` with `uv pip compile`). It keeps
Glassbox's own ceilings (numpy<2, torch<2.11, transformer_lens<3).

| Library | Version | Why |
|---|---|---|
| [nnsight](https://github.com/ndif-team/nnsight) | 0.7.0 | Interventions on any HF model, locally or remotely on NDIF |
| [circuit-tracer](https://github.com/safety-research/circuit-tracer) | 0.5.0 (PyPI; GitHub main is newer) | Transcoder attribution graphs |
| [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) | 0.3.276 | UK AISI evaluation framework |
| [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) | 0.4.13 | Standard benchmarks |
| [fairlearn](https://github.com/fairlearn/fairlearn) / [aif360](https://github.com/Trusted-AI/AIF360) | 0.14.0 / 0.6.1 | Fairness metrics and mitigation |
| [compliance-trestle](https://github.com/oscal-compass/compliance-trestle) | 5.1.0 | NIST OSCAL models and validation |

In the lab env, Glassbox's own test suite gives 942 passed, 0 failed. 124 more tests error
in a cloud session only because their setup downloads models from Hugging Face, which that
session's network policy blocked.

## Glassbox skills

| Skill | Does | Tested how |
|---|---|---|
| `glassbox-toolkit-guide` | Routes a task to the right tool or skill | n/a (instructions) |
| `glassbox-fairness-evidence` | fairlearn metrics → Annex IV vault entries | synthetic predictions |
| `glassbox-eval-evidence` | lm-eval / Inspect results → vault entries | lm-eval result file and a schema-valid Inspect log |
| `glassbox-oscal-export` | vault JSON → OSCAL Assessment Results | output re-read through trestle's OSCAL model |
| `glassbox-large-model-patching` | layer × position activation patching via nnsight / NDIF | tiny random GPT-2 and Llama: final-layer last-position patch restores 1.0, pre-diff positions 0 |
| `glassbox-grade-attribution-graph` | grades circuit-tracer graphs with suff / comp / F1 | tiny random model + random transcoders; not yet run on Gemma/Llama |
| `glassbox-spec-feature` | spec-kit workflow with a Glassbox constitution | `specify init` in a scratch project |

Every evidence skill writes `{"entries": [...]}` in Glassbox's `VaultEntry` format, so the
outputs merge with `AnnexIVEvidenceVault(...).build_vault(custom_entries=...)`.

### Using the skills in Cowork or the Claude apps

`package-skills.py` writes one zip per skill to `dist/cowork-skills/` (git-ignored), with
the folder layout claude.ai's skill upload uses. It also packages the Agent-Reach and
google-maps-scraper skills with their MIT licenses. Upload the zips under your Claude
settings (Capabilities → Skills at the time of writing; see
[Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
if the menu has moved). Each skill's Setup section lists the `pip install` it needs, because
Cowork runs in its own environment, not in this container.

## Caveats

- **MiroFish is AGPL-3.0.** Glassbox is MIT plus a commercial license. Keep MiroFish
  as a separate internal tool: do not copy its code into this repo, import it from
  Glassbox, or run it as part of a hosted Glassbox service. I am not a lawyer; get a
  proper review before mixing it with anything we ship.
- **Agency-agents costs context.** Every installed subagent's description is listed
  in each session, about 66 KB of text for all of them (my rough estimate: ~15k
  tokens). Use `--agency-divisions` to install only the teams you need.
- **agent-scripts is one person's setup.** Many skills target macOS or steipete's own
  services (Sonos, iMessage, Things, his Mac fleet) and will not work here. 15
  entries are links into his other repos and are skipped. `codex-first` is never
  linked because it reroutes implementation work to an external Codex CLI. His
  `AGENTS.MD` rules are not applied to this repo.
- **Agent-Reach** works without setup for web pages and RSS. Other platforms need
  extra CLIs or your own login cookies: `agent-reach doctor` lists what is missing.
  `--reach-channels twitter,reddit` runs its installer, which adds global pipx/npm
  packages. Logged-in scraping uses your own accounts and is subject to each
  platform's terms.
- **google-maps-scraper-kit** needs a running Docker daemon for its engine
  (`docker compose up -d` in `.agent-tools/google-maps-scraper-kit`). Cloud sessions
  have the Docker client but no daemon, so it only runs on a local machine. Its scripts
  use paths relative to the kit folder. Before using scraped data for outreach, check
  Google's terms and the data-protection and anti-spam rules where the recipients are
  (for example GDPR in the EU); the kit's own skill lists some guardrails.
- The kit's `.claude/settings.json` pre-approves Docker and curl commands. The
  installer does not merge it into any settings.

## Updating a pin

Review the upstream changes since the pinned commit, edit the SHA in `repos.lock`,
and re-run `install.sh`.
