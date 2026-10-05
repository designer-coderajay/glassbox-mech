# Glassbox

## Agent toolkit

Third-party agents, skills and CLIs for building Glassbox (agency-agents, agent-scripts,
CLI-Anything, Agent-Reach, google-maps-scraper-kit, MiroFish) are pinned in
`tools/agent-toolkit/`. They are not installed by default: run
`bash tools/agent-toolkit/install.sh` when a task needs them, and read
`tools/agent-toolkit/README.md` for licenses and caveats. MiroFish is AGPL-3.0; never copy
its code into this repo.

`--lab` adds a research virtualenv (`.agent-tools/lab`: nnsight, circuit-tracer, inspect-ai,
lm-eval, fairlearn, aif360, compliance-trestle) pinned in `tools/agent-toolkit/lab.lock.txt`.
Glassbox-specific skills (Annex IV evidence, OSCAL export, large-model patching, attribution-graph
grading, spec-kit, MiroFish simulation) live in `tools/agent-toolkit/skills/`; `package-skills.py` zips them for Cowork.
