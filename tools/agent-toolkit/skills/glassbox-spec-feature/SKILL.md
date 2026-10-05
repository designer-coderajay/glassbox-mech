---
name: glassbox-spec-feature
description: "Plan a new Glassbox feature spec-first with GitHub spec-kit: write the spec, clarify it, produce a technical plan and task list, then implement against it. Use when the user wants to design, scope or plan a new Glassbox capability, integration (nnsight, circuit-tracer, OSCAL, evals), or a larger change before coding."
---

# Spec-first feature work with spec-kit

spec-kit (github/spec-kit) turns a feature idea into `spec.md` → `plan.md` → `tasks.md` before
any code is written. It suits Glassbox, where claims must stay traceable to evidence.

## Setup (once per machine)

```bash
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
specify check
```

## Enable it in the Glassbox repo (once, and only if the user agrees)

```bash
cd glassbox-mech
specify init --here --integration claude --script sh
```

This adds `.specify/` (templates, scripts, `memory/constitution.md`) and `/speckit-*` skills under
`.claude/skills/`. Show the user the diff and let them decide whether to commit it.

## Glassbox constitution (suggested starting points for `.specify/memory/constitution.md`)

1. Every metric shown to a user is computed, never asserted. Faithfulness is measured by intervention.
2. When the model cannot support a claim, report that (grade C/D, NON-COMPLIANT); never smooth it over.
3. Compliance output is evidence and documentation, not legal advice or a conformity declaration.
4. Respect the dependency ceilings in `pyproject.toml` (numpy<2, torch<2.11, transformer_lens<3) unless the spec changes them deliberately.
5. Every new capability ships with tests that run offline, and with an entry in `CHANGELOG.md`.
6. Third-party code: check its license before adding it (MiroFish is AGPL-3.0, so keep it out).

## Workflow

| Step | Command | Output |
|---|---|---|
| Principles | `/speckit-constitution` | `.specify/memory/constitution.md` |
| What and why | `/speckit-specify <feature description>` | `specs/<n>-<name>/spec.md` |
| De-risk (optional) | `/speckit-clarify` | answered open questions in the spec |
| How | `/speckit-plan <tech constraints>` | `plan.md`, research, data model |
| Break down | `/speckit-tasks` | `tasks.md` |
| Check consistency (optional) | `/speckit-analyze` | cross-artifact report |
| Build | `/speckit-implement` | code + tests, task by task |

Keep the spec in user terms: who needs this, what decision it supports, and what evidence it produces.
Technology choices belong in the plan.
