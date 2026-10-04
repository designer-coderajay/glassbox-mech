---
name: glassbox-oscal-export
description: "Export a Glassbox EU AI Act Annex IV evidence vault as a NIST OSCAL Assessment Results document (validated with compliance-trestle) so GRC and compliance-automation tools can import it. Use when the user asks for OSCAL, machine-readable compliance output, GRC tool integration, or interoperable audit evidence."
---

# Annex IV vault → OSCAL Assessment Results

## Setup

```bash
pip install "glassbox-mech-interp==4.5.1" "compliance-trestle==5.1.0"
```

## Input

An Annex IV vault JSON written by Glassbox (`vault.save_json(...)` or `build_annex_iv_vault(..., output_json=...)`).
If the user has no vault yet, build one first. The `glassbox-fairness-evidence` and
`glassbox-eval-evidence` skills show how to add evidence to it.

## Run

```bash
python scripts/vault_to_oscal.py annex-iv.json -o annex-iv.oscal-ar.json
```

The script writes the file, then reads it back through compliance-trestle's OSCAL model, so a
successful run means the document validates against OSCAL 1.1.2.

## How it maps

| Glassbox vault | OSCAL |
|---|---|
| vault | one `result` inside `assessment-results` |
| every entry | an `observation` (method TEST if it has a metric, else EXAMINE) |
| entry with pass/fail | a `finding` per article: `satisfied` / `not-satisfied` |
| article refs | `reviewed-controls` with IDs like `eu-ai-act-art-15.1` |

Article 11 (the duty to keep documentation) gets no finding, because a failed metric does not make
the documentation itself non-compliant.

## Caveats to tell the user

- To my knowledge there is no official OSCAL catalog for the EU AI Act. The control IDs are
  Glassbox-defined, so a GRC tool may need a mapping to its own control framework.
- `import-ap` points to a placeholder assessment-plan href. Pass `--assessment-plan-href` if the
  user has a real OSCAL assessment plan.
- OSCAL "satisfied" here reflects Glassbox's threshold check, not a legal conformity assessment.
