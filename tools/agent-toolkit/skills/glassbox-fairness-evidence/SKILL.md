---
name: glassbox-fairness-evidence
description: "Measure group fairness (demographic parity, equalized odds, per-group accuracy/TPR/FPR) of a model's decisions with fairlearn and turn it into Glassbox EU AI Act Annex IV evidence. Use when the user has predictions plus a sensitive attribute (gender, age, ethnicity, region...) and asks about bias, fairness, discrimination, Article 10 data governance, or the bias part of Annex IV."
---

# Fairness evidence for Annex IV

Turns a CSV of decisions into fairness metrics and Glassbox vault entries.

## Setup (once per environment)

```bash
pip install "glassbox-mech-interp==4.5.1" "fairlearn==0.14.0" "pandas" "scikit-learn"
```

## Inputs to get from the user

A CSV with one row per decision:
- the true outcome column (`--y-true`)
- the model's decision column (`--y-pred`)
- one or more sensitive-attribute columns (`--sensitive`, repeatable)

Labels must be binary. If they are strings like `approved`/`denied`, pass `--positive approved`.
Ask before assuming which column is which. Never invent a sensitive attribute.

## Run

```bash
python scripts/fairness_evidence.py decisions.csv \
  --y-true actual --y-pred decision --sensitive gender --sensitive age_band \
  -o fairness-evidence.json
```

Defaults: demographic-parity ratio ≥ 0.8 and equalized-odds difference ≤ 0.1. Both are
screening conventions (0.8 is the US "four-fifths" heuristic). Neither is an EU legal threshold.
Change them with `--min-dp-ratio` / `--max-eo-diff` if the user has their own policy. Groups below
`--min-group-size` (default 30) are flagged as statistically unreliable.

## Add to the Annex IV vault

```python
import json
from glassbox.evidence_vault import AnnexIVEvidenceVault, VaultEntry

entries = [VaultEntry(**e) for e in json.load(open("fairness-evidence.json"))["entries"]]
vault = AnnexIVEvidenceVault(model_name="credit-model-v3", provider="Acme").build_vault(
    gb_result=result,          # optional: output of GlassboxV2.analyze()
    custom_entries=entries,
)
vault.save_json("annex-iv.json")
open("annex-iv.html", "w").write(vault.to_html())
```

## Report to the user

- Show each metric with PASS/FLAG, the per-group table from `raw.by_group`, and any small groups.
- Say plainly that a FLAG is a signal to investigate, not a legal finding, and a PASS does not prove
  the system is non-discriminatory (only the measured attributes and data were checked).
- Mitigation options, if asked: reweighting or threshold optimisation (`fairlearn.postprocessing.ThresholdOptimizer`),
  or AIF360's pre-processing algorithms. Re-run this skill after any change.
