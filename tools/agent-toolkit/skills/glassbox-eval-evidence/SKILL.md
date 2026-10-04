---
name: glassbox-eval-evidence
description: "Run accuracy and robustness benchmarks on a language model with EleutherAI lm-evaluation-harness or UK AISI Inspect AI, then convert the results into Glassbox EU AI Act Annex IV evidence (Article 15 accuracy/robustness). Use when the user asks to benchmark or evaluate a model, measure accuracy/robustness/safety, run lm_eval or inspect eval, or fill the performance part of Annex IV."
---

# Evaluation evidence for Annex IV

## Setup

```bash
pip install "glassbox-mech-interp==4.5.1" "lm-eval==0.4.13" "inspect-ai==0.3.276"
```

Running open-weight models locally needs `torch` and `transformers` and Hugging Face access.
Inspect can also call hosted models (set the provider's API key, e.g. `ANTHROPIC_API_KEY`).

## Pick benchmarks with the user

Match benchmarks to the system's intended purpose. Generic benchmarks say little about a credit or
hiring model. Ask what the system does, then suggest:
- **General capability / accuracy:** `hellaswag`, `arc_easy`, `mmlu` (lm-eval)
- **Truthfulness:** `truthfulqa_mc2` (lm-eval)
- **Safety / misuse:** Inspect's evals catalogue (https://inspect.aisi.org.uk/evals/)
- **The user's own task:** a small Inspect task over their labelled examples. This is usually the
  most relevant evidence.

## Run

```bash
# lm-evaluation-harness (open weights)
lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag,arc_easy \
  --device cpu --limit 200 --output_path out/lm-eval

# Inspect AI (any provider)
pip install inspect-evals   # community eval catalogue
inspect eval inspect_evals/gpqa_diamond --model anthropic/claude-sonnet-5-5 --log-dir logs
```

Use `--limit` only for a quick smoke test, and say so in the report. Documented evidence should use
full runs.

## Convert to vault entries

```bash
python scripts/evals_to_vault.py out/lm-eval logs \
  --min "hellaswag:acc_norm=0.5" --min "arc_easy:acc=0.6" \
  -o eval-evidence.json
```

`--min TASK:METRIC=VALUE` sets a pass threshold. Without one, the entry is recorded with no
pass/fail verdict. Agree thresholds with the user and don't invent them. Inspect metrics are named
`scorer/metric` (for example `includes/accuracy`) when the scorer name differs from the task name.

## Add to the Annex IV vault

```python
import json
from glassbox.evidence_vault import AnnexIVEvidenceVault, VaultEntry

entries = [VaultEntry(**e) for e in json.load(open("eval-evidence.json"))["entries"]]
vault = AnnexIVEvidenceVault(model_name="gpt2", provider="Acme").build_vault(custom_entries=entries)
vault.save_json("annex-iv.json")
```

## Report

Give each metric with its standard error where available, the sample count, and whether the run was
limited. Benchmark scores describe the benchmark, not real-world deployment. Say that when you
present them as Annex IV evidence.
