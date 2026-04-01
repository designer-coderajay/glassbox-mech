# /audit — Full Model Audit

Run a complete Glassbox audit: circuit discovery + faithfulness metrics + compliance report.

## Usage

```
/audit [model_name] [prompt] [--output <path>]
```

**Default**: `gpt2` model, IOI benchmark prompt from paper.

## Steps

1. **Load model**: `HookedTransformer.from_pretrained(model_name)` — validate against allowlist first
2. **Run attribution patching**: 3 forward passes, compute `(n_layers, n_heads)` attribution scores
3. **Identify circuit**: select heads above 5% threshold, sorted by score descending
4. **Compute faithfulness**: sufficiency, comprehensiveness, F1
5. **Compute correlation**: confidence vs faithfulness on a batch of prompts (or report stored r=0.009 for GPT-2 IOI)
6. **Generate Annex IV**: all 9 sections populated, grade from F1
7. **Export JSON**: GRC-ready schema, timestamped, saved to `./outputs/audit_<timestamp>.json`
8. **Print summary**: key metrics in terminal, path to full report

## Expected Output (GPT-2, IOI task)

```
=== GLASSBOX AUDIT REPORT ===
Model: gpt2
Task: Indirect Object Identification (IOI)
Analysis time: 1.2s (3 forward passes)

Circuit identified: [(9,9), (9,6), (10,0)]

Faithfulness Metrics:
  Sufficiency:       1.00
  Comprehensiveness: 0.22
  F1:                0.64

Confidence-Faithfulness Correlation: r = 0.009

Compliance Grade: B
EU AI Act Annex IV: 9/9 sections complete

Full report: ./outputs/audit_2026-04-01T10:00:00Z.json
```

## Error Handling

- Model not in allowlist → print allowlist, exit cleanly
- Prompt too long (> 512 tokens) → truncate with warning
- CUDA OOM → fall back to CPU with warning
- JSON serialization error → print offending field, exit with stack trace

## Code to Run

```python
from glassbox import GlassboxAnalyzer

analyzer = GlassboxAnalyzer(model_name=model_name)
results = analyzer.analyze(
    prompt=prompt,
    corrupted_prompt=corrupted_prompt,
    target_token=target,
    distractor_token=distractor
)
report = analyzer.generate_report(results)
report.save(output_path)
report.print_summary()
```
