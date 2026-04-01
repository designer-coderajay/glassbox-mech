# /circuit — Run Circuit Discovery on a Prompt

Run attribution patching on a single prompt and identify the minimal circuit responsible for the prediction.

## Usage

```
/circuit [--model gpt2] [--prompt "..."] [--corrupted "..."] [--target " Mary"] [--distractor " John"]
```

If `--corrupted` not provided, Glassbox will generate a corrupted version automatically (name-swap for IOI tasks).

## Steps

1. **Validate**: model allowlist, token lengths match, clean logit diff > 0
2. **Run analysis**: 3 forward passes, attribution scores per head
3. **Identify circuit**: heads above 5% threshold, sorted descending
4. **Print circuit**: (layer, head) pairs with attribution scores
5. **Visualize**: ASCII heatmap of attribution scores by layer/head
6. **Compute faithfulness**: sufficiency, comprehensiveness, F1
7. **Optionally save**: `--output <path>` for JSON export

## Expected Output (GPT-2 IOI)

```
=== CIRCUIT DISCOVERY ===
Model: gpt2  |  Prompt: "When Mary and John went..."
3 forward passes  |  1.2s

Top Circuit Heads:
  Layer 9, Head 9:  0.584  ████████████████
  Layer 9, Head 6:  0.211  ██████
  Layer 10, Head 0: 0.208  ██████

Attribution Heatmap (layer × head):
     H0   H1   H2   H3   H4   H5   H6   H7   H8   H9   H10  H11
L9 [ .02  .01  .00  .03  .01  .00  .21  .01  .02  .58  .01  .02 ]
L10[ .21  .01  .00  .02  .01  .00  .01  .01  .01  .02  .01  .01 ]

Faithfulness:
  Sufficiency:       1.00  ✓
  Comprehensiveness: 0.22  (distributed backup mechanisms present)
  F1:                0.64
```

## Debugging Circuit Results

If the circuit seems wrong:
```bash
# Check logit diff is positive (model gets it right on clean)
# If negative: model is getting the clean prompt wrong — try a different prompt

# Check top-k heads
python -c "
from glassbox import GlassboxAnalyzer
a = GlassboxAnalyzer('gpt2')
r = a.analyze(prompt='...', corrupted_prompt='...')
top = r.top_heads(k=10)
print(top)
"
```

Invoke `pytorch-build-resolver` if you get shape errors or hook failures.
