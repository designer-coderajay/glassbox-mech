---
name: circuit-discovery
description: Step-by-step circuit discovery workflow using attribution patching. Activates when running or extending the circuit analysis pipeline in Glassbox.
origin: Glassbox
---

# Circuit Discovery Skill

The Glassbox circuit discovery pipeline, from raw prompt to identified minimal circuit.

## When to Activate

- Running a full circuit analysis on a new model or task
- Extending the patching engine with new hook types
- Debugging incorrect attribution scores
- Adding new circuit visualization methods

---

## The 5-Step Pipeline

### Step 1: Prepare Prompt Pair

Choose a clean and corrupted prompt where only the factual answer changes:

```python
# Good: minimal edit, same structure
clean     = "When Mary and John went to the store, John gave a drink to"
corrupted = "When John and Mary went to the store, Mary gave a drink to"

# The model should predict "Mary" (clean) vs "John" (corrupted)
# Only the name positions differ — everything else is structurally identical
```

Validation:
```python
# Verify token length is identical
assert len(model.to_tokens(clean)[0]) == len(model.to_tokens(corrupted)[0]), \
    "Token lengths must match for patching to be valid"
```

### Step 2: Run Attribution Patching

```python
from glassbox import GlassboxAnalyzer

analyzer = GlassboxAnalyzer(model_name="gpt2")
results = analyzer.analyze(
    prompt=clean,
    corrupted_prompt=corrupted,
    target_token=" Mary",
    distractor_token=" John"
)

# results.attribution_scores: (n_layers, n_heads) tensor
# results.logit_diff_clean: scalar
# results.logit_diff_corrupted: scalar
```

### Step 3: Identify the Circuit

Select heads above the attribution threshold (default: ≥5% of total positive attribution):

```python
total_positive = results.attribution_scores[results.attribution_scores > 0].sum()
threshold = 0.05 * total_positive

circuit = []
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        score = results.attribution_scores[layer, head].item()
        if score >= threshold.item():
            circuit.append((layer, head, score))

# Sort by attribution score descending
circuit.sort(key=lambda x: x[2], reverse=True)
print(f"Circuit: {[(l,h) for l,h,_ in circuit]}")
# Expected for IOI/GPT-2: [(9,9), (9,6), (10,0)]
```

### Step 4: Compute Faithfulness Metrics

```python
from glassbox.faithfulness import compute_sufficiency, compute_comprehensiveness

cited_heads = [(l, h) for l, h, _ in circuit]
head_effects = {(l, h): s for l, h, s in circuit}

sufficiency = compute_sufficiency(
    cited_heads=cited_heads,
    head_effects=head_effects,
    clean_logit_diff=results.logit_diff_clean
)

comprehensiveness = compute_comprehensiveness(
    model=analyzer.model,
    clean_prompt=clean,
    cited_heads=cited_heads,
    clean_logit_diff=results.logit_diff_clean,
    target_token=" Mary",
    distractor_token=" John"
)

f1 = 2 * (sufficiency * comprehensiveness) / (sufficiency + comprehensiveness)

print(f"Sufficiency:       {sufficiency:.2f}")       # Expect: ~1.00
print(f"Comprehensiveness: {comprehensiveness:.2f}")  # Expect: ~0.22
print(f"F1:                {f1:.2f}")                 # Expect: ~0.64
```

### Step 5: Generate Report

```python
report = analyzer.generate_report(
    results=results,
    circuit=cited_heads,
    sufficiency=sufficiency,
    comprehensiveness=comprehensiveness,
    f1=f1
)

# Export for GRC / Annex IV
import json
with open("circuit_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

---

## Interpreting Results

### What Good Results Look Like

| Metric | Target | What It Means |
|--------|--------|---------------|
| Sufficiency | ≥ 0.80 | The circuit explains the prediction |
| Comprehensiveness | ≥ 0.50 | Ablating the circuit hurts the prediction |
| F1 | ≥ 0.65 | Balanced sufficiency + comprehensiveness |

### Why Low Comprehensiveness Is Expected

Glassbox finds sufficiency=1.00 but comprehensiveness=0.22. This is not a bug — it reflects the transformer's architecture:
- Transformers learn distributed, redundant representations
- Multiple backup circuits implement the same behavior
- This is known as **superposition** (Elhage et al. 2022)

For compliance purposes: document this finding. It means the model has multiple independent pathways implementing the same behavior — which is actually a robustness property.

---

## Extending the Pipeline

### Adding New Hook Types

```python
# Current: only attention heads (hook_z)
# To add: MLP neurons (hook_post)

def compute_mlp_attribution(
    model: HookedTransformer,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    layer: int
) -> torch.Tensor:
    """
    Attribution for MLP neurons at a given layer.
    Returns: (d_mlp,) tensor of attribution scores
    """
    act_diff = (
        clean_cache[f'blocks.{layer}.mlp.hook_post']
        - corrupted_cache[f'blocks.{layer}.mlp.hook_post']
    )
    grad = clean_cache[f'blocks.{layer}.mlp.hook_post'].grad
    return (act_diff * grad).sum(dim=(0, 1))  # sum over batch, seq
```

### Adding New Tasks

To extend beyond IOI:
1. Define `clean_prompt` and `corrupted_prompt` with identical token lengths
2. Define `target_token` and `distractor_token`
3. Verify logit_diff_clean > 0 (model gets it right on clean)
4. Run the pipeline — everything else is automatic

---

## Debugging Circuit Results

```python
# Sanity check 1: logit diff should be positive (model correct on clean)
assert results.logit_diff_clean > 0, f"Model gets clean prompt wrong: LD={results.logit_diff_clean}"

# Sanity check 2: attribution scores should sum to approximately LD_clean
total_attribution = results.attribution_scores.sum().item()
print(f"Total attribution: {total_attribution:.3f}, LD_clean: {results.logit_diff_clean:.3f}")
# These won't be exactly equal (gradient approximation), but should be similar in magnitude

# Sanity check 3: at least 1 head should have positive attribution
assert (results.attribution_scores > 0).any(), "No heads have positive attribution"

# Sanity check 4: hooks cleaned up
assert len(model.hook_dict) == 0 or all(
    len(hooks) == 0 for hooks in model.hook_dict.values()
), "Hooks not cleaned up after analysis"
```
