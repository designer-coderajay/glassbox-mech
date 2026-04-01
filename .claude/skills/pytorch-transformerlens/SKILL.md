---
name: pytorch-transformerlens
description: PyTorch + TransformerLens patterns for Glassbox. Activates when writing code that uses TransformerLens hooks, model loading, caching, or any PyTorch tensor operations in the interpretability pipeline.
origin: Glassbox
---

# PyTorch + TransformerLens Patterns

Correct, idiomatic patterns for TransformerLens and PyTorch in the Glassbox codebase.

## When to Activate

- Loading models with `HookedTransformer`
- Adding, using, or removing hooks
- Working with `ActivationCache`
- Writing tensor operations for attributions
- Debugging device or shape errors

---

## Model Loading

```python
from transformer_lens import HookedTransformer

# Correct: use from_pretrained, always specify device
model = HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed=True,     # Useful for logit lens
    center_writing_weights=True,
    fold_ln=True,            # Folds layer norm for cleaner analysis
    refactor_factored_attn_matrices=True
)
model.eval()  # Always eval mode for interpretability

# Get device — never hardcode 'cuda' or 'cpu'
device = model.cfg.device

# Move all tensors to this device
tokens = model.to_tokens(prompt).to(device)
```

---

## Running with Cache

```python
# Always use run_with_cache for interpretability work
# Never cache = {} and populate manually — use the API

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

# Access cache by string key
attn_pattern = cache['pattern', 9, 'attn']   # (batch, n_heads, seq, seq)
resid_post   = cache['resid_post', 9]         # (batch, seq, d_model)
hook_z       = cache[f'blocks.9.attn.hook_z'] # (batch, seq, n_heads, d_head)

# List all available keys
print(list(cache.keys())[:10])
```

### Cache Key Formats

| Component | Key Format |
|-----------|-----------|
| Residual pre-attention | `'resid_pre', layer` |
| Residual post-attention | `'resid_mid', layer` |
| Residual post-MLP | `'resid_post', layer` |
| Attention pattern | `'pattern', layer, 'attn'` |
| Q/K/V matrices | `f'blocks.{layer}.attn.hook_q'` etc |
| Attention output | `f'blocks.{layer}.attn.hook_z'` |
| MLP activation | `f'blocks.{layer}.mlp.hook_post'` |

---

## Hooks

### Adding Hooks

```python
# Preferred: use context manager pattern (auto-removes)
def hook_fn(value, hook):
    # value: the activation tensor at this hook point
    # hook: HookPoint object with .name, .layer, etc.
    print(f"Hook {hook.name}: shape {value.shape}")
    return value  # MUST return value

# Context manager — hooks removed automatically
with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
    logits = model(tokens)

# Manual hook (must clean up explicitly)
model.add_hook(hook_name, hook_fn)
logits = model(tokens)
model.reset_hooks()  # ALWAYS do this
```

### Hook Points for Patching

```python
# To patch attention output at layer N, head H:
def patch_head_hook(value, hook, clean_act, corrupted_act, head_idx):
    """Patch head H's output with corrupted activation."""
    # value: (batch, seq, n_heads, d_head)
    patched = value.clone()  # Never modify in-place
    patched[:, :, head_idx, :] = corrupted_act[:, :, head_idx, :]
    return patched

# Register with partial
from functools import partial
hook_fn = partial(patch_head_hook,
                  clean_act=clean_cache['blocks.9.attn.hook_z'],
                  corrupted_act=corrupted_cache['blocks.9.attn.hook_z'],
                  head_idx=9)

model.add_hook('blocks.9.attn.hook_z', hook_fn)
```

---

## Gradient-Based Attribution

```python
# Enable gradient tracking on the correct tensor
# DO NOT use torch.no_grad() when computing attributions

model.zero_grad()
logits, cache = model.run_with_cache(clean_tokens)

# Compute target metric
target_logit_diff = logits[0, -1, target_id] - logits[0, -1, distractor_id]

# Backward pass — computes gradients in cache tensors
target_logit_diff.backward()

# Read gradients
# Note: gradients are on the CACHE TENSORS, not model parameters
for layer in range(model.cfg.n_layers):
    hook_key = f'blocks.{layer}.attn.hook_z'
    if cache[hook_key].grad is not None:
        grad = cache[hook_key].grad  # same shape as activation
```

---

## Tensor Shape Conventions

Always comment shapes in non-obvious operations:

```python
# (batch, seq_len, d_model) — residual stream
# (batch, seq_len, n_heads, d_head) — per-head activations
# (batch, n_heads, seq_len, seq_len) — attention patterns
# (n_layers, n_heads) — attribution scores
# (vocab_size,) — logit vector at a position
```

---

## Device Management

```python
# Get device from model — never hardcode
device = model.cfg.device

# Move all new tensors to the same device
tokens = tokens.to(device)
activation = activation.to(device)

# Verify device before expensive operations
assert tokens.device == next(model.parameters()).device, \
    f"Token device {tokens.device} != model device {next(model.parameters()).device}"
```

---

## Common Mistakes and Fixes

### Mistake: Modifying cache tensors in-place
```python
# WRONG — corrupts gradient computation
cache['blocks.9.attn.hook_z'][:, :, 9, :] = 0

# CORRECT
patched = cache['blocks.9.attn.hook_z'].clone()
patched[:, :, 9, :] = 0
```

### Mistake: Not resetting hooks
```python
# WRONG — hooks accumulate across calls
model.add_hook('blocks.9.attn.hook_z', my_hook)
logits = model(tokens)
# next call still has the hook attached!

# CORRECT
model.add_hook('blocks.9.attn.hook_z', my_hook)
logits = model(tokens)
model.reset_hooks()  # Clean slate
```

### Mistake: Using no_grad during attribution
```python
# WRONG — can't compute gradients
with torch.no_grad():
    logits, cache = model.run_with_cache(clean_tokens)
target_logit_diff.backward()  # This will fail

# CORRECT — only use no_grad for non-attribution runs
logits, cache = model.run_with_cache(clean_tokens)  # Gradients enabled
target_logit_diff.backward()
```

### Mistake: Wrong cache key format
```python
# WRONG — old format
cache['blocks.9.attn.hook_pattern']

# CORRECT — TransformerLens key format
cache['pattern', 9, 'attn']
# OR
cache[f'blocks.9.attn.hook_pattern']  # string format also works
```

---

## Multi-Model Support (Beyond GPT-2)

Glassbox supports multiple architectures. Key differences:

```python
# GPT-2 Small (12 layers, 12 heads, d_model=768)
model = HookedTransformer.from_pretrained("gpt2")

# GPT-2 Medium (24 layers, 16 heads, d_model=1024)
model = HookedTransformer.from_pretrained("gpt2-medium")

# Pythia (EleutherAI) — same TransformerLens API
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m")

# Always check config after loading
print(f"Layers: {model.cfg.n_layers}")
print(f"Heads: {model.cfg.n_heads}")
print(f"d_model: {model.cfg.d_model}")
print(f"d_head: {model.cfg.d_head}")    # = d_model / n_heads
print(f"vocab_size: {model.cfg.d_vocab}")
```

## Memory-Efficient Analysis

For larger models, cache only what you need:

```python
# Filter cache to save memory
names_filter = lambda name: name.endswith("hook_z") or name.endswith("hook_resid_post")

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
# cache only contains hook_z and hook_resid_post — much smaller

# For attribution patching, you need gradients — can't use no_grad
# But you can still filter cache:
logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
```

## Batch Processing for Confidence-Faithfulness Correlation

The r=0.009 finding requires running many prompts:

```python
def compute_confidence_faithfulness_correlation(
    model: HookedTransformer,
    prompt_pairs: List[Tuple[str, str, str, str]],
    n_samples: int = 100
) -> float:
    """
    Compute Pearson r between model confidence and faithfulness F1.

    Args:
        prompt_pairs: List of (clean, corrupted, target, distractor) tuples
        n_samples: Number of prompt pairs to sample

    Returns:
        Pearson correlation coefficient r

    Reference: arXiv 2603.09988 — finding: r = 0.009
    """
    from scipy.stats import pearsonr
    import random

    sample = random.sample(prompt_pairs, min(n_samples, len(prompt_pairs)))
    confidences = []
    f1_scores = []

    for clean, corrupted, target, distractor in sample:
        # Confidence: model's softmax probability for target token
        tokens = model.to_tokens(clean)
        with torch.no_grad():
            logits = model(tokens)
        target_id = model.to_single_token(target)
        probs = torch.softmax(logits[0, -1], dim=0)
        confidence = probs[target_id].item()

        # Faithfulness F1: from circuit analysis
        # (simplified — full version uses attribution patching)
        f1 = compute_f1_for_pair(model, clean, corrupted, target, distractor)

        confidences.append(confidence)
        f1_scores.append(f1)

    r, p_value = pearsonr(confidences, f1_scores)
    return float(r)
```

## TransformerLens Version Compatibility

```python
# Check version
import transformer_lens
print(transformer_lens.__version__)

# Known breaking changes:
# v1.x → v2.x: cache key format changed
# Old: cache['blocks.9.attn.hook_pattern']
# New: cache['pattern', 9, 'attn'] OR cache['blocks.9.attn.hook_pattern'] (both work)

# Always test cache key access early in your script:
_, cache = model.run_with_cache(model.to_tokens("test"))
assert 'blocks.0.attn.hook_z' in cache.cache_dict, "Cache key format mismatch"
```
