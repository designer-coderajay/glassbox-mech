---
name: mechanistic-interpretability
description: Domain knowledge skill for mechanistic interpretability work in Glassbox. Activates when writing, reviewing, or extending any MI analysis code — attention patterns, circuits, logit lens, activation patching, faithfulness metrics, SAE features, or steering vectors.
origin: Glassbox
version: 2.0
updated: 2026-04-01
---

# Mechanistic Interpretability Skill

Core domain knowledge for working in the mechanistic interpretability codebase.

## When to Activate

- Writing or reviewing circuit discovery code
- Implementing attention analysis
- Working with activation patching or hook functions
- Computing faithfulness metrics
- Extending Glassbox with new interpretability methods

---

## Core Concepts

### The Residual Stream Paradigm

Transformer models process information via a residual stream — each layer adds its contribution to a running sum. Mechanistic interpretability reads this stream at every layer.

```
Input Embedding
    ↓
+ Attention Layer 0
+ MLP Layer 0
    ↓
+ Attention Layer 1
+ MLP Layer 1
    ...
    ↓
Logit Output (via Unembedding matrix)
```

Every component (attention head, MLP) can be analyzed independently because they all read from and write to the same residual stream.

### Attention Heads as Circuits

Each attention head performs:
1. **Query × Key** → attention pattern (which tokens to attend to)
2. **Attention × Value** → information moved from source to destination
3. Result added to residual stream

**Key insight**: not all heads matter equally. Circuit discovery identifies the heads that causally drive a specific behavior.

### Logit Difference as Metric

For token prediction tasks:
```
LD = logit(correct_token) - logit(incorrect_token)
```

Attribution patching measures each head's contribution to this LD.

---

## Attribution Patching

### How It Works

Given:
- **Clean prompt**: `"When Mary and John went to the store, John gave a drink to ___"` (answer: Mary)
- **Corrupted prompt**: same but with names swapped (answer changes)

Patching measures: if we replace head h's activation with the corrupted version, how much does LD change?

**Approximation via gradient**:
```python
Attribution(h) = (clean_activation - corrupt_activation) · ∇_activation[LD]
```

This requires 3 forward passes:
1. Clean forward pass → get clean activations and LD
2. Corrupted forward pass → get corrupted activations
3. Backward pass on clean → get gradients

### Implementation Pattern

```python
def compute_attribution(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    distractor_token: str
) -> torch.Tensor:
    """
    Compute attribution scores for all attention heads.

    Returns:
        Tensor of shape (n_layers, n_heads) with attribution scores.
        Higher = more responsible for the prediction.
    """
    # 1. Clean forward pass with gradient tracking
    clean_tokens = model.to_tokens(clean_prompt)
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    # 2. Corrupted forward pass
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # 3. Compute gradients w.r.t. attention activations
    target_id = model.to_single_token(target_token)
    distractor_id = model.to_single_token(distractor_token)
    logit_diff = clean_logits[0, -1, target_id] - clean_logits[0, -1, distractor_id]
    logit_diff.backward()

    # 4. Attribution = gradient × activation difference
    attributions = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            act_diff = (
                clean_cache[f'blocks.{layer}.attn.hook_z'][:, :, head]
                - corrupted_cache[f'blocks.{layer}.attn.hook_z'][:, :, head]
            )
            # gradient shape matches activation shape
            grad = clean_cache[f'blocks.{layer}.attn.hook_z'].grad[:, :, head]
            attributions[layer, head] = (act_diff * grad).sum().item()

    model.reset_hooks()
    return attributions
```

---

## Faithfulness Metrics

### Sufficiency (Eq. 2 from arXiv 2603.09988)

> "Do the cited heads, by themselves, account for the prediction?"

```python
sufficiency = sum(contrib[h] for h in cited_heads) / LD_clean
# Glassbox result: 1.00 (cited heads fully explain IOI task)
```

### Comprehensiveness

> "Does ablating (removing) the cited heads change the prediction?"

```python
comprehensiveness = (LD_clean - LD_ablated) / LD_clean
# Glassbox result: 0.22 (reveals distributed backup mechanisms)
```

### The Key Finding

Sufficiency = 1.00 but Comprehensiveness = 0.22. This means:
- The cited heads are sufficient to drive the prediction
- But the model has redundant backup mechanisms elsewhere
- Ablating the "main" circuit still leaves 78% of LD intact via other paths

This is why F1 = 0.64 (not 1.00) — sufficiency and comprehensiveness don't align.

---

## Logit Lens

Projects intermediate residual stream states through the unembedding matrix to show which tokens the model "predicts" at each layer.

```python
def logit_lens(
    model: HookedTransformer,
    prompt: str,
    position: int = -1
) -> torch.Tensor:
    """
    Returns (n_layers, vocab_size) logits from residual stream at each layer.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    logits_by_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][0, position]  # (d_model,)
        # Apply layer norm and unembed
        resid_normed = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        logits = model.unembed(resid_normed.unsqueeze(0).unsqueeze(0))[0, 0]
        logits_by_layer.append(logits)

    model.reset_hooks()
    return torch.stack(logits_by_layer)  # (n_layers, vocab_size)
```

---

## Shape Reference

Always comment expected tensor shapes in MI code:

| Variable | Shape | Meaning |
|----------|-------|---------|
| `tokens` | `(batch, seq_len)` | Token IDs |
| `logits` | `(batch, seq_len, vocab_size)` | Output logits |
| `resid_pre/post` | `(batch, seq_len, d_model)` | Residual stream states |
| `attn_pattern` | `(batch, n_heads, seq_len, seq_len)` | Attention weights |
| `hook_z` | `(batch, seq_len, n_heads, d_head)` | Attention output (pre-concat) |
| `hook_q/k/v` | `(batch, seq_len, n_heads, d_head)` | Q, K, V matrices |
| `attribution` | `(n_layers, n_heads)` | Attribution scores per head |

---

## Common Pitfalls

1. **Hook accumulation**: always call `model.reset_hooks()` after any patching or caching run
2. **Wrong position**: most tasks care about the final token position (`-1`), not position 0
3. **Logit diff sign**: `logit(correct) - logit(incorrect)` — if this is negative, the model is wrong
4. **Softmax vs logits**: always work in logit space for attribution (softmax destroys additive structure)
5. **Batching**: most research code runs batch=1 for interpretability — verify batch dim handling

---

## Sparse Autoencoders (SAEs)

SAEs are the current frontier of MI. They decompose the residual stream (or MLP activations) into sparse, interpretable features. Glassbox v4+ will support SAE feature analysis.

### How SAEs Work

A sparse autoencoder learns a dictionary of features:
```python
# Encoder: residual stream → sparse feature activations
features = ReLU(W_enc @ residual_stream + b_enc)  # (d_features,) — mostly zero

# Decoder: sparse features → residual stream approximation  
reconstructed = W_dec @ features + b_dec  # (d_model,)

# Loss: reconstruction + L1 sparsity penalty
loss = ||residual_stream - reconstructed||² + λ * ||features||₁
```

### Key Property: Superposition

Standard neurons activate for many unrelated features (superposition). SAE features are monosemantic — each feature corresponds to a single interpretable concept.

### Glassbox Integration (Future)

```python
# Planned API (v4.0)
from glassbox.sae import SAEAnalyzer

sae_analyzer = SAEAnalyzer(model, sae_checkpoint="sae_gpt2_layer9")
feature_activations = sae_analyzer.get_features(prompt)
# Returns: Dict[int, float] — feature_id → activation strength
```

### Literature

| Paper | Key Contribution |
|-------|-----------------|
| Cunningham et al. 2023 | SAEs find monosemantic features |
| Bricken et al. 2023 (Anthropic) | 1M features in Claude 3 |
| Templeton et al. 2024 (Anthropic) | Feature geometry and universality |
| Marks et al. 2024 | SAE circuit discovery |

---

## Steering Vectors / Representation Engineering

Steering vectors modify model behavior by adding a direction to the residual stream during the forward pass.

### How It Works

```python
# Find steering direction (contrast pair approach)
positive_prompts = ["The capital of France is Paris", "2+2=4"]
negative_prompts  = ["The capital of France is London", "2+2=5"]

# Run both through model, extract residual stream at layer L
pos_residuals = [get_residual(p, layer=L) for p in positive_prompts]  # Each: (seq, d_model)
neg_residuals = [get_residual(p, layer=L) for p in negative_prompts]

# Steering vector = mean difference
steering_vector = mean(pos_residuals) - mean(neg_residuals)  # (d_model,)

# Apply during inference
def steering_hook(value, hook, alpha=20.0):
    return value + alpha * steering_vector  # Add direction to residual stream

with model.hooks(fwd_hooks=[(f'blocks.{L}.hook_resid_post', steering_hook)]):
    output = model.generate(prompt, max_new_tokens=50)
```

### Relationship to Circuits

Steering vectors and circuit analysis are complementary:
- **Circuits**: identify *which heads* drive a behavior (localized)
- **Steering vectors**: identify *which direction* in residual stream encodes a concept (distributed)

Both approaches contribute to Annex IV compliance documentation.

### Literature

| Paper | Key Contribution |
|-------|-----------------|
| Turner et al. 2023 | Activation addition (steering) |
| Zou et al. 2023 | Representation Engineering |
| Arditi et al. 2024 | Refusal direction in LLMs |
| Templeton et al. 2024 | Feature steering in production models |

---

## Probing Classifiers

Linear probes test whether a concept is linearly represented in activations:

```python
from sklearn.linear_model import LogisticRegression

# Collect activations at layer L for labeled examples
X_train = []  # List of residual stream states (d_model,)
y_train = []  # Binary labels

for prompt, label in dataset:
    residual = get_residual(prompt, layer=L, position=-1)
    X_train.append(residual.numpy())
    y_train.append(label)

# Train linear probe
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)

# Probe accuracy tells us: is this concept linearly decodable at layer L?
print(f"Probe accuracy: {probe.score(X_test, y_test):.3f}")
```

**When to use probes vs. circuits:**
- Probes: test whether information *exists* in activations (global)
- Circuits: test which components *move* that information (local, causal)

For Annex IV compliance, **circuits are stronger evidence** than probes because they show causal responsibility, not just correlation.
