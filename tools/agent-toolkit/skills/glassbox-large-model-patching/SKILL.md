---
name: glassbox-large-model-patching
description: "Locate where a language model's decision is computed with residual-stream activation patching (layer x token position) using nnsight, locally or remotely on NDIF for models too large for local hardware (e.g. Llama-3.1-8B and larger, depending on what NDIF currently hosts). Use when the user wants Glassbox-style causal analysis on a big model, asks about NDIF or nnsight, or needs interpretability evidence beyond what fits on a laptop."
---

# Activation patching on large models (nnsight / NDIF)

Glassbox's own engine is validated up to about 12B locally. nnsight runs the same kind of causal
intervention on any Hugging Face causal LM. With `--remote` it runs on NDIF's shared GPUs, so the
model never has to fit on the user's machine.

## Setup

```bash
pip install "nnsight==0.7.0" "torch" "transformers"
```

Remote runs need an NDIF API key (sign up via https://ndif.us/): `export NDIF_API_KEY=...`.
Which models NDIF currently hosts changes over time. Check https://nnsight.net/status/ before
promising a model. Gated models (Llama, Gemma) also need Hugging Face access for the tokenizer.

## Design the prompt pair

Patching needs a **clean** and a **corrupted** prompt that tokenize to the same length and differ in
the fact the decision depends on, plus two answer tokens:

| | Example (IOI) |
|---|---|
| clean | `When Mary and John went to the store, John gave a drink to` |
| corrupt | `When Mary and John went to the store, Mary gave a drink to` |
| correct / incorrect | `" Mary"` / `" John"` (leading space matters) |

For an audit, build the pair from the real decision. For example, swap the applicant attribute
being tested and keep everything else identical.

## Run

```bash
# remote on NDIF
python scripts/layer_patching.py meta-llama/Llama-3.1-8B \
  --clean "..." --corrupt "..." --correct " Mary" --incorrect " John" --remote -o patching.json

# local (small models, or a GPU box)
python scripts/layer_patching.py gpt2 --clean "..." --corrupt "..." --correct " Mary" --incorrect " John"
```

Each layer costs one forward pass per `--batch-size` positions (default 16). Lower it if memory is tight.

## Read the result

`recovery[layer][position]` = share of the clean logit difference restored by patching that one
activation (1.0 = fully restored, 0 = no effect). Positions before the first differing token are
always ~0, as a sanity check. In the final layer, the last position is always ~1.0. Report the
`top_sites`, and if useful, plot `recovery` as a heatmap (layers on y, tokens on x).

`patching.json` also contains `vault_entries`. Add them to a Glassbox vault with
`VaultEntry(**e)` and `AnnexIVEvidenceVault(...).build_vault(custom_entries=...)`.

## Limits to state

- This is residual-stream patching: it shows *where* information flows, not which heads or
  features compute it. For head-level circuits on models that fit locally, use Glassbox itself
  (`GlassboxV2.analyze`).
- One prompt pair is an anecdote. Run several pairs before drawing conclusions.
- Tested offline with nnsight 0.7.0 on GPT-2 and Llama module layouts. Other layouts may need a path
  added to `decoder_layers()` in the script.
