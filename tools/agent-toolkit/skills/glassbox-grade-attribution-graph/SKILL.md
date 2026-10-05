---
name: glassbox-grade-attribution-graph
description: "Build a circuit-tracer attribution graph (transcoder-feature explanation, as in Anthropic's 2025 circuit tracing work) for a prompt on Gemma-2-2B or Llama-3.2-1B, then grade how faithful that explanation is with Glassbox-style sufficiency, comprehensiveness and F1. Use when the user mentions circuit-tracer, attribution graphs, transcoders, Neuronpedia graphs, or wants to check whether someone else's interpretability explanation holds up."
---

# Grade a circuit-tracer attribution graph

Glassbox's position: an explanation is only worth something if it is *faithful*. This skill applies
that test to circuit-tracer's feature graphs.

## Setup

```bash
pip install "circuit-tracer==0.5.0" "torch" "transformers<=4.57.3"
```

Needs Hugging Face access to download the model and transcoders (several GB; Gemma needs licence
acceptance on its HF page). A GPU is strongly recommended; CPU works only for very short prompts.
Supported transcoder sets: `gemma` (google/gemma-2-2b) and `llama` (meta-llama/Llama-3.2-1B), or a
HF repo / config path.

## Run

```bash
python scripts/grade_graph.py --model google/gemma-2-2b --transcoders gemma \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --correct " Austin" --incorrect " Dallas" \
  --save-graph graph.pt -o graded.json
```

Useful flags: `--node-threshold` (default 0.8: keep features carrying 80% of influence),
`--max-feature-nodes` (default 8192), `--freeze-attention` (circuit-tracer's default intervention
mode; off by default here so effects propagate through attention).

## What the numbers mean

| Field | Meaning |
|---|---|
| `sufficiency` | logit difference when **only** circuit features stay active, over the full one |
| `comprehensiveness` | 1 − logit difference with circuit features **zeroed**, over the full one |
| `f1` | harmonic mean, as in Glassbox |
| `graph_replacement_score` | circuit-tracer's own score: share of influence not routed through error nodes |

Both faithfulness metrics use zero-ablation of transcoder features and are clipped to [0, 1]. They are
**not numerically comparable** to Glassbox's head-level scores, which use attribution-sum sufficiency
and corrupted-patching comprehensiveness. Compare graphs graded by this script with each other.
The raw logit differences are in the JSON if a clipped value hides an overshoot.

A high replacement score with low comprehensiveness is the interesting case: the graph *looks*
complete, but zeroing its features barely moves the decision, so backup paths exist.

## Hand-offs

- `graded.json` → `vault_entries` can go into a Glassbox Annex IV vault (`VaultEntry(**e)`).
- `graph.pt` can be opened with circuit-tracer's visualiser (`circuit-tracer start-server`) to inspect features.

## Verified so far

The grading logic was tested end to end on a tiny random model with random transcoders (pipeline,
shapes, and that "no intervention" reproduces the model's own logits). It has not yet been run on
Gemma or Llama, because model downloads were blocked where it was built. Treat the first real run as a check.
