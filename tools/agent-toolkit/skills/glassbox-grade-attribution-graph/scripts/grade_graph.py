#!/usr/bin/env python3
"""Grade a circuit-tracer attribution graph with Glassbox-style faithfulness metrics.

circuit-tracer (safety-research/circuit-tracer) explains a prediction as a graph of
transcoder features. This script asks Glassbox's question of that graph: is the
explanation faithful? It prunes the graph to its most influential features (the
"circuit"), then intervenes on the real model:

  sufficiency       = LD(only circuit features active) / LD(all features active)
  comprehensiveness = 1 - LD(circuit features zeroed) / LD(all features active)
  f1                = harmonic mean of the two (as in Glassbox)

LD is the logit difference correct - incorrect at the last position. Both metrics
use zero-ablation of transcoder features via ReplacementModel.feature_intervention,
so they are NOT numerically comparable to Glassbox's head-level scores (Glassbox
uses attribution-sum sufficiency and corrupted-patching comprehensiveness). The
graph's own replacement and completeness scores are reported alongside.

    python grade_graph.py --model google/gemma-2-2b --transcoders gemma \
        --prompt "The capital of the state containing Dallas is" \
        --correct " Austin" --incorrect " Dallas" -o graded.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def _ld(logits: torch.Tensor, a: int, b: int) -> float:
    last = logits.reshape(-1, logits.shape[-2], logits.shape[-1])[0, -1] if logits.dim() == 3 else logits[-1]
    return (last[a] - last[b]).item()


def _zero(rows: torch.Tensor):
    return [(int(l), int(p), int(f), 0.0) for l, p, f in rows.tolist()]


def grade(model, graph, correct_id: int, incorrect_id: int, node_threshold: float = 0.8,
          freeze_attention: bool = False) -> dict:
    from circuit_tracer.graph import compute_graph_scores, prune_graph

    tokens = graph.input_tokens
    n_sel = len(graph.selected_features)
    node_mask, _, _ = prune_graph(graph, node_threshold=node_threshold, edge_threshold=0.98)
    kept_sel = torch.nonzero(node_mask[:n_sel].cpu()).flatten()
    kept_active = graph.selected_features.cpu()[kept_sel]
    circuit = graph.active_features.cpu()[kept_active]

    all_idx = torch.arange(len(graph.active_features))
    outside = graph.active_features.cpu()[~torch.isin(all_idx, kept_active)]

    def run(interventions):
        logits, _ = model.feature_intervention(tokens, interventions, freeze_attention=freeze_attention,
                                               return_activations=False)
        return _ld(logits, correct_id, incorrect_id)

    ld_full = run([])
    ld_without = run(_zero(circuit)) if len(circuit) else ld_full
    ld_only = run(_zero(outside)) if len(outside) else ld_full

    if abs(ld_full) < 1e-8:
        suff = comp = 0.0
    else:
        suff = float(min(max(ld_only / ld_full, 0.0), 1.0))
        comp = float(min(max(1.0 - ld_without / ld_full, 0.0), 1.0))
    f1 = 2 * suff * comp / (suff + comp) if suff + comp > 0 else 0.0
    replacement, completeness = compute_graph_scores(graph)
    return {
        "node_threshold": node_threshold,
        "freeze_attention": freeze_attention,
        "n_active_features": int(len(graph.active_features)),
        "n_selected_features": int(n_sel),
        "n_circuit_features": int(len(circuit)),
        "circuit_features": circuit.tolist(),  # (layer, position, feature)
        "ld_full": ld_full,
        "ld_circuit_zeroed": ld_without,
        "ld_circuit_only": ld_only,
        "sufficiency": suff,
        "comprehensiveness": comp,
        "f1": f1,
        "graph_replacement_score": replacement,
        "graph_completeness_score": completeness,
    }


def letter_grade(f1: float) -> str:
    # Same cut-offs as Glassbox's A-D grades are not assumed here; these are documented
    # local bands so readers can compare runs of this script with each other.
    return "A" if f1 >= 0.85 else "B" if f1 >= 0.65 else "C" if f1 >= 0.4 else "D"


def vault_entry(res: dict, model_name: str, prompt: str) -> dict:
    return {
        "section": "§2",
        "article_refs": ["Article 13(1)", "Article 15(1)"],
        "title": f"Faithfulness of circuit-tracer attribution graph ({model_name})",
        "description": (
            f"Attribution graph for {prompt!r} pruned to {res['n_circuit_features']} transcoder features "
            f"(node threshold {res['node_threshold']}). Zero-ablation sufficiency {res['sufficiency']:.3f}, "
            f"comprehensiveness {res['comprehensiveness']:.3f}, F1 {res['f1']:.3f}. Graph replacement score "
            f"{res['graph_replacement_score']:.3f}: the share of influence not routed through error nodes."
        ),
        "evidence_type": "faithfulness",
        "metric_name": "attribution_graph_f1",
        "metric_value": res["f1"],
        "threshold": None,
        "passed": None,
        "raw": {k: v for k, v in res.items() if k != "circuit_features"} | {"prompt": prompt},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--transcoders", default="gemma",
                    help="circuit-tracer transcoder set: 'gemma', 'llama', or a HF repo / config path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--correct", required=True, help="expected next token (include leading space)")
    ap.add_argument("--incorrect", required=True, help="contrast token")
    ap.add_argument("--node-threshold", type=float, default=0.8)
    ap.add_argument("--max-feature-nodes", type=int, default=8192)
    ap.add_argument("--freeze-attention", action="store_true",
                    help="freeze attention patterns during interventions (circuit-tracer default); off lets effects propagate")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--save-graph", help="also save the raw graph (.pt) for circuit-tracer's visualizer")
    ap.add_argument("-o", "--output", default="graded-graph.json")
    args = ap.parse_args(argv)

    from circuit_tracer import ReplacementModel, attribute

    model = ReplacementModel.from_pretrained(args.model, args.transcoders, dtype=getattr(torch, args.dtype))
    tok = model.tokenizer
    a = tok.encode(args.correct, add_special_tokens=False)
    b = tok.encode(args.incorrect, add_special_tokens=False)
    if len(a) != 1 or len(b) != 1:
        print(f"warning: answers tokenize to {len(a)} / {len(b)} tokens; using the first", file=sys.stderr)
    graph = attribute(args.prompt, model, attribution_targets=[args.correct, args.incorrect],
                      max_feature_nodes=args.max_feature_nodes)
    if args.save_graph:
        graph.to_pt(args.save_graph)
    res = grade(model, graph, a[0], b[0], args.node_threshold, args.freeze_attention)
    res["grade"] = letter_grade(res["f1"])
    out = {"model": args.model, "transcoders": args.transcoders, "prompt": args.prompt, **res,
           "vault_entries": [vault_entry(res, args.model, args.prompt)]}
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"circuit: {res['n_circuit_features']} features | suff {res['sufficiency']:.3f} | "
          f"comp {res['comprehensiveness']:.3f} | F1 {res['f1']:.3f} ({res['grade']}) | "
          f"replacement {res['graph_replacement_score']:.3f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
