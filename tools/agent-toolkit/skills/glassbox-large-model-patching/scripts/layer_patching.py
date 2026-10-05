#!/usr/bin/env python3
"""Residual-stream activation patching (layer x token position) with nnsight.

Runs on any Hugging Face causal LM nnsight can wrap, locally or on NDIF
(--remote, needs an NDIF API key), so it reaches models too large for a laptop.
For each (layer, position) it copies the clean run's residual stream into the
corrupted run and measures how much of the clean logit difference comes back:

    recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)

1.0 means that activation alone restores the clean behaviour; ~0 means it carries
nothing the decision needs. Clean and corrupted prompts must tokenize to the same
length (swap names/entities, as in IOI).

    python layer_patching.py meta-llama/Llama-3.1-8B \
        --clean "When Mary and John went to the store, John gave a drink to" \
        --corrupt "When Mary and John went to the store, Mary gave a drink to" \
        --correct " Mary" --incorrect " John" --remote -o patching.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def decoder_layers(model):
    """Return the list of transformer blocks for common HF architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers", "transformer.blocks"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj, path
        except AttributeError:
            continue
    raise SystemExit("could not find decoder layers; add this architecture's path to decoder_layers()")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        print(f"warning: {text!r} is {len(ids)} tokens; using the first", file=sys.stderr)
    return ids[0]


def patch_grid(model, clean: str, corrupt: str, correct: str, incorrect: str,
               remote: bool = False, batch_size: int = 16):
    tok = model.tokenizer
    c_ids, x_ids = tok(clean)["input_ids"], tok(corrupt)["input_ids"]
    if len(c_ids) != len(x_ids):
        raise SystemExit(f"prompts tokenize to different lengths ({len(c_ids)} vs {len(x_ids)})")
    a, b = token_id(tok, correct), token_id(tok, incorrect)
    layers, path = decoder_layers(model)
    n_layers, n_pos = len(layers), len(c_ids)

    # nnsight cannot rebind a function's locals from inside a trace on Python < 3.13,
    # so every value leaves the trace through a container created beforehand.
    clean_hs, out = [], {}
    with model.trace(clean, remote=remote):
        for i in range(n_layers):
            clean_hs.append(_hidden(layers[i].output).save())
        out["clean"] = model.output.logits[0, -1].save()
    with model.trace(corrupt, remote=remote):
        out["corrupt"] = model.output.logits[0, -1].save()
    clean_logits, corrupt_logits = out["clean"], out["corrupt"]

    ld_clean = (clean_logits[a] - clean_logits[b]).item()
    ld_corrupt = (corrupt_logits[a] - corrupt_logits[b]).item()
    denom = ld_clean - ld_corrupt
    if abs(denom) < 1e-6:
        raise SystemExit("clean and corrupted logit differences are equal; the prompt pair carries no signal")

    # One batched forward pass per chunk of positions: row i patches position i.
    # (Separate tracer.invoke blocks were not isolated from each other in nnsight 0.7.)
    grid = []
    for layer in range(n_layers):
        clean_layer = clean_hs[layer]
        row = []
        for start in range(0, n_pos, batch_size):
            positions = list(range(start, min(start + batch_size, n_pos)))
            saved = []
            with model.trace([corrupt] * len(positions), remote=remote):
                hs = _hidden(layers[layer].output)
                for i, pos in enumerate(positions):
                    hs[i, pos, :] = clean_layer[0, pos, :]
                saved.append(model.output.logits[:, -1].save())
            logits = saved[0]
            row += [((logits[i, a] - logits[i, b]).item() - ld_corrupt) / denom for i in range(len(positions))]
        grid.append(row)

    tokens = [tok.decode([t]) for t in c_ids]
    return {
        "layers_path": path,
        "n_layers": n_layers,
        "tokens": tokens,
        "ld_clean": ld_clean,
        "ld_corrupt": ld_corrupt,
        "recovery": grid,  # [layer][position]
    }


def summarize(res, top_k=10):
    cells = [(v, L, p) for L, row in enumerate(res["recovery"]) for p, v in enumerate(row)]
    cells.sort(reverse=True)
    return [{"layer": L, "position": p, "token": res["tokens"][p], "recovery": round(v, 4)} for v, L, p in cells[:top_k]]


def vault_entry(res, model_name, top):
    peak = top[0] if top else None
    return {
        "section": "§2",
        "article_refs": ["Article 13(1)", "Article 11"],
        "title": f"Residual-stream patching map ({model_name})",
        "description": (
            f"Layer-by-position activation patching on {res['n_layers']} layers. Clean logit "
            f"difference {res['ld_clean']:.3f}, corrupted {res['ld_corrupt']:.3f}. Strongest site: layer "
            f"{peak['layer']}, token {peak['token']!r} (recovery {peak['recovery']:.2f})." if peak else ""
        ),
        "evidence_type": "circuit",
        "metric_name": "max_patching_recovery",
        "metric_value": peak["recovery"] if peak else None,
        "threshold": None,
        "passed": None,
        "raw": {"top_sites": top, "recovery_grid": res["recovery"], "tokens": res["tokens"]},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face model id")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--corrupt", required=True)
    ap.add_argument("--correct", required=True, help="answer token for the clean prompt (include leading space)")
    ap.add_argument("--incorrect", required=True)
    ap.add_argument("--remote", action="store_true", help="run on NDIF (set NDIF_API_KEY first)")
    ap.add_argument("--batch-size", type=int, default=16, help="positions patched per forward pass")
    ap.add_argument("--dtype", default="auto", help="torch dtype for local runs (auto, float16, bfloat16, float32)")
    ap.add_argument("-o", "--output", default="patching.json")
    args = ap.parse_args(argv)

    from nnsight import CONFIG, LanguageModel

    if args.remote:
        key = os.environ.get("NDIF_API_KEY")
        if not key:
            raise SystemExit("set NDIF_API_KEY (sign up via https://ndif.us/)")
        CONFIG.set_default_api_key(key)
        model = LanguageModel(args.model)
    else:
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        model = LanguageModel(args.model, device_map="auto", torch_dtype=dtype, dispatch=True)

    res = patch_grid(model, args.clean, args.corrupt, args.correct, args.incorrect,
                     remote=args.remote, batch_size=args.batch_size)
    top = summarize(res)
    out = {"model": args.model, "remote": args.remote, **res, "top_sites": top,
           "vault_entries": [vault_entry(res, args.model, top)]}
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    for t in top[:5]:
        print(f"layer {t['layer']:>3}  pos {t['position']:>3} {t['token']!r:<12} recovery {t['recovery']:.3f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
