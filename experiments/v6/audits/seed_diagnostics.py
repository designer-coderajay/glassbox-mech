"""Post-pilot diagnostics for the Pythia-410M seed pilot (2026-09-29 audit).

For each model spec:
  1. Provenance: exact Hub commit + weights sha256 of the requested revision; for seed
     variants, a tensor-by-tensor comparison of the official ``pytorch_model.bin`` at the
     requested revision against every SFconvertbot ``model.safetensors`` PR that the
     original pilot may have loaded (transformers preferred ``refs/pr/N``).
  2. Loading: re-load through the fixed loader (``use_safetensors=False``) and check the
     IOI results equal the stored pilot record exactly.
  3. Behaviour (to diagnose seed4): IOI accuracy, probability mass on the two names,
     how often the top-1 token is a name, counterfactual-probe accuracy, and mean
     next-token loss on a fixed set of plain English sentences (task-independent
     language-modelling sanity check).

Usage:
    python experiments/v6/audits/seed_diagnostics.py experiments/v6/runs/pilot_410m_seeds
Writes ``seed_diagnostics.json`` into the run directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from glassbox.v6 import measure, tasks
from glassbox.v6.claims import canonical_json
from glassbox.v6.pilot import parse_model_spec

# Self-written, license-free sentences for a generic language-modelling sanity check.
SENTENCES = [
    "The train to the city leaves every morning at seven o'clock.",
    "She opened the window because the room was too warm.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The children walked to school together in the rain.",
    "He forgot his keys, so he had to wait outside the door.",
    "The library closes early on Sundays during the summer.",
    "Our neighbours planted three apple trees in their garden.",
    "The meeting was moved from Tuesday to Thursday afternoon.",
    "A small dog was sleeping under the kitchen table.",
    "The price of bread went up again this year.",
    "They painted the old fence white before the winter.",
    "The museum has a large collection of ancient coins.",
    "Please remember to turn off the lights when you leave.",
    "The river flows slowly through the centre of the town.",
    "My brother is learning to play the piano.",
    "The shop on the corner sells fresh fruit and vegetables.",
    "It snowed heavily during the night, and the roads were closed.",
    "The teacher asked the students to read the first chapter.",
    "The bus was late because of an accident on the bridge.",
    "We had soup and bread for dinner last night.",
]


def _tensor_equality(repo: str, revision: str) -> Dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    ref = torch.load(hf_hub_download(repo, "pytorch_model.bin", revision=revision),
                     map_location="cpu", weights_only=True)
    out: Dict[str, Any] = {
        "n_tensors": len(ref),
        # Excludes non-parameter buffers (rotary inv_freq, attention masks, masked_bias).
        "n_params": int(sum(v.numel() for k, v in ref.items()
                            if v.is_floating_point() and not k.endswith(
                                ("inv_freq", "masked_bias", "attention.bias")))),
        "prs": {},
    }
    for d in HfApi().get_repo_discussions(repo):
        if not (d.is_pull_request and d.author == "SFconvertbot"):
            continue
        rev = f"refs/pr/{d.num}"
        try:
            st = load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
        except Exception as exc:  # noqa: BLE001 - record and continue
            out["prs"][rev] = f"not compared: {type(exc).__name__}"
            continue
        same_keys = set(st) == set(ref)
        diff = [k for k in set(st) & set(ref)
                if st[k].shape != ref[k].shape or not torch.equal(st[k], ref[k])]
        out["prs"][rev] = {"same_tensor_names": same_keys, "n_mismatched": len(diff),
                           "identical": same_keys and not diff}
    return out


def _vocab_hash(model: Any) -> str:
    items = sorted(model.tokenizer.get_vocab().items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


@torch.no_grad()
def _behaviour(model: Any, ds: Any) -> Dict[str, Any]:
    mass, top_name, cf_ok = [], 0, 0
    for it in ds.items:
        p = torch.softmax(model(model.to_tokens(it.prompt))[0, -1].float(), -1)
        t, d = model.to_single_token(it.target), model.to_single_token(it.distractor)
        mass.append(float(p[t] + p[d]))
        top_name += int(p.argmax()) in (t, d)
    cf = [p for p in ds.probes if p.kind == "counterfactual"]
    for pr in cf:
        it = ds.items[pr.item_index]
        logits = model(model.to_tokens(pr.prompt))[0, -1]
        cf_ok += bool(logits[model.to_single_token(pr.expected)]
                      > logits[model.to_single_token(it.target)])
    nll = []
    for s in SENTENCES:
        toks = model.to_tokens(s)
        nll.append(float(model(toks, return_type="loss")))
    return {"mean_name_mass": float(np.mean(mass)), "top1_is_name": top_name,
            "n_items": len(ds.items), "counterfactual_correct": cf_ok,
            "n_counterfactual": len(cf), "mean_sentence_nll": float(np.mean(nll))}


def main() -> int:
    run_dir = Path(sys.argv[1])
    rec = json.loads((run_dir / "record.json").read_text())
    specs: List[str] = rec["config"]["models"]
    stored = {m["spec"]: m for m in rec["models"]}
    report: Dict[str, Any] = {"run_dir": str(run_dir), "models": {}}
    ds = None
    for spec in specs:
        name, ckpt = parse_model_spec(spec)
        entry: Dict[str, Any] = {"hub": measure.hub_provenance(name, ckpt)}
        if measure.seed_variant(name):
            entry["tensor_check"] = _tensor_equality(f"EleutherAI/{name}", f"step{ckpt}")
        model = measure.load_model(name, ckpt)
        entry["vocab_sha256"] = _vocab_hash(model)
        if ds is None:
            ds = tasks.build_ioi(rec["config"]["n_prompts"], rec["config"]["seed"],
                                 measure.single_token_predicate(model),
                                 model.tokenizer.name_or_path)
            report["dataset_hash_matches_pilot"] = ds.dataset_hash == rec["dataset"]["hash"]
        correct, lds = measure.evaluate_items(model, ds.items)
        entry["ioi_correct"] = int(sum(correct))
        entry["ioi_identical_to_pilot"] = correct == stored[spec]["correct"]
        entry["ld_max_abs_diff_vs_pilot"] = float(
            np.max(np.abs(np.array(lds) - np.array(stored[spec]["ld"]))))
        entry["behaviour"] = _behaviour(model, ds)
        report["models"][spec] = entry
        print(spec, json.dumps({k: v for k, v in entry.items() if k != "hub"}), flush=True)
        del model
    out = run_dir / "seed_diagnostics.json"
    out.write_text(json.dumps(json.loads(canonical_json(report)), indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
