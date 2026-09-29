"""Build the locked IOI prompt sets for the V6 confirmatory experiment (Amendment 3 §5).

Deterministic and model-independent. No model output or distance is consulted.
  Attribution set A: the first 200 items of the IOI generator stream with seed 0.
  Matching set   M: the IOI generator stream with seed 2, taken in order, skipping any item
                    whose prompt text is already in A or already in M, until 1,000 items.
The generator (`v6.tasks.ioi/1.0.0`) is called with all 38 names, which were verified to be
single tokens for the Pythia tokenizer (EleutherAI/pythia-410m) on 2026-09-29.

Run once:  python experiments/v6/prompts/build_prompt_sets.py
The runner never regenerates prompts; it loads these files and checks their hashes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

from glassbox.v6 import tasks
from glassbox.v6.claims import canonical_json

HERE = pathlib.Path(__file__).resolve().parent
N_ATTR, SEED_ATTR = 200, 0
N_MATCH, SEED_MATCH, CANDIDATES = 1000, 2, 2000


def items_hash(items: list) -> str:
    return hashlib.sha256(canonical_json(items).encode()).hexdigest()


def build() -> dict:
    a = [dataclasses.asdict(x) for x in tasks.build_ioi(N_ATTR, SEED_ATTR).items]
    seen = {x["prompt"] for x in a}
    assert len(seen) == N_ATTR, "attribution stream contains duplicates"
    m, skipped = [], {"overlap_with_attribution": 0, "duplicate": 0}
    taken, consumed = set(), 0
    for x in (dataclasses.asdict(i) for i in tasks.build_ioi(CANDIDATES, SEED_MATCH).items):
        consumed += 1
        if x["prompt"] in seen:
            skipped["overlap_with_attribution"] += 1
            continue
        if x["prompt"] in taken:
            skipped["duplicate"] += 1
            continue
        taken.add(x["prompt"])
        m.append(x)
        if len(m) == N_MATCH:
            break
    assert len(m) == N_MATCH, "candidate stream too short"
    for i, x in enumerate(a):
        x["id"] = f"A{i:04d}"
    for i, x in enumerate(m):
        x["id"] = f"M{i:04d}"
    return {
        "attribution": {"version": tasks.TASK_VERSION, "seed": SEED_ATTR, "n": N_ATTR,
                        "items": a, "sha256": items_hash(a)},
        "matching": {"version": tasks.TASK_VERSION, "seed": SEED_MATCH, "n": N_MATCH,
                     "candidates_consumed": consumed, "skipped": skipped,
                     "items": m, "sha256": items_hash(m)},
    }


if __name__ == "__main__":
    sets = build()
    for name in ("attribution", "matching"):
        (HERE / f"ioi_{name}_v1.json").write_text(json.dumps(sets[name], indent=1))
        print(name, sets[name]["n"], sets[name]["sha256"], sets[name].get("skipped", ""))
