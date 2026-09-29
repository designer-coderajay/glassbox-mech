"""R = 4 / R = 5 extension of Gate 5 (criterion: amendment3_gate_criteria.md, dated note 5).

Reuses gate5_simulation._experiment unchanged (same world, estimator, bootstrap and seed
formula); only the run count differs. Resumable: each finished experiment is appended to
gate5_small_r.jsonl, so the run can be split into chunks.

    python experiments/v6/audits/gate5_small_r.py run      # compute missing experiments
    python experiments/v6/audits/gate5_small_r.py summary  # apply the pre-declared rule
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("g5", HERE / "gate5_simulation.py")
g5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g5)

SIMS = 200
CELLS = [("H0", "sparse", 4), ("H0", "dense", 4), ("H0", "sparse", 5), ("H0", "dense", 5)]
LOG = HERE / "gate5_small_r.jsonl"
OUT = HERE / "gate5_small_r_results.json"


def _done() -> set:
    if not LOG.exists():
        return set()
    return {(r["kind"], r["regime"], r["R"], r["sim"])
            for r in map(json.loads, LOG.read_text().splitlines())}


def _job(args):
    kind, regime, n_runs, sim = args
    res = g5._experiment((kind, regime, n_runs, sim))
    return {"kind": kind, "regime": regime, "R": n_runs, "sim": sim, **res}


def run(budget_s: float) -> None:
    done = _done()
    todo = [(k, r, n, s) for k, r, n in CELLS for s in range(SIMS) if (k, r, n, s) not in done]
    start = time.time()
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 1))
    with Pool(workers) as pool:
        for rec in pool.imap(_job, todo):
            with open(LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if time.time() - start > budget_s:
                pool.terminate()
                break
    print(f"completed {len(_done())} / {len(CELLS) * SIMS}")


def summary() -> None:
    rows = [json.loads(x) for x in LOG.read_text().splitlines()]
    out = {"criterion_type1_max": float(g5.TYPE1_MAX), "cells": {}}
    for kind, regime, n in CELLS:
        rs = sorted({r["sim"]: r for r in rows
                     if (r["kind"], r["regime"], r["R"]) == (kind, regime, n)}.values(),
                    key=lambda r: r["sim"])
        rate = float(np.mean([r["boot_reject"] for r in rs]))
        out["cells"][f"{kind}/{regime}/R{n}"] = {
            "sims": len(rs), "reject_rate_bootstrap": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / max(len(rs), 1))),
            "reject_rate_jackknife": float(np.mean([r["jack_reject"] for r in rs])),
            "mean_delta": float(np.mean([r["delta"] for r in rs])),
            "complete": len(rs) == SIMS, "pass": bool(len(rs) == SIMS and rate <= g5.TYPE1_MAX)}
    c = out["cells"]
    out["R4_pass"] = c["H0/sparse/R4"]["pass"] and c["H0/dense/R4"]["pass"]
    out["R5_pass"] = c["H0/sparse/R5"]["pass"] and c["H0/dense/R5"]["pass"]
    out["R_min"] = 4 if (out["R4_pass"] and out["R5_pass"]) else 6
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(float(os.environ.get("BUDGET_S", "1e9")))
    else:
        summary()
