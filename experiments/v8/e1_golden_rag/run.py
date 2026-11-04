#!/usr/bin/env python3
"""Run E1 (PROTOCOL.md §8): competence gate → reference → main (GB, B0–B2) → recovery
→ scoring. Every finished unit is checkpointed (``glassbox.v7.checkpoint``), so a run
that disconnects can be resumed with the same command and the same ``--out``.

Usage (Colab GPU, see docs/v7/COLAB_GPU.md; --out on Google Drive):
    python3 experiments/v8/e1_golden_rag/run.py --backend hf \\
        --out /content/drive/MyDrive/glassbox_runs/e1/attempt1
Tests use ``--backend toy`` (no model).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from glassbox.v7 import compute
from glassbox.v7.checkpoint import Checkpoint
from glassbox.v7.intervene import InterventionSettings

HERE = Path(__file__).resolve().parent
PROTOCOL = "experiments/v8/e1_golden_rag/PROTOCOL.md v1.0"
N_MAIN, N_REF, N_REC, N_GATE = 30, 200, 60, 10
GATE_MIN = 0.80
N_PERM, ALPHA = 999, 0.05
SEEDS = {
    "main": "e1-main",
    "reference": "e1-reference",
    "recovery": "e1-recovery",
    "gate": "e1-gate",
}
EMBEDDER = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
LADDER = (
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "float32",
    ),
    ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "4bit"),
)
log = logging.getLogger("e1")


def _mod(name: str) -> Any:
    key = f"e1_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class Backend:
    """Builds embedder and generators for ``toy`` or ``hf``."""

    def __init__(self, kind: str, device: str) -> None:
        self.kind, self.device = kind, device
        sysm = _mod("system")
        if kind == "toy":
            self.embedder = sysm.ToyEmbedder()
            self.ladder: Sequence[Any] = (("toy", "-", "-"),)
        else:  # pragma: no cover - needs models; exercised on Colab
            self.embedder = sysm.STEmbedder(EMBEDDER[0], EMBEDDER[1], device)
            self.ladder = LADDER

    def generator(self, rung: Sequence[str]) -> Any:
        sysm = _mod("system")
        if self.kind == "toy":
            return sysm.ToyGenerator()
        return sysm.HFGenerator(
            rung[0], rung[1], self.device, rung[2]
        )  # pragma: no cover


def _gate(backend: Backend, dev: List[Dict[str, Any]]) -> Dict[str, Any]:
    sysm = _mod("system")
    rungs = []
    for rung in backend.ladder:
        gen = backend.generator(rung)
        s = sysm.System(backend.embedder, gen, SEEDS["gate"])
        hits = sum(
            s.answer(sysm.BASE, q, i)[0] == "correct"
            for q in dev
            for i in range(N_GATE)
        )
        acc = hits / (len(dev) * N_GATE)
        rungs.append(
            {
                "model": rung[0],
                "revision": rung[1],
                "precision": rung[2],
                "accuracy": round(acc, 4),
                "n": len(dev) * N_GATE,
            }
        )
        log.info("gate %s: %.3f", rung[0], acc)
        if acc >= GATE_MIN:
            return {"rungs": rungs, "selected": rung[0]}
        del gen, s
        _free_gpu()
    return {"rungs": rungs, "selected": None}


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _timed(work: Callable[[], Dict[str, Any]]) -> Callable[[], Dict[str, Any]]:
    def run() -> Dict[str, Any]:
        start = time.perf_counter()
        out = work()
        out["seconds"] = round(time.perf_counter() - start, 3)
        return out

    return run


def _stages(
    cp: Checkpoint, backend: Backend, rung: Sequence[str], test: List[Dict[str, Any]]
) -> Dict[str, Any]:
    sysm, ref, meth, rec = (
        _mod(n) for n in ("system", "reference", "methods", "recovery")
    )
    gen = backend.generator(rung)
    systems = {
        k: sysm.System(backend.embedder, gen, SEEDS[k])
        for k in ("main", "reference", "recovery")
    }
    settings = InterventionSettings(N_MAIN, N_PERM, ALPHA, SEEDS["main"])
    per_q: Dict[str, Any] = {}
    for q in test:
        r = cp.run(
            f"ref_{q['id']}",
            _timed(
                lambda q=q: ref.reference_question(
                    systems["reference"], q, sysm.FACTORS, N_REF
                )
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            m = cp.run(
                f"main_{q['id']}",
                _timed(
                    lambda q=q, tmp=tmp: meth.run_methods(
                        systems["main"], [q], Path(tmp), settings
                    )[q["id"]]
                ),
            )
        v = cp.run(
            f"rec_{q['id']}",
            _timed(
                lambda q=q, m=m: rec.recovery_question(
                    systems["recovery"], q, m["GB"], sysm.BASE, sysm.CANDIDATE, N_REC
                )
            ),
        )
        log.info("%s done", q["id"])
        per_q[q["id"]] = {
            "kind": q["kind"],
            "reference": r,
            "methods": m,
            "recovery": v,
        }
    return per_q


def _score(per_q: Dict[str, Any]) -> Dict[str, Any]:
    sc = _mod("score")
    labels = {
        q: {f: x["label"] for f, x in row["reference"]["factors"].items()}
        for q, row in per_q.items()
    }
    methods = {
        q: {m: row["methods"][m] for m in sc.METHODS} for q, row in per_q.items()
    }
    recovery = {q: row["recovery"] for q, row in per_q.items()}
    return {
        "criteria": sc.criteria(labels, methods, recovery),
        "descriptive": sc.descriptive(labels, methods),
    }


PACKAGES = (
    "glassbox-mech-interp",
    "torch",
    "transformers",
    "sentence-transformers",
    "bitsandbytes",
    "accelerate",
    "numpy",
)


def _software() -> Dict[str, Optional[str]]:
    """Installed versions of the packages that matter, plus the git commit."""
    found: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    try:
        found["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=HERE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        found["git_commit"] = None
    return found


def _environment(backend: Backend) -> Dict[str, Any]:
    if backend.kind == "toy":
        return {"device": "cpu", "backend": "toy (not a language model)"}
    cfg = compute.resolve(compute.ComputeConfig(backend.device))  # pragma: no cover
    return compute.environment_record(cfg)  # pragma: no cover


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run or resume E1 and write ``results.json`` in ``--out``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backend", choices=("toy", "hf"), default="hf")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N test questions (smoke tests; never scored)",
    )
    args = p.parse_args(argv)
    device = args.device or ("cpu" if args.backend == "toy" else "cuda")
    fingerprint = {
        "protocol": PROTOCOL,
        "n": [N_MAIN, N_REF, N_REC, N_GATE],
        "gate_min": GATE_MIN,
        "seeds": SEEDS,
        "backend": args.backend,
        "limit": args.limit,
        "ladder": [list(r) for r in LADDER],
        "embedder": list(EMBEDDER),
    }
    cp = Checkpoint(args.out / "checkpoint", fingerprint)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sysm = _mod("system")
    backend = Backend(args.backend, device)
    test = sysm.load_questions("test")[: args.limit]
    gate = cp.run("gate", _timed(lambda: _gate(backend, sysm.load_questions("dev"))))
    result: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "gate": gate,
        "limit": args.limit,
        "environment": _environment(backend),
        "software": _software(),
        "embedder": backend.embedder.name,
    }
    if gate["selected"] is None:
        result.update(
            {
                "per_question": {},
                "criteria": {"overall": "INCONCLUSIVE"},
                "note": "no generator passed the competence gate",
            }
        )
    else:
        rung = next(r for r in backend.ladder if r[0] == gate["selected"])
        per_q = _stages(cp, backend, rung, test)
        result.update({"per_question": per_q, **_score(per_q)})
    (args.out / "results.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"overall: {result['criteria']['overall']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
