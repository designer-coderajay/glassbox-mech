#!/usr/bin/env python3
"""Run E2 (PROTOCOL.md §9–10). Two stages, each its own invocation and ``--out``:

* ``--stage pilot``: competence gate, then the dev pilot (11 cells × 30 episodes per
  dev question), the per-factor effect check and a runtime projection. Its output goes
  into Addendum 2 before the test run.
* ``--stage test``: competence gate, then per test question (reference, main,
  recovery), then scoring.

Every finished unit is checkpointed (``glassbox.v7.checkpoint``), so a run that stops
can be resumed with the same command and the same ``--out``. The installed package
list is written to ``--out/pip_freeze.txt``, the GPU name to ``results.json``.

Usage (rented GPU host, see RUNBOOK.md):
    python3 experiments/v8/e2_golden_agent/run.py --stage pilot --out runs/e2/pilot1
    python3 experiments/v8/e2_golden_agent/run.py --stage test --out runs/e2/attempt1
Tests use ``--backend toy`` (scripted, not a language model).
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
PROTOCOL = "experiments/v8/e2_golden_agent/PROTOCOL.md v1.0"
N_MAIN, N_REF, N_REC, N_GATE, N_PILOT = 30, 200, 60, 10, 30
BLOCKS = {"gate": 10, "pilot": 30, "reference": 50, "main": 30, "recovery": 60}
GATE_MIN, PARSE_MIN, PILOT_DROP = 0.80, 0.95, 0.20
N_PERM, ALPHA = 999, 0.05
SEEDS = {
    "main": "e2-main",
    "reference": "e2-reference",
    "recovery": "e2-recovery",
    "gate": "e2-gate",
    "pilot": "e2-pilot",
}
LADDER = (
    ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "bfloat16"),
    ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
     "bfloat16"),
)
PILOT_TARGETS = {
    "invoice_tool_desc": (("B",), -1),
    "refund_tool_desc": (("C-inel",), -1),
    "runbook_order": (("D",), -1),
    "max_steps": (("C-def", "D"), -1),
    "answer_policy": (("E",), -1),
    "refund_fix": (("C-def",), +1),
}
REMAINING = 24 * 11 * N_REF + 24 * 20 * N_MAIN + 24 * 6 * N_REC
log = logging.getLogger("e2")


def _mod(name: str) -> Any:
    key = f"e2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class Backend:
    """Builds agent models for ``toy`` or ``hf``."""

    def __init__(self, kind: str, device: str) -> None:
        self.kind, self.device = kind, device
        self.ladder: Sequence[Any] = (
            (("toy", "-", "-"),) if kind == "toy" else LADDER
        )

    def model(self, rung: Sequence[str]) -> Any:
        sysm = _mod("system")
        if self.kind == "toy":
            return sysm.ToyAgentModel()
        return sysm.HFAgentModel(
            rung[0], rung[1], self.device, rung[2]
        )  # pragma: no cover


def _licence(backend: Backend, rung: Sequence[str]) -> Optional[str]:
    """sha256 of the model's LICENSE file at the pinned revision (Addendum 1 item 7)."""
    if backend.kind == "toy":
        return None
    import hashlib  # pragma: no cover - needs the Hub

    from huggingface_hub import hf_hub_download  # pragma: no cover

    path = hf_hub_download(rung[0], "LICENSE", revision=rung[1])  # pragma: no cover
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()  # pragma: no cover


def gate_scores(episodes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Accuracy, and the tool-call parse rate: valid calls / (valid + failed)."""
    acc = sum(e["label"] == "correct" for e in episodes) / len(episodes)
    valid = sum(len(e["calls"]) for e in episodes)
    failed = sum(int(e["parse_failures"]) for e in episodes)
    parse = valid / (valid + failed) if valid + failed else 0.0
    return {"accuracy": round(acc, 4), "parse_rate": round(parse, 4)}


def _gate(backend: Backend, dev: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PROTOCOL §9: baseline with ``refund_fix`` applied, 10 episodes per dev question."""
    sysm = _mod("system")
    cfg = {**sysm.BASE, "refund_fix": "fixed"}
    rungs = []
    for rung in backend.ladder:
        model = backend.model(rung)
        s = sysm.System(model, SEEDS["gate"], BLOCKS["gate"])
        eps = [s.episode(cfg, q, i) for q in dev for i in range(N_GATE)]
        sc = gate_scores(eps)
        ok = sc["accuracy"] >= GATE_MIN and sc["parse_rate"] >= PARSE_MIN
        rungs.append(
            {"model": rung[0], "revision": rung[1], "precision": rung[2],
             "n": len(eps), "passed": ok, "licence_sha256": _licence(backend, rung),
             **sc}
        )
        log.info("gate %s: %s", rung[0], sc)
        if ok:
            return {"rungs": rungs, "selected": rung[0]}
        del model, s
        _free_gpu()
    return {"rungs": rungs, "selected": None}


def pilot_check(
    dev: Sequence[Dict[str, Any]], acc: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Per designed factor: does a dev question of the targeted type move by ≥ 0.20
    in the designed direction (PROTOCOL §9)? ``acc[qid][cell]`` is accuracy."""
    out = {}
    for f, (kinds, sign) in PILOT_TARGETS.items():
        moves = {
            q["id"]: round(acc[q["id"]][f"only_{f}"] - acc[q["id"]]["baseline"], 4)
            for q in dev
            if q["kind"] in kinds
        }
        out[f] = {
            "moves": moves,
            "effect_present": any(sign * m >= PILOT_DROP for m in moves.values()),
        }
    return out


def _pilot(model: Any, dev: List[Dict[str, Any]]) -> Dict[str, Any]:
    sysm = _mod("system")
    s = sysm.System(model, SEEDS["pilot"], BLOCKS["pilot"])
    cells = {"baseline": sysm.BASE, "candidate": sysm.CANDIDATE}
    cells.update({f"only_{f.name}": {**sysm.BASE, f.name: f.candidate}
                  for f in sysm.FACTORS})
    start = time.perf_counter()
    acc: Dict[str, Dict[str, float]] = {}
    for q in dev:
        acc[q["id"]] = {
            name: sum(s.episode(cfg, q, i)["label"] == "correct"
                      for i in range(N_PILOT)) / N_PILOT
            for name, cfg in cells.items()
        }
        s.clear()
    seconds = time.perf_counter() - start
    n_eps = len(dev) * len(cells) * N_PILOT
    per_hour = n_eps / seconds * 3600 if seconds else 0.0
    return {
        "accuracy": acc,
        "check": pilot_check(dev, acc),
        "episodes": n_eps,
        "seconds": round(seconds, 1),
        "episodes_per_hour": round(per_hour, 1),
        "remaining_episodes": REMAINING,
        "projected_hours_remaining": round(REMAINING / per_hour, 2)
        if per_hour else None,
        "note": "projection assumes test episodes take as long as pilot episodes",
    }


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
    model = backend.model(rung)
    systems = {
        k: sysm.System(model, SEEDS[k], BLOCKS[k])
        for k in ("main", "reference", "recovery")
    }
    settings = InterventionSettings(N_MAIN, N_PERM, ALPHA, SEEDS["main"])
    per_q: Dict[str, Any] = {}
    for n, q in enumerate(test, 1):
        log.info("question %d/%d %s: reference, main, recovery", n, len(test), q["id"])
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
                    systems["recovery"],
                    q,
                    m["GB"],
                    m["B2"],
                    m["GB_attributed"],
                    sysm.BASE,
                    sysm.CANDIDATE,
                    N_REC,
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
    regressed = {q: bool(row["reference"]["regressed"]) for q, row in per_q.items()}
    methods = {
        q: {
            **{m: row["methods"][m] for m in sc.METHODS},
            "GB_attributed": row["methods"]["GB_attributed"],
            "GB_fault": row["methods"]["GB_fault"],
        }
        for q, row in per_q.items()
    }
    recovery = {q: row["recovery"] for q, row in per_q.items()}
    return {
        "criteria": sc.criteria(labels, regressed, methods, recovery),
        "descriptive": sc.descriptive(labels, regressed, methods, recovery),
    }


PACKAGES = (
    "glassbox-mech-interp",
    "torch",
    "transformers",
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


def _write_freeze(out: Path) -> None:
    """Installed packages into ``out/pip_freeze.txt`` (uv venv first, then pip)."""
    commands = (
        ["uv", "pip", "freeze", "--python", sys.executable],
        [sys.executable, "-m", "pip", "freeze"],
    )
    for cmd in commands:
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        (out / "pip_freeze.txt").write_text(done.stdout, encoding="utf-8")
        return
    (out / "pip_freeze.txt").write_text(
        "neither `uv pip freeze` nor `pip freeze` was available\n", encoding="utf-8"
    )


def _gpu_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - GPU only
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None


def _environment(backend: Backend) -> Dict[str, Any]:
    if backend.kind == "toy":
        return {"device": "cpu", "backend": "toy (not a language model)"}
    cfg = compute.resolve(compute.ComputeConfig(backend.device))  # pragma: no cover
    return compute.environment_record(cfg)  # pragma: no cover


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run or resume one E2 stage and write ``results.json`` in ``--out``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("pilot", "test"), required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backend", choices=("toy", "hf"), default="hf")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N questions (smoke tests; never scored)",
    )
    p.add_argument(
        "--gate-min",
        type=float,
        default=GATE_MIN,
        help="toy dry runs only; must stay 0.80 for a real run",
    )
    args = p.parse_args(argv)
    if args.backend == "hf" and args.gate_min != GATE_MIN:
        raise SystemExit("--gate-min may only be changed with --backend toy")
    globals()["GATE_MIN"] = args.gate_min
    device = args.device or ("cpu" if args.backend == "toy" else "cuda")
    fingerprint = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "n": [N_MAIN, N_REF, N_REC, N_GATE, N_PILOT],
        "blocks": BLOCKS,
        "gate_min": args.gate_min,
        "seeds": SEEDS,
        "backend": args.backend,
        "limit": args.limit,
        "ladder": [list(r) for r in LADDER],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(args.out / "checkpoint", fingerprint)
    _write_freeze(args.out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sysm = _mod("system")
    backend = Backend(args.backend, device)
    dev = sysm.load_questions("dev")
    gate = cp.run("gate", _timed(lambda: _gate(backend, dev)))
    result: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "gate": gate,
        "limit": args.limit,
        "environment": _environment(backend),
        "gpu": _gpu_name(),
        "software": _software(),
    }
    if gate["selected"] is None:
        result.update({"criteria": {"overall": "INCONCLUSIVE"},
                       "note": "no model passed the competence gate"})
    else:
        rung = next(r for r in backend.ladder if r[0] == gate["selected"])
        if args.stage == "pilot":
            model = backend.model(rung)
            result["pilot"] = cp.run(
                "pilot", lambda: _pilot(model, dev[: args.limit])
            )
            result["criteria"] = {"overall": "PILOT (not scored)"}
        else:
            test = sysm.load_questions("test")[: args.limit]
            per_q = _stages(cp, backend, rung, test)
            result.update({"per_question": per_q, **_score(per_q)})
    (args.out / "results.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"overall: {result['criteria']['overall']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
