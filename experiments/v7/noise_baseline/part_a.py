#!/usr/bin/env python3
"""V7 Experiment 3, Part A: synthetic pipeline with known ground truth (PROTOCOL.md §3).

For each world (W0–W3) and each of R=200 replications, writes N=30 baseline and
M=30 candidate traces of a seeded three-step stochastic pipeline to a temporary
directory, runs ``glassbox.v7.groupdiff.group_diff`` on them (and, in W0, the
pairwise diff on pairs), keeps a small summary, and deletes the traces. The
pre-declared criteria A1–A6 are evaluated and written to ``results.json``.

Usage (stdlib + glassbox only; no model needed):
    python3 experiments/v7/noise_baseline/part_a.py --out experiments/v7/noise_baseline/results/part_a
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from glassbox.v7.diff import diff_traces
from glassbox.v7.groupdiff import group_diff
from glassbox.v7.trace import Recorder

MASTER_SEED = 20261005
N_PER_GROUP = 30
REPLICATIONS = 200
N_PERM = 999
ALPHA = 0.05
WORLDS = ("W0", "W1", "W2", "W3")
RETRIEVAL = "retrieval#0"
CHAT = "chat#0"
OUTPUT_HASH = "glassbox.content.gen_ai.output.messages.sha256"


@dataclass(frozen=True)
class Params:
    """Pipeline parameters: P(right doc), P(correct | right), P(correct | wrong)."""

    r: float
    c_r: float
    c_w: float


BASELINE = Params(r=0.90, c_r=0.85, c_w=0.20)
CANDIDATE = {
    "W0": BASELINE,
    "W1": Params(r=0.90, c_r=0.35, c_w=0.20),
    "W2": Params(r=0.40, c_r=0.85, c_w=0.20),
    "W3": Params(r=0.40, c_r=0.35, c_w=0.20),
}


def simulate_run(rng: random.Random, p: Params, out_dir: Path, run_id: str) -> Path:
    """Write one trace of the retrieval → chat → (tool) pipeline; return its path."""
    doc = "right" if rng.random() < p.r else "wrong"
    p_correct = p.c_r if doc == "right" else p.c_w
    label = "correct" if rng.random() < p_correct else "distractor"
    nonce = f"{rng.getrandbits(64):016x}"
    with Recorder(out_dir=out_dir, run_id=run_id) as rec:
        with rec.span("retrieval", "kb", kind="INTERNAL") as s:
            rec.add_content(s, {"gen_ai.retrieval.documents": [doc]})
        with rec.span(
            "chat", "synthetic", attributes={"glassbox.answer.label": label}
        ) as s:
            rec.add_content(s, {"gen_ai.output.messages": f"answer {nonce}"})
        if label == "correct":
            with rec.span(
                "execute_tool",
                "send_confirmation",
                kind="INTERNAL",
                attributes={"gen_ai.tool.name": "send_confirmation"},
            ):
                pass
    return rec.path


def _group(seed: str, p: Params, out_dir: Path, prefix: str, n: int) -> List[Path]:
    rng = random.Random(seed)
    return [simulate_run(rng, p, out_dir, f"{prefix}{i:02d}") for i in range(n)]


def replicate(
    world: str, rep: int, n: int = N_PER_GROUP, n_perm: int = N_PERM
) -> Dict[str, Any]:
    """Run one replication of one world; return its summary (traces are deleted)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        a = _group(f"{MASTER_SEED}:{world}:{rep}:A", BASELINE, out, "a", n)
        b = _group(f"{MASTER_SEED}:{world}:{rep}:B", CANDIDATE[world], out, "b", n)
        report = group_diff(a, b, n_perm=n_perm, alpha=ALPHA, seed=rep)
        pairwise: Optional[float] = None
        if world == "W0":
            hits = sum(bool(diff_traces(x, y).divergences) for x, y in zip(a, b))
            pairwise = hits / n
    return summarise(report.to_dict(), pairwise)


def summarise(report: Dict[str, Any], pairwise: Optional[float]) -> Dict[str, Any]:
    """Reduce a group-diff report to what the criteria need."""
    first = report["first_beyond_noise"]
    sig_steps = sorted({f["step"] for f in report["significant"]})
    output_class = [
        f["classification"]
        for f in report["fields"]
        if f["step"] == CHAT and f["field"] == OUTPUT_HASH
    ]
    return {
        "any_significant": bool(report["significant"]),
        "first": [first["step"], first["field"]] if first else None,
        "significant_steps": sig_steps,
        "output_hash_classification": output_class[0] if output_class else None,
        "pairwise_divergence_rate": pairwise,
    }


def _rate(xs: Sequence[bool]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Apply PROTOCOL.md §5 Part A criteria to per-world replication summaries."""

    def first_at(world: str, step: str) -> float:
        return _rate(
            [bool(s["first"]) and s["first"][0] == step for s in summaries[world]]
        )

    def flagged(world: str, step: str) -> float:
        return _rate([step in s["significant_steps"] for s in summaries[world]])

    a1 = _rate([s["any_significant"] for s in summaries["W0"]])
    a2, a2b, a3 = (
        first_at("W1", CHAT),
        flagged("W1", RETRIEVAL),
        first_at("W2", RETRIEVAL),
    )
    a4 = all(
        s["output_hash_classification"] == "untestable"
        for w in WORLDS
        for s in summaries[w]
    )
    pw = [s["pairwise_divergence_rate"] for s in summaries["W0"]]
    return {
        "A1_type_I": {"rate": a1, "rule": "<= 0.08", "pass": a1 <= 0.08},
        "A2_chat_localisation": {"rate": a2, "rule": ">= 0.80", "pass": a2 >= 0.80},
        "A2b_no_false_upstream": {"rate": a2b, "rule": "<= 0.08", "pass": a2b <= 0.08},
        "A3_retrieval_localisation": {
            "rate": a3,
            "rule": ">= 0.80",
            "pass": a3 >= 0.80,
        },
        "A4_free_text_untestable": {"all_untestable": a4, "pass": a4},
        "A5_one_vs_two_causes": {
            "chat_flagged_W2": flagged("W2", CHAT),
            "chat_flagged_W3": flagged("W3", CHAT),
            "descriptive": True,
        },
        "A6_pairwise_noise": {
            "mean_pairwise_divergence_rate_W0": sum(pw) / len(pw) if pw else None,
            "descriptive": True,
        },
    }


GATED = (
    "A1_type_I",
    "A2_chat_localisation",
    "A2b_no_false_upstream",
    "A3_retrieval_localisation",
    "A4_free_text_untestable",
)


def overall(criteria: Dict[str, Any]) -> str:
    """PASS iff every gated Part A criterion passes."""
    return "PASS" if all(criteria[g]["pass"] for g in GATED) else "FAIL"


def run(out: Path, reps: int = REPLICATIONS) -> Dict[str, Any]:
    """Run all worlds and replications; write ``results.json`` (never overwrites)."""
    out.mkdir(parents=True, exist_ok=True)
    target = out / "results.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; results are never overwritten")
    summaries = {w: [replicate(w, i) for i in range(reps)] for w in WORLDS}
    criteria = evaluate(summaries)
    results = {
        "protocol": "experiments/v7/noise_baseline/PROTOCOL.md v1.0 (Part A)",
        "settings": {
            "master_seed": MASTER_SEED,
            "n_per_group": N_PER_GROUP,
            "replications": reps,
            "n_perm": N_PERM,
            "alpha": ALPHA,
            "baseline": BASELINE.__dict__,
            "candidates": {w: p.__dict__ for w, p in CANDIDATE.items()},
            "traces_written": reps * len(WORLDS) * 2 * N_PER_GROUP,
            "traces_committed": False,
        },
        "criteria": criteria,
        "overall": overall(criteria),
        "summaries": summaries,
    }
    with open(target, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; prints the criteria and the overall result."""
    parser = argparse.ArgumentParser(description="V7 Experiment 3 Part A (synthetic)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    results = run(args.out)
    for name, c in results["criteria"].items():
        tag = "" if "pass" not in c else ("PASS " if c["pass"] else "FAIL ")
        rest = {k: v for k, v in c.items() if k != "pass"}
        sys.stdout.write(f"{name}: {tag}{json.dumps(rest)}\n")
    sys.stdout.write(f"PART A OVERALL: {results['overall']}\n")
    return 0 if results["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
