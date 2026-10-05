#!/usr/bin/env python3
"""V7 Experiment 3, Part B: real sampling model, GPT-2 small (PROTOCOL.md §4).

Reuses Experiment 1's corpus, retriever (top_k=1) and prompt unchanged. For each of
8 questions it records 20 sampled runs per condition (baseline seeds 0–19,
baseline_more seeds 100–119, fault seeds 200–219; fault = d1–d4 missing). It then
evaluates B1–B4 with ``glassbox.v7.groupdiff`` and the pairwise diff.

Usage (owner's machine; needs torch + transformer_lens + GPT-2 weights):
    python3 experiments/v7/noise_baseline/part_b.py --out experiments/v7/noise_baseline/results/part_b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence

from glassbox.v7.diff import diff_traces
from glassbox.v7.groupdiff import group_diff
from glassbox.v7.trace import Recorder

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "rag_fault"
SEEDS = {
    "baseline": range(0, 20),
    "baseline_more": range(100, 120),
    "fault": range(200, 220),
}
N_PERM = 1999
ALPHA = 0.05
MAX_NEW_TOKENS = 8
TEMPERATURE = 1.0
RETRIEVAL = "retrieval#0"
AFFECTED = ("q1", "q2", "q3", "q4")
UNAFFECTED = ("q5", "q6", "q7", "q8")

#: ``sampler(prompt, seed) -> sampled continuation text``.
Sampler = Callable[[str, int], str]


def load_exp1() -> ModuleType:
    """Load Experiment 1's ``run.py`` by path (corpus, retriever, prompt)."""
    spec = importlib.util.spec_from_file_location("exp1_run", EXP1 / "run.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GPT2Sampler:
    """Seeded multinomial sampling of ``MAX_NEW_TOKENS`` tokens at ``TEMPERATURE``."""

    def __init__(self, model_name: str = "gpt2") -> None:
        import torch
        from transformer_lens import HookedTransformer

        self._torch = torch
        self.model = HookedTransformer.from_pretrained(model_name, device="cpu")
        self.model.eval()

    def __call__(self, prompt: str, seed: int) -> str:
        torch = self._torch
        torch.manual_seed(seed)
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            out = self.model.generate(
                tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=None,
                top_p=None,
                stop_at_eos=False,
                verbose=False,
            )
        return str(self.model.to_string(out[0, tokens.shape[1] :]))


def record_run(
    out_dir: Path, run_id: str, q: Dict[str, str], ctx: Dict[str, Any], seed: int
) -> Path:
    """One sampled run: retrieval span then chat span; returns the trace path.

    ``ctx`` holds ``retriever``, ``docs_by_id``, ``build_prompt``, ``sampler``, ``top_k``.
    """
    with Recorder(out_dir=out_dir, run_id=run_id) as rec:
        with rec.span(
            "retrieval",
            "kb",
            attributes={
                "gen_ai.data_source.id": "kb",
                "gen_ai.retrieval.top_k": ctx["top_k"],
            },
        ) as span:
            hits = ctx["retriever"].search(q["question"], ctx["top_k"])
            rec.add_content(
                span,
                {
                    "gen_ai.retrieval.query.text": q["question"],
                    "gen_ai.retrieval.documents": [
                        {"id": d, "score": s} for d, s in hits
                    ],
                },
            )
        prompt = ctx["build_prompt"](
            [ctx["docs_by_id"][d] for d, _ in hits], q["question"]
        )
        sample = ctx["sampler"](prompt, seed)
        contains = q["correct"].strip().lower() in sample.lower()
        messages = [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
        output = [{"role": "assistant", "parts": [{"type": "text", "content": sample}]}]
        with rec.span(
            "chat",
            "gpt2",
            attributes={
                "gen_ai.request.model": "gpt2",
                "gen_ai.request.temperature": TEMPERATURE,
                "glassbox.answer.contains_correct": contains,
            },
            content={
                "gen_ai.input.messages": messages,
                "gen_ai.output.messages": output,
            },
        ):
            pass
    return rec.path


def trace_paths(out: Path, condition: str, qid: str) -> List[Path]:
    """Trace files of one (condition, question) group, in seed order."""
    return [
        out / condition / qid / f"s{seed:03d}.trace.jsonl" for seed in SEEDS[condition]
    ]


def run_all(out: Path, exp1: ModuleType, sampler: Sampler) -> None:
    """Record every (condition, question, seed) run."""
    docs, questions = exp1.load_corpus(EXP1 / "corpus.json")
    docs_by_id = {d["id"]: d["text"] for d in docs}
    for condition, seeds in SEEDS.items():
        exclude = exp1.FAULT_EXCLUDED if condition == "fault" else frozenset()
        ctx = {
            "retriever": exp1.BowRetriever(docs, exclude=exclude),
            "docs_by_id": docs_by_id,
            "build_prompt": exp1.build_prompt,
            "sampler": sampler,
            "top_k": exp1.TOP_K,
        }
        for q in questions:
            for seed in seeds:
                record_run(out / condition / q["id"], f"s{seed:03d}", q, ctx, seed)


def _contains_rate(paths: List[Path]) -> float:
    from glassbox.v7.trace import read_trace

    vals = []
    for p in paths:
        spans, _ = read_trace(p)
        chat = [
            s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"
        ]
        vals.append(bool(chat[0]["attributes"]["glassbox.answer.contains_correct"]))
    return sum(vals) / len(vals)


def evaluate(out: Path, qids: Sequence[str]) -> Dict[str, Any]:
    """Apply PROTOCOL.md §5 Part B criteria."""
    null, fault = {}, {}
    for q in qids:
        base = trace_paths(out, "baseline", q)
        null[q] = group_diff(
            base, trace_paths(out, "baseline_more", q), N_PERM, ALPHA, 0
        ).to_dict()
        fault[q] = group_diff(
            base, trace_paths(out, "fault", q), N_PERM, ALPHA, 0
        ).to_dict()
    b1 = {q: not null[q]["significant"] for q in qids}
    b2 = {
        q: bool(fault[q]["first_beyond_noise"])
        and fault[q]["first_beyond_noise"]["step"] == RETRIEVAL
        for q in AFFECTED
    }
    b3 = {
        q: not any(f["step"] == RETRIEVAL for f in fault[q]["significant"])
        for q in UNAFFECTED
    }
    pairs = [
        bool(diff_traces(x, y).divergences)
        for q in qids
        for x, y in zip(
            trace_paths(out, "baseline", q), trace_paths(out, "baseline_more", q)
        )
    ]
    rates = {
        c: {q: _contains_rate(trace_paths(out, c, q)) for q in qids} for c in SEEDS
    }
    return {
        "B1_real_null": {
            "per_question": b1,
            "rule": ">= 7/8",
            "pass": sum(b1.values()) >= 7,
        },
        "B2_real_localisation": {
            "per_question": b2,
            "rule": "4/4",
            "pass": all(b2.values()),
        },
        "B3_real_specificity": {
            "per_question": b3,
            "rule": "4/4",
            "pass": all(b3.values()),
        },
        "B4_descriptive": {
            "pairwise_divergence_rate_baseline_vs_more": sum(pairs) / len(pairs),
            "contains_correct_rate": rates,
            "descriptive": True,
        },
        "_reports": {"null": null, "fault": fault},
    }


GATED = ("B1_real_null", "B2_real_localisation", "B3_real_specificity")


def run(out: Path, sampler: Sampler, model_name: str = "gpt2") -> Dict[str, Any]:
    """Record all runs, evaluate, write ``results.json``; never overwrites."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already holds results; results are never overwritten"
        )
    exp1 = load_exp1()
    run_all(out, exp1, sampler)
    qids = [q["id"] for q in exp1.load_corpus(EXP1 / "corpus.json")[1]]
    criteria = evaluate(out, qids)
    results = {
        "protocol": "experiments/v7/noise_baseline/PROTOCOL.md v1.0 (Part B)",
        "model": model_name,
        "settings": {
            "seeds": {c: [s.start, s.stop - 1] for c, s in SEEDS.items()},
            "n_perm": N_PERM,
            "alpha": ALPHA,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "fault_excluded_docs": sorted(exp1.FAULT_EXCLUDED),
        },
        "criteria": criteria,
        "overall": "PASS" if all(criteria[g]["pass"] for g in GATED) else "FAIL",
    }
    with open(out / "results.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; prints the criteria and the overall result."""
    parser = argparse.ArgumentParser(
        description="V7 Experiment 3 Part B (GPT-2 sampling)"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args(argv)
    results = run(args.out, GPT2Sampler(args.model), args.model)
    for name, c in results["criteria"].items():
        if name.startswith("_"):
            continue
        tag = "" if "pass" not in c else ("PASS " if c["pass"] else "FAIL ")
        rest = {k: v for k, v in c.items() if k != "pass"}
        sys.stdout.write(f"{name}: {tag}{json.dumps(rest)}\n")
    sys.stdout.write(f"PART B OVERALL: {results['overall']}\n")
    return 0 if results["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
