#!/usr/bin/env python3
"""V7 Experiment 1: controlled RAG fault (see PROTOCOL.md in this directory).

Runs four conditions (baseline, baseline_repeat, fault, restored) over eight
fictional questions, records one Glassbox trace per (condition, question), and
evaluates the protocol's pre-declared criteria C1–C5 with ``glassbox.v7.diff``.

Usage (owner's machine; needs transformer_lens + GPT-2 weights):
    python3 experiments/v7/rag_fault/run.py --out experiments/v7/rag_fault/results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Sequence, Tuple

from glassbox.v7.diff import diff_traces
from glassbox.v7.trace import Recorder

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TOP_K = 1
CONDITIONS = ("baseline", "baseline_repeat", "fault", "restored")
FAULT_EXCLUDED: FrozenSet[str] = frozenset({"d1", "d2", "d3", "d4"})
AFFECTED = ("q1", "q2", "q3", "q4")
UNAFFECTED = ("q5", "q6", "q7", "q8")
STOPWORDS = frozenset(
    "a an the of in on at by for from is was which who what where when to every".split()
)

Scorer = Callable[[str, str], float]
Corpus = Tuple[List[Dict[str, str]], List[Dict[str, str]]]


def load_corpus(path: Path) -> Corpus:
    """Return ``(documents, questions)`` from the corpus JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["documents"], data["questions"]


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens without stopwords."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BowRetriever:
    """Deterministic bag-of-words cosine retriever (ties broken by document id)."""

    def __init__(
        self, docs: Sequence[Dict[str, str]], exclude: Iterable[str] = ()
    ) -> None:
        excluded = set(exclude)
        self.docs = [d for d in docs if d["id"] not in excluded]
        self._vecs = {d["id"]: Counter(tokenize(d["text"])) for d in self.docs}

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:  # type: ignore[type-arg]
        dot = sum(a[t] * b[t] for t in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Top-``k`` ``(doc_id, score)`` pairs, score rounded to 6 decimals."""
        q = Counter(tokenize(query))
        scored = [(d, round(self._cosine(q, v), 6)) for d, v in self._vecs.items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def build_prompt(context: Sequence[str], question: str) -> str:
    """Prompt: retrieved documents, then the question, then ``Answer:``."""
    return "Context:\n" + "\n".join(context) + f"\nQuestion: {question}\nAnswer:"


class GPT2Scorer:
    """Teacher-forced log-probability of an answer continuation under GPT-2."""

    def __init__(self, model_name: str = "gpt2") -> None:
        import torch
        from transformer_lens import HookedTransformer

        self._torch = torch
        self.model = HookedTransformer.from_pretrained(model_name, device="cpu")
        self.model.eval()

    def __call__(self, prompt: str, answer: str) -> float:
        torch = self._torch
        p_tok = self.model.to_tokens(prompt, prepend_bos=True)
        a_tok = self.model.to_tokens(answer, prepend_bos=False)
        tokens = torch.cat([p_tok, a_tok], dim=1)
        with torch.no_grad():
            logp = torch.log_softmax(self.model(tokens)[0].double(), dim=-1)
        n = p_tok.shape[1]
        return float(sum(logp[n - 1 + i, a_tok[0, i]] for i in range(a_tok.shape[1])))


def _ask(
    rec: Recorder,
    retriever: BowRetriever,
    q: Dict[str, str],
    docs_by_id: Dict[str, str],
    scorer: Scorer,
    model_name: str,
) -> Dict[str, Any]:  # noqa: E501
    """Retrieve, build prompt, score both answers; record spans; return outcome."""
    with rec.span(
        "retrieval",
        "kb",
        attributes={"gen_ai.data_source.id": "kb", "gen_ai.retrieval.top_k": TOP_K},
    ) as span:
        hits = retriever.search(q["question"], TOP_K)
        rec.add_content(
            span,
            {
                "gen_ai.retrieval.query.text": q["question"],
                "gen_ai.retrieval.documents": [{"id": d, "score": s} for d, s in hits],
            },
        )
    prompt = build_prompt([docs_by_id[d] for d, _ in hits], q["question"])
    margin = scorer(prompt, q["correct"]) - scorer(prompt, q["distractor"])
    preferred = q["correct"] if margin > 0 else q["distractor"]
    messages = [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
    output = [{"role": "assistant", "parts": [{"type": "text", "content": preferred}]}]
    with rec.span(
        "chat",
        model_name,
        attributes={
            "gen_ai.request.model": model_name,
            "gen_ai.provider.name": "local",
        },
        content={"gen_ai.input.messages": messages, "gen_ai.output.messages": output},
    ):
        pass
    return {
        "retrieved": [d for d, _ in hits],
        "margin": round(margin, 6),
        "correct": margin > 0,
    }


def run_condition(
    name: str, corpus: Corpus, scorer: Scorer, out: Path, model_name: str
) -> Dict[str, Dict[str, Any]]:  # noqa: E501
    """Run all questions for one condition; one trace per question."""
    docs, questions = corpus
    exclude = FAULT_EXCLUDED if name == "fault" else frozenset()
    retriever = BowRetriever(docs, exclude=exclude)
    docs_by_id = {d["id"]: d["text"] for d in docs}
    outcomes = {}
    for q in questions:
        with Recorder(out_dir=out / name, run_id=q["id"], label=name) as rec:
            outcomes[q["id"]] = _ask(rec, retriever, q, docs_by_id, scorer, model_name)
    return outcomes


def _first_is_retrieval(path_a: Path, path_b: Path) -> bool:
    first = diff_traces(path_a, path_b).first_divergence
    return first is not None and (first.name_a or "").startswith("retrieval")


def _no_divergence(path_a: Path, path_b: Path) -> bool:
    return not diff_traces(path_a, path_b).divergences


def evaluate(
    out: Path, outcomes: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Apply PROTOCOL.md §4 criteria; returns per-criterion details and pass flags."""
    t = {c: {q: out / c / f"{q}.trace.jsonl" for q in outcomes[c]} for c in outcomes}
    c1 = {q: _first_is_retrieval(t["baseline"][q], t["fault"][q]) for q in AFFECTED}
    c2 = {}
    for q in UNAFFECTED:
        same = outcomes["baseline"][q]["retrieved"] == outcomes["fault"][q]["retrieved"]
        ok = (
            _no_divergence(t["baseline"][q], t["fault"][q])
            if same
            else _first_is_retrieval(t["baseline"][q], t["fault"][q])
        )
        c2[q] = {"retrieval_unchanged": same, "pass": ok}
    qs = list(outcomes["baseline"])
    c3 = {
        q: _no_divergence(t["baseline"][q], t["restored"][q])
        and outcomes["baseline"][q]["correct"] == outcomes["restored"][q]["correct"]
        for q in qs
    }
    c4 = {q: _no_divergence(t["baseline"][q], t["baseline_repeat"][q]) for q in qs}
    crit = {
        "C1_localisation": {"per_question": c1, "pass": all(c1.values())},
        "C2_specificity": {
            "per_question": c2,
            "pass": all(v["pass"] for v in c2.values()),
        },
        "C3_intervention": {"per_question": c3, "pass": all(c3.values())},
        "C4_determinism": {"per_question": c4, "pass": all(c4.values())},
        "C5_effect": {
            "baseline_correct_affected": sum(
                outcomes["baseline"][q]["correct"] for q in AFFECTED
            ),
            "fault_correct_affected": sum(
                outcomes["fault"][q]["correct"] for q in AFFECTED
            ),
            "n_affected": len(AFFECTED),
        },
    }
    return crit


def run_experiment(
    corpus: Corpus, scorer: Scorer, out: Path, model_name: str
) -> Dict[str, Any]:  # noqa: E501
    """Run all conditions, evaluate, write ``results.json``; never overwrites."""
    out = Path(out)
    if (out / "results.json").exists() or any((out / c).exists() for c in CONDITIONS):
        raise FileExistsError(
            f"{out} already holds results; results are never overwritten"
        )
    outcomes = {
        c: run_condition(c, corpus, scorer, out, model_name) for c in CONDITIONS
    }
    crit = evaluate(out, outcomes)
    gated = ("C1_localisation", "C2_specificity", "C3_intervention", "C4_determinism")
    results = {
        "protocol": "experiments/v7/rag_fault/PROTOCOL.md v1.0",
        "model": model_name,
        "fault_excluded_docs": sorted(FAULT_EXCLUDED),
        "outcomes": outcomes,
        "criteria": crit,
        "overall": "PASS" if all(crit[g]["pass"] for g in gated) else "FAIL",
    }
    (out / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("overall %s", results["overall"])
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run with GPT-2 and print the criteria summary."""
    parser = argparse.ArgumentParser(
        description="V7 Experiment 1: controlled RAG fault"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args(argv)
    corpus = load_corpus(HERE / "corpus.json")
    results = run_experiment(corpus, GPT2Scorer(args.model), args.out, args.model)
    for name, c in results["criteria"].items():
        status = "" if "pass" not in c else ("PASS" if c["pass"] else "FAIL")
        sys.stdout.write(f"{name}: {status} {json.dumps(c.get('per_question', c))}\n")
    sys.stdout.write(f"OVERALL: {results['overall']}\n")
    return 0 if results["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
