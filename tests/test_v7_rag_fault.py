"""Mechanics of V7 Experiment 1 (experiments/v7/rag_fault), with a fake scorer.

The real experiment uses GPT-2 on the owner's machine. These tests check the
pipeline, retriever, fault injection and criteria logic without any model: the fake
scorer prefers an answer iff it literally appears in the prompt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "v7" / "rag_fault"


@pytest.fixture(scope="module")
def rf():
    spec = importlib.util.spec_from_file_location("rag_fault_run", EXP / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus(rf):
    return rf.load_corpus(EXP / "corpus.json")


def fake_scorer(prompt: str, answer: str) -> float:
    """Prefer answers in the top-ranked document, then anywhere in context."""
    context = prompt.split("\nQuestion:")[0].splitlines()[1:]
    if context and answer.strip() in context[0]:
        return 0.0
    return -1.0 if answer.strip() in prompt else -5.0


def test_corpus_is_well_formed(corpus):
    docs, questions = corpus
    ids = {d["id"] for d in docs}
    assert len(docs) == 8 and len(questions) == 8
    for q in questions:
        assert q["doc"] in ids
        text = next(d["text"] for d in docs if d["id"] == q["doc"])
        assert q["correct"].strip() in text
        assert q["distractor"].strip() not in text
        assert any(q["distractor"].strip() in d["text"] for d in docs)


def test_retriever_finds_source_document_first(rf, corpus):
    docs, questions = corpus
    retriever = rf.BowRetriever(docs)
    for q in questions:
        top = retriever.search(q["question"], k=2)
        assert top[0][0] == q["doc"], (q["id"], top)


def test_retriever_is_deterministic_and_respects_exclusion(rf, corpus):
    docs, questions = corpus
    stale = rf.BowRetriever(docs, exclude=rf.FAULT_EXCLUDED)
    q1 = questions[0]
    first = stale.search(q1["question"], k=2)
    assert first == stale.search(q1["question"], k=2)
    assert all(doc_id not in rf.FAULT_EXCLUDED for doc_id, _ in first)


def test_full_pipeline_passes_criteria_with_fake_scorer(rf, corpus, tmp_path):
    results = rf.run_experiment(
        corpus, fake_scorer, tmp_path / "out", model_name="fake"
    )
    crit = results["criteria"]
    assert crit["C1_localisation"]["pass"], crit["C1_localisation"]
    assert crit["C2_specificity"]["pass"], crit["C2_specificity"]
    assert crit["C3_intervention"]["pass"]
    assert crit["C4_determinism"]["pass"]
    assert results["overall"] == "PASS"
    c5 = crit["C5_effect"]
    assert c5["baseline_correct_affected"] == 4
    assert c5["fault_correct_affected"] < 4


def test_traces_are_written_per_condition_and_question(rf, corpus, tmp_path):
    rf.run_experiment(corpus, fake_scorer, tmp_path / "out", model_name="fake")
    for cond in rf.CONDITIONS:
        files = sorted((tmp_path / "out" / cond).glob("*.trace.jsonl"))
        assert [f.name.split(".")[0] for f in files] == [f"q{i}" for i in range(1, 9)]


def test_results_contain_no_raw_text(rf, corpus, tmp_path):
    rf.run_experiment(corpus, fake_scorer, tmp_path / "out", model_name="fake")
    raw = (tmp_path / "out" / "results.json").read_text(encoding="utf-8")
    assert "Kestrel" not in raw
    json.loads(raw)
    traces = "".join(
        p.read_text(encoding="utf-8") for p in (tmp_path / "out").rglob("*.jsonl")
    )
    assert "Kestrel" not in traces and "Brindle" not in traces


def test_criteria_fail_when_fault_is_invisible(rf, corpus, tmp_path, monkeypatch):
    """If the fault did nothing, C1 must FAIL (no false pass)."""
    monkeypatch.setattr(rf, "FAULT_EXCLUDED", frozenset())
    results = rf.run_experiment(corpus, fake_scorer, tmp_path / "o", model_name="fake")
    assert not results["criteria"]["C1_localisation"]["pass"]
    assert results["overall"] == "FAIL"


def test_refuses_to_overwrite_results(rf, corpus, tmp_path):
    rf.run_experiment(corpus, fake_scorer, tmp_path / "out", model_name="fake")
    with pytest.raises(FileExistsError):
        rf.run_experiment(corpus, fake_scorer, tmp_path / "out", model_name="fake")


def test_design_lets_c2_test_silence(rf, corpus):
    """Unaffected questions must retrieve the same documents under the fault,
    otherwise C2 could never test that the diff stays silent (PROTOCOL §2 note)."""
    docs, questions = corpus
    full = rf.BowRetriever(docs)
    stale = rf.BowRetriever(docs, exclude=rf.FAULT_EXCLUDED)
    for q in questions:
        a = full.search(q["question"], rf.TOP_K)
        b = stale.search(q["question"], rf.TOP_K)
        if q["id"] in rf.UNAFFECTED:
            assert a == b, q["id"]
        else:
            assert a != b, q["id"]


def test_c2_silence_branch_is_exercised(rf, corpus, tmp_path):
    results = rf.run_experiment(
        corpus, fake_scorer, tmp_path / "out", model_name="fake"
    )
    per_q = results["criteria"]["C2_specificity"]["per_question"]
    assert all(v["retrieval_unchanged"] for v in per_q.values())
