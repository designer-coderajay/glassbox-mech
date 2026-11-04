"""V8 phase E1 golden RAG case (experiments/v8/e1_golden_rag): data, pipeline, methods,
reference labels, recovery rule, scoring and the staged runner. Toy embedder and toy
generator only, so no GPU or model download is needed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from glassbox.v7.intervene import InterventionSettings
from glassbox.v7.trace import read_trace

EXP = Path(__file__).resolve().parent.parent / "experiments" / "v8" / "e1_golden_rag"


def _load(name):
    key = f"e1_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, EXP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sysm():
    return _load("system")


@pytest.fixture(scope="module")
def toy(sysm):
    return sysm.System(sysm.ToyEmbedder(), sysm.ToyGenerator(), master="test")


# --- data (PROTOCOL §2) -------------------------------------------------------------


def test_corpus_and_splits_match_protocol(sysm):
    corpus = sysm.load_corpus()
    assert len(corpus) == 24
    assert sum(d["v1"] != d["v2"] for d in corpus) == 8
    dev, test = sysm.load_questions("dev"), sysm.load_questions("test")
    assert len(dev) == 8 and len(test) == 24
    for qs, n in ((dev, 4), (test, 12)):
        assert sum(q["kind"] == "updated" for q in qs) == n
    ids = [q["id"] for q in dev + test]
    assert len(ids) == len(set(ids))


def test_gold_answers_are_in_the_right_document_versions(sysm):
    corpus = {d["id"]: d for d in sysm.load_corpus()}
    for q in sysm.load_questions("dev") + sysm.load_questions("test"):
        doc = corpus[q["doc"]]
        assert sysm.matches(doc["v2"], q["accepted"]), q["id"]
        if q["kind"] == "updated":
            assert not sysm.matches(doc["v1"], q["accepted"]), q["id"]
            assert sysm.matches(doc["v1"], q["stale"]), q["id"]


def test_data_files_are_regenerated_identically(tmp_path, monkeypatch):
    make = _load("make_data")
    before = {p.name: p.read_bytes() for p in EXP.glob("*.json")}
    monkeypatch.setattr(make, "HERE", tmp_path)
    make.main()
    for name, data in before.items():
        if name in ("corpus.json", "questions_dev.json", "questions_test.json"):
            assert (tmp_path / name).read_bytes() == data, name


# --- matching ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out,ok",
    [
        ("You have 30 days.", True),
        ("Thirty days from delivery.", True),
        ("Within 300 days", False),
        ("14 days", False),
    ],
)
def test_matching_is_token_based(sysm, out, ok):
    assert sysm.matches(out, ["30 days", "thirty days"]) is ok


def test_matching_handles_euro_percent_and_time(sysm):
    assert sysm.matches("It is free above €35.", ["€35"])
    assert not sysm.matches("€350", ["€35"])
    assert sysm.matches("a 15% discount", ["15%"])
    assert sysm.matches("open until 20:00", ["20:00"])


# --- pipeline ------------------------------------------------------------------------


def test_chunking(sysm):
    text = " ".join(f"w{i}" for i in range(70))
    assert [len(c.split()) for c in sysm.chunk(text, 40)] == [40, 30]
    assert sysm.chunk(text, 120) == [text]


def test_stale_index_serves_old_text(sysm, toy):
    q = next(q for q in sysm.load_questions("dev") if q["id"] == "u_returns_a")
    fresh = toy.retrieve({**sysm.BASE, "index": "fresh"}, q["question"])
    stale = toy.retrieve({**sysm.BASE, "index": "stale"}, q["question"])
    assert any("30 days" in c["text"] for c in fresh)
    assert any("14 days" in c["text"] for c in stale)
    assert len(fresh) == sysm.TOP_K


def test_factors_match_protocol(sysm):
    assert [(f.name, f.base, f.candidate) for f in sysm.FACTORS] == [
        ("index", "fresh", "stale"),
        ("chunk_words", 120, 40),
        ("template", "T1", "T2"),
        ("log_level", "info", "debug"),
    ]
    assert sysm.TEMPLATES["T2"].startswith(sysm.TEMPLATES["T1"])
    assert "concisely in one sentence" in sysm.TEMPLATES["T2"]


def test_answer_is_deterministic_and_log_level_is_never_read(sysm, toy):
    q = sysm.load_questions("dev")[0]
    a = toy.answer(sysm.BASE, q, 3)
    assert a == toy.answer(sysm.BASE, q, 3)
    calls = []
    toy_spy = sysm.System(
        sysm.ToyEmbedder(), sysm.SpyGenerator(sysm.ToyGenerator(), calls), "test"
    )
    toy_spy.answer({**sysm.BASE, "log_level": "debug"}, q, 0)
    assert "debug" not in json.dumps(calls)  # the inert factor never reaches the model


def test_blocks_are_independent_per_configuration(sysm, toy):
    q = sysm.load_questions("dev")[0]
    s1 = toy.block_seed(sysm.BASE, q["id"], 0)
    s2 = toy.block_seed({**sysm.BASE, "log_level": "debug"}, q["id"], 0)
    assert s1 != s2 and s1 == toy.block_seed(dict(sysm.BASE), q["id"], 0)


def test_runner_writes_trace_with_outcome(sysm, toy, tmp_path):
    q = sysm.load_questions("dev")[0]
    path = sysm.make_runner(toy, q)(dict(sysm.BASE), 0, tmp_path, "r002")
    spans, _ = read_trace(path)
    ops = [s["attributes"].get("gen_ai.operation.name") for s in spans]
    assert [o for o in ops if o] == ["retrieval", "chat"]
    assert sysm.outcome(Path(path)) in ("correct", "wrong")


# --- methods -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(sysm, toy, tmp_path_factory):
    meth = _load("methods")
    qs = sysm.load_questions("dev")[:2]
    settings = InterventionSettings(6, 99, 0.05, "t")
    return meth.run_methods(toy, qs, tmp_path_factory.mktemp("m"), settings)


def test_methods_output_shape(small_run, sysm):
    for qid, row in small_run.items():
        assert set(row) >= {"GB", "B0", "B1", "B2", "B2_flips", "cells"}
        assert row["B0"] == [f.name for f in sysm.FACTORS]
        for m in ("GB", "B1", "B2"):
            assert set(row[m]) <= {f.name for f in sysm.FACTORS}
        assert 0 <= row["B2_flips"] <= 5


def test_b1_maps_first_divergence_to_owners():
    meth = _load("methods")
    assert meth.B1_OWNERS["retrieval"] == ("index", "chunk_words")
    assert meth.B1_OWNERS["chat"] == ("template",)


def test_b2_rule():
    meth = _load("methods")
    cand, reverted = "wrong", {"index": "correct", "template": "wrong"}
    assert meth.b2_blame(cand, reverted) == ["index"]
    assert meth.b2_blame("correct", reverted) == []


# --- reference labels (PROTOCOL §4) -----------------------------------------------------


def test_reference_labels():
    ref = _load("reference")
    assert ref.label([1] * 200, [0] * 200, "index", seed="s") == "large"
    hi_acc = [1] * 180 + [0] * 20  # n = 200 at 90%: a null is reachable
    assert ref.label(hi_acc, list(hi_acc), "template", "s") == "null"
    half = [1] * 100 + [0] * 100  # at 50% the CI is too wide for null at n = 200
    assert ref.label(half, list(half), "template", "s") == "small"
    assert ref.label([1] * 200, [0] * 200, "log_level", seed="s") == "null"
    mid = ref.label([1] * 120 + [0] * 80, [1] * 100 + [0] * 100, "template", "s")
    assert mid == "small"


def test_bootstrap_ci_is_deterministic():
    ref = _load("reference")
    a, b = [1] * 150 + [0] * 50, [1] * 100 + [0] * 100
    assert ref.tvd_ci(a, b, "x") == ref.tvd_ci(a, b, "x")
    lo, hi = ref.tvd_ci(a, b, "x")
    assert lo < 0.25 < hi


# --- recovery rule (PROTOCOL §6) -----------------------------------------------------


def test_recovery_rule():
    rec = _load("recovery")
    base, cand = [1] * 55 + [0] * 5, [1] * 10 + [0] * 50
    assert rec.verified(fixed=[1] * 54 + [0] * 6, cand=cand, base=base, seed="s")
    assert not rec.verified(fixed=cand, cand=cand, base=base, seed="s")
    assert not rec.verified(fixed=[1] * 40 + [0] * 20, cand=cand, base=base, seed="s")


def test_fix_reverts_only_attributed(sysm):
    rec = _load("recovery")
    cand = {f.name: f.candidate for f in sysm.FACTORS}
    fixed = rec.fix_config(cand, ["index"])
    assert fixed["index"] == "fresh" and fixed["chunk_words"] == 40
    assert rec.wrong_fix_config(cand)["template"] == "T1"
    assert rec.wrong_fix_config(cand)["index"] == "stale"


# --- scoring (PROTOCOL §7) -------------------------------------------------------------


def _labels():
    return {
        "q1": {
            "index": "large",
            "chunk_words": "null",
            "template": "null",
            "log_level": "null",
        },
        "q2": {
            "index": "large",
            "chunk_words": "small",
            "template": "null",
            "log_level": "null",
        },
    }


def test_scoring_gates():
    sc = _load("score")
    methods = {
        "q1": {
            "GB": ["index"],
            "B0": ["index", "chunk_words", "template", "log_level"],
            "B1": ["index", "chunk_words"],
            "B2": ["index"],
        },
        "q2": {
            "GB": ["index"],
            "B0": ["index", "chunk_words", "template", "log_level"],
            "B1": ["index", "chunk_words"],
            "B2": [],
        },
    }
    rec = {
        "q1": {"fixed": True, "wrong_fix": False},
        "q2": {"fixed": True, "wrong_fix": False},
    }
    out = sc.criteria(_labels(), methods, rec)
    assert out["G1"]["value"] == 1.0 and out["G2"]["value"] == 0
    assert out["G3"]["testable"] is False  # 2 null template pairs < 4
    assert out["G6"]["f1"]["GB"] == 1.0 and out["G6"]["pass"] is True
    assert out["overall"] == "INCONCLUSIVE"  # only 2 large pairs (< 6)


def test_g3_not_testable_with_few_null_template_pairs():
    sc = _load("score")
    labels = {
        "q1": {
            "index": "large",
            "chunk_words": "null",
            "template": "large",
            "log_level": "null",
        }
    }
    m = {"q1": {"GB": ["index"], "B0": [], "B1": [], "B2": []}}
    out = sc.criteria(labels, m, {"q1": {"fixed": True, "wrong_fix": True}})
    assert out["G3"]["testable"] is False


def test_wilson_interval():
    sc = _load("score")
    lo, hi = sc.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97


# --- staged runner ---------------------------------------------------------------------


def test_run_end_to_end_with_toy_backends(tmp_path, monkeypatch):
    run = _load("run")
    monkeypatch.setattr(run, "N_MAIN", 6)
    monkeypatch.setattr(run, "N_REF", 20)
    monkeypatch.setattr(run, "N_REC", 10)
    monkeypatch.setattr(run, "N_GATE", 2)
    monkeypatch.setattr(run, "GATE_MIN", 0.0)
    out = tmp_path / "out"
    assert run.main(["--out", str(out), "--backend", "toy", "--limit", "2"]) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["protocol"] == "experiments/v8/e1_golden_rag/PROTOCOL.md v1.0"
    assert res["gate"]["selected"] == "toy"
    assert "sentence-transformers" in res["software"] and "git_commit" in res["software"]
    assert set(res["per_question"]) and "criteria" in res
    assert (out / "checkpoint" / "_fingerprint.json").exists()
    # resume: a second call with the same settings reuses every saved unit
    assert run.main(["--out", str(out), "--backend", "toy", "--limit", "2"]) == 0


def test_run_refuses_changed_settings_on_resume(tmp_path, monkeypatch):
    run = _load("run")
    for k, v in (
        ("N_MAIN", 6),
        ("N_REF", 20),
        ("N_REC", 10),
        ("N_GATE", 2),
        ("GATE_MIN", 0.0),
    ):
        monkeypatch.setattr(run, k, v)
    out = tmp_path / "out"
    run.main(["--out", str(out), "--backend", "toy", "--limit", "1"])
    monkeypatch.setattr(run, "N_MAIN", 7)
    with pytest.raises(ValueError, match="different settings"):
        run.main(["--out", str(out), "--backend", "toy", "--limit", "1"])


def test_settings_match_protocol():
    run = _load("run")
    assert (run.N_MAIN, run.N_REF, run.N_REC, run.N_GATE) == (30, 200, 60, 10)
    assert run.GATE_MIN == 0.80
    assert [r[0] for r in run.LADDER] == [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    assert run.SEEDS == {
        "main": "e1-main",
        "reference": "e1-reference",
        "recovery": "e1-recovery",
        "gate": "e1-gate",
    }
