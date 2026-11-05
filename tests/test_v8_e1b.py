"""V8 phase E1b (experiments/v8/e1b_directed_rag): new data, directed reference
labels, GB's harmful-list blame, the hurt-only fix with a one-sided recovery check,
scoring G1–G7 and the staged runner. Toy embedder and generator only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from glassbox.v7.intervene import InterventionSettings

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "v8" / "e1b_directed_rag"
E1 = ROOT / "experiments" / "v8" / "e1_golden_rag"


def _load(name):
    key = f"e1b_{name}"
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


def test_no_document_or_question_reused_from_e1(sysm):
    e1_docs = json.loads((E1 / "corpus.json").read_text())
    e1_text = {d["v1"] for d in e1_docs} | {d["v2"] for d in e1_docs}
    e1_ids = {d["id"] for d in e1_docs}
    for d in sysm.load_corpus():
        assert d["id"] not in e1_ids
        assert d["v1"] not in e1_text and d["v2"] not in e1_text
        assert "Kestrel" not in d["v1"] + d["v2"]
    e1_q = {
        q["question"]
        for s in ("dev", "test")
        for q in json.loads((E1 / f"questions_{s}.json").read_text())
    }
    for s in ("dev", "test"):
        assert not {q["question"] for q in sysm.load_questions(s)} & e1_q


def test_gold_answers_are_in_the_right_document_versions(sysm):
    corpus = {d["id"]: d for d in sysm.load_corpus()}
    for q in sysm.load_questions("dev") + sysm.load_questions("test"):
        doc = corpus[q["doc"]]
        assert sysm.matches(doc["v2"], q["accepted"]), q["id"]
        if q["kind"] == "updated":
            assert not sysm.matches(doc["v1"], q["accepted"]), q["id"]
            assert sysm.matches(doc["v1"], q["stale"]), q["id"]
            assert not sysm.matches(doc["v2"], q["stale"]), q["id"]


def test_documents_are_short_and_fact_is_at_the_end(sysm):
    for d in sysm.load_corpus():
        n = len(d["v2"].split())
        assert 40 <= n <= 70, (d["id"], n)


def test_data_files_are_regenerated_identically(tmp_path, monkeypatch):
    make = _load("make_data")
    before = {
        n: (EXP / n).read_bytes()
        for n in ("corpus.json", "questions_dev.json", "questions_test.json")
    }
    monkeypatch.setattr(make, "HERE", tmp_path)
    make.main()
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data, name


# --- pipeline is E1's (PROTOCOL §2) --------------------------------------------------


def test_system_is_e1_pipeline_plus_two_additions(sysm):
    e1 = (E1 / "system.py").read_text()
    e1b = (EXP / "system.py").read_text()
    for marker in (
        "TOP_K = 3",
        "BLOCK = 50",
        "MAX_NEW_TOKENS = 48",
        "TEMPERATURE = 0.7",
    ):
        assert marker in e1 and marker in e1b
    assert (
        e1.split("TEMPLATES = {")[1].split("}")[0]
        == e1b.split("TEMPLATES = {")[1].split("}")[0]
    )
    assert [(f.name, f.base, f.candidate) for f in sysm.FACTORS] == [
        ("index", "fresh", "stale"),
        ("chunk_words", 120, 40),
        ("template", "T1", "T2"),
        ("log_level", "info", "debug"),
    ]


def test_incident_names_target(sysm, toy):
    inc = sysm.make_incident(toy, sysm.load_questions("dev")[:1])
    assert inc.target == "correct"


def test_retrieval_truth(sysm, toy):
    q = next(q for q in sysm.load_questions("dev") if q["id"] == "u_exit_fee_a")
    truth = sysm.retrieval_truth(
        toy, q, {"baseline": sysm.BASE, "stale": {**sysm.BASE, "index": "stale"}}
    )
    assert truth == {"baseline": True, "stale": False}


# --- reference labels (PROTOCOL §4) -------------------------------------------------


def test_directed_labels():
    ref = _load("reference")
    assert ref.label([0] * 200, [1] * 200, "index", "s") == "harmful"
    assert ref.label([1] * 200, [0] * 200, "chunk_words", "s") == "helpful"
    hi = [1] * 180 + [0] * 20
    assert ref.label(hi, list(hi), "template", "s") == "null"
    half = [1] * 100 + [0] * 100
    assert ref.label(half, list(half), "template", "s") == "small"
    assert ref.label([0] * 200, [1] * 200, "log_level", "s") == "null"


def test_regressed_rule():
    ref = _load("reference")
    assert ref.regressed([1] * 200, [0] * 200, "s") is True
    assert ref.regressed([1] * 200, [1] * 200, "s") is False
    # a 0.15 drop has a CI lower bound below 0.20
    assert ref.regressed([1] * 170 + [0] * 30, [1] * 140 + [0] * 60, "s") is False
    # a candidate that improves is not regressed
    assert ref.regressed([0] * 200, [1] * 200, "s") is False


def test_reference_question_has_candidate_cell_and_retrieval(sysm, toy):
    ref = _load("reference")
    q = sysm.load_questions("dev")[0]
    out = ref.reference_question(toy, q, sysm.FACTORS, 10)
    assert {"baseline_correct", "candidate_correct", "n", "regressed"} <= set(out)
    assert set(out["factors"]) == {f.name for f in sysm.FACTORS}
    assert out["factors"]["log_level"]["label"] == "null"
    assert set(out["retrieval"]) == {"baseline", "candidate"} | {
        f"only_{f.name}" for f in sysm.FACTORS
    }


# --- methods (PROTOCOL §5) ----------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(sysm, toy, tmp_path_factory):
    meth = _load("methods")
    qs = sysm.load_questions("dev")[:2]
    settings = InterventionSettings(6, 99, 0.05, "t")
    return meth.run_methods(toy, qs, tmp_path_factory.mktemp("m"), settings)


def test_methods_output_shape(small_run, sysm):
    names = {f.name for f in sysm.FACTORS}
    for row in small_run.values():
        assert set(row) >= {
            "GB",
            "GB_helpful",
            "GB_attributed",
            "GB_reverting_matters",
            "direction",
            "B0",
            "B1",
            "B2",
            "B2_flips",
            "cells",
            "timing",
        }
        assert row["B0"] == [f.name for f in sysm.FACTORS]
        for m in ("GB", "GB_helpful", "B1", "B2"):
            assert set(row[m]) <= names
        assert not set(row["GB"]) & set(row["GB_helpful"])


def test_gb_blame_is_the_harmful_list(sysm, toy, tmp_path):
    meth = _load("methods")
    q = next(q for q in sysm.load_questions("dev") if q["kind"] == "updated")
    rows = meth.run_methods(
        toy, [q], tmp_path, InterventionSettings(12, 199, 0.05, "h")
    )
    row = rows[q["id"]]
    harmful = [f for f, d in row["direction"].items() if d["harmful"]]
    assert row["GB"] == harmful


def test_b2_rule_is_e1s():
    meth = _load("methods")
    assert meth.b2_blame("wrong", {"index": "correct", "template": "wrong"}) == [
        "index"
    ]
    assert meth.b2_blame("correct", {"index": "correct"}) == []
    assert meth.B1_OWNERS == {
        "retrieval": ("index", "chunk_words"),
        "chat": ("template",),
    }


# --- recovery (PROTOCOL §6) ---------------------------------------------------------


def test_one_sided_recovery_allows_beating_baseline():
    rec = _load("recovery")
    base, cand = [1] * 40 + [0] * 20, [0] * 60
    better = [1] * 60
    assert rec.verified(better, cand, base, "s") is True
    assert rec.verified_two_sided(better, cand, base, "s") is False  # E1's rule
    assert rec.verified([1] * 30 + [0] * 30, cand, base, "s") is False  # too far below
    assert rec.verified(cand, cand, base, "s") is False  # not better than candidate


def test_fix_configs(sysm):
    rec = _load("recovery")
    cand = dict(sysm.CANDIDATE)
    fixed = rec.fix_config(cand, ["index"])
    assert fixed["index"] == "fresh" and fixed["chunk_words"] == 40
    assert rec.wrong_fix_config(cand) == {**cand, "template": "T1"}


def test_recovery_arms(sysm, toy):
    rec = _load("recovery")
    q = sysm.load_questions("dev")[0]
    out = rec.recovery_question(toy, q, ["index"], [], sysm.BASE, sysm.CANDIDATE, 8)
    assert out["n"] == 8
    assert set(out["arms"]) == {"gb_fix", "wrong_fix", "b2_fix"}
    assert out["arms"]["gb_fix"]["reverted"] == ["index"]
    assert out["arms"]["gb_fix"]["verified"] in (True, False)
    assert "verified_two_sided" in out["arms"]["gb_fix"]
    assert out["arms"]["b2_fix"] == {"reverted": [], "run": False}
    empty = rec.recovery_question(toy, q, [], [], sysm.BASE, sysm.CANDIDATE, 8)
    assert empty["arms"]["gb_fix"]["run"] is False


def test_recovery_reuses_identical_configs(sysm):
    rec = _load("recovery")
    calls = []

    class Counting:
        def answer(self, config, q, i):
            calls.append(json.dumps(config, sort_keys=True))
            return ("correct", "", [])

    q = {"id": "x"}
    rec.recovery_question(
        Counting(), q, ["template"], ["template"], sysm.BASE, sysm.CANDIDATE, 4
    )
    # gb_fix, wrong_fix and b2_fix are all "revert template": run once, not three times
    assert len(set(calls)) == 3 and len(calls) == 3 * 4


# --- scoring (PROTOCOL §7) ----------------------------------------------------------


def _q(index="harmful", chunk="null", template="null", log="null"):
    return {
        "index": index,
        "chunk_words": chunk,
        "template": template,
        "log_level": log,
    }


def _methods(gb):
    return {
        "GB": gb,
        "B0": ["index", "chunk_words", "template", "log_level"],
        "B1": ["index", "chunk_words"],
        "B2": [],
    }


def _arms(gb_ok, wrong_ok=False):
    return {
        "arms": {
            "gb_fix": {"verified": gb_ok, "run": gb_ok is not None},
            "wrong_fix": {"verified": wrong_ok, "run": True},
            "b2_fix": {"verified": False, "run": True},
        }
    }


def test_scoring_pass_case():
    sc = _load("score")
    labels = {f"q{i}": _q() for i in range(8)}
    regressed = {q: True for q in labels}
    methods = {q: _methods(["index"]) for q in labels}
    rec = {q: _arms(True) for q in labels}
    out = sc.criteria(labels, regressed, methods, rec)
    assert out["G1"]["value"] == 1.0 and out["G1"]["pass"]
    assert out["G2"]["pass"] and out["G3"]["pass"] and out["G3"]["testable"]
    assert out["G4"]["testable"] is False  # no helpful pairs
    assert out["G5"]["value"] == 1.0 and out["G6"]["value"] == 1.0
    assert out["G7"]["f1"]["GB"] == 1.0
    assert out["overall"] == "PASS"


def test_g4_blocks_helpful_changes_in_the_fix():
    sc = _load("score")
    labels = {f"q{i}": _q(chunk="helpful") for i in range(8)}
    regressed = {q: True for q in labels}
    methods = {q: _methods(["index", "chunk_words"]) for q in labels}
    rec = {q: _arms(True) for q in labels}
    out = sc.criteria(labels, regressed, methods, rec)
    assert out["G4"]["testable"] and out["G4"]["value"] == 1.0
    assert out["G4"]["pass"] is False and out["overall"] == "FAIL"


def test_g5_counts_empty_blame_as_not_verified():
    sc = _load("score")
    labels = {f"q{i}": _q() for i in range(8)}
    regressed = {q: True for q in labels}
    methods = {q: _methods(["index"] if i < 6 else []) for i, q in enumerate(labels)}
    rec = {q: _arms(True if i < 6 else None) for i, q in enumerate(labels)}
    out = sc.criteria(labels, regressed, methods, rec)
    assert out["G5"]["n"] == 8 and out["G5"]["verified"] == 6
    assert out["G5"]["pass"] is False


def test_g6_excludes_questions_where_template_is_harmful():
    sc = _load("score")
    labels = {f"q{i}": _q(template="harmful" if i == 0 else "null") for i in range(8)}
    regressed = {q: True for q in labels}
    methods = {q: _methods(["index"]) for q in labels}
    rec = {q: _arms(True) for q in labels}
    assert sc.criteria(labels, regressed, methods, rec)["G6"]["n"] == 7


def test_inconclusive_when_too_few_regressed_or_harmful():
    sc = _load("score")
    labels = {f"q{i}": _q() for i in range(8)}
    methods = {q: _methods(["index"]) for q in labels}
    rec = {q: _arms(True) for q in labels}
    few = {q: i < 5 for i, q in enumerate(labels)}
    assert sc.criteria(labels, few, methods, rec)["overall"] == "INCONCLUSIVE"
    weak = {f"q{i}": _q(index="small") for i in range(8)}
    every = {q: True for q in weak}
    assert sc.criteria(weak, every, methods, rec)["overall"] == "INCONCLUSIVE"


def test_descriptive_reports_b2_fix_and_undirected():
    sc = _load("score")
    labels = {"q0": _q(chunk="helpful")}
    methods = {"q0": {**_methods(["index"]), "GB_attributed": ["index", "chunk_words"]}}
    rec = {"q0": _arms(True)}
    d = sc.descriptive(labels, {"q0": True}, methods, rec)
    assert d["fix_verified"]["GB"] == {"verified": 1, "n": 1}
    assert d["fix_verified"]["B2"] == {"verified": 0, "n": 1}
    assert d["undirected_attributed"]["tp"] == 2  # E1-style: large in either direction
    assert set(d["methods"]) == {"GB", "B0", "B1", "B2"}


# --- staged runner ------------------------------------------------------------------


def _small(run, monkeypatch):
    for k, v in (
        ("N_MAIN", 6),
        ("N_REF", 20),
        ("N_REC", 10),
        ("N_GATE", 2),
        ("GATE_MIN", 0.0),
    ):
        monkeypatch.setattr(run, k, v)


def test_run_end_to_end_with_toy_backends(tmp_path, monkeypatch):
    run = _load("run")
    _small(run, monkeypatch)
    out = tmp_path / "out"
    assert run.main(["--out", str(out), "--backend", "toy", "--limit", "2"]) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["protocol"] == "experiments/v8/e1b_directed_rag/PROTOCOL.md v1.0"
    assert res["gate"]["selected"] == "toy"
    assert {"criteria", "descriptive", "per_question", "software"} <= set(res)
    assert set(res["criteria"]) >= {f"G{i}" for i in range(1, 8)} | {"overall"}
    assert (out / "pip_freeze.txt").exists()
    row = next(iter(res["per_question"].values()))
    assert {"reference", "methods", "recovery"} <= set(row)
    assert run.main(["--out", str(out), "--backend", "toy", "--limit", "2"]) == 0


def test_run_refuses_changed_settings_on_resume(tmp_path, monkeypatch):
    run = _load("run")
    _small(run, monkeypatch)
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
        "main": "e1b-main",
        "reference": "e1b-reference",
        "recovery": "e1b-recovery",
        "gate": "e1b-gate",
    }
    rec, ref = _load("recovery"), _load("reference")
    assert (rec.N_PERM, rec.N_BOOT, rec.ALPHA, rec.MARGIN) == (999, 2000, 0.05, 0.15)
    assert (ref.N_BOOT, ref.LARGE_LO, ref.NULL_HI, ref.REGRESSED_LO) == (
        2000,
        0.20,
        0.10,
        0.20,
    )
