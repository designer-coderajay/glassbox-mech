"""Tests for the V7 end-to-end investigation and the built-in RAG demo (toy scorer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glassbox.v7 import demo_rag
from glassbox.v7.evidence import verify_package
from glassbox.v7.intervene import InterventionSettings
from glassbox.v7.investigate import investigate, main, render_report

FAST = InterventionSettings(n_per_cell=8, n_perm=99, alpha=0.05, seed="test")


@pytest.fixture(scope="module")
def toy_investigation(tmp_path_factory):
    out = tmp_path_factory.mktemp("inv")
    inc = demo_rag.make_incident(demo_rag.toy_scorer)
    return investigate(inc, out / "work", FAST)


# --- demo system ------------------------------------------------------------------


def test_retrieval_matches_experiment_1_design():
    for q in ("q1", "q2", "q3", "q4"):
        doc = f"d{q[1]}"
        assert demo_rag.retrieve(demo_rag.QUESTIONS[q][0], "full")[0][0] == doc
        assert demo_rag.retrieve(demo_rag.QUESTIONS[q][0], "stale")[0][0] != doc
    for q in ("q5", "q6", "q7", "q8"):
        question = demo_rag.QUESTIONS[q][0]
        assert demo_rag.retrieve(question, "full") == demo_rag.retrieve(
            question, "stale"
        )


def test_corpus_matches_experiment_1_file():
    root = Path(__file__).resolve().parent.parent
    path = root / "experiments" / "v7" / "rag_fault" / "corpus.json"
    if not path.exists():
        pytest.skip("experiment corpus not present")
    data = json.loads(path.read_text())
    assert {d["id"]: d["text"] for d in data["documents"]} == demo_rag.DOCUMENTS
    assert {
        q["id"]: (q["question"], q["correct"], q["distractor"])
        for q in data["questions"]
    } == demo_rag.QUESTIONS


def test_templates_and_probability():
    assert demo_rag.build_prompt("context_first", ["A."], "Q?") == (
        "Context:\nA.\nQuestion: Q?\nAnswer:"
    )
    assert demo_rag.build_prompt("question_first", ["A."], "Q?") == (
        "Question: Q?\nContext:\nA.\nAnswer:"
    )
    with pytest.raises(ValueError):
        demo_rag.build_prompt("x", [], "Q?")
    assert demo_rag.p_correct(0.0) == 0.5
    assert demo_rag.p_correct(-900.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        demo_rag.make_scorer("nope")
    assert demo_rag.make_scorer("toy") is demo_rag.toy_scorer


def test_runner_trace_has_no_raw_text(tmp_path):
    run = demo_rag.make_runner("q1", lambda p, c, d: 5.0)
    cfg = {"index": "full", "template": "context_first", "log_level": "debug"}
    path = run(cfg, 3, tmp_path, "r000")
    text = path.read_text()
    assert "Kestrel" not in text and "order 4417" not in text.lower()
    assert demo_rag.answer_label(path) in {"correct", "wrong"}


# --- investigation ----------------------------------------------------------------


def test_toy_investigation_attributes_stale_index_only(toy_investigation):
    inv = toy_investigation
    summary = inv.factor_summary()
    assert summary["index"]["switching_alone_changed"] == ["q1", "q2", "q3", "q4"]
    assert summary["template"]["switching_alone_changed"] == []
    assert summary["log_level"]["switching_alone_changed"] == []
    first = {c.case: c.first_beyond_noise for c in inv.findings}
    assert first["q1"][0] == "retrieval#0"
    assert first["q5"][0] == "chat#0"  # prompt changed, outcome did not


def test_report_is_plain_and_does_not_say_caused(toy_investigation):
    md = render_report(toy_investigation)
    assert "## Answer" in md and "## What this does not show" in md
    assert "Switching index alone changed the outcome in 4/8 test cases" in md
    assert "No outcome change was detected from template, log_level alone" in md
    assert "caused" not in md.lower()
    assert "In 4 case(s) (q5, q6, q7, q8) the runs differed beyond noise" in md
    assert "| q1 | correct 8/8 | wrong 8/8 |" in md


def test_export_evidence_verifies(toy_investigation, tmp_path):
    pkg = toy_investigation.export_evidence(tmp_path / "evidence", "inv-test")
    assert verify_package(pkg) == []
    claim = json.loads((pkg / "claim.json").read_text())
    assert claim["status"] == "PRELIMINARY"
    assert "4/8" in claim["claim"]
    json.dumps(toy_investigation.to_dict())


def test_cli_demo_toy(tmp_path, capsys, monkeypatch):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--scorer", "toy", "--runs", "6"]) == 0
    printed = capsys.readouterr().out
    assert "toy scorer, not a language model" in printed
    assert (out / "REPORT.md").exists()
    assert verify_package(out / "evidence" / "investigation") == []
    with pytest.raises(FileExistsError):
        main(["demo", "--out", str(out), "--scorer", "toy", "--runs", "6"])


def test_incident_with_subset_of_changes(tmp_path):
    inc = demo_rag.make_incident(demo_rag.toy_scorer, ["template", "log_level"])
    assert [f.name for f in inc.factors] == ["template", "log_level"]
    assert "re-indexed" not in inc.description
    assert "prompt template was edited" in inc.description
    inv = investigate(inc, tmp_path, FAST)
    assert all(not c.attributed for c in inv.findings)  # index stays full
    with pytest.raises(ValueError):
        demo_rag.make_incident(demo_rag.toy_scorer, [])
    with pytest.raises(ValueError):
        demo_rag.make_incident(demo_rag.toy_scorer, ["model"])


def test_model_label_is_in_description():
    inc = demo_rag.make_incident(demo_rag.toy_scorer)
    assert "toy heuristic (NOT a language model)" in inc.description
    assert demo_rag.model_label(lambda p, a: 0.0).startswith("custom scorer")


def test_export_evidence_records_environment(toy_investigation, tmp_path):
    env = {"device": "cuda", "dtype": "bfloat16", "gpu_name": "Tesla T4"}
    pkg = toy_investigation.export_evidence(tmp_path / "ev", "inv-env", environment=env)
    assert verify_package(pkg) == []
    prov = json.loads((pkg / "provenance.json").read_text())
    assert json.loads(prov["environment"])["gpu_name"] == "Tesla T4"


def test_cli_demo_gpt2_passes_compute_and_records_it(tmp_path, monkeypatch):

    from glassbox.v7 import compute

    seen = {}

    class FakeScorer:
        def __init__(self, cfg):
            self.compute = cfg

        def __call__(self, prompt, answer):
            return demo_rag.toy_scorer(prompt, answer)

    def fake_make_scorer(kind, cfg=None):
        seen["kind"], seen["cfg"] = kind, cfg
        return FakeScorer(cfg)

    monkeypatch.setattr(demo_rag, "make_scorer", fake_make_scorer)
    monkeypatch.setattr(
        compute,
        "environment_record",
        lambda cfg, torch_module=None: {"device": cfg.device, "dtype": cfg.dtype},
    )
    out = tmp_path / "gpu"
    args = ["demo", "--out", str(out), "--scorer", "gpt2", "--runs", "6"]
    assert main(args + ["--device", "auto", "--dtype", "bfloat16"]) == 0
    assert seen["cfg"] == compute.ComputeConfig("auto", "bfloat16")
    prov = json.loads(
        (out / "evidence" / "investigation" / "provenance.json").read_text()
    )
    assert json.loads(prov["environment"]) == {"device": "auto", "dtype": "bfloat16"}


# --- timing log (V8 B2, PLAN §2 item 5) ---------------------------------------------


def test_investigation_records_per_case_timings(toy_investigation):
    t = toy_investigation.timings
    assert set(t) == set(demo_rag.QUESTIONS)
    for case in t.values():
        assert set(case) == {"runs_s", "analysis_s"}
        assert all(v >= 0 for v in case.values())


def test_timings_are_not_in_investigation_json(toy_investigation):
    assert "timings" not in json.dumps(toy_investigation.to_dict())


def test_cli_writes_timing_log(tmp_path):
    out = tmp_path / "o"
    assert main(["demo", "--out", str(out), "--scorer", "toy", "--runs", "6"]) == 0
    t = json.loads((out / "timing.json").read_text())
    for key in ("setup_s", "runs_s", "analysis_s", "report_s", "total_s"):
        assert t[key] >= 0
    assert t["total_s"] + 0.01 >= t["runs_s"] + t["analysis_s"]  # rounding
    assert t["unit"] == "seconds, wall clock" and t["started_utc"].endswith("Z")
    assert set(t["cases"]) == set(demo_rag.QUESTIONS)
