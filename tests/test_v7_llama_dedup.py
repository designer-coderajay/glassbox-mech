"""Mechanics of V7 Experiment 2 (experiments/v7/llama_dedup) with synthetic traces.

These tests deliberately do NOT run llama_index: the real repro runs only after the
protocol is committed (pre-registration). They check the evaluator's criteria logic,
including that a no-effect world FAILS, and the subject's stable node description.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glassbox.v7.trace import Recorder

EXP = Path(__file__).resolve().parent.parent / "experiments" / "v7" / "llama_dedup"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"exp2_{name}", EXP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ev():
    return _load("evaluate")


@pytest.fixture(scope="module")
def subj():
    return _load("subject")


DOC1 = {"ref_doc_id": "doc-1", "score": 0.9, "text_sha256": "a"}
DOC2 = {"ref_doc_id": "doc-2", "score": 0.8, "text_sha256": "a"}


def _write(results: Path, version: str, path: str, run: str, docs) -> None:
    rec = Recorder(out_dir=results / version / path, run_id=run)
    rec.environment["glassbox.env.packages"] = [f"llama-index-core=={version}"]
    with rec:
        with rec.span("retrieval", "repro-21033", kind="INTERNAL") as s:
            rec.add_content(s, {"gen_ai.retrieval.documents": docs})


def _world(results: Path, bug_sync_docs) -> None:
    for version in ("0.14.17", "0.14.19"):
        for path in ("sync", "async"):
            docs = (
                bug_sync_docs
                if (version, path) == ("0.14.17", "sync")
                else [DOC1, DOC2]
            )
            for run in ("r1", "r2"):
                _write(results, version, path, run, docs)


def test_expected_world_passes(ev, tmp_path):
    _world(tmp_path, bug_sync_docs=[DOC1])  # bug: sync drops doc-2
    r = ev.evaluate(tmp_path)
    assert r["overall"] == "PASS", r["criteria"]
    assert r["criteria"]["C5_n_documents"] == {
        "0.14.17/sync": 1,
        "0.14.17/async": 2,
        "0.14.19/sync": 2,
        "0.14.19/async": 2,
    }
    assert "glassbox.env.packages" in r["environment_differences_bug_vs_fix"]


def test_no_bug_world_fails(ev, tmp_path):
    """If the bug did not manifest, C1 and C3a must FAIL (no false pass)."""
    _world(tmp_path, bug_sync_docs=[DOC1, DOC2])
    r = ev.evaluate(tmp_path)
    assert not r["criteria"]["C1_localisation_bug"]["pass"]
    assert not r["criteria"]["C3a_intervention_sync"]["pass"]
    assert r["overall"] == "FAIL"


def test_nondeterminism_fails_c4(ev, tmp_path):
    _world(tmp_path, bug_sync_docs=[DOC1])
    victim = ev.trace_path(tmp_path, "0.14.19", "async", "r2")
    victim.unlink()
    _write(tmp_path, "0.14.19", "async", "r2", [DOC2, DOC1])
    r = ev.evaluate(tmp_path)
    assert not r["criteria"]["C4_determinism"]["pass"]
    assert r["overall"] == "FAIL"


def test_main_writes_results_once(ev, tmp_path):
    _world(tmp_path, bug_sync_docs=[DOC1])
    assert ev.main(["--results", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "results.json").read_text())["overall"] == "PASS"
    with pytest.raises(FileExistsError):
        ev.main(["--results", str(tmp_path)])


def test_subject_describes_nodes_without_random_ids(subj):
    def node(ref, text, score):
        inner = SimpleNamespace(ref_doc_id=ref, get_content=lambda: text, node_id="rnd")
        return SimpleNamespace(node=inner, score=score)

    out = subj.describe([node("doc-1", "shared content", 0.9)])
    assert out == [
        {
            "ref_doc_id": "doc-1",
            "score": 0.9,
            "text_sha256": hashlib.sha256(b"shared content").hexdigest(),
        }
    ]
    assert "node_id" not in out[0]


def test_subject_loads_trace_module_standalone(subj):
    mod = subj.load_trace_module()
    assert hasattr(mod, "Recorder") and mod.SCHEMA.startswith("glassbox.trace/")


def test_experiment2_package_builds_and_verifies(tmp_path):
    from glassbox.v7.evidence import build_package, verify_package

    pkg_mod = _load("package")
    if not (EXP / "results" / "results.json").exists():
        pytest.skip("official results not present")
    pkg = build_package(tmp_path, pkg_mod.PACKAGE_ID, pkg_mod.spec())
    assert verify_package(pkg) == []
    assert len(list((pkg / "artifacts").glob("traces_*/*.trace.jsonl"))) == 8
