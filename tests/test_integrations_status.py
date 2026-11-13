"""MLflow and Hugging Face integrations write the report's grade, status and version.

Both third-party packages are replaced by small fakes, so these tests run without
``mlflow`` or ``huggingface_hub`` installed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import glassbox
from glassbox import grading, hf_integration, mlflow_integration


def _result(suff: float, comp: float) -> Dict[str, Any]:
    f1 = 2 * suff * comp / (suff + comp) if suff + comp else 0.0
    return {
        "faithfulness": {
            "sufficiency": suff,
            "comprehensiveness": comp,
            "f1": f1,
            "category": "faithful",
        },
        "n_heads": 3,
    }


class _FakeClient:
    def __init__(self) -> None:
        self.artifacts: List[Dict[str, Any]] = []
        self.version_tags: Dict[str, str] = {}

    def log_artifact(self, run_id: str, path: str, artifact_path: str = "") -> None:
        with open(path, encoding="utf-8") as fh:
            self.artifacts.append(json.load(fh))

    def set_model_version_tag(self, name, version, key, value) -> None:
        self.version_tags[key] = value


class _FakeMlflow:
    class exceptions:  # noqa: N801 - mirrors mlflow.exceptions
        class MlflowException(Exception):
            pass

    def __init__(self) -> None:
        self.params: Dict[str, str] = {}
        self.tags: Dict[str, str] = {}
        self.metrics: Dict[str, float] = {}
        self.client = _FakeClient()
        outer = self

        class _Tracking:
            @staticmethod
            def MlflowClient() -> _FakeClient:  # noqa: N802
                return outer.client

        self.tracking = _Tracking()

    def active_run(self) -> object:
        return object()

    def log_metrics(self, metrics, step=None) -> None:
        self.metrics.update(metrics)

    def log_param(self, key, value) -> None:
        self.params[key] = value

    def set_tags(self, tags) -> None:
        self.tags.update(tags)


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setattr(mlflow_integration, "_require_mlflow", lambda: fake)
    return fake


CASES = [(0.9, 0.8), (0.75, 1.0), (0.6, 0.5), (0.8, 0.1), (1.0, 0.543)]


@pytest.mark.parametrize("suff,comp", CASES)
def test_mlflow_run_tags_match_the_report_rule(fake_mlflow, suff, comp):
    r = _result(suff, comp)
    f1 = r["faithfulness"]["f1"]
    mlflow_integration.log_glassbox_run(
        r, log_circuit_json=False, log_html_report=False
    )
    want = grading.check_status(f1)
    assert fake_mlflow.params["glassbox.compliance_status"] == want
    assert fake_mlflow.tags["glassbox.compliance_status"] == want
    assert fake_mlflow.tags["glassbox.explainability_grade"] == grading.grade_letter(
        suff, comp, f1
    )
    assert fake_mlflow.params["glassbox.version"] == glassbox.__version__
    assert want in grading.STATUSES


def test_mlflow_no_longer_says_passed_for_a_failing_circuit(fake_mlflow):
    # Up to 4.7.0: sufficiency 0.80 alone gave CHECKS_PASSED.
    mlflow_integration.log_glassbox_run(
        _result(0.8, 0.1), log_circuit_json=False, log_html_report=False
    )
    assert fake_mlflow.tags["glassbox.compliance_status"] == "checks_failed"
    assert fake_mlflow.tags["glassbox.explainability_grade"] == "D"
    assert fake_mlflow.tags["glassbox.faithfulness_grade"] == "Good"  # band kept


@pytest.mark.parametrize("suff,comp", CASES)
def test_mlflow_compliance_artifact(fake_mlflow, suff, comp):
    r = _result(suff, comp)
    f1 = r["faithfulness"]["f1"]
    mlflow_integration.register_compliance_artifact(
        r, run_id="run1", model_uri="models:/demo/3", model_name="demo"
    )
    payload = fake_mlflow.client.artifacts[0]
    assert payload["compliance_status"] == grading.check_status(f1)
    assert payload["explainability_grade"] == grading.grade_letter(suff, comp, f1)
    assert payload["glassbox_version"] == glassbox.__version__
    assert fake_mlflow.client.version_tags["glassbox.compliance_status"] == (
        grading.check_status(f1)
    )


@pytest.mark.parametrize("suff,comp", CASES)
def test_hf_model_card_section(suff, comp):
    card = object.__new__(hf_integration.HuggingFaceModelCard)
    card.repo_id = "org/model"
    r = _result(suff, comp)
    f1 = r["faithfulness"]["f1"]
    section = card._build_section(r, auditor="a", annex_iv_url=None)
    letter = grading.grade_letter(suff, comp, f1)
    status = grading.check_status(f1)
    assert f"v{glassbox.__version__}" in section
    assert "4.2.6" not in section
    assert f"**{letter} — {grading.GRADE_LABELS[letter]}**" in section
    assert f"**{status.replace('_', ' ')}**" in section
    assert "not a determination of compliance" in section
    assert "Compliance Grade" not in section
    start = section.index(hf_integration._COMPLIANCE_JSON_START)
    end = section.index(hf_integration._COMPLIANCE_JSON_END)
    meta = json.loads(
        section[start + len(hf_integration._COMPLIANCE_JSON_START) : end]
    )
    assert meta["compliance_status"] == status
    assert meta["explainability_grade"] == letter
    assert meta["glassbox_version"] == glassbox.__version__
