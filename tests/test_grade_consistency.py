"""One analysis, one grade: the audit log must agree with the Annex IV draft.

Before this change the audit log graded a raw ``analyze()`` result from F1 alone, so it
could record grade A where the report for the same analysis says B. A report JSON
passed directly was not recognised and was logged as grade D, model "unknown".
"""

from __future__ import annotations

import json
import logging

import pytest

from glassbox.audit_log import AuditLog
from glassbox.compliance import AnnexIVReport, ExplainabilityGrade, explainability_grade
from tests.test_compliance import _make_result


def _report(suff: float, comp: float) -> AnnexIVReport:
    r = AnnexIVReport(
        model_name="m", system_purpose="p", provider_name="x", provider_address="y"
    )
    r.add_analysis(_make_result(suff=suff, comp=comp), use_case="u")
    return r


def _report_grade(report_json: dict) -> str:
    return report_json["sections"]["3_monitoring_control"]["explainability_grade"][0]


GRID = [i / 10 for i in range(11)]


def test_public_rule_matches_the_report_rule():
    r = _report(0.9, 0.9)
    for suff in GRID:
        for comp in GRID:
            f1 = 2 * suff * comp / (suff + comp) if suff + comp else 0.0
            assert explainability_grade(suff, comp, f1) is r._compute_grade(
                suff, comp, f1
            )


def test_public_rule_known_points():
    assert explainability_grade(1.00, 0.543, 0.704) is ExplainabilityGrade.GRADE_B
    assert explainability_grade(0.75, 1.00, 0.857) is ExplainabilityGrade.GRADE_B
    assert explainability_grade(0.90, 0.80, 0.847) is ExplainabilityGrade.GRADE_A
    assert explainability_grade(0.30, 0.20, 0.24) is ExplainabilityGrade.GRADE_D


@pytest.mark.parametrize("suff,comp", [(0.75, 1.0), (0.6, 1.0), (1.0, 0.543), (0.9, 0.8)])
def test_raw_result_gets_the_report_grade(tmp_path, suff, comp):
    result = _make_result(suff=suff, comp=comp)
    want = _report_grade(json.loads(_report(suff, comp).to_json()))
    rec = AuditLog(tmp_path / "a.jsonl").append_from_result(result)
    assert rec.explainability_grade == want


def test_report_json_passed_directly_is_recognised(tmp_path):
    report = json.loads(_report(0.75, 1.0).to_json())
    rec = AuditLog(tmp_path / "a.jsonl").append_from_result(report)
    assert rec.explainability_grade == "B"
    assert rec.model_name == "m"
    assert rec.compliance_status == report["compliance_status"]
    assert rec.report_id == report["report_id"]
    assert rec.faithfulness_f1 == pytest.approx(0.857142, abs=1e-5)


def test_api_shaped_result_is_unchanged(tmp_path):
    report = json.loads(_report(0.75, 1.0).to_json())
    rec = AuditLog(tmp_path / "a.jsonl").append_from_result({"full_report": report})
    assert rec.explainability_grade == "B"


def test_f1_only_result_keeps_the_f1_fallback(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for f1, want in ((0.85, "A"), (0.70, "B"), (0.55, "C"), (0.20, "D")):
        rec = log.append_from_result({"faithfulness": {"f1": f1}})
        assert rec.explainability_grade == want


def test_explicit_grade_still_wins(tmp_path):
    result = {**_make_result(suff=0.75, comp=1.0), "explainability_grade": "C — x"}
    rec = AuditLog(tmp_path / "a.jsonl").append_from_result(result)
    assert rec.explainability_grade == "C"


def test_unrecognised_input_is_logged_loudly(tmp_path, caplog):
    log = AuditLog(tmp_path / "a.jsonl")
    with caplog.at_level(logging.WARNING, logger="glassbox.audit_log"):
        rec = log.append_from_result({"something": "else"})
    assert rec.explainability_grade == "D"  # behaviour kept: the record is written
    assert "no grade or faithfulness" in caplog.text
    with caplog.at_level(logging.WARNING, logger="glassbox.audit_log"):
        caplog.clear()
        log.append_from_result(_make_result(suff=0.9, comp=0.8))
    assert "no grade or faithfulness" not in caplog.text
