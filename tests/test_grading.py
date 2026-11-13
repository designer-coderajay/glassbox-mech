"""glassbox.grading: one grade rule and one status vocabulary for every surface."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from glassbox import grading
from glassbox.compliance import (
    AnnexIVReport,
    ComplianceStatus,
    ExplainabilityGrade,
    explainability_grade,
    normalize_status,
)
from tests.test_compliance import _make_result

ROOT = Path(__file__).resolve().parent.parent
GRID = [i / 20 for i in range(21)]


def _f1(s: float, c: float) -> float:
    return 2 * s * c / (s + c) if s + c else 0.0


# --- the rule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "suff,comp,f1,want",
    [
        (1.00, 0.543, 0.704, "B"),  # IOI default circuit (key number)
        (0.90, 0.80, 0.847, "A"),
        (0.75, 1.00, 0.857, "B"),  # high F1 but sufficiency below A
        (0.60, 1.00, 0.75, "C"),  # high F1 but sufficiency below B
        (0.80, 0.10, 0.178, "D"),
        (0.0, 0.0, 0.0, "D"),
    ],
)
def test_grade_letter(suff, comp, f1, want):
    assert grading.grade_letter(suff, comp, f1) == want


def test_thresholds_are_the_documented_ones():
    assert grading.GRADE_A_MIN == (0.80, 0.60, 0.80)
    assert grading.GRADE_B_MIN == (0.65, 0.40, 0.65)
    assert grading.GRADE_C_MIN_F1 == 0.50
    assert grading.CHECK_FAIL_BELOW_F1 == 0.50
    assert set(grading.GRADE_LABELS) == {"A", "B", "C", "D"}


@pytest.mark.parametrize(
    "f1,want",
    [(0.9, "checks_passed_with_gaps"), (0.65, "checks_passed_with_gaps"),
     (0.6, "incomplete"), (0.5, "incomplete"), (0.49, "checks_failed"),
     (0.0, "checks_failed")],
)
def test_check_status(f1, want):
    assert grading.check_status(f1) == want
    assert grading.check_status(f1) in grading.STATUSES


def test_normalize_status_covers_every_spelling_written():
    for old, new in grading.STATUS_ALIASES.items():
        assert grading.normalize_status(old) == new
        assert new in grading.STATUSES
    for s in grading.STATUSES:
        assert grading.normalize_status(s) == s
    assert grading.normalize_status("something_else") == "something_else"
    assert grading.normalize_status("") == ""


def test_module_is_mit_and_free_of_heavy_imports():
    src = inspect.getsource(grading)
    assert src.startswith("# SPDX-License-Identifier: MIT")
    assert "import torch" not in src and "from glassbox" not in src


# --- every surface agrees ---------------------------------------------------------


def test_report_uses_the_shared_rule():
    r = AnnexIVReport(
        model_name="m", system_purpose="p", provider_name="x", provider_address="y"
    )
    assert r._GRADE_A_THRESHOLD == grading.GRADE_A_MIN
    assert r._GRADE_B_THRESHOLD == grading.GRADE_B_MIN
    assert r._FAITHFULNESS_RISK_F1_THRESHOLD == grading.CHECK_FAIL_BELOW_F1
    for s in GRID:
        for c in GRID:
            f1 = _f1(s, c)
            letter = grading.grade_letter(s, c, f1)
            assert explainability_grade(s, c, f1).value[0] == letter
            assert r._compute_grade(s, c, f1).value[0] == letter


@pytest.mark.parametrize("suff,comp", [(0.9, 0.8), (0.75, 1.0), (0.6, 0.5), (0.8, 0.1)])
def test_report_json_status_and_grade_match_grading(suff, comp):
    r = AnnexIVReport(
        model_name="m", system_purpose="p", provider_name="x", provider_address="y"
    )
    r.add_analysis(_make_result(suff=suff, comp=comp), use_case="u")
    d = json.loads(r.to_json())
    f1 = _f1(suff, comp)
    assert d["compliance_status"] == grading.check_status(f1)
    s3 = d["sections"]["3_monitoring_control"]
    assert s3["explainability_grade"][0] == grading.grade_letter(suff, comp, f1)


def test_compliance_normalize_status_is_the_shared_one():
    for old, new in grading.STATUS_ALIASES.items():
        assert normalize_status(old) == new
        assert ComplianceStatus(old).value == new
    assert ExplainabilityGrade.GRADE_B.value[0] == "B"


@pytest.mark.parametrize("suff,comp", [(0.9, 0.8), (0.75, 1.0), (0.6, 0.5), (0.8, 0.1)])
def test_audit_log_grade_and_status_match_grading(tmp_path, suff, comp):
    from glassbox.audit_log import AuditLog

    rec = AuditLog(tmp_path / "a.jsonl").append_from_result(
        _make_result(suff=suff, comp=comp)
    )
    f1 = _f1(suff, comp)
    assert rec.explainability_grade == grading.grade_letter(suff, comp, f1)
    assert rec.compliance_status == grading.check_status(f1)


def test_audit_log_keeps_grade_fallback_without_f1(tmp_path):
    from glassbox.audit_log import AuditLog

    log = AuditLog(tmp_path / "a.jsonl")
    rec = log.append_from_result({"explainability_grade": "B — x"})
    assert rec.compliance_status == "checks_passed_with_gaps"
    rec = log.append_from_result({"explainability_grade": "C — x"})
    assert rec.compliance_status == "checks_failed"


def test_telemetry_grade_uses_the_shared_rule():
    from glassbox import telemetry

    src = inspect.getsource(telemetry)
    assert "grade_letter(" in src
    assert 'f1 >= 0.80 else "B"' not in src


@pytest.mark.parametrize(
    "module", ["glassbox.mlflow_integration", "glassbox.hf_integration"]
)
def test_integrations_use_the_report_status_rule(module):
    import importlib

    src = inspect.getsource(importlib.import_module(module))
    assert "check_status(" in src and "grade_letter(" in src
    assert "NEEDS_REVIEW" not in src
    assert 'if suff >= 0.75 else' not in src


def test_space_ui_uses_the_shared_rule():
    spec = importlib.util.spec_from_file_location(
        "space_ui_grading", ROOT / "dashboard" / "space_ui.py"
    )
    ui = importlib.util.module_from_spec(spec)
    sys.modules["space_ui_grading"] = ui
    spec.loader.exec_module(ui)
    for s in GRID[::2]:
        for c in GRID[::2]:
            f1 = _f1(s, c)
            letter, label = ui.grade(s, c, f1)
            assert letter == grading.grade_letter(s, c, f1)
            assert label == grading.GRADE_LABELS[letter]


@pytest.mark.parametrize("path", ["mcp/server.py", "mcp/glassbox_mcp/server.py"])
def test_mcp_servers_use_the_shared_rule_when_available(path):
    src = (ROOT / path).read_text(encoding="utf-8")
    assert "from glassbox.grading import grade_letter" in src
    assert "_compute_grade(f1, suff, comp)" in src
