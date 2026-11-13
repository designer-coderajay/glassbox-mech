"""4.7.0: status values say "Glassbox checks", not compliance.

Glassbox can say whether its own checks passed. It cannot determine compliance with
Regulation (EU) 2024/1689. Old values are still read for one release (removed in
4.8.0) so existing JSON files and audit logs keep working.
"""

from __future__ import annotations

import inspect
import json

import pytest

from glassbox.compliance import AnnexIVReport, ComplianceStatus, normalize_status
from tests.test_compliance import _make_result

NEW = {"checks_passed", "checks_passed_with_gaps", "incomplete", "checks_failed"}


def test_enum_values_are_check_statuses():
    assert {s.value for s in ComplianceStatus} == NEW
    assert ComplianceStatus.CHECKS_PASSED.value == "checks_passed"
    assert ComplianceStatus.CHECKS_FAILED.value == "checks_failed"


def test_old_member_names_are_aliases():
    assert ComplianceStatus.COMPLIANT is ComplianceStatus.CHECKS_PASSED
    assert (
        ComplianceStatus.CONDITIONALLY_COMPLIANT
        is ComplianceStatus.CHECKS_PASSED_WITH_GAPS
    )
    assert ComplianceStatus.NON_COMPLIANT is ComplianceStatus.CHECKS_FAILED


@pytest.mark.parametrize(
    "old,new",
    [
        ("compliant", "checks_passed"),
        ("conditionally_compliant", "checks_passed_with_gaps"),
        ("non_compliant", "checks_failed"),
        ("incomplete", "incomplete"),
        ("checks_failed", "checks_failed"),
    ],
)
def test_old_values_still_read(old, new):
    assert ComplianceStatus(old).value == new
    assert normalize_status(old) == new


def test_normalize_status_keeps_unknown_values():
    assert normalize_status("") == ""
    assert normalize_status("something_else") == "something_else"


@pytest.mark.parametrize("suff,comp", [(0.92, 0.72), (0.35, 0.18), (0.6, 0.5)])
def test_report_json_uses_new_values_and_a_note(suff, comp):
    r = AnnexIVReport(
        model_name="m", system_purpose="p", provider_name="x", provider_address="y"
    )
    r.add_analysis(_make_result(suff=suff, comp=comp), use_case="u")
    data = json.loads(r.to_json())
    assert data["compliance_status"] in NEW
    assert "not a determination of compliance" in data["status_note"]
    assert "compliant" not in data["compliance_status"]


def test_vault_overall_status_values():
    from glassbox.evidence_vault import AnnexIVEvidenceVault

    src = inspect.getsource(AnnexIVEvidenceVault)
    for old in ('"COMPLIANT"', '"MARGINAL"', '"NON-COMPLIANT"'):
        assert old not in src
    for new in ('"CHECKS_PASSED"', '"CHECKS_PARTIAL"', '"CHECKS_FAILED"'):
        assert new in src


@pytest.mark.parametrize(
    "module", ["glassbox.mlflow_integration", "glassbox.hf_integration"]
)
def test_integrations_do_not_tag_compliant(module):
    import importlib

    src = inspect.getsource(importlib.import_module(module))
    assert '"COMPLIANT"' not in src
    # From 4.8.0 the status comes from the report's rule (glassbox.grading).
    assert "check_status(" in src


def test_bias_markdown_does_not_say_compliant():
    from glassbox import bias

    assert "COMPLIANT" not in inspect.getsource(bias)


def test_audit_log_reads_old_and_new_failed_values(tmp_path):
    from glassbox.audit_log import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    for status in ("non_compliant", "checks_failed", "checks_passed_with_gaps"):
        log.append(
            model_name="m",
            analysis_mode="white_box",
            prompt="p",
            explainability_grade="C",
            compliance_status=status,
        )
    assert len(log.non_compliant()) == 2
    assert len(log.checks_failed()) == 2
    assert log.summary()["non_compliant_count"] == 2


def test_audit_log_derived_status_uses_new_values():
    from glassbox import audit_log

    src = inspect.getsource(audit_log)
    assert '"conditionally_compliant" if grade' not in src
    assert '"checks_passed_with_gaps" if grade' in src
