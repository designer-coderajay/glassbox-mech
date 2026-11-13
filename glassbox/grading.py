# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""One place for the explainability grade and the check status.

Every surface that shows a grade or a status for an analysis (the Annex IV draft, the
audit log, telemetry, MLflow, the Hugging Face model card, the Space, the MCP server)
uses these functions, so one analysis gets one grade and one status everywhere.

Grade (from the faithfulness metrics of the circuit):

* ``A``: sufficiency ≥ 0.80, comprehensiveness ≥ 0.60 and F1 ≥ 0.80;
* ``B``: sufficiency ≥ 0.65, comprehensiveness ≥ 0.40 and F1 ≥ 0.65;
* ``C``: F1 ≥ 0.50;
* ``D``: otherwise.

Check status (the status of Glassbox's own checks; **not** a determination of
compliance with Regulation (EU) 2024/1689):

* ``checks_passed_with_gaps``: F1 ≥ 0.65. Provider-only fields are still needed, which
  is why an analysis alone never yields ``checks_passed``;
* ``checks_failed``: F1 < 0.50;
* ``incomplete``: in between, or nothing to check.

``checks_passed`` exists in the vocabulary for documentation whose provider fields
are complete; Glassbox does not produce it from an analysis.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "GRADE_A_MIN",
    "GRADE_B_MIN",
    "GRADE_C_MIN_F1",
    "CHECK_FAIL_BELOW_F1",
    "GRADE_LABELS",
    "STATUSES",
    "STATUS_ALIASES",
    "grade_letter",
    "check_status",
    "normalize_status",
]

#: (min sufficiency, min comprehensiveness, min F1)
GRADE_A_MIN: Tuple[float, float, float] = (0.80, 0.60, 0.80)
GRADE_B_MIN: Tuple[float, float, float] = (0.65, 0.40, 0.65)
GRADE_C_MIN_F1: float = 0.50
#: Below this F1 the check status is ``checks_failed``.
CHECK_FAIL_BELOW_F1: float = 0.50

GRADE_LABELS: Dict[str, str] = {
    "A": "Fully explainable",
    "B": "Substantially explainable",
    "C": "Partially explainable",
    "D": "Minimally explainable",
}

#: The canonical status values.
STATUSES: Tuple[str, ...] = (
    "checks_passed",
    "checks_passed_with_gaps",
    "incomplete",
    "checks_failed",
)

#: Other spellings that Glassbox has written, and their canonical value.
#: Pre-4.7.0 report values, the evidence vault's upper-case values, and the values
#: the MLflow and Hugging Face integrations wrote up to 4.7.0.
STATUS_ALIASES: Dict[str, str] = {
    "compliant": "checks_passed",
    "conditionally_compliant": "checks_passed_with_gaps",
    "non_compliant": "checks_failed",
    "CHECKS_PASSED": "checks_passed",
    "CHECKS_PARTIAL": "incomplete",
    "CHECKS_FAILED": "checks_failed",
    "NEEDS_REVIEW": "incomplete",
}


def grade_letter(suff: float, comp: float, f1: float) -> str:
    """Explainability grade ``A``–``D`` from the faithfulness metrics.

    Args:
        suff: Sufficiency of the circuit.
        comp: Comprehensiveness of the circuit.
        f1: Faithfulness F1.

    Returns:
        ``"A"``, ``"B"``, ``"C"`` or ``"D"``.
    """
    for letter, (ms, mc, mf) in (("A", GRADE_A_MIN), ("B", GRADE_B_MIN)):
        if suff >= ms and comp >= mc and f1 >= mf:
            return letter
    return "C" if f1 >= GRADE_C_MIN_F1 else "D"


def check_status(f1: float) -> str:
    """Status of Glassbox's checks for one analysis, from its faithfulness F1.

    Returns:
        ``checks_passed_with_gaps``, ``incomplete`` or ``checks_failed``. Never
        ``checks_passed``: provider-only fields are always still needed.
    """
    if f1 >= GRADE_B_MIN[2]:
        return "checks_passed_with_gaps"
    if f1 < CHECK_FAIL_BELOW_F1:
        return "checks_failed"
    return "incomplete"


def normalize_status(value: str) -> str:
    """Map any status spelling Glassbox has written to its canonical value.

    Values that are already canonical, and values Glassbox never wrote, are returned
    unchanged.
    """
    return STATUS_ALIASES.get(value, value)
