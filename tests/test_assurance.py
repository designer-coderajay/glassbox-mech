"""ASSURANCE.md must stay machine-checkable (V7 Step 2).

The claim table is the guard against marketing outrunning evidence, so its
structure is enforced: every row has a known status, a unique ID, and any
evidence path it cites exists in the repository.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSURANCE = REPO_ROOT / "ASSURANCE.md"

STATUSES = {
    "VERIFIED",
    "EXPERIMENTALLY_SUPPORTED",
    "REPRODUCED",
    "PRELIMINARY",
    "IMPLEMENTED",
    "HYPOTHESIS",
    "UNRESOLVED",
    "REFUTED",
    "PLANNED",
    "NOT_SUPPORTED",
}
NEEDS_EVIDENCE = {"VERIFIED", "EXPERIMENTALLY_SUPPORTED", "REPRODUCED", "IMPLEMENTED"}
ROW = re.compile(r"^\|\s*([A-Z]\d+(?:\.\d+)?)\s*\|(.+)\|\s*$")
PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:md|py|json|jsonl|toml|sha256))`")


def parse_rows(text: str) -> List[Dict[str, str]]:
    """Parse claim-table rows: | ID | Claim | Status | Evidence | Limits |."""
    rows = []
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group(2).split("|")]
        rows.append(
            {
                "id": match.group(1),
                "claim": cells[0],
                "status": cells[1].strip("` *"),
                "evidence": cells[2],
            }
        )
    return rows


@pytest.fixture(scope="module")
def rows() -> List[Dict[str, str]]:
    assert ASSURANCE.exists(), "ASSURANCE.md missing at repo root"
    parsed = parse_rows(ASSURANCE.read_text(encoding="utf-8"))
    assert parsed, "no claim rows found"
    return parsed


def test_parser_reads_a_row():
    line = "| L1 | Claim text | `IMPLEMENTED` | `README.md` | none |"
    assert parse_rows(line) == [
        {
            "id": "L1",
            "claim": "Claim text",
            "status": "IMPLEMENTED",
            "evidence": "`README.md`",
        }
    ]


def test_every_status_is_in_the_taxonomy(rows):
    bad = [(r["id"], r["status"]) for r in rows if r["status"] not in STATUSES]
    assert bad == []


def test_claim_ids_are_unique(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_supported_claims_cite_existing_evidence(rows):
    problems = []
    for r in rows:
        paths = PATH.findall(r["evidence"])
        if r["status"] in NEEDS_EVIDENCE and not paths:
            problems.append((r["id"], "no evidence path"))
        for p in paths:
            if not (REPO_ROOT / p).exists():
                problems.append((r["id"], f"missing {p}"))
    assert problems == []


def _tracked(rel: str) -> bool:
    result = subprocess.run(
        ["git", "--no-optional-locks", "ls-files", "--error-unmatch", rel],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs the git checkout",
)
def test_evidence_files_are_git_tracked(rows):
    """Evidence must survive a fresh clone: no gitignored or local-only files."""
    untracked = [
        (r["id"], p)
        for r in rows
        for p in PATH.findall(r["evidence"])
        if not _tracked(p)
    ]
    assert untracked == []


def test_no_independent_reproduction_is_claimed(rows):
    """REPRODUCED means someone else reproduced it; none has, as of writing."""
    assert [r["id"] for r in rows if r["status"] == "REPRODUCED"] == []
