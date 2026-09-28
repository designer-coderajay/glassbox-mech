"""Regression tests for the security hardening pass (2026-09-28).

Covers: stored XSS in the Annex IV HTML vault, unsafe torch.load
deserialisation, and unbounded API request fields.
"""
from __future__ import annotations

import pathlib
import re

from glassbox.evidence_vault import AnnexIVEvidenceVault, _esc

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = '<script>alert("x")</script>'


def test_esc_escapes_markup_and_quotes() -> None:
    assert _esc(PAYLOAD) == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
    assert _esc(None) == ""


def test_vault_html_escapes_user_supplied_fields() -> None:
    vault = AnnexIVEvidenceVault(
        model_name=PAYLOAD,
        provider=PAYLOAD,
        use_case=PAYLOAD,
        deployment_ctx="credit_scoring",
        commit_sha="<img src=x>",
    )
    vault.build_vault()
    page = vault.to_html()
    assert "<script>" not in page
    assert "<img src=x>" not in page
    assert "&lt;script&gt;" in page


def test_torch_load_calls_use_weights_only() -> None:
    offenders = []
    for path in (ROOT / "glassbox").glob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\btorch\.load\(", line) and "weights_only=True" not in line:
                offenders.append(f"{path.name}:{n}")
    assert not offenders, f"torch.load without weights_only=True: {offenders}"


def test_api_string_fields_are_length_bounded() -> None:
    src = (ROOT / "api" / "main.py").read_text()
    for field in ("prompt", "decision_prompt", "model_name", "provider_name"):
        pattern = rf"\b{field}:\s+str\s*=\s*Field\([^)]*max_length="
        assert re.search(pattern, src), f"{field} has no max_length"
