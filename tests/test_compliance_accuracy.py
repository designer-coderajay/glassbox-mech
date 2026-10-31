"""Accuracy of generated compliance documents (V8 B2, release 4.6.1).

Covers the bugs found when the Annex IV draft was exercised end to end:
- ``to_model_card()`` crashed (``ExplainabilityGrade.D`` does not exist);
- it read misnamed fields, so every metric printed as 0.000;
- it overclaimed ("Complete", "audited", "authorised", sections marked done that the
  provider has not written);
- the draft stated human-oversight arrangements of the provider as facts.
"""

from __future__ import annotations

import json

import pytest

from glassbox.compliance import AnnexIVReport, DeploymentContext
from glassbox.evidence_vault import _ANNEX_IV_SECTIONS, AnnexIVEvidenceVault
from tests.test_compliance import _make_result


def _report(suff=0.92, comp=0.72, ctx=DeploymentContext.FINANCIAL_SERVICES):
    r = AnnexIVReport(
        model_name="CreditScorer v3.2",
        system_purpose="Credit risk assessment for loan applications",
        provider_name="Acme Bank NV",
        provider_address="1 Fintech Street, Amsterdam",
        deployment_context=ctx,
    )
    r.add_analysis(_make_result(suff=suff, comp=comp), use_case="Loan denial")
    return r


def test_model_card_does_not_crash_and_shows_real_metrics():
    r = _report()
    card = r.to_model_card()
    s3 = r._s3
    assert f"`{s3.sufficiency:.3f}`" in card and s3.sufficiency > 0
    assert f"`{s3.comprehensiveness:.3f}`" in card
    assert f"`{s3.f1_score:.3f}`" in card
    assert s3.explainability_grade in card
    assert "`0.000`" not in card


def test_model_card_status_is_the_computed_status():
    r = _report(suff=0.35, comp=0.18)
    status = r._compute_compliance_status().value
    assert status.replace("_", " ").title() in r.to_model_card()


@pytest.mark.parametrize(
    "phrase",
    [
        "Complete (all 9 sections)",
        "has been audited",
        "authorised",
        "✅ Completed",
        "✅ Documented",
        "Nanda et al. 2023",
        "EU AI Act Compliance Report",
    ],
)
def test_model_card_makes_no_overclaims(phrase):
    assert phrase not in _report().to_model_card()


def test_model_card_marks_provider_sections():
    card = _report().to_model_card()
    assert "not a conformity assessment" in card
    for article in ("Article 47", "Article 72"):
        line = next(ln for ln in card.splitlines() if article in ln)
        assert "Provider" in line


def test_model_card_lists_identified_risks():
    card = _report(suff=0.35, comp=0.18).to_model_card()
    assert "Insufficient circuit explainability" in card


def test_model_card_data_governance_reflects_provider_input():
    r = _report()
    assert "Provider to complete" in r.to_model_card()
    r._s4.training_dataset_description = "Loan book 2015-2024"
    assert "Provided by the provider" in r.to_model_card()


def test_model_card_writes_file(tmp_path):
    out = tmp_path / "MODEL_CARD.md"
    assert _report().to_model_card(str(out)) == out.read_text(encoding="utf-8")


@pytest.mark.parametrize("ctx", list(DeploymentContext))
def test_oversight_is_never_asserted_for_the_provider(ctx):
    s3 = _report(ctx=ctx)._s3
    for value in (s3.human_oversight_measures, s3.override_mechanism):
        assert value.startswith("[PROVIDER TO COMPLETE")


# --- Evidence vault: Annex IV numbering and provider statements ----------------
# Annex IV points as fetched from EUR-Lex (CELEX 32024R1689, Annex IV) on 2026-10-31.


GB = {
    "faithfulness": {"sufficiency": 1.0, "comprehensiveness": 0.22, "f1": 0.64},
    "n_heads": 4,
    "circuit": {(9, 6): 0.584, (9, 9): 0.431, (10, 0): 0.312, (3, 0): 0.067},
}


def _vault():
    return AnnexIVEvidenceVault(model_name="gpt2", provider="Acme").build_vault(
        gb_result=GB
    )


def test_vault_catalogue_follows_annex_iv_points_1_to_9():
    assert list(_ANNEX_IV_SECTIONS) == [f"§{i}" for i in range(1, 10)]
    for sec, words in {
        "§4": "performance metrics",
        "§5": "risk management",
        "§6": "changes",
        "§7": "harmonised standards",
        "§8": "declaration of conformity",
        "§9": "post-market",
    }.items():
        assert words in _ANNEX_IV_SECTIONS[sec].lower()


def test_vault_entries_use_catalogue_sections():
    assert {e.section for e in _vault().entries} <= set(_ANNEX_IV_SECTIONS)


def test_vault_conformity_entry_is_provider_only():
    (conf,) = [e for e in _vault().entries if "Conformity" in e.title]
    assert conf.section == "§8" and "Article 47" in conf.article_refs
    assert conf.description.startswith("[PROVIDER TO COMPLETE")
    assert "confirms" not in conf.description


def test_vault_risk_evidence_is_section_5():
    (f1,) = [e for e in _vault().entries if e.metric_name == "f1"]
    assert f1.section == "§5"
    assert "CI/CD" not in f1.description


def test_vault_standards_entry_claims_no_harmonised_standard():
    (std,) = [e for e in _vault().entries if e.section == "§7"]
    text = std.description + json.dumps(std.raw)
    assert "harmonised standard ISO/IEC 42001" not in text
    assert "Attribution patching — Conmy" not in text
    assert "[PROVIDER TO COMPLETE" in std.description


def test_vault_html_badge_is_not_a_compliance_determination():
    html_text = _vault().to_html()
    assert "Glassbox checks passed" in html_text
    assert "not a determination of compliance" in html_text
    assert (
        ">COMPLIANT &mdash;" not in html_text
        and "suitable for regulatory submission" not in html_text
    )


def test_status_label_says_draft_and_glassbox_checks_not_compliance():
    for suff, comp in ((0.92, 0.72), (0.35, 0.18), (0.71, 0.44)):
        label = _report(suff=suff, comp=comp)._status_label()
        assert label.startswith("Draft status (Glassbox checks): ")
        assert "not a determination of compliance" in label


def test_pdf_text_uses_draft_wording():
    import inspect

    from glassbox import compliance

    src = inspect.getsource(compliance.AnnexIVReport)
    assert "Documentation Status: {status.value" not in src
    assert "This Annex IV compliance report covers" not in src
    assert "EU AI Act Compliance Audit Platform" not in src
