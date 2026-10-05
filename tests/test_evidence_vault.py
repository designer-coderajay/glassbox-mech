"""Tests for glassbox/evidence_vault.py — Annex IV evidence vault builder."""
import json

from glassbox.evidence_vault import (
    _ANNEX_IV_SECTIONS,
    AnnexIVEvidenceVault,
    VaultEntry,
    build_annex_iv_vault,
)

# A realistic GlassboxV2.analyze() result (GPT-2 IOI numbers).
GB = {
    "faithfulness": {"sufficiency": 1.0, "comprehensiveness": 0.22, "f1": 0.64},
    "n_heads": 4,
    "circuit": {(9, 6): 0.584, (9, 9): 0.431, (10, 0): 0.312, (3, 0): 0.067},
}


# ── VaultEntry ─────────────────────────────────────────────────────────────
def test_vault_entry_to_dict():
    e = VaultEntry(
        section="§4", article_refs=["Article 9"], title="t",
        description="d", evidence_type="faithfulness",
        metric_name="f1", metric_value=0.64, threshold=0.65, passed=False,
    )
    d = e.to_dict()
    assert d["section"] == "§4"
    assert d["passed"] is False
    assert d["metric_value"] == 0.64
    assert "timestamp_utc" in d


# ── construction + baseline build ──────────────────────────────────────────
def test_empty_vault_has_baseline_entries():
    v = AnnexIVEvidenceVault(model_name="gpt2", provider="Acme").build_vault()
    d = v.to_dict()
    assert d["model_name"] == "gpt2"
    assert d["provider"] == "Acme"
    assert d["n_entries"] >= 3  # general description + standards + conformity
    assert d["n_entries"] == len(v.entries)
    assert isinstance(d["sections_covered"], list)


def test_build_vault_clears_between_runs():
    v = AnnexIVEvidenceVault()
    v.build_vault(gb_result=GB)
    first = len(v.entries)
    v.build_vault(gb_result=GB)  # should reset, not accumulate
    assert len(v.entries) == first


# ── gb_result population ───────────────────────────────────────────────────
def test_build_from_gb_result_metrics():
    v = AnnexIVEvidenceVault().build_vault(gb_result=GB)
    by_metric = {e.metric_name: e for e in v.entries if e.metric_name}
    assert by_metric["sufficiency"].passed is True       # 1.00 >= 0.70
    assert by_metric["comprehensiveness"].passed is False  # 0.22 < 0.60
    assert by_metric["f1"].passed is False                # 0.64 < 0.65
    assert by_metric["n_heads"].metric_value == 4.0
    # top-heads circuit entry is present
    assert any("circuit attention heads" in e.title.lower() for e in v.entries)


def test_compliance_summary_noncompliant():
    s = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_dict()["compliance_summary"]
    assert s["overall_status"] == "NON-COMPLIANT"  # only 1 of 3 thresholds pass
    assert 0.0 <= s["pass_rate"] <= 1.0
    assert s["n_failed"] >= 2


def test_compliance_summary_compliant():
    good = {
        "faithfulness": {"sufficiency": 1.0, "comprehensiveness": 0.85, "f1": 0.92},
        "n_heads": 4, "circuit": {(9, 6): 0.5},
    }
    s = AnnexIVEvidenceVault().build_vault(gb_result=good).to_dict()["compliance_summary"]
    assert s["overall_status"] == "COMPLIANT"
    assert s["pass_rate"] == 1.0


# ── other input channels ───────────────────────────────────────────────────
def test_stability_entries_skip_non_numeric():
    v = AnnexIVEvidenceVault().build_vault(
        stability_result={"jaccard": 0.9, "rank_corr": 0.8, "note": "ignored"}
    )
    stab = [e for e in v.entries if e.evidence_type == "stability"]
    assert len(stab) == 2  # the string value is skipped


def test_custom_entries_appended():
    ce = VaultEntry(section="§9", article_refs=["Article 72"], title="Custom item",
                    description="d", evidence_type="general")
    v = AnnexIVEvidenceVault().build_vault(custom_entries=[ce])
    assert any(e.title == "Custom item" for e in v.entries)


# ── Annex IV section numbering (Regulation (EU) 2024/1689, Annex IV points 1-9) ──
def test_section_catalogue_follows_annex_iv_points():
    assert list(_ANNEX_IV_SECTIONS) == [f"§{i}" for i in range(1, 10)]
    assert "performance metrics" in _ANNEX_IV_SECTIONS["§4"]
    assert "Article 9" in _ANNEX_IV_SECTIONS["§5"]
    assert "lifecycle" in _ANNEX_IV_SECTIONS["§6"]
    assert "standards" in _ANNEX_IV_SECTIONS["§7"].lower()
    assert "declaration of conformity" in _ANNEX_IV_SECTIONS["§8"]
    assert "Article 72" in _ANNEX_IV_SECTIONS["§9"]


def test_builder_entries_land_in_regulation_sections():
    v = AnnexIVEvidenceVault().build_vault(
        gb_result=GB,
        stability_result={"jaccard": 0.9},
        sae_features=[{"feature_id": 7, "activation": 1.2, "legal_risk_category": "gender_bias"}],
        multiagent_report={"chain_id": "c1", "chain_risk_level": "LOW", "annex_iv_text": "narrative"},
    )
    section_of = {}
    for e in v.entries:
        section_of.setdefault(e.evidence_type, set()).add(e.section)
    assert section_of["stability"] == {"§3"}
    assert section_of["sae_feature"] == {"§5"}       # risk management (Article 9)
    assert section_of["bias"] == {"§5"}              # multi-agent risk entries
    by_metric = {e.metric_name: e.section for e in v.entries if e.metric_name}
    assert by_metric["sufficiency"] == "§2"
    assert by_metric["f1"] == "§5"
    by_title = {e.title: e.section for e in v.entries}
    assert by_title["Technical standards and methodologies applied"] == "§7"
    assert by_title["EU Declaration of Conformity (placeholder)"] == "§8"
    assert v.to_dict()["sections_covered"] == ["§1", "§2", "§3", "§5", "§7", "§8"]


def test_html_section_titles_match_catalogue():
    html = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_html()
    assert "§8 &mdash; Copy of the EU declaration of conformity (Article 47)" in html
    assert "§5 &mdash; Detailed description of the risk management system" in html


# ── serialisation ──────────────────────────────────────────────────────────
def test_to_json_parses():
    parsed = json.loads(AnnexIVEvidenceVault().build_vault(gb_result=GB).to_json())
    assert parsed["n_entries"] > 0
    assert "compliance_summary" in parsed
    assert "entries" in parsed


def test_to_html_contains_status():
    html = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_html()
    assert "<" in html
    assert any(s in html for s in ("COMPLIANT", "MARGINAL", "NON-COMPLIANT"))


def test_articles_covered_collected():
    arts = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_dict()["articles_covered"]
    assert any("Article 15" in a for a in arts)


# ── static helper ──────────────────────────────────────────────────────────
def test_truncate_circuit_stringifies_keys_for_json_safety():
    big = {(i, 0): float(i) for i in range(30)}
    out = AnnexIVEvidenceVault._truncate_circuit(big, max_items=20)
    assert len(out) == 20
    assert all(isinstance(k, str) for k in out)  # JSON-serialisable keys
    # small circuits are also stringified now (the bug fix)
    assert AnnexIVEvidenceVault._truncate_circuit({(1, 0): 0.5}) == {"(1, 0)": 0.5}
    assert AnnexIVEvidenceVault._truncate_circuit("notadict") == "notadict"


# ── convenience function + file output ─────────────────────────────────────
def test_build_annex_iv_vault_writes_files(tmp_path):
    jp = tmp_path / "out" / "vault.json"
    hp = tmp_path / "out" / "vault.html"
    vault = build_annex_iv_vault(
        gb_result=GB, model_name="gpt2", provider="Acme Corp",
        output_json=str(jp), output_html=str(hp),
    )
    assert jp.exists() and hp.exists()
    assert vault.model_name == "gpt2"
    data = json.loads(jp.read_text())
    assert data["provider"] == "Acme Corp"
    assert "<" in hp.read_text()


def test_save_json_creates_dirs(tmp_path):
    v = AnnexIVEvidenceVault().build_vault(gb_result=GB)
    p = tmp_path / "nested" / "dir" / "v.json"
    v.save_json(str(p))
    assert p.exists()
    assert json.loads(p.read_text())["n_entries"] > 0
