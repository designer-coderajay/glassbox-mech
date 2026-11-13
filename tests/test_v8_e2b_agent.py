"""V8 E2b agent retry (PROTOCOL.md): new data, new factors, two-sided ground truth.

Uses the toy agent model only (scripted, not a language model).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
E2B = ROOT / "experiments" / "v8" / "e2b_agent_retry"


def _load(name):
    key = f"e2b_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, E2B / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


@pytest.fixture(scope="module")
def S():
    return _load("system")


@pytest.fixture(scope="module")
def R():
    return _load("reference")


# --- data ------------------------------------------------------------------------------


def test_make_data_is_deterministic():
    names = ("db.json", "questions_dev.json", "questions_test.json")
    before = {p: (E2B / p).read_bytes() for p in names}
    subprocess.run([sys.executable, str(E2B / "make_data.py")], check=True)
    for p, data in before.items():
        assert (E2B / p).read_bytes() == data


def test_question_counts_and_kinds(S):
    from collections import Counter

    dev, test = S.load_questions("dev"), S.load_questions("test")
    assert len(dev) == 12 and len(test) == 24
    assert Counter(q["kind"] for q in test) == {k: 4 for k in S.REQUIRED}
    assert Counter(q["kind"] for q in dev) == {k: 2 for k in S.REQUIRED}
    ids = [q["id"] for q in dev + test]
    assert len(ids) == len(set(ids))


def test_nothing_shared_with_earlier_experiments(S):
    mine = S.load_questions("dev") + S.load_questions("test")
    texts = {q["question"] for q in mine}
    db = S.load_db()
    v8 = ROOT / "experiments" / "v8"
    for old in ("e1_golden_rag", "e1b_directed_rag", "e2_golden_agent"):
        for split in ("dev", "test"):
            old_q = json.loads((v8 / old / f"questions_{split}.json").read_text())
            assert not texts & {q["question"] for q in old_q}
    e2_db = json.loads((v8 / "e2_golden_agent" / "db.json").read_text())
    assert db["company"] != e2_db["company"]
    assert not set(db["orders"]) & set(e2_db["orders"])
    assert not set(db["catalog"]) & set(e2_db["catalog"])
    assert not set(db["vouchers"]) & set(e2_db["vouchers"])
    assert not {p["text"] for p in db["policies"]} & {
        p["text"] for p in e2_db["policies"]
    }


KEY = re.compile(r"^(?:[A-Za-z]+|\d+(?:\.\d+)?|[A-Z]+-\d+)$")


def test_answer_keys_are_hard_to_paraphrase(S):
    """Every accepted answer is one word, one number or one code (PROTOCOL §2)."""
    for q in S.load_questions("dev") + S.load_questions("test"):
        assert q["accepted"], q["id"]
        for a in q["accepted"]:
            assert KEY.match(a), (q["id"], a)
        assert not set(q["accepted"]) & set(q["rejected"])


def _run_required(S, t, q):
    if q["kind"] == "D":
        m = re.search(r"quote (\d+) x (.+?) with voucher (\w+)", q["question"])
        r = t.call("start_quote", {"item": m.group(2), "qty": int(m.group(1))})[0]
        t.call("apply_voucher", {"quote_id": r["quote_id"], "code": m.group(3)})
        return t.call("compute_total", {"quote_id": r["quote_id"]})[0]
    if q["kind"] == "E":
        return t.call("search_policy", {"query": q["question"]})[0]
    oid = re.search(r"HC-\d{4}", q["question"]).group(0)
    if q["kind"] == "A":
        return t.call("lookup_order", {"order_id": oid})[0]["status"]
    if q["kind"] == "B":
        return t.call("lookup_invoice", {"order_id": oid})[0]["total_paid"]
    elig = t.call("check_eligibility", {"order_id": oid})[0]
    if not elig["eligible"]:
        return elig["decision_code"]
    return t.call("calc_refund", {"order_id": oid})[0]["refund_amount"]


def test_every_accepted_answer_is_reachable_with_tools(S):
    db = S.load_db()
    for q in S.load_questions("dev") + S.load_questions("test"):
        cfg = {**S.BASE, "refund_fix": "fixed", "retries": 50}
        text = json.dumps(_run_required(S, S.Tools(db, cfg, 0), q))
        assert S.is_correct(text.replace('"', " "), q), q["id"]


def test_policy_search_returns_the_right_passage(S):
    db = S.load_db()
    for q in S.load_questions("dev") + S.load_questions("test"):
        if q["kind"] == "E":
            passage = S.Tools(db, S.BASE, 0).call(
                "search_policy", {"query": q["question"]}
            )[0]["passage"]
            assert S.is_correct(passage, q), q["id"]


# --- factors ---------------------------------------------------------------------------


def test_factor_table_matches_protocol(S):
    assert [(f.name, f.base, f.candidate) for f in S.FACTORS] == [
        ("invoice_tool_desc", "status_only", "broad"),
        ("eligibility_tool", "registered", "missing"),
        ("quote_example", "apply_then_compute", "compute_then_apply"),
        ("max_steps", 6, 3),
        ("answer_policy", "always_tools", "own_knowledge"),
        ("refund_fix", "buggy", "fixed"),
        ("retries", 1, 0),
        ("persona", "short", "polite"),
        ("log_level", "info", "debug"),
    ]
    assert S.FAULT_CLASS == {
        "invoice_tool_desc": "wrong_tool", "eligibility_tool": "skipped_tool",
        "quote_example": "wrong_order", "max_steps": "premature_finish",
        "answer_policy": "no_action"}
    assert set(S.TEXT_FACTORS) < set(S.FAULT_CLASS)


def test_eligibility_tool_missing_from_schemas_and_runtime(S):
    names = lambda cfg: [t["function"]["name"] for t in S.tool_schemas(cfg)]  # noqa: E731
    assert "check_eligibility" in names(S.BASE) and len(names(S.BASE)) == 8
    gone = {**S.BASE, "eligibility_tool": "missing"}
    assert "check_eligibility" not in names(gone) and len(names(gone)) == 7
    t = S.Tools(S.load_db(), gone, 0)
    assert "unknown tool" in t.call("check_eligibility", {"order_id": "HC-4502"})[0][
        "error"]
    assert t.call("calc_refund", {"order_id": "HC-4502"})[0]["refund_amount"] == "33.50"


def test_prompt_factors(S):
    base, cand = S.system_prompt(S.BASE), S.system_prompt(S.CANDIDATE)
    assert "start_quote, then apply_voucher, then compute_total" in base
    assert "start_quote, then compute_total, then apply_voucher" in cand
    assert "Always use the tools" in base and "standard retail practice" in cand
    assert cand.count("polite, professional tone") == 1 and "polite" not in base
    assert "deprecated" in S.ORDER_DESC["broad"]
    for f in ("max_steps", "refund_fix", "retries", "log_level", "eligibility_tool"):
        assert S.system_prompt({**S.BASE, f: S.CANDIDATE[f]}) == base


def test_check_eligibility_returns_a_decision_code(S):
    t = S.Tools(S.load_db(), S.BASE, 0)
    r = t.call("check_eligibility", {"order_id": "hc-4502"})[0]
    assert r == {"eligible": False, "reason": "outside the 19-day return window",
                 "decision_code": "NR-417"}


def test_quote_uses_the_new_shipping_fee(S):
    t = S.Tools(S.load_db(), S.BASE, 0)
    q = t.call("start_quote", {"item": "Marlstone inner tube", "qty": 6})[0]
    assert t.call("compute_total", {"quote_id": q["quote_id"]})[0]["total"] == "51.90"
    t.call("apply_voucher", {"quote_id": q["quote_id"], "code": "SPOKE12"})
    assert t.call("compute_total", {"quote_id": q["quote_id"]})[0]["total"] == "38.28"


def test_toy_agent_shows_each_designed_fault(S):
    """The toy is scripted to follow the candidate wording; this checks the wiring."""
    from glassbox.v7.taxonomy import label_episode

    dev = {q["kind"]: q for q in S.load_questions("dev")}
    sysm = S.System(S.ToyAgentModel(), "t", 40)

    def labels(kind, factor):
        cfg = {**S.BASE, factor: S.CANDIDATE[factor]}
        eps = [sysm.episode(cfg, dev[kind], i) for i in range(40)]
        return [label_episode(e["calls"], S.REQUIRED[kind], e["hit_limit"],
                              e["label"] == "correct") for e in eps]

    for kind, factor in (("B", "invoice_tool_desc"), ("C-inel", "eligibility_tool"),
                         ("D", "quote_example"), ("D", "max_steps"),
                         ("E", "answer_policy")):
        got = labels(kind, factor)
        assert got.count(S.FAULT_CLASS[factor]) >= 20, (kind, factor, got[:5])


# --- two-sided reference labels --------------------------------------------------------


@pytest.mark.parametrize(
    "ci,want",
    [((-0.9, -0.2), "neg"), ((-0.9, -0.19), "unclear"), ((0.2, 0.5), "pos"),
     ((0.19, 0.5), "unclear"), ((-0.1, 0.1), "flat"), ((-0.11, 0.05), "unclear"),
     ((0.0, 0.0), "flat")],
)
def test_contrast_kind(R, ci, want):
    assert R.contrast_kind(*ci) == want


@pytest.mark.parametrize(
    "alone,rel,want",
    [("neg", "flat", "harmful"), ("flat", "neg", "harmful"), ("neg", "neg", "harmful"),
     ("unclear", "neg", "harmful"), ("pos", "flat", "helpful"),
     ("flat", "pos", "helpful"), ("neg", "pos", "mixed"), ("pos", "neg", "mixed"),
     ("flat", "flat", "null"), ("flat", "unclear", "small"),
     ("unclear", "unclear", "small")],
)
def test_pair_label(R, alone, rel, want):
    assert R.pair_label(alone, rel, "persona") == want
    assert R.pair_label(alone, rel, "log_level") == "null"


def test_one_sided_label_is_e2s_rule(R):
    assert R.one_sided_label("neg", "x") == "harmful"
    assert R.one_sided_label("pos", "x") == "helpful"
    assert R.one_sided_label("flat", "x") == "null"
    assert R.one_sided_label("unclear", "x") == "small"
    assert R.one_sided_label("neg", "log_level") == "null"


def test_key_suspect_rule(R):
    other = {"other": 180, "no_action": 0}
    assert R.key_suspect("A", 0.10, other) is True
    assert R.key_suspect("C-def", 0.0, other) is False  # wrong by design
    assert R.key_suspect("A", 0.60, other) is False  # baseline mostly right
    assert R.key_suspect("A", 0.10, {"other": 50, "no_action": 130}) is False
    assert R.key_suspect("A", 0.10, {}) is False


def test_reference_sees_a_harm_the_baseline_hides(S, R):
    """E2's blind spot: on C-def the baseline is 0, so a change that breaks only the
    new release is invisible alone and visible in the release."""
    q = next(x for x in S.load_questions("test") if x["kind"] == "C-def")
    # For this test only: a step limit that cuts the three-call refund task, and the
    # eligibility tool kept registered so the task stays three calls long.
    factors = [f for f in S.FACTORS if f.name not in ("max_steps", "eligibility_tool")]
    factors.append(type(S.FACTORS[0])("max_steps", 6, 2))
    r = R.reference_question(S.System(S.ToyAgentModel(), "e2b-reference", 50), q,
                             factors, 50)
    assert r["baseline_correct"] == 0  # the refund bug
    ms = r["factors"]["max_steps"]
    assert ms["alone"] == "flat" and ms["label_one_sided"] == "null"
    assert ms["in_release"] == "neg" and ms["label"] == "harmful"
    assert r["factors"]["refund_fix"]["label"] == "helpful"
    assert r["key_suspect"] is False
    cells = set(r["trajectories"])
    assert {"baseline", "candidate", "only_persona", "all_but_persona"} <= cells
    assert len(cells) == 2 + 2 * len(factors)
    assert len(r["trajectories"]["baseline"]["answers"]) == 3


# --- runner and scoring ----------------------------------------------------------------


def test_pilot_check_uses_both_contrasts():
    run = _load("run")
    dev = [{"id": "d1", "kind": "D"}, {"id": "c1", "kind": "C-def"}]
    acc = {"d1": {"baseline": 1.0, "candidate": 0.0}, "c1": {"baseline": 0.0,
                                                               "candidate": 0.9}}
    for f in run.PILOT_TARGETS:
        for row in acc.values():
            row[f"only_{f}"] = row["baseline"]
            row[f"all_but_{f}"] = row["candidate"]
    acc["d1"]["all_but_quote_example"] = 0.6  # visible only inside the release
    acc["c1"]["only_refund_fix"] = 0.95
    chk = run.pilot_check(dev, acc)
    assert chk["quote_example"]["effect_present"] is True
    assert chk["quote_example"]["moves"]["d1"] == {"alone": 0.0, "in_release": -0.6}
    assert chk["refund_fix"]["effect_present"] is True
    assert chk["max_steps"]["effect_present"] is False
    assert run.REMAINING == 24 * 20 * 200 + 24 * 20 * 30 + 24 * 6 * 60


def test_score_excludes_key_suspects_and_reports_one_sided():
    run, sc = _load("run"), _load("score")
    names = [f for f in sc.FAULT_CLASS] + ["refund_fix", "retries", "persona",
                                           "log_level"]
    arm = {"run": True, "verified": True, "verified_two_sided": True}
    per_q = {}
    for i in range(9):
        fac = {f: {"label": "null", "label_one_sided": "null"} for f in names}
        fac["invoice_tool_desc"] = {"label": "harmful", "label_one_sided": "harmful"}
        fac["max_steps"] = {"label": "harmful", "label_one_sided": "null"}
        per_q[f"q{i}"] = {
            "reference": {"factors": fac, "regressed": True, "key_suspect": i == 8},
            "methods": {"GB": ["invoice_tool_desc", "max_steps"], "B0": names,
                        "B1": ["persona"], "B2": [],
                        "GB_attributed": ["invoice_tool_desc"],
                        "GB_fault": {"invoice_tool_desc": "wrong_tool",
                                     "max_steps": "premature_finish"}},
            "recovery": {"arms": {"gb_fix": arm, "b2_fix": {"run": False},
                                  "wrong_fix": {"run": True, "verified": False},
                                  "undirected_fix": arm},
                         "gb_minus_undirected_ci": None},
        }
    out = run._score(per_q)
    c, one = out["criteria"], out["criteria_one_sided_labels"]
    assert c["key_suspect_questions"] == ["q8"]
    assert c["n_harmful_large"] == 16 and c["G10"]["hits"] == 0
    assert c["overall"] == "PASS"
    assert one["n_harmful_large"] == 8 and one["G10"]["hits"] == 8  # E2's blind spot
    for i in range(5):
        per_q[f"q{i}"]["reference"]["key_suspect"] = True
    assert run._score(per_q)["criteria"]["overall"] == "INCONCLUSIVE"


def test_sealed_hashes_match_addendum():
    text = (E2B / "PROTOCOL.md").read_text()
    if "## Addendum 1" not in text:
        pytest.skip("Addendum 1 not written yet")
    for name in ("questions_test.json", "questions_dev.json", "db.json"):
        digest = hashlib.sha256((E2B / name).read_bytes()).hexdigest()
        assert digest in text, name


def test_toy_end_to_end(tmp_path):
    run = _load("run")
    args = ["--backend", "toy", "--limit", "2", "--gate-min", "0"]
    assert run.main(["--stage", "pilot", "--out", str(tmp_path / "p"), *args]) == 0
    res = json.loads((tmp_path / "p" / "results.json").read_text())
    assert res["pilot"]["episodes"] == 2 * 20 * 30
    assert run.main(["--stage", "test", "--out", str(tmp_path / "t"), *args]) == 0
    res = json.loads((tmp_path / "t" / "results.json").read_text())
    assert set(res["criteria"]) >= {f"G{i}" for i in range(1, 11)}
    assert "criteria_one_sided_labels" in res
    assert res["protocol"].startswith("experiments/v8/e2b_agent_retry/")
    assert run.main(["--stage", "test", "--out", str(tmp_path / "t"), *args]) == 0
