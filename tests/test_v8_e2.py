"""V8 E2 golden agent case (PROTOCOL.md): data, tools, agent loop, runner, scoring.

Uses the toy agent model only (scripted, not a language model).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
E2 = ROOT / "experiments" / "v8" / "e2_golden_agent"


def _load(name):
    key = f"e2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, E2 / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


@pytest.fixture(scope="module")
def S():
    return _load("system")


# --- data ------------------------------------------------------------------------------


def test_make_data_is_deterministic(tmp_path):
    before = {p: (E2 / p).read_bytes() for p in
              ("db.json", "questions_dev.json", "questions_test.json")}
    subprocess.run([sys.executable, str(E2 / "make_data.py")], check=True)
    for p, data in before.items():
        assert (E2 / p).read_bytes() == data


def test_question_counts_and_kinds(S):
    from collections import Counter

    dev, test = S.load_questions("dev"), S.load_questions("test")
    assert len(dev) == 12 and len(test) == 24
    assert Counter(q["kind"] for q in test) == {k: 4 for k in S.REQUIRED}
    assert Counter(q["kind"] for q in dev) == {k: 2 for k in S.REQUIRED}
    ids = [q["id"] for q in dev + test]
    assert len(ids) == len(set(ids))
    with pytest.raises(ValueError):
        S.load_questions("train")


def test_nothing_shared_with_e1_or_e1b(S):
    texts = {q["question"] for q in S.load_questions("dev") + S.load_questions("test")}
    db = S.load_db()
    for old in ("e1_golden_rag", "e1b_directed_rag"):
        d = ROOT / "experiments" / "v8" / old
        old_q = {
            q["question"]
            for split in ("dev", "test")
            for q in json.loads((d / f"questions_{split}.json").read_text())
        }
        assert not texts & old_q
        corpus = (d / "corpus.json").read_text()
        assert db["company"] not in corpus
        assert not any(p["text"] in corpus for p in db["policies"])


def test_every_accepted_answer_is_reachable_with_tools(S):
    """Running the required tools under the baseline (with the fix for C-def) gives
    data that contains the accepted answer and no rejected one."""
    db = S.load_db()
    for q in S.load_questions("dev") + S.load_questions("test"):
        cfg = {**S.BASE, "refund_fix": "fixed", "retries": 50}
        t = S.Tools(db, cfg, 0)
        text = json.dumps(_run_required(S, t, q))
        assert S.is_correct(text.replace('"', " "), q), q["id"]


def _run_required(S, t, q):
    import re

    if q["kind"] == "D":
        m = re.search(r"quote (\d+) x (.+?) with voucher (\w+)", q["question"])
        r = t.call("start_quote", {"item": m.group(2), "qty": int(m.group(1))})[0]
        t.call("apply_voucher", {"quote_id": r["quote_id"], "code": m.group(3)})
        return t.call("compute_total", {"quote_id": r["quote_id"]})[0]
    if q["kind"] == "E":
        return t.call("search_policy", {"query": q["question"]})[0]
    oid = re.search(r"QS-\d{4}", q["question"]).group(0)
    if q["kind"] == "A":
        return t.call("lookup_order", {"order_id": oid})[0]["status"]
    if q["kind"] == "B":
        return t.call("lookup_invoice", {"order_id": oid})[0]["total_paid"]
    elig = t.call("check_eligibility", {"order_id": oid})[0]
    if not elig["eligible"]:
        return "not eligible"
    return t.call("calc_refund", {"order_id": oid})[0]["refund_amount"]


# --- outcome and parsing ---------------------------------------------------------------


def test_is_correct_handles_currency_and_rejections(S):
    q = {"accepted": ["59.95"], "rejected": ["54.00"]}
    assert S.is_correct("The refund is €59.95.", q)
    assert S.is_correct("EUR 59.95", q)
    assert not S.is_correct("54.00 or 59.95", q)
    assert not S.is_correct("59.9", q)
    q2 = {"accepted": S.load_questions("test")[8]["accepted"], "rejected": []}
    assert S.is_correct("Sorry, this order isn't eligible for a refund.", q2)


def test_parse_tool_call(S):
    ok = S.tool_call_text("lookup_order", {"order_id": "QS-4417"})
    assert S.parse_tool_call(ok) == (("lookup_order", {"order_id": "QS-4417"}), False)
    assert S.parse_tool_call("The status is shipped.") == (None, False)
    assert S.parse_tool_call("<tool_call>{not json}</tool_call>") == (None, True)
    assert S.parse_tool_call('<tool_call>{"arguments": {}}</tool_call>') == (None, True)
    s = '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
    assert S.parse_tool_call(s) == (("x", {"a": 1}), False)


# --- tools and factors -----------------------------------------------------------------


def test_refund_fix_changes_only_defective_refunds(S):
    db = S.load_db()
    for fix, want in (("buggy", "64.50"), ("fixed", "71.40")):
        t = S.Tools(db, {**S.BASE, "refund_fix": fix}, 0)
        assert t.call("calc_refund", {"order_id": "QS-7023"})[0]["refund_amount"] == want
        r = t.call("calc_refund", {"order_id": "qs-6011"})[0]
        assert r["refund_amount"] == "58.00"  # calc_refund never checks eligibility


def test_quote_order_matters(S):
    t = S.Tools(S.load_db(), S.BASE, 0)
    qid = t.call("start_quote", {"item": "ridgeline trekking pole", "qty": 3})[0]
    before = t.call("compute_total", {"quote_id": qid["quote_id"]})[0]["total"]
    t.call("apply_voucher", {"quote_id": qid["quote_id"], "code": "trail15"})
    after = t.call("compute_total", {"quote_id": qid["quote_id"]})[0]["total"]
    assert (before, after) == ("108.90", "86.70")


def test_tool_errors(S):
    t = S.Tools(S.load_db(), S.BASE, 0)
    assert "error" in t.call("no_such_tool", {})[0]
    assert "error" in t.call("lookup_invoice", {"order_id": "QS-0000"})[0]
    assert "error" in t.call("start_quote", {"item": "x"})[0]


def test_retries_reduce_transient_errors(S):
    db = S.load_db()

    def err_rate(retries):
        n = 0
        for seed in range(2000):
            r, _ = S.Tools(db, {**S.BASE, "retries": retries}, seed).call(
                "lookup_order", {"order_id": "QS-4417"}
            )
            n += "error" in r
        return n / 2000

    assert 0.07 < err_rate(0) < 0.13
    assert err_rate(1) < 0.03


def test_prompt_and_schema_factors(S):
    base, cand = S.system_prompt(S.BASE), S.system_prompt(S.CANDIDATE)
    assert "apply the voucher, then compute" in base
    assert "compute the total, then apply" in cand
    assert "without calling tools" in cand and "without calling tools" not in base
    d = {t["function"]["name"]: t["function"]["description"]
         for t in S.tool_schemas(S.CANDIDATE)}
    assert "amounts" in d["lookup_order"] and "automatically" in d["calc_refund"]
    for f in ("max_steps", "refund_fix", "retries", "log_level"):
        cfg = {**S.BASE, f: S.CANDIDATE[f]}
        assert S.system_prompt(cfg) == base
        assert S.tool_schemas(cfg) == S.tool_schemas(S.BASE)


def test_factor_table_matches_protocol(S):
    assert [f.name for f in S.FACTORS] == [
        "invoice_tool_desc", "refund_tool_desc", "runbook_order", "max_steps",
        "answer_policy", "refund_fix", "retries", "persona", "log_level",
    ]
    assert S.BASE["max_steps"] == 6 and S.CANDIDATE["max_steps"] == 2
    assert S.BASE["retries"] == 1 and S.CANDIDATE["retries"] == 0
    assert set(S.FAULT_CLASS.values()) == {
        "wrong_tool", "skipped_tool", "wrong_order", "premature_finish", "no_action"}


# --- the agent loop --------------------------------------------------------------------


class Scripted:
    """Returns the queued outputs in order, for every conversation in the batch."""

    name = "scripted"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate_batch(self, conversations, tools, seed):
        self.calls.append((len(conversations), seed, [list(c) for c in conversations]))
        out = self.outputs.pop(0)
        return [out] * len(conversations)


def test_loop_tool_then_answer(S):
    q = S.load_questions("dev")[0]
    m = Scripted([S.tool_call_text("lookup_order", {"order_id": "QS-4417"}),
                  "It is awaiting courier pickup."])
    sysm = S.System(m, "x", 3)
    ep = sysm.episode({**S.BASE, "retries": 50}, q, 0)
    assert ep["calls"] == ["lookup_order"] and ep["label"] == "correct"
    assert ep["hit_limit"] is False
    assert [c[0] for c in m.calls] == [3, 3]  # one batch per step
    conv = m.calls[1][2][0]
    assert conv[-1]["role"] == "tool" and conv[-2]["tool_calls"]


def test_forced_final_counts_in_budget(S):
    q = S.load_questions("dev")[0]
    call = S.tool_call_text("lookup_order", {"order_id": "QS-4417"})
    m = Scripted([call, call])
    ep = S.System(m, "x", 1).episode({**S.BASE, "max_steps": 2}, q, 0)
    assert len(m.calls) == 2
    assert m.calls[1][2][0][-1] == {"role": "user", "content": S.FORCED_MESSAGE}
    assert ep["hit_limit"] is True and ep["final"] == call
    assert ep["label"] == "wrong" and ep["calls"] == ["lookup_order"]


def test_parse_failure_is_final_answer(S):
    q = S.load_questions("dev")[0]
    ep = S.System(Scripted(["<tool_call>{oops</tool_call>"]), "x", 1).episode(
        S.BASE, q, 0)
    assert ep["parse_failures"] == 1 and ep["calls"] == [] and ep["label"] == "wrong"


def test_blocks_are_deterministic_and_config_specific(S):
    q = S.load_questions("dev")[4]
    a = S.System(S.ToyAgentModel(), "m", 10)
    b = S.System(S.ToyAgentModel(), "m", 10)
    run = [a.answer(S.BASE, q, i)[1] for i in range(20)]
    assert run == [b.answer(S.BASE, q, i)[1] for i in range(20)]
    assert a.step_seed(S.BASE, q["id"], 0, 0) != a.step_seed(
        {**S.BASE, "log_level": "debug"}, q["id"], 0, 0)
    assert a.step_seed(S.BASE, q["id"], 0, 0) != a.step_seed(S.BASE, q["id"], 0, 1)
    c = S.System(S.ToyAgentModel(), "other", 10)
    assert run != [c.answer(S.BASE, q, i)[1] for i in range(20)]
    with pytest.raises(ValueError):
        S.System(S.ToyAgentModel(), "m", 0)


def test_tool_seed_is_per_episode(S):
    sysm = S.System(S.ToyAgentModel(), "m", 5)
    seeds = {sysm.tool_seed(S.BASE, "a_4417", i) for i in range(10)}
    assert len(seeds) == 10


# --- runner, outcome, incident ---------------------------------------------------------


def test_runner_trace_outcome_and_faults(S, tmp_path):
    from glassbox.v7.taxonomy import episode_from_trace

    q = next(x for x in S.load_questions("dev") if x["kind"] == "D")
    sysm = S.System(S.ToyAgentModel(), "m", 5)
    run = S.make_runner(sysm, q)
    for cfg, name in ((S.BASE, "b"), ({**S.BASE, "max_steps": 2}, "c")):
        p = run(cfg, 0, tmp_path / name, "r000")
        ep = sysm.episode(cfg, q, 0)
        assert S.outcome(p) == ep["label"]
        calls, limit = episode_from_trace(p)
        assert calls == ep["calls"] and limit == ep["hit_limit"]
    assert limit is True


def test_incident(S):
    qs = S.load_questions("dev")[:3]
    inc = S.make_incident(S.System(S.ToyAgentModel(), "m", 5), qs)
    assert inc.target == "correct"
    assert inc.required_tools["a_4417"] == ("lookup_order",)
    assert inc.required_tools[qs[2]["id"]] == S.REQUIRED[qs[2]["kind"]]
    assert "toy-agent" in inc.description


def test_sealed_hashes_match_addendum():
    text = (E2 / "PROTOCOL.md").read_text()
    if "## Addendum 1" not in text:
        pytest.skip("Addendum 1 not written yet")
    for name in ("questions_test.json", "questions_dev.json", "db.json"):
        digest = hashlib.sha256((E2 / name).read_bytes()).hexdigest()
        assert digest in text, name


# --- reference, methods, recovery ------------------------------------------------------


def test_reference_question_toy(S):
    ref = _load("reference")
    q = next(x for x in S.load_questions("test") if x["kind"] == "C-def")
    sysm = S.System(S.ToyAgentModel(), "e2-reference", 50)
    r = ref.reference_question(sysm, q, S.FACTORS, 50)
    assert set(r["factors"]) == {f.name for f in S.FACTORS}
    assert r["factors"]["refund_fix"]["label"] == "helpful"  # toy: bug fix helps
    assert r["factors"]["log_level"]["label"] == "null"
    assert set(r["trajectories"]) == {"baseline", "candidate"} | {
        f"only_{f.name}" for f in S.FACTORS}
    assert r["trajectories"]["only_max_steps"]["forced_final"] > 0
    assert sysm._blocks == {}  # cleared after the question


def test_b1_and_b2_rules(S, tmp_path):
    meth = _load("methods")
    q = next(x for x in S.load_questions("dev") if x["kind"] == "A")
    sysm = S.System(S.ToyAgentModel(), "m", 5)
    run = S.make_runner(sysm, q)
    changed = [f.name for f in S.FACTORS]
    a = run(S.BASE, 0, tmp_path / "a", "r000")
    b = run({**S.BASE, "persona": "long"}, 0, tmp_path / "b", "r000")
    assert meth.b1_blame(str(a), str(b), changed) == list(S.PROMPT_VISIBLE)
    c = run({**S.BASE, "log_level": "debug"}, 0, tmp_path / "c", "r000")
    assert "log_level" not in meth.b1_blame(str(a), str(c), changed)
    assert meth.b2_blame("correct", {"x": "correct"}) == []
    assert meth.b2_blame("wrong", {"x": "correct", "y": "wrong"}) == ["x"]


def test_fault_labels():
    meth = _load("methods")
    from glassbox.v7.taxonomy import FAULTS

    zero = {f: 0 for f in FAULTS}
    counts = {"only_a": {**zero, "wrong_tool": 6}, "only_b": {**zero, "other": 2}}
    assert meth.fault_labels(counts, ["a", "b"]) == {"a": "wrong_tool", "b": None}


def test_run_methods_toy(S, tmp_path):
    from glassbox.v7.intervene import InterventionSettings

    meth = _load("methods")
    q = next(x for x in S.load_questions("test") if x["kind"] == "D")
    sysm = S.System(S.ToyAgentModel(), "e2-main", 30)
    rows = meth.run_methods(sysm, [q], tmp_path,
                            InterventionSettings(30, 999, 0.05, "e2-main"))
    row = rows[q["id"]]
    assert "runbook_order" in row["GB"]
    assert row["GB_fault"]["runbook_order"] == "wrong_order"
    assert row["GB_fault"]["max_steps"] == "premature_finish"
    assert row["B0"] == [f.name for f in S.FACTORS]
    assert set(row["fault_counts"]) >= {"baseline", "candidate"}
    with pytest.raises(ValueError):
        meth.run_methods(sysm, [q], tmp_path / "x",
                         InterventionSettings(5, 99, 0.05, "s"))


def test_recovery_arms_and_g8_ci(S):
    rec = _load("recovery")
    q = next(x for x in S.load_questions("test") if x["kind"] == "C-def")
    sysm = S.System(S.ToyAgentModel(), "e2-recovery", 60)
    r = rec.recovery_question(
        sysm, q, ["max_steps"], [], ["max_steps", "refund_fix"],
        S.BASE, S.CANDIDATE, 60)
    assert set(r["arms"]) == {"gb_fix", "wrong_fix", "b2_fix", "undirected_fix"}
    assert r["arms"]["wrong_fix"]["reverted"] == ["persona"]
    assert r["arms"]["b2_fix"]["run"] is False
    assert r["gb_minus_undirected_ci"] is not None
    assert r["gb_minus_undirected_ci"][0] >= 0.20  # toy: keeping the fix matters
    same = rec.recovery_question(sysm, q, ["max_steps"], [], ["max_steps"],
                                 S.BASE, S.CANDIDATE, 60)
    assert same["gb_minus_undirected_ci"] is None
    assert rec.fix_config({"a": 2, "b": 2}, {"a": 1, "b": 1}, ["a"]) == {"a": 1, "b": 2}


# --- scoring ---------------------------------------------------------------------------


def _scoring_inputs(S, gb_fault="wrong_tool", helpful_blamed=False):
    names = [f.name for f in S.FACTORS]
    labels, regressed, methods, recovery = {}, {}, {}, {}
    for i in range(8):
        q = f"q{i}"
        row = {f: "null" for f in names}
        row["invoice_tool_desc"] = "harmful"
        if i < 4:
            row["refund_fix"] = "helpful"
        labels[q], regressed[q] = row, True
        gb = ["invoice_tool_desc"] + (["refund_fix"] if helpful_blamed else [])
        methods[q] = {
            "GB": gb, "B0": names, "B1": ["persona", "invoice_tool_desc"], "B2": [],
            "GB_attributed": gb + ["refund_fix"],
            "GB_fault": {"invoice_tool_desc": gb_fault},
        }
        arm = {"run": True, "verified": True, "verified_two_sided": True}
        recovery[q] = {
            "arms": {"gb_fix": arm, "b2_fix": {"run": False},
                     "wrong_fix": {"run": True, "verified": False},
                     "undirected_fix": {"run": True, "verified": False}},
            "gb_minus_undirected_ci": [0.5, 0.9] if i < 4 else None,
        }
    return labels, regressed, methods, recovery


def test_criteria_pass_case(S):
    sc = _load("score")
    c = sc.criteria(*_scoring_inputs(S))
    for g in ("G1", "G2", "G3", "G5", "G6", "G7", "G9", "G10"):
        assert c[g]["pass"] is True, g
    assert c["G4"]["pass"] is True and c["G4"]["n"] == 4
    assert c["G8"]["pass"] is True and c["G8"]["n"] == 4
    assert c["overall"] == "PASS"
    d = sc.descriptive(*_scoring_inputs(S))
    assert d["coverage"]["wrong_tool"]["produced"] is True
    assert d["coverage"]["no_action"]["produced"] is False
    assert d["per_factor"]["invoice_tool_desc"]["harmful"] == 8


def test_criteria_fail_cases(S):
    sc = _load("score")
    c = sc.criteria(*_scoring_inputs(S, gb_fault="skipped_tool"))
    assert c["G9"]["pass"] is False and c["overall"] == "FAIL"
    c = sc.criteria(*_scoring_inputs(S, helpful_blamed=True))
    assert c["G4"]["pass"] is False
    c = sc.criteria(*_scoring_inputs(S, gb_fault=None))
    assert c["G9"]["testable"] is False and c["G9"]["unlabelled"] == 8


def test_criteria_inconclusive_when_too_few(S):
    sc = _load("score")
    keep = ["q0", "q1", "q2"]
    parts = [{k: v[k] for k in keep} for v in _scoring_inputs(S)]
    assert sc.criteria(*parts)["overall"] == "INCONCLUSIVE"


# --- runner ----------------------------------------------------------------------------


def test_gate_scores_and_pilot_check():
    run = _load("run")
    eps = [{"label": "correct", "calls": ["a"], "parse_failures": 0}] * 9 + [
        {"label": "wrong", "calls": [], "parse_failures": 1}]
    assert run.gate_scores(eps) == {"accuracy": 0.9, "parse_rate": 0.9}
    dev = [{"id": "b1", "kind": "B"}, {"id": "c1", "kind": "C-def"}]
    acc = {
        "b1": {"baseline": 0.9, "only_invoice_tool_desc": 0.5},
        "c1": {"baseline": 0.0, "only_refund_fix": 0.1},
    }
    for f in run.PILOT_TARGETS:
        for row in acc.values():
            row.setdefault(f"only_{f}", row["baseline"])
    chk = run.pilot_check(dev, acc)
    assert chk["invoice_tool_desc"]["effect_present"] is True
    assert chk["refund_fix"]["effect_present"] is False
    assert chk["answer_policy"]["moves"] == {}


def test_gate_min_only_changeable_for_toy(tmp_path):
    run = _load("run")
    with pytest.raises(SystemExit):
        run.main(["--stage", "pilot", "--out", str(tmp_path), "--backend", "hf",
                  "--gate-min", "0"])


def test_toy_end_to_end(tmp_path):
    run = _load("run")
    args = ["--backend", "toy", "--limit", "2", "--gate-min", "0"]
    assert run.main(["--stage", "pilot", "--out", str(tmp_path / "p"), *args]) == 0
    res = json.loads((tmp_path / "p" / "results.json").read_text())
    assert res["pilot"]["episodes"] == 2 * 11 * 30
    assert set(res["pilot"]["check"]) == set(run.PILOT_TARGETS)
    assert run.main(["--stage", "test", "--out", str(tmp_path / "t"), *args]) == 0
    res = json.loads((tmp_path / "t" / "results.json").read_text())
    assert set(res["criteria"]) >= {f"G{i}" for i in range(1, 11)}
    assert (tmp_path / "t" / "pip_freeze.txt").exists()
    # resume: same command and --out reuse the checkpoint
    assert run.main(["--stage", "test", "--out", str(tmp_path / "t"), *args]) == 0
