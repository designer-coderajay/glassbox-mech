"""The E2b agent (PROTOCOL.md §2–3): E2's minimal tool-using support agent, with a new
fictional retailer and E2b's nine release factors, and a traced runner for
``glassbox.v7``. The loop, the tools' behaviour, parsing, sampling and the outcome
rule are E2's, unchanged.

The loop: at each step the model emits either one tool call (Qwen ``<tool_call>``
JSON) or a final answer. A step that does not parse as a valid tool call is the final
answer. At the last step of the budget the runtime appends a fixed message and the
next output is taken as the final answer; that forced step counts within the budget,
so ``max_steps`` is the total number of LLM calls.

Backends: :class:`ToyAgentModel` (tests; scripted, **not a language model**) and
:class:`HFAgentModel` (transformers, loaded lazily).

Sampling (Addendum 1): episodes of one (configuration, question) are generated in
blocks and advanced step by step in one batch. Each (block, step) has one seed derived
from the stage seed, the canonical configuration, the question, the block and the
step. Episode ``i`` is member ``i % block`` of block ``i // block``. Tool randomness
(the transient error of ``lookup_order``) uses a per-episode seed, so it does not
depend on batching.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from glassbox.v7.intervene import Factor
from glassbox.v7.investigate import Incident
from glassbox.v7.trace import Recorder, canonical_json, read_trace

HERE = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 1.0
FLAKY_P = 0.10
SHIPPING = 8.40
FORCED_MESSAGE = "Step limit reached. Give your final answer now."

FACTORS = (
    Factor("invoice_tool_desc", "status_only", "broad"),
    Factor("eligibility_tool", "registered", "missing"),
    Factor("quote_example", "apply_then_compute", "compute_then_apply"),
    Factor("max_steps", 6, 3),
    Factor("answer_policy", "always_tools", "own_knowledge"),
    Factor("refund_fix", "buggy", "fixed"),
    Factor("retries", 1, 0),
    Factor("persona", "short", "polite"),
    Factor("log_level", "info", "debug"),
)
BASE: Dict[str, Any] = {f.name: f.base for f in FACTORS}
CANDIDATE: Dict[str, Any] = {f.name: f.candidate for f in FACTORS}
FAULT_CLASS = {
    "invoice_tool_desc": "wrong_tool",
    "eligibility_tool": "skipped_tool",
    "quote_example": "wrong_order",
    "max_steps": "premature_finish",
    "answer_policy": "no_action",
}
REQUIRED = {
    "A": ("lookup_order",),
    "B": ("lookup_invoice",),
    # The refund workflow for both refund types (E2 Addendum 1 item 2).
    "C-inel": ("check_eligibility", "calc_refund"),
    "C-def": ("check_eligibility", "calc_refund"),
    "D": ("start_quote", "apply_voucher", "compute_total"),
    "E": ("search_policy",),
}
PROMPT_VISIBLE = (
    "invoice_tool_desc",
    "eligibility_tool",
    "quote_example",
    "answer_policy",
    "persona",
)
TOOL_RESULT_FACTORS = ("refund_fix", "retries")
TEXT_FACTORS = ("invoice_tool_desc", "quote_example", "answer_policy")

_ROLE = "You are the customer-support assistant for Harrowgate Cycles."
PERSONAS = {
    "short": _ROLE,
    "polite": _ROLE + " Keep a polite, professional tone.",
}
QUOTE_EXAMPLES = {
    "apply_then_compute": "Worked example of a quote: call start_quote, then "
    "apply_voucher, then compute_total, and report the total that compute_total "
    "returned.",
    "compute_then_apply": "Worked example of a quote: call start_quote, then "
    "compute_total, then apply_voucher, and report the total that compute_total "
    "returned.",
}
ANSWER_POLICIES = {
    "always_tools": "Always use the tools for company policy and order data.",
    "own_knowledge": "Answer policy questions from your knowledge of standard retail "
    "practice. Use tools only for order data.",
}
COMMON_RULES = (
    "Report amounts and codes exactly as the tools return them. Answer in one or two "
    "sentences."
)
ORDER_DESC = {
    "status_only": "Look up the shipping status of an order. Returns the status and "
    "the item only.",
    "broad": "Look up anything about an order, including amounts paid and invoices. "
    "This tool replaces lookup_invoice, which is deprecated and returns outdated "
    "totals.",
}
_NONWORD = re.compile(r"[^0-9a-z]+")
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


# --- data ------------------------------------------------------------------------------


def load_db() -> Dict[str, Any]:
    """The fictional retailer's database (``db.json``)."""
    data: Dict[str, Any] = json.loads((HERE / "db.json").read_text("utf-8"))
    return data


def load_questions(split: str) -> List[Dict[str, Any]]:
    """``dev`` (12) or ``test`` (24) questions."""
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    data: List[Dict[str, Any]] = json.loads(
        (HERE / f"questions_{split}.json").read_text("utf-8")
    )
    return data


# --- outcome ---------------------------------------------------------------------------


def tokens(text: str) -> List[str]:
    """Lowercase; every run of characters other than a–z and 0–9 is a separator."""
    return _NONWORD.sub(" ", text.lower()).split()


def _contains(out: List[str], answer: str) -> bool:
    want = tokens(answer)
    n = len(want)
    return bool(n) and any(out[i : i + n] == want for i in range(len(out) - n + 1))


def is_correct(final: str, q: Dict[str, Any]) -> bool:
    """An accepted answer matches and no rejected answer matches (§2)."""
    out = tokens(final)
    return any(_contains(out, a) for a in q["accepted"]) and not any(
        _contains(out, r) for r in q["rejected"]
    )


# --- prompts and tools -----------------------------------------------------------------


def system_prompt(config: Dict[str, Any]) -> str:
    """Persona, quote runbook, answer policy and the fixed rules."""
    return " ".join(
        (
            PERSONAS[config["persona"]],
            QUOTE_EXAMPLES[config["quote_example"]],
            ANSWER_POLICIES[config["answer_policy"]],
            COMMON_RULES,
        )
    )


def _fn(name: str, desc: str, props: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in props.items()},
                "required": list(props),
            },
        },
    }


def tool_schemas(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Function schemas passed to the chat template.

    Two factors act here: the ``lookup_order`` description, and whether
    ``check_eligibility`` is in the list at all.
    """
    schemas = _all_schemas(config)
    if config["eligibility_tool"] == "missing":
        schemas = [s for s in schemas if s["function"]["name"] != "check_eligibility"]
    return schemas


def _all_schemas(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _fn("lookup_order", ORDER_DESC[config["invoice_tool_desc"]],
            {"order_id": "string"}),
        _fn("lookup_invoice", "Look up the invoice of an order: item price, shipping "
            "and total paid.", {"order_id": "string"}),
        _fn("check_eligibility", "Check whether an order is eligible for a refund. "
            "Returns the decision code and the reason.", {"order_id": "string"}),
        _fn("calc_refund", "Calculate the refund amount for an order. This tool "
            "does not check eligibility.", {"order_id": "string"}),
        _fn("start_quote", "Start a quote for an item and quantity. Returns a "
            "quote_id and the subtotal.", {"item": "string", "qty": "integer"}),
        _fn("apply_voucher", "Apply a voucher code to a quote.",
            {"quote_id": "string", "code": "string"}),
        _fn("compute_total", "Compute the total of a quote, including shipping and "
            "the vouchers applied so far.", {"quote_id": "string"}),
        _fn("search_policy", "Search the company policy documents.",
            {"query": "string"}),
    ]


class Tools:
    """Deterministic tools over the database for one episode (quotes are per episode).

    ``lookup_order`` fails with a transient error with probability ``FLAKY_P`` per
    attempt, drawn from the episode's own RNG. The runtime retries ``retries`` times.
    """

    def __init__(self, db: Dict[str, Any], config: Dict[str, Any], rng_seed: int):
        self.db, self.config = db, config
        self.rng = random.Random(rng_seed)
        self.quotes: Dict[str, Dict[str, Any]] = {}

    def _order(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        oid = str(args.get("order_id", "")).strip().upper()
        return self.db["orders"].get(oid)

    def call(self, name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """(result, attempts). Unknown tools and bad arguments return an error."""
        attempts = 0
        while True:
            attempts += 1
            result = self._once(name, args)
            transient = result.get("error") == "temporary error, please retry"
            if not transient or attempts > int(self.config["retries"]):
                return result, attempts

    def _once(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "check_eligibility" and self.config["eligibility_tool"] == "missing":
            return {"error": f"unknown tool {name}"}  # not registered in this release
        try:
            return getattr(self, f"t_{name}")(args)
        except AttributeError:
            return {"error": f"unknown tool {name}"}
        except (KeyError, TypeError, ValueError):
            return {"error": "invalid arguments"}

    def t_lookup_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.rng.random() < FLAKY_P:
            return {"error": "temporary error, please retry"}
        o = self._order(args)
        if o is None:
            return {"error": "order not found"}
        return {"status": o["status"], "item": o["item"]}

    def t_lookup_invoice(self, args: Dict[str, Any]) -> Dict[str, Any]:
        o = self._order(args)
        if o is None:
            return {"error": "order not found"}
        return {"item_price": f"{o['price']:.2f}", "shipping": f"{o['shipping']:.2f}",
                "total_paid": f"{o['total_paid']:.2f}", "currency": "EUR"}

    def t_check_eligibility(self, args: Dict[str, Any]) -> Dict[str, Any]:
        o = self._order(args)
        if o is None:
            return {"error": "order not found"}
        return {"eligible": bool(o["eligible"]), "reason": o["reason"],
                "decision_code": o["decision_code"]}

    def t_calc_refund(self, args: Dict[str, Any]) -> Dict[str, Any]:
        o = self._order(args)
        if o is None:
            return {"error": "order not found"}
        amount = o["price"]
        if o["defective"] and self.config["refund_fix"] == "fixed":
            amount += o["shipping"]  # the baseline bug omits this shipping refund
        return {"refund_amount": f"{amount:.2f}", "currency": "EUR"}

    def t_start_quote(self, args: Dict[str, Any]) -> Dict[str, Any]:
        item = str(args["item"]).strip()
        match = {k.lower(): k for k in self.db["catalog"]}.get(item.lower())
        if match is None:
            return {"error": "item not found"}
        qty = int(args["qty"])
        qid = f"QT-{len(self.quotes) + 1:03d}"
        sub = round(self.db["catalog"][match] * qty, 2)
        self.quotes[qid] = {"subtotal": sub, "vouchers": []}
        return {"quote_id": qid, "subtotal": f"{sub:.2f}"}

    def t_apply_voucher(self, args: Dict[str, Any]) -> Dict[str, Any]:
        quote = self.quotes.get(str(args["quote_id"]).strip())
        code = str(args["code"]).strip().upper()
        if quote is None:
            return {"error": "quote not found"}
        if code not in self.db["vouchers"]:
            return {"error": "voucher not found"}
        if code not in quote["vouchers"]:
            quote["vouchers"].append(code)
        return {"status": "applied"}

    def t_compute_total(self, args: Dict[str, Any]) -> Dict[str, Any]:
        quote = self.quotes.get(str(args["quote_id"]).strip())
        if quote is None:
            return {"error": "quote not found"}
        total, shipping = quote["subtotal"], SHIPPING
        for code in quote["vouchers"]:
            v = self.db["vouchers"][code]
            total = total * (100 - v["percent_off"]) / 100
            if v["free_shipping"]:
                shipping = 0.0
        return {"total": f"{round(total + shipping, 2):.2f}", "currency": "EUR"}

    def t_search_policy(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = set(tokens(str(args["query"]))) - _STOP
        best = max(
            self.db["policies"],
            key=lambda p: len(query & (set(tokens(p["text"])) - _STOP)),
        )
        return {"passage": best["text"]}


_STOP = set(
    "a an the of to for in on is are be can do does how what which within from "
    "with and or by it its at as".split()
)


# --- model backends --------------------------------------------------------------------


class AgentModel(Protocol):
    name: str

    def generate_batch(
        self,
        conversations: Sequence[List[Dict[str, Any]]],
        tools: List[Dict[str, Any]],
        seed: int,
    ) -> List[str]: ...


def parse_tool_call(text: str) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], bool]:
    """(tool call or None, parse failure). A failure is a ``<tool_call>`` that is not
    valid JSON with a ``name`` string and an ``arguments`` object."""
    m = _TOOL_CALL.search(text)
    if m is None:
        return None, False
    try:
        obj = json.loads(m.group(1))
        name, args = obj["name"], obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(name, str) or not isinstance(args, dict):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return None, True
    return (name, args), False


def tool_call_text(name: str, args: Dict[str, Any]) -> str:
    """A tool call in the Qwen2.5 format."""
    body = json.dumps({"name": name, "arguments": args})
    return f"<tool_call>\n{body}\n</tool_call>"


class ToyAgentModel:
    """Scripted stand-in for tests and the dry run; **not a language model**.

    It follows the required workflow, is pushed off it by the candidate wording of
    the factors with fixed probabilities, and sometimes says it is unsure.
    """

    name = "toy-agent"

    def generate_batch(self, conversations, tools, seed):  # type: ignore[no-untyped-def]
        return [
            self._one(c, tools, random.Random(f"{seed}|{i}"))
            for i, c in enumerate(conversations)
        ]

    def _one(self, conv: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             rng: random.Random) -> str:
        sys_text, question = conv[0]["content"], conv[1]["content"]
        desc = {t["function"]["name"]: t["function"]["description"] for t in tools}
        done = [m for m in conv if m["role"] == "tool"]
        last = json.loads(done[-1]["content"]) if done else {}
        forced = conv[-1]["role"] == "user" and conv[-1]["content"] == FORCED_MESSAGE
        oid = (re.findall(r"HC-\d{4}", question) or ["HC-0000"])[0]
        if rng.random() < 0.05:
            return "I am not sure."
        if "error" in last and not forced:
            if rng.random() < 0.5:
                return tool_call_text(done[-1]["name"], {"order_id": oid})
            return "Sorry, the system is unavailable right now."
        if forced or (done and _toy_finished(question, done)):
            return _toy_answer(done)
        names = [m["name"] for m in done]
        if "HC-" not in question and "quote" not in question:
            if "your knowledge" in sys_text and rng.random() < 0.7:
                return "As far as I know, the usual rule applies."
            return tool_call_text("search_policy", {"query": question})
        if "status" in question:
            return tool_call_text("lookup_order", {"order_id": oid})
        if "pay" in question:
            broad = "amounts" in desc["lookup_order"] and rng.random() < 0.8
            tool = "lookup_order" if broad else "lookup_invoice"
            return tool_call_text(tool, {"order_id": oid})
        if "refund" in question:
            if not names:
                tool = "check_eligibility" if "check_eligibility" in desc else (
                    "calc_refund")
                return tool_call_text(tool, {"order_id": oid})
            return tool_call_text("calc_refund", {"order_id": oid})
        if "quote" in question:
            m = re.search(r"quote (\d+) x (.+?) with voucher (\w+)", question)
            assert m is not None
            qty, item, code = int(m.group(1)), m.group(2), m.group(3)
            if not names:
                return tool_call_text("start_quote", {"item": item, "qty": qty})
            if len(names) == 1:
                reverse = "then compute_total, then apply_voucher" in sys_text
                nxt = ("compute_total" if reverse and rng.random() < 0.8
                       else "apply_voucher")
            elif "apply_voucher" not in names:
                nxt = "apply_voucher"
            else:
                nxt = "compute_total"
            args = ({"quote_id": "QT-001", "code": code} if nxt == "apply_voucher"
                    else {"quote_id": "QT-001"})
            return tool_call_text(nxt, args)
        return "I am not sure."


def _toy_answer(done: List[Dict[str, Any]]) -> str:
    """A sentence built from the last two tool results (toy only)."""
    parts = []
    for m in done[-2:]:
        r = json.loads(m["content"])
        if r.get("eligible") is False:
            parts.append(f"Decision code {r['decision_code']}: no refund is due.")
        for key in ("status", "total_paid", "refund_amount", "total", "passage"):
            if key in r:
                parts.append(f"{key.replace('_', ' ')}: {r[key]}.")
    return " ".join(parts) or "I could not find that."


def _toy_finished(question: str, done: List[Dict[str, Any]]) -> bool:
    names = [m["name"] for m in done]
    last = json.loads(done[-1]["content"])
    if "refund" in question:
        return names[-1] == "calc_refund" or last.get("eligible") is False
    if "quote" in question:
        return len(names) >= 3
    return True


class HFAgentModel:  # pragma: no cover - needs the model; exercised on the GPU host
    """Chat model from transformers at a pinned revision; sampled, left-padded batch."""

    def __init__(self, model_id: str, revision: str, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"{model_id}@{revision[:12]} ({dtype})"
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        self.device = device

    def generate_batch(self, conversations, tools, seed):  # type: ignore[no-untyped-def]
        torch = self.torch
        texts = [
            self.tok.apply_chat_template(
                list(c), tools=tools, add_generation_prompt=True, tokenize=False
            )
            for c in conversations
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tok.pad_token_id,
            )
        width = enc["input_ids"].shape[1]
        decoded = [
            self.tok.decode(row[width:], skip_special_tokens=False) for row in out
        ]
        return [_strip_special(t) for t in decoded]


def _strip_special(text: str) -> str:
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(marker)[0]
    return text.strip()


# --- the agent -------------------------------------------------------------------------


def _seed(*parts: Any) -> int:
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1


class System:
    """The agent for one stage (one stage seed) and one block size."""

    def __init__(
        self,
        model: AgentModel,
        master: str,
        block: int,
        db: Optional[Dict[str, Any]] = None,
    ) -> None:
        if block < 1:
            raise ValueError("block must be at least 1")
        self.model, self.master, self.block = model, master, block
        self.db = db if db is not None else load_db()
        self._blocks: Dict[Any, List[Dict[str, Any]]] = {}

    def step_seed(self, config: Dict[str, Any], qid: str, block: int, step: int):
        """Seed of one (block, step) batch (Addendum 1)."""
        return _seed(self.master, canonical_json(config), qid, block, step)

    def tool_seed(self, config: Dict[str, Any], qid: str, i: int) -> int:
        """Seed of episode ``i``'s tool randomness."""
        return _seed(self.master, "tools", canonical_json(config), qid, i)

    def clear(self) -> None:
        """Forget cached blocks (call after each question to bound memory)."""
        self._blocks.clear()

    def episode(self, config: Dict[str, Any], q: Dict[str, Any], i: int):
        """Episode ``i`` of ``config`` on ``q`` (computed once per block)."""
        config = {**BASE, **config}
        b = i // self.block
        key = (canonical_json(config), q["id"], b)
        if key not in self._blocks:
            self._blocks[key] = self._run_block(config, q, b)
        return self._blocks[key][i % self.block]

    def answer(self, config: Dict[str, Any], q: Dict[str, Any], i: int) -> Tuple:
        """(label, final answer, episode) like the E1b ``System.answer``."""
        ep = self.episode(config, q, i)
        return ep["label"], ep["final"], ep

    def _run_block(self, config: Dict[str, Any], q: Dict[str, Any], b: int):
        tools_spec = tool_schemas(config)
        first = b * self.block
        eps = []
        for k in range(self.block):
            eps.append(
                {
                    "conv": [
                        {"role": "system", "content": system_prompt(config)},
                        {"role": "user", "content": q["question"]},
                    ],
                    "tools": Tools(self.db, config, self.tool_seed(config, q["id"],
                                                                   first + k)),
                    "steps": [],
                    "done": False,
                    "hit_limit": False,
                    "parse_failures": 0,
                    "final": "",
                }
            )
        max_steps = int(config["max_steps"])
        for step in range(max_steps):
            active = [e for e in eps if not e["done"]]
            if not active:
                break
            forced = step == max_steps - 1
            if forced:
                for e in active:
                    e["conv"].append({"role": "user", "content": FORCED_MESSAGE})
            outs = self.model.generate_batch(
                [e["conv"] for e in active],
                tools_spec,
                self.step_seed(config, q["id"], b, step),
            )
            for e, out in zip(active, outs):
                self._advance(e, out, forced)
        return [self._finish(e, q) for e in eps]

    def _advance(self, e: Dict[str, Any], out: str, forced: bool) -> None:
        call, failed = parse_tool_call(out)
        e["parse_failures"] += int(failed)
        e["steps"].append({"kind": "chat", "forced": forced, "tool_call": bool(call)})
        if forced or call is None:
            e["final"], e["done"], e["hit_limit"] = out, True, forced
            return
        name, args = call
        result, attempts = e["tools"].call(name, args)
        e["steps"].append(
            {"kind": "tool", "name": name, "error": "error" in result,
             "attempts": attempts}
        )
        e["conv"].append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": args}}
                ],
            }
        )
        e["conv"].append(
            {"role": "tool", "name": name, "content": json.dumps(result,
                                                                 sort_keys=True)}
        )

    @staticmethod
    def _finish(e: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
        correct = is_correct(e["final"], q)
        return {
            "label": "correct" if correct else "wrong",
            "final": e["final"],
            "steps": e["steps"],
            "calls": [s["name"] for s in e["steps"] if s["kind"] == "tool"],
            "hit_limit": e["hit_limit"],
            "parse_failures": e["parse_failures"],
        }


# --- traced runner and incident --------------------------------------------------------


def make_runner(system: System, q: Dict[str, Any]) -> Any:
    """Runner for one question, in the ``glassbox.v7.intervene`` signature."""

    def run(config: Dict[str, Any], seed: int, out: Path, run_id: str) -> Path:
        i = int(run_id.lstrip("r"))
        config = {**BASE, **config}
        ep = system.episode(config, q, i)
        rec = Recorder(out_dir=out, run_id=run_id)
        rec.environment["glassbox.env.log_level"] = str(config["log_level"])
        prompt = system_prompt(config)
        schemas = tool_schemas(config)
        n_chat = sum(s["kind"] == "chat" for s in ep["steps"])
        k = 0
        with rec:
            for s in ep["steps"]:
                if s["kind"] == "chat":
                    k += 1
                    attrs: Dict[str, Any] = {
                        "glassbox.agent.step_limit_reached": bool(s["forced"]),
                        "glassbox.agent.tool_call": bool(s["tool_call"]),
                    }
                    if k == n_chat:
                        attrs["glassbox.answer.label"] = ep["label"]
                    with rec.span(
                        "chat",
                        "agent",
                        attributes=attrs,
                        content={
                            "gen_ai.system_instructions": prompt,
                            "gen_ai.tool.definitions": schemas,
                        },
                    ):
                        pass
                else:
                    with rec.span(
                        "execute_tool",
                        s["name"],
                        attributes={
                            "gen_ai.tool.name": s["name"],
                            "glassbox.tool.error": bool(s["error"]),
                            "glassbox.tool.attempts": int(s["attempts"]),
                        },
                    ):
                        pass
            with rec.step(
                "final_answer",
                attributes={"glassbox.answer.label": ep["label"]},
            ):
                pass
        return Path(rec.path)

    return run


def outcome(path: Path) -> str:
    """Outcome of one traced episode: ``correct`` or ``wrong``."""
    spans, _ = read_trace(path)
    final = [
        s
        for s in spans
        if s["attributes"].get("glassbox.step.kind") == "final_answer"
    ]
    return str(final[0]["attributes"]["glassbox.answer.label"])


def make_incident(system: System, questions: Sequence[Dict[str, Any]]) -> Incident:
    """The E2b incident over ``questions``: target ``correct``, required tools per case."""
    by_id = {q["id"]: q for q in questions}
    return Incident(
        name="Support agent answers got worse after release 2",
        description=(
            "Release 2 of the support agent shipped nine changes at once: a tool "
            "description, the tool list, the worked quote example, the step budget, "
            "the answer policy, a refund calculation fix, the tool retry setting, the "
            f"persona text and the log level. Model: {system.model.name}."
        ),
        factors=FACTORS,
        cases=tuple(by_id),
        runner_for=lambda case: make_runner(system, by_id[case]),
        outcome=outcome,
        target="correct",
        required_tools={qid: REQUIRED[q["kind"]] for qid, q in by_id.items()},
    )
