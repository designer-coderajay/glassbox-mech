#!/usr/bin/env python3
"""Write the E2b data (PROTOCOL.md §2): a new fictional retailer's database and the
dev and test questions. Deterministic: running it again rewrites the same bytes (a
test checks this).

The company, every order, item, voucher and policy are invented for E2b. Nothing is
taken from E1, E1b or E2 (a test checks this).

Every accepted answer is a number, a code or a single distinctive word, so that a
correct answer is hard to paraphrase past the key (PROTOCOL §0 point 3).

Usage:
    python3 experiments/v8/e2b_agent_retry/make_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
COMPANY = "Harrowgate Cycles"
SHIPPING = 8.40

# --- type A: order status (one distinctive word each) ----------------------------------
STATUS = [
    ("HC-2104", "backordered"),
    ("HC-2137", "quarantined"),
    ("HC-2169", "rerouted"),
    ("HC-2192", "impounded"),
    ("HC-2215", "palletised"),
    ("HC-2248", "consolidating"),
]
# --- type B: amount paid ---------------------------------------------------------------
PAID = [
    ("HC-3306", "Wend gravel tyre x2", 97.30),
    ("HC-3329", "Carrick pannier rack", 64.15),
    ("HC-3351", "Tollgate floor pump", 51.85),
    ("HC-3374", "Drover saddle bag", 38.95),
    ("HC-3398", "Inglenook dynamo light set", 143.60),
    ("HC-3417", "Sett cobble mudguards", 46.70),
]
# --- type C: refunds (ineligible / eligible defective) ---------------------------------
INELIGIBLE = [
    ("HC-4502", "Byre chain lock", 33.50, "outside the 19-day return window", "NR-417"),
    ("HC-4526", "Fold commuter helmet", 72.00, "safety item, opened", "NR-263"),
    ("HC-4549", "Lych bar tape", 14.25, "outside the 19-day return window", "NR-508"),
    ("HC-4571", "Garth cycling shoes", 118.00, "worn outdoors", "NR-342"),
    ("HC-4593", "Thwaite bottle cage", 11.80, "clearance item", "NR-695"),
    ("HC-4618", "Reeve track pump", 57.40, "outside the 19-day return window", "NR-781"),
]
DEFECTIVE = [
    ("HC-5703", "Hythe rear derailleur", 86.20, "cage bent on arrival"),
    ("HC-5727", "Stile pedal set", 43.75, "bearing seized"),
    ("HC-5748", "Coppice front light", 39.10, "does not charge"),
    ("HC-5762", "Wold brake lever", 28.60, "cracked clamp"),
    ("HC-5785", "Ginnel bike computer", 74.90, "screen dead on arrival"),
    ("HC-5809", "Beck wheel skewer", 16.30, "thread stripped"),
]
# --- type D: quotes with a voucher -----------------------------------------------------
CATALOG = {
    "Marlstone inner tube": 7.25,
    "Fellgate chain": 23.80,
    "Ashlar brake pads": 12.60,
    "Quoin bottle": 9.90,
    "Lintel kickstand": 18.50,
    "Mullion bell": 6.75,
}
VOUCHERS = {
    "SPOKE12": {"percent_off": 12, "free_shipping": True},
    "GEAR08": {"percent_off": 8, "free_shipping": False},
    "CHAIN25": {"percent_off": 25, "free_shipping": True},
}
QUOTES = [
    ("Marlstone inner tube", 6, "SPOKE12"),
    ("Fellgate chain", 2, "GEAR08"),
    ("Ashlar brake pads", 4, "CHAIN25"),
    ("Quoin bottle", 5, "GEAR08"),
    ("Lintel kickstand", 3, "SPOKE12"),
    ("Mullion bell", 8, "CHAIN25"),
]
# --- type E: policies whose numbers differ from common practice ------------------------
POLICIES = [
    {"id": "returns", "text": f"{COMPANY} returns: unused items can be returned "
     "within 19 days of delivery. Items worn outdoors cannot be returned."},
    {"id": "assembly", "text": "Assembly: a bike bought online is assembled in the "
     "workshop for a flat fee of 37 EUR."},
    {"id": "free_shipping", "text": "Shipping: standard shipping costs 8.40 EUR. "
     "Orders of 110 EUR or more ship free within the EU."},
    {"id": "frame_warranty", "text": "Warranty: frames carry a warranty of 42 months "
     "from the date of delivery."},
    {"id": "service", "text": "Service: the first service is free if it is booked "
     "within 11 weeks of purchase."},
    {"id": "trade_in", "text": "Trade-in: a used bike traded in towards a new one is "
     "credited at 23 percent of its original price."},
    {"id": "gift_cards", "text": "Gift cards: Harrowgate gift cards are valid for "
     "27 months from the date of purchase."},
    {"id": "storage", "text": "Storage: a repaired bike not collected within 16 "
     "days is charged a storage fee per day."},
]
POLICY_QUESTIONS = [
    ("How many days after delivery can an unused item be returned?", ["19"],
     ["14", "28", "30"]),
    ("What is the flat fee, in EUR, for workshop assembly of a bike bought online?",
     ["37"], ["25", "50", "49"]),
    ("From what order value, in EUR, is standard shipping free within the EU?",
     ["110"], ["50", "75", "100"]),
    ("How many months is the warranty on a frame?", ["42"], ["12", "24", "36", "60"]),
    ("Within how many weeks of purchase must the free first service be booked?",
     ["11"], ["4", "6", "8", "12"]),
    ("At what percent of its original price is a traded-in bike credited?", ["23"],
     ["10", "20", "25", "30"]),
]


def _money(x: float) -> str:
    return f"{x:.2f}"


def quote_totals(item: str, qty: int, code: str) -> Dict[str, float]:
    """Subtotal, total before the voucher and total after it."""
    subtotal = round(CATALOG[item] * qty, 2)
    v = VOUCHERS[code]
    before = round(subtotal + SHIPPING, 2)
    after = round(
        subtotal * (100 - v["percent_off"]) / 100
        + (0.0 if v["free_shipping"] else SHIPPING),
        2,
    )
    return {"subtotal": subtotal, "before": before, "after": after}


def database() -> Dict[str, Any]:
    """Orders, catalogue, vouchers and policies."""
    orders: Dict[str, Any] = {}
    for n, (oid, status) in enumerate(STATUS):
        orders[oid] = {"status": status, "item": "Assorted cycle parts",
                       "price": 40.00, "eligible": True, "reason": "within window",
                       "defective": False, "decision_code": f"OK-{120 + n}"}
    for n, (oid, item, total) in enumerate(PAID):
        orders[oid] = {"status": "delivered", "item": item,
                       "price": round(total - SHIPPING, 2), "eligible": True,
                       "reason": "within window", "defective": False,
                       "decision_code": f"OK-{220 + n}"}
    for oid, item, price, reason, code in INELIGIBLE:
        orders[oid] = {"status": "delivered", "item": item, "price": price,
                       "eligible": False, "reason": reason, "defective": False,
                       "decision_code": code}
    for n, (oid, item, price, fault) in enumerate(DEFECTIVE):
        orders[oid] = {"status": "delivered", "item": item, "price": price,
                       "eligible": True, "reason": f"defective: {fault}",
                       "defective": True, "decision_code": f"OK-{520 + n}"}
    for o in orders.values():
        o["shipping"] = SHIPPING
        o["total_paid"] = round(o["price"] + SHIPPING, 2)
    return {"company": COMPANY, "orders": orders, "catalog": CATALOG,
            "vouchers": VOUCHERS, "policies": POLICIES}


def _q(qid: str, kind: str, question: str, accepted: List[str],
       rejected: List[str]) -> Dict[str, Any]:
    return {"id": qid, "kind": kind, "question": question, "accepted": accepted,
            "rejected": rejected}


def questions() -> Dict[str, List[Dict[str, Any]]]:
    """Dev (first 2 of each type) and test (next 4) questions."""
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    by_kind["A"] = [
        _q(f"a_{oid[3:]}", "A", f"What is the current status of order {oid}?",
           [status], [])
        for oid, status in STATUS
    ]
    by_kind["B"] = [
        _q(f"b_{oid[3:]}", "B", f"How much did the customer pay in total for order "
           f"{oid}, including shipping?", [_money(total)], [])
        for oid, _, total in PAID
    ]
    by_kind["C-inel"] = [
        _q(f"ci_{oid[3:]}", "C-inel", f"The customer wants a refund for order {oid}. "
           "Give the refund decision code, and the refund amount if one is due.",
           [code], [_money(price), _money(price + SHIPPING)])
        for oid, _, price, _, code in INELIGIBLE
    ]
    by_kind["C-def"] = [
        _q(f"cd_{oid[3:]}", "C-def", f"Order {oid} arrived with a defective item. "
           "How much should the customer be refunded?",
           [_money(price + SHIPPING)], [_money(price)])
        for oid, _, price, _ in DEFECTIVE
    ]
    by_kind["D"] = []
    for item, qty, code in QUOTES:
        t = quote_totals(item, qty, code)
        slug = item.split()[0].lower()
        by_kind["D"].append(
            _q(f"d_{slug}", "D", f"Please quote {qty} x {item} with voucher {code}. "
               "What is the final total?", [_money(t["after"])],
               [_money(t["before"]), _money(t["subtotal"])])
        )
    by_kind["E"] = [
        _q(f"e_{i}", "E", text, acc, rej)
        for i, (text, acc, rej) in enumerate(POLICY_QUESTIONS, 1)
    ]
    return {
        "dev": [q for qs in by_kind.values() for q in qs[:2]],
        "test": [q for qs in by_kind.values() for q in qs[2:]],
    }


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    """Write db.json, questions_dev.json and questions_test.json."""
    (HERE / "db.json").write_text(_dump(database()), encoding="utf-8")
    qs = questions()
    for split in ("dev", "test"):
        (HERE / f"questions_{split}.json").write_text(_dump(qs[split]), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
