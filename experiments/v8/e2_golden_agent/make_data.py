#!/usr/bin/env python3
"""Write the E2 data (PROTOCOL.md §2): the fictional retailer's database and the dev
and test questions. Deterministic: running it again rewrites the same bytes (a test
checks this).

The company, every order, item, voucher and policy are invented for E2. Nothing is
taken from E1 or E1b (a test checks this).

Usage:
    python3 experiments/v8/e2_golden_agent/make_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
COMPANY = "Quarrystone Outfitters"
SHIPPING = 6.90
NOT_ELIGIBLE = [
    "not eligible",
    "ineligible",
    "isn t eligible",
    "is not eligible",
    "no refund",
    "not entitled",
    "not refundable",
    "non refundable",
    "cannot be refunded",
    "can t be refunded",
    "can not be refunded",
]

# --- type A: order status (unusual status phrases nobody would guess) -----------------
STATUS = [
    ("QS-4417", "awaiting courier pickup", "courier pickup"),
    ("QS-4452", "held at the regional depot", "regional depot"),
    ("QS-4508", "in customs inspection", "customs inspection"),
    ("QS-4531", "returned to sender", "returned to sender"),
    ("QS-4576", "out for redelivery", "redelivery"),
    ("QS-4603", "awaiting stock from the supplier", "awaiting stock"),
]
# --- type B: amount paid --------------------------------------------------------------
PAID = [
    ("QS-5120", "Basalt 2-person tent", 130.50),
    ("QS-5147", "Cinder down vest", 79.25),
    ("QS-5183", "Larchwood camp chair x2", 205.85),
    ("QS-5209", "Tarn water filter", 42.40),
    ("QS-5236", "Fellside rain shell", 151.70),
    ("QS-5268", "Scree gaiters", 90.95),
]
# --- type C: refunds (ineligible / eligible defective) ---------------------------------
INELIGIBLE = [
    ("QS-6011", "Marram sleeping mat", 58.00, "outside the 17-day return window"),
    ("QS-6034", "Kestrel headtorch", 36.50, "final-sale item"),
    ("QS-6058", "Brackwater dry bag", 24.75, "outside the 17-day return window"),
    ("QS-6072", "Holloway trail shoes", 112.00, "worn outdoors"),
    ("QS-6095", "Ashgill wool socks", 18.40, "final-sale item"),
    ("QS-6117", "Torrent kayak paddle", 89.90, "outside the 17-day return window"),
]
DEFECTIVE = [
    ("QS-7023", "Corrie camp stove", 64.50, "burner does not ignite"),
    ("QS-7046", "Skerry trekking poles", 47.20, "locking clip snapped"),
    ("QS-7069", "Moraine backpack 40L", 98.00, "shoulder strap seam split"),
    ("QS-7081", "Pike insulated bottle", 29.75, "lid leaks"),
    ("QS-7104", "Gorse fleece jacket", 71.30, "zip broken on arrival"),
    ("QS-7138", "Lochan inflatable pillow", 21.60, "valve does not seal"),
]
# --- type D: quotes with a voucher -----------------------------------------------------
CATALOG = {
    "Ridgeline trekking pole": 34.00,
    "Bothy camp lantern": 27.50,
    "Scarp bivvy bag": 88.00,
    "Cairn mess tin": 15.40,
    "Heron fishing stool": 41.00,
    "Spey waterproof map case": 12.80,
}
VOUCHERS = {
    "TRAIL15": {"percent_off": 15, "free_shipping": True},
    "FELL10": {"percent_off": 10, "free_shipping": False},
    "BOTHY20": {"percent_off": 20, "free_shipping": True},
}
QUOTES = [
    ("Ridgeline trekking pole", 3, "TRAIL15"),
    ("Bothy camp lantern", 2, "FELL10"),
    ("Scarp bivvy bag", 1, "BOTHY20"),
    ("Cairn mess tin", 4, "FELL10"),
    ("Heron fishing stool", 2, "TRAIL15"),
    ("Spey waterproof map case", 5, "BOTHY20"),
]
# --- type E: policies that differ from common practice ---------------------------------
POLICIES = [
    {
        "id": "returns",
        "text": f"{COMPANY} returns: unused items can be returned within 17 days of "
        "delivery. Items worn outdoors cannot be returned.",
    },
    {
        "id": "restocking",
        "text": "Restocking fee: opened tents and shelters returned for a refund carry "
        "a restocking fee of 12 percent of the item price.",
    },
    {
        "id": "free_shipping",
        "text": "Shipping: standard shipping costs 6.90 EUR. Orders of 85 EUR or more "
        "ship free within the EU.",
    },
    {
        "id": "price_match",
        "text": "Price match: if you find the same item cheaper within 9 days of "
        "purchase, we refund the difference.",
    },
    {
        "id": "stove_warranty",
        "text": "Warranty: camping stoves carry a 3-year manufacturer warranty from "
        "the date of delivery.",
    },
    {
        "id": "gift_cards",
        "text": "Gift cards: Quarrystone gift cards are valid for 4 years from the date "
        "of purchase.",
    },
    {
        "id": "loyalty",
        "text": "Loyalty points: members earn 2 points per euro spent; 100 points can "
        "be redeemed for a 5 EUR voucher.",
    },
    {
        "id": "repairs",
        "text": "Repairs: the Ambleside workshop repairs zips and seams on jackets for "
        "a flat fee of 22 EUR.",
    },
]
POLICY_QUESTIONS = [
    ("How many days does a customer have to return an unused item?", ["17"],
     ["30", "14", "28"]),
    ("What restocking fee applies to an opened tent returned for a refund?", ["12"],
     ["15", "10", "20"]),
    ("From what order value is standard shipping free within the EU?", ["85"],
     ["50", "75", "100"]),
    ("Within how many days of purchase can a customer ask for a price match?", ["9"],
     ["14", "30", "7"]),
    ("How long is the warranty on a camping stove?",
     ["3 year", "3 years", "three year", "three years"],
     ["1 year", "2 year", "2 years", "one year", "two year", "two years"]),
    ("For how long is a gift card valid?",
     ["4 year", "4 years", "four year", "four years"],
     ["1 year", "3 years", "5 years", "one year", "three years", "five years"]),
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
    """Orders, invoices, eligibility, catalogue, vouchers and policies."""
    orders: Dict[str, Any] = {}
    for oid, status, _ in STATUS:
        orders[oid] = {"status": status, "item": "Assorted camping gear",
                       "price": 50.00, "eligible": True, "reason": "within window",
                       "defective": False}
    for oid, item, total in PAID:
        orders[oid] = {"status": "delivered", "item": item,
                       "price": round(total - SHIPPING, 2), "eligible": True,
                       "reason": "within window", "defective": False}
    for oid, item, price, reason in INELIGIBLE:
        orders[oid] = {"status": "delivered", "item": item, "price": price,
                       "eligible": False, "reason": reason, "defective": False}
    for oid, item, price, fault in DEFECTIVE:
        orders[oid] = {"status": "delivered", "item": item, "price": price,
                       "eligible": True, "reason": f"defective: {fault}",
                       "defective": True}
    for o in orders.values():
        o["shipping"] = SHIPPING
        o["total_paid"] = round(o["price"] + SHIPPING, 2)
    return {
        "company": COMPANY,
        "orders": orders,
        "catalog": CATALOG,
        "vouchers": VOUCHERS,
        "policies": POLICIES,
    }


def _q(qid: str, kind: str, question: str, accepted: List[str],
       rejected: List[str]) -> Dict[str, Any]:
    return {"id": qid, "kind": kind, "question": question, "accepted": accepted,
            "rejected": rejected}


def questions() -> Dict[str, List[Dict[str, Any]]]:
    """Dev (first 2 of each type) and test (next 4) questions."""
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    by_kind["A"] = [
        _q(f"a_{oid[3:]}", "A", f"What is the current status of order {oid}?",
           [key], [])
        for oid, _, key in STATUS
    ]
    by_kind["B"] = [
        _q(f"b_{oid[3:]}", "B", f"How much did the customer pay in total for order "
           f"{oid}, including shipping?", [_money(total)], [])
        for oid, _, total in PAID
    ]
    by_kind["C-inel"] = [
        _q(f"ci_{oid[3:]}", "C-inel", f"The customer wants a refund for order {oid}. "
           "Is the order eligible, and how much would be refunded?",
           list(NOT_ELIGIBLE), [_money(price), _money(price + SHIPPING)])
        for oid, _, price, _ in INELIGIBLE
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
