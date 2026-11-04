#!/usr/bin/env python3
"""Write the E1 corpus and question files (PROTOCOL.md §2). Run once; the JSON files
are committed and the test file's sha256 is sealed in Addendum 1.

The company, its policies and every number are fictional and written for this
experiment. Eight documents change between v1 (old) and v2 (current).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each document: (id, title, v1 text, v2 text or None if unchanged). Facts sit near the
# end of each document on purpose, so chunk size can matter.
DOCS = [
    (
        "returns",
        "Returns policy",
        "Kestrel Supply Co. wants every customer to be happy with what they buy. Items must be unused, with tags attached and the original receipt or order number. Sale items can be returned on the same terms as full-price items. Refunds go back to the original payment method once the warehouse has checked the item, which usually takes three working days. You can return an item within 14 days of delivery.",
        "Kestrel Supply Co. wants every customer to be happy with what they buy. Items must be unused, with tags attached and the original receipt or order number. Sale items can be returned on the same terms as full-price items. Refunds go back to the original payment method once the warehouse has checked the item, which usually takes three working days. You can return an item within 30 days of delivery.",
    ),
    (
        "shipping",
        "Standard shipping",
        "Orders placed before noon on a working day leave our warehouse the same day. Standard delivery within the country takes two to four working days, and every parcel gets a tracking link by email. Parcels are handed to the national postal service unless a courier is chosen at checkout. Standard shipping costs €4.95, and it is free on orders over €50.",
        "Orders placed before noon on a working day leave our warehouse the same day. Standard delivery within the country takes two to four working days, and every parcel gets a tracking link by email. Parcels are handed to the national postal service unless a courier is chosen at checkout. Standard shipping costs €4.95, and it is free on orders over €35.",
    ),
    (
        "warranty",
        "Product warranty",
        "Kestrel tents, stoves and backpacks are tested in the field before they are sold. The warranty covers manufacturing faults in materials and workmanship, but not normal wear, misuse or damage from accidents. To make a claim, send a photo of the fault and your order number to the support team. All Kestrel-brand products carry a warranty of 1 year from the date of purchase.",
        "Kestrel tents, stoves and backpacks are tested in the field before they are sold. The warranty covers manufacturing faults in materials and workmanship, but not normal wear, misuse or damage from accidents. To make a claim, send a photo of the fault and your order number to the support team. All Kestrel-brand products carry a warranty of 2 years from the date of purchase.",
    ),
    (
        "loyalty",
        "Loyalty programme",
        "The Kestrel Trail Club is free to join online or in any store. Members get early access to seasonal sales and a birthday voucher every year. Points can be redeemed at checkout once you have collected at least 500, and they expire after two years without activity. Members earn 1 point per euro spent.",
        "The Kestrel Trail Club is free to join online or in any store. Members get early access to seasonal sales and a birthday voucher every year. Points can be redeemed at checkout once you have collected at least 500, and they expire after two years without activity. Members earn 2 points per euro spent.",
    ),
    (
        "support_hours",
        "Customer support hours",
        "Our support team answers questions about orders, products and returns by phone, chat and email. Emails are answered within one working day, and chat is usually answered within a few minutes. The phone line is closed on public holidays. On weekdays, phone and chat support is open from 08:00 until 18:00.",
        "Our support team answers questions about orders, products and returns by phone, chat and email. Emails are answered within one working day, and chat is usually answered within a few minutes. The phone line is closed on public holidays. On weekdays, phone and chat support is open from 08:00 until 20:00.",
    ),
    (
        "membership_fee",
        "Trail Club Plus membership",
        "Trail Club Plus is the paid tier of the loyalty programme. It adds free express shipping, free repairs labour and invitations to members-only events. Plus can be cancelled at any time and the remaining months are refunded. The Trail Club Plus fee is €20 per year.",
        "Trail Club Plus is the paid tier of the loyalty programme. It adds free express shipping, free repairs labour and invitations to members-only events. Plus can be cancelled at any time and the remaining months are refunded. The Trail Club Plus fee is €25 per year.",
    ),
    (
        "cancellation",
        "Cancelling an order",
        "You can cancel an order from your account page or by contacting support. Once an order has been handed to the carrier it can no longer be cancelled, but it can still be returned. Cancelled orders are refunded within five working days. Orders can be cancelled up to 2 hours after they are placed.",
        "You can cancel an order from your account page or by contacting support. Once an order has been handed to the carrier it can no longer be cancelled, but it can still be returned. Cancelled orders are refunded within five working days. Orders can be cancelled up to 24 hours after they are placed.",
    ),
    (
        "student_discount",
        "Student discount",
        "Students at any accredited university or college can register with their student email address. The discount works online and in stores, but it cannot be combined with other promotions or used on gift cards. Registration has to be renewed every academic year. Verified students get a discount of 10% on full-price items.",
        "Students at any accredited university or college can register with their student email address. The discount works online and in stores, but it cannot be combined with other promotions or used on gift cards. Registration has to be renewed every academic year. Verified students get a discount of 15% on full-price items.",
    ),
    (
        "gift_cards",
        "Gift cards",
        "Kestrel gift cards are sold in stores and online in any amount from €10 to €500. They can be used for any product, including sale items, but not to buy other gift cards. A lost card can be replaced if you still have the purchase receipt. Gift cards are valid for 3 years from the date of purchase.",
        None,
    ),
    (
        "price_match",
        "Price match",
        "If you find the same product cheaper at another national retailer, we will match the price. The product must be identical, in stock at the other retailer and not part of a clearance or members-only offer. Show the offer to staff or send a link to support. Price matching is available within 7 days of your purchase.",
        None,
    ),
    (
        "store_hours",
        "Store opening hours",
        "Our stores are open from Monday to Saturday and closed on Sundays and public holidays. On weekdays the doors open at 09:30 and close at 19:00. Some city-centre stores have longer hours during the December holidays, which are listed on each store page. On Saturdays, stores close at 16:00.",
        None,
    ),
    (
        "repairs",
        "Repair service",
        "Our workshop repairs zips, seams, tent poles and stove valves, whether or not the item is still under warranty. Bring the item to any store or send it in with the free repair label from your account. Repairs outside the warranty are charged for parts only for Trail Club Plus members. The usual repair turnaround is 10 working days.",
        None,
    ),
    (
        "rentals",
        "Equipment rental",
        "Tents, sleeping bags and climbing gear can be rented for weekends or full weeks from selected stores. Rented items are cleaned and checked after every return. Late returns are charged at the daily rate. A refundable deposit of €100 is required for every rental.",
        None,
    ),
    (
        "data_retention",
        "Order data retention",
        "We keep the information needed to process orders, returns and warranty claims. Payment card details are never stored by Kestrel; they are handled by our payment provider. Marketing preferences can be changed at any time in your account. Order records are kept for 6 years to meet accounting rules.",
        None,
    ),
    (
        "account_deletion",
        "Deleting your account",
        "You can ask for your customer account to be deleted from the account page or by writing to support. Deleting the account also removes your Trail Club points and saved addresses. Order records required by law are kept separately. Deletion requests are processed within 5 working days.",
        None,
    ),
    (
        "international",
        "International shipping",
        "We ship to every country in the European Union and to Norway, Switzerland and the United Kingdom. Delivery takes four to eight working days, and customs fees for non-EU countries are paid by the customer. International orders cannot be sent to parcel lockers. International shipping costs a flat fee of €15.",
        None,
    ),
    (
        "bulk_orders",
        "Bulk and club orders",
        "Outdoor clubs, schools and companies can order equipment in bulk through our business team. Bulk orders get a dedicated contact person and an invoice with 30-day payment terms. Custom logos can be added to many products for an extra charge. Orders of 50 units or more receive a discount of 8%.",
        None,
    ),
    (
        "delivery_damage",
        "Damaged deliveries",
        "Please check your parcel when it arrives. If the box or the product is damaged, take photos before unpacking anything else. We will send a replacement or a refund, whichever you prefer, and you do not need to pay for the return. Damage must be reported within 48 hours of delivery.",
        None,
    ),
    (
        "exchanges",
        "Size exchanges",
        "Shoes and clothing can be exchanged for a different size free of charge, online or in any store. The item must be unworn and in its original packaging. If the new size is out of stock, we refund the item instead. Size exchanges are possible within 60 days of delivery.",
        None,
    ),
    (
        "layaway",
        "Layaway",
        "Layaway lets you reserve large items such as tents and bikes and pay for them in instalments. The item is kept in store until it is fully paid. Instalments can be paid in store or online, and the plan can run for up to three months. Layaway requires a deposit of 20% of the price.",
        None,
    ),
    (
        "newsletter",
        "Newsletter",
        "The Kestrel newsletter brings trip ideas, gear guides and early news about sales, about twice a month. You can unsubscribe with one click at the bottom of every email. We never sell your email address. New subscribers receive a voucher worth €5 off their next order.",
        None,
    ),
    (
        "recycling",
        "Gear recycling",
        "Bring worn-out outdoor gear of any brand to one of our stores and we will recycle it or pass it on to repair partners. Items must be clean and dry. Electrical items and gas canisters cannot be accepted. You receive store credit of €3 for every item you hand in.",
        None,
    ),
    (
        "click_collect",
        "Click and collect",
        "Orders can be collected free of charge from any Kestrel store. You will get an email when the order is ready, usually within one working day. Bring the email and an ID to the collection desk. Orders are held for collection for 9 days before they are returned to the warehouse.",
        None,
    ),
    (
        "gift_wrap",
        "Gift wrapping",
        "Most items can be gift-wrapped in recycled paper, online at checkout or at any store counter. We can add a handwritten card with your message. Large items such as tents are wrapped with a ribbon only. Gift wrapping costs €4 per item.",
        None,
    ),
]

# Questions: id, doc, question, accepted answers (v2 / current), stale answers (v1),
# kind ("updated" or "unchanged"), split ("dev" or "test").
QUESTIONS = [
    # updated facts: dev uses returns + shipping, test uses the other six
    (
        "u_returns_a",
        "returns",
        "How many days do I have to return an item?",
        ["30 days", "thirty days"],
        ["14 days", "fourteen days"],
        "updated",
        "dev",
    ),
    (
        "u_returns_b",
        "returns",
        "What is the deadline for sending back a product after it is delivered?",
        ["30 days", "thirty days"],
        ["14 days", "fourteen days"],
        "updated",
        "dev",
    ),
    (
        "u_shipping_a",
        "shipping",
        "Above what order value is standard shipping free?",
        ["€35", "35 euro", "35 eur"],
        ["€50", "50 euro", "50 eur"],
        "updated",
        "dev",
    ),
    (
        "u_shipping_b",
        "shipping",
        "How much do I need to spend to get free standard delivery?",
        ["€35", "35 euro", "35 eur"],
        ["€50", "50 euro", "50 eur"],
        "updated",
        "dev",
    ),
    (
        "u_warranty_a",
        "warranty",
        "How long is the warranty on Kestrel-brand products?",
        ["2 years", "two years", "24 months"],
        ["1 year", "one year", "12 months"],
        "updated",
        "test",
    ),
    (
        "u_warranty_b",
        "warranty",
        "For how long after purchase are manufacturing faults covered?",
        ["2 years", "two years", "24 months"],
        ["1 year", "one year", "12 months"],
        "updated",
        "test",
    ),
    (
        "u_loyalty_a",
        "loyalty",
        "How many points do Trail Club members earn per euro?",
        ["2 points", "two points"],
        ["1 point", "one point"],
        "updated",
        "test",
    ),
    (
        "u_loyalty_b",
        "loyalty",
        "What is the points rate for each euro a member spends?",
        ["2 points", "two points"],
        ["1 point", "one point"],
        "updated",
        "test",
    ),
    (
        "u_support_a",
        "support_hours",
        "Until what time is phone support open on weekdays?",
        ["20:00", "8 pm", "8pm", "8 p m"],
        ["18:00", "6 pm", "6pm", "6 p m"],
        "updated",
        "test",
    ),
    (
        "u_support_b",
        "support_hours",
        "When does the support chat close on a weekday?",
        ["20:00", "8 pm", "8pm", "8 p m"],
        ["18:00", "6 pm", "6pm", "6 p m"],
        "updated",
        "test",
    ),
    (
        "u_fee_a",
        "membership_fee",
        "How much does Trail Club Plus cost per year?",
        ["€25", "25 euro", "25 eur"],
        ["€20", "20 euro", "20 eur"],
        "updated",
        "test",
    ),
    (
        "u_fee_b",
        "membership_fee",
        "What is the annual fee for the paid loyalty tier?",
        ["€25", "25 euro", "25 eur"],
        ["€20", "20 euro", "20 eur"],
        "updated",
        "test",
    ),
    (
        "u_cancel_a",
        "cancellation",
        "How long after placing an order can I still cancel it?",
        ["24 hours", "24 hour", "twenty four hours"],
        ["2 hours", "two hours"],
        "updated",
        "test",
    ),
    (
        "u_cancel_b",
        "cancellation",
        "What is the time limit for cancelling an order?",
        ["24 hours", "24 hour", "twenty four hours"],
        ["2 hours", "two hours"],
        "updated",
        "test",
    ),
    (
        "u_student_a",
        "student_discount",
        "What discount do verified students get?",
        ["15%", "15 percent", "fifteen percent"],
        ["10%", "10 percent", "ten percent"],
        "updated",
        "test",
    ),
    (
        "u_student_b",
        "student_discount",
        "How big is the student discount on full-price items?",
        ["15%", "15 percent", "fifteen percent"],
        ["10%", "10 percent", "ten percent"],
        "updated",
        "test",
    ),
    # unchanged facts: dev uses the first four, test the other twelve
    (
        "n_gift_cards",
        "gift_cards",
        "How long is a gift card valid?",
        ["3 years", "three years"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_price_match",
        "price_match",
        "Within how many days of purchase can I ask for a price match?",
        ["7 days", "seven days"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_store_hours",
        "store_hours",
        "What time do stores close on Saturdays?",
        ["16:00", "4 pm", "4pm", "4 p m"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_repairs",
        "repairs",
        "How long does a repair usually take?",
        ["10 working days", "10 business days", "ten working days"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_rentals",
        "rentals",
        "How much is the deposit for renting equipment?",
        ["€100", "100 euro", "100 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_data",
        "data_retention",
        "How long are order records kept?",
        ["6 years", "six years"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_deletion",
        "account_deletion",
        "How quickly is an account deletion request processed?",
        ["5 working days", "5 business days", "five working days"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_international",
        "international",
        "What does international shipping cost?",
        ["€15", "15 euro", "15 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_bulk",
        "bulk_orders",
        "What discount do orders of 50 units or more get?",
        ["8%", "8 percent", "eight percent"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_damage",
        "delivery_damage",
        "How soon must I report a damaged delivery?",
        ["48 hours", "48 hour", "forty eight hours"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_exchanges",
        "exchanges",
        "Within how many days can I exchange shoes for another size?",
        ["60 days", "sixty days"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_layaway",
        "layaway",
        "How much deposit does layaway require?",
        ["20%", "20 percent", "twenty percent"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_newsletter",
        "newsletter",
        "What voucher do new newsletter subscribers receive?",
        ["€5", "5 euro", "5 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_recycling",
        "recycling",
        "How much store credit do I get for each recycled item?",
        ["€3", "3 euro", "3 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_click_collect",
        "click_collect",
        "How long is a click-and-collect order held at the store?",
        ["9 days", "nine days"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_gift_wrap",
        "gift_wrap",
        "How much does gift wrapping cost?",
        ["€4", "4 euro", "4 eur"],
        [],
        "unchanged",
        "test",
    ),
]


def main() -> None:
    """Write corpus.json, questions_dev.json and questions_test.json next to this file."""
    corpus = [
        {"id": d, "title": t, "v1": v1, "v2": v2 if v2 is not None else v1}
        for d, t, v1, v2 in DOCS
    ]
    (HERE / "corpus.json").write_text(json.dumps(corpus, indent=1) + "\n", "utf-8")
    for split in ("dev", "test"):
        rows = [
            {
                "id": q,
                "doc": d,
                "question": text,
                "accepted": acc,
                "stale": st,
                "kind": kind,
            }
            for q, d, text, acc, st, kind, s in QUESTIONS
            if s == split
        ]
        path = HERE / f"questions_{split}.json"
        path.write_text(json.dumps(rows, indent=1) + "\n", "utf-8")


if __name__ == "__main__":
    main()
