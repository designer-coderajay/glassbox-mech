#!/usr/bin/env python3
"""Write the E1b corpus and question files (PROTOCOL.md §2). Run once; the JSON files
are committed and the test file's sha256 is sealed in Addendum 1.

The company (Velmora Home Energy), its policies and every number are fictional and
written for this experiment. No document is reused from E1. Eight documents change
between v1 (old) and v2 (current). As in E1, the fact sits at the end of each
document.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (id, title, v1 text, v2 text or None if unchanged)
DOCS = [
    # --- changed in v2; dev questions --------------------------------------------
    (
        "exit_fee",
        "Leaving a fixed tariff early",
        "Velmora Home Energy fixed tariffs keep your unit price the same for the whole contract. You can switch to another supplier at any time, and we will send a final bill within two weeks. There is no fee in the last 49 days of a contract. Leaving a fixed tariff before then costs an exit fee of €60 per fuel.",
        "Velmora Home Energy fixed tariffs keep your unit price the same for the whole contract. You can switch to another supplier at any time, and we will send a final bill within two weeks. There is no fee in the last 49 days of a contract. Leaving a fixed tariff before then costs an exit fee of €45 per fuel.",
    ),
    (
        "smart_meter",
        "Smart meter installation",
        "A smart meter sends readings automatically, so you never need to read the meter yourself. Installation is free and takes about two hours, and the power is off for part of that time. An adult must be at home during the visit. After you book, an engineer visits within 10 working days.",
        "A smart meter sends readings automatically, so you never need to read the meter yourself. Installation is free and takes about two hours, and the power is off for part of that time. An adult must be at home during the visit. After you book, an engineer visits within 6 working days.",
    ),
    # --- changed in v2; test questions -------------------------------------------
    (
        "late_payment",
        "Late payment",
        "Bills are due 14 days after they are issued. If a payment is missed we send a reminder by email and by letter, and we are happy to agree a payment plan instead. Missed payments can affect your credit file. A late payment fee of €8 is added to the next bill.",
        "Bills are due 14 days after they are issued. If a payment is missed we send a reminder by email and by letter, and we are happy to agree a payment plan instead. Missed payments can affect your credit file. A late payment fee of €12 is added to the next bill.",
    ),
    (
        "referral",
        "Refer a friend",
        "Share your personal referral link with friends and family. The reward is paid once your friend has been supplied by Velmora for 60 days, and there is no limit on the number of referrals. Your friend gets the same reward. Each successful referral earns a bill credit of €40.",
        "Share your personal referral link with friends and family. The reward is paid once your friend has been supplied by Velmora for 60 days, and there is no limit on the number of referrals. Your friend gets the same reward. Each successful referral earns a bill credit of €50.",
    ),
    (
        "solar_export",
        "Selling solar power back",
        "If you have solar panels or a home battery, you can sell the electricity you do not use back to the grid. You need a smart meter that records exports, and payments are added to your account every quarter. Exported electricity is paid at 4 cents per kWh.",
        "If you have solar panels or a home battery, you can sell the electricity you do not use back to the grid. You need a smart meter that records exports, and payments are added to your account every quarter. Exported electricity is paid at 7 cents per kWh.",
    ),
    (
        "cooling_off",
        "Changing your mind",
        "When you join Velmora online or by phone, you can still change your mind. Tell us by email, chat or phone, and we will stop the switch before your supply moves to us. You will not pay any exit fee. The cooling-off period is 14 days from the day you sign up.",
        "When you join Velmora online or by phone, you can still change your mind. Tell us by email, chat or phone, and we will stop the switch before your supply moves to us. You will not pay any exit fee. The cooling-off period is 21 days from the day you sign up.",
    ),
    (
        "direct_debit",
        "Paying by direct debit",
        "A monthly direct debit spreads the cost of energy evenly across the year. We review the amount every six months and tell you before it changes. You can change the payment date in your online account. Customers who pay by monthly direct debit get a discount of 3% on their bills.",
        "A monthly direct debit spreads the cost of energy evenly across the year. We review the amount every six months and tell you before it changes. You can change the payment date in your online account. Customers who pay by monthly direct debit get a discount of 5% on their bills.",
    ),
    (
        "emergency",
        "Gas emergencies and boiler breakdowns",
        "If you smell gas, leave the property and call the national emergency line first. Boiler breakdowns are covered for customers with boiler cover, and repairs are free under the cover. Call us at any time of day. An engineer will arrive within 4 hours of a breakdown call.",
        "If you smell gas, leave the property and call the national emergency line first. Boiler breakdowns are covered for customers with boiler cover, and repairs are free under the cover. Call us at any time of day. An engineer will arrive within 2 hours of a breakdown call.",
    ),
    # --- unchanged; dev questions ------------------------------------------------
    (
        "outage",
        "Power cuts",
        "Power cuts are handled by the local network operator, who will tell you when power is expected back. You can check live updates on the network operator's website. Customers on the priority register get a call during long cuts. If a cut lasts more than 12 hours, you get compensation of €30.",
        None,
    ),
    (
        "warm_home",
        "Warm Home Rebate",
        "The Warm Home Rebate helps households with low incomes during the winter months. You can apply online or by phone between October and January, and we will check your eligibility within a month. The rebate is paid directly into your energy account. Eligible households receive a rebate of €150.",
        None,
    ),
    (
        "letters",
        "Writing to us",
        "You can write to Velmora at our head office address, which is printed on every bill. Please include your account number so we can find your details quickly. We scan all letters on the day they arrive. Letters are answered within 10 working days.",
        None,
    ),
    (
        "appointments",
        "Engineer appointments",
        "When you book an engineer visit, you choose a morning or afternoon on the day that suits you. We send a text message on the day with a narrower estimate of the arrival time. You can rebook online for free. Appointments are offered in 4 hour time slots.",
        None,
    ),
    # --- unchanged; test questions -----------------------------------------------
    (
        "meter_reading",
        "Submitting meter readings",
        "If you do not have a smart meter, please send us a reading every month so your bill is accurate. You can submit readings in the app, online or by phone. Without a reading, your bill is estimated from past use. Readings must reach us by the 25th of each month.",
        None,
    ),
    (
        "priority_register",
        "Priority services register",
        "The priority services register gives extra help to customers who need it, such as advance notice of planned power cuts and bills in large print. It is free to join, and you can register for someone you care for. Customers aged 65 or over can join the register automatically.",
        None,
    ),
    (
        "payment_plan",
        "Payment plans",
        "If you are struggling to pay, contact us as early as you can. We can agree a plan that spreads what you owe across affordable monthly amounts, on top of your usual payments. No interest is charged on a payment plan. Plans can last up to 18 months.",
        None,
    ),
    (
        "credit_refund",
        "Credit on your account",
        "If you pay more than you use, your account builds up a credit balance. We check balances every year when we review your direct debit. You can ask for a refund at any time in the app. Credit balances above €80 are refunded automatically.",
        None,
    ),
    (
        "moving_home",
        "Moving home",
        "Tell us when you are moving so we can close your account at the old address and set one up at the new one. Take a meter reading on the day you move, and keep a photo of it. Your final bill is sent to the new address. Please give us at least 48 hours notice before you move.",
        None,
    ),
    (
        "ev_tariff",
        "Electric vehicle tariff",
        "The Velmora EV tariff is designed for customers who charge an electric car at home. The cheap rate applies to all electricity used in the off-peak window, not only the car charger. A smart meter is required. Off-peak electricity on the EV tariff costs 9 cents per kWh.",
        None,
    ),
    (
        "complaints",
        "Making a complaint",
        "We want to put things right when they go wrong. You can complain by phone, chat, email or letter, and we will confirm that we have received it within two working days. If you are not happy with our final answer, you can go to the energy ombudsman after 8 weeks.",
        None,
    ),
    (
        "boiler_cover",
        "Boiler cover",
        "Boiler cover includes a yearly service, parts and labour for repairs, and an emergency helpline. There is no excess to pay when you make a claim. Boilers older than fifteen years need an inspection before cover starts. Boiler cover costs €14 per month.",
        None,
    ),
    (
        "final_bill",
        "Closing your account",
        "When your supply moves to another company or you leave a property, we close your account. We use the final meter reading to work out what you owe or what we owe you. Any credit is refunded to your bank account. The final bill is sent within 6 weeks.",
        None,
    ),
    (
        "standing_charge",
        "Standing charge",
        "The standing charge is a fixed daily amount that covers the cost of keeping your home connected to the network. You pay it even on days when you use no energy. It is shown separately on every bill. The electricity standing charge is 52 cents per day.",
        None,
    ),
    (
        "heat_pump",
        "Heat pump grant",
        "Velmora helps customers replace old gas boilers with heat pumps. A surveyor visits first to check that your home is suitable, and the grant is paid directly to the installer. Only approved installers can be used. The heat pump grant is worth up to €700.",
        None,
    ),
    (
        "paperless",
        "Paperless billing",
        "With paperless billing, your bills and statements arrive by email and in the app instead of by post. You can still ask for a paper copy of any bill. Paperless billing also cuts the paper we use. Customers who choose paperless billing save €2 per month.",
        None,
    ),
]

# (id, doc, question, accepted, stale, kind, split)
QUESTIONS = [
    # dev: updated
    (
        "u_exit_fee_a",
        "exit_fee",
        "How much is the exit fee for leaving a fixed tariff early?",
        ["€45", "45 euro", "45 eur"],
        ["€60", "60 euro", "60 eur"],
        "updated",
        "dev",
    ),
    (
        "u_exit_fee_b",
        "exit_fee",
        "What does it cost per fuel to end a fixed tariff early?",
        ["€45", "45 euro", "45 eur"],
        ["€60", "60 euro", "60 eur"],
        "updated",
        "dev",
    ),
    (
        "u_smart_meter_a",
        "smart_meter",
        "How soon after booking does an engineer install a smart meter?",
        ["6 working days", "6 business days", "six working days"],
        ["10 working days", "10 business days", "ten working days"],
        "updated",
        "dev",
    ),
    (
        "u_smart_meter_b",
        "smart_meter",
        "Within how many working days is a smart meter installed after I book?",
        ["6 working days", "6 business days", "six working days", "6 days"],
        ["10 working days", "10 business days", "ten working days", "10 days"],
        "updated",
        "dev",
    ),
    # dev: unchanged
    (
        "n_outage",
        "outage",
        "How much compensation do I get for a power cut longer than 12 hours?",
        ["€30", "30 euro", "30 eur"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_warm_home",
        "warm_home",
        "How much is the Warm Home Rebate?",
        ["€150", "150 euro", "150 eur"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_letters",
        "letters",
        "How long does Velmora take to answer a letter?",
        ["10 working days", "10 business days", "ten working days"],
        [],
        "unchanged",
        "dev",
    ),
    (
        "n_appointments",
        "appointments",
        "How long are the time slots for engineer appointments?",
        ["4 hour", "4 hours", "four hour", "four hours"],
        [],
        "unchanged",
        "dev",
    ),
    # test: updated
    (
        "u_late_payment_a",
        "late_payment",
        "How much is the late payment fee?",
        ["€12", "12 euro", "12 eur"],
        ["€8", "8 euro", "8 eur"],
        "updated",
        "test",
    ),
    (
        "u_late_payment_b",
        "late_payment",
        "What fee is added to my next bill if I pay late?",
        ["€12", "12 euro", "12 eur"],
        ["€8", "8 euro", "8 eur"],
        "updated",
        "test",
    ),
    (
        "u_referral_a",
        "referral",
        "How much bill credit does a successful referral earn?",
        ["€50", "50 euro", "50 eur"],
        ["€40", "40 euro", "40 eur"],
        "updated",
        "test",
    ),
    (
        "u_referral_b",
        "referral",
        "What reward do I get for referring a friend?",
        ["€50", "50 euro", "50 eur"],
        ["€40", "40 euro", "40 eur"],
        "updated",
        "test",
    ),
    (
        "u_solar_export_a",
        "solar_export",
        "How much is exported solar electricity paid per kWh?",
        ["7 cents", "seven cents", "€0.07"],
        ["4 cents", "four cents", "€0.04"],
        "updated",
        "test",
    ),
    (
        "u_solar_export_b",
        "solar_export",
        "What rate do I get for selling unused solar power back to the grid?",
        ["7 cents", "seven cents", "€0.07"],
        ["4 cents", "four cents", "€0.04"],
        "updated",
        "test",
    ),
    (
        "u_cooling_off_a",
        "cooling_off",
        "How long is the cooling-off period after signing up?",
        ["21 days", "twenty one days", "twenty-one days"],
        ["14 days", "fourteen days"],
        "updated",
        "test",
    ),
    (
        "u_cooling_off_b",
        "cooling_off",
        "How many days do I have to change my mind after joining?",
        ["21 days", "twenty one days", "twenty-one days"],
        ["14 days", "fourteen days"],
        "updated",
        "test",
    ),
    (
        "u_direct_debit_a",
        "direct_debit",
        "What discount do customers get for paying by monthly direct debit?",
        ["5%", "5 percent", "five percent"],
        ["3%", "3 percent", "three percent"],
        "updated",
        "test",
    ),
    (
        "u_direct_debit_b",
        "direct_debit",
        "How much cheaper are bills if I pay by direct debit?",
        ["5%", "5 percent", "five percent"],
        ["3%", "3 percent", "three percent"],
        "updated",
        "test",
    ),
    (
        "u_emergency_a",
        "emergency",
        "How quickly does an engineer arrive after a boiler breakdown call?",
        ["2 hours", "two hours"],
        ["4 hours", "four hours"],
        "updated",
        "test",
    ),
    (
        "u_emergency_b",
        "emergency",
        "Within how many hours will a breakdown engineer reach me?",
        ["2 hours", "two hours"],
        ["4 hours", "four hours"],
        "updated",
        "test",
    ),
    # test: unchanged
    (
        "n_meter_reading",
        "meter_reading",
        "By which day of the month must meter readings reach Velmora?",
        ["25th", "25"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_priority_register",
        "priority_register",
        "From what age can customers join the priority register automatically?",
        ["65"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_payment_plan",
        "payment_plan",
        "How long can a payment plan last?",
        ["18 months", "eighteen months"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_credit_refund",
        "credit_refund",
        "Above what amount are credit balances refunded automatically?",
        ["€80", "80 euro", "80 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_moving_home",
        "moving_home",
        "How much notice should I give before moving home?",
        ["48 hours", "forty eight hours", "forty-eight hours"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_ev_tariff",
        "ev_tariff",
        "What does off-peak electricity cost on the EV tariff?",
        ["9 cents", "nine cents", "€0.09"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_complaints",
        "complaints",
        "After how long can I take a complaint to the energy ombudsman?",
        ["8 weeks", "eight weeks"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_boiler_cover",
        "boiler_cover",
        "How much does boiler cover cost?",
        ["€14", "14 euro", "14 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_final_bill",
        "final_bill",
        "How soon is the final bill sent after closing an account?",
        ["6 weeks", "six weeks"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_standing_charge",
        "standing_charge",
        "How much is the electricity standing charge?",
        ["52 cents", "fifty two cents", "€0.52"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_heat_pump",
        "heat_pump",
        "How much is the heat pump grant worth?",
        ["€700", "700 euro", "700 eur"],
        [],
        "unchanged",
        "test",
    ),
    (
        "n_paperless",
        "paperless",
        "How much do customers save with paperless billing?",
        ["€2", "2 euro", "2 eur"],
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
