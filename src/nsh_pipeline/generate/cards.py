"""Synthetic WEX cards: `debit_card_status` records, several per member (data contract A1.4)."""
from datetime import date

from nsh_pipeline.generate.activity import enrollments
from nsh_pipeline.generate.common import mdy, oid

PRIOR_YEAR_CARD_RATE = 0.4  # members who also have last year's (inactive) card record


def _card(n, person, employer_code, plan_name, plan_year, max_oop, remaining, status) -> dict:
    return {
        "_id": oid("e0", n),
        "employeeId": person["_id"],
        "nswid": person["_base64Id"],
        "employer_code": employer_code,
        "wex_id": employer_code,
        "plan_year": str(plan_year),
        "plan_name": plan_name,
        "card_status": status,
        "account_status": status,
        "pariticpant_status": "Active",  # (sic) misspelled in the real data
        "valid_start_dt": mdy(date(plan_year, 1, 1)),
        "valid_end_dt": mdy(date(plan_year, 12, 31)),
        "enroll_dt": mdy(date(plan_year, 1, 1)),
        "STATUS": "MAILED",
        "max_oop": max_oop,
        "remaining_balance": remaining,
    }


def make_cards(rng, clients: list[dict], people: list[dict], swipes: list[dict], reference: date) -> list[dict]:
    """Current-year card per enrolled member (+ a prior-year card for some); balance reflects spend."""
    spent = {}
    for swipe in swipes:
        spent[swipe["employeeId"]] = spent.get(swipe["employeeId"], 0.0) + float(swipe["paid_amt"].replace(",", ""))
    cards = []
    for person, client, benefit, _start, _end in enrollments(clients, people, reference):
        plan = next(p for p in client["benefits"] if p["name"] == benefit["CoverageName"])
        max_oop = float(plan["individualOopString"])
        code = client["employer"]["claimReferenceId"]
        if rng.random() < PRIOR_YEAR_CARD_RATE:
            cards.append(_card(len(cards) + 1, person, code, plan["name"], reference.year - 1,
                               max_oop, 0.0, "Inactive"))
        remaining = round(max(max_oop - spent.get(person["_id"], 0.0), 0.0), 2)
        status = "Inactive" if benefit["TerminationDate"] else "Active"
        cards.append(_card(len(cards) + 1, person, code, plan["name"], reference.year,
                           max_oop, remaining, status))
    return cards