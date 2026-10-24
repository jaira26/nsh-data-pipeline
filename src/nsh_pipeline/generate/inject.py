"""Deliberately injected data problems, so validation (M3) and bias analysis (M5) can prove they catch them.

Off by default (params.yaml: inject_issues.enabled). Uses its own RNG (seed + 1), so turning it on
changes only the injected records. Every injection is listed in manifest.json under `injected_issues`.
"""
import copy
from datetime import timedelta

from nsh_pipeline.constants import EXCLUDED_ORG_IDS
from nsh_pipeline.generate.common import iso, money_str

UNIT_ERROR_MIN_AMOUNT = 50.0  # only inflate swipes >= $50, so x100 is a clear outlier


def _num(money: str) -> float:
    return float(money.lstrip("$").replace(",", ""))


def _issue(name: str, collection: str, ids: list[str], description: str) -> dict:
    return {"issue": name, "collection": collection, "description": description, "count": len(ids), "ids": ids}


def _pick(rng, items: list, n: int) -> list:
    """Up to `n` distinct items chosen at random, kept in their original order."""
    if not items or n <= 0:
        return []
    chosen = rng.choice(len(items), size=min(n, len(items)), replace=False)
    return [items[i] for i in sorted(int(i) for i in chosen)]


def skew_denials(rng, swipes: list[dict], clients: list[dict], state: str, rate: float) -> dict:
    """Bias for M5 to detect: flip `rate` of one state's approved swipes to Denied.

    Runs before cards, bills and the ledger are built, so those stay consistent with the flipped swipes.
    """
    in_state = {
        c["_id"] for c in clients
        if c["employer"]["state"] == state and c["_id"] not in EXCLUDED_ORG_IDS
    }
    ids = []
    for swipe in swipes:
        eligible = (swipe["clientId"] in in_state and swipe["claim_status"] == "Approved"
                    and not swipe["ticketNumber"])
        if eligible and rng.random() < rate:
            swipe["claim_status"] = "Denied"
            swipe["denied_amt"] = swipe["amount"]
            swipe["paid_amt"] = money_str(0.0)
            ids.append(swipe["_id"])
    return _issue("denial_skew", "wex_claims", ids,
                  f"{rate:.0%} of approved swipes in {state} flipped to Denied (bias to find by slicing on state)")


def inject_anomalies(rng, collections: dict, settings: dict, reference) -> list[dict]:
    """Anomalies for M3 to detect (plan section 8, stage 6). Mutates `collections`; returns issue records."""
    real_swipes = [s for s in collections["wex_claims"] if s["clientId"] not in EXCLUDED_ORG_IDS]
    used = set()

    def pick_swipes(n, condition=lambda s: True):
        pool = [s for s in real_swipes if s["_id"] not in used and condition(s)]
        chosen = _pick(rng, pool, n)
        used.update(s["_id"] for s in chosen)
        return chosen

    issues = []

    ids = []
    bills = [b for b in collections["bills"] if b["appId"] not in EXCLUDED_ORG_IDS]
    for bill in _pick(rng, bills, settings.get("duplicate_bill_lines", 0)):
        medical = [line for line in bill["recurring"] if line["type"] == "medical"]
        line = medical[int(rng.integers(len(medical)))]
        bill["recurring"].append(copy.deepcopy(line))
        ids.append(f'{bill["_id"]}:{line["employeeId"]}')
    issues.append(_issue("duplicate_bill_lines", "bills", ids,
                         "Same member billed twice in one month (PMPM double-counting, contract A8 #1)"))

    ids = []
    for swipe in pick_swipes(settings.get("cents_as_dollars", 0),
                             lambda s: s["claim_status"] == "Approved"
                             and _num(s["amount"]) >= UNIT_ERROR_MIN_AMOUNT):
        swipe["amount"] = swipe["paid_amt"] = money_str(_num(swipe["amount"]) * 100)
        ids.append(swipe["_id"])
    issues.append(_issue("cents_as_dollars", "wex_claims", ids,
                         "Amount stored in cents but read as dollars: 100x too large (contract A8 #6)"))

    ids = []
    for i, swipe in enumerate(pick_swipes(settings.get("orphan_swipes", 0)), start=1):
        swipe["employeeId"] = f"ff{i:022d}"
        ids.append(swipe["_id"])
    issues.append(_issue("orphan_swipes", "wex_claims", ids,
                         "employeeId points to a member that does not exist"))

    ids = []
    for swipe in pick_swipes(settings.get("unparseable_money", 0)):
        swipe["paid_amt"] = "N/A"
        ids.append(swipe["_id"])
    issues.append(_issue("unparseable_money", "wex_claims", ids,
                         'paid_amt is "N/A" and becomes missing after preprocessing'))

    ids = []
    for swipe in pick_swipes(settings.get("future_service_dates", 0)):
        swipe["date_of_service"] = iso(reference + timedelta(days=30))
        ids.append(swipe["_id"])
    issues.append(_issue("future_service_dates", "wex_claims", ids,
                         "Service date is after the reference date"))

    ids = []
    real_tickets = [t for t in collections["tickets"] if t["clientId"] not in EXCLUDED_ORG_IDS]
    for ticket in _pick(rng, real_tickets, settings.get("status_casing_drift", 0)):
        ticket["status"] = ticket["status"].lower()
        ids.append(ticket["ticketNumber"])
    issues.append(_issue("status_casing_drift", "tickets", ids,
                         'Ticket status in lowercase (e.g. "closed"), outside the contract enum'))

    return issues