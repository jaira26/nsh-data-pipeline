"""Synthetic member activity: card swipes (`wex-claims`), support tickets and manual claims."""
import math
from datetime import date, datetime, timedelta

import numpy as np

from nsh_pipeline.generate.common import iso, iso_dt, money_str, oid, random_date
from nsh_pipeline.preprocess.people_current import current_entry

SWIPES_PER_MEMBER = 4.0          # Poisson mean per enrolled member
SWIPE_STATUSES = ["Approved", "Denied", "Pended", "Partial"]
NON_DENIED_STATUSES = ["Approved", "Pended", "Partial"]
NON_DENIED_WEIGHTS = np.array([0.88, 0.04, 0.03]) / 0.95
SERVICE_CATEGORIES = ["Medical", "Drugs & Medicine", "Dental", "Vision"]
SERVICE_CATEGORY_WEIGHTS = [0.55, 0.25, 0.12, 0.08]
DEBIT_CARD_SHARE = 0.92
SUBSTANTIATION_RATE = 0.025      # paid card swipes needing a receipt (NSH cohort ~2.2%)
SUPPORT_TICKET_RATE = 0.05
TICKET_STATUSES = ["Closed", "Waiting on Member", "In Progress", "New", "On Hold"]
TICKET_STATUS_WEIGHTS = [0.55, 0.20, 0.10, 0.10, 0.05]
SUPPORT_TYPES = ["Customer Support", "Technical Support", "General Questions", "Member Advocacy"]
MANUAL_CLAIM_RATE = 0.06
CLAIM_STATUSES = ["Approved", "Denied", "New", "In Progress", "Request More Info"]
CLAIM_STATUS_WEIGHTS = [0.70, 0.10, 0.08, 0.07, 0.05]
HIGH_DOLLAR = 1000.0

# Defaults for params.yaml -> denial_model (see the comments there).
DEFAULT_DENIAL_MODEL = {
    "intercept": -3.3,
    "log_amount": 0.6,
    "category": {"Dental": 0.9, "Vision": 0.7, "Drugs & Medicine": -0.2},
    "high_deductible": 0.5,
    "new_member": 0.8,
    "prior_denial": 0.7,
    "manual_claim": 0.3,
}
HIGH_DEDUCTIBLE = 3000.0
NEW_MEMBER_DAYS = 180
PRIOR_DENIAL_WINDOW_DAYS = 90
MAX_PRIOR_DENIALS = 3


def _to_float(money: str) -> float:
    return float(money.replace(",", ""))


def denial_probability(model: dict, amount: float, category: str, deductible: float,
                       tenure_days: int, recent_denials: int, is_card: bool) -> float:
    """Chance that a swipe is denied, from documented causes (no demographic effects)."""
    logit = (
        model["intercept"]
        + model["log_amount"] * math.log(max(amount, 1.0) / 75)
        + model["category"].get(category, 0.0)
        + model["high_deductible"] * (deductible >= HIGH_DEDUCTIBLE)
        + model["new_member"] * (tenure_days < NEW_MEMBER_DAYS)
        + model["prior_denial"] * min(recent_denials, MAX_PRIOR_DENIALS)
        + model["manual_claim"] * (not is_card)
    )
    return 1 / (1 + math.exp(-logit))


def _plan_deductible(client: dict, benefit: dict) -> float:
    plan = next(p for p in client["benefits"] if p["name"] == benefit["CoverageName"])
    return float(plan["individualDeductibleString"])


def _hire_date(person: dict) -> date:
    return date.fromisoformat(current_entry(person["Employment"])["HireDate"][:10])


def enrollments(clients: list[dict], people: list[dict], reference: date):
    """Yield (person, client, benefit, start, end) for members with a current medical enrollment.

    start..end is the window this plan year in which the member could have activity.
    """
    clients_by_id = {c["_id"]: c for c in clients}
    for person in people:
        benefit = current_entry(person.get("Benefits"))
        if benefit is None:
            continue
        start = max(date.fromisoformat(benefit["CurrentEffectiveDate"][:10]), date(reference.year, 1, 1))
        end = reference
        if benefit["TerminationDate"]:
            end = min(end, date.fromisoformat(benefit["TerminationDate"][:10]))
        if start <= end:
            yield person, clients_by_id[person["Application"]], benefit, start, end


def make_swipes(rng, fake, clients: list[dict], people: list[dict], reference: date,
                denial_model: dict | None = None) -> list[dict]:
    """Debit-card swipes; denial depends on amount, category, plan, tenure, history and channel.

    Swipes needing substantiation get a ticket number (NSET-xxxxx).
    """
    model = {**DEFAULT_DENIAL_MODEL, **(denial_model or {})}
    kinds = rng.choice(["Family Medicine", "Pharmacy", "Dental", "Eye Care", "Urgent Care"], size=40)
    providers = [f"{fake.last_name()} {kind}" for kind in kinds]
    swipes, n_tickets = [], 0
    for person, client, benefit, start, end in enrollments(clients, people, reference):
        deductible = _plan_deductible(client, benefit)
        hire = _hire_date(person)
        service_dates = sorted(random_date(rng, start, end) for _ in range(int(rng.poisson(SWIPES_PER_MEMBER))))
        denied_dates = []
        for service_date in service_dates:
            n = len(swipes) + 1
            amount = round(float(np.clip(rng.lognormal(4.3, 1.0), 5, 5000)), 2)
            category = str(rng.choice(SERVICE_CATEGORIES, p=SERVICE_CATEGORY_WEIGHTS))
            is_card = bool(rng.random() < DEBIT_CARD_SHARE)
            recent = sum(1 for d in denied_dates if 0 < (service_date - d).days <= PRIOR_DENIAL_WINDOW_DAYS)
            p_deny = denial_probability(model, amount, category, deductible,
                                        (service_date - hire).days, recent, is_card)
            if rng.random() < p_deny:
                status = "Denied"
                denied_dates.append(service_date)
            else:
                status = str(rng.choice(NON_DENIED_STATUSES, p=NON_DENIED_WEIGHTS))
            paid = {"Approved": amount, "Partial": round(amount / 2, 2)}.get(status, 0.0)
            denied = amount if status == "Denied" else 0.0
            needs_sub = bool(is_card and paid > 0 and rng.random() < SUBSTANTIATION_RATE)
            ticket = None
            if needs_sub:
                n_tickets += 1
                ticket = f"NSET-{n_tickets:05d}"
            swipes.append({
                "_id": f"WEX-{n:06d}",  # String _id, not an ObjectId
                "claim_nbr": str(900000 + n),
                "nswid": person["_base64Id"],
                "clientId": client["_id"],
                "employeeId": person["_id"],
                "alias": client["employer"]["claimReferenceId"],
                "date_of_service": iso(service_date),
                "provider": providers[int(rng.integers(len(providers)))],
                "amount": money_str(amount),
                "paid_amt": money_str(paid),
                "denied_amt": money_str(denied),
                "claim_status": status,
                "service_category": category,
                "claim_source_desc": "Debit card" if is_card else "Manual",
                "needsSubstantion": needs_sub,  # (sic) misspelled in the real data
                "ticketNumber": ticket,
            })
    return swipes


def _ticket(rng, number, person, client_id, ticket_type, category, opened: date, reference: date,
            amount: float) -> dict:
    created = datetime(opened.year, opened.month, opened.day, int(rng.integers(8, 18)))
    created = min(created, datetime(reference.year, reference.month, reference.day, 8))
    status = str(rng.choice(TICKET_STATUSES, p=TICKET_STATUS_WEIGHTS))
    closed = created + timedelta(hours=float(rng.lognormal(3.3, 0.8))) if status == "Closed" else None
    condition = None
    if category and status != "Closed":
        condition = str(rng.choice(["Notice 1", "Notice 2", "Notice 3"], p=[0.6, 0.3, 0.1]))
    return {
        "ticketNumber": number,
        "type": ticket_type,
        "category": category,
        "status": status,
        "priority": str(rng.choice(["Low", "Medium", "High"], p=[0.3, 0.55, 0.15])),
        "escalation": "standard" if rng.random() < 0.9 else "escalated",
        "condition": condition,
        "isVip": bool(person["MemberTags"]["vipStatus"]),
        "employeeId": person["_id"],
        "clientId": client_id,
        "createdAt": iso_dt(created),
        "closedAt": iso_dt(closed) if closed else None,
        "amountUnrecovered": money_str(0.0 if status == "Closed" else amount),
    }


def make_tickets(rng, people: list[dict], swipes: list[dict], reference: date) -> list[dict]:
    """A Substantiation ticket per flagged swipe, then general support tickets."""
    people_by_id = {p["_id"]: p for p in people}
    tickets = []
    for swipe in swipes:
        if swipe["ticketNumber"]:
            opened = date.fromisoformat(swipe["date_of_service"][:10]) + timedelta(days=int(rng.integers(1, 4)))
            tickets.append(_ticket(
                rng, swipe["ticketNumber"], people_by_id[swipe["employeeId"]], swipe["clientId"],
                "Claim", "Substantiation", opened, reference, _to_float(swipe["paid_amt"]),
            ))
    for person in people:
        if rng.random() < SUPPORT_TICKET_RATE:
            opened = random_date(rng, date(reference.year, 1, 1), reference)
            tickets.append(_ticket(
                rng, f"NSET-{len(tickets) + 1:05d}", person, person["Application"],
                str(rng.choice(SUPPORT_TYPES)), None, opened, reference, 0.0,
            ))
    return tickets


def make_claims(rng, clients: list[dict], people: list[dict], reference: date) -> list[dict]:
    """Manual reimbursement claims (v2 `claims` workqueue) for a share of enrolled members."""
    claims = []
    for person, client, _benefit, start, end in enrollments(clients, people, reference):
        if rng.random() >= MANUAL_CLAIM_RATE:
            continue
        n = len(claims) + 1
        amount = round(float(np.clip(rng.lognormal(5.5, 0.9), 20, 20000)), 2)
        status = str(rng.choice(CLAIM_STATUSES, p=CLAIM_STATUS_WEIGHTS))
        service_date = random_date(rng, start, end)
        created = service_date + timedelta(days=int(rng.integers(1, 11)))
        closed = created + timedelta(days=int(rng.integers(1, 21))) if status in ("Approved", "Denied") else None
        high = amount > HIGH_DOLLAR
        claims.append({
            "_id": oid("c0", n),
            "claimNumber": f"CLM-{100000 + n}",
            "clientId": client["_id"],
            "employeeId": person["_id"],
            "type": "Medical" if rng.random() < 0.95 else "Dental",
            "serviceCategory": str(rng.choice(SERVICE_CATEGORIES, p=SERVICE_CATEGORY_WEIGHTS)),
            "dateOfService": iso(service_date),
            "amount": amount,
            "requestedAmount": amount,
            "amountApproved": amount if status == "Approved" else 0.0,
            "amountDenied": amount if status == "Denied" else 0.0,
            "status": "Approved - High Dollar" if status == "Approved" and high else status,
            "escalation": "High Dollar" if high else str(
                rng.choice(["Standard", "Potential Duplicate", "Needs Review"], p=[0.9, 0.05, 0.05])),
            "priority": "High" if high else str(rng.choice(["Low", "Medium"], p=[0.4, 0.6])),
            "condition": {"Approved": "Paid", "Denied": "Denied"}.get(status, "Waiting on Member - Notice 1"),
            "payee": str(rng.choice(["Member", "Provider"])),
            "denialReason": "Not a qualified medical expense" if status == "Denied" else None,
            "createdAt": iso(created),
            "closedAt": iso(closed) if closed else None,
            "deleted": bool(rng.random() < 0.01),
        })
    return claims