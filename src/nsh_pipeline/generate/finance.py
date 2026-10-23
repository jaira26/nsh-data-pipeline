"""Synthetic finance collections: monthly `bills`, `reserves` and the NSE daily ledger."""
from datetime import date, timedelta

from nsh_pipeline.generate.activity import enrollments
from nsh_pipeline.generate.common import iso, money_str, oid
from nsh_pipeline.preprocess.people_current import current_entry

DOLLAR_SIGN_CLIENT_RATE = 0.05  # clients whose bill strings carry a "$" (e.g. "$35.00")
DENTAL_LINE_RATE = 0.05         # extra non-medical bill lines (exercise the medical filter)
RETRO_ADJUSTMENT_RATE = 0.05


def _num(money: str) -> float:
    return float(money.lstrip("$").replace(",", ""))


def _months(reference: date) -> list[date]:
    return [date(reference.year, m, 1) for m in range(1, reference.month + 1)]


def _month_end(month_start: date) -> date:
    next_month = date(month_start.year + month_start.month // 12, month_start.month % 12 + 1, 1)
    return next_month - timedelta(days=1)


def _ym(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def _paid_by_client_month(swipes: list[dict]) -> dict:
    totals = {}
    for swipe in swipes:
        key = (swipe["clientId"], swipe["date_of_service"][:7])
        totals[key] = totals.get(key, 0.0) + _num(swipe["paid_amt"])
    return totals


def make_bills(rng, clients: list[dict], people: list[dict], swipes: list[dict], reference: date) -> list[dict]:
    """One bill per client per month with a recurring line per enrolled member (money as strings)."""
    enrolled = {}
    for person, client, benefit, start, end in enrollments(clients, people, reference):
        enrolled.setdefault(client["_id"], []).append((person, benefit, start, end))
    paid = _paid_by_client_month(swipes)
    current_month = date(reference.year, reference.month, 1)
    bills = []
    for client in clients:
        members = enrolled.get(client["_id"], [])
        if not members:
            continue
        plans = {p["name"]: p for p in client["benefits"]}
        dollar_sign = bool(rng.random() < DOLLAR_SIGN_CLIENT_RATE)

        def fmt(amount, _dollar=dollar_sign):
            return ("$" if _dollar else "") + money_str(amount)

        for month in _months(reference):
            lines = []
            for person, benefit, start, end in members:
                if start > _month_end(month) or end < month:
                    continue
                plan = plans[benefit["CoverageName"]]
                lines.append({
                    "employeeId": person["_id"], "nswId": person["_base64Id"], "memberCount": "1",
                    "planName": plan["name"], "planFI": benefit["CoverageTier"], "carrier": plan["carrier"],
                    "premium": fmt(plan["premium"]), "adminFee": fmt(plan["adminFee"]),
                    "reserveCharge": fmt(plan["reserveCharge"]), "brokerFees": fmt(plan["brokerFees"]),
                    "type": "medical",
                })
                if rng.random() < DENTAL_LINE_RATE:
                    lines.append({
                        **lines[-1], "planName": "Dental Basic", "carrier": "Delta Dental",
                        "premium": fmt(40.0), "adminFee": fmt(0.0), "reserveCharge": fmt(0.0),
                        "brokerFees": fmt(0.0), "type": "dental",
                    })
            if not lines:
                continue
            adjustments = []
            if rng.random() < RETRO_ADJUSTMENT_RATE:
                adjustments.append({
                    "retroType": "termination",
                    "retroAmount": money_str(-round(float(rng.uniform(50, 500)), 2)),
                    "billingDateAffected": _ym(month - timedelta(days=1)),
                })
            closed = month < current_month
            bills.append({
                "_id": oid("f0", len(bills) + 1),
                "appId": client["_id"],
                "employer": client["employer"]["name"],
                "billingDate": _ym(month),
                "billingDateFormat": iso(month),
                "closed": closed,
                "locked": closed,
                "priorBalance": money_str(0.0),
                "clientPayments": money_str(sum(_num(line["premium"]) for line in lines)),
                "totalClaimSpend": money_str(paid.get((client["_id"], _ym(month)), 0.0)),
                "recurring": lines,
                "adjustments": adjustments,
            })
    return bills


def make_reserves(rng, clients: list[dict], bills: list[dict], swipes: list[dict]) -> list[dict]:
    """Monthly medical reserve balance per client: previous + reserve charges billed - claims paid."""
    paid = _paid_by_client_month(swipes)
    bills_by_client = {}
    for bill in bills:
        bills_by_client.setdefault(bill["appId"], []).append(bill)
    reserves = []
    for client in clients:
        client_bills = bills_by_client.get(client["_id"], [])
        if not client_bills:
            continue
        balance = round(float(rng.uniform(5_000, 50_000)), 2)  # opening balance
        for bill in client_bills:  # already in month order
            billed = round(sum(_num(line["reserveCharge"]) for line in bill["recurring"]
                               if line["type"] == "medical"), 2)
            claims_paid = round(paid.get((client["_id"], bill["billingDate"]), 0.0), 2)
            previous = balance
            balance = round(previous + billed - claims_paid, 2)
            year, month = (int(part) for part in bill["billingDate"].split("-"))
            reserves.append({
                "_id": oid("f1", len(reserves) + 1), "client": client["_id"], "bill": bill["_id"],
                "year": year, "month": month, "type": "Medical",
                "balance": balance, "previousBalance": previous, "reservesBilled": billed,
                "claimsPaid": claims_paid, "returnOfReserve": 0.0, "closed": bill["closed"],
            })
    return reserves


def make_ledger(clients: list[dict], people: list[dict], swipes: list[dict], claims: list[dict]) -> list[dict]:
    """NSE daily ledger in integer CENTS: one row per paid swipe and per approved manual claim."""
    codes = {c["_id"]: c["employer"]["claimReferenceId"] for c in clients}
    names = {p["_id"]: current_entry(p["Names"]) or {} for p in people}
    rows = []

    def add(person_id, client_id, cents, service_date, source, payee):
        name = names[person_id]
        effective = date.fromisoformat(service_date[:10]) + timedelta(days=2)
        rows.append({
            "_id": oid("f2", len(rows) + 1), "employerCode": codes[client_id], "employeeId": person_id,
            "consumerLastName": name.get("LastName"), "consumerFirstName": name.get("FirstName"),
            "amount": cents, "dateOfService": service_date, "effectiveDate": iso(effective),
            "method": "DC" if source == "wex" else "ACH", "payee": payee,
            "status": "processed", "nseSource": source, "processed": True,
        })

    for swipe in swipes:
        cents = round(_num(swipe["paid_amt"]) * 100)
        if cents > 0:
            add(swipe["employeeId"], swipe["clientId"], cents, swipe["date_of_service"], "wex", "Provider")
    for claim in claims:
        if claim["amountApproved"] > 0 and not claim["deleted"]:
            add(claim["employeeId"], claim["clientId"], round(claim["amountApproved"] * 100),
                claim["dateOfService"], "manual", claim["payee"])
    return rows