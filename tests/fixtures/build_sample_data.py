"""Build the tiny hand-written sample dataset in tests/fixtures/raw/.

Run from the repo root:  python tests/fixtures/build_sample_data.py

Records follow the shapes and quirks in 01-data-contract-package.md. Conventions:
ObjectIds are 24-hex strings, Mongo dates are ISO-8601 strings, and contract quirks
(M/D/YYYY string dates, money as strings or cents, misspelled field names) are kept as-is.
"""
import hashlib
import json
from pathlib import Path

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.utils.io import write_jsonl
from nsh_pipeline.utils.keys import make_nswid

OUT_DIR = Path(__file__).parent / "raw"


def oid(prefix, n):
    """Fake 24-hex ObjectId string, e.g. oid("a0", 1) -> "a0000...001"."""
    return f"{prefix}{n:0{24 - len(prefix)}d}"


# ---------------------------------------------------------------- clients
ACME, BIRCH, CEDAR = oid("a0", 1), oid("a0", 2), oid("a0", 3)

GOLD_PLAN = {
    "type": "Medical", "carrier": "Aetna", "name": "NSW Gold 2000/4000", "category": "nsw",
    "contracted": True, "familyIndicator": "EE", "premium": 650.0, "adminFee": 35.0,
    "reserveCharge": 210.0, "brokerFees": 12.5, "individualDeductibleString": "2000",
    "individualOopString": "4000", "Deleted": False,
}
BENCHMARK_PLAN = {
    "type": "Medical", "carrier": "Cigna", "name": "Cigna PPO 1500", "category": "benchmark",
    "contracted": False, "premium": 720.0, "Deleted": False,
}


def client(_id, name, alias, wex_id, ein, status, group_size, state, benefits=()):
    return {
        "_id": _id,
        "status": status.lower(),  # top-level status is lowercase: contract notes casing drift
        "dnnPortalAlias": alias,
        "healthStatus": "good",
        "employer": {
            "name": name, "legalName": f"{name} LLC", "ein": ein, "claimReferenceId": wex_id,
            "groupSize": group_size, "state": state, "status": status, "renewalMonth": 7,
            "clientSince": "2021-07-01T00:00:00Z",
        },
        "planYears": [{
            "name": "2025", "planYearEffectiveDate": "2025-01-01T00:00:00Z",
            "planYearEndDate": "2025-12-31T00:00:00Z",
        }],
        "benefits": list(benefits),
    }


APPLICATIONS = [
    client(ACME, "Acme Manufacturing", "acme", "ACMEWEX01", "12-3456789", "Active", 45, "OR",
           [GOLD_PLAN, BENCHMARK_PLAN]),
    client(BIRCH, "Birch Dental Group", "birch", "BIRCHWEX01", "98-7654321", "Active", 8, "WA",
           [GOLD_PLAN]),
    client(CEDAR, "Cedar Logistics", "cedar", "CEDARWEX01", "55-1234567", "Terminated", 120, "CA"),
    client(NS_DEMO_ID, "NS Demo", "demo", "NSDEMO", "00-0000001", "Active", 1, "OR"),
    client(NSAI_ID, "Nonstop Administrators Inc", "nsais", "NSAI", "00-0000002", "Active", 1, "OR"),
]


# ---------------------------------------------------------------- members
def person(n, client_id, first, last, ssn, dob, *, old_last=None, cobra=False,
           term_date=None, vip=False, tier="EE"):
    names = [{"FirstName": first, "LastName": old_last, "Deleted": True}] if old_last else []
    names.append({"FirstName": first, "LastName": last, "Deleted": False})
    return {
        "_id": oid("b0", n),
        "Application": client_id,
        "_base64Id": make_nswid(last, ssn),  # NSW ID = member join key
        "Names": names,
        "PersonalData": [{"Ssn": ssn, "Birthdate": dob, "Gender": "F" if n % 2 else "M",
                          "Deleted": False}],
        "Employment": [{"HireDate": "2020-03-01T00:00:00Z", "TerminationDate": term_date,
                        "EmployeeStatus": "cobra" if cobra else "active", "Active": True}],
        "Benefits": [{"Type": "Medical", "Carrier": "Aetna", "CoverageName": "NSW Gold 2000/4000",
                      "CoverageTier": tier, "CurrentEffectiveDate": "2025-01-01T00:00:00Z",
                      "TerminationDate": None, "IsCobra": cobra, "Premium": 650.0,
                      "Deleted": False}],
        "MemberTags": {"vipStatus": vip},
    }


P1 = person(1, ACME, "Jane", "Richardson", "123-45-6789", "4/12/1985", old_last="Miller", tier="ES")
P2 = person(2, ACME, "Wei", "Li", "987-65-4321", "11/3/1990", vip=True)
P3 = person(3, ACME, "Maria", "Garcia", "222-33-4444", "7/21/1978", cobra=True)
P4 = person(4, BIRCH, "Tom", "O'Brien", "555-12-3456", "1/30/1995",
            term_date="2025-03-15T00:00:00Z")
P5 = person(5, NS_DEMO_ID, "Demo", "User", "000-00-0001", "1/1/1980")
P6 = person(6, NSAI_ID, "Staff", "Member", "000-00-0002", "1/1/1985")
# P2 changed jobs: the old Employment entry is inactive.
P2["Employment"].insert(0, {"HireDate": "2018-06-01T00:00:00Z", "EmployeeStatus": "active",
                            "Title": "Intern", "Active": False})
PEOPLE = [P1, P2, P3, P4, P5, P6]


# ---------------------------------------------------------------- cards
def card(n, member, employer_code, plan_year, start, end, remaining, status="Active"):
    return {
        "_id": oid("e0", n), "employeeId": member["_id"], "nswid": member["_base64Id"],
        "employer_code": employer_code, "wex_id": employer_code, "plan_year": plan_year,
        "plan_name": "NSW Gold 2000/4000", "card_status": status, "account_status": status,
        "pariticpant_status": "Active",  # (sic) misspelled in the real data
        "valid_start_dt": start, "valid_end_dt": end, "enroll_dt": start, "STATUS": "MAILED",
        "max_oop": 4000.0, "remaining_balance": remaining,
    }


DEBIT_CARD_STATUS = [
    card(1, P1, "ACMEWEX01", "2024", "1/1/2024", "12/31/2024", 0.0, status="Inactive"),
    card(2, P1, "ACMEWEX01", "2025", "1/1/2025", "12/31/2025", 3624.25),
    card(3, P2, "ACMEWEX01", "2025", "1/1/2025", "12/31/2025", 4000.0),
    card(4, P4, "BIRCHWEX01", "2025", "1/1/2025", "12/31/2025", 3900.0),
]


# ---------------------------------------------------------------- swipes (wex-claims)
def swipe(n, member, client_id, alias, dos, amount, paid, denied, status, *,
          source="Debit card", needs_sub=False, ticket=None, category="Medical"):
    return {
        "_id": f"WEX-{n:06d}",  # String _id, not an ObjectId
        "claim_nbr": str(900000 + n), "nswid": member["_base64Id"], "clientId": client_id,
        "employeeId": member["_id"], "alias": alias, "date_of_service": dos,
        "provider": "Sample Clinic", "amount": amount, "paid_amt": paid, "denied_amt": denied,
        "claim_status": status, "service_category": category, "claim_source_desc": source,
        "needsSubstantion": needs_sub,  # (sic) misspelled in the real data
        "ticketNumber": ticket,
    }


WEX_CLAIMS = [
    swipe(1, P1, ACME, "ACMEWEX01", "2025-02-03T00:00:00Z", "125.00", "125.00", "0.00",
          "Approved", needs_sub=True, ticket="NSET-00001"),
    swipe(2, P1, ACME, "ACMEWEX01", "2025-02-10T00:00:00Z", "1,250.75", "1,250.75", "0.00",
          "Approved"),
    swipe(3, P2, ACME, "ACMEWEX01", "2025-02-14T00:00:00Z", "80.00", "0.00", "80.00",
          "Denied"),  # POS decline, not a claim denial
    swipe(4, P3, ACME, "ACMEWEX01", "2025-03-01T00:00:00Z", "45.10", "45.10", "0.00",
          "Approved", category="Drugs & Medicine"),
    swipe(5, P4, BIRCH, "BIRCHWEX01", "2025-03-05T00:00:00Z", "N/A", "N/A", "0.00",
          "Pended", source="Manual"),  # unparseable money must become NaN
    swipe(6, P5, NS_DEMO_ID, "NSDEMO", "2025-03-07T00:00:00Z", "60.00", "60.00", "0.00",
          "Approved"),  # demo org: must be excluded
]


# ---------------------------------------------------------------- tickets
TICKETS = [
    {"ticketNumber": "NSET-00001", "type": "Claim", "category": "Substantiation",
     "status": "Waiting on Member", "priority": "Medium", "escalation": "standard",
     "condition": "Notice 1", "isVip": False, "employeeId": P1["_id"], "clientId": ACME,
     "createdAt": "2025-02-04T15:00:00Z", "closedAt": None, "amountUnrecovered": "125.00"},
    {"ticketNumber": "NSET-00002", "type": "Customer Support", "category": None,
     "status": "Closed", "priority": "Low", "escalation": "standard", "condition": None,
     "isVip": True, "employeeId": P2["_id"], "clientId": ACME,
     "createdAt": "2025-02-15T10:00:00Z", "closedAt": "2025-02-16T09:30:00Z",
     "amountUnrecovered": "0.00"},
]


# ---------------------------------------------------------------- claims (v2 workqueue)
CLAIMS = [
    {"_id": oid("c0", 1), "claimNumber": "CLM-100001", "clientId": ACME, "employeeId": P1["_id"],
     "type": "Medical", "serviceCategory": "Medical", "dateOfService": "2025-01-20T00:00:00Z",
     "amount": 300.0, "requestedAmount": 300.0, "amountApproved": 300.0, "amountDenied": 0.0,
     "status": "Approved", "escalation": "Standard", "priority": "Medium", "condition": "Paid",
     "payee": "Member", "createdAt": "2025-01-22T00:00:00Z", "deleted": False},
    {"_id": oid("c0", 2), "claimNumber": "CLM-100002", "clientId": BIRCH, "employeeId": P4["_id"],
     "type": "Medical", "serviceCategory": "Medical", "dateOfService": "2025-02-11T00:00:00Z",
     "amount": 950.0, "requestedAmount": 950.0, "amountApproved": 0.0, "amountDenied": 950.0,
     "status": "Denied", "escalation": "High Dollar", "priority": "High", "condition": "Denied",
     "payee": "Provider", "denialReason": "Not a qualified medical expense",
     "createdAt": "2025-02-12T00:00:00Z", "deleted": False},
]


# ---------------------------------------------------------------- bills (money as STRINGS)
def bill_line(member, premium, admin="35.00", reserve="210.00", broker="12.50"):
    return {"employeeId": member["_id"], "nswId": member["_base64Id"], "memberCount": "1",
            "planName": "NSW Gold 2000/4000", "planFI": "EE", "carrier": "Aetna",
            "premium": premium, "adminFee": admin, "reserveCharge": reserve,
            "brokerFees": broker, "type": "medical"}


BILLS = [
    {"_id": oid("f0", 1), "appId": ACME, "employer": "Acme Manufacturing",
     "billingDate": "2025-01", "billingDateFormat": "2025-01-01T00:00:00Z",
     "closed": True, "locked": True, "priorBalance": "0.00", "clientPayments": "2,722.50",
     "totalClaimSpend": "1,420.85",
     "recurring": [bill_line(P1, "650.00"), bill_line(P2, "650.00"), bill_line(P3, "650.00")],
     "adjustments": [{"retroType": "termination", "retroAmount": "-325.00",
                      "billingDateAffected": "2024-12"}]},
    {"_id": oid("f0", 2), "appId": BIRCH, "employer": "Birch Dental Group",
     "billingDate": "2025-01", "billingDateFormat": "2025-01-01T00:00:00Z",
     "closed": True, "locked": True, "priorBalance": "0.00", "clientPayments": "1,357.50",
     "totalClaimSpend": "0.00",
     "recurring": [bill_line(P4, "1,100.00", admin="$35.00")],
     "adjustments": []},
]


# ---------------------------------------------------------------- reserves (plain dollars)
RESERVES = [
    {"_id": oid("f1", 1), "client": ACME, "bill": BILLS[0]["_id"], "year": 2025, "month": 1,
     "type": "Medical", "balance": 15630.0, "previousBalance": 15000.0, "reservesBilled": 630.0,
     "claimsPaid": 0.0, "returnOfReserve": 0.0, "closed": True},
    {"_id": oid("f1", 2), "client": BIRCH, "bill": BILLS[1]["_id"], "year": 2025, "month": 1,
     "type": "Medical", "balance": 4210.0, "previousBalance": 4000.0, "reservesBilled": 210.0,
     "claimsPaid": 0.0, "returnOfReserve": 0.0, "closed": True},
]


# ---------------------------------------------------------------- daily ledger (CENTS)
def ledger(n, member, employer_code, cents, dos, source="wex", payee="Provider"):
    last = member["Names"][-1]
    return {"_id": oid("f2", n), "employerCode": employer_code, "employeeId": member["_id"],
            "consumerLastName": last["LastName"], "consumerFirstName": last["FirstName"],
            "amount": cents, "dateOfService": dos, "effectiveDate": dos, "method": "DC",
            "payee": payee, "status": "processed", "nseSource": source, "processed": True}


NSE_TRANSACTION_DAILY_LOGS = [
    ledger(1, P1, "ACMEWEX01", 12500, "2025-02-03T00:00:00Z"),
    ledger(2, P1, "ACMEWEX01", 125075, "2025-02-10T00:00:00Z"),
    ledger(3, P3, "ACMEWEX01", 4510, "2025-03-01T00:00:00Z"),
    ledger(4, P1, "ACMEWEX01", 30000, "2025-01-20T00:00:00Z", source="manual", payee="Member"),
]


COLLECTIONS = {
    "applications": APPLICATIONS,
    "people": PEOPLE,
    "debit_card_status": DEBIT_CARD_STATUS,
    "wex_claims": WEX_CLAIMS,
    "tickets": TICKETS,
    "claims": CLAIMS,
    "bills": BILLS,
    "reserves": RESERVES,
    "nse_transaction_daily_logs": NSE_TRANSACTION_DAILY_LOGS,
}


def main():
    manifest = {"dataset": "sample-fixture", "version": "1", "files": {}}
    for name, records in COLLECTIONS.items():
        path = write_jsonl(records, OUT_DIR / f"{name}.jsonl")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][path.name] = {"rows": len(records), "sha256": digest}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(COLLECTIONS)} collections to {OUT_DIR}")


if __name__ == "__main__":
    main()