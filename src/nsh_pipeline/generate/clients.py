"""Synthetic clients: `applications` documents shaped like data contract A1.1."""
import re
from datetime import date

import numpy as np

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.generate.common import iso, oid, random_date

CARRIERS = ["Aetna", "Cigna", "Anthem", "BSC", "BCBS"]
STATES = ["OR", "WA", "CA", "ID", "NV", "AZ", "UT", "TX", "NY", "CO"]
STATE_WEIGHTS = [0.25, 0.20, 0.15, 0.08, 0.06, 0.06, 0.05, 0.07, 0.04, 0.04]
STATUSES = ["Active", "Terminated", "Run Out", "Pending", "Implementation"]
STATUS_WEIGHTS = [0.85, 0.08, 0.03, 0.02, 0.02]
DEDUCTIBLES = [1000, 2000, 3000, 5000]


def _nsw_plan(rng, carrier: str) -> dict:
    deductible = int(rng.choice(DEDUCTIBLES))
    premium = round(float(rng.uniform(450, 900)), 2)
    return {
        "type": "Medical", "carrier": carrier, "name": f"NSW {carrier} {deductible}/{deductible * 2}",
        "category": "nsw", "contracted": True, "familyIndicator": "EE", "premium": premium,
        "adminFee": 35.0, "reserveCharge": round(premium * 0.3, 2), "brokerFees": 12.5,
        "individualDeductibleString": str(deductible),
        "individualOopString": str(deductible * 2), "Deleted": False,
    }


def _benchmark_plan(rng, carrier: str) -> dict:
    return {
        "type": "Medical", "carrier": carrier, "name": f"{carrier} PPO Benchmark",
        "category": "benchmark", "contracted": False,
        "premium": round(float(rng.uniform(600, 1100)), 2), "Deleted": False,
    }


def _client(_id, name, alias, wex_id, ein, status, group_size, state, renewal_month,
            client_since: date, reference: date, benefits) -> dict:
    return {
        "_id": _id,
        "status": status.lower(),  # top-level status is lowercase: contract notes casing drift
        "dnnPortalAlias": alias,
        "healthStatus": "good",
        "employer": {
            "name": name, "legalName": f"{name} LLC", "ein": ein, "claimReferenceId": wex_id,
            "groupSize": group_size, "state": state, "status": status,
            "renewalMonth": renewal_month, "clientSince": iso(client_since),
        },
        "planYears": [{
            "name": str(reference.year),
            "planYearEffectiveDate": iso(date(reference.year, 1, 1)),
            "planYearEndDate": iso(date(reference.year, 12, 31)),
        }],
        "benefits": benefits,
    }


def make_clients(rng, fake, n_clients: int, reference: date) -> list[dict]:
    """`n_clients` synthetic clients plus the two special orgs (NS Demo, NSAI) at the end."""
    clients = []
    for i in range(1, n_clients + 1):
        name = fake.unique.company()
        code = re.sub(r"[^A-Za-z]", "", name).upper()[:6]
        carriers = [str(c) for c in rng.choice(CARRIERS, size=int(rng.integers(2, 5)), replace=False)]
        benefits = [_nsw_plan(rng, c) for c in carriers[:-1]] + [_benchmark_plan(rng, carriers[-1])]
        clients.append(_client(
            _id=oid("a0", i),
            name=name,
            alias=re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            wex_id=f"{code}{i:03d}",
            ein=f"{int(rng.integers(10, 100))}-{int(rng.integers(1_000_000, 10_000_000))}",
            status=str(rng.choice(STATUSES, p=STATUS_WEIGHTS)),
            group_size=int(np.clip(round(rng.lognormal(2.8, 1.0)), 2, 1500)),  # many small, few large
            state=str(rng.choice(STATES, p=STATE_WEIGHTS)),
            renewal_month=int(rng.integers(1, 13)),
            client_since=random_date(rng, date(2016, 1, 1), date(reference.year - 1, 12, 31)),
            reference=reference,
            benefits=benefits,
        ))
    for _id, name, alias, wex_id, size in [
        (NS_DEMO_ID, "NS Demo", "demo", "NSDEMO", 3),
        (NSAI_ID, "Nonstop Administrators Inc", "nsais", "NSAI", 5),
    ]:
        clients.append(_client(
            _id, name, alias, wex_id, "00-0000000", "Active", size, "OR", 1,
            date(2016, 1, 1), reference, [_nsw_plan(rng, "Aetna")],
        ))
    return clients