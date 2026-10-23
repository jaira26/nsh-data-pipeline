"""Synthetic members: `people` documents with array-of-versions history (data contract A1.2)."""
from datetime import date

from nsh_pipeline.generate.common import iso, mdy, oid, random_date
from nsh_pipeline.utils.keys import make_nswid

TIERS = ["EE", "ES", "EC", "EF"]
TIER_WEIGHTS = [0.50, 0.15, 0.15, 0.20]
NAME_CHANGE_RATE = 0.05   # old last name kept as a Deleted entry
JOB_CHANGE_RATE = 0.10    # old job kept as an inactive entry
TERMINATION_RATE = 0.08
COBRA_RATE = 0.35         # share of terminated members who continue on COBRA
WAIVE_RATE = 0.10         # members with no medical enrollment
VIP_RATE = 0.001


def _person(rng, fake, n: int, client_id: str, plans: list[dict], client_since: date, reference: date) -> dict:
    first, last = fake.first_name(), fake.last_name()
    ssn = fake.ssn()
    birthdate = random_date(rng, date(reference.year - 70, 1, 1), date(reference.year - 18, 1, 1))

    names = []
    if rng.random() < NAME_CHANGE_RATE:
        names.append({"FirstName": first, "LastName": fake.last_name(), "Deleted": True})
    names.append({"FirstName": first, "LastName": last, "Deleted": False})

    hire = random_date(rng, max(client_since, date(reference.year - 15, 1, 1)), reference)
    term = random_date(rng, hire, reference) if rng.random() < TERMINATION_RATE else None
    cobra = term is not None and rng.random() < COBRA_RATE

    employment = []
    if rng.random() < JOB_CHANGE_RATE:
        employment.append({"HireDate": iso(hire), "EmployeeStatus": "active",
                           "Title": "Previous role", "Active": False})
    employment.append({"HireDate": iso(hire), "TerminationDate": iso(term) if term else None,
                       "EmployeeStatus": "cobra" if cobra else "active", "Active": True})

    benefits = []
    if plans and rng.random() >= WAIVE_RATE:
        plan = plans[int(rng.integers(len(plans)))]
        benefits.append({
            "Type": "Medical", "Carrier": plan["carrier"], "CoverageName": plan["name"],
            "CoverageTier": str(rng.choice(TIERS, p=TIER_WEIGHTS)),
            "CurrentEffectiveDate": iso(max(hire, date(reference.year, 1, 1))),
            "TerminationDate": iso(term) if term and not cobra else None,
            "IsCobra": cobra, "Premium": plan["premium"], "Deleted": False,
        })

    return {
        "_id": oid("b0", n),
        "Application": client_id,
        "_base64Id": make_nswid(last, ssn),
        "Names": names,
        "PersonalData": [{"Ssn": ssn, "Birthdate": mdy(birthdate),
                          "Gender": str(rng.choice(["F", "M"])), "Deleted": False}],
        "Employment": employment,
        "Benefits": benefits,
        "MemberTags": {"vipStatus": bool(rng.random() < VIP_RATE)},
    }


def make_people(rng, fake, clients: list[dict], reference: date, max_members_per_client: int) -> list[dict]:
    """Members for every client: min(group size, cap) people each."""
    people = []
    for client in clients:
        employer = client["employer"]
        plans = [p for p in client["benefits"] if p["category"] == "nsw"]
        client_since = date.fromisoformat(employer["clientSince"][:10])
        for _ in range(min(employer["groupSize"], max_members_per_client)):
            people.append(_person(rng, fake, len(people) + 1, client["_id"], plans,
                                  client_since, reference))
    return people