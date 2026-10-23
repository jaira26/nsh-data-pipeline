"""Flatten raw NSH collections into clean, typed, one-row-per-entity tables (plan section 6.2).

Each `clean_<table>` takes the raw documents of one collection (a list of dicts, as read
from data/raw/*.jsonl) and returns a DataFrame with exactly the columns the team agreed on.
"""
import numpy as np
import pandas as pd

from nsh_pipeline.constants import EXCLUDED_ORG_IDS
from nsh_pipeline.preprocess.dates import parse_dates
from nsh_pipeline.preprocess.exclusions import drop_excluded, flag_excluded_orgs
from nsh_pipeline.preprocess.money import normalize_money
from nsh_pipeline.preprocess.people_current import current_entry
from nsh_pipeline.preprocess.privacy import age_bands, assert_no_phi, drop_phi, mask_ssn
from nsh_pipeline.utils.keys import make_nswid

CLIENT_COLUMNS = [
    "client_id", "name", "ein", "wex_id", "employer_code", "group_size", "group_size_bucket",
    "state", "client_type", "status", "renewal_month", "client_since", "is_excluded_org",
]
MEMBER_COLUMNS = [
    "member_id", "client_id", "nswid", "first_name", "last_name", "ssn_masked", "age_band",
    "gender", "employee_status", "hire_date", "term_date", "is_vip", "coverage_tier",
]

GROUP_SIZE_BINS = [0, 9, 24, 49, 99, 249, float("inf")]
GROUP_SIZE_LABELS = ["1-9", "10-24", "25-49", "50-99", "100-249", "250+"]


def _utc(as_of) -> pd.Timestamp:
    """Return `as_of` as a UTC timestamp (naive values are treated as UTC)."""
    ts = pd.Timestamp(as_of)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _title(value):
    return value.strip().title() if isinstance(value, str) else None


def clean_clients(applications: list[dict]) -> pd.DataFrame:
    """One row per client from `applications`; NS Demo / NSAI are kept but flagged."""
    rows = []
    for app in applications:
        employer = app.get("employer") or {}
        rows.append({
            "client_id": app.get("_id"),
            "name": employer.get("name"),
            "ein": employer.get("ein"),
            "wex_id": employer.get("claimReferenceId"),
            # claimReferenceId is the client's WEX ID; card and ledger feeds call it employer_code (contract A3)
            "employer_code": employer.get("claimReferenceId"),
            "group_size": employer.get("groupSize"),
            "state": employer.get("state"),
            "client_type": employer.get("clientType"),
            "status": _title(employer.get("status") or app.get("status")),
            "renewal_month": employer.get("renewalMonth"),
            "client_since": employer.get("clientSince"),
            "dnn_alias": app.get("dnnPortalAlias"),
        })
    df = pd.DataFrame(rows)
    df["group_size"] = pd.to_numeric(df["group_size"], errors="coerce")
    df["group_size_bucket"] = pd.cut(
        df["group_size"], bins=GROUP_SIZE_BINS, labels=GROUP_SIZE_LABELS
    ).astype(object)
    df["renewal_month"] = pd.to_numeric(df["renewal_month"], errors="coerce")
    df = parse_dates(df, {"client_since": "iso"})
    df = flag_excluded_orgs(df, id_col="client_id", alias_col="dnn_alias")
    return df[CLIENT_COLUMNS]


def _employee_status(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """4-bucket census status as of `as_of` (contract A5). Our reading of the rules; NSH to confirm."""
    has_benefit = (
        df["benefit_start"].notna()
        & (df["benefit_start"] <= as_of)
        & (df["benefit_end"].isna() | (df["benefit_end"] > as_of))
    )
    terminated = df["term_date"].notna() & (df["term_date"] <= as_of)
    cobra = df["employment_status"].fillna("").str.lower().eq("cobra") | df["is_cobra"]
    status = np.select(
        [cobra & has_benefit, terminated, has_benefit],
        ["COBRA Enrolled", "Other", "Enrolled"],
        default="Not Enrolled",
    )
    return pd.Series(status, index=df.index)


def clean_members(people: list[dict], as_of, excluded_ids=EXCLUDED_ORG_IDS) -> pd.DataFrame:
    """One row per member from `people`, using current versions; excluded orgs and PHI removed."""
    as_of = _utc(as_of)
    rows = []
    for person in people:
        name = current_entry(person.get("Names")) or {}
        personal = current_entry(person.get("PersonalData")) or {}
        job = current_entry(person.get("Employment")) or {}
        medical_plans = [
            b for b in person.get("Benefits") or [] if isinstance(b, dict) and b.get("Type") == "Medical"
        ]
        medical = current_entry(medical_plans) or {}
        ssn = personal.get("Ssn")
        rows.append({
            "member_id": person.get("_id"),
            "client_id": person.get("Application"),
            "nswid": person.get("_base64Id") or make_nswid(name.get("LastName"), ssn),
            "first_name": name.get("FirstName"),
            "last_name": name.get("LastName"),
            "ssn": ssn,
            "birthdate": personal.get("Birthdate"),
            "gender": personal.get("Gender"),
            "employment_status": job.get("EmployeeStatus"),
            "hire_date": job.get("HireDate"),
            "term_date": job.get("TerminationDate"),
            "is_vip": bool((person.get("MemberTags") or {}).get("vipStatus", False)),
            "coverage_tier": medical.get("CoverageTier"),
            "is_cobra": bool(medical.get("IsCobra", False)),
            "benefit_start": medical.get("CurrentEffectiveDate"),
            "benefit_end": medical.get("TerminationDate"),
        })
    df = drop_excluded(pd.DataFrame(rows), "client_id", excluded_ids)
    df = parse_dates(df, {
        "birthdate": "mdy", "hire_date": "iso", "term_date": "iso",
        "benefit_start": "iso", "benefit_end": "iso",
    })
    df["ssn_masked"] = df["ssn"].map(mask_ssn)
    df["age_band"] = age_bands(df["birthdate"], as_of)
    df["employee_status"] = _employee_status(df, as_of)
    df = drop_phi(df)
    assert_no_phi(df, "members")
    return df[MEMBER_COLUMNS]

# ---------------------------------------------------------------- swipes, claims, tickets
SWIPE_COLUMNS = [
    "swipe_id", "member_id", "client_id", "nswid", "service_date", "amount_usd", "paid_usd",
    "denied_usd", "status", "service_category", "is_debit_card", "needs_substantiation",
    "ticket_number",
]
CLAIM_COLUMNS = [
    "claim_number", "member_id", "client_id", "service_category", "requested_usd",
    "approved_usd", "denied_usd", "status", "escalation", "priority", "created_at", "closed_at",
]
TICKET_COLUMNS = [
    "ticket_number", "member_id", "client_id", "type", "category", "status", "priority",
    "is_vip", "created_at", "closed_at", "sla_breached",
]

SWIPE_MONEY_UNITS = {"amount": "string", "paid_amt": "string", "denied_amt": "string"}
CLAIM_MONEY_UNITS = {"requestedAmount": "dollars", "amountApproved": "dollars", "amountDenied": "dollars"}
TICKET_SLA_HOURS = 40  # contract A1.6; measured in wall-clock hours (NSH to confirm)


def _frame(records: list[dict], fields: list[str]) -> pd.DataFrame:
    """DataFrame of `records` with exactly `fields` as columns (fields missing everywhere become NaN)."""
    return pd.DataFrame(records).reindex(columns=fields)


def clean_swipes(wex_claims: list[dict], excluded_ids=EXCLUDED_ORG_IDS) -> pd.DataFrame:
    """One row per debit-card swipe from `wex-claims`; money strings become dollars."""
    df = _frame(wex_claims, [
        "_id", "employeeId", "clientId", "nswid", "date_of_service", "amount", "paid_amt",
        "denied_amt", "claim_status", "service_category", "claim_source_desc",
        "needsSubstantion", "ticketNumber",
    ])
    df = drop_excluded(df, "clientId", excluded_ids)
    df = normalize_money(df, SWIPE_MONEY_UNITS)
    df = parse_dates(df, {"date_of_service": "iso"})
    df["is_debit_card"] = df["claim_source_desc"].eq("Debit card")
    df["needs_substantiation"] = df["needsSubstantion"].eq(True)  # (sic) field name in raw data
    df = df.rename(columns={
        "_id": "swipe_id", "employeeId": "member_id", "clientId": "client_id",
        "date_of_service": "service_date", "paid_amt_usd": "paid_usd",
        "denied_amt_usd": "denied_usd", "claim_status": "status", "ticketNumber": "ticket_number",
    })
    return df[SWIPE_COLUMNS]


def clean_claims(claims: list[dict], excluded_ids=EXCLUDED_ORG_IDS) -> pd.DataFrame:
    """One row per manual claim from `claims`; deleted claims are removed."""
    df = _frame(claims, [
        "claimNumber", "employeeId", "clientId", "serviceCategory", "requestedAmount",
        "amountApproved", "amountDenied", "status", "escalation", "priority", "createdAt",
        "closedAt", "deleted",
    ])
    df = df[~df["deleted"].eq(True)]
    df = drop_excluded(df, "clientId", excluded_ids)
    df = normalize_money(df, CLAIM_MONEY_UNITS)
    df = parse_dates(df, {"createdAt": "iso", "closedAt": "iso"})
    df = df.rename(columns={
        "claimNumber": "claim_number", "employeeId": "member_id", "clientId": "client_id",
        "serviceCategory": "service_category", "requestedAmount_usd": "requested_usd",
        "amountApproved_usd": "approved_usd", "amountDenied_usd": "denied_usd",
        "createdAt": "created_at", "closedAt": "closed_at",
    })
    return df[CLAIM_COLUMNS]


def clean_tickets(tickets: list[dict], as_of, excluded_ids=EXCLUDED_ORG_IDS) -> pd.DataFrame:
    """One row per support ticket; `sla_breached` if open or resolved longer than the SLA."""
    as_of = _utc(as_of)
    df = _frame(tickets, [
        "ticketNumber", "employeeId", "clientId", "type", "category", "status", "priority",
        "isVip", "createdAt", "closedAt",
    ])
    df = drop_excluded(df, "clientId", excluded_ids)
    df = parse_dates(df, {"createdAt": "iso", "closedAt": "iso"})
    df["is_vip"] = df["isVip"].eq(True)
    end = df["closedAt"].fillna(as_of)
    df["sla_breached"] = (end - df["createdAt"]) > pd.Timedelta(hours=TICKET_SLA_HOURS)
    df = df.rename(columns={
        "ticketNumber": "ticket_number", "employeeId": "member_id", "clientId": "client_id",
        "createdAt": "created_at", "closedAt": "closed_at",
    })
    return df[TICKET_COLUMNS]