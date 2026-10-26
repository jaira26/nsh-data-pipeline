"""Strict schemas for the 9 processed tables (plan section 6.2).

Columns, and their order, come from the *_COLUMNS lists in `preprocess/flatten.py`: those
lists are the contract. Schemas are strict, so an extra column (e.g. a leaked raw SSN) fails.

What the schemas enforce: types (money = float dollars, dates = UTC), primary keys unique and
not null, id formats, ranges, and enums our own preprocessing derives. What they leave to
`anomalies`: null or orphan join keys, missing money, source-system enum drift and impossible
values. Those have their own severities (plan section 8, stage 6) instead of failing the schema.
"""
from pandera.pandas import DataFrameSchema

from nsh_pipeline.preprocess import flatten
from nsh_pipeline.validation.enums import DERIVED_ENUMS
from nsh_pipeline.validation.schemas.checks import flag, money, number, text, utc

OBJECT_ID = r"[0-9a-f]{24}"


def _derived(table: str, column: str, nullable: bool = True):
    return text(nullable=nullable, isin=DERIVED_ENUMS[(table, column)])


def _schema(name: str, columns: dict, column_order: list[str], unique=None) -> DataFrameSchema:
    if list(columns) != column_order:
        raise ValueError(f"{name} schema columns {list(columns)} do not match flatten.py {column_order}")
    return DataFrameSchema(columns, name=name, strict=True, ordered=True, unique=unique)


CLIENTS = _schema("clients", {
    "client_id": text(nullable=False, unique=True, pattern=OBJECT_ID),
    "name": text(nullable=False),
    "ein": text(pattern=r"\d{2}-\d{7}"),
    "wex_id": text(),
    "employer_code": text(),
    "group_size": number(nullable=True, ge=1, whole=True),
    "group_size_bucket": _derived("clients", "group_size_bucket"),
    "state": text(pattern=r"[A-Z]{2}"),
    "client_type": text(),
    "status": text(),
    "renewal_month": number(nullable=True, ge=1, le=12, whole=True),
    "client_since": utc(),
    "is_excluded_org": flag(),
}, flatten.CLIENT_COLUMNS)

MEMBERS = _schema("members", {
    "member_id": text(nullable=False, unique=True, pattern=OBJECT_ID),
    "client_id": text(),
    "nswid": text(),
    "first_name": text(),
    "last_name": text(),
    "ssn_masked": text(pattern=r"\*\*\*-\*\*-\d{4}"),
    "age_band": _derived("members", "age_band"),
    "gender": text(),
    "employee_status": _derived("members", "employee_status", nullable=False),
    "hire_date": utc(),
    "term_date": utc(),
    "is_vip": flag(),
    "coverage_tier": text(),
}, flatten.MEMBER_COLUMNS)

MEMBER_BENEFITS = _schema("member_benefits", {
    "member_id": text(),
    "client_id": text(),
    "benefit_type": text(),
    "carrier": text(),
    "plan_name": text(),
    "coverage_tier": text(),
    "effective_date": utc(),
    "term_date": utc(),
    "is_cobra": flag(),
    "premium_usd": money(),
}, flatten.MEMBER_BENEFIT_COLUMNS)

SWIPES = _schema("swipes", {
    "swipe_id": text(nullable=False, unique=True),
    "member_id": text(),
    "client_id": text(),
    "nswid": text(),
    "service_date": utc(),
    "amount_usd": money(),
    "paid_usd": money(),
    "denied_usd": money(),
    "status": text(),
    "service_category": text(),
    "is_debit_card": flag(),
    "needs_substantiation": flag(),
    "ticket_number": text(pattern=r"NSET-\d{5}"),
}, flatten.SWIPE_COLUMNS)

CLAIMS = _schema("claims", {
    "claim_number": text(nullable=False, unique=True, pattern=r"CLM-1\d{5}"),
    "member_id": text(),
    "client_id": text(),
    "service_category": text(),
    "requested_usd": money(),
    "approved_usd": money(),
    "denied_usd": money(),
    "status": text(),
    "escalation": text(),
    "priority": text(),
    "created_at": utc(),
    "closed_at": utc(),
}, flatten.CLAIM_COLUMNS)

TICKETS = _schema("tickets", {
    "ticket_number": text(nullable=False, unique=True, pattern=r"NSET-\d{5}"),
    "member_id": text(),
    "client_id": text(),
    "type": text(),
    "category": text(),
    "status": text(),
    "priority": text(),
    "is_vip": flag(),
    "created_at": utc(nullable=False),  # sla_breached is computed from it
    "closed_at": utc(),
    "sla_breached": flag(),
}, flatten.TICKET_COLUMNS)

# No uniqueness on (member_id, billing_month): duplicate bill lines are kept on purpose
# (decisions.md #2) and reported by anomalies.duplicate_bill_lines as a critical finding.
BILL_LINES = _schema("bill_lines", {
    "client_id": text(),
    "billing_month": utc(nullable=False),
    "member_id": text(),
    "plan_name": text(),
    "carrier": text(),
    "premium_usd": money(),
    "admin_fee_usd": money(),
    "reserve_charge_usd": money(),
    "broker_fee_usd": money(),
    "member_count": number(nullable=True, ge=1, whole=True),
}, flatten.BILL_LINE_COLUMNS)

RESERVES = _schema("reserves", {
    "client_id": text(nullable=False),
    "month": utc(nullable=False),
    "balance_usd": money(),
    "reserves_billed_usd": money(),
    "claims_paid_usd": money(),
    "return_of_reserve_usd": money(),
}, flatten.RESERVE_COLUMNS, unique=["client_id", "month"])

LEDGER = _schema("ledger", {
    "txn_id": text(nullable=False, unique=True),
    "client_id": text(),
    "member_id": text(),
    "source": text(),
    "amount_usd": money(),
    "effective_date": utc(),
    "status": text(),
}, flatten.LEDGER_COLUMNS)

PROCESSED_SCHEMAS: dict[str, DataFrameSchema] = {
    "clients": CLIENTS,
    "members": MEMBERS,
    "member_benefits": MEMBER_BENEFITS,
    "swipes": SWIPES,
    "claims": CLAIMS,
    "tickets": TICKETS,
    "bill_lines": BILL_LINES,
    "reserves": RESERVES,
    "ledger": LEDGER,
}
