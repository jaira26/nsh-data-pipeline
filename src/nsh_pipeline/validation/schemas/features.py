"""Schemas for the 4 feature tables (plan section 6.3).

`src/nsh_pipeline/features/` is not built yet, so these follow the agreed plan: key columns
are required and unique together; metric columns are checked for range when present. A
missing features directory or table is reported as "skipped", never as a failure.
"""
from pandera.pandas import Column, DataFrameSchema

from nsh_pipeline.validation.schemas.checks import any_value, money, number, text


def _metric(ge=0, le=None) -> Column:
    return number(nullable=True, ge=ge, le=le, required=False)


def _schema(name: str, columns: dict, unique: list[str]) -> DataFrameSchema:
    return DataFrameSchema(columns, name=name, strict=False, unique=unique)


FEATURE_SCHEMAS: dict[str, DataFrameSchema] = {
    "book_of_business": _schema("book_of_business", {
        "client_id": text(nullable=False),
        "member_count": _metric(),
        "enrollment_count": _metric(),
        "ytd_billed_usd": money(ge=0, required=False),
        "ytd_claims_usd": money(ge=0, required=False),
    }, unique=["client_id"]),
    "pmpm_monthly": _schema("pmpm_monthly", {
        "client_id": text(nullable=False),
        "month": any_value(nullable=False),
        "member_months": _metric(),
        "claims_pmpm": _metric(),
        "billed_pmpm": _metric(),
        "mlr": _metric(),
    }, unique=["client_id", "month"]),
    "substantiation_rate": _schema("substantiation_rate", {
        "client_id": text(nullable=False),
        "quarter": any_value(nullable=False),
        "debit_swipe_count": _metric(),
        "sub_ticket_count": _metric(),
        "sub_rate_pct": _metric(le=100),
    }, unique=["client_id", "quarter"]),
    "claims_aging": _schema("claims_aging", {
        "client_id": text(nullable=False),
        "status": text(nullable=False),
        "open_count": _metric(),
        "avg_age_days": _metric(),
        "p90_age_days": _metric(),
        "sla_breach_pct": _metric(le=100),
    }, unique=["client_id", "status"]),
}
