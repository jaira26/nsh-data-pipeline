import pandas as pd
import pytest

from nsh_pipeline.validation.schemas import validate_frame
from nsh_pipeline.validation.schemas.features import FEATURE_SCHEMAS

A, B = "a00000000000000000000001", "a00000000000000000000002"

GOOD = {
    "book_of_business": pd.DataFrame({
        "client_id": [A, B], "member_count": [3, 1], "enrollment_count": [3, 1],
        "ytd_billed_usd": [2000.0, 500.0], "ytd_claims_usd": [900.0, 0.0],
    }),
    "pmpm_monthly": pd.DataFrame({
        "client_id": [A, A], "month": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True),
        "member_months": [3, 3], "claims_pmpm": [100.0, 120.0], "billed_pmpm": [300.0, 300.0],
        "mlr": [0.33, 0.4],
    }),
    "substantiation_rate": pd.DataFrame({
        "client_id": [A, B], "quarter": ["2025Q1", "2025Q1"], "debit_swipe_count": [40, 10],
        "sub_ticket_count": [1, 0], "sub_rate_pct": [2.5, 0.0],
    }),
    "claims_aging": pd.DataFrame({
        "client_id": [A, A], "status": ["New", "In Progress"], "open_count": [2, 1],
        "avg_age_days": [4.0, 10.0], "p90_age_days": [6.0, 10.0], "sla_breach_pct": [0.0, 100.0],
    }),
}


def errors_for(df, table):
    result = validate_frame(df, FEATURE_SCHEMAS[table], table)
    assert result["status"] == "failed"
    return result["errors"]


@pytest.mark.parametrize("table", list(FEATURE_SCHEMAS))
def test_good_feature_tables_pass(table):
    assert validate_frame(GOOD[table], FEATURE_SCHEMAS[table], table)["status"] == "passed"


@pytest.mark.parametrize("table", list(FEATURE_SCHEMAS))
def test_metric_columns_are_optional(table):
    keys = [c for c, col in FEATURE_SCHEMAS[table].columns.items() if col.required]
    assert validate_frame(GOOD[table][keys], FEATURE_SCHEMAS[table], table)["status"] == "passed"


@pytest.mark.parametrize("table, column, value, check", [
    ("pmpm_monthly", "mlr", -0.1, "greater_than_or_equal_to"),
    ("substantiation_rate", "sub_rate_pct", 101.0, "less_than_or_equal_to"),
    ("claims_aging", "sla_breach_pct", 150.0, "less_than_or_equal_to"),
    ("book_of_business", "ytd_billed_usd", -5.0, "greater_than_or_equal_to"),
])
def test_out_of_range_metrics_fail(table, column, value, check):
    df = GOOD[table].copy()
    df.loc[0, column] = value
    assert any(e["column"] == column and check in e["check"] for e in errors_for(df, table))


@pytest.mark.parametrize("table", list(FEATURE_SCHEMAS))
def test_one_row_per_key(table):
    df = pd.concat([GOOD[table], GOOD[table].head(1)], ignore_index=True)
    assert any("unique" in e["check"] for e in errors_for(df, table))


def test_missing_key_column_fails():
    df = GOOD["pmpm_monthly"].drop(columns=["month"])
    assert any(e["check"] == "column_in_dataframe" for e in errors_for(df, "pmpm_monthly"))
