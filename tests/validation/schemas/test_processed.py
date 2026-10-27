import pandas as pd
import pytest

from nsh_pipeline.preprocess import flatten
from nsh_pipeline.validation.schemas import validate_frame
from nsh_pipeline.validation.schemas.processed import PROCESSED_SCHEMAS, _schema
from nsh_pipeline.validation.schemas.checks import text


def errors_for(df, table):
    result = validate_frame(df, PROCESSED_SCHEMAS[table], table)
    assert result["status"] == "failed", f"{table} unexpectedly passed"
    return result["errors"]


def assert_error(errors, column, check):
    assert any(e["column"] == column and check in e["check"] for e in errors), errors


@pytest.mark.parametrize("table", list(PROCESSED_SCHEMAS))
def test_sample_tables_pass(tables, table):
    result = validate_frame(tables[table], PROCESSED_SCHEMAS[table], table)
    assert result == {"table": table, "status": "passed", "rows": len(tables[table]), "errors": []}


def test_real_pandas2_style_frames_pass(tables):
    """The Airflow image runs pandas 2.1: text as object, timestamps in ns. Schemas must accept both."""
    for table, df in tables.items():
        df = df.copy()
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype(object)
            elif isinstance(df[col].dtype, pd.DatetimeTZDtype):
                df[col] = df[col].astype("datetime64[ns, UTC]")
        assert validate_frame(df, PROCESSED_SCHEMAS[table], table)["status"] == "passed", table


def test_schemas_follow_flatten_column_lists():
    pairs = {
        "clients": flatten.CLIENT_COLUMNS, "members": flatten.MEMBER_COLUMNS,
        "member_benefits": flatten.MEMBER_BENEFIT_COLUMNS, "swipes": flatten.SWIPE_COLUMNS,
        "claims": flatten.CLAIM_COLUMNS, "tickets": flatten.TICKET_COLUMNS,
        "bill_lines": flatten.BILL_LINE_COLUMNS, "reserves": flatten.RESERVE_COLUMNS,
        "ledger": flatten.LEDGER_COLUMNS,
    }
    for table, columns in pairs.items():
        assert list(PROCESSED_SCHEMAS[table].columns) == columns


def test_schema_builder_rejects_column_list_mismatch():
    with pytest.raises(ValueError, match="do not match flatten.py"):
        _schema("x", {"a": text()}, ["a", "b"])


def test_duplicate_primary_key_fails(tables):
    swipes = pd.concat([tables["swipes"], tables["swipes"].head(1)], ignore_index=True)
    assert_error(errors_for(swipes, "swipes"), "swipe_id", "unique")


def test_leaked_phi_column_fails(tables):
    members = tables["members"].assign(ssn="123-45-6789")
    assert_error(errors_for(members, "members"), None, "column_in_schema")


def test_missing_column_fails(tables):
    assert_error(errors_for(tables["claims"].drop(columns=["status"]), "claims"), None, "column_in_dataframe")


def test_naive_timestamps_fail(tables):
    swipes = tables["swipes"].assign(service_date=tables["swipes"]["service_date"].dt.tz_localize(None))
    assert_error(errors_for(swipes, "swipes"), "service_date", "utc_datetime")


def test_money_as_text_fails(tables):
    swipes = tables["swipes"].assign(amount_usd=tables["swipes"]["amount_usd"].astype(str))
    assert_error(errors_for(swipes, "swipes"), "amount_usd", "float_dollars")


def test_cents_stored_as_integers_fail(tables):
    ledger = tables["ledger"].assign(amount_usd=(tables["ledger"]["amount_usd"] * 100).astype(int))
    assert_error(errors_for(ledger, "ledger"), "amount_usd", "float_dollars")


def test_derived_enum_outside_list_fails(tables):
    members = tables["members"].copy()
    members.loc[0, "employee_status"] = "Retired"
    assert_error(errors_for(members, "members"), "employee_status", "isin")


def test_out_of_range_month_fails(tables):
    clients = tables["clients"].copy()
    clients.loc[0, "renewal_month"] = 13
    assert_error(errors_for(clients, "clients"), "renewal_month", "less_than_or_equal_to")


@pytest.mark.parametrize("table, column, value", [
    ("tickets", "ticket_number", "TICKET-1"),
    ("claims", "claim_number", "CLM-9"),
    ("members", "ssn_masked", "123-45-6789"),
    ("clients", "client_id", "not-an-object-id"),
])
def test_bad_id_formats_fail(tables, table, column, value):
    df = tables[table].copy()
    df.loc[0, column] = value
    assert_error(errors_for(df, table), column, "str_matches")


def test_duplicate_reserve_month_fails(tables):
    reserves = pd.concat([tables["reserves"], tables["reserves"].head(1)], ignore_index=True)
    errors = errors_for(reserves, "reserves")
    assert any("unique" in e["check"] for e in errors), errors


def test_duplicate_bill_lines_are_not_a_schema_error(tables):
    """Kept on purpose (decisions.md #2); anomalies.duplicate_bill_lines reports them instead."""
    bills = pd.concat([tables["bill_lines"], tables["bill_lines"].head(1)], ignore_index=True)
    assert validate_frame(bills, PROCESSED_SCHEMAS["bill_lines"], "bill_lines")["status"] == "passed"


def test_error_summary_counts_and_samples(tables):
    tickets = tables["tickets"].copy()
    tickets["ticket_number"] = ["BAD-1", "BAD-2"]
    error = next(e for e in errors_for(tickets, "tickets") if e["column"] == "ticket_number")
    assert error["count"] == 2
    assert error["failure_cases"] == ["BAD-1", "BAD-2"]
