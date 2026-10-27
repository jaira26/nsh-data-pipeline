import json

import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.orchestration.summary import anomaly_counts
from nsh_pipeline.validation import anomalies as a
from nsh_pipeline.validation.stats import compute_stats

AS_OF = "2025-06-15"
ACME = "a00000000000000000000001"
JANE = "b00000000000000000000001"


def test_clean_sample_passes_every_check(clean_tables, params, settings):
    stats = compute_stats(clean_tables, settings)
    results = a.run_checks(clean_tables, {}, params, settings, AS_OF, current_stats=stats, baseline_stats=stats)
    assert [r.check for r in results if not r.passed] == []
    assert [r.check for r in results] == [
        "duplicate_bill_lines", "money_unit_error", "orphan_keys", "excluded_org_leak", "unparseable_money",
        "active_client_count", "enum_violations", "future_dates", "negative_amounts", "term_before_hire",
        "distribution_drift",
    ]
    assert a.critical_failures(results) == []


# ---------------------------------------------------------------- critical
def test_duplicate_bill_lines(clean_tables):
    bills = clean_tables["bill_lines"]
    clean_tables["bill_lines"] = pd.concat([bills, bills.head(1)], ignore_index=True)
    result = a.duplicate_bill_lines(clean_tables)
    assert (result.passed, result.severity, result.count) == (False, "critical", 1)
    assert result.record_ids == [f"bill_lines:{JANE}|2025-01-01"]
    assert "billed more than once" in result.message


def test_missing_table_skips_the_check():
    result = a.duplicate_bill_lines({})
    assert (result.status, result.passed, result.count) == ("skipped", True, 0)
    assert a.term_before_hire({}).status == "skipped"
    assert a.active_client_count({}, {}, {}).status == "skipped"


def swipes_with_amounts(amounts):
    return pd.DataFrame({"swipe_id": [f"WEX-{i:06d}" for i in range(len(amounts))],
                         "amount_usd": amounts, "paid_usd": amounts})


def test_money_unit_error_flags_cents_read_as_dollars(settings):
    amounts = [50.0 + i for i in range(30)] + [7000.0]  # median 65, last value ~108x
    result = a.money_unit_errors({"swipes": swipes_with_amounts(amounts)}, settings)
    assert (result.passed, result.severity) == (False, "critical")
    assert result.record_ids == ["swipes:WEX-000030"]  # flagged in two columns, counted once
    assert set(result.details["swipes"]) == {"amount_usd", "paid_usd"}


def test_money_unit_error_ignores_normal_tail_and_small_columns(settings):
    amounts = [50.0 + i for i in range(30)] + [5000.0]  # ~70x, like clean data's $5,000 swipe
    assert a.money_unit_errors({"swipes": swipes_with_amounts(amounts)}, settings).passed
    tiny = {**settings, "money_unit": {"max_median_multiple": 100, "min_values": 20}}
    assert a.money_unit_errors({"swipes": swipes_with_amounts([1.0, 1000.0])}, tiny).passed


def test_orphan_and_null_keys(clean_tables):
    swipes, tickets = clean_tables["swipes"], clean_tables["tickets"]
    swipes.loc[0, "member_id"] = "ff0000000000000000000001"
    swipes.loc[0, "client_id"] = "ff0000000000000000000009"  # same row, second bad key
    tickets.loc[1, "client_id"] = None
    result = a.orphan_keys(clean_tables)
    assert (result.passed, result.count) == (False, 2)
    assert result.record_ids == ["swipes:WEX-000001", "tickets:NSET-00002"]
    assert result.details["swipes.member_id -> members"] == {"null": 0, "orphan": 1}
    assert result.details["tickets.client_id -> clients"] == {"null": 1, "orphan": 0}


def test_excluded_org_leak(clean_tables):
    leaked = clean_tables["swipes"].head(1).assign(swipe_id="WEX-999999", client_id=NS_DEMO_ID)
    clean_tables["swipes"] = pd.concat([clean_tables["swipes"], leaked], ignore_index=True)
    clients = clean_tables["clients"]
    clients.loc[clients["client_id"] == NSAI_ID, "is_excluded_org"] = False
    features = {"book_of_business": pd.DataFrame({"client_id": [ACME, NSAI_ID]})}
    result = a.excluded_org_leak(clean_tables, features)
    assert (result.passed, result.severity) == (False, "critical")
    assert set(result.record_ids) == {f"clients:{NSAI_ID}", "swipes:WEX-999999", f"book_of_business:{NSAI_ID}"}


def test_unparseable_money_on_processed_and_raw(tables, sample_raw):
    result = a.unparseable_money(tables, sample_raw)
    assert (result.passed, result.severity) == (False, "critical")
    assert result.record_ids == ["swipes:WEX-000005"]
    assert result.details["swipes"] == {"amount_usd": 1, "paid_usd": 1}
    assert result.details["raw wex_claims values"] == {"'N/A'": 2}


def test_unparseable_money_without_raw(tables):
    result = a.unparseable_money(tables)
    assert result.record_ids == ["swipes:WEX-000005"] and "raw wex_claims values" not in result.details


# ---------------------------------------------------------------- warnings
def test_active_client_count(clean_tables, params, settings):
    assert a.active_client_count(clean_tables, params, settings).passed
    clean_tables["clients"]["status"] = "Terminated"
    result = a.active_client_count(clean_tables, params, settings)
    assert (result.passed, result.severity, result.count) == (False, "warning", 1)
    assert result.details["active_clients"] == 0


def test_enum_violations_report_casing_drift(clean_tables, sample_raw):
    tickets = clean_tables["tickets"]
    tickets.loc[0, "status"] = "closed"
    tickets.loc[1, "status"] = "Escalated"
    raw = {"applications": [dict(doc) for doc in sample_raw["applications"]]}
    raw["applications"][0]["employer"] = {**raw["applications"][0]["employer"], "status": "active"}
    result = a.enum_violations(clean_tables, raw)
    assert (result.passed, result.severity) == (False, "warning")
    assert result.record_ids == ["tickets:NSET-00001", "tickets:NSET-00002", f"raw.applications:{ACME}"]
    assert result.details["tickets.status"] == {"values": {"Escalated": 1, "closed": 1}, "casing_drift": ["closed"]}
    assert result.details["raw.applications.status"]["casing_drift"] == ["active"]


def test_future_dates_compare_by_day(clean_tables):
    swipes = clean_tables["swipes"]
    swipes.loc[0, "service_date"] = pd.Timestamp("2025-06-16", tz="UTC")
    swipes.loc[1, "service_date"] = pd.Timestamp("2025-06-15 23:00", tz="UTC")  # same day: fine
    result = a.future_dates(clean_tables, AS_OF)
    assert (result.passed, result.record_ids) == (False, ["swipes:WEX-000001"])
    assert result.details["swipes.service_date"]["count"] == 1
    assert a.future_dates(clean_tables, pd.Timestamp("2025-07-01", tz="UTC")).passed


def test_negative_amounts_but_not_ledger_refunds(clean_tables):
    clean_tables["ledger"].loc[0, "amount_usd"] = -125.0  # refunds are allowed
    assert a.negative_amounts(clean_tables).passed
    clean_tables["swipes"].loc[2, "paid_usd"] = -5.0
    result = a.negative_amounts(clean_tables)
    assert (result.passed, result.record_ids) == (False, ["swipes:WEX-000003"])


def test_term_before_hire(clean_tables):
    members = clean_tables["members"]
    members.loc[0, "term_date"] = members.loc[0, "hire_date"] - pd.Timedelta(days=1)
    result = a.term_before_hire(clean_tables)
    assert (result.passed, result.record_ids) == (False, [f"members:{JANE}"])


def test_distribution_drift(clean_tables, settings):
    stats = compute_stats(clean_tables, settings)
    assert a.distribution_drift(stats, {}, settings).status == "skipped"
    assert a.distribution_drift(stats, stats, settings).passed
    changed = compute_stats({**clean_tables, "swipes": clean_tables["swipes"].assign(status="Denied")}, settings)
    result = a.distribution_drift(changed, stats, settings)
    assert (result.passed, result.severity) == (False, "warning")
    assert "swipes.status:share of 'Denied'" in result.record_ids


# ---------------------------------------------------------------- record ids and reports
def test_record_ids_join_keys_and_fall_back_to_row_numbers():
    df = pd.DataFrame({"member_id": ["m1", None],
                       "billing_month": pd.to_datetime(["2025-01-01", None], utc=True)})
    assert a.record_ids(df, "bill_lines").tolist() == ["bill_lines:m1|2025-01-01", "bill_lines:|"]
    assert a.record_ids(pd.DataFrame({"x": [1, 2]}), "other").tolist() == ["other:row0", "other:row1"]


@pytest.fixture
def results(tables, params, settings):
    return a.run_checks(tables, {}, params, settings, AS_OF)


def test_reports_are_a_list_that_run_summary_can_count(results, tmp_path):
    paths = a.write_reports(results, tmp_path, {"mode": "strict", "critical_failed": 1}, sample=1)
    records = json.loads(paths["json"].read_text())
    assert isinstance(records, list) and len(records) == len(results)
    assert set(records[0]) == {"check", "severity", "status", "passed", "count", "sample_record_ids",
                               "message", "details"}
    assert anomaly_counts(paths["json"]) == {"critical": 1, "warning": 0}  # M4's run summary reader
    markdown = paths["markdown"].read_text()
    assert "| `unparseable_money` | critical | **FAIL** | 1 |" in markdown
    assert "`swipes:WEX-000005`" in markdown
    assert "| `distribution_drift` | warning | skipped | 0 |" in markdown


def test_to_record_samples_ids():
    result = a.CheckResult("c", "warning", False, 3, "m", ["x:1", "x:2", "x:3"])
    assert result.to_record(sample=2)["sample_record_ids"] == ["x:1", "x:2"]


def test_checks_handle_nan_only_money_columns(settings):
    df = pd.DataFrame({"swipe_id": ["WEX-1"], "amount_usd": [np.nan]})
    assert a.money_unit_errors({"swipes": df}, settings).passed
