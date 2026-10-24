import pandas as pd
import pytest

from nsh_pipeline.constants import EXCLUDED_ORG_IDS
from nsh_pipeline.generate.activity import TICKET_STATUSES
from nsh_pipeline.generate.run import build_collections, generate_data
from nsh_pipeline.preprocess.flatten import clean_bill_lines, clean_swipes
from nsh_pipeline.preprocess.run import run_preprocess

REF = pd.Timestamp("2025-06-15", tz="UTC")
BASE = {
    "seed": 42,
    "scale": "test",
    "reference_date": "2025-06-15",
    "scales": {"test": {"clients": 20, "max_members_per_client": 30}},
}
ANOMALIES = {
    "duplicate_bill_lines": 3, "cents_as_dollars": 3, "orphan_swipes": 3,
    "unparseable_money": 3, "future_service_dates": 2, "status_casing_drift": 3,
}


def with_injection(state: str) -> dict:
    return {**BASE, "inject_issues": {"enabled": True, **ANOMALIES,
                                      "denial_skew": {"state": state, "rate": 0.3}}}


@pytest.fixture(scope="module")
def clean():
    return build_collections(BASE)


@pytest.fixture(scope="module")
def skew_state(clean):
    collections, _ = clean
    return next(c["employer"]["state"] for c in collections["applications"]
                if c["_id"] not in EXCLUDED_ORG_IDS)


@pytest.fixture(scope="module")
def injected(skew_state):
    return build_collections(with_injection(skew_state))


def issues_by_name(injected):
    return {issue["issue"]: issue for issue in injected[1]}


def test_injection_is_off_by_default(clean):
    assert clean[1] == []


def test_every_issue_is_recorded_with_requested_count(injected):
    issues = issues_by_name(injected)
    assert set(issues) == set(ANOMALIES) | {"denial_skew"}
    for name, count in ANOMALIES.items():
        assert issues[name]["count"] == count == len(issues[name]["ids"])
    assert issues["denial_skew"]["count"] > 0


def test_injection_leaves_other_data_unchanged(clean, injected):
    for name in ("applications", "people", "claims"):
        assert clean[0][name] == injected[0][name]


def test_duplicate_bill_lines_create_duplicate_member_months(injected):
    lines = clean_bill_lines(injected[0]["bills"])
    keys = ["client_id", "billing_month", "member_id"]
    assert lines[lines.duplicated(keys, keep=False)].groupby(keys).ngroups == ANOMALIES["duplicate_bill_lines"]


def test_cents_as_dollars_are_extreme_outliers(injected):
    swipes = clean_swipes(injected[0]["wex_claims"]).set_index("swipe_id")
    median = swipes["amount_usd"].median()
    for swipe_id in issues_by_name(injected)["cents_as_dollars"]["ids"]:
        assert swipes.loc[swipe_id, "amount_usd"] > 50 * median


def test_orphans_unparseable_money_and_future_dates(injected):
    collections = injected[0]
    issues = issues_by_name(injected)
    people = {p["_id"] for p in collections["people"]}
    raw = {s["_id"]: s for s in collections["wex_claims"]}
    assert all(raw[i]["employeeId"] not in people for i in issues["orphan_swipes"]["ids"])
    swipes = clean_swipes(collections["wex_claims"]).set_index("swipe_id")
    assert swipes.loc[issues["unparseable_money"]["ids"], "paid_usd"].isna().all()
    assert (swipes.loc[issues["future_service_dates"]["ids"], "service_date"] > REF).all()


def test_ticket_status_casing_drift(injected):
    tickets = {t["ticketNumber"]: t for t in injected[0]["tickets"]}
    for number in issues_by_name(injected)["status_casing_drift"]["ids"]:
        status = tickets[number]["status"]
        assert status not in TICKET_STATUSES and status == status.lower()


def test_denial_skew_creates_a_detectable_slice(injected, skew_state):
    collections = injected[0]
    state_of = {c["_id"]: c["employer"]["state"] for c in collections["applications"]}
    swipes = clean_swipes(collections["wex_claims"])
    denied = swipes["status"].eq("Denied")
    in_state = swipes["client_id"].map(state_of).eq(skew_state)
    assert denied[in_state].mean() >= denied[~in_state].mean() + 0.15


def test_manifest_lists_issues_and_preprocessing_survives(skew_state, tmp_path):
    manifest = generate_data(with_injection(skew_state), tmp_path / "raw")
    assert {i["issue"] for i in manifest["injected_issues"]} == set(ANOMALIES) | {"denial_skew"}
    counts = run_preprocess(tmp_path / "raw", tmp_path / "processed", "2025-06-15")
    assert all(rows > 0 for rows in counts.values()), counts