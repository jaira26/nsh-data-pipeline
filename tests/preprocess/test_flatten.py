import pandas as pd
import pytest

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.preprocess.flatten import (
    CLAIM_COLUMNS, CLIENT_COLUMNS, MEMBER_COLUMNS, SWIPE_COLUMNS, TICKET_COLUMNS,
    clean_claims, clean_clients, clean_members, clean_swipes, clean_tickets,
)

AS_OF = pd.Timestamp("2025-06-15", tz="UTC")
ACME = "a00000000000000000000001"


@pytest.fixture
def clients(sample_raw):
    return clean_clients(sample_raw["applications"])


@pytest.fixture
def members(sample_raw):
    return clean_members(sample_raw["people"], AS_OF).set_index("last_name")


# ---------------------------------------------------------------- clients
def test_clients_have_agreed_columns(clients):
    assert list(clients.columns) == CLIENT_COLUMNS


def test_clients_keep_all_orgs_and_flag_excluded(clients):
    assert len(clients) == 5
    assert set(clients.loc[clients["is_excluded_org"], "client_id"]) == {NS_DEMO_ID, NSAI_ID}


def test_client_fields_are_flattened_and_typed(clients):
    acme = clients.set_index("client_id").loc[ACME]
    assert acme["name"] == "Acme Manufacturing"
    assert acme["wex_id"] == "ACMEWEX01"
    assert acme["group_size"] == 45
    assert acme["group_size_bucket"] == "25-49"
    assert acme["status"] == "Active"
    assert acme["client_since"] == pd.Timestamp("2021-07-01", tz="UTC")


def test_client_status_casing_is_normalized():
    out = clean_clients([{"_id": "x", "status": "active", "employer": {"name": "X"}}])
    assert out.loc[0, "status"] == "Active"


def test_client_without_employer_block_does_not_crash():
    out = clean_clients([{"_id": "x"}])
    assert out.loc[0, "client_id"] == "x"
    assert pd.isna(out.loc[0, "name"])


@pytest.mark.parametrize(
    "size, bucket", [(1, "1-9"), (9, "1-9"), (10, "10-24"), (120, "100-249"), (600, "250+")]
)
def test_group_size_buckets(size, bucket):
    out = clean_clients([{"_id": "x", "employer": {"groupSize": size}}])
    assert out.loc[0, "group_size_bucket"] == bucket


# ---------------------------------------------------------------- members
def test_members_have_agreed_columns_and_no_phi(sample_raw):
    out = clean_members(sample_raw["people"], AS_OF)
    assert list(out.columns) == MEMBER_COLUMNS
    assert "ssn" not in out.columns and "birthdate" not in out.columns


def test_excluded_org_members_are_dropped(sample_raw):
    out = clean_members(sample_raw["people"], AS_OF)
    assert len(out) == 4
    assert not out["client_id"].isin([NS_DEMO_ID, NSAI_ID]).any()


def test_current_name_and_derived_fields(members):
    jane = members.loc["Richardson"]
    assert "Miller" not in members.index  # her old, deleted last name
    assert jane["first_name"] == "Jane"
    assert jane["nswid"] == "richard6789"
    assert jane["ssn_masked"] == "***-**-6789"
    assert jane["age_band"] == "35-44"
    assert jane["coverage_tier"] == "ES"


def test_employee_status_buckets(members):
    assert members["employee_status"].to_dict() == {
        "Richardson": "Enrolled",
        "Li": "Enrolled",
        "Garcia": "COBRA Enrolled",
        "O'Brien": "Other",  # terminated 2025-03-15
    }


def test_member_without_benefits_is_not_enrolled_and_gets_computed_nswid():
    person = {
        "_id": "m1", "Application": ACME,
        "Names": [{"FirstName": "Ann", "LastName": "Bell"}],
        "PersonalData": [{"Ssn": "111-22-3333", "Birthdate": "1/1/1990"}],
        "Employment": [{"EmployeeStatus": "active", "Active": True}],
    }
    out = clean_members([person], AS_OF)
    assert out.loc[0, "employee_status"] == "Not Enrolled"
    assert out.loc[0, "nswid"] == "bell3333"  # computed because _base64Id is missing


def test_vip_flag(members):
    assert members["is_vip"].to_dict() == {
        "Richardson": False, "Li": True, "Garcia": False, "O'Brien": False,
    }


def test_custom_excluded_ids_drop_extra_clients(sample_raw):
    out = clean_members(sample_raw["people"], AS_OF, excluded_ids={NS_DEMO_ID, NSAI_ID, ACME})
    assert out["last_name"].tolist() == ["O'Brien"]


def test_naive_as_of_is_treated_as_utc(sample_raw):
    out = clean_members(sample_raw["people"], pd.Timestamp("2025-06-15"))
    assert len(out) == 4

# ---------------------------------------------------------------- swipes
@pytest.fixture
def swipes(sample_raw):
    return clean_swipes(sample_raw["wex_claims"]).set_index("swipe_id")


def test_swipes_have_agreed_columns_and_drop_excluded(sample_raw):
    out = clean_swipes(sample_raw["wex_claims"])
    assert list(out.columns) == SWIPE_COLUMNS
    assert len(out) == 5
    assert NS_DEMO_ID not in set(out["client_id"])


def test_swipe_money_strings_become_dollars(swipes):
    assert swipes.loc["WEX-000002", "amount_usd"] == pytest.approx(1250.75)
    assert swipes.loc["WEX-000003", "denied_usd"] == pytest.approx(80.0)
    assert pd.isna(swipes.loc["WEX-000005", "paid_usd"])  # "N/A" in the raw data


def test_swipe_flags_and_ticket_link(swipes):
    assert swipes.loc["WEX-000001", "is_debit_card"]
    assert not swipes.loc["WEX-000005", "is_debit_card"]  # manual claim, not a card swipe
    assert swipes["needs_substantiation"].sum() == 1
    assert swipes.loc["WEX-000001", "needs_substantiation"]
    assert swipes.loc["WEX-000001", "ticket_number"] == "NSET-00001"


def test_swipe_service_date_is_utc(swipes):
    assert swipes.loc["WEX-000001", "service_date"] == pd.Timestamp("2025-02-03", tz="UTC")


def test_swipe_with_missing_fields_does_not_crash():
    out = clean_swipes([{"_id": "WEX-1", "clientId": "c1"}])
    assert pd.isna(out.loc[0, "amount_usd"])
    assert not out.loc[0, "is_debit_card"]
    assert not out.loc[0, "needs_substantiation"]


def test_empty_input_gives_empty_table_with_columns():
    out = clean_swipes([])
    assert out.empty
    assert list(out.columns) == SWIPE_COLUMNS


# ---------------------------------------------------------------- claims
def test_claims_columns_and_values(sample_raw):
    out = clean_claims(sample_raw["claims"])
    assert list(out.columns) == CLAIM_COLUMNS
    claims = out.set_index("claim_number")
    assert claims.loc["CLM-100002", "status"] == "Denied"
    assert claims.loc["CLM-100002", "denied_usd"] == pytest.approx(950.0)
    assert claims.loc["CLM-100002", "approved_usd"] == 0
    assert claims.loc["CLM-100001", "created_at"] == pd.Timestamp("2025-01-22", tz="UTC")
    assert pd.isna(claims.loc["CLM-100001", "closed_at"])


def test_deleted_and_excluded_claims_are_dropped():
    raw = [
        {"claimNumber": "CLM-1", "clientId": "c1"},
        {"claimNumber": "CLM-2", "clientId": "c1", "deleted": True},
        {"claimNumber": "CLM-3", "clientId": NS_DEMO_ID},
    ]
    assert clean_claims(raw)["claim_number"].tolist() == ["CLM-1"]


# ---------------------------------------------------------------- tickets
def test_tickets_columns_flags_and_sla(sample_raw):
    out = clean_tickets(sample_raw["tickets"], AS_OF)
    assert list(out.columns) == TICKET_COLUMNS
    tickets = out.set_index("ticket_number")
    # NSET-00001 is still open months later; NSET-00002 closed after 23.5 hours
    assert tickets["sla_breached"].to_dict() == {"NSET-00001": True, "NSET-00002": False}
    assert tickets["is_vip"].to_dict() == {"NSET-00001": False, "NSET-00002": True}
    assert tickets.loc["NSET-00002", "closed_at"] == pd.Timestamp("2025-02-16 09:30", tz="UTC")


def test_open_ticket_within_sla_is_not_breached():
    raw = [{"ticketNumber": "NSET-9", "clientId": "c1", "createdAt": "2025-06-14T00:00:00Z"}]
    assert not clean_tickets(raw, AS_OF).loc[0, "sla_breached"]  # open for 24h of 40h


def test_excluded_org_tickets_are_dropped():
    raw = [{"ticketNumber": "NSET-9", "clientId": NSAI_ID, "createdAt": "2025-06-14T00:00:00Z"}]
    assert clean_tickets(raw, AS_OF).empty