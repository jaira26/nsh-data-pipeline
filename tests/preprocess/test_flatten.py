import pandas as pd
import pytest

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.preprocess.flatten import (
    CLIENT_COLUMNS, MEMBER_COLUMNS, clean_clients, clean_members,
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