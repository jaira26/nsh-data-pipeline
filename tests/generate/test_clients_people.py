from datetime import date

import pandas as pd
import pytest

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.generate.clients import STATUSES, make_clients
from nsh_pipeline.generate.common import make_generators
from nsh_pipeline.generate.people import make_people
from nsh_pipeline.preprocess.flatten import clean_clients, clean_members
from nsh_pipeline.preprocess.people_current import current_value
from nsh_pipeline.utils.keys import make_nswid

REF = date(2025, 6, 15)
EMPLOYEE_STATUSES = {"Enrolled", "Not Enrolled", "COBRA Enrolled", "Other"}


def generate(seed=42, n_clients=20, max_members=30):
    rng, fake = make_generators(seed)
    clients = make_clients(rng, fake, n_clients, REF)
    people = make_people(rng, fake, clients, REF, max_members)
    return clients, people


@pytest.fixture(scope="module")
def data():
    return generate()


def test_same_seed_gives_identical_data():
    assert generate(seed=7) == generate(seed=7)


def test_different_seed_gives_different_data():
    assert generate(seed=7) != generate(seed=8)


def test_clients_include_special_orgs_with_unique_ids(data):
    clients, _ = data
    ids = [c["_id"] for c in clients]
    assert len(clients) == 22  # 20 + NS Demo + NSAI
    assert NS_DEMO_ID in ids and NSAI_ID in ids
    assert len(set(ids)) == len(ids)


def test_client_names_and_wex_ids_are_unique(data):
    clients, _ = data
    names = [c["employer"]["name"] for c in clients]
    wex_ids = [c["employer"]["claimReferenceId"] for c in clients]
    assert len(set(names)) == len(names)
    assert len(set(wex_ids)) == len(wex_ids)


def test_client_fields_follow_contract(data):
    clients, _ = data
    for client in clients:
        employer = client["employer"]
        assert employer["status"] in STATUSES
        assert 2 <= employer["groupSize"] <= 1500
        assert 1 <= employer["renewalMonth"] <= 12
        assert any(p["category"] == "nsw" and p["contracted"] for p in client["benefits"])


def test_people_belong_to_clients_and_respect_cap(data):
    clients, people = data
    assert {p["Application"] for p in people} <= {c["_id"] for c in clients}
    assert pd.Series([p["Application"] for p in people]).value_counts().max() <= 30


def test_nswid_matches_current_name_and_ssn(data):
    _, people = data
    for p in people:
        last = current_value(p["Names"], "LastName")
        ssn = current_value(p["PersonalData"], "Ssn")
        assert p["_base64Id"] == make_nswid(last, ssn)


def test_people_include_contract_quirks(data):
    _, people = data
    assert any(len(p["Names"]) > 1 for p in people)       # name history
    assert any(len(p["Employment"]) > 1 for p in people)  # job history
    assert any(not p["Benefits"] for p in people)         # waived coverage
    assert all("/" in p["PersonalData"][0]["Birthdate"] for p in people)  # M/D/YYYY strings


def test_generated_data_runs_through_preprocessing(data):
    clients, people = data
    cleaned = clean_clients(clients)
    members = clean_members(people, pd.Timestamp("2025-06-15", tz="UTC"))
    assert cleaned["is_excluded_org"].sum() == 2
    assert len(members) > 0
    assert members["age_band"].notna().all()
    assert set(members["employee_status"]) <= EMPLOYEE_STATUSES