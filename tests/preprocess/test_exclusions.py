import pandas as pd

from nsh_pipeline.constants import NS_DEMO_ID, NSAI_ID
from nsh_pipeline.preprocess.exclusions import drop_excluded, flag_excluded_orgs

REAL_CLIENT = "a00000000000000000000001"


def test_flag_by_id():
    clients = pd.DataFrame({"_id": [REAL_CLIENT, NS_DEMO_ID, NSAI_ID]})
    assert flag_excluded_orgs(clients)["is_excluded_org"].tolist() == [False, True, True]


def test_flag_by_alias_is_case_insensitive():
    clients = pd.DataFrame({"_id": ["x1", "x2", "x3"], "dnnPortalAlias": ["acme", "Demo", "NSAIS"]})
    assert flag_excluded_orgs(clients)["is_excluded_org"].tolist() == [False, True, True]


def test_flag_works_without_alias_column():
    clients = pd.DataFrame({"_id": [REAL_CLIENT]})
    assert flag_excluded_orgs(clients)["is_excluded_org"].tolist() == [False]


def test_drop_excluded_keeps_real_clients_and_rows_without_client():
    df = pd.DataFrame({"clientId": [REAL_CLIENT, NS_DEMO_ID, NSAI_ID, None], "amount": [1, 2, 3, 4]})
    out = drop_excluded(df, "clientId")
    assert out["amount"].tolist() == [1, 4]
    assert list(out.index) == [0, 1]


def test_drop_excluded_accepts_custom_ids():
    df = pd.DataFrame({"clientId": ["keep", "alias-only-demo"]})
    assert drop_excluded(df, "clientId", excluded_ids={"alias-only-demo"})["clientId"].tolist() == ["keep"]


def test_drop_excluded_does_not_modify_original():
    df = pd.DataFrame({"clientId": [NS_DEMO_ID]})
    drop_excluded(df, "clientId")
    assert len(df) == 1


def test_sample_has_exactly_two_excluded_orgs(sample_raw):
    clients = flag_excluded_orgs(pd.DataFrame(sample_raw["applications"]))
    assert clients["is_excluded_org"].sum() == 2


def test_sample_people_and_swipes_lose_excluded_rows(sample_raw):
    people = drop_excluded(pd.DataFrame(sample_raw["people"]), "Application")
    swipes = drop_excluded(pd.DataFrame(sample_raw["wex_claims"]), "clientId")
    assert len(people) == 4   # 6 people minus the demo user and the NSAI staff member
    assert len(swipes) == 5   # 6 swipes minus the demo swipe