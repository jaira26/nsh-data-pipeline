import pytest

from nsh_pipeline.preprocess.people_current import current_entry, current_value


def test_last_entry_is_current():
    names = [{"LastName": "Old", "Deleted": False}, {"LastName": "New", "Deleted": False}]
    assert current_entry(names)["LastName"] == "New"


def test_deleted_last_entry_is_skipped():
    names = [{"LastName": "Keep", "Deleted": False}, {"LastName": "Gone", "Deleted": True}]
    assert current_entry(names)["LastName"] == "Keep"


def test_inactive_entry_is_skipped():
    jobs = [{"Title": "Analyst", "Active": True}, {"Title": "Intern", "Active": False}]
    assert current_entry(jobs)["Title"] == "Analyst"


def test_entry_without_flags_counts_as_current():
    assert current_entry([{"LastName": "Plain"}])["LastName"] == "Plain"


@pytest.mark.parametrize(
    "versions",
    [[], None, "not a list", [{"Deleted": True}, {"Deleted": True}], ["junk", 5]],
)
def test_no_current_entry_returns_none(versions):
    assert current_entry(versions) is None


def test_current_value_reads_key_and_falls_back_to_default():
    names = [{"LastName": "Smith", "Deleted": False}]
    assert current_value(names, "LastName") == "Smith"
    assert current_value(names, "MiddleName", default="") == ""
    assert current_value([], "LastName", default="n/a") == "n/a"