import pytest

from nsh_pipeline.utils.keys import make_nswid


@pytest.mark.parametrize(
    "last_name, ssn, expected",
    [
        ("Richardson", "123-45-6789", "richard6789"),  # long name is cut to 7 chars
        ("Li", "987654321", "li4321"),                 # short name kept whole, SSN without dashes
        ("  SMITH ", "111-22-3333", "smith3333"),      # whitespace stripped, lowercased
    ],
)
def test_make_nswid_valid(last_name, ssn, expected):
    assert make_nswid(last_name, ssn) == expected


@pytest.mark.parametrize(
    "last_name, ssn",
    [(None, "123-45-6789"), ("", "123-45-6789"), ("Smith", None), ("Smith", "12"), ("Smith", 123456789)],
)
def test_make_nswid_missing_or_bad_input_returns_none(last_name, ssn):
    assert make_nswid(last_name, ssn) is None