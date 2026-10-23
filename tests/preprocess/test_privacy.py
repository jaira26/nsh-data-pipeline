import pandas as pd
import pytest

from nsh_pipeline.preprocess.people_current import current_value
from nsh_pipeline.preprocess.privacy import age_bands, assert_no_phi, drop_phi, mask_ssn

AS_OF = pd.Timestamp("2025-06-15", tz="UTC")


@pytest.mark.parametrize(
    "ssn, expected", [("123-45-6789", "***-**-6789"), ("987654321", "***-**-4321")]
)
def test_mask_ssn_keeps_only_last_four(ssn, expected):
    assert mask_ssn(ssn) == expected


@pytest.mark.parametrize("ssn", [None, "", "12", 123456789])
def test_mask_ssn_unusable_input_returns_none(ssn):
    assert mask_ssn(ssn) is None


def test_age_bands_respect_birthday_boundaries():
    dobs = pd.Series(pd.to_datetime(["2007-06-15", "2007-06-16", "1990-01-01", "1960-06-15"], utc=True))
    # 18 on the day of the birthday, 17 the day before it; 35; exactly 65
    assert age_bands(dobs, AS_OF).tolist() == ["18-25", "<18", "35-44", "65+"]


def test_age_bands_missing_and_future_dates_are_nan():
    dobs = pd.Series(pd.to_datetime([None, "2030-01-01"], utc=True))
    assert age_bands(dobs, AS_OF).isna().all()


def test_drop_phi_removes_only_phi_columns():
    df = pd.DataFrame({"Ssn": ["123-45-6789"], "Birthdate": ["4/12/1985"], "nswid": ["richard6789"]})
    assert list(drop_phi(df).columns) == ["nswid"]
    assert "Ssn" in df.columns  # original DataFrame is not modified


def test_drop_phi_without_phi_columns_is_a_no_op():
    df = pd.DataFrame({"nswid": ["li4321"]})
    assert drop_phi(df).equals(df)


def test_assert_no_phi_raises_on_leak():
    with pytest.raises(ValueError, match="Ssn"):
        assert_no_phi(pd.DataFrame({"Ssn": ["x"], "nswid": ["y"]}), "members")


def test_assert_no_phi_passes_clean_table():
    assert_no_phi(pd.DataFrame({"nswid": ["y"], "ssn_masked": ["***-**-1234"]}), "members")


def test_masks_every_sample_member_ssn(sample_raw):
    for person in sample_raw["people"]:
        ssn = current_value(person["PersonalData"], "Ssn")
        assert mask_ssn(ssn) == "***-**-" + ssn[-4:]