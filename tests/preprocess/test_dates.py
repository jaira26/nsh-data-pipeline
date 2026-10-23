import pandas as pd
import pytest

from nsh_pipeline.preprocess.dates import parse_dates, to_utc


def ts(text):
    return pd.Timestamp(text, tz="UTC")


def test_mdy_without_zero_padding():
    result = to_utc(pd.Series(["4/12/1985", "11/3/1990"]), "mdy")
    assert result.tolist() == [ts("1985-04-12"), ts("1990-11-03")]


def test_mdy_is_month_first_not_day_first():
    assert to_utc(pd.Series(["3/7/2025"]), "mdy").iloc[0] == ts("2025-03-07")


def test_ym_becomes_first_of_month():
    assert to_utc(pd.Series(["2025-01"]), "ym").iloc[0] == ts("2025-01-01")


def test_iso_with_z_and_offset_are_converted_to_utc():
    result = to_utc(pd.Series(["2025-02-03T00:00:00Z", "2025-02-03T10:00:00-08:00"]), "iso")
    assert result.tolist() == [ts("2025-02-03 00:00"), ts("2025-02-03 18:00")]


@pytest.mark.parametrize(
    "kind, bad",
    [("mdy", "13/45/2025"), ("mdy", "2025-01-01"), ("ym", "January"), ("iso", "not a date")],
)
def test_bad_values_become_nat(kind, bad):
    assert pd.isna(to_utc(pd.Series([bad]), kind).iloc[0])


def test_missing_values_become_nat():
    assert to_utc(pd.Series([None, "4/12/1985"]), "mdy").isna().tolist() == [True, False]


def test_result_is_timezone_aware_utc():
    assert str(to_utc(pd.Series(["2025-01"]), "ym").dt.tz) == "UTC"


def test_parse_dates_converts_listed_columns_only():
    df = pd.DataFrame({"dob": ["4/12/1985"], "month": ["2025-01"], "name": ["Jane"]})
    out = parse_dates(df, {"dob": "mdy", "month": "ym"})
    assert out.loc[0, "dob"] == ts("1985-04-12")
    assert out.loc[0, "month"] == ts("2025-01-01")
    assert out.loc[0, "name"] == "Jane"
    assert df.loc[0, "dob"] == "4/12/1985"  # original DataFrame is not modified


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        to_utc(pd.Series(["x"]), "dmy")


def test_parses_sample_card_dates(sample_raw):
    cards = pd.DataFrame(sample_raw["debit_card_status"])
    out = parse_dates(cards, {"valid_start_dt": "mdy", "valid_end_dt": "mdy"})
    assert out["valid_start_dt"].notna().all()
    assert out["valid_end_dt"].notna().all()