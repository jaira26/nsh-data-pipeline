import math

import pandas as pd
import pytest

from nsh_pipeline.preprocess.money import cents_to_usd, normalize_money, parse_money_str


@pytest.mark.parametrize(
    "raw, expected",
    [("1,234.50", 1234.5), ("$99", 99.0), (" 12.3 ", 12.3), ("-5.25", -5.25), (42, 42.0)],
)
def test_parse_money_str_valid(raw, expected):
    assert parse_money_str(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "abc", "NaN", "inf", float("nan"), True])
def test_parse_money_str_bad_values_become_nan(raw):
    assert math.isnan(parse_money_str(raw))


def test_cents_to_usd_divides_by_100():
    result = cents_to_usd(pd.Series([12345, 0, None]))
    assert result.iloc[0] == pytest.approx(123.45)
    assert result.iloc[1] == 0
    assert math.isnan(result.iloc[2])


def test_normalize_money_handles_each_unit():
    df = pd.DataFrame({"a": [500], "b": ["5.00"], "c": [5.0]})
    out = normalize_money(df, {"a": "cents", "b": "string", "c": "dollars"})
    assert out[["a_usd", "b_usd", "c_usd"]].iloc[0].tolist() == [5.0, 5.0, 5.0]


def test_normalize_money_rejects_unknown_unit():
    with pytest.raises(ValueError):
        normalize_money(pd.DataFrame({"a": [1]}), {"a": "euros"})