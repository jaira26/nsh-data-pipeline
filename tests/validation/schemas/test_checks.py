import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.validation.schemas.checks import (
    is_bool, is_float, is_number, is_text, is_utc_datetime, is_whole_number,
)


@pytest.mark.parametrize("series", [
    pd.Series(["a", None], dtype="str"),              # pandas 3 default text dtype
    pd.Series(["a", None], dtype="string"),           # explicit StringDtype
    pd.Series(["a", None], dtype=object),             # pandas 2 text
    pd.Series([None, None], dtype=object),            # all missing
])
def test_is_text_accepts_every_text_representation(series):
    assert is_text(series)


@pytest.mark.parametrize("series", [
    pd.Series([1, 2]),
    pd.Series(["a", 1], dtype=object),
    pd.Series([1.5, np.nan]),
])
def test_is_text_rejects_non_text(series):
    assert not is_text(series)


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_is_utc_datetime_accepts_any_unit(unit):
    series = pd.Series(pd.to_datetime(["2025-01-01"], utc=True)).astype(f"datetime64[{unit}, UTC]")
    assert is_utc_datetime(series)


@pytest.mark.parametrize("series", [
    pd.Series(pd.to_datetime(["2025-01-01"])),                                        # naive
    pd.Series(pd.to_datetime(["2025-01-01"])).dt.tz_localize("America/Los_Angeles"),  # not UTC
    pd.Series(["2025-01-01"]),                                                        # text
])
def test_is_utc_datetime_rejects_naive_other_zones_and_text(series):
    assert not is_utc_datetime(series)


def test_number_checks():
    assert is_whole_number(pd.Series([12500, 300]))
    assert is_whole_number(pd.Series([12500.0, np.nan]))
    assert not is_whole_number(pd.Series([125.5]))
    assert not is_whole_number(pd.Series(["12500"]))
    assert is_float(pd.Series([1.0])) and not is_float(pd.Series([1]))
    assert is_number(pd.Series([1])) and not is_number(pd.Series([True]))
    assert is_bool(pd.Series([True, False])) and not is_bool(pd.Series([1, 0]))
