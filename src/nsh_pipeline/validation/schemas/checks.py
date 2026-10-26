"""Dtype-agnostic column builders for pandera schemas.

The same schemas must pass on pandas 3 (local, CI: text is the "str" dtype, timestamps are
datetime64[us, UTC]) and on the pandas 2.1 inside the Airflow image (text is object). So no
schema pins a string dtype or a datetime unit; checks test the kind of data instead.
"""
import pandas as pd
from pandas.api import types
from pandera.pandas import Check, Column


def is_text(series: pd.Series) -> bool:
    """True if every non-null value is a str (pandas 3 "str"/StringDtype or object of str)."""
    if isinstance(series.dtype, pd.StringDtype):
        return True
    if series.dtype != object:
        return False
    return bool(series.dropna().map(lambda value: isinstance(value, str)).all())


def is_utc_datetime(series: pd.Series) -> bool:
    """True for timezone-aware UTC timestamps of any unit (ns, us, ...)."""
    return isinstance(series.dtype, pd.DatetimeTZDtype) and str(series.dtype.tz) == "UTC"


def is_float(series: pd.Series) -> bool:
    return types.is_float_dtype(series.dtype)


def is_number(series: pd.Series) -> bool:
    return types.is_numeric_dtype(series.dtype) and not types.is_bool_dtype(series.dtype)


def is_whole_number(series: pd.Series) -> bool:
    """True for integer dtypes, or numbers that are all whole (e.g. cents read as float)."""
    if types.is_integer_dtype(series.dtype):
        return True
    return is_number(series) and bool((series.dropna() % 1 == 0).all())


def is_bool(series: pd.Series) -> bool:
    return types.is_bool_dtype(series.dtype)


TEXT = Check(is_text, name="text")
UTC_DATETIME = Check(is_utc_datetime, name="utc_datetime")
FLOAT_DOLLARS = Check(is_float, name="float_dollars")
NUMBER = Check(is_number, name="number")
WHOLE_NUMBER = Check(is_whole_number, name="whole_number")
BOOLEAN = Check(is_bool, name="boolean")


def text(nullable=True, unique=False, isin=None, pattern=None, required=True) -> Column:
    """A text column, optionally restricted to `isin` values or a full-match regex `pattern`."""
    checks = [TEXT]
    if isin is not None:
        checks.append(Check.isin(list(isin)))
    if pattern is not None:
        checks.append(Check.str_matches(f"^(?:{pattern})$"))
    return Column(None, checks, nullable=nullable, unique=unique, required=required)


def utc(nullable=True, required=True) -> Column:
    return Column(None, [UTC_DATETIME], nullable=nullable, required=required)


def money(nullable=True, ge=None, required=True) -> Column:
    """Float dollars. Nullable by default: missing money is an anomaly check, not a schema error."""
    checks = [FLOAT_DOLLARS] + ([Check.ge(ge)] if ge is not None else [])
    return Column(None, checks, nullable=nullable, required=required)


def number(nullable=False, ge=None, le=None, whole=False, required=True) -> Column:
    checks = [WHOLE_NUMBER if whole else NUMBER]
    if ge is not None:
        checks.append(Check.ge(ge))
    if le is not None:
        checks.append(Check.le(le))
    return Column(None, checks, nullable=nullable, required=required)


def flag() -> Column:
    return Column(None, [BOOLEAN], nullable=False)


def any_value(nullable=True, required=True) -> Column:
    """A column that only has to exist (raw documents: nested arrays and objects)."""
    return Column(None, nullable=nullable, required=required)
