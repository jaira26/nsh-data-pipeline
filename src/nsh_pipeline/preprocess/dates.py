"""Date normalization: parse NSH's mixed date formats into UTC timestamps.

Formats are declared per field, never guessed (data contract A4):
  "iso" - ISO-8601 strings from Mongo Date fields, e.g. "2025-02-03T00:00:00Z"
  "mdy" - M/D/YYYY strings, e.g. people Birthdate "4/12/1985" and card dates
  "ym"  - "YYYY-MM" billing months, e.g. bills.billingDate "2025-01" (-> first of month)
Date-only values are treated as midnight UTC so the calendar date never shifts.
"""
import pandas as pd

_FORMATS = {"iso": "ISO8601", "mdy": "%m/%d/%Y", "ym": "%Y-%m"}


def to_utc(series: pd.Series, kind: str) -> pd.Series:
    """Parse a Series of date strings of one `kind` into UTC timestamps; bad values become NaT."""
    if kind not in _FORMATS:
        raise ValueError(f"Unknown date format {kind!r}; expected one of {sorted(_FORMATS)}")
    return pd.to_datetime(series, format=_FORMATS[kind], errors="coerce", utc=True)


def parse_dates(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Return a copy of `df` with each column in `formats` parsed to UTC timestamps."""
    out = df.copy()
    for field, kind in formats.items():
        out[field] = to_utc(out[field], kind)
    return out