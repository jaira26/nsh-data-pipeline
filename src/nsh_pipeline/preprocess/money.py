"""Money normalization: convert the three NSH money storage forms to float dollars.

NSH stores money as integer cents (mongoose-currency fields), as numeric strings
(e.g. bills.recurring[] lines) or as plain dollar numbers. See data contract A4.
"""
import math
import re

import numpy as np
import pandas as pd

_MONEY_NOISE = re.compile(r"[,$\s]")


def parse_money_str(value) -> float:
    """Parse a money value stored as a string (e.g. "1,234.50") into float dollars.

    Returns NaN for None, empty, non-numeric or infinite values instead of raising.
    """
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else np.nan
    text = _MONEY_NOISE.sub("", str(value))
    if not text:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return number if math.isfinite(number) else np.nan


def cents_to_usd(series: pd.Series) -> pd.Series:
    """Convert integer cents (mongoose-currency fields) to float dollars."""
    return pd.to_numeric(series, errors="coerce") / 100


def normalize_money(df: pd.DataFrame, units: dict[str, str]) -> pd.DataFrame:
    """Add a `<field>_usd` column for each field in `units` ("cents" | "string" | "dollars")."""
    out = df.copy()
    for field, unit in units.items():
        if unit == "cents":
            out[f"{field}_usd"] = cents_to_usd(out[field])
        elif unit == "string":
            out[f"{field}_usd"] = out[field].map(parse_money_str).astype(float)
        elif unit == "dollars":
            out[f"{field}_usd"] = pd.to_numeric(out[field], errors="coerce")
        else:
            raise ValueError(f"Unknown money unit {unit!r} for field {field!r}")
    return out