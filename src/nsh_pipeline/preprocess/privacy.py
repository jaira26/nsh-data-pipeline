"""PHI minimization: mask or coarsen direct identifiers, then drop the raw values.

Even though our data is synthetic, the pipeline treats SSN, birthdate and addresses as
PHI (scoping plan 9.2): processed tables keep only nswid, a masked SSN and an age band.
"""
import re

import pandas as pd

PHI_COLUMNS = frozenset({
    "Ssn", "ssn", "unmaskedSsn", "employeeSsn",
    "Birthdate", "birthdate", "dob",
    "Addresses", "address",
})

AGE_BINS = [-1, 17, 25, 34, 44, 54, 64, 150]
AGE_LABELS = ["<18", "18-25", "26-34", "35-44", "45-54", "55-64", "65+"]


def mask_ssn(ssn):
    """Return an SSN masked to its last 4 digits (***-**-1234), or None if unusable."""
    if not isinstance(ssn, str):
        return None
    digits = re.sub(r"\D", "", ssn)
    if len(digits) < 4:
        return None
    return f"***-**-{digits[-4:]}"


def age_bands(birthdates: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Convert UTC birthdates to age-band labels as of `as_of`; missing or future dates -> NaN."""
    ages = pd.Series(float("nan"), index=birthdates.index)
    valid = birthdates.notna()
    known = birthdates[valid]
    years = as_of.year - known.dt.year
    before_birthday = (known.dt.month > as_of.month) | (
        (known.dt.month == as_of.month) & (known.dt.day > as_of.day)
    )
    ages[valid] = (years - before_birthday.astype(int)).astype(float)
    return pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS).astype(object)


def drop_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` without any PHI columns."""
    return df.drop(columns=[col for col in df.columns if col in PHI_COLUMNS])


def assert_no_phi(df: pd.DataFrame, table_name: str = "table") -> None:
    """Raise ValueError if `df` still has a PHI column (used before writing outputs)."""
    leaked = sorted(set(df.columns) & PHI_COLUMNS)
    if leaked:
        raise ValueError(f"{table_name} still contains PHI columns: {leaked}")