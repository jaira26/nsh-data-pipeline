"""Build the model's training rows (one per decided swipe) from the phase-1 processed tables.

Label: denied (1) vs approved / partially approved (0); pended swipes are undecided and dropped.
Every feature uses only information available on or before the swipe's service date.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nsh_pipeline.modeling.dataset import LABEL_COLUMN, TRAINING_COLUMNS

DECIDED_STATUSES = {"Approved": 0, "Partial": 0, "Denied": 1}
PRIOR_DENIAL_WINDOW_DAYS = 90
PROCESSED_TABLES = ("swipes", "members", "member_benefits", "clients")
_DEDUCTIBLE_OOP = re.compile(r"(\d{3,5})/(\d{3,5})")


def load_processed(processed_dir) -> dict[str, pd.DataFrame]:
    """Read the processed tables the training set needs."""
    processed_dir = Path(processed_dir)
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in PROCESSED_TABLES}


def plan_deductible(plan_name) -> float:
    """Individual deductible from the first 'deductible/OOP' pair in a plan name (contract glossary)."""
    match = _DEDUCTIBLE_OOP.search(plan_name) if isinstance(plan_name, str) else None
    return float(match.group(1)) if match else np.nan


def prior_denials(swipes: pd.DataFrame, window_days: int = PRIOR_DENIAL_WINDOW_DAYS) -> pd.Series:
    """Per swipe: the member's denied swipes dated 1..window_days days earlier (never same day or later)."""
    denied = swipes.loc[swipes["status"].eq("Denied"), ["member_id", "service_date"]]
    pairs = swipes[["swipe_id", "member_id", "service_date"]].merge(
        denied, on="member_id", suffixes=("", "_denied"))
    gap = (pairs["service_date"] - pairs["service_date_denied"]).dt.days
    counts = pairs[(gap > 0) & (gap <= window_days)].groupby("swipe_id").size()
    return swipes["swipe_id"].map(counts).fillna(0).astype(int)


def _current_medical_plan(member_benefits: pd.DataFrame) -> pd.DataFrame:
    medical = member_benefits[member_benefits["benefit_type"].astype(str).str.lower().eq("medical")]
    latest = medical.sort_values("effective_date").groupby("member_id").tail(1)
    return latest[["member_id", "carrier", "plan_name"]]


def build_training_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per decided swipe with exactly TRAINING_COLUMNS, sorted by service date."""
    swipes = tables["swipes"].copy()
    swipes["prior_denials_90d"] = prior_denials(swipes)  # history uses every swipe, before filtering
    swipes = swipes[swipes["status"].isin(list(DECIDED_STATUSES))]
    members = tables["members"][["member_id", "age_band", "coverage_tier", "hire_date", "is_vip"]]
    clients = tables["clients"][["client_id", "group_size_bucket", "state"]]
    df = (swipes
          .merge(members, on="member_id", how="inner")   # drops orphan swipes
          .merge(clients, on="client_id", how="inner")
          .merge(_current_medical_plan(tables["member_benefits"]), on="member_id", how="left"))
    df["plan_deductible"] = df["plan_name"].map(plan_deductible)
    df["tenure_days"] = (df["service_date"] - df["hire_date"]).dt.days
    df["service_month"] = df["service_date"].dt.month
    df[LABEL_COLUMN] = df["status"].map(DECIDED_STATUSES).astype(int)
    df = df.dropna(subset=["amount_usd", "service_date", "tenure_days", "plan_deductible"])
    df["tenure_days"] = df["tenure_days"].astype(int)
    return df.sort_values(["service_date", "swipe_id"]).reset_index(drop=True)[TRAINING_COLUMNS]