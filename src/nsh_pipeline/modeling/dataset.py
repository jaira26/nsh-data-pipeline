"""Training-data contract for the model pipeline: column names, file layout and loaders.

Every other modelling module imports these names instead of typing column names, so the
training data and the models always agree. Built from the phase-1 processed tables.
"""
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/training")
SPLITS = ("train", "validation", "test")
DATASET_CARD = "dataset_card.json"

ID_COLUMNS = ["swipe_id", "member_id", "client_id", "service_date"]
LABEL_COLUMN = "denied"  # 1 = swipe denied, 0 = approved or partially approved

NUMERIC_FEATURES = [
    "amount_usd",          # swipe amount in dollars
    "plan_deductible",     # individual deductible of the member's plan
    "tenure_days",         # days between hire date and service date
    "prior_denials_90d",   # member's denied swipes in the 90 days BEFORE this swipe
    "service_month",       # 1-12
]
CATEGORICAL_FEATURES = [
    "service_category",    # Medical, Drugs & Medicine, Dental, Vision
    "coverage_tier",       # EE, ES, EC, EF
    "carrier",
    "age_band",
    "group_size_bucket",
    "state",
    "is_debit_card",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Groups the model is checked for bias on. Some are also features; is_vip is slice-only.
SLICE_COLUMNS = ["state", "group_size_bucket", "age_band", "coverage_tier", "is_vip"]

# Every column of a split file, in order, without repeats.
TRAINING_COLUMNS = list(dict.fromkeys(ID_COLUMNS + FEATURE_COLUMNS + SLICE_COLUMNS + [LABEL_COLUMN]))


def split_path(name: str, data_dir=DEFAULT_DATA_DIR) -> Path:
    """Path of one split file, e.g. data/training/train.parquet."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLITS}")
    return Path(data_dir) / f"{name}.parquet"


def load_split(name: str, data_dir=DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load one split (train, validation or test) as a DataFrame with TRAINING_COLUMNS."""
    return pd.read_parquet(split_path(name, data_dir))


def load_dataset_card(data_dir=DEFAULT_DATA_DIR) -> dict:
    """Load dataset_card.json: data version, rows and positive rate per split, split dates, seed."""
    return json.loads((Path(data_dir) / DATASET_CARD).read_text(encoding="utf-8"))