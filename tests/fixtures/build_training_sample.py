"""Build the small sample training set in tests/fixtures/training/ (contract-shaped, seeded).

Run from the repo root:  python tests/fixtures/build_training_sample.py

A stand-in with a simple, known signal so model code can be built and tested before the real
training set exists. It follows nsh_pipeline.modeling.dataset exactly.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nsh_pipeline.modeling.dataset import DATASET_CARD, LABEL_COLUMN, SPLITS, TRAINING_COLUMNS, split_path

OUT_DIR = Path(__file__).parent / "training"
SEED = 42
N_ROWS = 600
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-06-15", tz="UTC")
VALIDATION_START = pd.Timestamp("2025-05-01", tz="UTC")
TEST_START = pd.Timestamp("2025-06-01", tz="UTC")


def build_sample(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    """Contract-shaped swipes with a known denial signal (category, amount, deductible, history, tenure)."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.integers(0, (END - START).days + 1, size=n_rows))
    dates = START + pd.to_timedelta(days, unit="D")
    category = rng.choice(["Medical", "Drugs & Medicine", "Dental", "Vision"], size=n_rows,
                          p=[0.55, 0.25, 0.12, 0.08])
    amount = np.clip(rng.lognormal(4.3, 1.0, n_rows), 5, 5000).round(2)
    deductible = rng.choice([1000, 2000, 3000, 5000], size=n_rows)
    tenure = rng.integers(0, 3000, size=n_rows)
    prior = rng.poisson(0.3, size=n_rows)
    logit = (-2.6 + 0.6 * np.log(amount / 75) + 0.9 * np.isin(category, ["Dental", "Vision"])
             + 0.5 * (deductible >= 3000) + 0.6 * prior - 0.0004 * tenure)
    denied = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "swipe_id": [f"S-{i:05d}" for i in range(1, n_rows + 1)],
        "member_id": [f"m{int(m):03d}" for m in rng.integers(1, 81, n_rows)],
        "client_id": [f"c{int(c):02d}" for c in rng.integers(1, 11, n_rows)],
        "service_date": dates,
        "amount_usd": amount,
        "plan_deductible": deductible.astype(float),
        "tenure_days": tenure,
        "prior_denials_90d": prior,
        "service_month": dates.month,
        "service_category": category,
        "coverage_tier": rng.choice(["EE", "ES", "EC", "EF"], n_rows, p=[0.5, 0.15, 0.15, 0.2]),
        "carrier": rng.choice(["Aetna", "Cigna", "Anthem", "BSC", "BCBS"], n_rows),
        "age_band": rng.choice(["18-25", "26-34", "35-44", "45-54", "55-64", "65+"], n_rows),
        "group_size_bucket": rng.choice(["1-9", "10-24", "25-49", "50-99", "100-249", "250+"], n_rows),
        "state": rng.choice(["OR", "WA", "CA", "TX", "NY"], n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "is_debit_card": rng.random(n_rows) < 0.92,
        "is_vip": rng.random(n_rows) < 0.01,
        LABEL_COLUMN: denied,
    })
    return df[TRAINING_COLUMNS]


def split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based split: train before May, validation in May, test from June."""
    dates = df["service_date"]
    return {
        "train": df[dates < VALIDATION_START],
        "validation": df[(dates >= VALIDATION_START) & (dates < TEST_START)],
        "test": df[dates >= TEST_START],
    }


def main():
    splits = split_by_date(build_sample())
    card = {"dataset": "training-sample-fixture", "data_version": "sample", "seed": SEED,
            "label": LABEL_COLUMN, "splits": {}}
    for name in SPLITS:
        part = splits[name].reset_index(drop=True)
        path = split_path(name, OUT_DIR)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        card["splits"][name] = {
            "rows": len(part),
            "positive_rate": round(float(part[LABEL_COLUMN].mean()), 4),
            "start": part["service_date"].min().date().isoformat(),
            "end": part["service_date"].max().date().isoformat(),
        }
    (OUT_DIR / DATASET_CARD).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(card["splits"], indent=2))


if __name__ == "__main__":
    main()