"""Time-based train / validation / test split and the dataset card (params.yaml -> training_data)."""
import json
from pathlib import Path

import pandas as pd

from nsh_pipeline.modeling.dataset import (
    CATEGORICAL_FEATURES, DATASET_CARD, LABEL_COLUMN, NUMERIC_FEATURES, SLICE_COLUMNS, SPLITS, split_path,
)


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def split_by_time(frame: pd.DataFrame, validation_start, test_start) -> dict[str, pd.DataFrame]:
    """train < validation_start <= validation < test_start <= test, by service_date."""
    validation_start, test_start = _utc(validation_start), _utc(test_start)
    if not validation_start < test_start:
        raise ValueError("validation_start must be before test_start")
    dates = frame["service_date"]
    splits = {
        "train": frame[dates < validation_start],
        "validation": frame[(dates >= validation_start) & (dates < test_start)],
        "test": frame[dates >= test_start],
    }
    empty = [name for name, part in splits.items() if part.empty]
    if empty:
        raise ValueError(f"Empty split(s) {empty}: check training_data dates against the data's date range")
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def dataset_card(splits: dict[str, pd.DataFrame], settings: dict, source: dict | None = None) -> dict:
    """Facts about the training set: label rule, features, split dates, rows and denial rate per split."""
    return {
        "label": LABEL_COLUMN,
        "label_rule": "1 = Denied; 0 = Approved or Partial; Pended swipes excluded",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "slice_columns": SLICE_COLUMNS,
        "split_rule": "by service_date",
        "validation_start": str(settings["validation_start"]),
        "test_start": str(settings["test_start"]),
        "source_data": source or {},
        "splits": {
            name: {
                "rows": len(part),
                "positive_rate": round(float(part[LABEL_COLUMN].mean()), 4),
                "start": part["service_date"].min().date().isoformat(),
                "end": part["service_date"].max().date().isoformat(),
            }
            for name, part in splits.items()
        },
    }


def write_training_set(splits: dict[str, pd.DataFrame], out_dir, card: dict) -> Path:
    """Write train/validation/test Parquet files and dataset_card.json into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        splits[name].to_parquet(split_path(name, out_dir), index=False)
    (out_dir / DATASET_CARD).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return out_dir