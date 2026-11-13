import pandas as pd
import pytest

from nsh_pipeline.modeling.dataset import LABEL_COLUMN, SPLITS, TRAINING_COLUMNS, load_dataset_card, load_split
from nsh_pipeline.modeling.split import dataset_card, split_by_time, write_training_set

SETTINGS = {"validation_start": "2025-05-01", "test_start": "2025-06-01"}


def frame(dates, labels=None):
    dates = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({"swipe_id": [f"s{i}" for i in range(len(dates))], "service_date": dates,
                         LABEL_COLUMN: labels or [0] * len(dates)})


def test_rows_go_to_the_right_split_including_boundaries():
    df = frame(["2025-04-30", "2025-05-01", "2025-05-31", "2025-06-01", "2025-06-15"])
    splits = split_by_time(df, **SETTINGS)
    assert splits["train"]["swipe_id"].tolist() == ["s0"]
    assert splits["validation"]["swipe_id"].tolist() == ["s1", "s2"]  # 1 May belongs to validation
    assert splits["test"]["swipe_id"].tolist() == ["s3", "s4"]        # 1 June belongs to test


def test_splits_are_disjoint_complete_and_in_time_order():
    df = frame(pd.date_range("2025-01-01", "2025-06-15", freq="D"))
    splits = split_by_time(df, **SETTINGS)
    assert sum(len(part) for part in splits.values()) == len(df)
    assert splits["train"]["service_date"].max() < splits["validation"]["service_date"].min()
    assert splits["validation"]["service_date"].max() < splits["test"]["service_date"].min()


def test_wrong_date_order_is_rejected():
    with pytest.raises(ValueError, match="before"):
        split_by_time(frame(["2025-01-01"]), "2025-06-01", "2025-05-01")


def test_empty_split_is_rejected():
    with pytest.raises(ValueError, match="test"):
        split_by_time(frame(["2025-01-01", "2025-05-02"]), **SETTINGS)


def test_card_records_rows_rates_and_dates():
    df = frame(["2025-01-05", "2025-01-06", "2025-05-02", "2025-06-03"], labels=[1, 0, 0, 1])
    card = dataset_card(split_by_time(df, **SETTINGS), SETTINGS, source={"seed": 42})
    assert card["splits"]["train"] == {"rows": 2, "positive_rate": 0.5, "start": "2025-01-05", "end": "2025-01-06"}
    assert card["splits"]["test"]["positive_rate"] == 1.0
    assert card["source_data"] == {"seed": 42}
    assert card["test_start"] == "2025-06-01"


def test_resplitting_the_sample_reproduces_its_splits(training_sample_dir, tmp_path):
    whole = pd.concat([load_split(name, training_sample_dir) for name in SPLITS], ignore_index=True)
    splits = split_by_time(whole, **SETTINGS)
    expected = load_dataset_card(training_sample_dir)["splits"]
    assert {name: len(part) for name, part in splits.items()} == {name: expected[name]["rows"] for name in SPLITS}
    write_training_set(splits, tmp_path, dataset_card(splits, SETTINGS))
    for name in SPLITS:
        assert list(load_split(name, tmp_path).columns) == TRAINING_COLUMNS
    assert load_dataset_card(tmp_path)["splits"]["train"]["rows"] == expected["train"]["rows"]