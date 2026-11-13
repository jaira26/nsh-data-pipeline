import json

import pandas as pd
import pytest

from nsh_pipeline.modeling.dataset import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURES,
    SLICE_COLUMNS, SPLITS, TRAINING_COLUMNS, load_dataset_card, load_split, split_path,
)


def test_feature_lists_do_not_overlap():
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_label_and_ids_are_never_features():
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert not set(ID_COLUMNS) & set(FEATURE_COLUMNS)


def test_training_columns_contain_everything_once():
    assert len(TRAINING_COLUMNS) == len(set(TRAINING_COLUMNS))
    assert set(ID_COLUMNS + FEATURE_COLUMNS + SLICE_COLUMNS + [LABEL_COLUMN]) == set(TRAINING_COLUMNS)


def test_split_path_and_unknown_split():
    assert split_path("train", "data/training").name == "train.parquet"
    with pytest.raises(ValueError, match="holdout"):
        split_path("holdout")


def test_load_split_and_card_round_trip(tmp_path):
    df = pd.DataFrame({column: [1] for column in TRAINING_COLUMNS})
    df.to_parquet(tmp_path / "train.parquet", index=False)
    (tmp_path / "dataset_card.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    assert list(load_split("train", tmp_path).columns) == TRAINING_COLUMNS
    assert load_dataset_card(tmp_path) == {"seed": 42}
    assert SPLITS == ("train", "validation", "test")