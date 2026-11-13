import joblib
import numpy as np
import pytest

from nsh_pipeline.modeling.candidates import (
    FIXED_PARAMS, MODEL_NAMES, build_estimator, build_pipeline, features,
)
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, LABEL_COLUMN, load_split

FAST = {"random_forest": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20}}


@pytest.fixture(scope="module")
def splits(training_sample_dir):
    return load_split("train", training_sample_dir), load_split("validation", training_sample_dir)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_family_fits_and_predicts_probabilities(name, splits):
    train, validation = splits
    model = build_pipeline(name, FAST.get(name)).fit(features(train), train[LABEL_COLUMN])
    proba = model.predict_proba(features(validation))
    assert proba.shape == (len(validation), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_unseen_and_missing_categories_do_not_break_prediction(splits):
    train, validation = splits
    model = build_pipeline("logistic_regression").fit(features(train), train[LABEL_COLUMN])
    odd = validation.head(3).copy()
    odd["state"] = ["ZZ", None, "CA"]
    odd["is_debit_card"] = [True, None, False]
    assert model.predict_proba(features(odd)).shape == (3, 2)


def test_class_imbalance_is_handled_in_every_family():
    for name in MODEL_NAMES:
        assert "balanced" in FIXED_PARAMS[name]["class_weight"]


def test_tuned_params_override_fixed_ones_and_seed_is_set():
    model = build_estimator("random_forest", {"n_estimators": 7, "class_weight": None}, seed=3)
    assert model.n_estimators == 7 and model.class_weight is None and model.random_state == 3


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_estimator("svm")


def test_features_selects_contract_columns_in_order(splits):
    train, _ = splits
    assert list(features(train[list(reversed(train.columns))]).columns) == FEATURE_COLUMNS
    with pytest.raises(KeyError, match="amount_usd"):
        features(train.drop(columns="amount_usd"))


def test_fitted_pipeline_survives_joblib_round_trip(splits, tmp_path):
    train, validation = splits
    model = build_pipeline("lightgbm", FAST["lightgbm"]).fit(features(train), train[LABEL_COLUMN])
    joblib.dump(model, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(loaded.predict_proba(features(validation)), model.predict_proba(features(validation)))
