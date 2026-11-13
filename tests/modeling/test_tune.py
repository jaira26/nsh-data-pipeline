import optuna
import pandas as pd
import pytest

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling import tune
from nsh_pipeline.modeling.candidates import MODEL_NAMES
from nsh_pipeline.modeling.dataset import load_split

SPACE = {
    "n_estimators": {"type": "int", "low": 10, "high": 30, "step": 10},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
    "num_leaves": {"type": "categorical", "choices": [4, 8]},
}


@pytest.fixture(scope="module")
def train(training_sample_dir):
    return load_split("train", training_sample_dir)


def test_suggest_supports_each_type():
    trial = optuna.trial.FixedTrial({"a": 3, "b": 0.5, "c": "x"})
    assert tune.suggest(trial, "a", {"type": "int", "low": 1, "high": 5}) == 3
    assert tune.suggest(trial, "b", {"type": "float", "low": 0.0, "high": 1.0}) == 0.5
    assert tune.suggest(trial, "c", {"type": "categorical", "choices": ["x", "y"]}) == "x"
    with pytest.raises(ValueError, match="type must be one of"):
        tune.suggest(trial, "a", {"type": "uniform"})


def test_search_space_in_params_yaml_is_valid_for_every_candidate():
    space = load_params("params.yaml")["model"]["search_space"]
    assert set(space) == set(MODEL_NAMES)
    for params in space.values():
        for spec in params.values():
            assert spec["type"] in tune.PARAM_TYPES
            if spec["type"] != "categorical":
                assert spec["low"] < spec["high"]


def test_tuning_returns_best_params_inside_the_space(train):
    result = tune.tune_model("lightgbm", train, SPACE, n_trials=3, folds=3, seed=1, log_to_mlflow=False)
    best = result["best_params"]
    assert set(best) == set(SPACE)
    assert 10 <= best["n_estimators"] <= 30 and 0.05 <= best["learning_rate"] <= 0.3 and best["num_leaves"] in (4, 8)
    assert len(result["trials"]) == 3
    assert result["cv_metrics"]["cv_pr_auc"] == max(t["cv_pr_auc"] for t in result["trials"])
    assert set(result["cv_metrics"]) == {"cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc", "cv_roc_auc_std"}


def test_tuning_sees_only_the_training_rows(train, monkeypatch):
    seen = []
    real = tune.cross_validate

    def spy(pipeline, X, y, folds, seed, sample_weight=None):
        seen.append(X.index)
        return real(pipeline, X, y, folds, seed, sample_weight)

    monkeypatch.setattr(tune, "cross_validate", spy)
    tune.tune_model("logistic_regression", train, {"C": {"type": "float", "low": 0.1, "high": 1.0}},
                    n_trials=2, folds=3, seed=0, log_to_mlflow=False)
    assert seen and all(index.equals(train.index) for index in seen)


def test_same_seed_gives_same_result(train):
    runs = [tune.tune_model("random_forest", train, {"n_estimators": SPACE["n_estimators"]}, n_trials=2,
                            folds=3, seed=7, log_to_mlflow=False) for _ in range(2)]
    assert runs[0]["best_params"] == runs[1]["best_params"]
    assert runs[0]["cv_metrics"] == runs[1]["cv_metrics"]


def test_cross_validate_accepts_sample_weights(train):
    from nsh_pipeline.modeling.candidates import build_pipeline, features

    weights = pd.Series(1.0, index=train.index)
    metrics = tune.cross_validate(build_pipeline("logistic_regression"), features(train), train["denied"],
                                  3, 0, sample_weight=weights)
    assert 0 <= metrics["cv_pr_auc"] <= 1
