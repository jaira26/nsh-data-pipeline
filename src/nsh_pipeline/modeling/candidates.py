"""The three candidate model families as scikit-learn Pipelines (shared preprocessing + estimator).

Every model is a Pipeline whose input is ``X[FEATURE_COLUMNS]`` and that supports ``predict_proba``,
so validation (Raj), bias (Het) and the registry (Hiral) can treat all candidates the same way.

    preprocess:  numeric      -> median impute -> standard scale
                 categorical  -> as text (missing = "Unknown") -> one-hot (unseen values ignored)
    estimator:   logistic_regression | random_forest | lightgbm, all class-weighted (denials are ~10%)
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from nsh_pipeline.modeling.dataset import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

MODEL_NAMES = ("logistic_regression", "random_forest", "lightgbm")

# Settings that are not tuned. class_weight handles the class imbalance in every family.
FIXED_PARAMS: dict[str, dict] = {
    "logistic_regression": {"class_weight": "balanced", "solver": "saga", "max_iter": 5000},
    "random_forest": {"class_weight": "balanced_subsample", "n_jobs": -1},
    # LightGBM: one thread; on a few thousand rows threading overhead made CV ~13x slower.
    "lightgbm": {"class_weight": "balanced", "subsample_freq": 1, "verbose": -1, "n_jobs": 1},
}


def as_text(values):
    """Cast categorical inputs (strings, booleans) to plain text so one encoder handles them all.

    A missing value becomes the text "Unknown", its own category. Module-level (not a lambda)
    so the fitted Pipeline can be pickled with joblib.
    """
    frame = pd.DataFrame(values)
    return frame.astype(object).where(frame.notna(), "Unknown").astype(str).to_numpy(dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing: scaled numerics + one-hot categoricals, in FEATURE_COLUMNS order."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("as_text", FunctionTransformer(as_text, feature_names_out="one-to-one")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimator(model_name: str, params: dict | None = None, seed: int = 42):
    """One untrained estimator: fixed settings, then the given (tuned) params on top."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")
    settings = {**FIXED_PARAMS[model_name], **(params or {}), "random_state": seed}
    if model_name == "logistic_regression":
        return LogisticRegression(**settings)
    if model_name == "random_forest":
        return RandomForestClassifier(**settings)
    return LGBMClassifier(**settings)


def build_pipeline(model_name: str, params: dict | None = None, seed: int = 42) -> Pipeline:
    """Preprocessing + estimator, ready to ``fit(X[FEATURE_COLUMNS], y)``."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", build_estimator(model_name, params, seed)),
    ])


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """The model input: the contract's feature columns, in contract order."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return frame[FEATURE_COLUMNS]
