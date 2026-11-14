"""Bias mitigation for the denial model: reweighing on the flagged slice, then refit (phase 2, model bias).

Reweighing (Kamiran & Calders, the phase-1 helper in bias/mitigation.py) gives every TRAIN row the
weight P(group) * P(denied) / P(group, denied), so in the weighted training data the denial rate is
the same in every group of the slice. The candidate is refit with its tuned hyperparameters and
these weights as sample_weight; nothing is re-tuned, so the before/after difference is the effect
of the weights alone. The result is an ordinary Pipeline with predict_proba(X[FEATURE_COLUMNS]),
so validation, registry and serving code need no change.
"""
from __future__ import annotations

import pandas as pd

from nsh_pipeline.bias.mitigation import reweigh
from nsh_pipeline.modeling.candidates import MODEL_NAMES, build_pipeline, features
from nsh_pipeline.modeling.dataset import LABEL_COLUMN

METHOD = "reweighing"
SUFFIX = "_reweighed"


def mitigated_name(model_name: str) -> str:
    """Name of the mitigated version of a candidate, e.g. random_forest_reweighed."""
    return f"{model_name}{SUFFIX}"


def slice_groups(frame: pd.DataFrame, column: str) -> pd.Series:
    """Slice values as text; a missing value is its own group, "Unknown" (as in the fairness audit)."""
    return frame[column].astype(object).where(frame[column].notna(), "Unknown").astype(str)


def reweigh_and_refit(model_name: str, params: dict, train: pd.DataFrame, column: str, seed: int = 42):
    """Refit one candidate on the train split, reweighed on `column`. Returns (pipeline, mitigation info)."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Cannot refit unknown model {model_name!r}; expected one of {MODEL_NAMES}")
    if column not in train.columns:
        raise KeyError(f"Slice column {column!r} is not in the train split")
    labels = train[LABEL_COLUMN].astype(int)
    result = reweigh(labels, slice_groups(train, column))
    model = build_pipeline(model_name, params, seed)
    model.fit(features(train), labels, model__sample_weight=result["weights"].to_numpy())
    comparison = result["comparison"]
    info = {
        "method": METHOD,
        "slice": column,
        "base_model": model_name,
        "train_rows": int(result["n"]),
        "effective_n": round(result["effective_n"], 1),
        "effective_share": round(result["effective_n"] / result["n"], 4),
        "weight_min": round(result["weight_min"], 4),
        "weight_max": round(result["weight_max"], 4),
        "train_denial_gap_before_pts": round(result["max_gap_before_pts"], 2),
        "train_denial_gap_after_pts": round(result["max_gap_after_pts"], 2),
        "train_denial_rate_by_group": {
            str(group): {"n": int(row.n), "before": round(float(row.before), 4), "after": round(float(row.after), 4)}
            for group, row in comparison.iterrows()
        },
    }
    return model, info
