"""Hyperparameter tuning: Optuna (TPE sampler) with stratified K-fold CV on the TRAINING split only.

The search space for each model lives in params.yaml -> model.search_space, e.g.

    random_forest:
      max_depth: {type: int, low: 3, high: 20}
      max_features: {type: categorical, choices: [sqrt, log2, 0.5]}
      learning_rate: {type: float, low: 0.01, high: 0.3, log: true}

Every trial is logged to MLflow as a nested run (params + CV metrics), so the hyperparameter
sensitivity analysis can read all of them back.
"""
from __future__ import annotations

import logging

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from nsh_pipeline.modeling.candidates import build_pipeline, features
from nsh_pipeline.modeling.dataset import LABEL_COLUMN

logger = logging.getLogger(__name__)

PARAM_TYPES = ("int", "float", "categorical")


def suggest(trial: optuna.Trial, name: str, spec: dict):
    """Draw one hyperparameter value from its params.yaml spec."""
    kind = spec.get("type")
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Hyperparameter {name!r}: type must be one of {PARAM_TYPES}, got {kind!r}")


def cross_validate(pipeline, X: pd.DataFrame, y: pd.Series, folds: int, seed: int,
                   sample_weight: pd.Series | None = None) -> dict:
    """Mean and std of PR-AUC and ROC-AUC over stratified folds of the given (training) rows."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pr, roc = [], []
    for fit_idx, eval_idx in splitter.split(X, y):
        model = clone(pipeline)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = np.asarray(sample_weight)[fit_idx]
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx], **fit_kwargs)
        scores = model.predict_proba(X.iloc[eval_idx])[:, 1]
        pr.append(average_precision_score(y.iloc[eval_idx], scores))
        roc.append(roc_auc_score(y.iloc[eval_idx], scores))
    return {
        "cv_pr_auc": float(np.mean(pr)), "cv_pr_auc_std": float(np.std(pr)),
        "cv_roc_auc": float(np.mean(roc)), "cv_roc_auc_std": float(np.std(roc)),
    }


def tune_model(model_name: str, train: pd.DataFrame, search_space: dict, *, n_trials: int, folds: int,
               seed: int, metric: str = "cv_pr_auc", sample_weight: pd.Series | None = None,
               log_to_mlflow: bool = True, tags: dict | None = None) -> dict:
    """Search the space for one model family; return best params, its CV metrics and every trial.

    Only `train` is ever seen here: the validation split is kept for model selection and the
    test split for Raj's hold-out validation. `tags` (data_version, git_sha, ...) go on every trial run.
    """
    X, y = features(train), train[LABEL_COLUMN].astype(int)
    trials: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest(trial, name, spec) for name, spec in (search_space or {}).items()}
        metrics = cross_validate(build_pipeline(model_name, params, seed), X, y, folds, seed, sample_weight)
        trials.append({"trial": trial.number, **params, **metrics})
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{model_name}-trial-{trial.number:03d}", nested=True):
                mlflow.set_tags({**(tags or {}), "model_name": model_name, "stage": "tuning_trial",
                                 "trial": trial.number})
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
        return metrics[metric]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=model_name)
    study.optimize(objective, n_trials=n_trials)
    best = next(t for t in trials if t["trial"] == study.best_trial.number)
    best_params = {name: best[name] for name in (search_space or {})}
    cv_metrics = {key: best[key] for key in best if key.startswith("cv_")}
    logger.info("%s: best %s = %.4f after %d trials", model_name, metric, study.best_value, n_trials)
    return {"best_params": best_params, "cv_metrics": cv_metrics, "trials": trials}
