"""Train, tune and select the swipe-denial model (phase 2, step 2). One command does all of it:

    python -m nsh_pipeline.modeling.train --data-dir data/training --params params.yaml

For each candidate in params.yaml -> model.candidates:
  1. tune with Optuna + stratified CV on the train split (every trial = nested MLflow run)
  2. refit the best params on the whole train split
  3. score on the validation split (PR-AUC, ROC-AUC, F1 at the best-F1 threshold, confusion matrix)
  4. save models/candidates/<name>/ and log the model, params, metrics and tags to MLflow
Then the best candidate by validation PR-AUC is copied to models/best/, and the comparison
charts and reports/model/training/candidates.json are written and logged.

Stub for teammates (contract layout, no tuning; a plain logistic regression):

    python -m nsh_pipeline.modeling.train --stub --data-dir tests/fixtures/training
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling import charts
from nsh_pipeline.modeling.artifacts import (
    best_dir, candidate_dir, copy_model, save_model, utc_now,
)
from nsh_pipeline.modeling.candidates import MODEL_NAMES, build_pipeline, features
from nsh_pipeline.modeling.dataset import LABEL_COLUMN, load_dataset_card, load_split
from nsh_pipeline.modeling.select import best_f1_threshold, evaluate, rank_candidates
from nsh_pipeline.modeling.tracking import data_version, run_tags, setup_mlflow
from nsh_pipeline.modeling.tune import tune_model

logger = logging.getLogger(__name__)

DEFAULTS = {
    "candidates": list(MODEL_NAMES),
    "selection_metric": "pr_auc",
    "cv_folds": 5,
    "n_trials": 25,
    "models_dir": "models",
    "reports_dir": "reports/model/training",
    "search_space": {},
    "mlflow": {},
}


def model_settings(params: dict) -> dict:
    """params.yaml -> model, with defaults filled in; the seed falls back to the global seed."""
    settings = {**DEFAULTS, **(params.get("model") or {})}
    settings.setdefault("seed", params.get("seed", 42))
    unknown = [name for name in settings["candidates"] if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown candidate(s) {unknown}; expected some of {MODEL_NAMES}")
    return settings


def load_sample_weights(path, train: pd.DataFrame) -> pd.Series | None:
    """Optional per-swipe training weights (e.g. Het's reweighing: swipe_id, weight); missing swipes get 1."""
    if not path:
        return None
    weights = pd.read_parquet(path).set_index("swipe_id")["weight"]
    return train["swipe_id"].map(weights).fillna(1.0).astype(float).reset_index(drop=True)


def train_candidate(model_name: str, train: pd.DataFrame, validation: pd.DataFrame, settings: dict,
                    tags: dict, card: dict, *, n_trials: int, sample_weight=None) -> dict:
    """Tune, refit and score one model family inside its own MLflow run; save it as a candidate."""
    seed, metric = settings["seed"], settings["selection_metric"]
    # Nested under the train-and-select run when one is active; its tuning trials nest under this run.
    with mlflow.start_run(run_name=model_name, nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({**tags, "model_name": model_name, "stage": "candidate"})
        tuned = tune_model(model_name, train, settings["search_space"].get(model_name, {}), n_trials=n_trials,
                           folds=settings["cv_folds"], seed=seed, metric=f"cv_{metric}", sample_weight=sample_weight,
                           tags=tags)

        model = build_pipeline(model_name, tuned["best_params"], seed)
        fit_kwargs = {"model__sample_weight": sample_weight.to_numpy()} if sample_weight is not None else {}
        model.fit(features(train), train[LABEL_COLUMN].astype(int), **fit_kwargs)

        y_val = validation[LABEL_COLUMN].astype(int)
        scores = model.predict_proba(features(validation))[:, 1]
        val_metrics = evaluate(y_val, scores, best_f1_threshold(y_val, scores))

        mlflow.log_params({**tuned["best_params"], "n_trials": n_trials, "cv_folds": settings["cv_folds"],
                           "sample_weighted": sample_weight is not None})
        mlflow.log_metrics(tuned["cv_metrics"])
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)})
        trials_path = Path(settings["reports_dir"]) / "trials" / f"{model_name}.csv"
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tuned["trials"]).to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path), artifact_path="tuning")
        if settings["mlflow"].get("log_models", True):
            # cloudpickle: skops (MLflow's default) rejects our custom as_text step. Same trust model as
            # models/*.joblib: only load models this pipeline produced.
            sample = features(train).head(5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # integer-column schema hint; our features have no NaNs
                mlflow.sklearn.log_model(model, name="model", input_example=sample,
                                         signature=infer_signature(sample, model.predict_proba(sample)),
                                         serialization_format="cloudpickle")

        metadata = {
            "model_name": model_name,
            "params": tuned["best_params"],
            "cv_metrics": tuned["cv_metrics"],
            "validation_metrics": val_metrics,
            "mlflow_run_id": run.info.run_id,
            "data_version": data_version(card),
            "git_sha": tags["git_sha"],
            "created_at": utc_now(),
            "seed": seed,
            "n_trials": n_trials,
            "sample_weighted": sample_weight is not None,
        }
        save_model(model, metadata, candidate_dir(model_name, settings["models_dir"]))
        logger.info("%s: CV PR-AUC %.4f, validation PR-AUC %.4f, ROC-AUC %.4f", model_name,
                    tuned["cv_metrics"]["cv_pr_auc"], val_metrics["pr_auc"], val_metrics["roc_auc"])
        return {**metadata, "_scores": (y_val.to_numpy(), scores)}


def run_training(params: dict, data_dir, *, n_trials: int | None = None, candidates: list[str] | None = None,
                 sample_weights=None, models_dir=None, reports_dir=None) -> dict:
    """Train every candidate, select the best into models/best/, write charts and candidates.json."""
    settings = model_settings(params)
    if models_dir:
        settings["models_dir"] = str(models_dir)
    if reports_dir:
        settings["reports_dir"] = str(reports_dir)
    names = candidates or settings["candidates"]
    n_trials = n_trials or settings["n_trials"]
    reports = Path(settings["reports_dir"])

    card = load_dataset_card(data_dir)
    train, validation = load_split("train", data_dir), load_split("validation", data_dir)
    sample_weight = load_sample_weights(sample_weights, train)
    setup_mlflow(settings["mlflow"])
    tags = run_tags(card, data_dir, {"sample_weights": sample_weights or "none"})

    results = {}
    reports.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="train-and-select") as parent:
        mlflow.set_tags({**tags, "stage": "selection"})
        mlflow.log_params({"candidates": ",".join(names), "n_trials": n_trials, "cv_folds": settings["cv_folds"],
                           "selection_metric": settings["selection_metric"], "seed": settings["seed"]})
        for name in names:
            results[name] = train_candidate(name, train, validation, settings, tags, card,
                                            n_trials=n_trials, sample_weight=sample_weight)

        curves = {name: result.pop("_scores") for name, result in results.items()}
        ranking = rank_candidates(results, settings["selection_metric"])
        best = ranking[0]
        copy_model(candidate_dir(best, settings["models_dir"]), best_dir(settings["models_dir"]))

        summary = {
            "selected_model": best,
            "selection_metric": f"validation {settings['selection_metric']}",
            "ranking": ranking,
            "data_version": data_version(card),
            "git_sha": tags["git_sha"],
            "mlflow_parent_run_id": parent.info.run_id,
            "candidates": {name: {key: result[key] for key in
                                  ("params", "cv_metrics", "validation_metrics", "mlflow_run_id")}
                           for name, result in results.items()},
        }
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "candidates.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        charts.comparison_chart(results, best, reports / "model_comparison.png")
        charts.confusion_matrices(results, reports / "confusion_matrices.png")
        charts.pr_curves(curves, reports / "pr_curves.png")
        for file in ("candidates.json", "model_comparison.png", "confusion_matrices.png", "pr_curves.png"):
            mlflow.log_artifact(str(reports / file), artifact_path="selection")
        mlflow.set_tag("selected_model", best)
        mlflow.log_metric(f"best_val_{settings['selection_metric']}",
                          results[best]["validation_metrics"][settings["selection_metric"]])
    logger.info("Selected %s -> %s", best, best_dir(settings["models_dir"]))
    return summary


def train_stub(data_dir, models_dir="models", seed: int = 42) -> Path:
    """Contract-shaped stub: an untuned logistic regression in candidates/ and best/ (no MLflow)."""
    card = load_dataset_card(data_dir)
    train, validation = load_split("train", data_dir), load_split("validation", data_dir)
    model = build_pipeline("logistic_regression", {}, seed).fit(features(train), train[LABEL_COLUMN].astype(int))
    y_val = validation[LABEL_COLUMN].astype(int)
    scores = model.predict_proba(features(validation))[:, 1]
    metadata = {
        "model_name": "logistic_regression", "params": {}, "cv_metrics": {},
        "validation_metrics": evaluate(y_val, scores, best_f1_threshold(y_val, scores)),
        "mlflow_run_id": None, "data_version": data_version(card), "git_sha": "stub", "created_at": utc_now(),
        "seed": seed, "stub": True,
    }
    save_model(model, metadata, candidate_dir("logistic_regression", models_dir))
    return copy_model(candidate_dir("logistic_regression", models_dir), best_dir(models_dir))


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train, tune and select the swipe-denial model.")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials per model (default: params.yaml)")
    parser.add_argument("--candidates", nargs="+", default=None, choices=MODEL_NAMES)
    parser.add_argument("--sample-weights", default=None, help="parquet with swipe_id, weight (bias mitigation)")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--stub", action="store_true", help="write the untuned stub model only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.stub:
        out = train_stub(args.data_dir, args.models_dir or "models")
        print(f"Stub model written to {out}")
        return {"selected_model": "logistic_regression", "stub": True}
    summary = run_training(load_params(args.params), args.data_dir, n_trials=args.n_trials,
                           candidates=args.candidates, sample_weights=args.sample_weights,
                           models_dir=args.models_dir, reports_dir=args.reports_dir)
    print(json.dumps({"selected_model": summary["selected_model"], "ranking": summary["ranking"]}, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
