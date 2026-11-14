"""Sensitivity analysis of the selected model (phase 2, step 3). One command:

    python -m nsh_pipeline.modeling.sensitivity --params params.yaml --model-dir models/best \
        --data-dir data/training --out-dir reports/model/sensitivity

Two parts, both written to --out-dir and logged to MLflow:

SHAP (feature sensitivity)
  * explains the model on a fixed-seed sample of the test split. The Pipeline's preprocessing step
    transforms the rows, and the explainer is picked by estimator type (TreeExplainer for tree
    ensembles, LinearExplainer for linear models, a model-agnostic permutation explainer otherwise)
  * one-hot SHAP columns are summed back to the original feature (SHAP values are additive)
  * shap_summary.png (beeswarm), shap_importance.png (mean |SHAP| bar), three waterfall plots
    (a true positive, a false positive, a false negative) and top_features.json
  * signal check: are the documented denial drivers of data-v1.1 near the top? Logged as a warning when not

Hyperparameter sensitivity
  * reads the tuning trials from MLflow (experiment nsh-denial-model, tags.stage = 'tuning_trial'),
    groups them by model family and plots the CV metric against every hyperparameter
  * estimates each hyperparameter's importance (random-forest permutation importance, or Spearman /
    correlation ratio when there are too few runs) -> hyperparameter_importance.json
  * no runs / too few runs: a clear message in the JSON and the log, never a crash
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from nsh_pipeline.config import load_params  # noqa: E402
from nsh_pipeline.modeling.artifacts import load_model  # noqa: E402
from nsh_pipeline.modeling.dataset import (  # noqa: E402
    CATEGORICAL_FEATURES, DATASET_CARD, FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURES, load_dataset_card,
    load_split,
)
from nsh_pipeline.modeling.evaluate import (  # noqa: E402
    REPORT_FILE, THRESHOLD_SPLIT, mlflow_run, predict_scores, run_tags, threshold_from_split, validation_settings,
    write_json,
)
from nsh_pipeline.modeling.tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI  # noqa: E402

logger = logging.getLogger(__name__)

SENSITIVITY_DEFAULTS = {
    "out_dir": "reports/model/sensitivity",
    "shap_sample_size": 500,          # test rows explained (fixed seed)
    "shap_background_size": 100,      # train rows the linear / permutation explainers integrate over
    # Documented causes of a denial in data-v1.1 (params.yaml -> denial_model).
    "expected_drivers": ["amount_usd", "service_category", "tenure_days", "plan_deductible", "prior_denials_90d"],
    "driver_top_n": 7,                # every expected driver should rank in the top N
    "tuning_metric": "cv_pr_auc",
    "tuning_run_filter": "tags.stage = 'tuning_trial'",
    "min_runs": 3,                    # fewer runs per family: skipped
    "min_runs_for_forest": 10,        # fewer: Spearman / correlation-ratio fallback
}

TREE_ESTIMATORS = {
    "RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier",
    "HistGradientBoostingClassifier", "LGBMClassifier", "XGBClassifier", "CatBoostClassifier",
}
LINEAR_ESTIMATORS = {"LogisticRegression", "LogisticRegressionCV", "SGDClassifier", "RidgeClassifier",
                     "LinearSVC", "Perceptron"}
# TreeExplainer explains a random forest in probability space; boosted trees in log-odds (margin) space.
PROBABILITY_TREES = {"RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier"}

LOCAL_CASES = ("true_positive", "false_positive", "false_negative")
TOP_FEATURES_FILE = "top_features.json"
HP_IMPORTANCE_FILE = "hyperparameter_importance.json"
BAR = "#2b6cb0"
DRIVER = "#2f855a"
MARK = "#c53030"


def sensitivity_settings(params: dict) -> dict:
    """model_validation settings plus model_validation.sensitivity with defaults filled in."""
    settings = validation_settings(params)
    settings["sensitivity"] = {**SENSITIVITY_DEFAULTS, **(settings.get("sensitivity") or {})}
    settings["search_space"] = (params.get("model") or {}).get("search_space") or {}
    return settings


# --------------------------------------------------------------------------------------------- SHAP

def split_pipeline(model):
    """(preprocessing, estimator) of a fitted Pipeline; the estimator is the last step."""
    steps = getattr(model, "steps", None)
    if not steps:
        raise TypeError(f"Expected a scikit-learn Pipeline (preprocessing + estimator), got {type(model).__name__}")
    return model[:-1], model[-1]


def transform(preprocess, frame) -> np.ndarray:
    values = preprocess.transform(frame[FEATURE_COLUMNS])
    values = values.toarray() if hasattr(values, "toarray") else values
    return np.asarray(values, dtype=float)


def original_feature(name: str, features=FEATURE_COLUMNS) -> str:
    """Original feature of a transformed column: amount_usd -> amount_usd, service_category_Dental ->
    service_category, categorical__state_CA -> state. Longest matching feature name wins."""
    candidates = [name, name.split("__", 1)[1]] if "__" in name else [name]
    for candidate in candidates:
        if candidate in features:
            return candidate
        matches = [f for f in features if candidate.startswith(f"{f}_")]
        if matches:
            return max(matches, key=len)
    return name


def aggregate_shap(values: np.ndarray, transformed_names, features=FEATURE_COLUMNS) -> pd.DataFrame:
    """Sum SHAP values of the one-hot columns of each original feature (rows x original features)."""
    values = np.asarray(values, dtype=float)
    owners = [original_feature(str(name), features) for name in transformed_names]
    frame = pd.DataFrame(values, columns=range(values.shape[1])).T.groupby(owners, sort=False).sum().T
    ordered = [f for f in features if f in frame.columns] + [c for c in frame.columns if c not in features]
    return frame[ordered]


def explainer_kind(estimator) -> str:
    name = type(estimator).__name__
    if name in TREE_ESTIMATORS:
        return "tree"
    if name in LINEAR_ESTIMATORS:
        return "linear"
    return "permutation"


def _positive_class(values) -> np.ndarray:
    """Binary-classifier SHAP output -> one (rows x columns) array for the denied class."""
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    return values[..., -1] if values.ndim == 3 else values


def _base_value(expected) -> float:
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(expected[-1]) if expected.size == 2 else float(expected.mean())


class PipelineExplainer:
    """SHAP for a preprocessing + estimator Pipeline, reported per original feature."""

    def __init__(self, model, background: pd.DataFrame):
        self.preprocess, self.estimator = split_pipeline(model)
        self.names = [str(n) for n in self.preprocess.get_feature_names_out()]
        self.kind = explainer_kind(self.estimator)
        background_values = transform(self.preprocess, background)
        name = type(self.estimator).__name__
        if self.kind == "tree":
            self._explainer = shap.TreeExplainer(self.estimator)
            self.output_space = "probability" if name in PROBABILITY_TREES else "log-odds"
        elif self.kind == "linear":
            self._explainer = shap.LinearExplainer(self.estimator, background_values)
            self.output_space = "log-odds"
        else:
            def positive_proba(x):
                return self.estimator.predict_proba(x)[:, 1]
            self._explainer = shap.Explainer(positive_proba, shap.maskers.Independent(background_values),
                                             algorithm="permutation")
            self.output_space = "probability"
        self.explainer_name = type(self._explainer).__name__

    def explain(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """(rows x original features SHAP values, base value) for these rows."""
        values = transform(self.preprocess, frame)
        if self.kind == "tree":
            raw = self._explainer.shap_values(values, check_additivity=False)
            base = _base_value(self._explainer.expected_value)
        elif self.kind == "linear":
            raw = self._explainer.shap_values(values)
            base = _base_value(self._explainer.expected_value)
        else:
            explanation = self._explainer(values)
            raw, base = explanation.values, _base_value(explanation.base_values)
        shap_values = aggregate_shap(_positive_class(raw), self.names)
        shap_values.index = frame.index
        return shap_values, base


def rank_features(shap_values: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per original feature, largest first."""
    return shap_values.abs().mean().sort_values(ascending=False)


def signal_check(ranking: pd.Series, expected_drivers, top_n: int) -> dict:
    """Compare the top features with the documented denial drivers."""
    ranks = {feature: i + 1 for i, feature in enumerate(ranking.index)}
    top = list(ranking.index[:top_n])
    found = [d for d in expected_drivers if d in top]
    missing = [d for d in expected_drivers if d not in top]
    return {
        "expected_drivers": list(expected_drivers),
        "top_n": int(top_n),
        "found_in_top_n": found,
        "missing_from_top_n": missing,
        "unexpected_in_top_n": [f for f in top if f not in expected_drivers],
        "driver_ranks": {d: ranks.get(d) for d in expected_drivers},
        "matches": not missing,
    }


def _display_value(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    return str(value)


def _colour_data(frame: pd.DataFrame, columns) -> np.ndarray:
    """Beeswarm colours: numeric feature values; categoricals have no order, so they are grey (NaN)."""
    out = np.full((len(frame), len(columns)), np.nan)
    for j, column in enumerate(columns):
        if column in NUMERIC_FEATURES:
            out[:, j] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return out


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_plot(shap_values: pd.DataFrame, frame: pd.DataFrame, base: float, output_space: str, path) -> Path:
    columns = list(shap_values.columns)
    explanation = shap.Explanation(values=shap_values.to_numpy(), base_values=np.full(len(shap_values), base),
                                   data=_colour_data(frame, columns), feature_names=columns)
    plt.figure()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")  # the grey categorical rows
        shap.plots.beeswarm(explanation, max_display=len(columns), show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary, test sample (n={len(shap_values)}, {output_space}); categoricals in grey",
                 fontsize=10)
    return _save(fig, path)


def importance_plot(ranking: pd.Series, expected_drivers, output_space: str, path) -> Path:
    ordered = ranking.iloc[::-1]
    colours = [DRIVER if f in expected_drivers else BAR for f in ordered.index]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(ordered) + 1.5))
    ax.barh(ordered.index, ordered.to_numpy(), color=colours)
    ax.set(xlabel=f"mean |SHAP value| ({output_space})", title="Global feature importance (test sample)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DRIVER), plt.Rectangle((0, 0), 1, 1, color=BAR)],
              labels=["documented denial driver", "other feature"], loc="lower right")
    return _save(fig, path)


def waterfall_plot(row_shap: pd.Series, row: pd.Series, base: float, title: str, path) -> Path:
    columns = list(row_shap.index)
    data = np.array([_display_value(row[c]) for c in columns], dtype=object)
    explanation = shap.Explanation(values=row_shap.to_numpy(dtype=float), base_values=base, data=data,
                                   feature_names=columns)
    plt.figure()
    shap.plots.waterfall(explanation, max_display=len(columns), show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def pick_local_cases(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Row positions of the clearest true positive, false positive and false negative."""
    y = frame[LABEL_COLUMN].astype(int).to_numpy()
    predicted = scores >= threshold
    masks = {"true_positive": (y == 1) & predicted, "false_positive": (y == 0) & predicted,
             "false_negative": (y == 1) & ~predicted}
    cases = {}
    for case, mask in masks.items():
        positions = np.flatnonzero(mask)
        if len(positions):
            pick = positions[np.argmin(scores[positions])] if case == "false_negative" else \
                positions[np.argmax(scores[positions])]
            cases[case] = int(pick)
    return cases


def resolve_threshold(model, settings: dict, data_dir, validation_report) -> tuple[float, str]:
    """The validation step's threshold (validation.json), else chosen again on the validation split."""
    path = Path(validation_report) if validation_report else None
    if path and path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["threshold"]), str(path)
    threshold, _ = threshold_from_split(model, load_split(THRESHOLD_SPLIT, data_dir), THRESHOLD_SPLIT, settings)
    return threshold, f"chosen on the validation split ({settings['threshold_method']})"


def run_shap(model, metadata: dict, settings: dict, data_dir, out_dir, validation_report=None) -> dict:
    """SHAP plots, local explanations and top_features.json; returns the top_features dict."""
    sens, seed = settings["sensitivity"], int(settings["seed"])
    out_dir = Path(out_dir)
    test = load_split("test", data_dir).reset_index(drop=True)
    train = load_split("train", data_dir)
    sample = test.sample(n=min(int(sens["shap_sample_size"]), len(test)), random_state=seed)
    background = train.sample(n=min(int(sens["shap_background_size"]), len(train)), random_state=seed)

    explainer = PipelineExplainer(model, background)
    shap_values, base = explainer.explain(sample)
    ranking = rank_features(shap_values)
    drivers = list(sens["expected_drivers"])
    check = signal_check(ranking, drivers, int(sens["driver_top_n"]))
    if check["matches"]:
        logger.info("Signal check passed: every documented driver is in the top %d (%s)", check["top_n"],
                    ", ".join(drivers))
    else:
        logger.warning("Signal check: documented driver(s) %s are not in the top %d SHAP features (ranks: %s). "
                       "The model may not have learned the documented denial signal.", check["missing_from_top_n"],
                       check["top_n"], {d: check["driver_ranks"][d] for d in check["missing_from_top_n"]})

    summary_plot(shap_values, sample, base, explainer.output_space, out_dir / "shap_summary.png")
    importance_plot(ranking, drivers, explainer.output_space, out_dir / "shap_importance.png")

    threshold, threshold_source = resolve_threshold(model, settings, data_dir, validation_report)
    scores = predict_scores(model, test)
    local = []
    cases = pick_local_cases(test, scores, threshold)
    for case in LOCAL_CASES:
        if case not in cases:
            logger.warning("No %s in the test split at threshold %.4f; its waterfall plot is skipped", case, threshold)
            local.append({"case": case, "found": False})
            continue
        position = cases[case]
        rows = test.iloc[[position]]
        row_shap, row_base = explainer.explain(rows)
        row_shap = row_shap.iloc[0]
        plot = f"waterfall_{case}.png"
        title = (f"{case.replace('_', ' ')}: swipe {rows['swipe_id'].iloc[0]}, "
                 f"score {scores[position]:.3f} vs threshold {threshold:.3f}")
        waterfall_plot(row_shap, rows.iloc[0], row_base, title, out_dir / plot)
        top = row_shap.reindex(row_shap.abs().sort_values(ascending=False).index).head(5)
        local.append({"case": case, "found": True, "swipe_id": str(rows["swipe_id"].iloc[0]),
                      "score": float(scores[position]), "denied": int(rows[LABEL_COLUMN].iloc[0]), "plot": plot,
                      "top_contributions": [{"feature": f, "shap": float(v)} for f, v in top.items()]})

    result = {
        "model_name": metadata.get("model_name"),
        "estimator": type(explainer.estimator).__name__,
        "explainer": explainer.explainer_name,
        "output_space": explainer.output_space,
        "base_value": base,
        "evaluated_split": "test",
        "n_samples": int(len(sample)),
        "seed": seed,
        "features": [{"rank": i + 1, "feature": f, "mean_abs_shap": float(v), "expected_driver": f in drivers,
                      "type": "categorical" if f in CATEGORICAL_FEATURES else "numeric"}
                     for i, (f, v) in enumerate(ranking.items())],
        "signal_check": check,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "local_explanations": local,
        "plots": ["shap_summary.png", "shap_importance.png"] + [c["plot"] for c in local if c["found"]],
    }
    write_json(result, out_dir / TOP_FEATURES_FILE)
    return result


# ------------------------------------------------------------------------- hyperparameter sensitivity

def tracking_uri(settings: dict) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or settings["mlflow"].get("tracking_uri") or DEFAULT_TRACKING_URI


def fetch_tuning_runs(settings: dict) -> tuple[pd.DataFrame, str | None]:
    """Tuning trials from MLflow as rows (run_id, family, metric, param columns); (frame, problem message)."""
    sens = settings["sensitivity"]
    uri = tracking_uri(settings)
    if uri.startswith("sqlite:///") and not Path(uri.removeprefix("sqlite:///")).exists():
        return pd.DataFrame(), f"No MLflow store at {uri}; run training (python -m nsh_pipeline.modeling.train) first"
    experiment_name = settings["mlflow"].get("experiment", DEFAULT_EXPERIMENT)
    try:
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame(), f"MLflow experiment {experiment_name!r} not found at {uri}"
        runs, token = [], None
        while True:
            page = client.search_runs([experiment.experiment_id], filter_string=sens["tuning_run_filter"],
                                      max_results=1000, page_token=token)
            runs.extend(page)
            token = page.token
            if not token:
                break
    except Exception as error:  # noqa: BLE001 - an unreachable store must not crash the stage
        return pd.DataFrame(), f"Could not read tuning runs from MLflow ({uri}): {error}"
    metric = sens["tuning_metric"]
    rows = []
    for run in runs:
        family = run.data.tags.get("model_name") or run.data.params.get("model_name")
        if family is None or metric not in run.data.metrics:
            continue
        rows.append({"run_id": run.info.run_id, "family": family, "metric": run.data.metrics[metric],
                     **{f"param.{k}": v for k, v in run.data.params.items() if k != "model_name"}})
    if not rows:
        return pd.DataFrame(), (f"No tuning runs with metric {metric!r} in experiment {experiment_name!r} "
                                f"(filter: {sens['tuning_run_filter']})")
    return pd.DataFrame(rows), None


def typed_params(frame: pd.DataFrame) -> pd.DataFrame:
    """Hyperparameter columns of one family: numeric where every value parses, else text. Unused params dropped."""
    out = {}
    for column in [c for c in frame.columns if c.startswith("param.")]:
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        name = column.removeprefix("param.")
        out[name] = numeric if numeric.notna().all() else values.astype(str)
    return pd.DataFrame(out, index=frame.index)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _use_log_axis(series: pd.Series, spec: dict | None) -> bool:
    if spec and "log" in spec:
        return bool(spec["log"])
    values = series.dropna()
    return bool(len(values) and values.min() > 0 and values.max() / values.min() >= 100)


def correlation_ratio(categories: pd.Series, metric: pd.Series) -> float:
    """Eta: share of the metric's spread explained by a categorical hyperparameter (0-1)."""
    total = ((metric - metric.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    between = sum(len(g) * (g.mean() - metric.mean()) ** 2 for _, g in metric.groupby(categories))
    return float(np.sqrt(between / total))


def hyperparameter_importance(params: pd.DataFrame, metric: pd.Series, seed: int, min_runs_for_forest: int) -> dict:
    """Importance per hyperparameter, normalised to sum 1, with the method used."""
    varying = [c for c in params.columns if params[c].nunique(dropna=True) > 1]
    constant = [c for c in params.columns if c not in varying]
    stats = {}
    for column in varying:
        if _is_numeric(params[column]):
            mask = params[column].notna()
            rho = spearmanr(params.loc[mask, column], metric[mask]).statistic
            stats[column] = {"type": "numeric", "spearman_rho": None if np.isnan(rho) else float(rho)}
        else:
            stats[column] = {"type": "categorical", "correlation_ratio": correlation_ratio(params[column], metric)}
    if not varying:
        return {"method": "none", "importance": [], "constant_params": constant,
                "message": "No hyperparameter varies across these runs"}

    if len(metric) >= min_runs_for_forest:
        X = pd.DataFrame({c: params[c] if _is_numeric(params[c]) else params[c].astype("category").cat.codes
                          for c in varying}).fillna(-1)
        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed)
        forest.fit(X, metric)
        perm = permutation_importance(forest, X, metric, n_repeats=20, random_state=seed)
        raw = dict(zip(varying, np.clip(perm.importances_mean, 0, None)))
        method = "random_forest_permutation"
    else:
        raw = {c: abs(s["spearman_rho"] or 0.0) if s["type"] == "numeric" else s["correlation_ratio"]
               for c, s in stats.items()}
        method = "spearman_abs_rho"
    total = sum(raw.values())
    ranked = sorted(raw, key=lambda c: -raw[c])
    importance = [{"param": c, "importance": float(raw[c] / total) if total else 0.0,
                   "raw_importance": float(raw[c]), **stats[c]} for c in ranked]
    return {"method": method, "importance": importance, "constant_params": constant}


def family_plot(family: str, params: pd.DataFrame, metric: pd.Series, metric_name: str, space: dict, path) -> Path:
    """One panel per hyperparameter: scatter (numeric, log axis where appropriate) or box (categorical)."""
    columns = list(params.columns)
    ncols = min(3, len(columns))
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    best = metric.idxmax()
    for ax, column in zip(axes.flat, columns):
        series = params[column]
        if _is_numeric(series):
            ax.scatter(series, metric, color=BAR, alpha=0.7, s=22)
            ax.scatter(series[best], metric[best], color=MARK, marker="*", s=140, zorder=3, label="best run")
            if _use_log_axis(series, space.get(column)):
                ax.set_xscale("log")
        else:
            groups = sorted(series.dropna().unique())
            ax.boxplot([metric[series == g].to_numpy() for g in groups], tick_labels=groups)
            ax.scatter(groups.index(series[best]) + 1, metric[best], color=MARK, marker="*", s=140, zorder=3,
                       label="best run")
        ax.set(xlabel=column, ylabel=metric_name)
    for ax in list(axes.flat)[len(columns):]:
        ax.axis("off")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{family}: {metric_name} vs hyperparameters ({len(metric)} tuning runs)")
    fig.tight_layout()
    return _save(fig, path)


def importance_chart(families: dict, path) -> Path | None:
    ok = {f: r for f, r in families.items() if r.get("status") == "ok" and r["importance"]}
    if not ok:
        return None
    fig, axes = plt.subplots(1, len(ok), figsize=(4.5 * len(ok), 3.6), squeeze=False)
    for ax, (family, result) in zip(axes.flat, ok.items()):
        items = result["importance"][::-1]
        ax.barh([i["param"] for i in items], [i["importance"] for i in items], color=BAR)
        ax.set(title=f"{family}\n({result['method']})", xlabel="relative importance")
    fig.tight_layout()
    return _save(fig, path)


def run_hyperparameter_sensitivity(settings: dict, out_dir) -> dict:
    """Plots + hyperparameter_importance.json from the MLflow tuning runs (graceful when there are none)."""
    sens, out_dir = settings["sensitivity"], Path(out_dir)
    runs, problem = fetch_tuning_runs(settings)
    result = {"metric": sens["tuning_metric"], "experiment": settings["mlflow"].get("experiment", DEFAULT_EXPERIMENT),
              "run_filter": sens["tuning_run_filter"], "n_runs": int(len(runs)), "families": {}}
    if problem:
        logger.warning("Hyperparameter sensitivity skipped: %s", problem)
        result.update({"status": "no_runs", "message": problem})
        write_json(result, out_dir / HP_IMPORTANCE_FILE)
        return result

    for family, group in runs.groupby("family", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < int(sens["min_runs"]):
            message = f"{len(group)} run(s); at least {sens['min_runs']} are needed"
            logger.warning("Hyperparameter sensitivity for %s skipped: %s", family, message)
            result["families"][family] = {"status": "too_few_runs", "n_runs": int(len(group)), "message": message}
            continue
        params, metric = typed_params(group), group["metric"].astype(float)
        importance = hyperparameter_importance(params, metric, int(settings["seed"]), int(sens["min_runs_for_forest"]))
        plot = f"hyperparameter_sensitivity_{family}.png"
        family_plot(family, params, metric, sens["tuning_metric"], settings["search_space"].get(family) or {},
                    out_dir / plot)
        best = int(metric.idxmax())
        result["families"][family] = {
            "status": "ok", "n_runs": int(len(group)), **importance, "plot": plot,
            "best_run": {"run_id": group.loc[best, "run_id"], "metric": float(metric[best]),
                         "params": {c: _display_value(params.loc[best, c]) for c in params.columns}},
            "metric_range": [float(metric.min()), float(metric.max())],
        }
        top = importance["importance"][0]["param"] if importance["importance"] else "none"
        logger.info("%s: %d tuning runs, most influential hyperparameter = %s (%s)", family, len(group), top,
                    importance["method"])

    chart = importance_chart(result["families"], out_dir / "hyperparameter_importance.png")
    ok = any(r["status"] == "ok" for r in result["families"].values())
    result["status"] = "ok" if ok else "too_few_runs"
    result["message"] = None if ok else "No model family has enough tuning runs"
    result["plots"] = [r["plot"] for r in result["families"].values() if "plot" in r] + ([chart.name] if chart else [])
    write_json(result, out_dir / HP_IMPORTANCE_FILE)
    return result


# ---------------------------------------------------------------------------------------------- run

def run_sensitivity(params: dict, model_dir="models/best", data_dir="data/training", out_dir=None,
                    validation_report=None, use_mlflow: bool | None = None) -> dict:
    """SHAP + hyperparameter sensitivity; writes every output to out_dir and logs it to MLflow."""
    settings = sensitivity_settings(params)
    if use_mlflow is not None:
        settings["mlflow"]["enabled"] = use_mlflow
    out_dir = Path(out_dir or settings["sensitivity"]["out_dir"])
    if validation_report is None:
        validation_report = Path(settings["out_dir"]) / REPORT_FILE
    model, metadata = load_model(model_dir)
    card = load_dataset_card(data_dir) if (Path(data_dir) / DATASET_CARD).exists() else {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(X does not have valid feature names|n_jobs).*")
        hp = run_hyperparameter_sensitivity(settings, out_dir)   # read tuning runs before logging our own
        tags = run_tags(metadata, card, data_dir)
        with mlflow_run(settings, "sensitivity", {**tags, "stage": "sensitivity"}) as run:
            top = run_shap(model, metadata, settings, data_dir, out_dir, validation_report)
            if run:
                mlflow.log_params({"shap_sample_size": top["n_samples"], "explainer": top["explainer"],
                                   "driver_top_n": top["signal_check"]["top_n"], "tuning_metric": hp["metric"]})
                mlflow.log_metrics({"signal_drivers_in_top_n": len(top["signal_check"]["found_in_top_n"]),
                                    "tuning_runs_analysed": hp["n_runs"]})
                mlflow.set_tags({"signal_check_matches": str(top["signal_check"]["matches"]).lower(),
                                 "hyperparameter_status": hp["status"]})
                mlflow.log_artifacts(str(out_dir), artifact_path="sensitivity")
    logger.info("Top features: %s", ", ".join(f["feature"] for f in top["features"][:5]))
    logger.info("Sensitivity outputs written to %s", out_dir)
    return {"top_features": top, "hyperparameters": hp}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="SHAP and hyperparameter sensitivity analysis for models/best.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-dir", default="models/best")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--out-dir", default=None, help="default: params.yaml -> model_validation.sensitivity.out_dir")
    parser.add_argument("--validation-report", default=None,
                        help="validation.json with the chosen threshold (default: <model_validation.out_dir>/"
                             "validation.json; chosen again on the validation split if missing)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging (tuning runs are still read)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_sensitivity(load_params(args.params), args.model_dir, args.data_dir, args.out_dir,
                             args.validation_report, use_mlflow=False if args.no_mlflow else None)
    print(json.dumps({"top_features": [f["feature"] for f in result["top_features"]["features"][:5]],
                      "signal_check_matches": result["top_features"]["signal_check"]["matches"],
                      "hyperparameter_status": result["hyperparameters"]["status"]}, indent=2))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
