"""Sensitivity analysis: SHAP on Pipelines (linear, tree, permutation), one-hot aggregation, signal check,
hyperparameter sensitivity from a throwaway MLflow store, and graceful handling of missing runs."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from nsh_pipeline.modeling import sensitivity
from nsh_pipeline.modeling.artifacts import load_model
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, load_split

from .conftest import fit_fixture_pipeline


# ------------------------------------------------------------------------------------- aggregation

@pytest.mark.parametrize("name, expected", [
    ("amount_usd", "amount_usd"),
    ("service_category_Dental", "service_category"),
    ("service_category_Drugs & Medicine", "service_category"),
    ("numeric__tenure_days", "tenure_days"),
    ("categorical__state_CA", "state"),
    ("is_debit_card_True", "is_debit_card"),
    ("something_else", "something_else"),
])
def test_original_feature(name, expected):
    assert sensitivity.original_feature(name) == expected


def test_original_feature_prefers_longest_match():
    assert sensitivity.original_feature("plan_type_HDHP", ["plan", "plan_type"]) == "plan_type"


def test_aggregate_shap_sums_one_hot_columns():
    values = np.array([[1.0, 2.0, 3.0, 0.5], [-1.0, 0.5, -0.25, 0.0]])
    names = ["service_category_Dental", "amount_usd", "service_category_Vision", "state_CA"]
    out = sensitivity.aggregate_shap(values, names)
    assert list(out.columns) == ["amount_usd", "service_category", "state"]     # contract order
    np.testing.assert_allclose(out["service_category"], [4.0, -1.25])
    np.testing.assert_allclose(out["amount_usd"], [2.0, 0.5])
    np.testing.assert_allclose(out.sum(axis=1), values.sum(axis=1))           # additivity kept


@pytest.mark.parametrize("estimator, kind", [
    (LogisticRegression(), "linear"), (RandomForestClassifier(), "tree"), (LGBMClassifier(), "tree"),
    (KNeighborsClassifier(), "permutation"),
])
def test_explainer_kind(estimator, kind):
    assert sensitivity.explainer_kind(estimator) == kind


def test_split_pipeline_requires_a_pipeline():
    with pytest.raises(TypeError, match="Pipeline"):
        sensitivity.split_pipeline(LogisticRegression())


# ------------------------------------------------------------------------------------------ SHAP

def test_linear_shap_is_additive_in_log_odds(linear_model_dir, contract_data_dir):
    model, _ = load_model(linear_model_dir)
    train, test = load_split("train", contract_data_dir), load_split("test", contract_data_dir).head(40)
    explainer = sensitivity.PipelineExplainer(model, train.sample(100, random_state=0))
    values, base = explainer.explain(test)
    assert explainer.explainer_name == "LinearExplainer" and explainer.output_space == "log-odds"
    assert list(values.columns) == FEATURE_COLUMNS            # one column per ORIGINAL feature
    log_odds = model.decision_function(test[FEATURE_COLUMNS])
    # LinearExplainer is exact against the background mean, so base + sum(SHAP) recovers the log-odds.
    np.testing.assert_allclose(values.sum(axis=1) + base, log_odds, atol=1e-6)


def test_tree_shap_is_additive_in_probability(tree_model_dir, contract_data_dir):
    model, _ = load_model(tree_model_dir)
    train, test = load_split("train", contract_data_dir), load_split("test", contract_data_dir).head(40)
    explainer = sensitivity.PipelineExplainer(model, train.head(50))
    values, base = explainer.explain(test)
    assert explainer.explainer_name == "TreeExplainer" and explainer.output_space == "probability"
    assert list(values.columns) == FEATURE_COLUMNS
    np.testing.assert_allclose(values.sum(axis=1) + base, model.predict_proba(test[FEATURE_COLUMNS])[:, 1], atol=1e-6)


def test_permutation_explainer_for_other_estimators(contract_data_dir):
    train = load_split("train", contract_data_dir)
    model = fit_fixture_pipeline(KNeighborsClassifier(n_neighbors=25), train, verbose_names=False)
    explainer = sensitivity.PipelineExplainer(model, train.head(20))
    values, _ = explainer.explain(load_split("test", contract_data_dir).head(5))
    assert explainer.kind == "permutation" and explainer.output_space == "probability"
    assert values.shape == (5, len(FEATURE_COLUMNS))


def test_signal_check_reports_matches_and_mismatches():
    ranking = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1], index=["amount_usd", "state", "tenure_days", "carrier", "age_band"])
    check = sensitivity.signal_check(ranking, ["amount_usd", "tenure_days", "plan_deductible"], top_n=3)
    assert check["found_in_top_n"] == ["amount_usd", "tenure_days"]
    assert check["missing_from_top_n"] == ["plan_deductible"]
    assert check["unexpected_in_top_n"] == ["state"]
    assert check["driver_ranks"] == {"amount_usd": 1, "tenure_days": 3, "plan_deductible": None}
    assert check["matches"] is False
    assert sensitivity.signal_check(ranking, ["amount_usd"], top_n=1)["matches"] is True


def test_pick_local_cases():
    frame = pd.DataFrame({"denied": [1, 1, 0, 0, 1, 0]})
    scores = np.array([0.9, 0.6, 0.8, 0.1, 0.2, 0.55])
    assert sensitivity.pick_local_cases(frame, scores, 0.5) == {
        "true_positive": 0, "false_positive": 2, "false_negative": 4}
    assert "false_negative" not in sensitivity.pick_local_cases(frame.head(4), scores[:4], 0.5)


def test_resolve_threshold_from_report_or_validation(linear_model_dir, contract_data_dir, tmp_path):
    model, _ = load_model(linear_model_dir)
    settings = sensitivity.sensitivity_settings({})
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({"threshold": 0.4321}), encoding="utf-8")
    assert sensitivity.resolve_threshold(model, settings, contract_data_dir, report) == (0.4321, str(report))
    threshold, source = sensitivity.resolve_threshold(model, settings, contract_data_dir, tmp_path / "missing.json")
    assert 0 < threshold < 1 and "validation split" in source


@pytest.mark.parametrize("model_fixture", ["linear_model_dir", "tree_model_dir"])
def test_run_shap_writes_top_features_and_plots(model_fixture, request, contract_data_dir, tmp_path, caplog):
    model_dir = request.getfixturevalue(model_fixture)
    model, metadata = load_model(model_dir)
    settings = sensitivity.sensitivity_settings({"model_validation": {"sensitivity": {"shap_sample_size": 200}}})
    result = sensitivity.run_shap(model, metadata, settings, contract_data_dir, tmp_path, validation_report=None)

    saved = json.loads((tmp_path / "top_features.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(result))
    features = [f["feature"] for f in saved["features"]]
    assert sorted(features) == sorted(FEATURE_COLUMNS)             # one-hot columns aggregated back
    shap_values = [f["mean_abs_shap"] for f in saved["features"]]
    assert shap_values == sorted(shap_values, reverse=True) and saved["features"][0]["rank"] == 1
    assert saved["n_samples"] == 200 and saved["seed"] == 42
    # The fixture data has a strong documented signal: the top features are the drivers.
    assert set(features[:3]) <= set(saved["signal_check"]["expected_drivers"])
    for case in saved["local_explanations"]:
        assert case["found"] and (tmp_path / case["plot"]).stat().st_size > 0
    assert [c["case"] for c in saved["local_explanations"]] == list(sensitivity.LOCAL_CASES)
    for plot in ("shap_summary.png", "shap_importance.png"):
        assert (tmp_path / plot).stat().st_size > 0


def test_signal_mismatch_logs_a_warning(linear_model_dir, contract_data_dir, tmp_path, caplog):
    model, metadata = load_model(linear_model_dir)
    settings = sensitivity.sensitivity_settings({"model_validation": {"sensitivity": {
        "shap_sample_size": 100, "expected_drivers": ["carrier", "state"], "driver_top_n": 2}}})
    result = sensitivity.run_shap(model, metadata, settings, contract_data_dir, tmp_path)
    assert result["signal_check"]["matches"] is False
    assert "Signal check: documented driver(s)" in caplog.text


# --------------------------------------------------------------------------- hyperparameter sensitivity

def log_fake_tuning_runs(uri: str, experiment: str = "nsh-denial-model") -> None:
    """12 logistic-regression trials where C drives the metric, 5 random-forest trials (with a categorical
    param), 2 LightGBM trials (too few) and one unrelated run that must be ignored."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    rng = np.random.default_rng(0)
    trials = []
    for c in np.logspace(-3, 2, 12):
        trials.append(("logistic_regression", {"C": c, "l1_ratio": rng.random()},
                       0.15 + 0.02 * np.log10(c) + rng.normal(0, 0.001)))
    for leaf, feat in [(1, "sqrt"), (5, "log2"), (10, "0.5"), (20, "sqrt"), (30, "0.5")]:
        trials.append(("random_forest", {"min_samples_leaf": leaf, "max_features": feat, "n_estimators": 100},
                       0.2 + 0.002 * leaf))
    for lr in (0.01, 0.1):
        trials.append(("lightgbm", {"learning_rate": lr}, 0.2))
    for i, (family, params, metric) in enumerate(trials):
        with mlflow.start_run(run_name=f"{family}-trial-{i:03d}"):
            mlflow.set_tags({"model_name": family, "stage": "tuning_trial"})
            mlflow.log_params(params)
            mlflow.log_metrics({"cv_pr_auc": metric, "cv_roc_auc": 0.7})
    with mlflow.start_run(run_name="train-and-select"):
        mlflow.set_tags({"stage": "selection"})
        mlflow.log_metrics({"cv_pr_auc": 0.99})


def test_hyperparameter_sensitivity_from_mlflow(mlflow_store, tmp_path):
    log_fake_tuning_runs(mlflow_store)
    settings = sensitivity.sensitivity_settings({"model": {"search_space": {
        "logistic_regression": {"C": {"type": "float", "low": 0.001, "high": 100.0, "log": True}}}}})
    result = sensitivity.run_hyperparameter_sensitivity(settings, tmp_path / "out")

    assert result["status"] == "ok" and result["n_runs"] == 19       # the selection run is filtered out
    lr = result["families"]["logistic_regression"]
    assert lr["status"] == "ok" and lr["n_runs"] == 12
    assert lr["method"] == "random_forest_permutation"
    assert lr["importance"][0]["param"] == "C"
    assert lr["importance"][0]["spearman_rho"] > 0.9
    assert sum(i["importance"] for i in lr["importance"]) == pytest.approx(1.0)
    rf = result["families"]["random_forest"]
    assert rf["method"] == "spearman_abs_rho"                           # 5 runs < min_runs_for_forest
    assert rf["importance"][0]["param"] == "min_samples_leaf"
    assert rf["constant_params"] == ["n_estimators"]
    assert {i["param"]: i["type"] for i in rf["importance"]}["max_features"] == "categorical"
    assert result["families"]["lightgbm"]["status"] == "too_few_runs"
    saved = json.loads((tmp_path / "out" / "hyperparameter_importance.json").read_text(encoding="utf-8"))
    assert saved["families"]["logistic_regression"]["best_run"]["params"]["C"] == pytest.approx(100.0)
    for plot in result["plots"]:
        assert (tmp_path / "out" / plot).stat().st_size > 0
    assert "hyperparameter_sensitivity_logistic_regression.png" in result["plots"]
    assert "hyperparameter_importance.png" in result["plots"]


def test_zero_runs_degrades_gracefully(mlflow_store, tmp_path, caplog):
    mlflow.set_tracking_uri(mlflow_store)
    mlflow.set_experiment("nsh-denial-model")                           # experiment exists, no trials
    result = sensitivity.run_hyperparameter_sensitivity(sensitivity.sensitivity_settings({}), tmp_path)
    assert result["status"] == "no_runs" and result["families"] == {}
    assert "No tuning runs" in result["message"] and "skipped" in caplog.text
    assert json.loads((tmp_path / "hyperparameter_importance.json").read_text(encoding="utf-8"))["status"] == "no_runs"


def test_missing_store_and_experiment(tmp_path, monkeypatch, mlflow_store):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'nope.db').as_posix()}")
    result = sensitivity.run_hyperparameter_sensitivity(sensitivity.sensitivity_settings({}), tmp_path / "a")
    assert result["status"] == "no_runs" and "No MLflow store" in result["message"]
    assert not (tmp_path / "nope.db").exists()                          # reading never creates a store

    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_store)
    mlflow.set_tracking_uri(mlflow_store)
    mlflow.set_experiment("other-experiment")
    result = sensitivity.run_hyperparameter_sensitivity(sensitivity.sensitivity_settings({}), tmp_path / "b")
    assert result["status"] == "no_runs" and "not found" in result["message"]


def test_only_too_few_runs(mlflow_store, tmp_path):
    mlflow.set_tracking_uri(mlflow_store)
    mlflow.set_experiment("nsh-denial-model")
    with mlflow.start_run():
        mlflow.set_tags({"model_name": "lightgbm", "stage": "tuning_trial"})
        mlflow.log_params({"learning_rate": 0.1})
        mlflow.log_metric("cv_pr_auc", 0.2)
    result = sensitivity.run_hyperparameter_sensitivity(sensitivity.sensitivity_settings({}), tmp_path)
    assert result["status"] == "too_few_runs" and result["plots"] == []


def test_importance_when_nothing_varies():
    params = pd.DataFrame({"C": [1.0, 1.0, 1.0]})
    result = sensitivity.hyperparameter_importance(params, pd.Series([0.1, 0.2, 0.3]), seed=0, min_runs_for_forest=10)
    assert result["method"] == "none" and result["constant_params"] == ["C"]


def test_correlation_ratio():
    metric = pd.Series([1.0, 1.0, 3.0, 3.0])
    assert sensitivity.correlation_ratio(pd.Series(["a", "a", "b", "b"]), metric) == pytest.approx(1.0)
    assert sensitivity.correlation_ratio(pd.Series(["a", "b", "a", "b"]), metric) == pytest.approx(0.0)
    assert sensitivity.correlation_ratio(pd.Series(["a", "b"]), pd.Series([2.0, 2.0])) == 0.0


# -------------------------------------------------------------------------------------------- CLI

def test_cli_end_to_end_logs_to_mlflow(linear_model_dir, contract_data_dir, tmp_path, mlflow_store, write_params):
    log_fake_tuning_runs(mlflow_store)
    out = tmp_path / "sens"
    result = sensitivity.main(["--params", str(write_params()), "--model-dir", str(linear_model_dir),
                               "--data-dir", str(contract_data_dir), "--out-dir", str(out)])
    assert result["hyperparameters"]["status"] == "ok"
    assert result["top_features"]["threshold_source"].startswith("chosen on the validation split")
    for name in ("top_features.json", "hyperparameter_importance.json", "shap_summary.png", "shap_importance.png",
                 "waterfall_true_positive.png", "waterfall_false_positive.png", "waterfall_false_negative.png"):
        assert (out / name).exists()

    client = mlflow.MlflowClient(mlflow_store)
    experiment = client.get_experiment_by_name("nsh-denial-model")
    runs = client.search_runs([experiment.experiment_id], "tags.stage = 'sensitivity'")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["data_version"] == "fixture-v1" and run.data.tags["git_sha"]
    assert run.data.params["explainer"] == "LinearExplainer"
    logged = {a.path for a in client.list_artifacts(run.info.run_id, "sensitivity")}
    assert "sensitivity/top_features.json" in logged and "sensitivity/hyperparameter_importance.json" in logged


def test_cli_without_mlflow_and_without_runs(tree_model_dir, contract_data_dir, tmp_path, monkeypatch, write_params):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'absent.db').as_posix()}")
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({"threshold": 0.3}), encoding="utf-8")
    result = sensitivity.main(["--params", str(write_params()), "--model-dir", str(tree_model_dir),
                               "--data-dir", str(contract_data_dir), "--out-dir", str(tmp_path / "s"),
                               "--validation-report", str(report), "--no-mlflow"])
    assert result["hyperparameters"]["status"] == "no_runs"
    assert result["top_features"]["threshold"] == 0.3
    assert not (tmp_path / "absent.db").exists() and not (tmp_path / "mlruns").exists()
