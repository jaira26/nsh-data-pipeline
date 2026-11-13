"""End-to-end: one command trains, tunes and selects on the sample, tracked in a throwaway MLflow store."""
import json

import mlflow
import pandas as pd
import pytest

from nsh_pipeline.modeling import train as train_module
from nsh_pipeline.modeling.artifacts import METADATA_KEYS, load_model
from nsh_pipeline.modeling.candidates import MODEL_NAMES
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, load_split

N_TRIALS = 2


@pytest.fixture(scope="module")
def trained(training_sample_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlflow.db")
        summary = train_module.main([
            "--data-dir", str(training_sample_dir), "--params", "params.yaml", "--n-trials", str(N_TRIALS),
            "--models-dir", str(root / "models"), "--reports-dir", str(root / "reports"),
        ])
        runs = mlflow.search_runs(experiment_names=["nsh-denial-model"], output_format="pandas")
    return {"root": root, "summary": summary, "runs": runs}


def test_every_candidate_and_best_follow_the_model_contract(trained, training_sample_dir):
    test = load_split("test", training_sample_dir)
    for folder in [*(trained["root"] / "models" / "candidates" / n for n in MODEL_NAMES),
                   trained["root"] / "models" / "best"]:
        model, metadata = load_model(folder)
        assert set(METADATA_KEYS) <= set(metadata)
        assert metadata["data_version"] == "sample" and metadata["mlflow_run_id"]
        assert model.predict_proba(test[FEATURE_COLUMNS]).shape == (len(test), 2)


def test_best_is_the_top_validation_pr_auc(trained):
    summary = json.loads((trained["root"] / "reports" / "candidates.json").read_text())
    scores = {name: c["validation_metrics"]["pr_auc"] for name, c in summary["candidates"].items()}
    assert summary["selected_model"] == max(scores, key=scores.get) == trained["summary"]["selected_model"]
    _, best = load_model(trained["root"] / "models" / "best")
    assert best["model_name"] == summary["selected_model"]


def test_mlflow_has_every_run_with_required_tags(trained):
    runs = trained["runs"]
    assert len(runs) == 1 + len(MODEL_NAMES) * (1 + N_TRIALS)  # selection + candidates + trials
    assert runs["tags.data_version"].eq("sample").all()
    assert runs["tags.git_sha"].notna().all() and runs["tags.git_sha"].ne("").all()
    assert (runs["tags.stage"] == "tuning_trial").sum() == len(MODEL_NAMES) * N_TRIALS
    candidates = runs[runs["tags.stage"] == "candidate"]
    assert candidates["metrics.val_pr_auc"].notna().all() and candidates["metrics.cv_pr_auc"].notna().all()
    selection = runs[runs["tags.stage"] == "selection"].iloc[0]
    assert selection["tags.selected_model"] == trained["summary"]["selected_model"]


def test_charts_and_trial_tables_are_written(trained):
    reports = trained["root"] / "reports"
    for name in ("model_comparison.png", "confusion_matrices.png", "pr_curves.png"):
        assert (reports / name).stat().st_size > 1000
    for name in MODEL_NAMES:
        assert len(pd.read_csv(reports / "trials" / f"{name}.csv")) == N_TRIALS


def test_stub_cli(training_sample_dir, tmp_path):
    assert train_module.main(["--stub", "--data-dir", str(training_sample_dir),
                              "--models-dir", str(tmp_path)])["stub"] is True
    assert (tmp_path / "best" / "model.joblib").exists()


def test_sample_weights_map_by_swipe_and_default_to_one(training_sample_dir, tmp_path):
    train = load_split("train", training_sample_dir)
    pd.DataFrame({"swipe_id": [train["swipe_id"].iloc[0]], "weight": [3.0]}).to_parquet(tmp_path / "w.parquet")
    weights = train_module.load_sample_weights(tmp_path / "w.parquet", train)
    assert weights.iloc[0] == 3.0 and (weights.iloc[1:] == 1.0).all() and len(weights) == len(train)
    assert train_module.load_sample_weights(None, train) is None


def test_weighted_training_run(training_sample_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    train = load_split("train", training_sample_dir)
    pd.DataFrame({"swipe_id": train["swipe_id"], "weight": 1.0}).to_parquet(tmp_path / "w.parquet")
    params = {"seed": 1, "model": {"candidates": ["logistic_regression"], "cv_folds": 3, "n_trials": 1,
                                   "mlflow": {"log_models": False}}}
    summary = train_module.run_training(params, training_sample_dir, sample_weights=str(tmp_path / "w.parquet"),
                                        models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    _, metadata = load_model(tmp_path / "models" / "best")
    assert summary["selected_model"] == "logistic_regression" and metadata["sample_weighted"] is True


def test_unknown_candidate_in_params_is_rejected():
    with pytest.raises(ValueError, match="Unknown candidate"):
        train_module.model_settings({"model": {"candidates": ["svm"]}})
