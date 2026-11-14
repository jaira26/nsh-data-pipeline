"""Model-bias stage end to end: audit, reweighing mitigation, fairness-aware selection, report and CLI.

Offline and fast: a synthetic training set is bootstrapped from the sample fixture rows (optionally with a
planted denial skew in one state), and small untuned models stand in for the tuned candidates.
"""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.modeling import fairness
from nsh_pipeline.modeling.artifacts import METADATA_KEYS, candidate_dir, load_model, save_model, utc_now
from nsh_pipeline.modeling.candidates import build_pipeline, features
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, LABEL_COLUMN, SPLITS, load_split, split_path
from nsh_pipeline.modeling.fairness_mitigation import mitigated_name, reweigh_and_refit, slice_groups
from nsh_pipeline.modeling.fairness_report import REPORT_MD, SCATTER_PNG, SLICE_CSV, ordered_groups

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "training"
SKEW_STATE = "CA"


def fair_labels(frame: pd.DataFrame, rng) -> np.ndarray:
    """Denials from documented drivers only (amount, category, recent denials, tenure), as in data-v1.1:
    no slice column has any effect, so a model trained on these labels should be fair by construction."""
    category = frame["service_category"].map({"Dental": 0.9, "Vision": 0.7}).fillna(0.0)
    logit = (-2.6 + 0.6 * np.log(frame["amount_usd"].clip(lower=1) / 75) + category
             + 0.7 * frame["prior_denials_90d"].clip(upper=3) + 0.8 * frame["tenure_days"].lt(180))
    return (rng.random(len(frame)) < 1 / (1 + np.exp(-logit))).astype(int)


def build_training_set(out_dir: Path, skew_rate: float = 0.0, sizes=(3000, 1500, 1500), seed: int = 7) -> Path:
    """Bootstrap the fixture rows into bigger splits with fair labels; skew_rate > 0 then flips that share
    of CA swipes to denied (the planted bias, like inject_issues.denial_skew)."""
    rng = np.random.default_rng(seed)
    pool = pd.concat([load_split(name, FIXTURE_DIR) for name in SPLITS], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, size in zip(SPLITS, sizes):
        frame = pool.sample(n=size, replace=True, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
        frame["swipe_id"] = [f"{name}-{i}" for i in range(size)]
        frame["amount_usd"] = frame["amount_usd"] * rng.uniform(0.9, 1.1, size)
        frame[LABEL_COLUMN] = fair_labels(frame, rng)
        if skew_rate:
            flip = frame["state"].eq(SKEW_STATE).to_numpy() & (rng.random(size) < skew_rate)
            frame.loc[flip, LABEL_COLUMN] = 1
        frame.to_parquet(split_path(name, out_dir), index=False)
    card = json.loads((FIXTURE_DIR / "dataset_card.json").read_text(encoding="utf-8"))
    (out_dir / "dataset_card.json").write_text(json.dumps({**card, "data_version": "synthetic-test"}),
                                               encoding="utf-8")
    return out_dir


def save_candidate(name: str, data_dir: Path, models_dir: Path, params: dict | None = None) -> Path:
    """Fit an untuned candidate on the train split and save it in the model contract layout."""
    train = load_split("train", data_dir)
    model = build_pipeline(name, params or {}, 42).fit(features(train), train[LABEL_COLUMN].astype(int))
    metadata = {"model_name": name, "params": params or {}, "cv_metrics": {}, "mlflow_run_id": None,
                "data_version": "synthetic-test", "git_sha": "test", "created_at": utc_now()}
    return save_model(model, metadata, candidate_dir(name, models_dir))


def make_params(tmp_path: Path, mlflow_enabled: bool = False, inject: bool = False, **model_bias) -> dict:
    params = {
        "seed": 42,
        "model": {"seed": 42, "selection_metric": "pr_auc",
                  "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test-model-bias"}},
        "model_bias": {"mlflow": {"enabled": mlflow_enabled}, **model_bias},
    }
    if inject:
        params["inject_issues"] = {"enabled": True, "denial_skew": {"state": SKEW_STATE, "rate": 0.6}}
    return params


@pytest.fixture(autouse=True)
def _no_tracking_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


@pytest.fixture(scope="module")
def clean_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    data_dir = build_training_set(root / "training")
    save_candidate("logistic_regression", data_dir, root / "models")
    return data_dir, root / "models"


@pytest.fixture(scope="module")
def skewed_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("skewed")
    data_dir = build_training_set(root / "training", skew_rate=0.6)
    save_candidate("logistic_regression", data_dir, root / "models")
    save_candidate("random_forest", data_dir, root / "models", {"n_estimators": 60, "max_depth": 6})
    return data_dir, root / "models"


def run(params, data_dir, models_dir, tmp_path):
    return fairness.run_model_bias(params, data_dir, models_dir, tmp_path / "reports", tmp_path / "mitigated")


def test_clean_data_passes_and_writes_every_output(clean_setup, tmp_path):
    data_dir, models_dir = clean_setup
    report = run(make_params(tmp_path), data_dir, models_dir, tmp_path)

    saved = json.loads((tmp_path / "reports" / "bias.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(report))
    assert all(field in saved for field in fairness.CONTRACT_FIELDS)
    assert saved["passed"] is True
    assert saved["selected_model"] == "logistic_regression"
    assert saved["selected_model_dir"] == str(candidate_dir("logistic_regression", models_dir))
    assert saved["mitigation_applied"] is None
    assert saved["audit_split"] == "test" and saved["selection_split"] == "validation"
    assert set(saved["max_disparity"]) == set(saved["gate"]["slice_columns"])
    assert {r["slice"] for r in saved["per_slice_metrics"]} == set(saved["gate"]["slice_columns"])
    assert not (tmp_path / "mitigated" / "candidates").exists()

    out = tmp_path / "reports" / "bias"
    for file in (REPORT_MD, SLICE_CSV, SCATTER_PNG, *(f"slices_{c}.png" for c in saved["gate"]["slice_columns"])):
        assert (out / file).stat().st_size > 0, file
    markdown = (out / REPORT_MD).read_text(encoding="utf-8")
    assert "Fairness gate passed" in markdown and "no mitigation was needed" in markdown
    csv = pd.read_csv(out / SLICE_CSV)
    assert set(csv["split"]) == {"validation", "test"} and set(csv["candidate"]) == {"logistic_regression"}


def test_vip_slice_is_too_small_to_judge(clean_setup, tmp_path):
    data_dir, models_dir = clean_setup
    report = run(make_params(tmp_path), data_dir, models_dir, tmp_path)
    vip = [r for r in report["per_slice_metrics"] if r["slice"] == "is_vip" and r["group"] == "True"]
    assert all(set(r["too_small_for"]) == {"false_positive_rate", "recall"} for r in vip)


def test_planted_skew_is_detected_mitigated_and_a_fair_model_selected(skewed_setup, tmp_path):
    data_dir, models_dir = skewed_setup
    report = run(make_params(tmp_path, inject=True), data_dir, models_dir, tmp_path)

    for base in ("logistic_regression", "random_forest"):
        validation = report["candidates"][base]["validation"]
        assert not validation["passed"]
        assert any(f["slice"] == "state" and f["group"] == SKEW_STATE for f in validation["flags"]), base
        assert mitigated_name(base) in report["candidates"]
        assert report["candidates"][mitigated_name(base)]["mitigation"]["slice"] == "state"

    selected = report["selected_model"]
    assert selected.endswith("_reweighed")
    assert report["passed"] is True
    assert report["mitigation_applied"]["method"] == "reweighing"
    assert report["mitigation_applied"]["train_denial_gap_after_pts"] == pytest.approx(0.0, abs=1e-6)
    assert report["ranking"].index(selected) > 0  # a better-scoring but unfair model was passed over
    assert "failed it" in report["selection_reason"]

    model, metadata = load_model(report["selected_model_dir"])
    assert all(key in metadata for key in METADATA_KEYS)
    assert metadata["base_model"] == selected.removesuffix("_reweighed")
    test = load_split("test", data_dir)
    assert model.predict_proba(test[FEATURE_COLUMNS]).shape == (len(test), 2)

    markdown = (tmp_path / "reports" / "bias" / REPORT_MD).read_text(encoding="utf-8")
    assert "Planted bias check" in markdown and "not detected" not in markdown
    assert "Trade-offs" in markdown
    assert (tmp_path / "reports" / "bias" / f"mitigation_{selected}.png").exists()


def test_mitigation_closes_the_planted_gap(skewed_setup, tmp_path):
    data_dir, models_dir = skewed_setup
    report = run(make_params(tmp_path), data_dir, models_dir, tmp_path)
    before = report["candidates"]["logistic_regression"]["validation"]["worst_gap_pts"]["false_positive_rate"]
    after = report["candidates"]["logistic_regression_reweighed"]["validation"]["worst_gap_pts"]["false_positive_rate"]
    assert before["group"] == SKEW_STATE
    assert after["gap_pts"] < before["gap_pts"] / 2


def test_gate_failure_exits_1_after_writing_the_report(skewed_setup, tmp_path, monkeypatch):
    data_dir, models_dir = skewed_setup
    params = make_params(tmp_path, mitigation={"enabled": False})
    monkeypatch.setattr(fairness, "load_params", lambda path: params)
    argv = ["--data-dir", str(data_dir), "--models-dir", str(models_dir), "--out-dir", str(tmp_path / "reports"),
            "--mitigated-dir", str(tmp_path / "mitigated")]
    with pytest.raises(SystemExit) as exit_info:
        fairness.main(argv)
    assert exit_info.value.code == 1
    saved = json.loads((tmp_path / "reports" / "bias.json").read_text(encoding="utf-8"))
    assert saved["selected_model"] is None and saved["passed"] is False
    assert saved["audited_model"] == saved["ranking"][0]
    assert any(f["group"] == SKEW_STATE for f in saved["flags"])
    assert "no candidate passes" in (tmp_path / "reports" / "bias" / REPORT_MD).read_text(encoding="utf-8")

    report = fairness.main([*argv, "--no-fail"])  # exploratory runs can opt out of the exit code
    assert report["passed"] is False


def test_cli_success_returns_report(clean_setup, tmp_path, monkeypatch, capsys):
    data_dir, models_dir = clean_setup
    monkeypatch.setattr(fairness, "load_params", lambda path: make_params(tmp_path))
    report = fairness.main(["--data-dir", str(data_dir), "--models-dir", str(models_dir),
                            "--out-dir", str(tmp_path / "reports"), "--mitigated-dir", str(tmp_path / "m")])
    assert report["passed"]
    assert json.loads(capsys.readouterr().out)["selected_model"] == "logistic_regression"


def test_mlflow_run_is_logged_with_tags_metrics_and_artifacts(skewed_setup, tmp_path):
    data_dir, models_dir = skewed_setup
    report = run(make_params(tmp_path, mlflow_enabled=True), data_dir, models_dir, tmp_path)
    assert report["mlflow_run_id"]
    run_info = mlflow.get_run(report["mlflow_run_id"])
    assert run_info.data.tags["stage"] == "model_bias"
    assert run_info.data.tags["data_version"] == "synthetic-test"
    assert run_info.data.tags["selected_model"] == report["selected_model"]
    assert run_info.data.metrics["fairness_passed"] == 1.0
    assert "logistic_regression.validation.passed" in run_info.data.metrics
    artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(report["mlflow_run_id"], "model_bias")}
    assert "model_bias/bias.json" in artifacts
    _, metadata = load_model(report["selected_model_dir"])
    assert metadata["mlflow_run_id"] == report["mlflow_run_id"]


def test_rerun_removes_stale_mitigated_models(skewed_setup, clean_setup, tmp_path):
    run(make_params(tmp_path), *skewed_setup, tmp_path)
    stale = candidate_dir("logistic_regression_reweighed", tmp_path / "mitigated")
    assert (stale / "model.joblib").exists()
    run(make_params(tmp_path), *clean_setup, tmp_path)  # clean data: nothing to mitigate
    assert not (stale / "model.joblib").exists()


def test_settings_defaults_and_validation():
    model = {"selection_metric": "roc_auc", "seed": 7, "mlflow": {"experiment": "x"}}
    settings = fairness.bias_settings({"model": model})
    assert settings["selection_metric"] == "roc_auc" and settings["seed"] == 7
    assert settings["mlflow"] == {"experiment": "x", "enabled": True}
    assert settings["planted_skew"] is None
    assert settings["max_gap_pts"]["false_positive_rate"] == 20
    with pytest.raises(ValueError, match="Unknown metric"):
        fairness.bias_settings({"model_bias": {"max_gap_pts": {"accuracy": 5}}})
    with pytest.raises(ValueError, match="at least one metric"):
        fairness.bias_settings({"model_bias": {"max_gap_pts": {"recall": None}}})
    with pytest.raises(ValueError, match="mitigation method"):
        fairness.bias_settings({"model_bias": {"mitigation": {"method": "threshold_optimizer"}}})


def test_missing_models_or_slice_columns_fail_clearly(clean_setup, tmp_path):
    data_dir, models_dir = clean_setup
    with pytest.raises(FileNotFoundError, match="train stage"):
        run(make_params(tmp_path), data_dir, tmp_path / "no_models", tmp_path)
    with pytest.raises(KeyError, match="employee_status"):
        run(make_params(tmp_path, slice_columns=["state", "employee_status"]), data_dir, models_dir, tmp_path)


def test_rank_prefers_original_then_simpler_model_on_ties():
    def result(base, mitigated, pr_auc=0.3):
        return {"base_model": base, "mitigation": {"slice": "state"} if mitigated else None,
                "validation": {"performance": {"pr_auc": pr_auc}}}
    results = {"lightgbm": result("lightgbm", False), "random_forest_reweighed": result("random_forest", True),
               "logistic_regression": result("logistic_regression", False),
               "random_forest": result("random_forest", False, pr_auc=0.31)}
    assert fairness.rank(results, "pr_auc") == ["random_forest", "logistic_regression", "lightgbm",
                                                "random_forest_reweighed"]


def test_reweigh_and_refit_equalises_training_rates_and_rejects_bad_input(skewed_setup):
    data_dir, _ = skewed_setup
    train = load_split("train", data_dir)
    model, info = reweigh_and_refit("logistic_regression", {}, train, "state", seed=42)
    assert model.predict_proba(train[FEATURE_COLUMNS].head()).shape == (5, 2)
    after = {group: rates["after"] for group, rates in info["train_denial_rate_by_group"].items()}
    assert max(after.values()) - min(after.values()) < 1e-9
    assert info["train_denial_gap_before_pts"] > 20 and 0 < info["effective_share"] <= 1
    with pytest.raises(ValueError, match="unknown model"):
        reweigh_and_refit("svm", {}, train, "state")
    with pytest.raises(KeyError, match="employee_status"):
        reweigh_and_refit("logistic_regression", {}, train, "employee_status")


def test_slice_groups_and_group_order():
    assert slice_groups(pd.DataFrame({"s": ["a", None]}), "s").tolist() == ["a", "Unknown"]
    assert ordered_groups("group_size_bucket", ["250+", "1-9", "100-249", "10-24"]) == ["1-9", "10-24", "100-249",
                                                                                         "250+"]
    assert ordered_groups("state", ["WA", "CA"]) == ["CA", "WA"]
