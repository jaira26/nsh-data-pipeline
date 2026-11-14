"""Hold-out validation: metrics, threshold choice on validation only, the gate, the report contract, CLI, MLflow."""
import json

import mlflow
import numpy as np
import pytest

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling import evaluate
from nsh_pipeline.modeling.artifacts import load_model
from nsh_pipeline.modeling.dataset import load_split

# Hand-computed toy prediction. At threshold 0.5 the predicted positives are 0.9, 0.8 (denied) and 0.7 (approved):
# TP=2, FP=1, FN=1 (0.3), TN=4. ROC-AUC: 13 of 15 (positive, negative) pairs ranked right.
# PR-AUC (average precision): positives at ranks 1, 2, 5 -> (1 + 1 + 3/5) / 3.
Y_TOY = [1, 1, 1, 0, 0, 0, 0, 0]
S_TOY = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.4, 0.05]


def test_holdout_metrics_match_hand_computed_toy():
    m = evaluate.holdout_metrics(Y_TOY, np.array(S_TOY), 0.5)
    assert m["confusion_matrix"] == [[4, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(13 / 15)
    assert m["pr_auc"] == pytest.approx((1 + 1 + 3 / 5) / 3)
    assert m["positive_rate"] == pytest.approx(3 / 8)
    assert m["pr_auc_lift"] == pytest.approx(((1 + 1 + 3 / 5) / 3) / (3 / 8))
    assert m["threshold"] == 0.5


def test_holdout_metrics_need_both_classes():
    with pytest.raises(ValueError, match="both denied and approved"):
        evaluate.holdout_metrics([0, 0, 0], np.array([0.1, 0.2, 0.3]), 0.5)


def test_best_f1_threshold_on_toy():
    # F1 by cut-off: 0.9 -> 0.50, 0.8 -> 0.80, 0.7 -> 0.67, 0.4 -> 0.57, 0.3 -> 0.75
    threshold, details = evaluate.choose_threshold(Y_TOY, S_TOY, "best_f1")
    assert threshold == pytest.approx(0.8)
    assert details["precision"] == pytest.approx(1.0) and details["recall"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("target, expected", [(0.6, 0.3), (0.99, 0.8), (0.65, 0.8)])
def test_target_precision_takes_highest_recall_meeting_target(target, expected):
    # precision / recall by cut-off: 0.9 1/.33, 0.8 1/.67, 0.7 .67/.67, 0.4 .5/.67, 0.3 .6/1
    # (0.65: 0.8 and 0.7 tie on recall; the tie goes to the higher precision, 0.8)
    threshold, details = evaluate.choose_threshold(Y_TOY, S_TOY, "target_precision", target)
    assert threshold == pytest.approx(expected)
    assert details["precision"] >= target and not details["fallback"]


def test_target_precision_unreachable_falls_back_to_best_f1(caplog):
    y, s = [0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6]
    threshold, details = evaluate.choose_threshold(y, s, "target_precision", 0.9)
    assert details["fallback"] is True
    assert threshold == pytest.approx(evaluate.choose_threshold(y, s, "best_f1")[0])
    assert "falling back to best F1" in caplog.text


def test_unknown_threshold_method():
    with pytest.raises(ValueError, match="Unknown threshold method"):
        evaluate.choose_threshold(Y_TOY, S_TOY, "youden")


def test_threshold_cannot_be_chosen_on_test(linear_model_dir, contract_data_dir):
    model, _ = load_model(linear_model_dir)
    test = load_split("test", contract_data_dir)
    with pytest.raises(evaluate.ThresholdLeakError, match="validation"):
        evaluate.threshold_from_split(model, test, "test", evaluate.validation_settings({}))


def test_overlapping_splits_are_rejected(contract_data_dir):
    validation = load_split("validation", contract_data_dir)
    with pytest.raises(evaluate.ThresholdLeakError, match="both the validation and test"):
        evaluate.check_disjoint(validation, validation.head(5))


def test_threshold_comes_from_validation_not_test(linear_model_dir, contract_data_dir, tmp_path, monkeypatch):
    """Spy on load_split and choose_threshold: the threshold sees only validation rows, before test is loaded."""
    loads, chosen_on = [], []
    real_load, real_choose = evaluate.load_split, evaluate.choose_threshold

    def spy_load(name, data_dir):
        loads.append(name)
        return real_load(name, data_dir)

    def spy_choose(y_true, scores, *args, **kwargs):
        chosen_on.append((list(loads), np.asarray(y_true).copy()))
        return real_choose(y_true, scores, *args, **kwargs)

    monkeypatch.setattr(evaluate, "load_split", spy_load)
    monkeypatch.setattr(evaluate, "choose_threshold", spy_choose)
    report = evaluate.run_validation({}, linear_model_dir, contract_data_dir, tmp_path, use_mlflow=False)

    assert loads == ["validation", "test"]
    assert len(chosen_on) == 1
    loaded_before, y_used = chosen_on[0]
    assert loaded_before == ["validation"]                    # the test split was not loaded yet
    validation, test = real_load("validation", contract_data_dir), real_load("test", contract_data_dir)
    np.testing.assert_array_equal(y_used, validation["denied"].to_numpy())

    # The two splits imply different thresholds; the report carries the validation one.
    model, _ = load_model(linear_model_dir)
    on_validation = real_choose(evaluate.labels(validation), evaluate.predict_scores(model, validation))[0]
    on_test = real_choose(evaluate.labels(test), evaluate.predict_scores(model, test))[0]
    assert on_validation != pytest.approx(on_test)
    assert report["threshold"] == pytest.approx(on_validation)
    assert report["threshold_selection"]["split"] == "validation"
    assert report["evaluated_split"] == "test"


def test_gate_passes_above_and_fails_below():
    metrics = {"pr_auc": 0.35, "pr_auc_lift": 2.7, "roc_auc": 0.71, "recall": 0.69, "precision": 0.24, "f1": 0.35}
    settings = evaluate.validation_settings({})
    assert all(c["passed"] for c in evaluate.apply_gate(metrics, settings))
    weak = {**metrics, "roc_auc": 0.55, "pr_auc": 0.14}
    failed = {c["metric"] for c in evaluate.apply_gate(weak, settings) if not c["passed"]}
    assert failed == {"roc_auc", "pr_auc"}


def test_gate_boundary_is_inclusive_and_null_minimums_are_skipped():
    settings = evaluate.validation_settings({"model_validation": {
        "min_pr_auc": 0.3, "min_pr_auc_lift": None, "min_roc_auc": None, "min_recall": None, "min_precision": 0.2}})
    checks = evaluate.apply_gate({"pr_auc": 0.3, "precision": 0.19}, settings)
    assert [(c["metric"], c["passed"]) for c in checks] == [("pr_auc", True), ("precision", False)]


def test_settings_defaults_and_validation():
    settings = evaluate.validation_settings({"seed": 7, "model": {"mlflow": {"experiment": "x", "tracking_uri": "u"}}})
    assert settings["seed"] == 7 and settings["threshold_method"] == "best_f1"
    assert settings["mlflow"] == {"experiment": "x", "tracking_uri": "u", "enabled": True}
    with pytest.raises(ValueError, match="threshold_method"):
        evaluate.validation_settings({"model_validation": {"threshold_method": "max_recall"}})
    with pytest.raises(ValueError, match="target_precision"):
        evaluate.validation_settings({"model_validation": {"target_precision": 1.5}})


def test_repo_params_have_a_valid_model_validation_section():
    params = load_params("params.yaml")
    settings = evaluate.validation_settings(params)
    assert settings["threshold_method"] in evaluate.THRESHOLD_METHODS
    assert settings["mlflow"]["experiment"] == "nsh-denial-model"
    assert "mode" in params["validation"]          # M3's data-validation section is untouched


def test_report_has_contract_fields_with_types(linear_model_dir, contract_data_dir, tmp_path, write_params):
    report = evaluate.main(["--params", str(write_params()), "--model-dir", str(linear_model_dir),
                            "--data-dir", str(contract_data_dir), "--out-dir", str(tmp_path / "out"), "--no-mlflow"])
    saved = json.loads((tmp_path / "out" / "validation.json").read_text(encoding="utf-8"))
    assert set(evaluate.CONTRACT_FIELDS) <= set(saved)
    for field in ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"):
        assert isinstance(saved[field], float) and 0.0 <= saved[field] <= 1.0
    cm = saved["confusion_matrix"]
    assert len(cm) == 2 and all(len(row) == 2 and all(isinstance(v, int) for v in row) for row in cm)
    assert sum(map(sum, cm)) == saved["n_test"] == 600
    assert saved["passed"] is True and report["passed"] is True
    assert saved["data_version"] == "fixture-v1" and saved["model_name"] == "logistic_regression"
    assert saved["mlflow_run_id"] is None
    for plot in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png", "precision_recall_vs_threshold.png"):
        assert (tmp_path / "out" / "validation" / plot).stat().st_size > 0


def test_cli_exits_non_zero_on_failed_gate_and_still_writes_report(linear_model_dir, contract_data_dir, tmp_path,
                                                                   write_params, caplog):
    params = write_params({"min_roc_auc": 0.99})
    with pytest.raises(SystemExit) as exit_info:
        evaluate.main(["--params", str(params), "--model-dir", str(linear_model_dir),
                       "--data-dir", str(contract_data_dir), "--out-dir", str(tmp_path / "out"), "--no-mlflow"])
    assert exit_info.value.code == 1
    saved = json.loads((tmp_path / "out" / "validation.json").read_text(encoding="utf-8"))
    assert saved["passed"] is False
    assert [c["metric"] for c in saved["gate"] if not c["passed"]] == ["roc_auc"]
    assert "Validation gate FAILED" in caplog.text and "roc_auc" in caplog.text


def test_target_precision_method_end_to_end(linear_model_dir, contract_data_dir, tmp_path):
    params = {"model_validation": {"threshold_method": "target_precision", "target_precision": 0.5,
                                   "min_recall": None}}
    report = evaluate.run_validation(params, linear_model_dir, contract_data_dir, tmp_path, use_mlflow=False)
    assert report["threshold_method"] == "target_precision"
    assert report["threshold_selection"]["precision"] >= 0.5


def test_validation_is_logged_to_mlflow(tree_model_dir, contract_data_dir, tmp_path, mlflow_store, write_params):
    report = evaluate.main(["--params", str(write_params()), "--model-dir", str(tree_model_dir),
                            "--data-dir", str(contract_data_dir), "--out-dir", str(tmp_path / "out")])
    run = mlflow.MlflowClient(mlflow_store).get_run(report["mlflow_run_id"])
    assert run.data.tags["data_version"] == "fixture-v1"
    assert run.data.tags["git_sha"] and run.data.tags["stage"] == "validation"
    assert run.data.tags["model_name"] == "random_forest"
    assert run.data.metrics["test_pr_auc"] == pytest.approx(report["pr_auc"])
    assert run.data.params["threshold_method"] == "best_f1"
    experiment = mlflow.MlflowClient(mlflow_store).get_experiment(run.info.experiment_id)
    assert experiment.name == "nsh-denial-model"
    artifacts = {a.path for a in mlflow.MlflowClient(mlflow_store).list_artifacts(run.info.run_id, "validation")}
    assert "validation/validation.json" in artifacts and "validation/plots" in artifacts
