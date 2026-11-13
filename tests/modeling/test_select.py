import numpy as np
import pytest

from nsh_pipeline.modeling.select import best_f1_threshold, evaluate, rank_candidates, select_best


def result(val_pr_auc, cv_pr_auc=0.5):
    return {"validation_metrics": {"pr_auc": val_pr_auc}, "cv_metrics": {"cv_pr_auc": cv_pr_auc}}


def test_evaluate_on_a_known_toy_prediction():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = evaluate(y, scores, threshold=0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert metrics["positive_rate"] == 0.5


def test_perfect_scores_give_perfect_metrics():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9])))
    assert metrics["pr_auc"] == metrics["roc_auc"] == metrics["f1"] == 1.0


def test_selection_picks_the_best_validation_metric():
    results = {"logistic_regression": result(0.30), "random_forest": result(0.42), "lightgbm": result(0.40)}
    assert select_best(results) == "random_forest"
    assert rank_candidates(results) == ["random_forest", "lightgbm", "logistic_regression"]


def test_ties_go_to_cv_metric_then_to_the_simpler_model():
    assert select_best({"lightgbm": result(0.4, 0.6), "random_forest": result(0.4, 0.5)}) == "lightgbm"
    assert select_best({"lightgbm": result(0.4), "logistic_regression": result(0.4)}) == "logistic_regression"


def test_selection_needs_candidates():
    with pytest.raises(ValueError):
        select_best({})
