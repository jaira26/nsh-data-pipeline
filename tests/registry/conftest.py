"""Registry fixtures: a contract-shaped selected model, validation.json + bias.json, all under tmp_path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nsh_pipeline.modeling.artifacts import save_model

TRAINING_SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "training"


class AmountModel:
    """Tiny picklable stand-in for a Pipeline: denial probability rises with amount_usd."""

    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(X["amount_usd"], dtype=float) - self.scale) / self.scale))
        return np.column_stack([1 - p, p])


def metadata(name: str) -> dict:
    return {"model_name": name, "params": {}, "cv_metrics": {}, "mlflow_run_id": "run-1",
            "data_version": "data-v1.1", "git_sha": "abc1234", "created_at": "2026-11-14T00:00:00+00:00"}


def split_metrics(pr_auc: float, positive_rate: float = 0.13, roc_auc: float = 0.71, recall: float = 0.68) -> dict:
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "precision": 0.25, "recall": recall, "f1": 0.36,
            "threshold": 0.53, "positive_rate": positive_rate, "confusion_matrix": [[418, 191], [29, 62]]}


def params(tmp_path: Path, **registry) -> dict:
    return {
        "model_validation": {"min_pr_auc": 0.20, "min_pr_auc_lift": 1.5, "min_roc_auc": 0.65, "min_recall": 0.40},
        "registry": {"backend": "local", "local_dir": str(tmp_path / "registry"),
                     "production_file": str(tmp_path / "registry" / "production.json"),
                     "decision_file": str(tmp_path / "reports" / "registry_decision.json"),
                     "alerts": True, "rescore_production": False, **registry},
    }


@pytest.fixture
def workspace(tmp_path):
    """write(pr_auc=..., validation_passed=..., bias_passed=...) writes a fresh selected model + reports."""
    reports = tmp_path / "reports"

    def write(pr_auc=0.32, validation_passed=True, bias_passed=True, name="logistic_regression_reweighed", **kw):
        model_dir = tmp_path / "models_mitigated" / "candidates" / name
        save_model(AmountModel(), metadata(name), model_dir)
        reports.mkdir(exist_ok=True)
        (reports / "validation.json").write_text(json.dumps({
            "model_name": "logistic_regression", "passed": validation_passed, "pr_auc": 0.35,
            "gate": [{"metric": "pr_auc", "value": 0.35 if validation_passed else 0.1, "minimum": 0.2,
                      "passed": validation_passed}]}))
        (reports / "bias.json").write_text(json.dumps({
            "selected_model": name if bias_passed else None, "selected_model_dir": str(model_dir),
            "selected_model_mlflow_run_id": "run-2", "base_model": "logistic_regression", "passed": bias_passed,
            "threshold": 0.53, "data_version": "data-v1.1", "selection_reason": "fairness gate failed",
            "performance": {"test": split_metrics(pr_auc, **kw)}}))
        return model_dir

    write.tmp_path = tmp_path
    write.reports = reports
    return write


@pytest.fixture
def alerts():
    sent = []

    def alert(title, body, details=None, env=None):
        sent.append((title, body))
        return []

    alert.sent = sent
    return alert
