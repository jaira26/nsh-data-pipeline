"""Model selection: the candidate with the best validation metric becomes models/best.

Ties (equal validation metric) go to the higher CV metric, then to the simpler model
(MODEL_NAMES order: logistic regression < random forest < LightGBM), so selection is deterministic.

Het's fairness-aware final selection (bias.json -> selected_model) runs after this and may pick
another candidate from models/candidates/; this step only ranks by performance.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

from nsh_pipeline.modeling.candidates import MODEL_NAMES

DEFAULT_METRIC = "pr_auc"


def best_f1_threshold(y_true, scores) -> float:
    """Score cut-off with the highest F1 on these rows (a starting point; Raj sets the final threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(f1[:-1].argmax())]) if len(thresholds) else 0.5


def evaluate(y_true, scores, threshold: float) -> dict:
    """Ranking metrics plus threshold metrics and the confusion matrix [[tn, fp], [fn, tp]]."""
    y_true = pd.Series(y_true).astype(int).to_numpy()
    predicted = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def rank_candidates(results: dict[str, dict], metric: str = DEFAULT_METRIC) -> list[str]:
    """Model names, best first, by validation metric, then CV metric, then simplicity."""
    def key(name):
        result = results[name]
        order = MODEL_NAMES.index(name) if name in MODEL_NAMES else len(MODEL_NAMES)
        return (-result["validation_metrics"][metric], -result["cv_metrics"].get(f"cv_{metric}", 0.0), order)
    return sorted(results, key=key)


def select_best(results: dict[str, dict], metric: str = DEFAULT_METRIC) -> str:
    """Name of the best candidate."""
    if not results:
        raise ValueError("No candidates to select from")
    return rank_candidates(results, metric)[0]
