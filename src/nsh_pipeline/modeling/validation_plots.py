"""Hold-out validation plots (phase 2, step 3): ROC, PR, confusion matrix, precision/recall vs threshold.

Every function draws one PNG from test-split labels and scores, marks the chosen decision
threshold where it applies, and returns the file path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

LINE = "#2b6cb0"
SECOND = "#b7791f"
CHANCE = "#a0aec0"
MARK = "#c53030"

PLOT_FILES = {
    "roc": "roc_curve.png",
    "pr": "pr_curve.png",
    "confusion": "confusion_matrix.png",
    "threshold": "precision_recall_vs_threshold.png",
}


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def roc_plot(y_true, scores, threshold: float, roc_auc: float, path) -> Path:
    """ROC curve with the chance diagonal and the operating point at the chosen threshold."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    predicted = np.asarray(scores) >= threshold
    y = np.asarray(y_true).astype(bool)
    point = (predicted[~y].mean() if (~y).any() else 0.0, predicted[y].mean() if y.any() else 0.0)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=LINE, lw=2, label=f"Model (ROC-AUC {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=CHANCE, ls="--", lw=1, label="Chance")
    ax.scatter(*point, color=MARK, zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (test split)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    return _save(fig, path)


def pr_plot(y_true, scores, threshold: float, pr_auc: float, path) -> Path:
    """Precision-recall curve with the no-skill line (the positive rate) and the operating point."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    y = np.asarray(y_true).astype(bool)
    predicted = np.asarray(scores) >= threshold
    point_recall = predicted[y].mean() if y.any() else 0.0
    point_precision = y[predicted].mean() if predicted.any() else 0.0
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color=LINE, lw=2, label=f"Model (PR-AUC {pr_auc:.3f})")
    ax.axhline(y.mean(), color=CHANCE, ls="--", lw=1, label=f"No skill ({y.mean():.3f})")
    ax.scatter(point_recall, point_precision, color=MARK, zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test split)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    return _save(fig, path)


def confusion_plot(matrix, threshold: float, path) -> Path:
    """2 x 2 confusion matrix [[tn, fp], [fn, tp]] with counts and row shares."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    rows = matrix.sum(axis=1, keepdims=True).clip(min=1)
    for (i, j), count in np.ndenumerate(matrix):
        colour = "white" if count > matrix.max() / 2 else "black"
        ax.text(j, i, f"{count}\n({count / rows[i, 0]:.0%})", ha="center", va="center", color=colour)
    ax.set_xticks([0, 1], ["Approved", "Denied"])
    ax.set_yticks([0, 1], ["Approved", "Denied"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (test, threshold {threshold:.3f})")
    return _save(fig, path)


def threshold_plot(y_true, scores, threshold: float, path) -> Path:
    """Precision and recall as the threshold moves, with the chosen threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(thresholds, precision[:-1], color=LINE, lw=2, label="Precision")
    ax.plot(thresholds, recall[:-1], color=SECOND, lw=2, label="Recall")
    ax.axvline(threshold, color=MARK, ls="--", lw=1, label=f"Chosen threshold {threshold:.3f}")
    ax.set(xlabel="Decision threshold (predicted denial probability)", ylabel="Score",
           title="Precision and recall vs threshold (test split)", ylim=(0, 1.02))
    ax.legend(loc="best")
    return _save(fig, path)


def validation_plots(y_true, scores, report: dict, out_dir) -> dict[str, Path]:
    """All four plots into out_dir; returns {kind: path}."""
    out_dir = Path(out_dir)
    threshold = report["threshold"]
    return {
        "roc": roc_plot(y_true, scores, threshold, report["roc_auc"], out_dir / PLOT_FILES["roc"]),
        "pr": pr_plot(y_true, scores, threshold, report["pr_auc"], out_dir / PLOT_FILES["pr"]),
        "confusion": confusion_plot(report["confusion_matrix"], threshold, out_dir / PLOT_FILES["confusion"]),
        "threshold": threshold_plot(y_true, scores, threshold, out_dir / PLOT_FILES["threshold"]),
    }
