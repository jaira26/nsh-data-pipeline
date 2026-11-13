"""Result charts for the README and MLflow: model comparison, confusion matrices, PR curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM"}
BAR_METRICS = [("cv_pr_auc", "CV PR-AUC (train)"), ("pr_auc", "Validation PR-AUC"),
               ("roc_auc", "Validation ROC-AUC"), ("f1", "Validation F1")]
COLORS = ["#9db4d1", "#2b6cb0", "#2f855a", "#b7791f"]


def _label(name: str) -> str:
    return LABELS.get(name, name)


def comparison_chart(results: dict[str, dict], best: str, path) -> Path:
    """Grouped bars: CV and validation metrics per model; the selected model is starred."""
    names = list(results)
    x = np.arange(len(names))
    width = 0.8 / len(BAR_METRICS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, ((key, label), color) in enumerate(zip(BAR_METRICS, COLORS)):
        values = [results[n]["cv_metrics"].get(key) if key.startswith("cv_") else results[n]["validation_metrics"][key]
                  for n in names]
        bars = ax.bar(x + (i - (len(BAR_METRICS) - 1) / 2) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    positive_rate = next(iter(results.values()))["validation_metrics"]["positive_rate"]
    ax.axhline(positive_rate, color="grey", linestyle="--", linewidth=1,
               label=f"PR-AUC of random guessing ({positive_rate:.2f})")
    ax.set_xticks(x, [f"{_label(n)} ★" if n == best else _label(n) for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Candidate models: swipe-denial prediction (★ = selected)")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def confusion_matrices(results: dict[str, dict], path) -> Path:
    """One confusion matrix per model on the validation split, at its best-F1 threshold."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.8), squeeze=False)
    for ax, (name, result) in zip(axes[0], results.items()):
        metrics = result["validation_metrics"]
        matrix = np.array(metrics["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Approved", "Denied"])
        ax.set_yticks([0, 1], ["Approved", "Denied"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{_label(name)}\nthreshold {metrics['threshold']:.2f} · F1 {metrics['f1']:.2f}", fontsize=10)
    fig.suptitle("Confusion matrices (validation split)")
    return _save(fig, path)


def pr_curves(curves: dict[str, tuple], path) -> Path:
    """Precision-recall curve of every model on the validation split. curves: name -> (y_true, scores)."""
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for name, (y_true, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=_label(name))
    base = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base, color="grey", linestyle="--", linewidth=1, label=f"Random ({base:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-recall curves (validation split)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
