"""Hold-out validation of the selected model and the validation gate (phase 2, step 3). One command:

    python -m nsh_pipeline.modeling.evaluate --params params.yaml --model-dir models/best \
        --data-dir data/training --out-dir reports/model

  1. load models/best (model.joblib + metadata.json) and the validation and test splits
  2. choose the decision threshold on the VALIDATION split only (params.yaml -> model_validation.threshold_method)
  3. score the TEST split once at that threshold: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix
  4. gate: passed = every metric meets its minimum in params.yaml -> model_validation
  5. write reports/model/validation.json + plots in reports/model/validation/, log them to MLflow
  6. exit with code 1 when the gate fails (after writing the report), so CI and `dvc repro` stop

The test split is never used to choose the threshold or anything else: `threshold_from_split` refuses
any split but validation, and the test split is loaded only after the threshold is fixed.
"""
from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import precision_recall_curve

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling import validation_plots
from nsh_pipeline.modeling.artifacts import load_model
from nsh_pipeline.modeling.dataset import DATASET_CARD, FEATURE_COLUMNS, LABEL_COLUMN, load_dataset_card, load_split
from nsh_pipeline.modeling.select import best_f1_threshold, evaluate
from nsh_pipeline.modeling.tracking import data_hash, data_version, git_sha, setup_mlflow

logger = logging.getLogger(__name__)

# Report contract (task assigner section 2.3): validation.json always has these fields.
CONTRACT_FIELDS = ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold", "confusion_matrix", "passed")
REPORT_FILE = "validation.json"
PLOTS_DIR = "validation"

THRESHOLD_METHODS = ("best_f1", "target_precision")
THRESHOLD_SPLIT = "validation"   # the only split the threshold may be chosen on
HOLDOUT_SPLIT = "test"           # scored once, never used for any choice

# Gate keys in params.yaml -> model_validation, and the metric each one bounds from below.
GATE_KEYS = {
    "min_pr_auc": "pr_auc",
    "min_pr_auc_lift": "pr_auc_lift",   # PR-AUC / positive rate (1.0 = no better than guessing)
    "min_roc_auc": "roc_auc",
    "min_recall": "recall",
    "min_precision": "precision",
    "min_f1": "f1",
}

DEFAULTS = {
    "threshold_method": "best_f1",
    "target_precision": 0.30,
    "min_pr_auc": 0.20,
    "min_pr_auc_lift": 1.5,
    "min_roc_auc": 0.65,
    "min_recall": 0.40,
    "min_precision": None,
    "min_f1": None,
    "out_dir": "reports/model",
    "mlflow": {"enabled": True},
    "sensitivity": {},
}


class ThresholdLeakError(ValueError):
    """Raised when the threshold would be chosen on (or overlap with) the hold-out test split."""


def validation_settings(params: dict) -> dict:
    """params.yaml -> model_validation with defaults filled in.

    The seed falls back to the global seed. MLflow settings (tracking URI, experiment) come from
    model.mlflow so validation runs land in the same store and experiment as training.
    """
    section = params.get("model_validation") or {}
    settings = {**DEFAULTS, "seed": params.get("seed", 42), **section}
    model_mlflow = (params.get("model") or {}).get("mlflow") or {}
    settings["mlflow"] = {**model_mlflow, **DEFAULTS["mlflow"], **(section.get("mlflow") or {})}
    if settings["threshold_method"] not in THRESHOLD_METHODS:
        raise ValueError(f"Unknown threshold_method {settings['threshold_method']!r}; expected one of "
                         f"{THRESHOLD_METHODS}")
    if not 0 < float(settings["target_precision"]) <= 1:
        raise ValueError(f"target_precision must be in (0, 1], got {settings['target_precision']}")
    return settings


def labels(frame) -> np.ndarray:
    return frame[LABEL_COLUMN].astype(int).to_numpy()


def predict_scores(model, frame) -> np.ndarray:
    """Predicted denial probability for every row (model contract: predict_proba(X[FEATURE_COLUMNS]))."""
    return np.asarray(model.predict_proba(frame[FEATURE_COLUMNS])[:, 1], dtype=float)


def choose_threshold(y_true, scores, method: str = "best_f1", target_precision: float = 0.30) -> tuple[float, dict]:
    """Decision threshold from labels and scores; returns (threshold, details).

    best_f1: the cut-off with the highest F1.
    target_precision: the cut-off with the highest recall among those with precision >= target_precision
    (ties to the higher precision); falls back to best_f1 with a warning when no cut-off reaches it.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    details = {"method": method, "fallback": False}
    if method == "target_precision":
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        ok = np.flatnonzero(precision[:-1] >= target_precision)
        if len(ok):
            best = ok[np.lexsort((-precision[ok], -recall[ok]))[0]]
            threshold = float(thresholds[best])
        else:
            logger.warning("No threshold reaches precision %.3f on the validation split; falling back to best F1",
                           target_precision)
            details["fallback"] = True
            threshold = best_f1_threshold(y_true, scores)
        details["target_precision"] = float(target_precision)
    elif method == "best_f1":
        threshold = best_f1_threshold(y_true, scores)
    else:
        raise ValueError(f"Unknown threshold method {method!r}; expected one of {THRESHOLD_METHODS}")
    predicted = scores >= threshold
    tp = int((predicted & (y_true == 1)).sum())
    details["precision"] = tp / int(predicted.sum()) if predicted.any() else 0.0
    details["recall"] = tp / int((y_true == 1).sum()) if (y_true == 1).any() else 0.0
    return float(threshold), details


def threshold_from_split(model, frame, split: str, settings: dict) -> tuple[float, dict]:
    """Choose the threshold on one split. Only the validation split is allowed."""
    if split != THRESHOLD_SPLIT:
        raise ThresholdLeakError(f"The decision threshold must be chosen on the {THRESHOLD_SPLIT!r} split, "
                                 f"not {split!r}; the test split is for hold-out validation only")
    threshold, details = choose_threshold(labels(frame), predict_scores(model, frame),
                                          settings["threshold_method"], settings["target_precision"])
    details.update({"split": split, "n_rows": int(len(frame))})
    return threshold, details


def check_disjoint(validation, test) -> None:
    """The splits must not share swipes, or the threshold would have seen test rows."""
    shared = set(validation["swipe_id"]) & set(test["swipe_id"])
    if shared:
        raise ThresholdLeakError(f"{len(shared)} swipe(s) are in both the validation and test splits, "
                                 f"e.g. {sorted(shared)[:3]}")


def holdout_metrics(y_true, scores, threshold: float) -> dict:
    """Test-split metrics at a fixed threshold, plus the PR-AUC lift over the positive rate."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("The test split needs both denied and approved swipes to compute ROC-AUC / PR-AUC")
    metrics = evaluate(y_true, scores, threshold)
    metrics["confusion_matrix"] = [[int(v) for v in row] for row in metrics["confusion_matrix"]]
    metrics["pr_auc_lift"] = metrics["pr_auc"] / metrics["positive_rate"]
    return metrics


def apply_gate(metrics: dict, settings: dict) -> list[dict]:
    """One check per configured minimum (unset / null minimums are skipped)."""
    checks = []
    for key, metric in GATE_KEYS.items():
        minimum = settings.get(key)
        if minimum is None:
            continue
        value = float(metrics[metric])
        checks.append({"metric": metric, "value": value, "minimum": float(minimum), "passed": value >= minimum})
    return checks


def run_tags(metadata: dict, card: dict, data_dir) -> dict:
    """MLflow tags: data_version and git_sha (team rule), plus the model's lineage."""
    return {
        "data_version": str(metadata.get("data_version") or data_version(card)),
        "git_sha": git_sha(),
        "model_git_sha": str(metadata.get("git_sha", "unknown")),
        "model_name": str(metadata.get("model_name", "unknown")),
        "model_run_id": str(metadata.get("mlflow_run_id")),
        "data_hash": data_hash(data_dir),
    }


def build_report(metrics: dict, checks: list[dict], threshold_details: dict, tags: dict, n_test: int) -> dict:
    """validation.json: the contract fields first, then informational fields."""
    report = {field: metrics[field] for field in CONTRACT_FIELDS if field != "passed"}
    report["passed"] = bool(all(check["passed"] for check in checks))
    report.update({
        "model_name": tags["model_name"],
        "data_version": tags["data_version"],
        "git_sha": tags["git_sha"],
        "model_git_sha": tags["model_git_sha"],
        "evaluated_split": HOLDOUT_SPLIT,
        "n_test": int(n_test),
        "positive_rate": metrics["positive_rate"],
        "pr_auc_lift": metrics["pr_auc_lift"],
        "threshold_method": threshold_details["method"],
        "threshold_selection": threshold_details,
        "gate": checks,
    })
    return report


@contextmanager
def mlflow_run(settings: dict, run_name: str, tags: dict):
    """An MLflow run in the team experiment, or nothing when model_validation.mlflow.enabled is false."""
    if not settings["mlflow"].get("enabled", True):
        yield None
        return
    setup_mlflow(settings["mlflow"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        yield run


def write_json(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _log_validation(report: dict, settings: dict, report_path: Path, plots: dict) -> None:
    mlflow.log_params({
        "threshold": report["threshold"],
        "threshold_method": report["threshold_method"],
        "target_precision": settings["target_precision"],
        **{key: settings.get(key) for key in GATE_KEYS if settings.get(key) is not None},
    })
    mlflow.log_metrics({
        **{f"test_{key}": report[key] for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "pr_auc_lift")},
        "threshold": report["threshold"],
        "gate_passed": float(report["passed"]),
    })
    mlflow.set_tag("gate_passed", str(report["passed"]).lower())
    mlflow.log_artifact(str(report_path), artifact_path="validation")
    for path in plots.values():
        mlflow.log_artifact(str(path), artifact_path="validation/plots")


def run_validation(params: dict, model_dir="models/best", data_dir="data/training", out_dir=None,
                   use_mlflow: bool | None = None) -> dict:
    """Threshold on validation, metrics + gate on test, reports + MLflow. Returns the report (never raises on a
    failed gate; the CLI turns passed=False into exit code 1)."""
    settings = validation_settings(params)
    if use_mlflow is not None:
        settings["mlflow"]["enabled"] = use_mlflow
    out_dir = Path(out_dir or settings["out_dir"])
    model, metadata = load_model(model_dir)
    card = load_dataset_card(data_dir) if (Path(data_dir) / DATASET_CARD).exists() else {}

    # 1. Threshold: validation split only. The test split is not loaded yet.
    validation = load_split(THRESHOLD_SPLIT, data_dir)
    threshold, threshold_details = threshold_from_split(model, validation, THRESHOLD_SPLIT, settings)
    logger.info("Threshold %.4f chosen on the validation split (%s, %d rows)", threshold,
                threshold_details["method"], len(validation))

    # 2. Hold-out: score the test split once at the fixed threshold.
    test = load_split(HOLDOUT_SPLIT, data_dir)
    check_disjoint(validation, test)
    y_test, scores = labels(test), predict_scores(model, test)
    metrics = holdout_metrics(y_test, scores, threshold)
    checks = apply_gate(metrics, settings)

    tags = run_tags(metadata, card, data_dir)
    with mlflow_run(settings, "validation", {**tags, "stage": "validation"}) as run:
        report = build_report(metrics, checks, threshold_details, tags, len(test))
        report["mlflow_run_id"] = run.info.run_id if run else None
        report_path = write_json(report, out_dir / REPORT_FILE)
        plots = validation_plots.validation_plots(y_test, scores, report, out_dir / PLOTS_DIR)
        if run:
            _log_validation(report, settings, report_path, plots)

    logger.info("Test split (%d rows): ROC-AUC %.4f, PR-AUC %.4f (lift %.2fx), precision %.4f, recall %.4f, "
                "F1 %.4f at threshold %.4f", len(test), report["roc_auc"], report["pr_auc"], report["pr_auc_lift"],
                report["precision"], report["recall"], report["f1"], threshold)
    logger.info("Report written to %s", report_path)
    return report


def gate_message(report: dict) -> str:
    failed = [check for check in report["gate"] if not check["passed"]]
    if not failed:
        return "Validation gate passed"
    details = "; ".join(f"{c['metric']} {c['value']:.4f} < {c['minimum']:.4f}" for c in failed)
    return f"Validation gate FAILED for model {report['model_name']!r}: {details}"


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Hold-out validation and validation gate for models/best.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-dir", default="models/best")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--out-dir", default=None, help="default: params.yaml -> model_validation.out_dir")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging (reports are still written)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_validation(load_params(args.params), args.model_dir, args.data_dir, args.out_dir,
                            use_mlflow=False if args.no_mlflow else None)
    print(json.dumps({field: report[field] for field in CONTRACT_FIELDS}, indent=2))
    if not report["passed"]:
        logger.error(gate_message(report))
        raise SystemExit(1)
    logger.info(gate_message(report))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
