"""Model bias: fairness audit, mitigation and fairness-aware final selection (phase 2, step 4). One command:

    python -m nsh_pipeline.modeling.fairness --params params.yaml --data-dir data/training \
        --models-dir models --out-dir reports/model

  1. load every candidate in models/candidates/ and the train, validation and test splits
  2. per candidate: decision threshold = best F1 on the VALIDATION split (as in training and in model
     validation); audit the validation and test splits per slice (fairness_metrics.audit)
  3. every candidate that fails the gate on validation is refit with reweighing on its root slice
     (fairness_mitigation), audited again and saved to models_mitigated/candidates/<name>_reweighed/
  4. rank all candidates (original and mitigated) by validation PR-AUC, training's selection metric.
     The selected model is the best-ranked one that passes the gate on VALIDATION; the test split is
     never used to choose, it only confirms the choice (bias.json -> passed)
  5. write reports/model/bias.json and reports/model/bias/ (Markdown report, CSV, charts), log to MLflow
  6. exit 1 when no candidate passes or the selected one fails on test (after writing the report)

Settings: params.yaml -> model_bias (named so because `bias:` is the phase-1 data-bias section).
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling import fairness_report
from nsh_pipeline.modeling.artifacts import METADATA_FILE, MODEL_FILE, candidate_dir, load_model, save_model, utc_now
from nsh_pipeline.modeling.candidates import MODEL_NAMES, features
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, LABEL_COLUMN, SLICE_COLUMNS, load_dataset_card, load_split
from nsh_pipeline.modeling.fairness_metrics import RATE_METRICS, audit, root_slice
from nsh_pipeline.modeling.fairness_mitigation import mitigated_name, reweigh_and_refit
from nsh_pipeline.modeling.select import best_f1_threshold, evaluate
from nsh_pipeline.modeling.tracking import data_version, run_tags, setup_mlflow

logger = logging.getLogger(__name__)

REPORT_FILE = "bias.json"
REPORT_DIR = "bias"
SCHEMA_VERSION = "1.0"
# Report contract (task assigner section 2.3): bias.json always has these fields.
CONTRACT_FIELDS = ("per_slice_metrics", "max_disparity", "mitigation_applied", "passed", "selected_model")
SELECTION_SPLIT = "validation"   # the gate that chooses the model
AUDIT_SPLIT = "test"             # hold-out confirmation; never used to choose

DEFAULTS = {
    "min_support": 30,
    "alpha": 0.01,
    # Gate limits in percentage points (group worse than overall); null = reported, not gated.
    "max_gap_pts": {"false_positive_rate": 20, "recall": 20, "selection_rate": None, "precision": None},
    "slice_columns": list(SLICE_COLUMNS),
    "mitigation": {"enabled": True, "method": "reweighing"},
    "fail_on_gate": True,
    "out_dir": "reports/model",
    "mitigated_models_dir": "models_mitigated",
    "mlflow": {"enabled": True},
}


def bias_settings(params: dict) -> dict:
    """params.yaml -> model_bias with defaults; MLflow store, selection metric and seed from model."""
    section = params.get("model_bias") or {}
    model = params.get("model") or {}
    settings = {**DEFAULTS, **section}
    settings["mitigation"] = {**DEFAULTS["mitigation"], **(section.get("mitigation") or {})}
    settings["mlflow"] = {**(model.get("mlflow") or {}), **DEFAULTS["mlflow"], **(section.get("mlflow") or {})}
    settings["selection_metric"] = model.get("selection_metric", "pr_auc")
    settings["seed"] = model.get("seed", params.get("seed", 42))
    inject = params.get("inject_issues") or {}
    settings["planted_skew"] = inject.get("denial_skew") if inject.get("enabled") else None  # report's planted check
    unknown = [metric for metric in settings["max_gap_pts"] if metric not in RATE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) in model_bias.max_gap_pts: {unknown}; expected some of {RATE_METRICS}")
    if not any(limit is not None for limit in settings["max_gap_pts"].values()):
        raise ValueError("model_bias.max_gap_pts must give a limit for at least one metric")
    if settings["mitigation"]["method"] != "reweighing":
        raise ValueError(f"Unknown mitigation method {settings['mitigation']['method']!r}; expected 'reweighing'")
    return settings


def load_candidates(models_dir) -> dict:
    """{name: (pipeline, metadata)} for every saved candidate, simplest model family first."""
    root = Path(models_dir) / "candidates"
    folders = [path for path in root.glob("*") if (path / MODEL_FILE).exists()] if root.exists() else []
    if not folders:
        raise FileNotFoundError(f"No candidate models in {root}; run the train stage first")

    def order(path):
        return (MODEL_NAMES.index(path.name) if path.name in MODEL_NAMES else len(MODEL_NAMES), path.name)

    return {path.name: load_model(path) for path in sorted(folders, key=order)}


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL_COLUMN].astype(int).to_numpy()


def scores(model, frame: pd.DataFrame) -> np.ndarray:
    """Predicted denial probability per row (model contract: predict_proba(X[FEATURE_COLUMNS]))."""
    return np.asarray(model.predict_proba(frame[FEATURE_COLUMNS])[:, 1], dtype=float)


def worst_gaps(flags: pd.DataFrame) -> dict:
    """Per gated metric, the judged group furthest worse than overall: {metric: {gap_pts, slice, group}}."""
    out = {}
    for metric, rows in flags[flags["judged"]].groupby("metric", sort=False):
        worst = rows.loc[rows["gap_pts"].idxmax()]
        out[metric] = {"gap_pts": round(float(worst["gap_pts"]), 2), "slice": str(worst["slice"]),
                       "group": str(worst["group"])}
    return out


def assess(model, splits: dict, settings: dict) -> dict:
    """Threshold on validation, then performance and fairness audit on validation and test."""
    validation = splits[SELECTION_SPLIT]
    val_scores = scores(model, validation)
    threshold = best_f1_threshold(labels(validation), val_scores)
    out = {"threshold": float(threshold)}
    for split_name, split_scores in ((SELECTION_SPLIT, val_scores), (AUDIT_SPLIT, scores(model, splits[AUDIT_SPLIT]))):
        frame = splits[split_name]
        result = audit(frame, labels(frame), split_scores, threshold, settings["slice_columns"], settings)
        out[split_name] = {"performance": evaluate(labels(frame), split_scores, threshold), "audit": result,
                           "passed": result["passed"], "worst_gap_pts": worst_gaps(result["flags"])}
    return out


def rank(results: dict, metric: str) -> list[str]:
    """Candidate names, best first: validation metric, then original before mitigated, then simpler model."""
    def key(name):
        result = results[name]
        base = result["base_model"]
        order = MODEL_NAMES.index(base) if base in MODEL_NAMES else len(MODEL_NAMES)
        return (-result[SELECTION_SPLIT]["performance"][metric], result["mitigation"] is not None, order, name)
    return sorted(results, key=key)


def _clean(value):
    """JSON-safe copy: NaN -> None, numpy scalars -> Python numbers, DataFrames dropped by the caller."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 4)
    return value


def slice_records(table: pd.DataFrame, flags: pd.DataFrame) -> list[dict]:
    """Per-slice metrics as records, each with the gated metrics it is flagged on or too small to judge for."""
    records = []
    for row in table.to_dict("records"):
        mine = flags[(flags["slice"] == row["slice"]) & (flags["group"] == row["group"])]
        records.append({**row, "flagged_metrics": sorted(mine.loc[mine["flagged"], "metric"]),
                        "too_small_for": sorted(mine.loc[~mine["judged"], "metric"])})
    return _clean(records)


def flag_records(flagged: pd.DataFrame) -> list[dict]:
    columns = ["slice", "group", "metric", "value", "overall", "gap_pts", "support", "p_value"]
    return _clean(flagged[columns].to_dict("records"))


def split_summary(split: dict) -> dict:
    performance = {key: split["performance"][key] for key in ("pr_auc", "roc_auc", "precision", "recall", "f1")}
    return _clean({**performance, "passed": split["passed"], "worst_gap_pts": split["worst_gap_pts"],
                   "n_tests": split["audit"]["n_tests"], "flags": flag_records(split["audit"]["flagged"])})


def build_report(results: dict, ranking: list[str], selected: str | None, settings: dict, card: dict,
                 tags: dict) -> dict:
    """bias.json: the selected model's per-slice audit on test, plus every candidate's gate results."""
    audited = selected or ranking[0]  # nothing selected: show the best-performing model's problems
    result = results[audited]
    test = result[AUDIT_SPLIT]
    passed = selected is not None and bool(test["passed"])
    if selected is None:
        reason = "No candidate passes the fairness gate on the validation split, even after mitigation."
    elif selected == ranking[0]:
        reason = f"{selected} has the best validation {settings['selection_metric']} and passes the fairness gate."
    else:
        skipped = ranking[:ranking.index(selected)]
        reason = (f"{selected} is the best-ranked candidate that passes the fairness gate on validation; "
                  f"better-scoring {', '.join(skipped)} failed it.")
    reason = reason.replace(settings["selection_metric"], settings["selection_metric"].replace("_", "-").upper())
    gated = {metric: limit for metric, limit in settings["max_gap_pts"].items() if limit is not None}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "data_version": data_version(card),
        "git_sha": tags.get("git_sha"),
        "selected_model": selected,
        "selected_model_dir": result["model_dir"] if selected else None,
        "selected_model_mlflow_run_id": result.get("mlflow_run_id") if selected else None,
        "base_model": result["base_model"],
        "passed": passed,
        "mitigation_applied": result["mitigation"] if selected else None,
        "selection_reason": reason,
        "audited_model": audited,
        "audit_split": AUDIT_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "threshold": round(result["threshold"], 4),
        "gate": {"max_gap_pts": gated, "reported_only": [m for m in RATE_METRICS if m not in gated],
                 "min_support": settings["min_support"], "alpha": settings["alpha"],
                 "n_tests": test["audit"]["n_tests"], "alpha_corrected": test["audit"]["alpha_corrected"],
                 "slice_columns": list(settings["slice_columns"])},
        "overall": _clean(test["audit"]["overall"]),
        "performance": {split: _clean({k: v for k, v in result[split]["performance"].items()})
                        for split in (SELECTION_SPLIT, AUDIT_SPLIT)},
        "max_disparity": _clean(test["audit"]["max_disparity"]),
        "flags": flag_records(test["audit"]["flagged"]),
        "per_slice_metrics": slice_records(test["audit"]["table"], test["audit"]["flags"]),
        "ranking": ranking,
        "candidates": {
            name: {"base_model": res["base_model"], "mitigation": res["mitigation"], "model_dir": res["model_dir"],
                   "mlflow_run_id": res.get("mlflow_run_id"), "threshold": round(res["threshold"], 4),
                   SELECTION_SPLIT: split_summary(res[SELECTION_SPLIT]), AUDIT_SPLIT: split_summary(res[AUDIT_SPLIT])}
            for name, res in results.items()
        },
    }


@contextmanager
def tracking_run(settings: dict, tags: dict):
    """An MLflow run for this stage when enabled, else None."""
    if not settings["mlflow"].get("enabled", True):
        yield None
        return
    setup_mlflow(settings["mlflow"])
    with mlflow.start_run(run_name="model-bias") as run:
        mlflow.set_tags({**tags, "stage": "model_bias"})
        yield run


def log_mitigated_model(name: str, model, result: dict, settings: dict, tags: dict, train: pd.DataFrame) -> str:
    """A nested MLflow run per mitigated candidate, like training's candidate runs: params, validation and test
    metrics, fairness gate results, tags and the fitted model (cloudpickle, as decision 33). Returns the run id."""
    info = result["mitigation"]
    with mlflow.start_run(run_name=name, nested=True) as child:
        mlflow.set_tags({**tags, "stage": "mitigated_candidate", "model_name": name, "base_model": info["base_model"],
                         "mitigation": f"{info['method']} on {info['slice']}"})
        mlflow.log_params({**{key: value for key, value in (result["params"] or {}).items()},
                           "mitigation_method": info["method"], "mitigation_slice": info["slice"],
                           "threshold": round(result["threshold"], 6)})
        metrics = {"effective_share": info["effective_share"]}
        for split, prefix in ((SELECTION_SPLIT, "val"), (AUDIT_SPLIT, "test")):
            performance = result[split]["performance"]
            metrics.update({f"{prefix}_{key}": value for key, value in performance.items()
                            if isinstance(value, float) and not np.isnan(value)})
            metrics[f"{prefix}_fairness_passed"] = float(result[split]["passed"])
            for metric, worst in result[split]["worst_gap_pts"].items():
                metrics[f"{prefix}_worst_gap_pts.{metric}"] = worst["gap_pts"]
        mlflow.log_metrics(metrics)
        if settings["mlflow"].get("log_models", True):
            sample = features(train).head(5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # integer-column schema hint, as in training
                mlflow.sklearn.log_model(model, name="model", input_example=sample,
                                         signature=infer_signature(sample, model.predict_proba(sample)),
                                         serialization_format="cloudpickle")
        return child.info.run_id


def log_results(report: dict, settings: dict, report_path: Path, report_dir: Path, mitigated: dict) -> None:
    """Params, gate metrics, the report files and the mitigated models into the active MLflow run."""
    gate = report["gate"]
    mlflow.log_params({"min_support": gate["min_support"], "alpha": gate["alpha"],
                       "slice_columns": ",".join(gate["slice_columns"]),
                       **{f"max_gap_pts.{metric}": limit for metric, limit in gate["max_gap_pts"].items()}})
    metrics = {"fairness_passed": float(report["passed"])}
    for name, candidate in report["candidates"].items():
        for split in (SELECTION_SPLIT, AUDIT_SPLIT):
            metrics[f"{name}.{split}.passed"] = float(candidate[split]["passed"])
            for metric, worst in candidate[split]["worst_gap_pts"].items():
                metrics[f"{name}.{split}.worst_gap_pts.{metric}"] = worst["gap_pts"]
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"selected_model": str(report["selected_model"]), "fairness_passed": str(report["passed"])})
    mlflow.log_artifact(str(report_path), artifact_path="model_bias")
    mlflow.log_artifacts(str(report_dir), artifact_path="model_bias/report")
    for name, folder in mitigated.items():
        mlflow.log_artifacts(str(folder), artifact_path=f"mitigated_models/{name}")
    if report["selected_model_mlflow_run_id"]:
        mlflow.set_tag("selected_model_run_id", report["selected_model_mlflow_run_id"])


def run_model_bias(params: dict, data_dir="data/training", models_dir="models", out_dir=None,
                   mitigated_dir=None) -> dict:
    """Audit, mitigate, select; write bias.json + the report folder; return the bias.json content."""
    settings = bias_settings(params)
    out_dir = Path(out_dir or settings["out_dir"])
    report_dir = out_dir / REPORT_DIR
    mitigated_dir = Path(mitigated_dir or settings["mitigated_models_dir"])
    card = load_dataset_card(data_dir)
    splits = {name: load_split(name, data_dir) for name in ("train", SELECTION_SPLIT, AUDIT_SPLIT)}
    missing = [column for column in settings["slice_columns"]
               if any(column not in splits[name].columns for name in splits)]
    if missing:
        raise KeyError(f"Slice column(s) {missing} missing from the training data in {data_dir}")
    candidates = load_candidates(models_dir)
    tags = run_tags(card, data_dir)

    with tracking_run(settings, tags) as run:
        results, mitigated = {}, {}
        for name, (model, metadata) in candidates.items():
            results[name] = {**assess(model, splits, settings), "base_model": name, "mitigation": None,
                             "model_dir": str(candidate_dir(name, models_dir)), "params": metadata.get("params"),
                             "mlflow_run_id": metadata.get("mlflow_run_id")}
            logger.info("%s: validation gate %s, test gate %s", name, "passed" if results[name][SELECTION_SPLIT][
                "passed"] else "FAILED", "passed" if results[name][AUDIT_SPLIT]["passed"] else "FAILED")

        if settings["mitigation"]["enabled"]:
            for name, (_, metadata) in candidates.items():
                folder = candidate_dir(mitigated_name(name), mitigated_dir)
                if folder.exists():  # a stale model from an earlier run must not be mistaken for this run's
                    for path in (folder / MODEL_FILE, folder / METADATA_FILE):
                        path.unlink(missing_ok=True)
                validation = results[name][SELECTION_SPLIT]["audit"]
                column = root_slice(validation["flagged"], validation["table"])
                if column is None or name not in MODEL_NAMES:
                    continue
                model, info = reweigh_and_refit(name, metadata.get("params") or {}, splits["train"], column,
                                                settings["seed"])
                new = mitigated_name(name)
                results[new] = {**assess(model, splits, settings), "base_model": name, "mitigation": info,
                                "model_dir": str(folder), "params": metadata.get("params") or {}}
                run_id = log_mitigated_model(new, model, results[new], settings, tags, splits["train"]) if run else None
                results[new]["mlflow_run_id"] = run_id
                save_model(model, {
                    "model_name": new, "params": metadata.get("params") or {}, "cv_metrics": {},
                    "validation_metrics": _clean(results[new][SELECTION_SPLIT]["performance"]),
                    "mlflow_run_id": run_id,
                    "data_version": data_version(card), "git_sha": tags["git_sha"], "created_at": utc_now(),
                    "seed": settings["seed"], "base_model": name, "mitigation": info,
                }, folder)
                mitigated[new] = folder
                logger.info("%s: reweighed on %s -> %s, validation gate %s", name, column, new,
                            "passed" if results[new][SELECTION_SPLIT]["passed"] else "FAILED")

        ranking = rank(results, settings["selection_metric"])
        selected = next((name for name in ranking if results[name][SELECTION_SPLIT]["passed"]), None)
        report = build_report(results, ranking, selected, settings, card, tags)
        report["mlflow_run_id"] = run.info.run_id if run else None

        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / REPORT_FILE
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        fairness_report.write_all(report, results, settings, report_dir)
        if run:
            log_results(report, settings, report_path, report_dir, mitigated)
    logger.info("Model bias: selected %s, passed=%s -> %s", selected, report["passed"], report_path)
    return report


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Fairness audit, mitigation and fairness-aware model selection.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default=None, help="default: params.yaml model_bias.out_dir (reports/model)")
    parser.add_argument("--mitigated-dir", default=None, help="default: model_bias.mitigated_models_dir")
    parser.add_argument("--no-fail", action="store_true", help="exit 0 even when the fairness gate fails")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params(args.params)
    report = run_model_bias(params, args.data_dir, args.models_dir, args.out_dir, args.mitigated_dir)
    mitigation = report["mitigation_applied"]
    print(json.dumps({"selected_model": report["selected_model"], "passed": report["passed"],
                      "mitigation": f"{mitigation['method']} on {mitigation['slice']}" if mitigation else None,
                      "selection_reason": report["selection_reason"]}, indent=2))
    if not report["passed"] and bias_settings(params)["fail_on_gate"] and not args.no_fail:
        logger.error("Fairness gate FAILED: %s", report["selection_reason"] if report["selected_model"] is None
                     else f"{report['selected_model']} is flagged on the {AUDIT_SPLIT} split")
        raise SystemExit(1)
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
