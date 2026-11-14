"""Per-slice model metrics, disparities and the fairness gate (phase 2, model bias).

The model's yes/no decisions (score >= threshold) are compared across the slice columns of the
training-data contract with Fairlearn's MetricFrame. Per group:

    count, positives (actual denials), base_rate
    selection_rate       share of swipes the model predicts as denied        worse when higher
    false_positive_rate  share of APPROVED swipes wrongly predicted denied   worse when higher
    recall               share of DENIED swipes the model catches            worse when lower
    precision            share of predicted denials that really were denied  worse when lower

A group is judged on a metric only if its denominator (support) has at least min_support rows;
smaller groups (e.g. is_vip = True) are reported as too small, never flagged. A group is flagged
on a gated metric when it is worse than the overall value by more than that metric's limit
(params.yaml -> model_bias.max_gap_pts) AND a one-sided Fisher exact test against all other rows
is significant at alpha / number of tests (Bonferroni), the same rule as the phase-1 data-bias
report. The gate passes when no group is flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count

from nsh_pipeline.bias.metrics import one_sided_p

RATE_METRICS = ("selection_rate", "false_positive_rate", "recall", "precision")
METRIC_LABELS = {
    "selection_rate": "Selection rate (predicted denied)",
    "false_positive_rate": "False-positive rate (approved swipes flagged)",
    "recall": "Recall (denials caught)",
    "precision": "Precision (flags that were denials)",
}
# Per metric: numerator and denominator counts, and whether a higher value is the worse one.
# The Fisher test always compares the "bad" count: flagged approvals, missed denials, false alarms.
_COUNTS = {
    "selection_rate": ("predicted_positive", "count", True),
    "false_positive_rate": ("fp", "negatives", True),
    "recall": ("tp", "positives", False),
    "precision": ("tp", "predicted_positive", False),
}

TABLE_COLUMNS = ["slice", "group", "count", "positives", "negatives", "predicted_positive", "tp", "fp", "fn", "tn",
                 "base_rate", *RATE_METRICS]
FLAG_COLUMNS = ["slice", "group", "metric", "value", "overall", "gap_pts", "support", "judged", "p_value",
                "limit_pts", "flagged"]


def _tp(y_true, y_pred):
    return int(((y_true == 1) & (y_pred == 1)).sum())


def _fp(y_true, y_pred):
    return int(((y_true == 0) & (y_pred == 1)).sum())


def _fn(y_true, y_pred):
    return int(((y_true == 1) & (y_pred == 0)).sum())


def _tn(y_true, y_pred):
    return int(((y_true == 0) & (y_pred == 0)).sum())


def _ratio(numerator, denominator):
    """numerator / denominator, NaN where the denominator is 0 (metric undefined for that group)."""
    numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def _with_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Add derived counts and the rate metrics to a frame of tp / fp / fn / tn / count."""
    out = counts.copy()
    out["positives"] = out["tp"] + out["fn"]
    out["negatives"] = out["fp"] + out["tn"]
    out["predicted_positive"] = out["tp"] + out["fp"]
    out["base_rate"] = _ratio(out["positives"], out["count"])
    for metric, (numerator, denominator, _) in _COUNTS.items():
        out[metric] = _ratio(out[numerator], out[denominator])
    return out


def predictions(scores, threshold: float) -> np.ndarray:
    """0/1 decisions: 1 (predicted denied) when the score reaches the threshold."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def slice_metrics(frame: pd.DataFrame, y_true, y_pred, slice_columns) -> tuple[pd.DataFrame, dict]:
    """Per-group metrics for every slice column (one row per slice + group) and the overall metrics.

    A missing slice value becomes its own group, "Unknown", so no row silently drops out.
    """
    y_true = pd.Series(np.asarray(y_true).astype(int), index=frame.index)
    y_pred = pd.Series(np.asarray(y_pred).astype(int), index=frame.index)
    metrics = {"count": count, "tp": _tp, "fp": _fp, "fn": _fn, "tn": _tn}
    tables, overall = [], None
    for column in slice_columns:
        groups = frame[column].astype(object).where(frame[column].notna(), "Unknown").astype(str)
        result = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=groups.rename(column))
        by_group = _with_rates(result.by_group.astype(int)).reset_index().rename(columns={column: "group"})
        by_group.insert(0, "slice", column)
        tables.append(by_group)
        if overall is None:
            overall = _with_rates(result.overall.astype(int).to_frame().T).iloc[0].to_dict()
    if overall is None:  # no slice columns: overall metrics only
        counts = {name: fn(y_true, y_pred) for name, fn in metrics.items() if name != "count"}
        overall = _with_rates(pd.DataFrame([{**counts, "count": len(y_true)}])).iloc[0].to_dict()
    table = pd.concat(tables, ignore_index=True)[TABLE_COLUMNS] if tables else pd.DataFrame(columns=TABLE_COLUMNS)
    overall = {key: (int(value) if key in ("count", "positives", "negatives", "predicted_positive", "tp", "fp",
                                           "fn", "tn") else float(value)) for key, value in overall.items()}
    return table, overall


def _bad_counts(row, metric: str) -> tuple[int, int]:
    """(bad outcomes, support) for one group: flagged approvals, all predicted, missed denials, false alarms."""
    if metric == "selection_rate":
        return int(row["predicted_positive"]), int(row["count"])
    if metric == "false_positive_rate":
        return int(row["fp"]), int(row["negatives"])
    if metric == "recall":
        return int(row["fn"]), int(row["positives"])
    return int(row["fp"]), int(row["predicted_positive"])  # precision: false alarms among predicted denials


def flag_disparities(table: pd.DataFrame, overall: dict, max_gap_pts: dict, min_support: int,
                     alpha: float) -> pd.DataFrame:
    """One row per slice, group and gated metric (a metric with a limit in max_gap_pts).

    gap_pts > 0 means the group is worse off than overall (in percentage points). Judged = support
    >= min_support. Flagged = judged, gap above the metric's limit, and Fisher p < alpha / n_tests.
    """
    gated = {metric: float(limit) for metric, limit in (max_gap_pts or {}).items() if limit is not None}
    unknown = [metric for metric in gated if metric not in RATE_METRICS]
    if unknown:
        raise ValueError(f"Unknown fairness metric(s) {unknown}; expected some of {RATE_METRICS}")
    rows = []
    for _, row in table.iterrows():
        rest = table[(table["slice"] == row["slice"]) & (table["group"] != row["group"])]
        for metric, limit in gated.items():
            higher_is_worse = _COUNTS[metric][2]
            value, base = float(row[metric]), float(overall[metric])
            bad, support = _bad_counts(row, metric)
            rest_bad = sum(_bad_counts(other, metric)[0] for _, other in rest.iterrows())
            rest_support = sum(_bad_counts(other, metric)[1] for _, other in rest.iterrows())
            judged = support >= min_support and not np.isnan(value)
            gap = (value - base if higher_is_worse else base - value) * 100 if judged else np.nan
            p_value = one_sided_p(bad, support, rest_bad, rest_support) if judged else np.nan
            rows.append({"slice": row["slice"], "group": row["group"], "metric": metric, "value": value,
                         "overall": base, "gap_pts": gap, "support": support, "judged": judged,
                         "p_value": p_value, "limit_pts": limit})
    flags = pd.DataFrame(rows, columns=FLAG_COLUMNS[:-1])
    n_tests = max(int(flags["judged"].sum()), 1) if len(flags) else 1
    flags["flagged"] = (flags["judged"] & flags["gap_pts"].gt(flags["limit_pts"])
                        & flags["p_value"].lt(alpha / n_tests)).astype(bool)
    flags.attrs["n_tests"] = n_tests
    flags.attrs["alpha_corrected"] = alpha / n_tests
    return flags


def max_disparity(table: pd.DataFrame, min_support: int, metrics=RATE_METRICS) -> dict:
    """Largest gap between two groups of each slice, per metric, among groups with enough support.

    {slice: {metric: {gap_pts, worst_group, worst_value, best_group, best_value, groups_judged,
    groups_too_small}}}; gap_pts is None when fewer than two groups can be judged.
    """
    out: dict = {}
    for column, groups in table.groupby("slice", sort=False):
        out[column] = {}
        for metric in metrics:
            numerator, denominator, higher_is_worse = _COUNTS[metric]
            judged = groups[groups[denominator].ge(min_support) & groups[metric].notna()]
            entry = {"gap_pts": None, "worst_group": None, "worst_value": None, "best_group": None,
                     "best_value": None, "groups_judged": int(len(judged)),
                     "groups_too_small": sorted(set(groups["group"]) - set(judged["group"]))}
            if len(judged) >= 2:
                ordered = judged.sort_values(metric, ascending=not higher_is_worse)
                worst, best = ordered.iloc[0], ordered.iloc[-1]
                entry.update(gap_pts=round(abs(float(worst[metric]) - float(best[metric])) * 100, 2),
                             worst_group=str(worst["group"]), worst_value=round(float(worst[metric]), 4),
                             best_group=str(best["group"]), best_value=round(float(best[metric]), 4))
            out[column][metric] = entry
    return out


def audit(frame: pd.DataFrame, y_true, scores, threshold: float, slice_columns, settings: dict) -> dict:
    """Everything the fairness gate needs for one model on one split.

    settings: min_support, alpha, max_gap_pts ({metric: limit in points, or None = report only}).
    """
    y_pred = predictions(scores, threshold)
    table, overall = slice_metrics(frame, y_true, y_pred, slice_columns)
    flags = flag_disparities(table, overall, settings["max_gap_pts"], settings["min_support"], settings["alpha"])
    flagged = flags[flags["flagged"]]
    return {
        "threshold": float(threshold),
        "overall": overall,
        "table": table,
        "flags": flags,
        "flagged": flagged,
        "max_disparity": max_disparity(table, settings["min_support"]),
        "n_tests": flags.attrs["n_tests"],
        "alpha_corrected": flags.attrs["alpha_corrected"],
        "passed": flagged.empty,
    }


def root_slice(flagged: pd.DataFrame, table: pd.DataFrame) -> str | None:
    """Slice column of the flag with the most excess bad outcomes (support x gap): the one to mitigate.

    Same idea as the phase-1 proxy root: one large skewed group outweighs a small one with a
    bigger gap, and flags on other columns are often proxies of it.
    """
    if flagged.empty:
        return None
    excess = flagged["support"] * flagged["gap_pts"]
    return str(flagged.loc[excess.idxmax(), "slice"])
