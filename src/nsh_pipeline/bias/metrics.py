"""Per-slice outcome rates for bias analysis, computed with Fairlearn's MetricFrame.

The pipeline has no model yet, so a slice's "performance" is the outcome its members get:
how often their swipes are denied, and how often a paid debit-card swipe needs substantiation.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count
from scipy.stats import fisher_exact

RATE_METRICS = ("denial_rate", "substantiation_rate")
METRIC_LABELS = {"denial_rate": "Denial rate", "substantiation_rate": "Substantiation rate"}


def metric_outcome(swipes: pd.DataFrame, metric: str) -> pd.Series:
    """0/1 outcome per swipe for `metric`, only on the swipes the metric applies to.

    denial_rate: every swipe; 1 if status is Denied.
    substantiation_rate: debit-card swipes with paid_usd > 0; 1 if it needs substantiation.
    """
    if metric == "denial_rate":
        return swipes["status"].eq("Denied").astype(int)
    if metric == "substantiation_rate":
        paid_debit = swipes["is_debit_card"] & swipes["paid_usd"].gt(0)
        return swipes.loc[paid_debit, "needs_substantiation"].astype(int)
    raise ValueError(f"Unknown metric {metric!r}; expected one of {RATE_METRICS}")


def _rate(y_true, y_pred, sample_weight=None) -> float:
    """Share of rows whose outcome is 1. y_pred is unused: there are only observed outcomes."""
    return float(np.average(y_true, weights=sample_weight))


def group_rates(outcome: pd.Series, groups: pd.Series, sample_weight=None) -> tuple[pd.DataFrame, float]:
    """Rate and row count per group (index = group), plus the overall rate."""
    sample_params = None if sample_weight is None else {"rate": {"sample_weight": np.asarray(sample_weight)}}
    frame = MetricFrame(metrics={"rate": _rate, "n": count}, y_true=outcome, y_pred=outcome,
                        sensitive_features=groups, sample_params=sample_params)
    by_group = frame.by_group.astype({"rate": float, "n": int})
    return by_group, float(frame.overall["rate"])


def one_sided_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """p-value that rate x1/n1 is higher than x2/n2 (one-sided Fisher exact test).

    Exact rather than a z-test: substantiation is rare (about 2.5%), so a small slice expects
    only a few events and the normal approximation would overstate significance.
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="greater").pvalue)
