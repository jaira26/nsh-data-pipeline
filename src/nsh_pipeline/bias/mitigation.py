"""Bias mitigation by reweighing (Kamiran & Calders, 2012).

Every row gets the weight P(group) * P(outcome) / P(group, outcome). Under these weights the
outcome rate is the same in every group, so a model trained with them as `sample_weight` does
not learn the flagged group's skew. Rows and outcomes are left untouched; the cost is a smaller
effective sample size, reported next to the before/after rates.
"""
import numpy as np
import pandas as pd

from nsh_pipeline.bias.metrics import group_rates


def reweighing_weights(groups: pd.Series, outcome: pd.Series) -> pd.Series:
    """Weight per row that makes `outcome` independent of `groups`; weights average 1."""
    frame = pd.DataFrame({"g": groups.to_numpy(), "y": outcome.to_numpy()}, index=outcome.index)
    p_group = frame["g"].map(frame["g"].value_counts(normalize=True))
    p_outcome = frame["y"].map(frame["y"].value_counts(normalize=True))
    p_joint = frame.groupby(["g", "y"])["y"].transform("size") / len(frame)
    return (p_group * p_outcome / p_joint).rename("weight")


def effective_sample_size(weights) -> float:
    """Kish's effective sample size: how many unweighted rows the weighted data is worth."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())


def reweigh(outcome: pd.Series, groups: pd.Series) -> dict:
    """Reweigh on `groups` and compare each group's rate before and after."""
    weights = reweighing_weights(groups, outcome)
    before, overall = group_rates(outcome, groups)
    after, _ = group_rates(outcome, groups, sample_weight=weights)
    comparison = pd.DataFrame({"n": before["n"], "before": before["rate"], "after": after["rate"]})
    return {
        "weights": weights,
        "comparison": comparison.sort_values("before", ascending=False),
        "overall": overall,
        "max_gap_before_pts": float(np.ptp(comparison["before"]) * 100),
        "max_gap_after_pts": float(np.ptp(comparison["after"]) * 100),
        "n": len(weights),
        "effective_n": effective_sample_size(weights),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
