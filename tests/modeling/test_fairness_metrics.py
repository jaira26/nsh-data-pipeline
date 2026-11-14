"""Per-slice model metrics, disparity flags and the fairness gate (modeling/fairness_metrics.py)."""
import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.modeling.fairness_metrics import (
    RATE_METRICS, audit, flag_disparities, max_disparity, predictions, root_slice, slice_metrics,
)

SETTINGS = {"min_support": 30, "alpha": 0.01, "max_gap_pts": {"false_positive_rate": 20, "recall": 20}}


def decisions(n=3000, seed=0, groups=("CA", "OR", "WA"), shares=(0.3, 0.4, 0.3), base_rate=0.15,
              tpr=0.7, fpr=0.2, skew=None):
    """Synthetic labels and 0/1 decisions; skew = {group: {"fpr": x} or {"tpr": x}} plants a disparity."""
    rng = np.random.default_rng(seed)
    state = rng.choice(list(groups), size=n, p=list(shares))
    y = rng.binomial(1, base_rate, n)
    tpr_row = np.full(n, tpr, dtype=float)
    fpr_row = np.full(n, fpr, dtype=float)
    for group, rates in (skew or {}).items():
        tpr_row[state == group] = rates.get("tpr", tpr)
        fpr_row[state == group] = rates.get("fpr", fpr)
    pred = np.where(y == 1, rng.random(n) < tpr_row, rng.random(n) < fpr_row).astype(int)
    frame = pd.DataFrame({"state": state, "tier": rng.choice(["EE", "EF"], size=n)})
    return frame, y, pred


def test_slice_metrics_counts_and_rates_on_a_hand_made_example():
    frame = pd.DataFrame({"state": ["A"] * 4 + ["B"] * 4})
    y_true = [1, 1, 0, 0, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 0, 1, 1, 1, 0]
    table, overall = slice_metrics(frame, y_true, y_pred, ["state"])
    a = table.set_index("group").loc["A"]
    assert (a["count"], a["tp"], a["fn"], a["fp"], a["tn"]) == (4, 1, 1, 1, 1)
    assert a["selection_rate"] == 0.5 and a["recall"] == 0.5 and a["false_positive_rate"] == 0.5
    assert a["precision"] == 0.5 and a["base_rate"] == 0.5
    b = table.set_index("group").loc["B"]
    assert b["false_positive_rate"] == pytest.approx(2 / 3) and b["recall"] == 1.0
    assert overall["count"] == 8 and overall["positives"] == 3
    assert overall["false_positive_rate"] == pytest.approx(3 / 5)
    assert overall["recall"] == pytest.approx(2 / 3)


def test_missing_slice_value_is_its_own_group_and_undefined_rates_are_nan():
    frame = pd.DataFrame({"state": ["A", None, None]})
    table, _ = slice_metrics(frame, [0, 0, 0], [0, 1, 0], ["state"])
    unknown = table.set_index("group").loc["Unknown"]
    assert unknown["count"] == 2
    assert np.isnan(unknown["recall"])  # no denials in the group: recall undefined, not 0


def test_overall_only_when_no_slice_columns():
    table, overall = slice_metrics(pd.DataFrame(index=range(4)), [1, 0, 1, 0], [1, 1, 0, 0], [])
    assert table.empty
    assert overall["recall"] == 0.5 and overall["false_positive_rate"] == 0.5


def test_predictions_threshold_is_inclusive():
    assert predictions([0.2, 0.5, 0.7], 0.5).tolist() == [0, 1, 1]


def test_planted_false_positive_disparity_is_flagged():
    frame, y, pred = decisions(skew={"CA": {"fpr": 0.6}})
    result = audit(frame, y, pred, 0.5, ["state"], SETTINGS)
    flagged = result["flagged"]
    assert not result["passed"]
    assert list(zip(flagged["group"], flagged["metric"])) == [("CA", "false_positive_rate")]
    row = flagged.iloc[0]
    assert row["gap_pts"] > 20 and row["p_value"] < result["alpha_corrected"]


def test_planted_recall_disparity_is_flagged_in_the_worse_direction():
    frame, y, pred = decisions(n=6000, skew={"OR": {"tpr": 0.25}})
    result = audit(frame, y, pred, 0.5, ["state"], SETTINGS)
    assert list(zip(result["flagged"]["group"], result["flagged"]["metric"])) == [("OR", "recall")]
    better = result["flags"].query("metric == 'recall' and group != 'OR'")
    assert (better["gap_pts"] < 0).all()  # groups doing better than overall are never "worse off"


@pytest.mark.parametrize("seed", range(5))
def test_balanced_data_is_not_flagged(seed):
    frame, y, pred = decisions(seed=seed)
    result = audit(frame, y, pred, 0.5, ["state", "tier"], SETTINGS)
    assert result["passed"]
    assert result["flagged"].empty


def test_small_groups_are_skipped_not_flagged():
    frame, y, pred = decisions(groups=("CA", "OR", "VIP"), shares=(0.5, 0.495, 0.005), skew={"VIP": {"fpr": 1.0}})
    result = audit(frame, y, pred, 0.5, ["state"], SETTINGS)
    vip = result["flags"][result["flags"]["group"] == "VIP"]
    assert not vip["judged"].any() and not vip["flagged"].any()
    assert vip["gap_pts"].isna().all()
    assert "VIP" in result["max_disparity"]["state"]["false_positive_rate"]["groups_too_small"]
    assert result["passed"]


def test_gate_fails_only_above_the_limit():
    frame, y, pred = decisions(n=20000, skew={"CA": {"fpr": 0.3}})  # ~7-pt gap, highly significant at this n
    loose = audit(frame, y, pred, 0.5, ["state"], SETTINGS)
    assert loose["passed"]
    strict = audit(frame, y, pred, 0.5, ["state"], {**SETTINGS, "max_gap_pts": {"false_positive_rate": 5}})
    assert not strict["passed"]
    assert strict["flagged"]["group"].tolist() == ["CA"]


def test_large_gap_that_is_not_significant_is_not_flagged():
    # 35 approved swipes in WA: a 15-pt gap is plausible by chance after Bonferroni
    frame = pd.DataFrame({"state": ["WA"] * 35 + ["OR"] * 400})
    y = np.zeros(435, dtype=int)
    pred = np.r_[np.ones(13), np.zeros(22), np.ones(88), np.zeros(312)].astype(int)  # 37% vs 22%
    result = audit(frame, y, pred, 0.5, ["state"], {**SETTINGS, "max_gap_pts": {"false_positive_rate": 10}})
    wa = result["flags"].query("group == 'WA'").iloc[0]
    assert wa["gap_pts"] > 10 and wa["judged"]
    assert not wa["flagged"]


def test_bonferroni_counts_only_judged_tests():
    frame, y, pred = decisions(groups=("CA", "OR", "VIP"), shares=(0.5, 0.495, 0.005))
    flags = flag_disparities(*slice_metrics(frame, y, pred, ["state"]), SETTINGS["max_gap_pts"], 30, 0.01)
    assert flags.attrs["n_tests"] == int(flags["judged"].sum())
    assert flags.attrs["alpha_corrected"] == pytest.approx(0.01 / flags.attrs["n_tests"])


def test_unknown_metric_is_rejected():
    frame, y, pred = decisions(n=200)
    with pytest.raises(ValueError, match="Unknown fairness metric"):
        flag_disparities(*slice_metrics(frame, y, pred, ["state"]), {"accuracy": 5}, 30, 0.01)


def test_null_limit_means_reported_not_gated():
    frame, y, pred = decisions(skew={"CA": {"fpr": 0.6}})
    result = audit(frame, y, pred, 0.5, ["state"], {**SETTINGS, "max_gap_pts": {"recall": 20,
                                                                                 "false_positive_rate": None}})
    assert result["passed"]
    assert set(result["flags"]["metric"]) == {"recall"}
    assert result["max_disparity"]["state"]["false_positive_rate"]["gap_pts"] > 30  # still reported


def test_max_disparity_reports_worst_and_best_group_per_metric():
    frame = pd.DataFrame({"state": ["A"] * 40 + ["B"] * 40 + ["C"] * 5})
    y = np.zeros(85, dtype=int)
    pred = np.r_[np.ones(20), np.zeros(20), np.ones(4), np.zeros(36), np.ones(5)].astype(int)
    table, _ = slice_metrics(frame, y, pred, ["state"])
    fpr = max_disparity(table, min_support=30)["state"]["false_positive_rate"]
    assert (fpr["worst_group"], fpr["best_group"]) == ("A", "B")
    assert fpr["gap_pts"] == pytest.approx(40.0)
    assert fpr["groups_judged"] == 2 and fpr["groups_too_small"] == ["C"]
    recall = max_disparity(table, min_support=30)["state"]["recall"]
    assert recall["gap_pts"] is None  # no denials anywhere: nothing to compare
    assert set(max_disparity(table, 30)["state"]) == set(RATE_METRICS)


def test_root_slice_is_the_flag_with_most_excess():
    flagged = pd.DataFrame({"slice": ["age_band", "state"], "support": [40, 300], "gap_pts": [40.0, 25.0]})
    assert root_slice(flagged, pd.DataFrame()) == "state"
    assert root_slice(flagged.iloc[0:0], pd.DataFrame()) is None
