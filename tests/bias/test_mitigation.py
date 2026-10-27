import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.bias.mitigation import effective_sample_size, reweigh, reweighing_weights


@pytest.fixture
def skewed():
    groups = pd.Series(["CA"] * 100 + ["OR"] * 300, name="state")
    outcome = pd.Series([1] * 30 + [0] * 70 + [1] * 15 + [0] * 285)  # CA 30% denied, OR 5%
    return groups, outcome


def test_weights_make_every_group_match_the_overall_rate(skewed):
    groups, outcome = skewed
    weights = reweighing_weights(groups, outcome)
    assert weights.sum() == pytest.approx(len(weights))  # weights average 1
    for group in ("CA", "OR"):
        mask = groups.eq(group)
        assert np.average(outcome[mask], weights=weights[mask]) == pytest.approx(outcome.mean())
    assert weights[0] == pytest.approx(outcome.mean() / 0.30)  # a denied CA swipe counts less


def test_effective_sample_size():
    assert effective_sample_size([1, 1, 1, 1]) == pytest.approx(4)
    assert effective_sample_size([0.5, 1.5, 1, 1]) < 4


def test_reweigh_reports_before_after_and_cost(skewed):
    groups, outcome = skewed
    result = reweigh(outcome, groups)
    assert result["comparison"].index[0] == "CA"  # sorted by the before rate
    assert result["comparison"].loc["CA", "before"] == pytest.approx(0.30)
    assert result["max_gap_before_pts"] == pytest.approx(25.0)
    assert result["max_gap_after_pts"] == pytest.approx(0.0, abs=1e-9)
    assert result["effective_n"] < result["n"] == 400
    assert result["weight_min"] < 1 < result["weight_max"]
