import numpy as np
import pandas as pd
import pytest

from nsh_pipeline.bias.metrics import group_rates, metric_outcome, one_sided_p


def _swipes():
    return pd.DataFrame({
        "status": ["Denied", "Approved", "Denied", "Pended"],
        "is_debit_card": [True, True, False, True],
        "paid_usd": [10.0, 0.0, 5.0, np.nan],
        "needs_substantiation": [True, False, True, False],
    })


def test_denial_outcome_covers_every_swipe():
    assert metric_outcome(_swipes(), "denial_rate").tolist() == [1, 0, 1, 0]


def test_substantiation_outcome_only_on_paid_debit_card_swipes():
    outcome = metric_outcome(_swipes(), "substantiation_rate")
    assert outcome.index.tolist() == [0]  # unpaid, non-debit and unparseable (NaN) swipes are left out
    assert outcome.tolist() == [1]


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="premium_rate"):
        metric_outcome(_swipes(), "premium_rate")


def test_group_rates_match_pandas_and_respect_weights():
    outcome = pd.Series([1, 0, 0, 1, 1, 0])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"], name="state")
    by_group, overall = group_rates(outcome, groups)
    assert by_group["rate"].to_dict() == pytest.approx(outcome.groupby(groups).mean().to_dict())
    assert by_group["n"].to_dict() == {"a": 3, "b": 3}
    assert overall == pytest.approx(0.5)

    weighted, _ = group_rates(outcome, groups, sample_weight=[2, 1, 1, 1, 1, 1])
    assert weighted.loc["a", "rate"] == pytest.approx(0.5)  # (2*1) / (2+1+1)


def test_one_sided_p():
    assert one_sided_p(30, 100, 5, 100) < 0.001  # clearly higher
    assert one_sided_p(5, 100, 30, 100) > 0.99  # lower: not "worse off"
    assert one_sided_p(0, 0, 5, 100) == 1.0
    assert one_sided_p(5, 100, 0, 0) == 1.0
