import pandas as pd

from nsh_pipeline.generate.activity import DEFAULT_DENIAL_MODEL, denial_probability

BASE = dict(amount=75.0, category="Medical", deductible=1000.0, tenure_days=1000, recent_denials=0, is_card=True)


def p(**changes):
    return denial_probability(DEFAULT_DENIAL_MODEL, **{**BASE, **changes})


def test_probability_is_between_zero_and_one():
    assert 0 < p() < 1
    assert 0 < p(amount=5000, category="Dental", recent_denials=10, tenure_days=0, is_card=False) < 1


def test_each_documented_cause_raises_the_chance():
    assert p(amount=1000) > p()
    assert p(category="Dental") > p() and p(category="Vision") > p()
    assert p(category="Drugs & Medicine") < p()
    assert p(deductible=5000) > p()
    assert p(tenure_days=30) > p()
    assert p(recent_denials=2) > p(recent_denials=1) > p()
    assert p(is_card=False) > p()


def test_recent_denials_are_capped():
    assert p(recent_denials=3) == p(recent_denials=10)


def test_params_can_override_the_defaults():
    stricter = {**DEFAULT_DENIAL_MODEL, "intercept": -1.0}
    assert denial_probability(stricter, **BASE) > p()


def test_generated_denials_follow_the_causes(world):
    swipes = pd.DataFrame(world["wex_claims"])
    swipes = swipes[swipes["claim_status"] != "Pended"]
    denied = swipes["claim_status"].eq("Denied")
    amount = swipes["amount"].str.replace(",", "").astype(float)
    assert 0.03 < denied.mean() < 0.15
    by_category = denied.groupby(swipes["service_category"]).mean()
    assert by_category["Dental"] > by_category["Medical"]
    assert denied[amount > amount.quantile(0.75)].mean() > denied[amount < amount.quantile(0.25)].mean()