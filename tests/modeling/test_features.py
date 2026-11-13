import math

import pandas as pd
import pytest

from nsh_pipeline.generate.run import build_collections
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, LABEL_COLUMN, TRAINING_COLUMNS
from nsh_pipeline.modeling.features import build_training_frame, plan_deductible, prior_denials
from nsh_pipeline.preprocess.run import build_tables, load_raw

AS_OF = "2025-06-15"


@pytest.fixture(scope="module")
def sample_frame(sample_raw_dir):
    """Training rows from the hand-built phase-1 sample (tests/fixtures/raw)."""
    return build_training_frame(build_tables(load_raw(sample_raw_dir), AS_OF)).set_index("swipe_id")


def test_columns_follow_the_contract(sample_frame):
    assert list(sample_frame.reset_index().columns) == TRAINING_COLUMNS


def test_pended_and_excluded_org_swipes_are_dropped(sample_frame):
    # WEX-000005 is Pended, WEX-000006 belongs to NS Demo; the other four are decided
    assert sorted(sample_frame.index) == ["WEX-000001", "WEX-000002", "WEX-000003", "WEX-000004"]


def test_label_marks_denied_swipes(sample_frame):
    assert sample_frame[LABEL_COLUMN].to_dict() == {
        "WEX-000001": 0, "WEX-000002": 0, "WEX-000003": 1, "WEX-000004": 0}


def test_member_client_and_plan_features_are_joined(sample_frame):
    jane = sample_frame.loc["WEX-000001"]
    assert jane["state"] == "OR"
    assert jane["group_size_bucket"] == "25-49"
    assert jane["age_band"] == "35-44"
    assert jane["coverage_tier"] == "ES"
    assert jane["carrier"] == "Aetna"
    assert jane["plan_deductible"] == 2000.0
    assert jane["tenure_days"] == (pd.Timestamp("2025-02-03") - pd.Timestamp("2020-03-01")).days
    assert jane["service_month"] == 2


@pytest.mark.parametrize("name, expected", [
    ("NSW Aetna 2000/4000", 2000.0), ("NSW Gold 5000/10000", 5000.0),
])
def test_plan_deductible_reads_the_first_pair(name, expected):
    assert plan_deductible(name) == expected


@pytest.mark.parametrize("name", ["Cigna PPO Benchmark", None])
def test_plan_deductible_missing_is_nan(name):
    assert math.isnan(plan_deductible(name))


def test_prior_denials_only_count_strictly_earlier_within_90_days():
    dates = pd.to_datetime(["2025-01-01", "2025-01-10", "2025-01-10", "2025-02-01", "2025-06-01"], utc=True)
    swipes = pd.DataFrame({
        "swipe_id": ["a", "b", "c", "d", "e"],
        "member_id": ["m1"] * 5,
        "service_date": dates,
        "status": ["Denied", "Denied", "Approved", "Approved", "Approved"],
    })
    # b and c: one denial (Jan 1) earlier; the same-day denial b does not count for c
    # d: both January denials; e: none within 90 days
    assert prior_denials(swipes).tolist() == [0, 1, 1, 2, 0]


def test_orphan_swipes_are_dropped(sample_raw_dir):
    tables = build_tables(load_raw(sample_raw_dir), AS_OF)
    ghost = tables["swipes"].iloc[[0]].assign(swipe_id="WEX-GHOST", member_id="no-such-member")
    tables["swipes"] = pd.concat([tables["swipes"], ghost], ignore_index=True)
    assert "WEX-GHOST" not in set(build_training_frame(tables)["swipe_id"])


def test_generated_data_gives_complete_learnable_rows():
    params = {"seed": 42, "scale": "test", "reference_date": AS_OF,
              "scales": {"test": {"clients": 20, "max_members_per_client": 30}}}
    collections, _ = build_collections(params)
    frame = build_training_frame(build_tables(collections, AS_OF))
    assert frame[FEATURE_COLUMNS].notna().all().all()
    assert 0.03 < frame[LABEL_COLUMN].mean() < 0.25
    by_category = frame.groupby("service_category")[LABEL_COLUMN].mean()
    assert by_category["Dental"] > by_category["Medical"]
    assert frame["service_date"].is_monotonic_increasing