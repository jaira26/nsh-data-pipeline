import copy

import pandas as pd
import pytest

from nsh_pipeline.validation.drift import compare_stats, headline, run_drift
from nsh_pipeline.validation.stats import compute_stats, read_stats


@pytest.fixture
def cfg(settings):
    return settings["drift"]


@pytest.fixture
def stats(clean_tables, settings):
    return compute_stats(clean_tables, settings)


def swipe_stats(statuses, amounts=None, settings=None):
    n = len(statuses)
    df = pd.DataFrame({"status": statuses, "amount_usd": amounts or [100.0] * n, "is_debit_card": [True] * n})
    return compute_stats({"swipes": df}, settings or {})


def metrics(findings):
    return {(f["table"], f["column"], f["metric"]) for f in findings}


def test_no_baseline_writes_one_and_does_not_fail(stats, cfg, tmp_path):
    result = run_drift(stats, tmp_path / "baseline", cfg)
    assert result == {"status": "no_baseline", "baseline_dir": str(tmp_path / "baseline"),
                      "headline": None, "findings": []}
    assert sorted(read_stats(tmp_path / "baseline")) == sorted(stats)


def test_identical_stats_are_ok(stats, cfg, tmp_path):
    run_drift(stats, tmp_path / "baseline", cfg)
    result = run_drift(stats, tmp_path / "baseline", cfg)
    assert result["status"] == "ok" and result["findings"] == []
    assert result["headline"]["change_pts"] == 0.0 and result["headline"]["drifted"] is False


def test_update_baseline_replaces_it(stats, cfg, tmp_path):
    run_drift(stats, tmp_path / "baseline", cfg)
    smaller = {"swipes": stats["swipes"]}
    assert run_drift(smaller, tmp_path / "baseline", cfg, update_baseline=True)["status"] == "baseline_updated"
    assert sorted(read_stats(tmp_path / "baseline")) == ["swipes"]


def test_category_share_shift_is_the_headline(cfg, settings):
    base = swipe_stats(["Approved"] * 95 + ["Denied"] * 5, settings=settings)
    cur = swipe_stats(["Approved"] * 90 + ["Denied"] * 10, settings=settings)
    found = metrics(compare_stats(cur, base, cfg))
    assert ("swipes", "status", "share of 'Denied'") in found
    assert ("swipes", "status", "share of 'Approved'") in found
    top = headline(cur, base, cfg)
    assert top["baseline_pct"] == 5.0 and top["current_pct"] == 10.0
    assert top["change_pts"] == 5.0 and top["drifted"] is True


def test_small_share_shift_is_ignored(cfg, settings):
    base = swipe_stats(["Approved"] * 95 + ["Denied"] * 5, settings=settings)
    cur = swipe_stats(["Approved"] * 94 + ["Denied"] * 6, settings=settings)
    assert compare_stats(cur, base, cfg) == []
    assert headline(cur, base, cfg)["drifted"] is False


def test_new_category_counts_from_zero_when_top_k_is_complete(cfg, settings):
    base = swipe_stats(["Approved"] * 100, settings=settings)
    cur = swipe_stats(["Approved"] * 90 + ["Hold"] * 10, settings=settings)
    assert ("swipes", "status", "share of 'Hold'") in metrics(compare_stats(cur, base, cfg))
    assert headline(cur, base, cfg)["baseline_pct"] == 0.0  # Denied absent from a complete list = 0%


def test_row_count_mean_and_null_rate(cfg, stats):
    cur = copy.deepcopy(stats)
    cur["swipes"]["row_count"] = int(stats["swipes"]["row_count"] * 1.5)
    amount = cur["swipes"]["columns"]["amount_usd"]
    amount["mean"] = amount["mean"] * 2
    amount["null_rate"] = (amount["null_rate"] or 0) + 0.05
    found = metrics(compare_stats(cur, stats, cfg))
    assert {("swipes", None, "row_count"), ("swipes", "amount_usd", "mean"),
            ("swipes", "amount_usd", "null_rate")} <= found


def test_small_row_count_change_is_ignored(cfg, stats):
    cur = copy.deepcopy(stats)
    cur["bill_lines"]["row_count"] = round(stats["bill_lines"]["row_count"] * 1.05)
    assert compare_stats(cur, stats, cfg) == []


def test_schema_drift_tables_columns_and_kinds(cfg, stats):
    cur = copy.deepcopy(stats)
    del cur["reserves"]
    cur["new_table"] = cur["claims"]
    cur["swipes"]["columns"]["extra"] = cur["swipes"]["columns"].pop("nswid")
    cur["swipes"]["columns"]["status"]["kind"] = "number"
    found = {m for _, _, m in metrics(compare_stats(cur, stats, cfg))}
    assert {"table missing (in baseline only)", "table added (not in baseline)",
            "column added", "column removed", "kind text -> number"} <= found


def test_bool_true_rate_shift(cfg, stats):
    cur = copy.deepcopy(stats)
    cur["swipes"]["columns"]["is_debit_card"]["true_rate"] = 0.0
    assert ("swipes", "is_debit_card", "true_rate") in metrics(compare_stats(cur, stats, cfg))


def test_headline_missing_or_disabled(cfg, stats):
    assert headline(stats, stats, {**cfg, "headline": None}) is None
    spec = {**cfg, "headline": {"table": "nope", "column": "x", "value": "y"}}
    assert headline(stats, stats, spec)["change_pts"] is None
