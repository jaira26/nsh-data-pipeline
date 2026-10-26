import json

import pandas as pd
import pytest
import yaml

from nsh_pipeline.bias.slicing import (
    analyse, explain_proxies, flag_slices, join_swipes, load_tables, main, paid_table, slice_table,
)

THRESHOLDS = {"min_n": 30, "abs_diff_pts": 10, "rel_ratio": 1.5, "alpha": 0.01}


# --- join -------------------------------------------------------------------------------------

def _tiny_tables():
    swipes = pd.DataFrame({
        "swipe_id": ["s1", "s2", "s3", "s4"], "client_id": ["c1", "c1", "demo", "c1"],
        "member_id": ["m1", "m2", "m3", "ghost"], "status": ["Denied", "Approved", "Approved", "Approved"],
        "paid_usd": [0.0, 20.0, 5.0, float("nan")],
    })
    clients = pd.DataFrame({"client_id": ["c1", "demo"], "state": ["CA", "OR"], "status": ["Active", "Active"],
                            "is_excluded_org": [False, True]})
    members = pd.DataFrame({"member_id": ["m1", "m2", "m3"], "client_id": ["c1", "c1", "demo"],
                            "coverage_tier": ["EE", None, "EE"], "is_vip": [False, True, False]})
    return swipes, clients, members


def test_join_drops_excluded_orgs_and_orphans_and_renames_clashing_columns():
    joined, dims, counts = join_swipes(*_tiny_tables(), ["state", "status"], ["coverage_tier", "is_vip"])
    assert joined["swipe_id"].tolist() == ["s1", "s2"]
    assert dims == ["state", "client_status", "coverage_tier", "is_vip"]
    assert joined["status"].tolist() == ["Denied", "Approved"]  # the swipe's own status is untouched
    assert joined["client_status"].tolist() == ["Active", "Active"]
    assert joined["coverage_tier"].tolist() == ["EE", "Unknown"]
    assert joined["is_vip"].tolist() == ["False", "True"]
    assert counts == {"swipes": 4, "analysed": 2, "dropped_no_client": 1, "orphan_swipe_ids": ["s4"],
                      "unparseable_paid": 0}


def test_join_on_injected_data_drops_exactly_the_planted_orphans(injected, repo_params):
    tables, issues = injected
    slices = repo_params["bias"]["slices"]
    _, _, counts = join_swipes(tables["swipes"], tables["clients"], tables["members"],
                               slices["clients"], slices["members"])
    assert counts["orphan_swipe_ids"] == sorted(issues["orphan_swipes"]["ids"])
    assert counts["unparseable_paid"] == issues["unparseable_money"]["count"]


# --- flag rules -------------------------------------------------------------------------------

def _table(rows):
    return pd.DataFrame(rows, columns=["metric", "dimension", "group", "n", "rate", "overall", "rest_rate", "p_value"])


def test_flag_rules():
    table = _table([
        ("denial_rate", "state", "big_gap", 500, 0.30, 0.10, 0.08, 1e-9),    # +20 pts
        ("denial_rate", "state", "ratio", 500, 0.04, 0.02, 0.019, 1e-9),     # +2 pts but 2x
        ("denial_rate", "state", "lower", 500, 0.01, 0.10, 0.11, 1.0),       # better off: never flagged
        ("denial_rate", "state", "small", 10, 0.90, 0.10, 0.09, 1e-9),       # below min_n
        ("denial_rate", "state", "chance", 500, 0.30, 0.10, 0.08, 0.004),    # p < alpha, not < alpha / 5
    ])
    verdicts = flag_slices(table, **THRESHOLDS).set_index("group")["verdict"]
    assert verdicts.to_dict() == {"big_gap": "flagged", "ratio": "flagged", "lower": "ok",
                                  "small": "too small to judge", "chance": "ok"}
    assert flag_slices(table, **THRESHOLDS, n_tests=1).set_index("group").loc["chance", "verdict"] == "flagged"


def test_proxy_flag_is_explained_by_the_slice_with_most_excess_denials():
    # Every "Run Out" swipe is in CA, and CA is denied far more often: Run Out is only a proxy.
    state = ["CA"] * 400 + ["OR"] * 600
    status = ["Run Out"] * 100 + ["Active"] * 900
    denied = [1] * 160 + [0] * 240 + [1] * 30 + [0] * 570  # CA 40% (Run Out 100%), OR 5%
    swipes = pd.DataFrame({"state": state, "client_status": status,
                           "status": ["Denied" if d else "Approved" for d in denied]})
    flagged = flag_slices(slice_table(swipes, ["state", "client_status"], ("denial_rate",)), **THRESHOLDS)
    explained = explain_proxies(swipes, flagged, THRESHOLDS).set_index(["dimension", "group"])
    assert explained.loc[("client_status", "Run Out"), "verdict"] == "flagged"  # largest gap of all...
    assert explained.loc[("client_status", "Run Out"), "explained_by"] == "state = CA"  # ...but a proxy
    assert pd.isna(explained.loc[("state", "CA"), "explained_by"])


def test_paid_table_averages_parseable_amounts():
    swipes = pd.DataFrame({"state": ["CA", "CA", "OR"], "paid_usd": [10.0, float("nan"), 4.0]})
    assert paid_table(swipes, ["state"]).set_index("group")["avg_paid_usd"].to_dict() == {"CA": 10.0, "OR": 4.0}


# --- end to end on generated data --------------------------------------------------------------

def test_clean_data_flags_nothing(clean, repo_params):
    result = analyse(clean[0], repo_params["bias"])
    assert not result["slices"]["verdict"].eq("flagged").any()
    assert result["mitigation"] is None
    assert result["weights"].eq(1.0).all()


def test_injected_data_flags_only_the_planted_state(injected, injected_params):
    result = analyse(injected[0], injected_params["bias"])
    slices = result["slices"]
    roots = slices[slices["verdict"].eq("flagged") & slices["explained_by"].isna()]
    skew_state = injected_params["inject_issues"]["denial_skew"]["state"]
    assert roots[["metric", "dimension", "group"]].values.tolist() == [["denial_rate", "state", skew_state]]
    mitigation = result["mitigation"]
    assert (mitigation["dimension"], mitigation["group"]) == ("state", skew_state)
    assert mitigation["max_gap_after_pts"] < 0.01 < mitigation["max_gap_before_pts"]


def test_vip_slice_is_too_small_to_judge(clean, repo_params):
    slices = analyse(clean[0], repo_params["bias"])["slices"]
    vip = slices[slices["dimension"].eq("is_vip") & slices["group"].eq("True")]
    assert vip["verdict"].eq("too small to judge").all()


# --- CLI ----------------------------------------------------------------------------------------

def _write_processed(tables, path):
    path.mkdir(parents=True)
    for name in ("swipes", "clients", "members"):
        tables[name].to_parquet(path / f"{name}.parquet", index=False)
    return path


def test_cli_writes_report_charts_and_weights(injected, injected_params, tmp_path, capsys):
    processed = _write_processed(injected[0], tmp_path / "processed")
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(injected_params), encoding="utf-8")
    summary = main(["--processed-dir", str(processed), "--params", str(params_path),
                    "--report-dir", str(tmp_path / "reports"), "--out-dir", str(tmp_path / "bias")])

    assert summary["flagged"] == ["denial_rate: state = CA"]
    assert summary["mitigation"]["dimension"] == "state"
    assert json.loads(capsys.readouterr().out)["flagged"] == summary["flagged"]
    for name in ("bias_report.md", "bias/slices.csv", "bias/denial_rate_by_state.png", "bias/reweighing_state.png"):
        assert (tmp_path / "reports" / name).stat().st_size > 0
    weights = pd.read_parquet(tmp_path / "bias" / "swipe_weights.parquet")
    assert weights.columns.tolist() == ["swipe_id", "weight"]
    assert len(weights) == summary["swipes_analysed"]


def test_cli_on_clean_data(clean, repo_params, tmp_path):
    processed = _write_processed(clean[0], tmp_path / "processed")
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(repo_params), encoding="utf-8")
    summary = main(["--processed-dir", str(processed), "--params", str(params_path),
                    "--report-dir", str(tmp_path / "reports"), "--out-dir", str(tmp_path / "bias")])
    assert summary["flagged"] == [] and summary["mitigation"] is None
    report = (tmp_path / "reports" / "bias_report.md").read_text(encoding="utf-8")
    assert "No slice is flagged" in report
    assert pd.read_parquet(tmp_path / "bias" / "swipe_weights.parquet")["weight"].eq(1.0).all()


def test_missing_processed_table_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="swipes"):
        load_tables(tmp_path)
