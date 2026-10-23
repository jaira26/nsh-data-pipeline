import pandas as pd
import pytest

from nsh_pipeline.preprocess.flatten import MEMBER_COLUMNS
from nsh_pipeline.preprocess.run import build_tables, load_raw, main, run_preprocess

AS_OF = pd.Timestamp("2025-06-15", tz="UTC")

EXPECTED_COUNTS = {
    "clients": 5, "members": 4, "member_benefits": 4, "swipes": 5, "claims": 2,
    "tickets": 2, "bill_lines": 4, "reserves": 2, "ledger": 4,
}


def test_run_preprocess_writes_all_nine_tables(sample_raw_dir, tmp_path):
    counts = run_preprocess(sample_raw_dir, tmp_path, AS_OF)
    assert counts == EXPECTED_COUNTS
    for name in EXPECTED_COUNTS:
        assert (tmp_path / f"{name}.parquet").exists()


def test_parquet_round_trip_keeps_columns_types_and_no_phi(sample_raw_dir, tmp_path):
    run_preprocess(sample_raw_dir, tmp_path, AS_OF)
    members = pd.read_parquet(tmp_path / "members.parquet")
    assert list(members.columns) == MEMBER_COLUMNS
    assert "ssn" not in members.columns
    swipes = pd.read_parquet(tmp_path / "swipes.parquet")
    assert str(swipes["service_date"].dt.tz) == "UTC"


def test_missing_raw_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="applications"):
        load_raw(tmp_path)


def test_as_of_defaults_to_today(sample_raw_dir, tmp_path):
    assert set(run_preprocess(sample_raw_dir, tmp_path)) == set(EXPECTED_COUNTS)


def test_alias_only_excluded_org_is_removed_everywhere(sample_raw_dir):
    raw = load_raw(sample_raw_dir)
    raw["applications"] = raw["applications"] + [
        {"_id": "x-demo", "dnnPortalAlias": "demo", "employer": {"name": "Second demo"}}
    ]
    raw["people"] = raw["people"] + [
        {"_id": "m-x", "Application": "x-demo", "Names": [{"FirstName": "A", "LastName": "B"}]}
    ]
    tables = build_tables(raw, AS_OF)
    assert "m-x" not in set(tables["members"]["member_id"])


def test_cli_main_runs_end_to_end(sample_raw_dir, tmp_path, capsys):
    counts = main(["--raw-dir", str(sample_raw_dir), "--out-dir", str(tmp_path), "--as-of", "2025-06-15"])
    assert counts == EXPECTED_COUNTS
    assert '"members": 4' in capsys.readouterr().out