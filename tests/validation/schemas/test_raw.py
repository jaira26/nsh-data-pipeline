import json
import shutil

import pandas as pd
import pytest

from nsh_pipeline.validation.schemas import validate_frame
from nsh_pipeline.validation.schemas.raw import RAW_SCHEMAS, check_files, raw_frame


def frame(sample_raw, collection):
    return raw_frame(sample_raw[collection])


def errors_for(df, collection):
    result = validate_frame(df, RAW_SCHEMAS[collection], collection)
    assert result["status"] == "failed"
    return result["errors"]


@pytest.fixture
def raw_copy(sample_raw_dir, tmp_path):
    return shutil.copytree(sample_raw_dir, tmp_path / "raw")


@pytest.mark.parametrize("collection", list(RAW_SCHEMAS))
def test_sample_collections_pass(sample_raw, collection):
    assert validate_frame(frame(sample_raw, collection), RAW_SCHEMAS[collection], collection)["status"] == "passed"


def test_sample_files_match_manifest(sample_raw_dir):
    manifest = json.loads((sample_raw_dir / "manifest.json").read_text())
    assert check_files(sample_raw_dir, manifest) == []


def test_missing_file_is_reported(raw_copy):
    manifest = json.loads((raw_copy / "manifest.json").read_text())
    (raw_copy / "claims.jsonl").unlink()
    missing = {"file": "claims.jsonl", "check": "exists", "message": "file is missing"}
    assert check_files(raw_copy, manifest) == [missing]


def test_changed_file_fails_row_count_and_checksum(raw_copy):
    manifest = json.loads((raw_copy / "manifest.json").read_text())
    with (raw_copy / "tickets.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"ticketNumber": "NSET-99999"}\n')
    checks = {e["check"] for e in check_files(raw_copy, manifest) if e["file"] == "tickets.jsonl"}
    assert checks == {"row_count", "sha256"}


def test_file_not_in_manifest_and_no_manifest(raw_copy):
    manifest = json.loads((raw_copy / "manifest.json").read_text())
    del manifest["files"]["bills.jsonl"]
    assert check_files(raw_copy, manifest)[0]["check"] == "manifest"
    assert check_files(raw_copy, None) == []  # without a manifest only existence is checked


def test_misspelled_field_must_be_present(sample_raw):
    df = frame(sample_raw, "wex_claims").drop(columns=["needsSubstantion"])
    assert any(e["check"] == "column_in_dataframe" and e["failure_cases"] == ["needsSubstantion"]
               for e in errors_for(df, "wex_claims"))


def test_correctly_spelled_field_fails(sample_raw):
    df = frame(sample_raw, "wex_claims").assign(needsSubstantiation=False)
    assert any(e["check"] == "no_correctly_spelled_needsSubstantiation" for e in errors_for(df, "wex_claims"))


def test_card_status_misspelling_required(sample_raw):
    df = frame(sample_raw, "debit_card_status").drop(columns=["pariticpant_status"])
    assert any(e["failure_cases"] == ["pariticpant_status"] for e in errors_for(df, "debit_card_status"))


def test_swipe_id_must_be_unique_text(sample_raw):
    df = frame(sample_raw, "wex_claims")
    as_numbers = df.assign(_id=range(len(df)))
    assert any(e["column"] == "_id" and e["check"] == "text" for e in errors_for(as_numbers, "wex_claims"))
    duplicated = pd.concat([df, df.head(1)], ignore_index=True)
    assert any(e["column"] == "_id" and "unique" in e["check"] for e in errors_for(duplicated, "wex_claims"))


@pytest.mark.parametrize("amount", [125.5, "N/A"])
def test_ledger_amount_must_be_integer_cents(sample_raw, amount):
    df = frame(sample_raw, "nse_transaction_daily_logs").astype({"amount": object})
    df.loc[0, "amount"] = amount
    assert any(e["column"] == "amount" for e in errors_for(df, "nse_transaction_daily_logs"))


def test_card_dates_must_be_mdy_strings(sample_raw):
    df = frame(sample_raw, "debit_card_status")
    df.loc[0, "valid_start_dt"] = "2025-01-01"
    assert any(e["column"] == "valid_start_dt" and "str_matches" in e["check"]
               for e in errors_for(df, "debit_card_status"))
