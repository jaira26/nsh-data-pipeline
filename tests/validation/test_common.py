import logging

import pandas as pd
import pytest

from nsh_pipeline.validation.common import (
    ValidationFailed, enforce_mode, load_tables, validation_settings, write_json,
)


def test_validation_settings_requires_a_valid_section():
    assert validation_settings({"validation": {"mode": "report"}}) == {"mode": "report"}
    with pytest.raises(ValueError, match="no 'validation:' section"):
        validation_settings({})
    with pytest.raises(ValueError, match="validation.mode must be one of"):
        validation_settings({"validation": {"mode": "loud"}})


def test_enforce_mode(caplog):
    enforce_mode("strict", False, "fine")
    with pytest.raises(ValidationFailed, match="broken"):
        enforce_mode("strict", True, "broken")
    with caplog.at_level(logging.WARNING):
        enforce_mode("report", True, "broken")
    assert "broken (validation.mode=report: continuing)" in caplog.text


def test_load_tables(tmp_path):
    assert load_tables(tmp_path / "missing") == {}
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "one.parquet")
    pd.DataFrame({"b": [2]}).to_parquet(tmp_path / "two.parquet")
    assert sorted(load_tables(tmp_path)) == ["one", "two"]
    assert list(load_tables(tmp_path, ["two", "three"])) == ["two"]


def test_write_json_creates_folders(tmp_path):
    path = write_json({"when": pd.Timestamp("2025-01-01")}, tmp_path / "a" / "b.json")
    assert path.read_text(encoding="utf-8") == '{\n  "when": "2025-01-01 00:00:00"\n}\n'
