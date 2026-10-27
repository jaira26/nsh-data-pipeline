"""Fixtures for the validation tests: processed tables built from the hand-made sample dataset."""
import copy
from pathlib import Path

import pytest

from nsh_pipeline.config import load_params
from nsh_pipeline.preprocess.run import build_tables, load_raw

ROOT = Path(__file__).resolve().parents[2]
AS_OF = "2025-06-15"
NAN_SWIPE = "WEX-000005"  # the sample swipe whose raw amount and paid_amt are "N/A"


@pytest.fixture(scope="session")
def repo_params():
    return load_params(ROOT / "params.yaml")


@pytest.fixture
def params(repo_params, tmp_path):
    """params.yaml sized for the sample (2 active real clients), drift baseline in a temp dir."""
    params = copy.deepcopy(repo_params)
    params["scales"]["small"]["clients"] = 2
    params["validation"]["money_unit"]["min_values"] = 3
    params["validation"]["drift"]["baseline_dir"] = str(tmp_path / "baseline")
    return params


@pytest.fixture
def settings(params):
    return params["validation"]


@pytest.fixture(scope="session")
def _sample_tables(sample_raw_dir):
    return build_tables(load_raw(sample_raw_dir), AS_OF)


@pytest.fixture
def tables(_sample_tables):
    """Fresh copies of the 9 processed sample tables (tests may change them)."""
    return {name: df.copy() for name, df in _sample_tables.items()}


@pytest.fixture
def clean_tables(tables):
    """Sample tables without the one swipe that has unparseable money: every check passes."""
    swipes = tables["swipes"]
    tables["swipes"] = swipes[swipes["swipe_id"] != NAN_SWIPE].reset_index(drop=True)
    return tables


@pytest.fixture
def processed_dir(clean_tables, tmp_path):
    """The clean sample tables written as Parquet, like data/processed/."""
    out = tmp_path / "processed"
    out.mkdir()
    for name, df in clean_tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return out
