"""Processed tables for bias tests, generated in memory from the repo's params.yaml (small scale, ~0.3 s each)."""
import copy
from pathlib import Path

import pytest

from nsh_pipeline.config import load_params
from nsh_pipeline.generate.run import build_collections
from nsh_pipeline.preprocess.run import build_tables

PARAMS_PATH = Path(__file__).parents[2] / "params.yaml"


def _build(params: dict):
    collections, issues = build_collections(params)
    return build_tables(collections, params["reference_date"]), {i["issue"]: i for i in issues}


@pytest.fixture(scope="session")
def repo_params():
    return load_params(PARAMS_PATH)


@pytest.fixture(scope="session")
def injected_params(repo_params):
    params = copy.deepcopy(repo_params)
    params["inject_issues"]["enabled"] = True
    return params


@pytest.fixture(scope="session")
def clean(repo_params):
    """(tables, issues) for the default dataset: no injected problems."""
    return _build(repo_params)


@pytest.fixture(scope="session")
def injected(injected_params):
    """(tables, issues by name) with every injected problem, including denial_skew."""
    return _build(injected_params)
