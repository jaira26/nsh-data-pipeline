"""The committed drift baseline (reports/baseline/) must match a fresh clean run from the seed.

If this fails after a generator or preprocessing change, regenerate the baseline:
    python -m nsh_pipeline.validation.run --step stats --update-baseline
(after the download/preprocess steps) and commit reports/baseline/.
"""
import math

from nsh_pipeline.config import load_params
from nsh_pipeline.generate.run import build_collections
from nsh_pipeline.preprocess.run import build_tables
from nsh_pipeline.validation.stats import compute_stats, read_stats
from tests.validation.conftest import ROOT


def assert_same(expected, actual, path="baseline"):
    """Equal JSON trees; floats may differ in the last digits between platforms."""
    if isinstance(expected, float) and isinstance(actual, float):
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-6), f"{path}: {expected} != {actual}"
    elif isinstance(expected, dict) and isinstance(actual, dict):
        assert expected.keys() == actual.keys(), f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in expected:
            assert_same(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual), f"{path}: {len(expected)} items != {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert_same(e, a, f"{path}[{i}]")
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def test_committed_baseline_matches_a_clean_run_from_the_seed():
    params = load_params(ROOT / "params.yaml")
    assert params["inject_issues"]["enabled"] is False
    collections, _ = build_collections(params)
    tables = build_tables(collections, params["reference_date"])
    fresh = compute_stats(tables, params["validation"])
    committed = read_stats(ROOT / params["validation"]["drift"]["baseline_dir"])
    assert committed, "reports/baseline/ is missing"
    assert_same(committed, fresh)
