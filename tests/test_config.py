from datetime import date
from pathlib import Path

import pytest

from nsh_pipeline.config import load_params, reference_date, scale_settings


def test_load_params_scale_and_reference_date(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        'seed: 1\nscale: small\nreference_date: "2025-06-15"\n'
        "scales:\n  small:\n    clients: 5\n    max_members_per_client: 10\n",
        encoding="utf-8",
    )
    params = load_params(path)
    assert scale_settings(params) == {"clients": 5, "max_members_per_client": 10}
    assert reference_date(params) == date(2025, 6, 15)


def test_unknown_scale_raises():
    with pytest.raises(ValueError, match="huge"):
        scale_settings({"scale": "huge", "scales": {"small": {}}})


def test_repo_params_file_is_valid():
    params = load_params(Path(__file__).parents[1] / "params.yaml")
    assert params["scale"] in params["scales"]
    reference_date(params)  # parses without error