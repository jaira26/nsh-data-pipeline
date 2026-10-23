"""Load pipeline parameters from params.yaml (shared by generator, preprocessing, Airflow and DVC)."""
from datetime import date
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")


def load_params(path=DEFAULT_PARAMS_PATH) -> dict:
    """Read params.yaml into a dict."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_settings(params: dict) -> dict:
    """Return the settings for the selected scale (e.g. clients, max_members_per_client)."""
    name = params["scale"]
    try:
        return params["scales"][name]
    except KeyError:
        raise ValueError(f"Unknown scale {name!r}; expected one of {sorted(params['scales'])}") from None


def reference_date(params: dict) -> date:
    """The pipeline's fixed 'as of' date, so every run is reproducible."""
    return date.fromisoformat(str(params["reference_date"]))