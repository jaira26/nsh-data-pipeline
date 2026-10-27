"""Pieces shared by the validation stages: settings, table loading, JSON output, failure mode."""
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODES = ("strict", "report")


class ValidationFailed(RuntimeError):
    """A strict-mode stage found a schema failure or a critical anomaly."""


def validation_settings(params: dict) -> dict:
    """Return the `validation:` section of params.yaml; fail clearly if it is missing or invalid."""
    settings = params.get("validation")
    if not isinstance(settings, dict):
        raise ValueError("params.yaml has no 'validation:' section (see docs/validation.md)")
    if settings.get("mode") not in MODES:
        raise ValueError(f"validation.mode must be one of {MODES}, got {settings.get('mode')!r}")
    return settings


def load_tables(directory, names=None) -> dict[str, pd.DataFrame]:
    """Read `<directory>/<name>.parquet` into {name: DataFrame}; a missing directory gives {}.

    With `names`, only those tables are read (missing files are skipped).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    paths = sorted(directory.glob("*.parquet"))
    if names is not None:
        paths = [p for p in paths if p.stem in set(names)]
    return {path.stem: pd.read_parquet(path) for path in paths}


def write_json(obj, path) -> Path:
    """Write `obj` as indented JSON (dates and other objects via str) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def enforce_mode(mode: str, failed: bool, message: str) -> None:
    """In strict mode raise `ValidationFailed` when `failed`; in report mode only log it."""
    if not failed:
        return
    if mode == "strict":
        raise ValidationFailed(message)
    logger.warning("%s (validation.mode=report: continuing)", message)
