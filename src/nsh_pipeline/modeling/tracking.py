"""MLflow setup and the tags every run must carry (data_version, git_sha).

Tracking URI, in order of precedence:
  1. the MLFLOW_TRACKING_URI environment variable (e.g. http://localhost:5000 for the Docker server)
  2. params.yaml -> model.mlflow.tracking_uri (default sqlite:///mlflow.db, a local file in the repo)
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import mlflow

from nsh_pipeline.modeling.dataset import SPLITS, split_path

DEFAULT_EXPERIMENT = "nsh-denial-model"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
REQUIRED_TAGS = ("data_version", "git_sha")

_RELEASE_TAG = re.compile(r"/download/([^/]+)/")


def git_sha(repo_dir=".") -> str:
    """Commit the code was run from ("-dirty" when there are uncommitted changes); GITHUB_SHA in CI."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def data_hash(data_dir) -> str:
    """Short SHA-256 over the three split files: changes whenever the training data changes."""
    digest = hashlib.sha256()
    for name in SPLITS:
        path = split_path(name, data_dir)
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def data_version(card: dict) -> str:
    """Data release the training set was built from, e.g. "data-v1.1" (from the dataset card).

    Uses the card's data_version if set, else the release tag in the configured download URL,
    else the generator version. "unknown" if the card carries none of these.
    """
    if card.get("data_version"):
        return str(card["data_version"])
    source = card.get("source_data") or {}
    match = _RELEASE_TAG.search(source.get("configured_release_url") or "")
    if match:
        return match.group(1)
    if source.get("generator_version"):
        return f"generated-v{source['generator_version']}"
    return "unknown"


def run_tags(card: dict, data_dir, extra: dict | None = None) -> dict:
    """Tags for every MLflow run: data_version and git_sha (required), plus data lineage."""
    source = card.get("source_data") or {}
    tags = {
        "data_version": data_version(card),
        "git_sha": git_sha(),
        "data_hash": data_hash(data_dir),
        "data_scale": str(source.get("scale", "unknown")),
        "injected_issues": str(source.get("injected_issues", 0)),
    }
    tags.update({key: str(value) for key, value in (extra or {}).items()})
    return tags


def setup_mlflow(settings: dict | None = None) -> str:
    """Point MLflow at the tracking store and select the experiment; return the experiment id."""
    settings = settings or {}
    uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.get("tracking_uri") or DEFAULT_TRACKING_URI
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(settings.get("experiment", DEFAULT_EXPERIMENT))
    return experiment.experiment_id
