"""Model contract (task assigner section 2.2): where models live and how to save / load them.

    models/candidates/<model_name>/model.joblib + metadata.json
    models/best/model.joblib + metadata.json

model.joblib is a scikit-learn Pipeline with predict_proba(X[FEATURE_COLUMNS]).
metadata.json always has METADATA_KEYS. Teammates load models only through load_model().
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

DEFAULT_MODELS_DIR = Path("models")
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
METADATA_KEYS = ("model_name", "params", "cv_metrics", "mlflow_run_id", "data_version", "git_sha", "created_at")


def candidate_dir(model_name: str, models_dir=DEFAULT_MODELS_DIR) -> Path:
    return Path(models_dir) / "candidates" / model_name


def best_dir(models_dir=DEFAULT_MODELS_DIR) -> Path:
    return Path(models_dir) / "best"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_model(model, metadata: dict, out_dir) -> Path:
    """Write model.joblib + metadata.json into out_dir (replacing what was there)."""
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata.json is missing required keys: {missing}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return out_dir


def load_metadata(model_dir) -> dict:
    return json.loads((Path(model_dir) / METADATA_FILE).read_text(encoding="utf-8"))


def load_model(model_dir=None):
    """Return (pipeline, metadata) for a model folder; default models/best."""
    model_dir = Path(model_dir) if model_dir else best_dir()
    return joblib.load(model_dir / MODEL_FILE), load_metadata(model_dir)


def copy_model(src_dir, dst_dir) -> Path:
    """Copy a saved model folder (used to publish the selected candidate as models/best)."""
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir
