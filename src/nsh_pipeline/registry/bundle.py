"""One registry version = one .tar.gz bundle:

    manifest.json              the production record for this version (see production_record)
    model/model.joblib         the selected model (model contract, task assigner section 2.2)
    model/metadata.json
    reports/validation.json    hold-out validation report
    reports/bias.json          fairness report (names the selected model)
    reports/bias_report.md     (when present)
"""
from __future__ import annotations

import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from nsh_pipeline.modeling.artifacts import METADATA_FILE, MODEL_FILE

MANIFEST = "manifest.json"
BUNDLE_NAME = "bundle.tar.gz"
_VERSION_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def make_version(git_sha: str | None, now: datetime | None = None) -> str:
    """Sortable, registry-safe version: v20261114-153617-2f68d25."""
    now = now or datetime.now(timezone.utc)
    sha = re.sub(r"[^0-9a-f]", "", (git_sha or "").split("-")[0].lower())[:7] or "nogit"
    return f"v{now:%Y%m%d-%H%M%S}-{sha}"


def check_version(version: str) -> str:
    if not _VERSION_OK.match(version):
        raise ValueError(f"invalid registry version {version!r}")
    return version


def production_record(version: str, bias_report: dict, model_metadata: dict, metrics: dict, *,
                      previous_version: str | None, uri: str, backend: str, promoted_at: str) -> dict:
    """registry/production.json contract (task assigner 2.3): version, metrics, promoted_at, previous_version."""
    return {
        "version": version,
        "model_name": bias_report.get("selected_model"),
        "base_model": bias_report.get("base_model"),
        "threshold": bias_report.get("threshold"),
        "metrics": {k: metrics[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "positive_rate")
                    if k in metrics},
        "metrics_split": "test",
        "data_version": bias_report.get("data_version") or model_metadata.get("data_version"),
        "git_sha": model_metadata.get("git_sha"),
        "mlflow_run_id": bias_report.get("selected_model_mlflow_run_id") or model_metadata.get("mlflow_run_id"),
        "promoted_at": promoted_at,
        "previous_version": previous_version,
        "uri": uri,
        "backend": backend,
    }


def build_bundle(out_path, model_dir, manifest: dict, report_files: dict[str, Path]) -> Path:
    """Write the bundle; report_files maps the name inside reports/ to a local file (missing ones skipped)."""
    model_dir = Path(model_dir)
    for name in (MODEL_FILE, METADATA_FILE):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"{model_dir / name} is missing: the selected model folder is incomplete")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.parent / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(manifest_path, arcname=MANIFEST)
        for name in (MODEL_FILE, METADATA_FILE):
            tar.add(model_dir / name, arcname=f"model/{name}")
        for name, path in report_files.items():
            if path and Path(path).exists():
                tar.add(path, arcname=f"reports/{name}")
    manifest_path.unlink()
    return out_path


def extract_bundle(bundle_path, dest_dir) -> Path:
    """Unpack a bundle (only regular files under the expected folders; no absolute or '..' paths)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and _safe_member(m.name)]
        tar.extractall(dest_dir, members=members, filter="data")
    return dest_dir


def _safe_member(name: str) -> bool:
    parts = Path(name).parts
    return (not Path(name).is_absolute() and ".." not in parts
            and (name == MANIFEST or parts[0] in ("model", "reports")))


def read_manifest(bundle_dir) -> dict:
    return json.loads((Path(bundle_dir) / MANIFEST).read_text(encoding="utf-8"))
