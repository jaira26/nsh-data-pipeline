"""Where registry versions live. Two backends with one interface:

    LocalRegistry   a folder (default registry/): tests, pull requests, offline runs
                        versions/<version>/bundle.tar.gz, production.json, history.jsonl
    GcpRegistry     GCP Artifact Registry, generic-format repository, through the gcloud CLI
                        package <package>             one version per promoted model (the bundle)
                        package <package>-production  the production pointer: one small version per
                                                      promotion or manual rollback; the newest one wins

Artifact Registry versions are immutable, so the production pointer is a log of pointer versions rather
than a mutable file: every promotion and every rollback stays auditable in the console.

Backend choice: REGISTRY_BACKEND env var, else params.yaml -> registry.backend. The GCP project comes from
GCP_PROJECT_ID (env only, so no project id is committed); GCP_REGION / GCP_AR_REPOSITORY override the
location / repository in params.yaml. Credentials: gcloud's own login (locally) or the service account
that google-github-actions/auth activates in CI through Workload Identity Federation. No key file is used.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nsh_pipeline.registry.bundle import BUNDLE_NAME, check_version

logger = logging.getLogger(__name__)

POINTER_FILE = "production.json"
HISTORY_FILE = "history.jsonl"


class RegistryError(RuntimeError):
    pass


class LocalRegistry:
    name = "local"

    def __init__(self, root="registry"):
        self.root = Path(root)

    def uri(self, version: str) -> str:
        return str(self.root / "versions" / version / BUNDLE_NAME)

    def push(self, bundle_path, version: str) -> str:
        target = Path(self.uri(check_version(version)))
        if target.exists():
            raise RegistryError(f"version {version} already exists in {self.root}")
        target.parent.mkdir(parents=True)
        shutil.copy2(bundle_path, target)
        return str(target)

    def fetch(self, version: str, dest_dir) -> Path:
        source = Path(self.uri(check_version(version)))
        if not source.exists():
            raise RegistryError(f"version {version} not found in {self.root}")
        dest = Path(dest_dir) / BUNDLE_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return dest

    def list_versions(self) -> list[str]:
        folder = self.root / "versions"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []

    def get_production(self) -> dict | None:
        path = self.root / POINTER_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def set_production(self, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / POINTER_FILE).write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


class GcpRegistry:
    name = "gcp"

    def __init__(self, project: str, location: str, repository: str, package: str, runner=subprocess.run):
        if not project:
            raise RegistryError("GCP backend needs GCP_PROJECT_ID (env var)")
        self.project, self.location, self.repository, self.package = project, location, repository, package
        self.pointer_package = f"{package}-production"
        self._runner = runner

    def _gcloud(self, *args: str) -> str:
        cmd = ["gcloud", "artifacts", *args, f"--project={self.project}", f"--location={self.location}",
               f"--repository={self.repository}", "--quiet"]
        try:
            result = self._runner(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RegistryError("gcloud CLI not found: install the Google Cloud CLI or use the local backend") from exc
        if result.returncode != 0:
            raise RegistryError(f"{' '.join(cmd[:4])} failed: {(result.stderr or result.stdout).strip()}")
        return result.stdout

    def uri(self, version: str, package: str | None = None) -> str:
        return (f"{self.location}-generic.pkg.dev/{self.project}/{self.repository}/"
                f"{package or self.package}:{version}")

    def push(self, bundle_path, version: str) -> str:
        self._gcloud("generic", "upload", f"--package={self.package}", f"--version={check_version(version)}",
                     f"--source={bundle_path}")
        return self.uri(version)

    def _download(self, package: str, version: str, dest_dir) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._gcloud("generic", "download", f"--package={package}", f"--version={check_version(version)}",
                     f"--destination={dest_dir}")
        return dest_dir

    def fetch(self, version: str, dest_dir) -> Path:
        path = self._download(self.package, version, dest_dir) / BUNDLE_NAME
        if not path.exists():
            raise RegistryError(f"{self.uri(version)} has no {BUNDLE_NAME}")
        return path

    def _versions(self, package: str) -> list[dict]:
        try:
            out = self._gcloud("versions", "list", f"--package={package}", "--sort-by=~createTime", "--format=json")
        except RegistryError as exc:
            if "NOT_FOUND" in str(exc):   # package not created yet: nothing pushed so far
                return []
            raise
        return json.loads(out or "[]")

    def list_versions(self) -> list[str]:
        return sorted(v["name"].rsplit("/", 1)[-1] for v in self._versions(self.package))

    def get_production(self) -> dict | None:
        versions = self._versions(self.pointer_package)
        if not versions:
            return None
        latest = versions[0]["name"].rsplit("/", 1)[-1]   # sorted newest first
        with tempfile.TemporaryDirectory() as tmp:
            path = self._download(self.pointer_package, latest, tmp) / POINTER_FILE
            return json.loads(path.read_text(encoding="utf-8"))

    def set_production(self, record: dict) -> None:
        pointer_version = f"p{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{record['version']}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / POINTER_FILE
            path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
            self._gcloud("generic", "upload", f"--package={self.pointer_package}",
                         f"--version={check_version(pointer_version)}", f"--source={path}")


def make_registry(settings: dict, env: dict | None = None, runner=subprocess.run):
    """The backend named by REGISTRY_BACKEND or params.yaml -> registry.backend."""
    env = os.environ if env is None else env
    backend = (env.get("REGISTRY_BACKEND") or settings["backend"]).lower()
    if backend == "local":
        return LocalRegistry(settings["local_dir"])
    if backend == "gcp":
        gcp = settings["gcp"]
        return GcpRegistry(project=env.get("GCP_PROJECT_ID") or gcp.get("project"),
                           location=env.get("GCP_REGION") or gcp["location"],
                           repository=env.get("GCP_AR_REPOSITORY") or gcp["repository"],
                           package=gcp["package"], runner=runner)
    raise RegistryError(f"unknown registry backend {backend!r}; expected local or gcp")


def append_history(path, entry: dict) -> None:
    """Every decision (promotion, rollback, manual rollback) is appended as one JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
