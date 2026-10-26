"""Record the data produced by an Airflow run in DVC.

``dvc commit -f <stage>`` stores the current outputs of each ``dvc.yaml`` stage in
the DVC cache and updates ``dvc.lock``, exactly as if ``dvc repro`` had produced them.
``dvc push`` (optional) uploads them to the remote.

Nothing here raises: if DVC is missing or a stage has no outputs yet, the step reports
it and the run continues. Commit ``dvc.lock`` from the host afterwards.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def dvc_stages(root: Path) -> list[str]:
    dvc_file = root / "dvc.yaml"
    if not dvc_file.exists():
        return []
    with dvc_file.open() as handle:
        return list((yaml.safe_load(handle) or {}).get("stages", {}))


def _run(cmd: list[str], root: Path, timeout: int = 900) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output[-500:]


def dvc_version(root: Path, push: bool = False) -> dict[str, Any]:
    root = Path(root)
    if shutil.which("dvc") is None:
        logger.warning("dvc is not installed in this environment; skipping versioning")
        return {"status": "skipped", "reason": "dvc not installed"}
    if not (root / ".dvc").exists():
        logger.warning("%s is not a DVC repo (run `dvc init`); skipping versioning", root)
        return {"status": "skipped", "reason": "not a dvc repo"}

    stages: dict[str, str] = {}
    for stage in dvc_stages(root):
        ok, output = _run(["dvc", "commit", "-f", stage], root)
        stages[stage] = "committed" if ok else f"failed: {output}"
        (logger.info if ok else logger.warning)("dvc commit %s: %s", stage, stages[stage])

    result: dict[str, Any] = {"stages": stages, "push": "not requested"}
    if push:
        ok, output = _run(["dvc", "push"], root, timeout=1800)
        result["push"] = "ok" if ok else f"failed: {output}"
        (logger.info if ok else logger.warning)("dvc push: %s", result["push"])

    committed = [s for s, v in stages.items() if v == "committed"]
    result["status"] = "ok" if stages and len(committed) == len(stages) else ("partial" if committed else "failed")
    return result
