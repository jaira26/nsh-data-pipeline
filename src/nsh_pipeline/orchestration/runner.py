"""Run a pipeline stage's CLI as a subprocess and stream its output into the task log."""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StageFailed(RuntimeError):
    """A stage CLI exited with a non-zero status."""


def module_available(module: str) -> bool:
    """True if ``python -m module`` would find the module."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package does not exist yet
        return False


def run_module(
    module: str,
    args: list[str],
    cwd: str | Path,
    timeout: float | None = 3600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``python -m module *args`` from ``cwd``.

    Returns a JSON-safe dict (it becomes the task's XCom). If the module is not merged
    yet, logs a placeholder message and returns ``status="placeholder"`` without failing.
    Raises ``StageFailed`` on a non-zero exit so the Airflow task fails.
    """
    if not module_available(module):
        logger.warning("%s is not merged yet; placeholder task, nothing to run", module)
        return {"module": module, "status": "placeholder", "seconds": 0.0}

    cmd = [sys.executable, "-m", module, *args]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )
    tail: list[str] = []
    name = module.rsplit(".", 1)[-1]

    def _stream() -> None:  # read output on a thread so the timeout below still works
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            logger.info("[%s] %s", name, line)
            tail.append(line)
            del tail[:-20]

    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise StageFailed(f"{module} timed out after {timeout}s") from exc
    finally:
        reader.join(timeout=5)

    seconds = round(time.monotonic() - start, 1)
    if returncode != 0:
        raise StageFailed(f"{module} exited with code {returncode}. Last output:\n" + "\n".join(tail))
    logger.info("%s finished in %.1fs", module, seconds)
    return {"module": module, "status": "ok", "seconds": seconds}


def jsonable(value: Any) -> Any:
    """Convert numpy scalars and paths so a value can be stored as XCom."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return value.item()
    return value
