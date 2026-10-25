"""Run summary (reports/run_summary.json) and the DAG's failure alert."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def anomaly_counts(path: Path) -> dict[str, int] | None:
    """Sum anomaly ``count`` by severity from M3's reports/anomalies.json."""
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s", path)
        return None
    counts: dict[str, int] = {}
    for item in records if isinstance(records, list) else []:
        severity = str(item.get("severity", "unknown")).lower()
        counts[severity] = counts.get(severity, 0) + int(item.get("count") or 0)
    return counts


def build_run_summary(
    run_id: str,
    source: str | None,
    row_counts: dict[str, int] | None,
    durations: dict[str, float | None],
    anomalies_path: Path,
    stage_results: dict[str, Any] | None,
    dvc: dict[str, Any] | None,
    commit: str,
) -> dict[str, Any]:
    known = [d for d in durations.values() if d is not None]
    return {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "dataset_source": source,
        "row_counts": row_counts,
        "anomalies_by_severity": anomaly_counts(anomalies_path),
        "stages": stage_results or {},
        "task_durations_s": {k: (round(v, 1) if v is not None else None) for k, v in durations.items()},
        "total_task_time_s": round(sum(known), 1),
        "dvc": dvc,
    }


def write_run_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Run summary written to %s", path)
    return path


def notify_failure(context: dict[str, Any]) -> None:
    """``on_failure_callback``: send M3's alert if it exists, otherwise log the failure."""
    ti = context.get("task_instance")
    title = f"NSH pipeline: task {getattr(ti, 'task_id', '?')} failed"
    body = "\n".join([
        f"DAG: {getattr(ti, 'dag_id', '?')}",
        f"Run: {context.get('run_id', '?')}",
        f"Try: {getattr(ti, 'try_number', '?')}",
        f"Error: {context.get('exception')}",
        f"Log: {getattr(ti, 'log_url', 'n/a')}",
    ])
    try:
        from nsh_pipeline.utils.alerts import send_alert
    except ImportError:
        logger.error("%s (alerts module not merged yet)\n%s", title, body)
        return
    try:
        send_alert(title, body)
    except Exception as exc:  # an alert failure must never mask the original error
        logger.error("Alert could not be sent: %s\n%s\n%s", exc, title, body)
