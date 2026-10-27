"""Compare this run's stats with a baseline (validation.drift.baseline_dir) and flag big shifts.

The baseline is the committed clean-run stats in reports/baseline/. If it does not exist yet,
the current stats are written as the baseline and the result is "no_baseline", not a failure.

Signals (thresholds in params.yaml -> validation.drift):
  * row count change (%), columns added or removed
  * numeric mean change (%) and null-rate change (percentage points)
  * category share change (points) for text columns with a top-k, and true-rate for flags
The headline signal (by default the swipe Denied share) is reported first.
"""
from pathlib import Path

from nsh_pipeline.validation.stats import read_stats, write_stats


def _pct_change(old, new) -> float | None:
    if old in (None, 0) or new is None:
        return None
    return 100.0 * (new - old) / abs(old)


def _finding(table, column, metric, baseline, current, change, threshold, unit) -> dict:
    where = f"{table}.{column}" if column else table
    return {
        "table": table, "column": column, "metric": metric,
        "baseline": baseline, "current": current,
        "change": round(change, 4) if isinstance(change, float) else change,
        "threshold": threshold,
        "message": f"{where} {metric} changed by {change:+.2f}{unit} (threshold {threshold}{unit})"
        if isinstance(change, (int, float)) else f"{where}: {metric}",
    }


def _shares(col_stats: dict) -> dict[str, float] | None:
    top = col_stats.get("top")
    return None if top is None else {item["value"]: item["share"] or 0.0 for item in top}


def _compare_column(table, column, base, cur, cfg) -> list[dict]:
    findings = []
    null_pts = 100.0 * ((cur.get("null_rate") or 0) - (base.get("null_rate") or 0))
    if abs(null_pts) > cfg["null_rate_pts"]:
        findings.append(_finding(table, column, "null_rate", base.get("null_rate"), cur.get("null_rate"),
                                 null_pts, cfg["null_rate_pts"], " pts"))
    if base.get("kind") != cur.get("kind"):
        findings.append(_finding(table, column, f"kind {base.get('kind')} -> {cur.get('kind')}",
                                 base.get("kind"), cur.get("kind"), None, None, ""))
        return findings
    if cur["kind"] == "number":
        change = _pct_change(base.get("mean"), cur.get("mean"))
        if change is not None and abs(change) > cfg["mean_pct"]:
            findings.append(_finding(table, column, "mean", base["mean"], cur["mean"], change, cfg["mean_pct"], "%"))
    elif cur["kind"] == "bool":
        if base.get("true_rate") is not None and cur.get("true_rate") is not None:
            pts = 100.0 * (cur["true_rate"] - base["true_rate"])
            if abs(pts) > cfg["category_share_pts"]:
                findings.append(_finding(table, column, "true_rate", base["true_rate"], cur["true_rate"],
                                         pts, cfg["category_share_pts"], " pts"))
    elif cur["kind"] == "text":
        old, new = _shares(base), _shares(cur)
        if old is not None and new is not None:
            complete = base.get("top_complete") and cur.get("top_complete")
            values = sorted(set(old) | set(new)) if complete else sorted(set(old) & set(new))
            for value in values:
                pts = 100.0 * (new.get(value, 0.0) - old.get(value, 0.0))
                if abs(pts) > cfg["category_share_pts"]:
                    findings.append(_finding(table, column, f"share of {value!r}", old.get(value, 0.0),
                                             new.get(value, 0.0), pts, cfg["category_share_pts"], " pts"))
    return findings


def compare_stats(current: dict[str, dict], baseline: dict[str, dict], cfg: dict) -> list[dict]:
    """All drift findings of `current` vs `baseline` stats, using `validation.drift` thresholds."""
    findings = []
    for table in sorted(set(baseline) - set(current)):
        findings.append(_finding(table, None, "table missing (in baseline only)", None, None, None, None, ""))
    for table in sorted(set(current) - set(baseline)):
        findings.append(_finding(table, None, "table added (not in baseline)", None, None, None, None, ""))
    for table in sorted(set(current) & set(baseline)):
        base, cur = baseline[table], current[table]
        change = _pct_change(base["row_count"], cur["row_count"])
        if change is not None and abs(change) > cfg["row_count_pct"]:
            findings.append(_finding(table, None, "row_count", base["row_count"], cur["row_count"],
                                     change, cfg["row_count_pct"], "%"))
        for column in sorted(set(base["columns"]) ^ set(cur["columns"])):
            state = "removed" if column in base["columns"] else "added"
            findings.append(_finding(table, column, f"column {state}", None, None, None, None, ""))
        for column in sorted(set(base["columns"]) & set(cur["columns"])):
            findings += _compare_column(table, column, base["columns"][column], cur["columns"][column], cfg)
    return findings


def headline(current: dict[str, dict], baseline: dict[str, dict], cfg: dict) -> dict | None:
    """The configured headline share (e.g. swipes.status == "Denied") before and after."""
    spec = cfg.get("headline")
    if not spec:
        return None
    table, column, value = spec["table"], spec["column"], spec["value"]

    def share(stats):
        col = (stats.get(table) or {}).get("columns", {}).get(column) or {}
        return (_shares(col) or {}).get(value, 0.0 if col.get("top_complete") else None)

    old, new = share(baseline), share(current)
    change = None if old is None or new is None else round(100.0 * (new - old), 4)
    return {
        "signal": f"share of {table}.{column} == {value!r}",
        "baseline_pct": None if old is None else round(100.0 * old, 4),
        "current_pct": None if new is None else round(100.0 * new, 4),
        "change_pts": change,
        "threshold_pts": cfg["category_share_pts"],
        "drifted": change is not None and abs(change) > cfg["category_share_pts"],
    }


def run_drift(current: dict[str, dict], baseline_dir, cfg: dict, update_baseline: bool = False) -> dict:
    """Compare with the baseline in `baseline_dir`; create it if missing (status "no_baseline").

    With `update_baseline`, the current stats replace the baseline (status "baseline_updated").
    Otherwise the status is "drift" if any finding exceeds its threshold, else "ok".
    """
    baseline_dir = Path(baseline_dir)
    baseline = read_stats(baseline_dir)
    if update_baseline or not baseline:
        for old in baseline_dir.glob("*.json") if update_baseline else []:
            old.unlink()
        write_stats(current, baseline_dir)
        status = "baseline_updated" if update_baseline else "no_baseline"
        return {"status": status, "baseline_dir": str(baseline_dir), "headline": None, "findings": []}
    findings = compare_stats(current, baseline, cfg)
    return {
        "status": "drift" if findings else "ok",
        "baseline_dir": str(baseline_dir),
        "headline": headline(current, baseline, cfg),
        "findings": findings,
    }
