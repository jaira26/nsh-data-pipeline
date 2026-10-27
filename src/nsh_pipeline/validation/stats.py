"""Per-table statistics -> reports/stats/<table>.json (plan section 8, stage 6).

The output is deterministic: no timestamps, floats rounded, ties in top-k broken by value, and
each column described by a `kind` (number, text, bool, datetime) instead of a pandas dtype
name ("str" in pandas 3, "object" in pandas 2). That way the committed drift baseline can be
regenerated exactly, on any machine.
"""
import json
import math
from pathlib import Path

import pandas as pd
from pandas.api import types

from nsh_pipeline.validation.common import write_json

DECIMALS = 6


def _num(value) -> float | None:
    """Round a number for JSON; NaN/inf become None."""
    if value is None:
        return None
    value = float(value)
    return round(value, DECIMALS) if math.isfinite(value) else None


def column_kind(series: pd.Series) -> str:
    if types.is_bool_dtype(series.dtype):
        return "bool"
    if types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if types.is_numeric_dtype(series.dtype):
        return "number"
    return "text"


def column_stats(series: pd.Series, top_k: int = 5, max_categories: int = 50) -> dict:
    """Null rate plus kind-specific stats: min/max/mean/std/median, true rate, date range or top-k."""
    rows = len(series)
    nulls = int(series.isna().sum())
    kind = column_kind(series)
    out = {"kind": kind, "null_count": nulls, "null_rate": _num(nulls / rows) if rows else None}
    values = series.dropna()
    if kind == "number":
        numbers = values.astype(float)
        out.update({
            "min": _num(numbers.min()) if len(numbers) else None,
            "max": _num(numbers.max()) if len(numbers) else None,
            "mean": _num(numbers.mean()) if len(numbers) else None,
            "std": _num(numbers.std()) if len(numbers) > 1 else None,
            "median": _num(numbers.median()) if len(numbers) else None,
        })
    elif kind == "bool":
        out["true_rate"] = _num(values.astype(bool).mean()) if len(values) else None
    elif kind == "datetime":
        out["min"] = values.min().isoformat() if len(values) else None
        out["max"] = values.max().isoformat() if len(values) else None
    else:
        text = values.astype(str)
        counts = text.value_counts()
        out["distinct"] = int(len(counts))
        if len(counts) <= max_categories:
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            out["top"] = [{"value": v, "count": int(c), "share": _num(c / rows)} for v, c in ranked]
            out["top_complete"] = len(counts) <= top_k  # every value is listed
    return out


def table_stats(df: pd.DataFrame, table: str, top_k: int = 5, max_categories: int = 50) -> dict:
    """Stats for every column of one table."""
    return {
        "table": table,
        "row_count": int(len(df)),
        "columns": {col: column_stats(df[col], top_k, max_categories) for col in df.columns},
    }


def compute_stats(tables: dict[str, pd.DataFrame], settings: dict) -> dict[str, dict]:
    """{table: stats} using `validation.stats` settings (top_k, max_categories)."""
    cfg = settings.get("stats") or {}
    top_k, max_categories = int(cfg.get("top_k", 5)), int(cfg.get("max_categories", 50))
    return {name: table_stats(df, name, top_k, max_categories) for name, df in sorted(tables.items())}


def write_stats(stats: dict[str, dict], out_dir) -> list[Path]:
    """Write one <table>.json per table into `out_dir`; return the paths."""
    return [write_json(table, Path(out_dir) / f"{name}.json") for name, table in stats.items()]


def read_stats(directory) -> dict[str, dict]:
    """Read every <table>.json in `directory` ({} if it does not exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}
