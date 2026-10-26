"""Rule-based anomaly checks for the known NSH data hazards (plan section 8, stage 6).

Each check is a pure function returning a `CheckResult`: check name, severity (critical or
warning), passed, count of offending records, record ids ("<table>:<key>") and a message.
`run_checks` runs them all in a fixed order; `write_reports` writes reports/anomalies.json
(a top-level list, as `orchestration/summary.py` expects) and reports/anomalies.md.

Severity follows the plan: wrong-number hazards (double-counting, unit errors, orphans,
excluded orgs, missing money) are critical; the rest are warnings.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nsh_pipeline.config import scale_settings
from nsh_pipeline.constants import EXCLUDED_ORG_IDS
from nsh_pipeline.preprocess.money import parse_money_str
from nsh_pipeline.validation.common import write_json
from nsh_pipeline.validation.drift import compare_stats
from nsh_pipeline.validation.enums import RAW_ENUMS, SOURCE_ENUMS

CRITICAL, WARNING = "critical", "warning"

# Columns that identify one record of each table, used to build "<table>:<key>" record ids.
ID_COLUMNS: dict[str, list[str]] = {
    "clients": ["client_id"],
    "members": ["member_id"],
    "member_benefits": ["member_id", "plan_name"],
    "swipes": ["swipe_id"],
    "claims": ["claim_number"],
    "tickets": ["ticket_number"],
    "bill_lines": ["member_id", "billing_month"],
    "reserves": ["client_id", "month"],
    "ledger": ["txn_id"],
    "book_of_business": ["client_id"],
    "pmpm_monthly": ["client_id", "month"],
    "substantiation_rate": ["client_id", "quarter"],
    "claims_aging": ["client_id", "status"],
}

# (child table, child column, parent table, parent column): every key must resolve, none null.
FOREIGN_KEYS = [
    ("members", "client_id", "clients", "client_id"),
    ("member_benefits", "member_id", "members", "member_id"),
    ("member_benefits", "client_id", "clients", "client_id"),
    ("swipes", "member_id", "members", "member_id"),
    ("swipes", "client_id", "clients", "client_id"),
    ("claims", "member_id", "members", "member_id"),
    ("claims", "client_id", "clients", "client_id"),
    ("tickets", "member_id", "members", "member_id"),
    ("tickets", "client_id", "clients", "client_id"),
    ("bill_lines", "member_id", "members", "member_id"),
    ("bill_lines", "client_id", "clients", "client_id"),
    ("reserves", "client_id", "clients", "client_id"),
    ("ledger", "member_id", "members", "member_id"),
    ("ledger", "client_id", "clients", "client_id"),
]

# Events that cannot happen after the as-of date (compared by calendar day). Ledger effective
# dates are excluded: they legitimately trail the service date by a couple of days.
EVENT_DATES = [("swipes", "service_date"), ("claims", "created_at"), ("tickets", "created_at")]

# Money that should never be negative. Ledger (refunds) and reserves (deficits) can be.
NON_NEGATIVE_MONEY = ["swipes", "claims", "bill_lines", "member_benefits"]


@dataclass
class CheckResult:
    """Outcome of one anomaly check. `record_ids` holds every flagged id; reports show a sample."""

    check: str
    severity: str
    passed: bool
    count: int
    message: str
    record_ids: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    status: str = "ran"  # "ran" or "skipped"

    def to_record(self, sample: int = 10) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "status": self.status,
            "passed": self.passed,
            "count": self.count,
            "sample_record_ids": self.record_ids[:sample],
            "message": self.message,
            "details": self.details,
        }


def _skipped(check: str, severity: str, reason: str) -> CheckResult:
    return CheckResult(check, severity, True, 0, f"skipped: {reason}", status="skipped")


def _result(check: str, severity: str, ids: list[str], ok_message: str, bad_message: str,
            details: dict | None = None, count: int | None = None) -> CheckResult:
    count = len(ids) if count is None else count
    passed = count == 0
    return CheckResult(check, severity, passed, count, ok_message if passed else bad_message,
                       ids, details or {})


def record_ids(df: pd.DataFrame, table: str) -> pd.Series:
    """"<table>:<key>" for each row; multi-column keys are joined with "|", dates as YYYY-MM-DD."""
    columns = [c for c in ID_COLUMNS.get(table, []) if c in df.columns]
    if not columns:
        return pd.Series([f"{table}:row{i}" for i in range(len(df))], index=df.index)
    parts = []
    for col in columns:
        values = df[col]
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            values = values.dt.strftime("%Y-%m-%d")
        parts.append(values.astype(object).where(values.notna(), "").astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return f"{table}:" + key


def _ids(df: pd.DataFrame, table: str, mask) -> list[str]:
    return record_ids(df[mask], table).tolist()


def _money_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_usd")]


# ---------------------------------------------------------------- critical checks
def duplicate_bill_lines(tables: dict[str, pd.DataFrame]) -> CheckResult:
    """Same member billed more than once in a month: PMPM double-counting (contract A8 #1)."""
    check = "duplicate_bill_lines"
    bills = tables.get("bill_lines")
    if bills is None:
        return _skipped(check, CRITICAL, "bill_lines table not found")
    keyed = bills.dropna(subset=["member_id"])
    in_group = keyed.duplicated(["member_id", "billing_month"], keep=False)
    extra_rows = int(keyed.duplicated(["member_id", "billing_month"], keep="first").sum())
    ids = sorted(set(record_ids(keyed[in_group], "bill_lines")))
    return _result(
        check, CRITICAL, ids,
        "every (member_id, billing_month) appears once in bill_lines",
        f"{len(ids)} (member_id, billing_month) keys billed more than once ({extra_rows} extra rows)",
        {"duplicate_keys": len(ids), "extra_rows": extra_rows}, count=extra_rows,
    )


def money_unit_errors(tables: dict[str, pd.DataFrame], settings: dict) -> CheckResult:
    """Money values far above their column median: cents read as dollars (contract A4, A8 #6)."""
    cfg = settings.get("money_unit") or {}
    multiple, min_values = float(cfg.get("max_median_multiple", 100)), int(cfg.get("min_values", 20))
    ids, details = [], {}
    for table, df in sorted(tables.items()):
        flagged = pd.Series(False, index=df.index)
        for col in _money_columns(df):
            positive = df[col][df[col] > 0]
            if len(positive) < min_values:
                continue
            median = float(positive.median())
            hits = df[col] > multiple * median
            if hits.any():
                details.setdefault(table, {})[col] = {
                    "median": round(median, 2), "threshold": round(multiple * median, 2),
                    "flagged": int(hits.sum()), "max_multiple": round(float(df[col].max()) / median, 1),
                }
            flagged |= hits
        ids += _ids(df, table, flagged)
    return _result(
        "money_unit_error", CRITICAL, ids,
        f"no money value above {multiple:g}x its column median",
        f"{len(ids)} records have money above {multiple:g}x the column median (cents read as dollars?)",
        details,
    )


def orphan_keys(tables: dict[str, pd.DataFrame]) -> CheckResult:
    """Join keys that are null or point to no parent row (members, clients)."""
    ids, details = [], {}
    for child, col, parent, parent_col in FOREIGN_KEYS:
        if child not in tables or parent not in tables or col not in tables[child]:
            continue
        df = tables[child]
        nulls = df[col].isna()
        orphans = df[col].notna() & ~df[col].isin(set(tables[parent][parent_col].dropna()))
        if nulls.any() or orphans.any():
            details[f"{child}.{col} -> {parent}"] = {"null": int(nulls.sum()), "orphan": int(orphans.sum())}
        ids += _ids(df, child, nulls | orphans)
    ids = list(dict.fromkeys(ids))  # a row with two bad keys counts once
    return _result(
        "orphan_keys", CRITICAL, ids,
        "every member_id / client_id resolves to its parent table",
        f"{len(ids)} records have a null or unresolved member_id / client_id",
        details,
    )


def excluded_org_leak(tables: dict[str, pd.DataFrame], feature_tables: dict[str, pd.DataFrame]) -> CheckResult:
    """NS Demo / NSAI rows in member, activity or feature tables, or unflagged in clients (A0 rule 2)."""
    clients = tables.get("clients")
    excluded = set(EXCLUDED_ORG_IDS)
    ids, details = [], {}
    if clients is not None:
        excluded |= set(clients.loc[clients["is_excluded_org"].astype(bool), "client_id"])
        unflagged = clients["client_id"].isin(EXCLUDED_ORG_IDS) & ~clients["is_excluded_org"].astype(bool)
        if unflagged.any():
            details["clients (not flagged is_excluded_org)"] = int(unflagged.sum())
        ids += _ids(clients, "clients", unflagged)
    for table, df in sorted({**tables, **feature_tables}.items()):
        if table == "clients" or "client_id" not in df:
            continue
        leaked = df["client_id"].isin(excluded)
        if leaked.any():
            details[table] = int(leaked.sum())
        ids += _ids(df, table, leaked)
    return _result(
        "excluded_org_leak", CRITICAL, ids,
        "NS Demo / NSAI appear only in clients, flagged is_excluded_org",
        f"{len(ids)} NS Demo / NSAI records leaked into reporting tables",
        details,
    )


def unparseable_money(tables: dict[str, pd.DataFrame], raw: dict[str, list[dict]] | None = None) -> CheckResult:
    """Money that is missing after preprocessing (e.g. "N/A" in the raw feed): silently dropped from totals.

    Detected on the processed tables (NaN in a *_usd column). When the raw swipes are available,
    the unparseable raw values are listed too, so the report shows what the feed contained.
    """
    ids, details = [], {}
    for table, df in sorted(tables.items()):
        cols = _money_columns(df)
        if not cols:
            continue
        missing = df[cols].isna()
        if missing.any().any():
            details[table] = {c: int(n) for c, n in missing.sum().items() if n}
        ids += _ids(df, table, missing.any(axis=1))
    swipe_ids = {i.split(":", 1)[1] for i in ids if i.startswith("swipes:")}
    if raw and raw.get("wex_claims") and swipe_ids:
        values: dict[str, int] = {}
        for doc in raw["wex_claims"]:
            if doc.get("_id") not in swipe_ids:
                continue
            for name in ("amount", "paid_amt", "denied_amt"):
                value = doc.get(name)
                if pd.isna(parse_money_str(value)):
                    values[repr(value)] = values.get(repr(value), 0) + 1
        details["raw wex_claims values"] = values
    return _result(
        "unparseable_money", CRITICAL, ids,
        "every money value parsed to a number",
        f"{len(ids)} records have missing money after parsing (unparseable or empty in the raw data)",
        details,
    )


# ---------------------------------------------------------------- warnings
def active_client_count(tables: dict[str, pd.DataFrame], params: dict, settings: dict) -> CheckResult:
    """Active, non-excluded clients within the expected share of the scale's client count."""
    check = "active_client_count"
    clients = tables.get("clients")
    if clients is None:
        return _skipped(check, WARNING, "clients table not found")
    cfg = settings.get("active_clients") or {}
    expected = scale_settings(params)["clients"]
    low, high = cfg.get("min_share", 0.7) * expected, cfg.get("max_share", 1.0) * expected
    active = int((clients["status"].astype(str).str.lower().eq("active") & ~clients["is_excluded_org"]).sum())
    ok = low <= active <= high
    details = {"active_clients": active, "expected_range": [round(low, 1), round(high, 1)],
               "scale": params.get("scale")}
    return CheckResult(
        check, WARNING, ok, 0 if ok else 1,
        f"{active} active clients, expected {low:g}-{high:g} for scale {params.get('scale')!r}",
        details=details,
    )


def _enum_hits(values: pd.Series, allowed: tuple[str, ...]) -> tuple[pd.Series, dict]:
    bad = values.notna() & ~values.isin(allowed)
    folded = {a.casefold() for a in allowed}
    found = values[bad].astype(str).value_counts().to_dict()
    info = {
        "values": {k: int(v) for k, v in sorted(found.items())},
        "casing_drift": sorted(v for v in found if v.casefold() in folded),
    }
    return bad, info


def enum_violations(tables: dict[str, pd.DataFrame], raw: dict[str, list[dict]] | None = None) -> CheckResult:
    """Source-system values outside the contract A5 lists, incl. casing drift ("closed" vs "Closed").

    Runs on processed tables for fields preprocessing passes through, and on raw documents for
    fields it normalizes (client status is title-cased, so drift there is only visible raw).
    """
    ids, details = [], {}
    for (table, col), allowed in SOURCE_ENUMS.items():
        df = tables.get(table)
        if df is None or col not in df:
            continue
        bad, info = _enum_hits(df[col], allowed)
        if bad.any():
            details[f"{table}.{col}"] = info
        ids += _ids(df, table, bad)
    for (collection, name), allowed in RAW_ENUMS.items():
        docs = (raw or {}).get(collection)
        if not docs:
            continue
        frame = pd.DataFrame({
            "_id": [d.get("_id") for d in docs],
            name: [(d.get("employer") or {}).get(name) or d.get(name) for d in docs],
        })
        bad, info = _enum_hits(frame[name], allowed)
        if bad.any():
            details[f"raw.{collection}.{name}"] = info
        ids += [f"raw.{collection}:{i}" for i in frame.loc[bad, "_id"]]
    return _result(
        "enum_violations", WARNING, ids,
        "every enum value is in the contract A5 lists",
        f"{len(ids)} values outside the contract enums ({len(details)} fields)",
        details,
    )


def future_dates(tables: dict[str, pd.DataFrame], as_of) -> CheckResult:
    """Service / creation dates after the as-of date (compared by calendar day)."""
    as_of = pd.Timestamp(as_of)
    as_of = (as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")).normalize()
    ids, details = [], {}
    for table, col in EVENT_DATES:
        df = tables.get(table)
        if df is None or col not in df:
            continue
        future = df[col].dt.normalize() > as_of
        if future.any():
            details[f"{table}.{col}"] = {"count": int(future.sum()), "max": df.loc[future, col].max().isoformat()}
        ids += _ids(df, table, future)
    return _result(
        "future_dates", WARNING, ids,
        f"no service or creation date after {as_of.date()}",
        f"{len(ids)} records dated after the as-of date {as_of.date()}",
        details,
    )


def negative_amounts(tables: dict[str, pd.DataFrame]) -> CheckResult:
    """Negative money where it cannot be negative (swipes, claims, bill lines, premiums)."""
    ids, details = [], {}
    for table in NON_NEGATIVE_MONEY:
        df = tables.get(table)
        if df is None:
            continue
        cols = _money_columns(df)
        negative = (df[cols] < 0)
        if negative.any().any():
            details[table] = {c: int(n) for c, n in negative.sum().items() if n}
        ids += _ids(df, table, negative.any(axis=1))
    return _result(
        "negative_amounts", WARNING, ids,
        "no negative paid, requested or billed amounts",
        f"{len(ids)} records have negative money",
        details,
    )


def term_before_hire(tables: dict[str, pd.DataFrame]) -> CheckResult:
    """Members whose termination date is before their hire date."""
    check = "term_before_hire"
    members = tables.get("members")
    if members is None:
        return _skipped(check, WARNING, "members table not found")
    bad = members["term_date"].notna() & members["hire_date"].notna() & (members["term_date"] < members["hire_date"])
    ids = _ids(members, "members", bad)
    return _result(check, WARNING, ids, "no termination date before a hire date",
                   f"{len(ids)} members terminated before they were hired")


def distribution_drift(current_stats: dict[str, dict], baseline_stats: dict[str, dict], settings: dict) -> CheckResult:
    """Stats moved past the drift thresholds vs the baseline (see drift.py)."""
    check = "distribution_drift"
    if not baseline_stats:
        return _skipped(check, WARNING, "no drift baseline yet (the stats step creates it)")
    findings = compare_stats(current_stats, baseline_stats, settings["drift"])
    ids = [f"{f['table']}.{f['column'] or '*'}:{f['metric']}" for f in findings]
    return _result(check, WARNING, ids, "no drift beyond the thresholds",
                   f"{len(findings)} statistics drifted beyond the thresholds",
                   {"findings": [f["message"] for f in findings]})


def run_checks(tables, feature_tables, params, settings, as_of, raw=None,
               current_stats=None, baseline_stats=None) -> list[CheckResult]:
    """Run every check in a fixed order: critical first, then warnings."""
    every = {**tables, **feature_tables}
    return [
        duplicate_bill_lines(tables),
        money_unit_errors(every, settings),
        orphan_keys(tables),
        excluded_org_leak(tables, feature_tables),
        unparseable_money(tables, raw),
        active_client_count(tables, params, settings),
        enum_violations(tables, raw),
        future_dates(tables, as_of),
        negative_amounts(tables),
        term_before_hire(tables),
        distribution_drift(current_stats or {}, baseline_stats or {}, settings),
    ]


def critical_failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.severity == CRITICAL and not r.passed]


def to_markdown(results: list[CheckResult], meta: dict, sample: int = 10) -> str:
    """Human-readable summary of every check (reports/anomalies.md)."""
    lines = [
        "# Anomaly report", "",
        f"- Processed data: `{meta.get('processed_dir')}`",
        f"- As of: {meta.get('as_of')}  |  mode: `{meta.get('mode')}`",
        f"- Critical failures: **{meta.get('critical_failed')}**  |  warnings: **{meta.get('warnings_failed')}**",
        "", "| Check | Severity | Result | Count | Message |", "|---|---|---|---|---|",
    ]
    for r in results:
        outcome = "skipped" if r.status == "skipped" else ("pass" if r.passed else "**FAIL**")
        lines.append(f"| `{r.check}` | {r.severity} | {outcome} | {r.count} | {r.message} |")
    for r in results:
        if r.passed:
            continue
        lines += ["", f"## `{r.check}` ({r.severity})", "", r.message, ""]
        if r.record_ids:
            lines.append("Sample records: " + ", ".join(f"`{i}`" for i in r.record_ids[:sample]))
        if r.details:
            lines += ["", "```", *[f"{k}: {v}" for k, v in r.details.items()], "```"]
    return "\n".join(lines) + "\n"


def write_reports(results: list[CheckResult], report_dir, meta: dict, sample: int = 10) -> dict[str, Path]:
    """Write anomalies.json (top-level list of check records) and anomalies.md."""
    report_dir = Path(report_dir)
    json_path = write_json([r.to_record(sample) for r in results], report_dir / "anomalies.json")
    md_path = report_dir / "anomalies.md"
    md_path.write_text(to_markdown(results, meta, sample), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
