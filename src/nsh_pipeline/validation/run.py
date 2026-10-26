"""Validation entry point (M3). Airflow (orchestration/stages.py) runs:

    python -m nsh_pipeline.validation.run --processed-dir data/processed --params params.yaml --step schema
    (and --step stats, --step anomalies)

Every other option has a default; the as-of date defaults to params.yaml -> reference_date.

Step        Checks                                              Writes (under --report-dir)
raw         raw files vs manifest.json + light raw schemas      validation/raw_schema.json
schema      strict processed schemas; feature schemas if built  validation/processed_schema.json, features_schema.json
stats       per-table stats, drift vs the baseline              stats/<table>.json, drift.json
anomalies   rule-based checks (+ answer key if injected)        anomalies.json, anomalies.md,
                                                                validation/anomalies_summary.json,
                                                                injection_detection.json
all         raw (if the raw dir exists), schema, stats, anomalies

validation.mode = strict: a schema failure or critical anomaly exits with code 1 (the task fails).
validation.mode = report: everything is written and alerted, exit code 0.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from nsh_pipeline.config import load_params
from nsh_pipeline.utils.alerts import send_alert
from nsh_pipeline.utils.io import read_jsonl
from nsh_pipeline.validation import anomalies
from nsh_pipeline.validation.answer_key import detect_injected
from nsh_pipeline.validation.common import (
    ValidationFailed, enforce_mode, load_tables, validation_settings, write_json,
)
from nsh_pipeline.validation.drift import run_drift
from nsh_pipeline.validation.schemas import failed_tables, validate_frame
from nsh_pipeline.validation.schemas.features import FEATURE_SCHEMAS
from nsh_pipeline.validation.schemas.processed import PROCESSED_SCHEMAS
from nsh_pipeline.validation.schemas.raw import RAW_SCHEMAS, check_files, raw_frame
from nsh_pipeline.validation.stats import compute_stats, read_stats, write_stats

logger = logging.getLogger(__name__)

STEPS = ("raw", "schema", "stats", "anomalies", "all")
RAW_FOR_ANOMALIES = ("applications", "wex_claims", "bills")


def _read_manifest(raw_dir) -> dict | None:
    path = Path(raw_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _alert(title: str, lines: list[str], details: dict | None = None) -> None:
    send_alert(f"NSH validation: {title}", "\n".join(lines), details)


# ---------------------------------------------------------------- steps
def run_raw(raw_dir, params: dict, report_dir="reports") -> dict:
    """Check raw files against manifest.json and the light raw schemas."""
    settings = validation_settings(params)
    sample = int(settings.get("sample_ids", 10))
    manifest = _read_manifest(raw_dir)
    file_errors = check_files(raw_dir, manifest)
    if manifest is None:
        file_errors.insert(0, {"file": "manifest.json", "check": "exists", "message": "file is missing"})
    tables = []
    for name, schema in RAW_SCHEMAS.items():
        path = Path(raw_dir) / f"{name}.jsonl"
        if path.exists():
            tables.append(validate_frame(raw_frame(read_jsonl(path)), schema, name, sample))
    failed = bool(file_errors) or bool(failed_tables(tables))
    result = {"step": "raw", "status": "failed" if failed else "passed", "raw_dir": str(raw_dir),
              "files": file_errors, "tables": tables}
    write_json(result, Path(report_dir) / "validation" / "raw_schema.json")
    if failed:
        _alert("raw data failed validation", [f"Raw dir: {raw_dir}",
               f"File problems: {len(file_errors)}", f"Failed tables: {failed_tables(tables)}"])
    enforce_mode(settings["mode"], failed,
                 f"Raw validation failed: {len(file_errors)} file problems, tables {failed_tables(tables)}")
    return result


def run_schema(processed_dir, params: dict, report_dir="reports", features_dir="data/features") -> dict:
    """Validate the 9 processed tables (strict) and any feature tables that exist."""
    settings = validation_settings(params)
    sample = int(settings.get("sample_ids", 10))
    tables = load_tables(processed_dir)
    processed = []
    for name, schema in PROCESSED_SCHEMAS.items():
        if name not in tables:
            processed.append({"table": name, "status": "failed", "rows": 0, "errors": [
                {"column": None, "check": "exists", "count": 1,
                 "failure_cases": [f"{name}.parquet not found in {processed_dir}"]}]})
        else:
            processed.append(validate_frame(tables[name], schema, name, sample))
    feature_tables = load_tables(features_dir, FEATURE_SCHEMAS)
    if not feature_tables:
        features = {"status": "skipped", "reason": f"no feature tables in {features_dir}", "tables": []}
    else:
        results = [validate_frame(feature_tables[n], s, n, sample) if n in feature_tables
                   else {"table": n, "status": "skipped", "rows": 0, "errors": []}
                   for n, s in FEATURE_SCHEMAS.items()]
        features = {"status": "failed" if failed_tables(results) else "passed", "tables": results}
    report_dir = Path(report_dir)
    write_json({"step": "schema", "processed_dir": str(processed_dir), "tables": processed},
               report_dir / "validation" / "processed_schema.json")
    write_json({"step": "schema", "features_dir": str(features_dir), **features},
               report_dir / "validation" / "features_schema.json")
    failed = failed_tables(processed) + failed_tables(features["tables"])
    if failed:
        _alert("schema validation failed", [f"Processed dir: {processed_dir}", f"Failed tables: {failed}"],
               {r["table"]: r["errors"][:3] for r in processed + features["tables"] if r["status"] == "failed"})
    enforce_mode(settings["mode"], bool(failed), f"Schema validation failed for tables: {failed}")
    return {"step": "schema", "status": "failed" if failed else "passed", "failed_tables": failed,
            "features": features["status"]}


def run_stats(processed_dir, params: dict, report_dir="reports", features_dir="data/features",
              update_baseline: bool = False) -> dict:
    """Write per-table stats and compare them with the drift baseline (never fails the run)."""
    settings = validation_settings(params)
    tables = {**load_tables(processed_dir), **load_tables(features_dir, FEATURE_SCHEMAS)}
    stats = compute_stats(tables, settings)
    stats_dir = Path(report_dir) / "stats"
    for old in stats_dir.glob("*.json"):
        old.unlink()
    write_stats(stats, stats_dir)
    drift = run_drift(stats, settings["drift"]["baseline_dir"], settings["drift"], update_baseline)
    write_json(drift, Path(report_dir) / "drift.json")
    if drift["status"] == "drift":
        logger.warning("Drift vs baseline: %d findings; headline %s", len(drift["findings"]), drift["headline"])
    return {"step": "stats", "tables": sorted(stats), "drift_status": drift["status"],
            "drift_findings": len(drift["findings"]), "headline": drift["headline"]}


def run_anomalies(processed_dir, params: dict, report_dir="reports", features_dir="data/features",
                  raw_dir="data/raw", as_of=None) -> dict:
    """Run every anomaly check; write the reports and, for injected data, the answer-key comparison."""
    settings = validation_settings(params)
    sample = int(settings.get("sample_ids", 10))
    as_of = as_of or params["reference_date"]
    tables = load_tables(processed_dir)
    feature_tables = load_tables(features_dir, FEATURE_SCHEMAS)
    raw = {n: read_jsonl(Path(raw_dir) / f"{n}.jsonl") for n in RAW_FOR_ANOMALIES
           if (Path(raw_dir) / f"{n}.jsonl").exists()}
    current = compute_stats({**tables, **feature_tables}, settings)
    baseline = read_stats(settings["drift"]["baseline_dir"])
    results = anomalies.run_checks(tables, feature_tables, params, settings, as_of, raw, current, baseline)

    critical = anomalies.critical_failures(results)
    warnings = [r for r in results if r.severity == anomalies.WARNING and not r.passed]
    meta = {
        "step": "anomalies",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed_dir": str(processed_dir), "raw_dir": str(raw_dir), "as_of": str(as_of),
        "mode": settings["mode"], "critical_failed": len(critical), "warnings_failed": len(warnings),
    }
    report_dir = Path(report_dir)
    anomalies.write_reports(results, report_dir, meta, sample)
    summary = {**meta, "checks": {r.check: {"severity": r.severity, "status": r.status,
                                           "passed": r.passed, "count": r.count} for r in results}}
    write_json(summary, report_dir / "validation" / "anomalies_summary.json")

    manifest = _read_manifest(raw_dir)
    if manifest is not None:
        if manifest.get("injected_issues"):
            detection = detect_injected(manifest, results, raw.get("bills"))
            summary["all_injected_detected"] = detection["all_in_scope_detected"]
        else:
            detection = {"status": "no_injected_issues", "dataset": manifest.get("dataset")}
        write_json(detection, report_dir / "injection_detection.json")

    if critical:
        _alert(f"{len(critical)} critical anomalies", [f"Processed dir: {processed_dir}", f"As of: {as_of}"]
               + [f"- {r.check}: {r.message}" for r in critical],
               {r.check: r.record_ids[:sample] for r in critical})
    enforce_mode(settings["mode"], bool(critical),
                 f"{len(critical)} critical anomalies: {[r.check for r in critical]}")
    return summary


def run_step(step: str, params: dict, processed_dir="data/processed", raw_dir="data/raw",
             features_dir="data/features", report_dir="reports", as_of=None,
             update_baseline: bool = False) -> dict:
    """Run one step (or "all" in order) and return its result."""
    if step == "raw":
        return run_raw(raw_dir, params, report_dir)
    if step == "schema":
        return run_schema(processed_dir, params, report_dir, features_dir)
    if step == "stats":
        return run_stats(processed_dir, params, report_dir, features_dir, update_baseline)
    if step == "anomalies":
        return run_anomalies(processed_dir, params, report_dir, features_dir, raw_dir, as_of)
    if step == "all":
        out = {"raw": run_raw(raw_dir, params, report_dir) if Path(raw_dir).is_dir()
               else {"step": "raw", "status": "skipped", "reason": f"{raw_dir} not found"}}
        out["schema"] = run_schema(processed_dir, params, report_dir, features_dir)
        out["stats"] = run_stats(processed_dir, params, report_dir, features_dir, update_baseline)
        out["anomalies"] = run_anomalies(processed_dir, params, report_dir, features_dir, raw_dir, as_of)
        return out
    raise ValueError(f"Unknown step {step!r}; expected one of {STEPS}")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Validate NSH pipeline data (schemas, stats, drift, anomalies).")
    parser.add_argument("--step", choices=STEPS, default="all")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; default params.yaml reference_date")
    parser.add_argument("--update-baseline", action="store_true",
                        help="stats step: replace the drift baseline with this run's stats")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params(args.params)
    try:
        result = run_step(args.step, params, args.processed_dir, args.raw_dir, args.features_dir,
                          args.report_dir, args.as_of, args.update_baseline)
    except ValidationFailed as exc:
        logger.error("Validation failed (validation.mode=strict): %s", exc)
        sys.exit(1)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
