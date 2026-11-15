"""Push the selected model to the registry, or keep production (rollback). DVC stage `register`:

    python -m nsh_pipeline.registry.promote --params params.yaml

  1. read reports/model/validation.json and reports/model/bias.json (the bias stage names the selected model)
  2. read the current production record from the registry
  3. re-score the production model on the current test split, so both models are compared on the same rows
  4. decide (registry/decide.py): promote only if both gates passed, the selected model meets the validation
     minimums, and it beats production on the registry metric
  5. promote: bundle model + reports, push a new version, move the production pointer
     otherwise: keep production and record why
  6. write registry/production.json (local mirror), reports/model/registry_decision.json, registry/history.jsonl
  7. alert (Slack / email): training completed + promoted or kept

Exit code 0 for both outcomes (a rollback is a valid result); 1 only when the registry itself fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling.artifacts import load_metadata, load_model, utc_now
from nsh_pipeline.modeling.dataset import load_split, split_path
from nsh_pipeline.modeling.evaluate import labels, predict_scores, validation_settings
from nsh_pipeline.modeling.select import evaluate
from nsh_pipeline.modeling.tracking import git_sha
from nsh_pipeline.registry.backends import HISTORY_FILE, append_history, make_registry
from nsh_pipeline.registry.bundle import (
    BUNDLE_NAME, build_bundle, extract_bundle, make_version, production_record,
)
from nsh_pipeline.registry.decide import candidate_metrics, decide, gate_checks, registry_settings
from nsh_pipeline.utils.alerts import send_alert

logger = logging.getLogger(__name__)


def read_json(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def ci_run_url(env=None) -> str | None:
    env = os.environ if env is None else env
    if env.get("GITHUB_RUN_ID") and env.get("GITHUB_REPOSITORY"):
        server = env.get("GITHUB_SERVER_URL", "https://github.com")
        return f"{server}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
    return None


def score_on_test(model_dir, data_dir, threshold: float | None, metric: str) -> float:
    """One model's metric on the current test split (the same computation for candidate and production)."""
    model, _ = load_model(model_dir)
    test = load_split("test", data_dir)
    return float(evaluate(labels(test), predict_scores(model, test), float(threshold or 0.5))[metric])


def candidate_metric(bias: dict | None, metrics: dict, data_dir, metric: str,
                     rescore: bool) -> tuple[float | None, str | None]:
    """The candidate scored exactly like production when re-scoring (bias.json rounds to 4 decimals)."""
    if bias and rescore and bias.get("selected_model_dir") and split_path("test", data_dir).exists():
        try:
            return score_on_test(bias["selected_model_dir"], data_dir, bias.get("threshold"), metric), "rescored"
        except Exception as exc:
            logger.warning("Could not re-score the candidate (%s: %s); using bias.json", type(exc).__name__, exc)
    value = metrics.get(metric)
    return (float(value), "recorded") if value is not None else (None, None)


def production_metric(registry, production: dict | None, data_dir, metric: str,
                      rescore: bool) -> tuple[float | None, str | None]:
    """Production's metric on the CURRENT test split ("rescored"); the recorded value if that is not possible."""
    if production is None:
        return None, None
    if rescore and split_path("test", data_dir).exists():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bundle_dir = extract_bundle(registry.fetch(production["version"], tmp), Path(tmp) / "bundle")
                return score_on_test(bundle_dir / "model", data_dir, production.get("threshold"), metric), "rescored"
        except Exception as exc:  # fall back to the recorded metric rather than block the decision
            logger.warning("Could not re-score production %s (%s: %s); using its recorded %s",
                           production["version"], type(exc).__name__, exc, metric)
    value = (production.get("metrics") or {}).get(metric)
    return (float(value), "recorded") if value is not None else (None, None)


def unique_version(registry, version: str) -> str:
    """Add -2, -3, ... when the version already exists (two promotions from one commit in the same second)."""
    existing = set(registry.list_versions())
    candidate, n = version, 1
    while candidate in existing:
        n += 1
        candidate = f"{version}-{n}"
    return candidate


def run_register(params: dict, reports_dir="reports/model", data_dir="data/training", *, registry=None,
                 env: dict | None = None, alerts: bool | None = None, alert=send_alert) -> dict:
    settings = registry_settings(params)
    env = dict(os.environ if env is None else env)
    registry = registry or make_registry(settings, env)
    reports_dir = Path(reports_dir)
    validation = read_json(reports_dir / "validation.json")
    bias = read_json(reports_dir / "bias.json")
    metric = settings["metric"]

    production = registry.get_production()
    checks = gate_checks(validation, bias, validation_settings(params))
    metrics = candidate_metrics(bias) if bias else {}
    prod_value, prod_source = production_metric(registry, production, data_dir, metric,
                                                settings["rescore_production"])
    # Same computation on both sides: re-score the candidate only when production was re-scored.
    cand_value, _ = candidate_metric(bias, metrics, data_dir, metric, rescore=prod_source == "rescored")
    decision = decide(checks, cand_value, production, prod_value, prod_source,
                      metric=metric, min_improvement=float(settings["min_improvement"]))
    logger.info("Registry decision: %s", decision.reason)

    record = None
    if decision.promote:
        model_dir = Path(bias["selected_model_dir"])
        version = unique_version(registry, make_version(git_sha()))
        record = production_record(
            version, bias, load_metadata(model_dir), metrics,
            previous_version=production["version"] if production else None,
            uri=registry.uri(version), backend=registry.name, promoted_at=utc_now())
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_bundle(Path(tmp) / BUNDLE_NAME, model_dir, record, {
                "validation.json": reports_dir / "validation.json",
                "bias.json": reports_dir / "bias.json",
                "bias_report.md": reports_dir / "bias" / "bias_report.md",
            })
            record["uri"] = registry.push(bundle, version)
        registry.set_production(record)
        logger.info("Pushed %s to %s and made it production", version, record["uri"])

    current = record or production
    if current:
        write_json(current, settings["production_file"])
    report = {
        "generated_at": utc_now(),
        "backend": registry.name,
        "promoted": decision.promote,
        "version": record["version"] if record else None,
        "production_version": current["version"] if current else None,
        "reason": decision.reason,
        "decision": decision.to_dict(),
        "candidate": {"model_name": (bias or {}).get("selected_model"), "metrics": metrics,
                      "data_version": (bias or {}).get("data_version")},
        "previous_production": production,
        "ci_run": ci_run_url(env),
    }
    write_json(report, settings["decision_file"])
    append_history(Path(settings["local_dir"]) / HISTORY_FILE,
                   {"at": report["generated_at"], "event": "promoted" if decision.promote else "kept",
                    "version": report["version"], "production_version": report["production_version"],
                    "backend": registry.name, "reason": decision.reason})

    if settings["alerts"] if alerts is None else alerts:
        _alert(report, metric, alert, env)
    return report


def _alert(report: dict, metric: str, alert, env: dict) -> None:
    candidate = report["candidate"]
    value = candidate["metrics"].get(metric)
    lines = [
        "Model training pipeline completed.",
        f"Candidate: {candidate['model_name']} ({metric} {value:.4f} on test)" if value is not None
        else f"Candidate: {candidate['model_name']}",
        f"Decision: {report['reason']}",
        f"Production now: {report['production_version']} ({report['backend']} registry)",
    ]
    if report["ci_run"]:
        lines.append(f"Run: {report['ci_run']}")
    title = (f"NSH model promoted: {report['version']}" if report["promoted"]
             else f"NSH model not promoted: kept {report['production_version']}")
    if report["backend"] == "local":   # pull requests and local runs: say it is not the production registry
        title += " [local test registry]"
    alert(title, "\n".join(lines), env=env)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Push the selected model to the registry or keep production.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--reports-dir", default="reports/model")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--no-alerts", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_register(load_params(args.params), args.reports_dir, args.data_dir,
                          alerts=False if args.no_alerts else None)
    print(json.dumps({k: report[k] for k in ("promoted", "version", "production_version", "reason")}, indent=2))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
