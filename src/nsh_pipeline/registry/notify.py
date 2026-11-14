"""Failure alerts for the CI model pipeline (GitHub Actions `if: failure()` step):

    python -m nsh_pipeline.registry.notify --event failure --reports-dir reports/model

Tells a blocked gate apart from a crash by reading the reports the stages wrote before exiting 1:
validation.json passed=false -> validation gate blocked; bias.json passed=false -> bias gate blocked;
anything else -> pipeline failed. Never exits non-zero itself (the job is already failing).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from nsh_pipeline.registry.promote import ci_run_url, read_json
from nsh_pipeline.utils.alerts import send_alert


def classify_failure(reports_dir) -> tuple[str, str]:
    """(title, body) describing why the pipeline stopped."""
    reports_dir = Path(reports_dir)
    validation = read_json(reports_dir / "validation.json")
    if validation is not None and not validation.get("passed", True):
        failed = [c for c in validation.get("gate", []) if not c.get("passed")]
        detail = "; ".join(f"{c['metric']} {c['value']:.4f} < {c['minimum']:.4f}" for c in failed) or "see report"
        return ("NSH model blocked by the validation gate",
                f"Model {validation.get('model_name')!r} failed hold-out validation: {detail}. Nothing was pushed.")
    bias = read_json(reports_dir / "bias.json")
    if bias is not None and not bias.get("passed", True):
        return ("NSH model blocked by the bias gate",
                f"{bias.get('selection_reason', 'No candidate passed the fairness gate.')} Nothing was pushed.")
    return ("NSH model pipeline failed", "A pipeline step failed before a registry decision. See the CI log.")


def main(argv=None) -> list[str]:
    parser = argparse.ArgumentParser(description="Send a model-pipeline alert.")
    parser.add_argument("--event", choices=["failure"], default="failure")
    parser.add_argument("--reports-dir", default="reports/model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    title, body = classify_failure(args.reports_dir)
    url = ci_run_url(os.environ)
    return send_alert(title, f"{body}\nRun: {url}" if url else body)


if __name__ == "__main__":  # pragma: no cover
    main()
