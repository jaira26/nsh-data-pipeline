"""Registry of the CLI stages the DAG runs.

Each entry is ``task_id -> (module, args)``. The DAG runs ``python -m <module> <args>``
from the repo root, which is the interface contract in the hand-off guide (section 9):
every stage is a function plus a ``python -m ...`` CLI taking paths and ``--params``.

When a teammate's module is not merged yet, the task logs a placeholder message and
succeeds, so the DAG stays runnable end to end at every point in the week.

To change a command (e.g. M3 names a flag differently), edit it here only.
"""
from __future__ import annotations

PROCESSED = ["--processed-dir", "data/processed", "--params", "params.yaml"]

STAGES: dict[str, tuple[str, list[str]]] = {
    # M3: raw files vs manifest.json + light raw schemas, before preprocessing (plan section 8, stage 3)
    "validate_raw": ("nsh_pipeline.validation.run", ["--raw-dir", "data/raw", "--params", "params.yaml",
                                                     "--step", "raw"]),
    # M3: schema validation of the processed tables (fails the run on a schema error)
    "validate_processed": ("nsh_pipeline.validation.run", [*PROCESSED, "--step", "schema"]),
    # FE: feature tables into data/features/
    "build_features": (
        "nsh_pipeline.features.run",
        ["--processed-dir", "data/processed", "--out-dir", "data/features", "--params", "params.yaml"],
    ),
    # M3: statistics + drift (reports/stats/)
    "stats": ("nsh_pipeline.validation.run", [*PROCESSED, "--step", "stats"]),
    # M3: anomaly checks + alert (reports/anomalies.json, fails the run on a critical anomaly)
    "anomalies": ("nsh_pipeline.validation.run", [*PROCESSED, "--step", "anomalies"]),
    # M5: bias slicing (reports/bias_report.md)
    "bias": ("nsh_pipeline.bias.slicing", PROCESSED),
}
