"""
### NSH data pipeline

acquire_data -> validate_raw -> preprocess -> validate_processed -> build_features
    -> [stats | anomalies | bias] -> dvc_version -> run_summary

* `acquire_data` and `preprocess` call M1/M2's functions directly (row counts go to XCom).
* The other stages run their owner's CLI (`python -m ...`), registered in
  `src/nsh_pipeline/orchestration/stages.py`. Until a module is merged, its task logs a
  placeholder and succeeds.
* Set `NSH_PARALLEL_REPORTS=false` in `.env` to run stats, anomalies and bias one after
  another (the "before" Gantt screenshot).
* Every failure calls `notify_failure`, which uses M3's `send_alert` once it exists.

No business logic lives here: this file only wires tasks together.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(os.environ.get("NSH_PROJECT_ROOT", "/opt/airflow/project"))
PARAMS = ROOT / "params.yaml"
PARALLEL_REPORTS = os.environ.get("NSH_PARALLEL_REPORTS", "true").strip().lower() == "true"
DVC_PUSH = os.environ.get("DVC_PUSH", "false").strip().lower() == "true"


# Imports live inside the callables so the scheduler parses this file quickly.
def _load_params():
    from nsh_pipeline.config import load_params

    return load_params(PARAMS)


def acquire():
    from nsh_pipeline.acquire.download import acquire_data

    return acquire_data(_load_params(), ROOT / "data/raw")["source"]


def preprocess():
    from nsh_pipeline.orchestration.runner import jsonable
    from nsh_pipeline.preprocess.run import run_preprocess

    params = _load_params()
    counts = run_preprocess(ROOT / "data/raw", ROOT / "data/processed", params["reference_date"])
    return jsonable(counts)


def run_stage(task_id: str):
    from nsh_pipeline.orchestration.runner import run_module
    from nsh_pipeline.orchestration.stages import STAGES

    module, args = STAGES[task_id]
    return run_module(module, args, cwd=ROOT)


def version_data():
    from nsh_pipeline.orchestration.versioning import dvc_version

    return dvc_version(ROOT, push=DVC_PUSH)


def summarize(**context):
    from nsh_pipeline.orchestration.stages import STAGES
    from nsh_pipeline.orchestration.summary import build_run_summary, git_commit, write_run_summary

    ti, dag_run = context["ti"], context["dag_run"]
    durations = {t.task_id: t.duration for t in dag_run.get_task_instances() if t.task_id != "run_summary"}
    summary = build_run_summary(
        run_id=dag_run.run_id,
        source=ti.xcom_pull(task_ids="acquire_data"),
        row_counts=ti.xcom_pull(task_ids="preprocess"),
        durations=durations,
        anomalies_path=ROOT / "reports/anomalies.json",
        stage_results={task_id: ti.xcom_pull(task_ids=task_id) for task_id in STAGES},
        dvc=ti.xcom_pull(task_ids="dvc_version"),
        commit=git_commit(ROOT),
    )
    return str(write_run_summary(summary, ROOT / "reports/run_summary.json"))


def _on_failure(context):
    from nsh_pipeline.orchestration.summary import notify_failure

    notify_failure(context)


with DAG(
    dag_id="nsh_data_pipeline",
    description="Acquire, preprocess, validate, feature-engineer and analyse synthetic NSH data",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # triggered manually; the as-of date comes from params.yaml
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "nsh-team",
        "retries": 0,  # stages are deterministic; a retry would fail the same way
        "execution_timeout": timedelta(hours=1),
        "on_failure_callback": _on_failure,
    },
    tags=["nsh", "mlops"],
    doc_md=__doc__,
) as dag:
    acquire_task = PythonOperator(
        task_id="acquire_data", python_callable=acquire, retries=2, retry_delay=timedelta(minutes=1)
    )
    validate_raw_task = PythonOperator(
        task_id="validate_raw", python_callable=run_stage, op_kwargs={"task_id": "validate_raw"}
    )
    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)
    validate_task = PythonOperator(
        task_id="validate_processed", python_callable=run_stage, op_kwargs={"task_id": "validate_processed"}
    )
    features_task = PythonOperator(
        task_id="build_features", python_callable=run_stage, op_kwargs={"task_id": "build_features"}
    )
    report_tasks = [
        PythonOperator(task_id=name, python_callable=run_stage, op_kwargs={"task_id": name})
        for name in ("stats", "anomalies", "bias")
    ]
    dvc_task = PythonOperator(task_id="dvc_version", python_callable=version_data)
    summary_task = PythonOperator(task_id="run_summary", python_callable=summarize)

    acquire_task >> validate_raw_task >> preprocess_task >> validate_task >> features_task
    if PARALLEL_REPORTS:
        features_task >> report_tasks >> dvc_task
    else:
        stats_task, anomalies_task, bias_task = report_tasks
        features_task >> stats_task >> anomalies_task >> bias_task >> dvc_task
    dvc_task >> summary_task
