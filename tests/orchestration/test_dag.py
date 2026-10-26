"""DAG integrity. Runs where Airflow is installed (inside the container); skipped in CI."""
import importlib
from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[2] / "dags"
TASKS = {"acquire_data", "validate_raw", "preprocess", "validate_processed", "build_features",
         "stats", "anomalies", "bias", "dvc_version", "run_summary"}


def _dag():
    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.get_dag("nsh_data_pipeline")


def test_dag_loads_with_expected_tasks():
    dag = _dag()
    assert set(dag.task_ids) == TASKS
    assert dag.get_task("validate_raw").upstream_task_ids == {"acquire_data"}
    assert dag.get_task("preprocess").upstream_task_ids == {"validate_raw"}
    assert dag.get_task("dvc_version").upstream_task_ids == {"stats", "anomalies", "bias"}


def test_sequential_layout(monkeypatch):
    monkeypatch.setenv("NSH_PARALLEL_REPORTS", "false")
    dag = _dag()
    assert dag.get_task("anomalies").upstream_task_ids == {"stats"}
    assert dag.get_task("dvc_version").upstream_task_ids == {"bias"}


def test_every_cli_task_is_registered():
    stages = importlib.import_module("nsh_pipeline.orchestration.stages").STAGES
    assert {"validate_raw", "validate_processed", "build_features", "stats", "anomalies", "bias"} == set(stages)
