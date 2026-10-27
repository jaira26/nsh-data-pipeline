import copy
import json
import shutil
from unittest import mock

import pytest
import yaml

from nsh_pipeline.orchestration.stages import STAGES
from nsh_pipeline.validation import run
from nsh_pipeline.validation.common import ValidationFailed
from tests.validation.schemas.test_features import GOOD

GOOD_PMPM = GOOD["pmpm_monthly"]
M3_TASKS = ("validate_processed", "stats", "anomalies")


def as_report_mode(params):
    params = copy.deepcopy(params)
    params["validation"]["mode"] = "report"
    return params


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def alerts():
    with mock.patch.object(run, "send_alert") as send:
        yield send


@pytest.fixture
def full_processed_dir(tables, tmp_path):
    """All sample tables, including the swipe with unparseable money (a critical anomaly)."""
    out = tmp_path / "processed_full"
    out.mkdir()
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return out


@pytest.fixture
def injected_raw_dir(sample_raw_dir, tmp_path):
    """The sample raw data with a manifest that lists the "N/A" swipe as an injected issue."""
    raw = shutil.copytree(sample_raw_dir, tmp_path / "raw_injected")
    manifest = read(raw / "manifest.json")
    manifest["injected_issues"] = [
        {"issue": "unparseable_money", "collection": "wex_claims", "count": 1, "ids": ["WEX-000005"]},
        {"issue": "denial_skew", "collection": "wex_claims", "count": 1, "ids": ["WEX-000003"]},
    ]
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw


# ---------------------------------------------------------------- raw
def test_raw_step_passes_on_the_sample(sample_raw_dir, params, tmp_path, alerts):
    result = run.run_raw(sample_raw_dir, params, tmp_path)
    assert result["status"] == "passed" and result["files"] == []
    assert len(result["tables"]) == 9
    assert read(tmp_path / "validation" / "raw_schema.json")["status"] == "passed"
    alerts.assert_not_called()


def test_raw_step_without_manifest(sample_raw_dir, params, tmp_path, alerts):
    raw = shutil.copytree(sample_raw_dir, tmp_path / "raw")
    (raw / "manifest.json").unlink()
    with pytest.raises(ValidationFailed, match="raw data failed validation"):
        run.run_raw(raw, params, tmp_path / "reports")
    result = run.run_raw(raw, as_report_mode(params), tmp_path / "reports")
    assert result["status"] == "failed" and result["files"][0]["file"] == "manifest.json"
    assert alerts.call_count == 2


# ---------------------------------------------------------------- schema
def test_schema_step_passes_and_skips_missing_features(processed_dir, params, tmp_path, alerts):
    result = run.run_schema(processed_dir, params, tmp_path, tmp_path / "no_features")
    assert result == {"step": "schema", "status": "passed", "failed_tables": [], "features": "skipped"}
    assert read(tmp_path / "validation" / "features_schema.json")["status"] == "skipped"
    assert len(read(tmp_path / "validation" / "processed_schema.json")["tables"]) == 9
    alerts.assert_not_called()


def test_schema_step_missing_table_fails_in_strict_mode(processed_dir, params, tmp_path, alerts):
    (processed_dir / "ledger.parquet").unlink()
    with pytest.raises(ValidationFailed, match="ledger"):
        run.run_schema(processed_dir, params, tmp_path)
    result = run.run_schema(processed_dir, as_report_mode(params), tmp_path)
    assert result["failed_tables"] == ["ledger"]
    assert alerts.called


def test_schema_step_validates_feature_tables_that_exist(processed_dir, params, tmp_path, alerts):
    features = tmp_path / "features"
    features.mkdir()
    GOOD_PMPM.to_parquet(features / "pmpm_monthly.parquet", index=False)
    assert run.run_schema(processed_dir, params, tmp_path, features)["features"] == "passed"
    tables = {t["table"]: t["status"] for t in read(tmp_path / "validation" / "features_schema.json")["tables"]}
    assert tables == {"book_of_business": "skipped", "pmpm_monthly": "passed",
                      "substantiation_rate": "skipped", "claims_aging": "skipped"}
    GOOD_PMPM.assign(mlr=-1.0).to_parquet(features / "pmpm_monthly.parquet", index=False)
    with pytest.raises(ValidationFailed, match="pmpm_monthly"):
        run.run_schema(processed_dir, params, tmp_path, features)


# ---------------------------------------------------------------- stats
def test_stats_step_writes_stats_and_creates_then_uses_the_baseline(processed_dir, params, tmp_path):
    stale = tmp_path / "stats" / "old_table.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    first = run.run_stats(processed_dir, params, tmp_path)
    assert first["drift_status"] == "no_baseline" and not stale.exists()
    assert sorted(p.stem for p in (tmp_path / "stats").glob("*.json")) == first["tables"]
    second = run.run_stats(processed_dir, params, tmp_path)
    assert second["drift_status"] == "ok" and second["headline"]["drifted"] is False
    assert read(tmp_path / "drift.json")["status"] == "ok"
    assert run.run_stats(processed_dir, params, tmp_path, update_baseline=True)["drift_status"] == "baseline_updated"


# ---------------------------------------------------------------- anomalies
def test_anomalies_step_on_clean_data(processed_dir, sample_raw_dir, params, tmp_path, alerts):
    summary = run.run_anomalies(processed_dir, params, tmp_path, raw_dir=sample_raw_dir)
    assert (summary["critical_failed"], summary["warnings_failed"]) == (0, 0)
    assert summary["as_of"] == params["reference_date"] and summary["mode"] == "strict"
    assert isinstance(read(tmp_path / "anomalies.json"), list)
    assert (tmp_path / "anomalies.md").exists()
    assert read(tmp_path / "validation" / "anomalies_summary.json")["checks"]["orphan_keys"]["passed"] is True
    assert read(tmp_path / "injection_detection.json")["status"] == "no_injected_issues"
    alerts.assert_not_called()


def test_anomalies_step_critical_fails_strict_and_alerts(full_processed_dir, sample_raw_dir, params,
                                                         tmp_path, alerts):
    with pytest.raises(ValidationFailed, match="unparseable_money"):
        run.run_anomalies(full_processed_dir, params, tmp_path, raw_dir=sample_raw_dir)
    title, body, details = alerts.call_args.args
    assert title == "NSH validation: 1 critical anomalies"
    assert "unparseable_money" in body and details == {"unparseable_money": ["swipes:WEX-000005"]}
    assert read(tmp_path / "anomalies.json")  # reports are written before the failure


def test_under_airflow_strict_mode_raises_details_instead_of_alerting(full_processed_dir, sample_raw_dir, params,
                                                                      tmp_path, alerts, monkeypatch):
    """The DAG's on_failure_callback alerts with the exception text; M3 must not alert twice."""
    monkeypatch.setenv("AIRFLOW_CTX_DAG_ID", "nsh_data_pipeline")
    with pytest.raises(ValidationFailed) as failure:
        run.run_anomalies(full_processed_dir, params, tmp_path, raw_dir=sample_raw_dir)
    alerts.assert_not_called()
    message = str(failure.value)
    assert message.startswith("1 critical anomalies")
    assert "- unparseable_money: 1 records have missing money" in message
    assert "unparseable_money: ['swipes:WEX-000005']" in message


def test_under_airflow_report_mode_still_alerts(full_processed_dir, sample_raw_dir, params, tmp_path,
                                                alerts, monkeypatch):
    """In report mode no task fails, so no callback fires: M3 sends the alert itself."""
    monkeypatch.setenv("AIRFLOW_CTX_DAG_ID", "nsh_data_pipeline")
    run.run_anomalies(full_processed_dir, as_report_mode(params), tmp_path, raw_dir=sample_raw_dir)
    assert alerts.call_count == 1


def test_under_airflow_detection(monkeypatch):
    monkeypatch.delenv("AIRFLOW_CTX_DAG_ID", raising=False)
    assert run.under_airflow() is False
    monkeypatch.setenv("AIRFLOW_CTX_DAG_ID", "nsh_data_pipeline")
    assert run.under_airflow() is True


def test_anomalies_step_report_mode_and_answer_key(full_processed_dir, injected_raw_dir, params,
                                                   tmp_path, alerts):
    summary = run.run_anomalies(full_processed_dir, as_report_mode(params), tmp_path, raw_dir=injected_raw_dir,
                                as_of="2025-06-15")
    assert summary["critical_failed"] == 1 and summary["all_injected_detected"] is True
    detection = read(tmp_path / "injection_detection.json")
    statuses = {i["issue"]: i["status"] for i in detection["issues"]}
    assert statuses == {"unparseable_money": "detected", "denial_skew": "out_of_scope"}
    assert alerts.called


def test_anomalies_without_raw_dir_skip_raw_parts(processed_dir, params, tmp_path):
    summary = run.run_anomalies(processed_dir, params, tmp_path, raw_dir=tmp_path / "missing")
    assert summary["critical_failed"] == 0
    assert not (tmp_path / "injection_detection.json").exists()


# ---------------------------------------------------------------- steps and CLI
def test_run_step_all_and_unknown_step(processed_dir, sample_raw_dir, params, tmp_path):
    out = run.run_step("all", params, processed_dir, sample_raw_dir, tmp_path / "f", tmp_path / "r")
    assert [out[s]["step"] for s in ("raw", "schema", "stats")] == ["raw", "schema", "stats"]
    assert out["anomalies"]["critical_failed"] == 0
    no_raw = run.run_step("all", params, processed_dir, tmp_path / "missing", tmp_path / "f", tmp_path / "r2")
    assert no_raw["raw"]["status"] == "skipped"
    with pytest.raises(ValueError, match="Unknown step"):
        run.run_step("nope", params)


@pytest.fixture
def project(processed_dir, params, tmp_path, monkeypatch):
    """A repo-like folder: data/processed + params.yaml, as the DAG's working directory."""
    root = tmp_path / "project"
    shutil.copytree(processed_dir, root / "data" / "processed")
    params = copy.deepcopy(params)
    params["validation"]["drift"]["baseline_dir"] = "reports/baseline"
    (root / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.chdir(root)
    return root


@pytest.mark.parametrize("task_id", M3_TASKS)
def test_cli_accepts_the_exact_dag_commands(project, task_id, alerts):
    module, args = STAGES[task_id]
    assert module == "nsh_pipeline.validation.run"
    result = run.main(args)
    assert result["step"] == {"validate_processed": "schema"}.get(task_id, task_id)


def test_cli_exits_1_on_strict_failure(project, alerts):
    (project / "data" / "processed" / "members.parquet").unlink()
    with pytest.raises(SystemExit) as exit_info:
        run.main(["--step", "schema"])
    assert exit_info.value.code == 1
