import json
import sys
import types

from nsh_pipeline.orchestration import summary


def test_anomaly_counts_sum_by_severity(tmp_path):
    path = tmp_path / "anomalies.json"
    path.write_text(json.dumps([
        {"check": "orphan_swipes", "severity": "Critical", "count": 5, "ids": []},
        {"check": "cents_as_dollars", "severity": "critical", "count": 5, "ids": []},
        {"check": "unparseable_money", "severity": "warning", "count": 5, "ids": []},
        {"check": "term_before_hire", "severity": "warning", "count": 0, "ids": []},
    ]))
    assert summary.anomaly_counts(path) == {"critical": 10, "warning": 5}


def test_anomaly_counts_missing_or_corrupt(tmp_path):
    assert summary.anomaly_counts(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert summary.anomaly_counts(bad) is None


def test_build_and_write_summary(tmp_path):
    result = summary.build_run_summary(
        run_id="manual__1", source="download", row_counts={"swipes": 4800},
        durations={"acquire_data": 12.34, "preprocess": 3.0, "bias": None},
        anomalies_path=tmp_path / "anomalies.json",
        stage_results={"bias": {"status": "placeholder"}}, dvc={"status": "skipped"}, commit="abc123",
    )
    assert result["total_task_time_s"] == 15.3
    assert result["task_durations_s"]["acquire_data"] == 12.3
    assert result["anomalies_by_severity"] is None
    out = summary.write_run_summary(result, tmp_path / "reports" / "run_summary.json")
    assert json.loads(out.read_text())["dataset_source"] == "download"


def test_git_commit_outside_repo(tmp_path):
    assert summary.git_commit(tmp_path) == "unknown"


class _TI:
    task_id, dag_id, try_number, log_url = "anomalies", "nsh_data_pipeline", 1, "http://log"


def test_notify_failure_uses_send_alert(monkeypatch):
    sent = {}
    fake = types.ModuleType("nsh_pipeline.utils.alerts")
    fake.send_alert = lambda title, body: sent.update(title=title, body=body)
    monkeypatch.setitem(sys.modules, "nsh_pipeline.utils.alerts", fake)
    summary.notify_failure({"task_instance": _TI(), "run_id": "manual__1", "exception": ValueError("x")})
    assert "anomalies" in sent["title"] and "manual__1" in sent["body"]


def test_notify_failure_without_alerts_module_logs(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "nsh_pipeline.utils.alerts", None)  # import raises ImportError
    summary.notify_failure({"task_instance": _TI(), "run_id": "r", "exception": "e"})
    assert "not merged yet" in caplog.text


def test_notify_failure_survives_broken_alert(monkeypatch, caplog):
    fake = types.ModuleType("nsh_pipeline.utils.alerts")

    def broken(title, body):
        raise RuntimeError("slack down")

    fake.send_alert = broken
    monkeypatch.setitem(sys.modules, "nsh_pipeline.utils.alerts", fake)
    summary.notify_failure({"task_instance": _TI(), "run_id": "r", "exception": "e"})
    assert "could not be sent" in caplog.text
