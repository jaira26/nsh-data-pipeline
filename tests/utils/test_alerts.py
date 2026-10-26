import json
import logging
from types import SimpleNamespace
from unittest import mock

import pytest

from nsh_pipeline.utils import alerts

SLACK = {"SLACK_WEBHOOK_URL": "https://hooks.slack.test/T000/B000/XXX"}
SMTP = {"SMTP_HOST": "smtp.test", "SMTP_PORT": "2525", "SMTP_USER": "bot@test", "SMTP_PASSWORD": "pw",
        "ALERT_EMAIL_TO": "team@test"}


def test_no_credentials_logs_and_returns_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        assert alerts.send_alert("Title", "Body", {"x": 1}, env={}) == []
    assert "no Slack/SMTP configured" in caplog.text and '"x": 1' in caplog.text


def test_default_env_is_os_environ(monkeypatch):
    for name in [*SLACK, *SMTP]:
        monkeypatch.delenv(name, raising=False)
    assert alerts.send_alert("Title", "Body") == []


def test_slack_payload():
    response = mock.MagicMock(status=200)
    with mock.patch.object(alerts.urllib.request, "urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = response
        assert alerts.send_alert("Title", "Body", {"check": "orphan_keys"}, env=SLACK) == ["slack"]
    request = urlopen.call_args.args[0]
    assert request.full_url == SLACK["SLACK_WEBHOOK_URL"]
    text = json.loads(request.data)["text"]
    assert text.startswith("*Title*\nBody") and '"check": "orphan_keys"' in text


def test_email_message():
    with mock.patch.object(alerts.smtplib, "SMTP") as smtp_cls:
        assert alerts.send_alert("Title", "Body", env=SMTP) == ["email"]
    smtp_cls.assert_called_once_with("smtp.test", 2525, timeout=alerts.TIMEOUT_SECONDS)
    smtp = smtp_cls.return_value.__enter__.return_value
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("bot@test", "pw")
    message = smtp.send_message.call_args.args[0]
    assert (message["Subject"], message["To"], message["From"]) == ("Title", "team@test", "bot@test")


def test_email_without_login():
    env = {"SMTP_HOST": "smtp.test", "ALERT_EMAIL_TO": "team@test", "ALERT_EMAIL_FROM": "pipe@test"}
    with mock.patch.object(alerts.smtplib, "SMTP") as smtp_cls:
        assert alerts.send_alert("T", "B", env=env) == ["email"]
    smtp = smtp_cls.return_value.__enter__.return_value
    smtp.login.assert_not_called()
    assert smtp.send_message.call_args.args[0]["From"] == "pipe@test"


@pytest.mark.parametrize("error", [OSError("network down"), RuntimeError("boom")])
def test_failures_never_raise_and_other_channels_still_send(error, caplog):
    with mock.patch.object(alerts.urllib.request, "urlopen", side_effect=error), \
            mock.patch.object(alerts.smtplib, "SMTP"):
        assert alerts.send_alert("Title", "Body", env={**SLACK, **SMTP}) == ["email"]
    assert "Alert via slack failed" in caplog.text
    assert SLACK["SLACK_WEBHOOK_URL"] not in caplog.text  # the webhook URL is a secret


def test_slack_http_error_status_is_a_failure():
    with mock.patch.object(alerts.urllib.request, "urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = mock.MagicMock(status=500)
        assert alerts.send_alert("Title", "Body", env=SLACK) == []


def test_on_failure_callback_sends_task_details():
    ti = SimpleNamespace(task_id="anomalies", dag_id="nsh_data_pipeline", try_number=1, log_url="http://log")
    with mock.patch.object(alerts, "send_alert") as send:
        alerts.on_failure_callback({"task_instance": ti, "run_id": "manual_1", "exception": ValueError("bad")})
    title, body = send.call_args.args
    assert title == "NSH pipeline: task anomalies failed"
    assert "Run: manual_1" in body and "Error: bad" in body and "Log: http://log" in body


def test_run_summary_notify_failure_uses_send_alert():
    """M4's DAG callback imports send_alert(title, body) from this module."""
    from nsh_pipeline.orchestration.summary import notify_failure

    with mock.patch.object(alerts, "send_alert") as send:
        notify_failure({"task_instance": SimpleNamespace(task_id="stats"), "run_id": "r1"})
    assert send.call_args.args[0] == "NSH pipeline: task stats failed"
