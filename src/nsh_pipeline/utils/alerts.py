"""Pipeline alerts by Slack incoming webhook and/or email (SMTP).

Credentials come only from environment variables (a local .env; see .env.example):
    SLACK_WEBHOOK_URL
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO, ALERT_EMAIL_FROM
With none set, the alert is logged and nothing is sent. Sending never raises: an alert that
fails must not hide the error that triggered it.
"""
import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10


def _format(body: str, details: dict | None) -> str:
    if not details:
        return body
    return f"{body}\n\nDetails:\n{json.dumps(details, indent=2, default=str)}"


def _send_slack(url: str, title: str, text: str) -> None:
    payload = json.dumps({"text": f"*{title}*\n{text}"}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if response.status >= 300:
            raise RuntimeError(f"Slack webhook returned HTTP {response.status}")


def _send_email(env: dict, title: str, text: str) -> None:
    message = EmailMessage()
    message["Subject"] = title
    message["From"] = env.get("ALERT_EMAIL_FROM") or env.get("SMTP_USER") or "nsh-pipeline@localhost"
    message["To"] = env["ALERT_EMAIL_TO"]
    message.set_content(text)
    with smtplib.SMTP(env["SMTP_HOST"], int(env.get("SMTP_PORT") or 587), timeout=TIMEOUT_SECONDS) as smtp:
        smtp.starttls()
        if env.get("SMTP_USER"):
            smtp.login(env["SMTP_USER"], env.get("SMTP_PASSWORD") or "")
        smtp.send_message(message)


def send_alert(title: str, body: str, details: dict | None = None, env: dict | None = None) -> list[str]:
    """Send an alert to every configured channel; return the channels it was delivered to.

    `details` is appended as JSON. `env` defaults to os.environ (tests pass their own). Returns
    [] when no channel is configured or every channel failed; never raises.
    """
    env = dict(os.environ if env is None else env)
    text = _format(body, details)
    channels = []
    if env.get("SLACK_WEBHOOK_URL"):
        channels.append(("slack", lambda: _send_slack(env["SLACK_WEBHOOK_URL"], title, text)))
    if env.get("SMTP_HOST") and env.get("ALERT_EMAIL_TO"):
        channels.append(("email", lambda: _send_email(env, title, text)))
    if not channels:
        logger.warning("ALERT (no Slack/SMTP configured, logged only): %s\n%s", title, text)
        return []
    delivered = []
    for name, send in channels:
        try:
            send()
            delivered.append(name)
            logger.info("Alert sent via %s: %s", name, title)
        except Exception as exc:  # an alert must never crash the pipeline
            logger.error("Alert via %s failed (%s): %s\n%s", name, type(exc).__name__, title, text)
    return delivered


def on_failure_callback(context: dict) -> None:
    """Airflow `on_failure_callback`: alert with the failed task, run, error and log link."""
    ti = context.get("task_instance")
    title = f"NSH pipeline: task {getattr(ti, 'task_id', '?')} failed"
    body = "\n".join([
        f"DAG: {getattr(ti, 'dag_id', '?')}",
        f"Run: {context.get('run_id', '?')}",
        f"Try: {getattr(ti, 'try_number', '?')}",
        f"Error: {context.get('exception')}",
        f"Log: {getattr(ti, 'log_url', 'n/a')}",
    ])
    send_alert(title, body)
