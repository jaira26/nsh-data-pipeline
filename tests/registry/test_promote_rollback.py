"""End-to-end registry behaviour on the local backend: the rules the task assigner's tests require."""
import json

import pytest

from nsh_pipeline.modeling.dataset import load_split
from nsh_pipeline.modeling.evaluate import labels, predict_scores
from nsh_pipeline.modeling.select import evaluate
from nsh_pipeline.registry import notify, promote, rollback
from nsh_pipeline.registry.backends import LocalRegistry, RegistryError
from tests.registry.conftest import TRAINING_SAMPLE_DIR, AmountModel, params


def run(workspace, sender, p=None, **kw):
    p = p or params(workspace.tmp_path)
    reg = LocalRegistry(p["registry"]["local_dir"])
    report = promote.run_register(p, workspace.reports, TRAINING_SAMPLE_DIR, registry=reg, env={}, alert=sender, **kw)
    return report, reg


def test_first_model_is_promoted_and_recorded(workspace, alerts):
    workspace(pr_auc=0.32)
    report, reg = run(workspace, alerts)
    assert report["promoted"] and report["version"] == report["production_version"]
    prod = json.loads((workspace.tmp_path / "registry" / "production.json").read_text())
    assert prod["version"] == report["version"] and prod["previous_version"] is None
    assert prod["metrics"]["pr_auc"] == 0.32 and prod["model_name"] == "logistic_regression_reweighed"
    assert reg.list_versions() == [report["version"]]
    decision = json.loads((workspace.tmp_path / "reports" / "registry_decision.json").read_text())
    assert decision["promoted"] and decision["decision"]["checks"][-1]["name"] == "beats_production"
    assert alerts.sent[0][0].startswith("NSH model promoted")
    assert "training pipeline completed" in alerts.sent[0][1].lower()


def test_worse_model_is_never_promoted(workspace, alerts):
    workspace(pr_auc=0.32)
    first, reg = run(workspace, alerts)
    workspace(pr_auc=0.25)
    second, _ = run(workspace, alerts)
    assert not second["promoted"] and second["production_version"] == first["version"]
    assert reg.get_production()["version"] == first["version"]
    assert reg.list_versions() == [first["version"]]          # nothing pushed
    assert "rolled back" in second["reason"] and alerts.sent[-1][0].startswith("NSH model not promoted")
    history = (workspace.tmp_path / "registry" / "history.jsonl").read_text().splitlines()
    assert [json.loads(h)["event"] for h in history] == ["promoted", "kept"]


def test_better_model_replaces_production(workspace, alerts, monkeypatch):
    workspace(pr_auc=0.30)
    first, reg = run(workspace, alerts)
    monkeypatch.setattr(promote, "make_version", lambda sha: "v2-better")
    workspace(pr_auc=0.36)
    second, _ = run(workspace, alerts)
    assert second["promoted"] and reg.get_production()["previous_version"] == first["version"]


@pytest.mark.parametrize("kwargs", [{"validation_passed": False}, {"bias_passed": False}, {"recall": 0.2}])
def test_failing_candidate_is_never_promoted(workspace, alerts, kwargs):
    workspace(pr_auc=0.30)
    first, reg = run(workspace, alerts)
    workspace(pr_auc=0.50, **kwargs)                        # better metric, but a gate failed
    second, _ = run(workspace, alerts)
    assert not second["promoted"] and reg.get_production()["version"] == first["version"]


def test_failing_first_candidate_leaves_registry_empty(workspace, alerts):
    workspace(bias_passed=False)
    report, reg = run(workspace, alerts)
    assert not report["promoted"] and report["production_version"] is None
    assert reg.list_versions() == [] and not (workspace.tmp_path / "registry" / "production.json").exists()
    assert alerts.sent[0][0] == "NSH model not promoted: kept None [local test registry]"


def test_production_is_rescored_on_the_current_test_split(workspace, alerts):
    workspace(pr_auc=0.30)
    p = params(workspace.tmp_path, rescore_production=True)
    run(workspace, alerts, p)
    workspace(pr_auc=0.99)
    report, _ = run(workspace, alerts, p)
    test = load_split("test", TRAINING_SAMPLE_DIR)
    expected = evaluate(labels(test), predict_scores(AmountModel(), test), 0.53)["pr_auc"]
    decision = report["decision"]
    assert decision["production_metric_source"] == "rescored"
    assert decision["production_metric"] == pytest.approx(expected)


def test_rescore_falls_back_to_recorded_metric(workspace, alerts, tmp_path):
    workspace(pr_auc=0.30)
    p = params(workspace.tmp_path, rescore_production=True)
    _, reg = run(workspace, alerts, p)
    for bundle in (workspace.tmp_path / "registry" / "versions").rglob("bundle.tar.gz"):
        bundle.write_bytes(b"corrupt")
    workspace(pr_auc=0.31)
    report, _ = run(workspace, alerts, p)
    assert report["decision"]["production_metric_source"] == "recorded" and report["promoted"]


def test_alerts_can_be_disabled(workspace, alerts):
    workspace()
    run(workspace, alerts, alerts=False)
    assert alerts.sent == []


def test_promote_cli(workspace, monkeypatch, capsys):
    workspace()
    p = params(workspace.tmp_path)
    monkeypatch.setattr(promote, "load_params", lambda path: p)
    report = promote.main(["--reports-dir", str(workspace.reports), "--data-dir", str(TRAINING_SAMPLE_DIR),
                           "--no-alerts"])
    assert report["promoted"] and '"promoted": true' in capsys.readouterr().out


def test_same_second_versions_get_a_suffix(workspace, alerts, monkeypatch):
    monkeypatch.setattr(promote, "make_version", lambda sha: "v1")
    workspace(pr_auc=0.30)
    run(workspace, alerts)
    workspace(pr_auc=0.40)
    report, reg = run(workspace, alerts)
    assert report["version"] == "v1-2" and reg.list_versions() == ["v1", "v1-2"]


def test_ci_run_url():
    env = {"GITHUB_RUN_ID": "7", "GITHUB_REPOSITORY": "o/r"}
    assert promote.ci_run_url(env) == "https://github.com/o/r/actions/runs/7"
    assert promote.ci_run_url({}) is None


# --- manual rollback ---------------------------------------------------------------------------------

def two_versions(workspace, alerts, monkeypatch):
    workspace(pr_auc=0.30)
    first, reg = run(workspace, alerts)
    monkeypatch.setattr(promote, "make_version", lambda sha: "v2-better")
    workspace(pr_auc=0.36)
    run(workspace, alerts)
    return first["version"], reg


def test_manual_rollback_to_previous(workspace, alerts, monkeypatch):
    v1, reg = two_versions(workspace, alerts, monkeypatch)
    p = params(workspace.tmp_path)
    record = rollback.run_rollback(p, previous=True, reason="drift", registry=reg, env={}, alert=alerts)
    assert record["version"] == v1 and record["rolled_back_from"] == "v2-better"
    assert reg.get_production()["version"] == v1
    assert json.loads((workspace.tmp_path / "registry" / "production.json").read_text())["version"] == v1
    assert alerts.sent[-1][0] == f"NSH model rolled back to {v1}"
    assert '"manual_rollback"' in (workspace.tmp_path / "registry" / "history.jsonl").read_text()


def test_manual_rollback_errors(workspace, alerts, monkeypatch):
    v1, reg = two_versions(workspace, alerts, monkeypatch)
    p = params(workspace.tmp_path)
    with pytest.raises(RegistryError, match="already production"):
        rollback.run_rollback(p, "v2-better", registry=reg, env={}, alert=alerts)
    with pytest.raises(RegistryError, match="not found"):
        rollback.run_rollback(p, "v9", registry=reg, env={}, alert=alerts)
    with pytest.raises(RegistryError, match="--to"):
        rollback.run_rollback(p, registry=reg, env={}, alert=alerts)
    with pytest.raises(RegistryError, match="no previous"):
        rollback.run_rollback(p, previous=True, registry=LocalRegistry(workspace.tmp_path / "empty"), env={},
                              alert=alerts)


def test_rollback_cli(workspace, alerts, monkeypatch, capsys):
    v1, _ = two_versions(workspace, alerts, monkeypatch)
    p = params(workspace.tmp_path, alerts=False)
    monkeypatch.setattr(rollback, "load_params", lambda path: p)
    rollback.main(["--to", v1])
    assert v1 in capsys.readouterr().out


# --- failure notifications ---------------------------------------------------------------------------

def test_notify_classifies_validation_and_bias_failures(workspace):
    workspace(validation_passed=False)
    assert "validation gate" in notify.classify_failure(workspace.reports)[0]
    workspace(bias_passed=False)
    title, body = notify.classify_failure(workspace.reports)
    assert "bias gate" in title and "Nothing was pushed" in body
    workspace()
    assert notify.classify_failure(workspace.reports)[0] == "NSH model pipeline failed"


def test_notify_cli(workspace, monkeypatch):
    workspace(validation_passed=False)
    sent = []
    monkeypatch.setattr(notify, "send_alert", lambda title, body: sent.append((title, body)) or [])
    monkeypatch.setenv("GITHUB_RUN_ID", "9")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    notify.main(["--reports-dir", str(workspace.reports)])
    assert "validation gate" in sent[0][0] and "actions/runs/9" in sent[0][1]


def test_same_model_twice_is_kept_when_rescored(workspace, alerts):
    """Regression: bias.json rounds to 4 decimals; the re-scored production value must not lose to it."""
    workspace(pr_auc=0.3248)
    p = params(workspace.tmp_path, rescore_production=True)
    first, reg = run(workspace, alerts, p)
    second, _ = run(workspace, alerts, p)
    assert not second["promoted"] and reg.list_versions() == [first["version"]]
    assert second["decision"]["candidate_metric"] == second["decision"]["production_metric"]


def test_candidate_metric_falls_back_to_bias_json(tmp_path):
    bias = {"selected_model_dir": str(tmp_path / "missing"), "threshold": 0.5}
    assert promote.candidate_metric(bias, {"pr_auc": 0.3}, TRAINING_SAMPLE_DIR, "pr_auc", True) == (0.3, "recorded")
    assert promote.candidate_metric(None, {}, TRAINING_SAMPLE_DIR, "pr_auc", True) == (None, None)
