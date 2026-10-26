import subprocess

from nsh_pipeline.orchestration import versioning

DVC_YAML = "stages:\n  acquire:\n    cmd: x\n  preprocess:\n    cmd: y\n"


def test_skips_when_dvc_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning.shutil, "which", lambda _: None)
    assert versioning.dvc_version(tmp_path)["status"] == "skipped"


def test_skips_when_not_a_dvc_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning.shutil, "which", lambda _: "/usr/bin/dvc")
    assert versioning.dvc_version(tmp_path)["reason"] == "not a dvc repo"


def test_stage_names_read_from_dvc_yaml(tmp_path):
    (tmp_path / "dvc.yaml").write_text(DVC_YAML)
    assert versioning.dvc_stages(tmp_path) == ["acquire", "preprocess"]
    assert versioning.dvc_stages(tmp_path / "missing") == []


def _fake_repo(tmp_path, monkeypatch, failing=()):
    (tmp_path / ".dvc").mkdir()
    (tmp_path / "dvc.yaml").write_text(DVC_YAML)
    monkeypatch.setattr(versioning.shutil, "which", lambda _: "/usr/bin/dvc")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        code = 1 if cmd[-1] in failing else 0
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr="boom" if code else "")

    monkeypatch.setattr(versioning.subprocess, "run", fake_run)
    return calls


def test_commits_every_stage_and_pushes(tmp_path, monkeypatch):
    calls = _fake_repo(tmp_path, monkeypatch)
    result = versioning.dvc_version(tmp_path, push=True)
    assert result["status"] == "ok" and result["push"] == "ok"
    assert ["dvc", "commit", "-f", "acquire"] in calls and ["dvc", "push"] in calls


def test_partial_when_a_stage_fails(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, failing=("preprocess",))
    result = versioning.dvc_version(tmp_path)
    assert result["status"] == "partial"
    assert result["stages"]["preprocess"].startswith("failed")
    assert result["push"] == "not requested"


def test_failed_when_dvc_cannot_run(tmp_path, monkeypatch):
    (tmp_path / ".dvc").mkdir()
    (tmp_path / "dvc.yaml").write_text(DVC_YAML)
    monkeypatch.setattr(versioning.shutil, "which", lambda _: "/usr/bin/dvc")

    def boom(*args, **kwargs):
        raise OSError("dvc crashed")

    monkeypatch.setattr(versioning.subprocess, "run", boom)
    assert versioning.dvc_version(tmp_path)["status"] == "failed"
