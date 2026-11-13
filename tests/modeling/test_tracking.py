import re
import shutil
import subprocess

import mlflow
import pytest

from nsh_pipeline.modeling import tracking

URL = "https://github.com/jaira26/nsh-data-pipeline/releases/download/data-v1.1/nsh-raw-small-v1.1.tar.gz"


@pytest.mark.parametrize("card, expected", [
    ({"data_version": "sample"}, "sample"),
    ({"source_data": {"configured_release_url": URL, "generator_version": "1.1"}}, "data-v1.1"),
    ({"source_data": {"generator_version": "1.1"}}, "generated-v1.1"),
    ({}, "unknown"),
])
def test_data_version_comes_from_the_dataset_card(card, expected):
    assert tracking.data_version(card) == expected


def test_git_sha_outside_a_repo_falls_back_to_ci_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert tracking.git_sha(tmp_path) == "abc123"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_sha_reports_commit_and_dirty_state(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f.txt").write_text("a")
    git("add", "f.txt")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    assert re.fullmatch(r"[0-9a-f]{40}", tracking.git_sha(tmp_path))
    (tmp_path / "f.txt").write_text("b")
    assert tracking.git_sha(tmp_path).endswith("-dirty")


def test_data_hash_changes_with_the_data(training_sample_dir, tmp_path):
    shutil.copytree(training_sample_dir, tmp_path / "t")
    before = tracking.data_hash(tmp_path / "t")
    assert before == tracking.data_hash(training_sample_dir)
    (tmp_path / "t" / "test.parquet").write_bytes(b"changed")
    assert tracking.data_hash(tmp_path / "t") != before


def test_run_tags_have_the_required_tags(training_sample_dir):
    tags = tracking.run_tags({"data_version": "sample"}, training_sample_dir, {"extra": 1})
    assert all(tags[key] for key in tracking.REQUIRED_TAGS)
    assert tags["data_version"] == "sample" and tags["extra"] == "1"


def test_setup_mlflow_uses_env_uri_and_experiment(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/env.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    experiment_id = tracking.setup_mlflow({"tracking_uri": "sqlite:///ignored.db", "experiment": "nsh-test"})
    assert mlflow.get_tracking_uri() == uri
    assert mlflow.get_experiment(experiment_id).name == "nsh-test"


def test_setup_mlflow_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    experiment_id = tracking.setup_mlflow({"tracking_uri": "sqlite:///sub/m.db"})
    assert (tmp_path / "sub" / "m.db").exists()
    assert mlflow.get_experiment(experiment_id).name == tracking.DEFAULT_EXPERIMENT
