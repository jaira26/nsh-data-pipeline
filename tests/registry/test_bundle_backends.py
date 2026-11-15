import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nsh_pipeline.registry.backends import GcpRegistry, LocalRegistry, RegistryError, append_history, make_registry
from nsh_pipeline.registry.bundle import (
    build_bundle, check_version, extract_bundle, make_version, production_record, read_manifest,
)
from nsh_pipeline.registry.decide import registry_settings
from tests.registry.conftest import AmountModel, metadata

from nsh_pipeline.modeling.artifacts import load_model, save_model


def test_make_version_is_sortable_and_safe():
    v = make_version("2f68d25b10cd-dirty", datetime(2026, 11, 14, 15, 36, 17, tzinfo=timezone.utc))
    assert v == "v20261114-153617-2f68d25"
    assert make_version(None).endswith("-nogit")
    with pytest.raises(ValueError):
        check_version("../evil")


def test_production_record_contract():
    record = production_record("v1", {"selected_model": "lr", "threshold": 0.5, "data_version": "data-v1.1"},
                               metadata("lr"), {"pr_auc": 0.3, "confusion_matrix": [[1]]},
                               previous_version=None, uri="u", backend="local", promoted_at="t")
    for key in ("version", "metrics", "promoted_at", "previous_version"):
        assert key in record
    assert record["metrics"] == {"pr_auc": 0.3}


def test_bundle_round_trip(tmp_path):
    model_dir = save_model(AmountModel(), metadata("lr"), tmp_path / "m")
    report = tmp_path / "bias.json"
    report.write_text("{}")
    bundle = build_bundle(tmp_path / "out" / "bundle.tar.gz", model_dir, {"version": "v1"},
                          {"bias.json": report, "missing.json": tmp_path / "nope.json"})
    out = extract_bundle(bundle, tmp_path / "x")
    assert read_manifest(out) == {"version": "v1"}
    assert (out / "reports" / "bias.json").exists() and not (out / "reports" / "missing.json").exists()
    model, meta = load_model(out / "model")
    assert meta["model_name"] == "lr" and hasattr(model, "predict_proba")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_bundle_needs_complete_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_bundle(tmp_path / "b.tar.gz", tmp_path, {}, {})


def test_extract_skips_unsafe_members(tmp_path):
    bundle = tmp_path / "b.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name in ("manifest.json", "other/x.txt"):
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    out = extract_bundle(bundle, tmp_path / "x")
    assert (out / "manifest.json").exists() and not (out / "other").exists()


def test_local_registry(tmp_path):
    reg = LocalRegistry(tmp_path / "reg")
    assert reg.get_production() is None and reg.list_versions() == []
    src = tmp_path / "bundle.tar.gz"
    src.write_bytes(b"data")
    reg.push(src, "v1")
    with pytest.raises(RegistryError):
        reg.push(src, "v1")
    assert reg.list_versions() == ["v1"]
    assert reg.fetch("v1", tmp_path / "f").read_bytes() == b"data"
    with pytest.raises(RegistryError):
        reg.fetch("v2", tmp_path / "f")
    reg.set_production({"version": "v1"})
    assert reg.get_production() == {"version": "v1"}


class FakeGcloud:
    """Records gcloud calls; answers `versions list` and `generic download` from a dict of package -> files."""

    def __init__(self, packages=None, fail=None):
        self.calls, self.packages, self.fail = [], packages or {}, fail

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", self.fail)
        args = dict(a.split("=", 1) for a in cmd if a.startswith("--") and "=" in a)
        package = args.get("--package")
        if cmd[2:4] == ["versions", "list"]:
            if package not in self.packages:
                return subprocess.CompletedProcess(cmd, 1, "", "ERROR: NOT_FOUND: package")
            names = [{"name": f"projects/p/.../versions/{v}"} for v in self.packages[package]]
            return subprocess.CompletedProcess(cmd, 0, json.dumps(names), "")
        if cmd[2:4] == ["generic", "download"]:
            for name, content in self.packages[package][args["--version"]].items():
                (Path(args["--destination"]) / name).write_text(content)
        return subprocess.CompletedProcess(cmd, 0, "", "")


def gcp(runner):
    return GcpRegistry("my-proj", "us-east1", "nsh-models", "nsh-denial-model", runner=runner)


def test_gcp_push_uses_generic_upload(tmp_path):
    runner = FakeGcloud()
    uri = gcp(runner).push(tmp_path / "bundle.tar.gz", "v1")
    assert uri == "us-east1-generic.pkg.dev/my-proj/nsh-models/nsh-denial-model:v1"
    cmd = runner.calls[0]
    assert cmd[:4] == ["gcloud", "artifacts", "generic", "upload"]
    assert "--project=my-proj" in cmd and "--package=nsh-denial-model" in cmd and "--version=v1" in cmd


def test_gcp_production_pointer(tmp_path):
    runner = FakeGcloud({"nsh-denial-model-production": {"p2-v2": {"production.json": '{"version": "v2"}'},
                                                         "p1-v1": {"production.json": '{"version": "v1"}'}}})
    reg = gcp(runner)
    assert reg.get_production() == {"version": "v2"}   # newest first (sort-by=~createTime)
    reg.set_production({"version": "v3"})
    upload = runner.calls[-1]
    assert "--package=nsh-denial-model-production" in upload
    assert any(a.startswith("--version=p") and a.endswith("-v3") for a in upload)


def test_gcp_no_pointer_yet_and_errors(tmp_path):
    assert gcp(FakeGcloud()).get_production() is None
    assert gcp(FakeGcloud()).list_versions() == []
    with pytest.raises(RegistryError, match="PERMISSION_DENIED"):
        gcp(FakeGcloud(fail="PERMISSION_DENIED")).push(tmp_path / "b", "v1")
    with pytest.raises(RegistryError, match="GCP_PROJECT_ID"):
        GcpRegistry("", "l", "r", "p")


def test_gcp_fetch(tmp_path):
    runner = FakeGcloud({"nsh-denial-model": {"v1": {"bundle.tar.gz": "x"}, "v0": {}}})
    assert gcp(runner).fetch("v1", tmp_path).read_text() == "x"
    assert gcp(runner).list_versions() == ["v0", "v1"]
    with pytest.raises(RegistryError):
        gcp(runner).fetch("v0", tmp_path / "empty")


def test_gcp_missing_cli(tmp_path):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("gcloud")
    with pytest.raises(RegistryError, match="gcloud CLI not found"):
        gcp(missing).push(tmp_path / "b", "v1")


def test_make_registry_selects_backend(tmp_path):
    settings = registry_settings({"registry": {"local_dir": str(tmp_path)}})
    assert isinstance(make_registry(settings, env={}), LocalRegistry)
    reg = make_registry(settings, env={"REGISTRY_BACKEND": "gcp", "GCP_PROJECT_ID": "p", "GCP_REGION": "eu"})
    assert isinstance(reg, GcpRegistry) and reg.location == "eu" and reg.repository == "nsh-models"
    with pytest.raises(RegistryError):
        make_registry(settings, env={"REGISTRY_BACKEND": "s3"})


def test_append_history(tmp_path):
    path = tmp_path / "h" / "history.jsonl"
    append_history(path, {"a": 1})
    append_history(path, {"a": 2})
    assert [json.loads(line)["a"] for line in path.read_text().splitlines()] == [1, 2]
