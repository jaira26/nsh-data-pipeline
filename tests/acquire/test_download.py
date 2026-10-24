import json
import tarfile

import pytest
import yaml

from nsh_pipeline.acquire.download import (
    DownloadError, acquire_data, download_dataset, download_file, main, sha256_of, verify_raw_dir,
)
from nsh_pipeline.acquire.publish import build_archive
from nsh_pipeline.acquire.publish import main as publish_main
from nsh_pipeline.generate.run import generate_data

PARAMS = {
    "seed": 42,
    "scale": "test",
    "reference_date": "2025-06-15",
    "scales": {"test": {"clients": 5, "max_members_per_client": 10}},
}


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A generated dataset packed like a GitHub Release asset, served from a file:// URL."""
    root = tmp_path_factory.mktemp("published")
    generate_data(PARAMS, root / "raw")
    archive = root / "dataset.tar.gz"
    digest = build_archive(root / "raw", archive)
    return {"raw": root / "raw", "archive": archive, "sha256": digest, "url": archive.as_uri()}


def read_manifest(raw_dir):
    return json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))


def test_build_archive_contains_every_file(published):
    with tarfile.open(published["archive"]) as tar:
        names = set(tar.getnames())
    assert {"manifest.json", "people.jsonl", "wex_claims.jsonl"} <= names
    assert published["sha256"] == sha256_of(published["archive"])


def test_download_verifies_and_unpacks(published, tmp_path):
    manifest = download_dataset(published["url"], published["sha256"], tmp_path / "raw", retries=1)
    assert (tmp_path / "raw" / "people.jsonl").exists()
    assert manifest == read_manifest(published["raw"])


def test_checksum_mismatch_is_rejected_before_unpacking(published, tmp_path):
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download_dataset(published["url"], "0" * 64, tmp_path / "raw", retries=1)
    assert not (tmp_path / "raw").exists()


def test_tampered_file_fails_verification(published, tmp_path):
    download_dataset(published["url"], published["sha256"], tmp_path, retries=1)
    with (tmp_path / "people.jsonl").open("a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(DownloadError, match="people.jsonl"):
        verify_raw_dir(tmp_path)


def test_missing_manifest_fails_verification(tmp_path):
    with pytest.raises(DownloadError, match="manifest"):
        verify_raw_dir(tmp_path)


def test_download_retries_then_fails(tmp_path, caplog):
    missing = (tmp_path / "nope.tar.gz").as_uri()
    with pytest.raises(DownloadError, match="after 2 attempts"):
        download_file(missing, tmp_path / "out", retries=2, backoff_seconds=0)
    assert caplog.text.count("Download attempt") == 2


def test_acquire_downloads_when_url_is_configured(published, tmp_path):
    params = {**PARAMS, "dataset": {"url": published["url"], "sha256": published["sha256"]}}
    assert acquire_data(params, tmp_path, retries=1, backoff_seconds=0)["source"] == "download"


def test_acquire_falls_back_to_identical_generated_data(published, tmp_path):
    params = {**PARAMS, "dataset": {"url": (tmp_path / "missing.tar.gz").as_uri(),
                                    "sha256": published["sha256"]}}
    result = acquire_data(params, tmp_path / "raw", retries=1, backoff_seconds=0)
    assert result["source"] == "generated"
    assert result["manifest"]["files"] == read_manifest(published["raw"])["files"]  # same SHA-256s


def test_acquire_without_url_generates(tmp_path):
    assert acquire_data(PARAMS, tmp_path)["source"] == "generated"


def test_cli_download_and_publish(published, tmp_path, capsys):
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(
        {**PARAMS, "dataset": {"url": published["url"], "sha256": published["sha256"]}}), encoding="utf-8")
    result = main(["--params", str(params_path), "--out-dir", str(tmp_path / "raw")])
    assert result["source"] == "download"
    digest = publish_main(["--raw-dir", str(tmp_path / "raw"), "--out", str(tmp_path / "again.tar.gz")])
    assert len(digest) == 64
    assert "SHA-256" in capsys.readouterr().out