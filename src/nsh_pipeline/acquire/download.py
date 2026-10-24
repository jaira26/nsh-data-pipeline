"""Data acquisition: download the published raw dataset, verify it, unpack it into data/raw/.

If no dataset URL is configured, or the download fails, the same dataset is generated locally
from the seed in params.yaml (identical files), so the pipeline never depends on the network.

    python -m nsh_pipeline.acquire.download --params params.yaml --out-dir data/raw
"""
import argparse
import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from nsh_pipeline.config import load_params
from nsh_pipeline.generate.run import generate_data

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """The dataset could not be downloaded or failed verification."""


def sha256_of(path) -> str:
    """SHA-256 of a file, read in chunks so large files don't fill memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest, retries: int = 3, timeout: int = 60, backoff_seconds: float = 2.0) -> Path:
    """Download `url` to `dest`, retrying with a growing pause; raise DownloadError if all attempts fail."""
    dest = Path(dest)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, dest.open("wb") as f:
                shutil.copyfileobj(response, f)
            return dest
        except OSError as error:  # URLError, HTTPError and timeouts are all OSErrors
            last_error = error
            logger.warning("Download attempt %d/%d failed: %s", attempt, retries, error)
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    raise DownloadError(f"Could not download {url} after {retries} attempts: {last_error}")


def verify_raw_dir(raw_dir) -> dict:
    """Check every file listed in manifest.json exists with the recorded SHA-256; return the manifest."""
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise DownloadError(f"No manifest.json in {raw_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        path = raw_dir / name
        if not path.exists() or sha256_of(path) != info["sha256"]:
            raise DownloadError(f"{name} is missing or does not match manifest.json")
    return manifest


def download_dataset(url: str, sha256: str, out_dir, retries: int = 3, backoff_seconds: float = 2.0) -> dict:
    """Download the dataset archive, check its checksum BEFORE unpacking, unpack, verify every file."""
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        archive = download_file(url, Path(tmp) / "dataset.tar.gz", retries=retries,
                                backoff_seconds=backoff_seconds)
        actual = sha256_of(archive)
        if actual != sha256:
            raise DownloadError(f"Checksum mismatch for {url}: expected {sha256}, got {actual}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_dir, filter="data")  # refuses paths that would escape out_dir
    return verify_raw_dir(out_dir)


def acquire_data(params: dict, out_dir, retries: int = 3, backoff_seconds: float = 2.0) -> dict:
    """Download the published dataset if configured; otherwise (or on failure) generate it locally."""
    dataset = params.get("dataset") or {}
    url = dataset.get("url")
    if url:
        try:
            manifest = download_dataset(url, dataset["sha256"], out_dir, retries, backoff_seconds)
            logger.info("Downloaded and verified dataset from %s", url)
            return {"source": "download", "manifest": manifest}
        except DownloadError as error:
            logger.warning("%s; generating the dataset locally instead", error)
    manifest = generate_data(params, out_dir)
    return {"source": "generated", "manifest": manifest}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Download (or generate) the raw NSH dataset into data/raw.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = acquire_data(load_params(args.params), args.out_dir)
    rows = {name: info["rows"] for name, info in result["manifest"]["files"].items()}
    print(json.dumps({"source": result["source"], "files": rows}, indent=2))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()