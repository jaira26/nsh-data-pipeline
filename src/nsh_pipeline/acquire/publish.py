"""Package a raw dataset folder into a .tar.gz for a GitHub Release and print its SHA-256.

    python -m nsh_pipeline.acquire.publish --raw-dir data/raw --out dist/nsh-raw-small-v1.0.tar.gz
"""
import argparse
import tarfile
from pathlib import Path

from nsh_pipeline.acquire.download import sha256_of, verify_raw_dir


def build_archive(raw_dir, out_path) -> str:
    """Write every file in `raw_dir` into a .tar.gz at `out_path`; return the archive's SHA-256."""
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    verify_raw_dir(raw_dir)  # never publish a dataset that fails its own manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for path in sorted(raw_dir.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
    return sha256_of(out_path)


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description="Package a raw dataset for a GitHub Release.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="dist/nsh-raw-small-v1.0.tar.gz")
    args = parser.parse_args(argv)
    digest = build_archive(args.raw_dir, args.out)
    print(f"Archive: {args.out}")
    print(f"SHA-256: {digest}")
    print("Next: upload the archive to a GitHub Release, then set dataset.url and dataset.sha256 in params.yaml")
    return digest


if __name__ == "__main__":  # pragma: no cover
    main()