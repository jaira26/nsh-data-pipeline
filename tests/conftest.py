from pathlib import Path

import pytest

from nsh_pipeline.utils.io import read_jsonl

SAMPLE_RAW_DIR = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture(scope="session")
def sample_raw_dir():
    return SAMPLE_RAW_DIR


@pytest.fixture(scope="session")
def sample_raw(sample_raw_dir):
    """All sample collections as {collection_name: list_of_documents}."""
    return {path.stem: read_jsonl(path) for path in sorted(sample_raw_dir.glob("*.jsonl"))}
