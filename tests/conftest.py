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


TRAINING_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "training"


@pytest.fixture(scope="session")
def stub_model_dir(tmp_path_factory):
    """models/ folder with the contract-shaped stub model (candidates/logistic_regression + best).

    For validation, bias and registry tests: load it with
    nsh_pipeline.modeling.artifacts.load_model(stub_model_dir / "best").
    """
    from nsh_pipeline.modeling.train import train_stub

    models_dir = tmp_path_factory.mktemp("stub_models")
    train_stub(TRAINING_FIXTURE_DIR, models_dir)
    return models_dir
