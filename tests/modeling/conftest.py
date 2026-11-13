from pathlib import Path

import pytest

TRAINING_SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "training"


@pytest.fixture(scope="session")
def training_sample_dir():
    """Folder with the sample train/validation/test splits and dataset_card.json."""
    return TRAINING_SAMPLE_DIR