import pytest

from nsh_pipeline.modeling import artifacts
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, load_split

META = {key: "x" for key in artifacts.METADATA_KEYS}


def test_save_requires_every_contract_key(tmp_path):
    with pytest.raises(ValueError, match="model_name"):
        artifacts.save_model({}, {k: v for k, v in META.items() if k != "model_name"}, tmp_path / "m")


def test_save_load_and_copy_round_trip(tmp_path):
    out = artifacts.save_model({"w": 1}, META, tmp_path / "a")
    (out / "stale.txt").write_text("old")
    artifacts.save_model({"w": 2}, META, tmp_path / "a")  # replaces the folder
    assert not (out / "stale.txt").exists()
    artifacts.copy_model(tmp_path / "a", tmp_path / "b")
    artifacts.copy_model(tmp_path / "a", tmp_path / "b")  # overwrite is fine
    model, metadata = artifacts.load_model(tmp_path / "b")
    assert model == {"w": 2} and metadata == META


def test_paths_follow_the_contract():
    assert str(artifacts.candidate_dir("lightgbm")) == "models/candidates/lightgbm"
    assert str(artifacts.best_dir()) == "models/best"


def test_stub_model_follows_the_contract(stub_model_dir, training_sample_dir):
    for folder in (stub_model_dir / "best", stub_model_dir / "candidates" / "logistic_regression"):
        model, metadata = artifacts.load_model(folder)
        assert set(artifacts.METADATA_KEYS) <= set(metadata)
        assert metadata["stub"] is True and metadata["data_version"] == "sample"
        test = load_split("test", training_sample_dir)
        assert model.predict_proba(test[FEATURE_COLUMNS]).shape == (len(test), 2)
