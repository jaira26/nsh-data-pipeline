import json

import pytest
import yaml

from nsh_pipeline.generate.run import generate_data
from nsh_pipeline.modeling.build import build_training_set, main, source_info
from nsh_pipeline.modeling.dataset import SPLITS, TRAINING_COLUMNS, load_dataset_card, load_split
from nsh_pipeline.preprocess.run import run_preprocess

AS_OF = "2025-06-15"


def make_params(out_dir):
    return {
        "seed": 42, "scale": "test", "reference_date": AS_OF,
        "scales": {"test": {"clients": 20, "max_members_per_client": 30}},
        "training_data": {"validation_start": "2025-05-01", "test_start": "2025-06-01", "out_dir": str(out_dir)},
    }


@pytest.fixture(scope="module")
def pipeline_dirs(tmp_path_factory):
    """Generated raw data and its processed tables, like the acquire + preprocess stages produce."""
    root = tmp_path_factory.mktemp("pipeline")
    generate_data(make_params(root / "training"), root / "raw")
    run_preprocess(root / "raw", root / "processed", AS_OF)
    return root


def test_build_writes_all_splits_and_the_card(pipeline_dirs, tmp_path):
    card = build_training_set(pipeline_dirs / "processed", make_params(tmp_path), raw_dir=pipeline_dirs / "raw")
    for name in SPLITS:
        part = load_split(name, tmp_path)
        assert list(part.columns) == TRAINING_COLUMNS
        assert len(part) == card["splits"][name]["rows"] > 0
    assert load_dataset_card(tmp_path) == card


def test_card_records_where_the_data_came_from(pipeline_dirs, tmp_path):
    card = build_training_set(pipeline_dirs / "processed", make_params(tmp_path), raw_dir=pipeline_dirs / "raw")
    source = card["source_data"]
    assert source["seed"] == 42 and source["scale"] == "test"
    assert source["dataset"] == "nsh-synthetic"
    assert source["injected_issues"] == 0


def test_source_info_without_a_manifest(tmp_path):
    params = {"dataset": {"url": "https://example.test/data.tar.gz", "sha256": "abc"}}
    assert source_info(tmp_path, params) == {
        "configured_release_url": "https://example.test/data.tar.gz", "configured_release_sha256": "abc"}
    assert source_info(None, {}) == {}


def test_cli_main(pipeline_dirs, tmp_path, capsys):
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(make_params(tmp_path / "training")), encoding="utf-8")
    card = main(["--processed-dir", str(pipeline_dirs / "processed"), "--raw-dir", str(pipeline_dirs / "raw"),
                 "--params", str(params_path)])
    assert (tmp_path / "training" / "train.parquet").exists()
    assert json.loads(capsys.readouterr().out)["train"]["rows"] == card["splits"]["train"]["rows"]