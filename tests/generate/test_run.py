import hashlib
import json

import yaml

from nsh_pipeline.generate.run import generate_data, main
from nsh_pipeline.preprocess.run import run_preprocess

PARAMS = {
    "seed": 42,
    "scale": "test",
    "reference_date": "2025-06-15",
    "scales": {"test": {"clients": 8, "max_members_per_client": 20}},
}
EXPECTED_FILES = {
    f"{name}.jsonl" for name in (
        "applications", "people", "debit_card_status", "wex_claims", "tickets", "claims",
        "bills", "reserves", "nse_transaction_daily_logs",
    )
}


def test_generate_data_writes_collections_and_manifest(tmp_path):
    manifest = generate_data(PARAMS, tmp_path)
    assert set(manifest["files"]) == EXPECTED_FILES
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    for name, info in manifest["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == info["sha256"]


def test_same_params_give_identical_files(tmp_path):
    first = generate_data(PARAMS, tmp_path / "a")
    second = generate_data(PARAMS, tmp_path / "b")
    assert first["files"] == second["files"]  # same row counts and same SHA-256 for every file


def test_generated_data_runs_through_full_preprocessing(tmp_path):
    generate_data(PARAMS, tmp_path / "raw")
    counts = run_preprocess(tmp_path / "raw", tmp_path / "processed", "2025-06-15")
    assert all(rows > 0 for rows in counts.values()), counts


def test_cli_main(tmp_path, capsys):
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(PARAMS), encoding="utf-8")
    manifest = main(["--params", str(params_path), "--out-dir", str(tmp_path / "raw")])
    assert manifest["seed"] == 42
    assert '"people.jsonl"' in capsys.readouterr().out