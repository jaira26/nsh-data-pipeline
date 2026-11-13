"""Training-set stage: processed tables -> data/training/{train,validation,test}.parquet + dataset_card.json.

    python -m nsh_pipeline.modeling.build --processed-dir data/processed --raw-dir data/raw --params params.yaml
"""
import argparse
import json
import logging
from pathlib import Path

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling.features import build_training_frame, load_processed
from nsh_pipeline.modeling.split import dataset_card, split_by_time, write_training_set

logger = logging.getLogger(__name__)

MANIFEST_KEYS = ("dataset", "generator_version", "seed", "scale", "reference_date")


def source_info(raw_dir, params: dict) -> dict:
    """Where the training data came from: raw manifest facts plus the configured dataset release."""
    info = {}
    manifest_path = Path(raw_dir) / "manifest.json" if raw_dir else None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info.update({key: manifest[key] for key in MANIFEST_KEYS if key in manifest})
        info["injected_issues"] = len(manifest.get("injected_issues", []))  # 0 for clean training data
    dataset = params.get("dataset") or {}
    if dataset.get("url"):
        info["configured_release_url"] = dataset["url"]
        info["configured_release_sha256"] = dataset.get("sha256")
    return info


def build_training_set(processed_dir, params: dict, out_dir=None, raw_dir=None) -> dict:
    """Build features + label, split by date, write the splits and the dataset card; return the card."""
    settings = params["training_data"]
    out_dir = out_dir or settings["out_dir"]
    frame = build_training_frame(load_processed(processed_dir))
    splits = split_by_time(frame, settings["validation_start"], settings["test_start"])
    card = dataset_card(splits, settings, source_info(raw_dir, params))
    write_training_set(splits, out_dir, card)
    for name, info in card["splits"].items():
        logger.info("Wrote %s: %d rows, %.1f%% denied", name, info["rows"], 100 * info["positive_rate"])
    return card


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Build the model training set from the processed tables.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw", help="for manifest.json (data lineage)")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out-dir", default=None, help="default: params.yaml training_data.out_dir")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    card = build_training_set(args.processed_dir, load_params(args.params), args.out_dir, args.raw_dir)
    print(json.dumps(card["splits"], indent=2))
    return card


if __name__ == "__main__":  # pragma: no cover
    main()