"""Generator entry point: params.yaml -> raw JSONL collections + manifest.json.

    python -m nsh_pipeline.generate.run --params params.yaml --out-dir data/raw
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

from nsh_pipeline.config import load_params, reference_date, scale_settings
from nsh_pipeline.generate.activity import make_claims, make_swipes, make_tickets
from nsh_pipeline.generate.cards import make_cards
from nsh_pipeline.generate.clients import make_clients
from nsh_pipeline.generate.common import make_generators
from nsh_pipeline.generate.finance import make_bills, make_ledger, make_reserves
from nsh_pipeline.generate.people import make_people
from nsh_pipeline.utils.io import write_jsonl

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "1.0"


def build_collections(params: dict) -> dict[str, list[dict]]:
    """Generate every raw collection in memory. Order matters: later collections build on earlier ones."""
    reference = reference_date(params)
    scale = scale_settings(params)
    rng, fake = make_generators(params["seed"])
    clients = make_clients(rng, fake, scale["clients"], reference)
    people = make_people(rng, fake, clients, reference, scale["max_members_per_client"])
    swipes = make_swipes(rng, fake, clients, people, reference)
    tickets = make_tickets(rng, people, swipes, reference)
    claims = make_claims(rng, clients, people, reference)
    cards = make_cards(rng, clients, people, swipes, reference)
    bills = make_bills(rng, clients, people, swipes, reference)
    reserves = make_reserves(rng, clients, bills, swipes)
    ledger = make_ledger(clients, people, swipes, claims)
    return {
        "applications": clients,
        "people": people,
        "debit_card_status": cards,
        "wex_claims": swipes,
        "tickets": tickets,
        "claims": claims,
        "bills": bills,
        "reserves": reserves,
        "nse_transaction_daily_logs": ledger,
    }


def generate_data(params: dict, out_dir) -> dict:
    """Write every collection to `out_dir/<name>.jsonl` plus `manifest.json`; return the manifest."""
    out_dir = Path(out_dir)
    manifest = {
        "dataset": "nsh-synthetic",
        "generator_version": GENERATOR_VERSION,
        "seed": params["seed"],
        "scale": params["scale"],
        "reference_date": str(reference_date(params)),
        "files": {},
    }
    for name, records in build_collections(params).items():
        path = write_jsonl(records, out_dir / f"{name}.jsonl")
        manifest["files"][path.name] = {
            "rows": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        logger.info("Wrote %s: %d rows", path.name, len(records))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Generate the synthetic NSH raw dataset.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = generate_data(load_params(args.params), args.out_dir)
    print(json.dumps({name: info["rows"] for name, info in manifest["files"].items()}, indent=2))
    return manifest


if __name__ == "__main__":  # pragma: no cover
    main()