"""Preprocessing entry point: raw JSONL collections -> 9 processed Parquet tables (plan section 6.2).

Airflow and DVC both call `run_preprocess`. It also runs from the command line:
    python -m nsh_pipeline.preprocess.run --raw-dir data/raw --out-dir data/processed --as-of 2025-06-15
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from nsh_pipeline.preprocess import flatten
from nsh_pipeline.preprocess.privacy import assert_no_phi
from nsh_pipeline.utils.io import read_jsonl

logger = logging.getLogger(__name__)

RAW_COLLECTIONS = (
    "applications", "people", "wex_claims", "claims", "tickets", "bills", "reserves",
    "nse_transaction_daily_logs",
)


def load_raw(raw_dir) -> dict[str, list[dict]]:
    """Read every raw collection the cleaners need; fail clearly if one is missing."""
    raw_dir = Path(raw_dir)
    raw = {}
    for name in RAW_COLLECTIONS:
        path = raw_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw collection {name!r}: expected {path}")
        raw[name] = read_jsonl(path)
    return raw


def build_tables(raw: dict[str, list[dict]], as_of) -> dict[str, pd.DataFrame]:
    """Run all cleaners. Excluded orgs come from the clients table so every table agrees."""
    clients = flatten.clean_clients(raw["applications"])
    excluded = frozenset(clients.loc[clients["is_excluded_org"], "client_id"])
    return {
        "clients": clients,
        "members": flatten.clean_members(raw["people"], as_of, excluded),
        "member_benefits": flatten.clean_member_benefits(raw["people"], excluded),
        "swipes": flatten.clean_swipes(raw["wex_claims"], excluded),
        "claims": flatten.clean_claims(raw["claims"], excluded),
        "tickets": flatten.clean_tickets(raw["tickets"], as_of, excluded),
        "bill_lines": flatten.clean_bill_lines(raw["bills"], excluded),
        "reserves": flatten.clean_reserves(raw["reserves"], excluded),
        "ledger": flatten.clean_ledger(raw["nse_transaction_daily_logs"], clients, excluded),
    }


def run_preprocess(raw_dir, out_dir, as_of=None) -> dict[str, int]:
    """Clean `raw_dir` into Parquet tables in `out_dir`; return row counts per table.

    `as_of` (date for age bands, employee status and SLA) defaults to today in UTC.
    """
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").normalize()
    tables = build_tables(load_raw(raw_dir), as_of)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in tables.items():
        assert_no_phi(df, name)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(df)
        logger.info("Wrote %s: %d rows", name, len(df))
    return counts


def main(argv=None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description="Clean raw NSH collections into processed Parquet tables.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--as-of", default=None,
                        help="Reference date (YYYY-MM-DD) for age bands, status and SLA; default today (UTC)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    counts = run_preprocess(args.raw_dir, args.out_dir, args.as_of)
    print(json.dumps(counts, indent=2))
    return counts


if __name__ == "__main__":  # pragma: no cover
    main()