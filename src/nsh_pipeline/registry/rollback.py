"""Manual rollback: point production back at an earlier registry version.

    python -m nsh_pipeline.registry.rollback --to v20261114-153617-2f68d25
    python -m nsh_pipeline.registry.rollback --previous        # the version before the current one

The automatic rollback (a worse candidate is never promoted) lives in promote.py; this command is for a
model that passed every gate but misbehaves after release. Nothing is deleted: the pointer moves, and the
move is recorded in registry/history.jsonl (and, on GCP, as a new production-pointer version).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from nsh_pipeline.config import load_params
from nsh_pipeline.modeling.artifacts import utc_now
from nsh_pipeline.registry.backends import HISTORY_FILE, RegistryError, append_history, make_registry
from nsh_pipeline.registry.bundle import extract_bundle, read_manifest
from nsh_pipeline.registry.decide import registry_settings
from nsh_pipeline.registry.promote import write_json
from nsh_pipeline.utils.alerts import send_alert

logger = logging.getLogger(__name__)


def run_rollback(params: dict, to_version: str | None = None, *, previous: bool = False, reason: str = "",
                 registry=None, env: dict | None = None, alert=send_alert) -> dict:
    settings = registry_settings(params)
    env = dict(os.environ if env is None else env)
    registry = registry or make_registry(settings, env)
    current = registry.get_production()
    if previous:
        if not current or not current.get("previous_version"):
            raise RegistryError("no previous production version to roll back to")
        to_version = current["previous_version"]
    if not to_version:
        raise RegistryError("give --to <version> or --previous")
    if current and current["version"] == to_version:
        raise RegistryError(f"{to_version} is already production")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = read_manifest(extract_bundle(registry.fetch(to_version, tmp), Path(tmp) / "bundle"))
    record = {**manifest, "promoted_at": utc_now(), "previous_version": current["version"] if current else None,
              "rolled_back_from": current["version"] if current else None, "uri": registry.uri(to_version),
              "rollback_reason": reason or "manual rollback"}
    registry.set_production(record)
    write_json(record, settings["production_file"])
    append_history(Path(settings["local_dir"]) / HISTORY_FILE,
                   {"at": record["promoted_at"], "event": "manual_rollback", "version": to_version,
                    "production_version": to_version, "backend": registry.name, "reason": record["rollback_reason"]})
    if settings["alerts"]:
        alert(f"NSH model rolled back to {to_version}",
              f"Production moved from {record['rolled_back_from']} to {to_version}.\nReason: "
              f"{record['rollback_reason']}", env=env)
    logger.info("Production is now %s (was %s)", to_version, record["rolled_back_from"])
    return record


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Point production back at an earlier registry version.")
    parser.add_argument("--params", default="params.yaml")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--to", dest="to_version")
    target.add_argument("--previous", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    record = run_rollback(load_params(args.params), args.to_version, previous=args.previous, reason=args.reason)
    print(json.dumps({k: record.get(k) for k in ("version", "rolled_back_from", "uri")}, indent=2))
    return record


if __name__ == "__main__":  # pragma: no cover
    main()
