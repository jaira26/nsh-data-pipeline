import hashlib
import json

from nsh_pipeline.constants import EXCLUDED_ORG_IDS
from nsh_pipeline.preprocess.people_current import current_value
from nsh_pipeline.utils.keys import make_nswid

EXPECTED_COLLECTIONS = {
    "applications", "people", "debit_card_status", "wex_claims", "tickets",
    "claims", "bills", "reserves", "nse_transaction_daily_logs",
}


def test_all_collections_present(sample_raw):
    assert set(sample_raw) == EXPECTED_COLLECTIONS


def test_manifest_matches_files(sample_raw_dir):
    manifest = json.loads((sample_raw_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        data = (sample_raw_dir / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], name
        assert len(data.splitlines()) == info["rows"], name


def test_excluded_orgs_are_present(sample_raw):
    assert EXCLUDED_ORG_IDS <= {a["_id"] for a in sample_raw["applications"]}


def test_foreign_keys_resolve(sample_raw):
    client_ids = {a["_id"] for a in sample_raw["applications"]}
    member_ids = {p["_id"] for p in sample_raw["people"]}
    ticket_ids = {t["ticketNumber"] for t in sample_raw["tickets"]}
    assert {p["Application"] for p in sample_raw["people"]} <= client_ids
    for name in ("wex_claims", "claims", "tickets", "debit_card_status", "nse_transaction_daily_logs"):
        assert {r["employeeId"] for r in sample_raw[name]} <= member_ids, name
    swipe_tickets = {s["ticketNumber"] for s in sample_raw["wex_claims"] if s["ticketNumber"]}
    assert swipe_tickets <= ticket_ids


def test_nswid_matches_current_name_and_ssn(sample_raw):
    for p in sample_raw["people"]:
        last = current_value(p["Names"], "LastName")
        ssn = current_value(p["PersonalData"], "Ssn")
        assert p["_base64Id"] == make_nswid(last, ssn)


def test_contract_quirks_are_present(sample_raw):
    swipe = sample_raw["wex_claims"][0]
    assert isinstance(swipe["_id"], str)
    assert "needsSubstantion" in swipe
    assert any("," in s["amount"] for s in sample_raw["wex_claims"])
    assert "pariticpant_status" in sample_raw["debit_card_status"][0]
    assert all(isinstance(r["amount"], int) for r in sample_raw["nse_transaction_daily_logs"])
    assert isinstance(sample_raw["bills"][0]["recurring"][0]["premium"], str)