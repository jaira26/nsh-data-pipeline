from datetime import date

import pandas as pd

from nsh_pipeline.generate.activity import CLAIM_STATUSES, HIGH_DOLLAR, TICKET_STATUSES
from nsh_pipeline.preprocess.flatten import clean_claims, clean_swipes, clean_tickets
from nsh_pipeline.preprocess.money import parse_money_str

REF = date(2025, 6, 15)


def test_same_seed_gives_identical_activity(build):
    assert build(seed=3) == build(seed=3)


def test_swipe_keys_resolve(world):
    clients = {c["_id"]: c for c in world["applications"]}
    people = {p["_id"]: p for p in world["people"]}
    for swipe in world["wex_claims"]:
        assert swipe["employeeId"] in people
        assert people[swipe["employeeId"]]["Application"] == swipe["clientId"]
        assert swipe["alias"] == clients[swipe["clientId"]]["employer"]["claimReferenceId"]
        assert swipe["nswid"] == people[swipe["employeeId"]]["_base64Id"]


def test_swipe_money_is_stored_as_strings_and_consistent(world):
    for swipe in world["wex_claims"]:
        amount, paid, denied = (parse_money_str(swipe[k]) for k in ("amount", "paid_amt", "denied_amt"))
        assert isinstance(swipe["amount"], str)
        if swipe["claim_status"] == "Approved":
            assert paid == amount and denied == 0
        if swipe["claim_status"] == "Denied":
            assert paid == 0 and denied == amount


def test_swipe_dates_fall_in_this_plan_year(world):
    dates = pd.to_datetime([s["date_of_service"] for s in world["wex_claims"]], utc=True)
    assert dates.min() >= pd.Timestamp(f"{REF.year}-01-01", tz="UTC")
    assert dates.max() <= pd.Timestamp(REF, tz="UTC")


def test_every_substantiation_swipe_has_its_ticket(world):
    tickets = {t["ticketNumber"]: t for t in world["tickets"]}
    for swipe in world["wex_claims"]:
        assert swipe["needsSubstantion"] == (swipe["ticketNumber"] is not None)
        if swipe["ticketNumber"]:
            assert tickets[swipe["ticketNumber"]]["category"] == "Substantiation"


def test_substantiation_rate_is_realistic(world):
    card_swipes = [s for s in world["wex_claims"] if s["claim_source_desc"] == "Debit card"]
    rate = 100 * sum(s["needsSubstantion"] for s in card_swipes) / len(card_swipes)
    assert 0.5 <= rate <= 6.0  # NSH cohort ~2.2%, industry benchmark 5-15%


def test_tickets_are_unique_and_consistent(world):
    numbers = [t["ticketNumber"] for t in world["tickets"]]
    assert len(set(numbers)) == len(numbers)
    for ticket in world["tickets"]:
        assert ticket["status"] in TICKET_STATUSES
        assert (ticket["closedAt"] is not None) == (ticket["status"] == "Closed")


def test_claims_are_consistent(world):
    numbers = [c["claimNumber"] for c in world["claims"]]
    assert len(numbers) > 0
    assert len(set(numbers)) == len(numbers)
    for claim in world["claims"]:
        assert claim["status"] in CLAIM_STATUSES + ["Approved - High Dollar"]
        if claim["amount"] > HIGH_DOLLAR:
            assert claim["escalation"] == "High Dollar"
        if claim["status"] == "Denied":
            assert claim["amountDenied"] == claim["amount"]


def test_cards_have_multiple_records_and_valid_balances(world):
    cards = world["debit_card_status"]
    people = {p["_id"] for p in world["people"]}
    per_member = pd.Series([c["employeeId"] for c in cards]).value_counts()
    assert per_member.max() == 2  # prior-year + current-year card
    for card in cards:
        assert card["employeeId"] in people
        assert 0 <= card["remaining_balance"] <= card["max_oop"]
        assert "pariticpant_status" in card
        assert card["valid_start_dt"].count("/") == 2  # M/D/YYYY string


def test_generated_activity_runs_through_preprocessing(world):
    swipes = clean_swipes(world["wex_claims"])
    tickets = clean_tickets(world["tickets"], pd.Timestamp(REF, tz="UTC"))
    claims = clean_claims(world["claims"])
    assert len(swipes) > 0 and swipes["amount_usd"].notna().all()
    assert len(tickets) > 0 and tickets["sla_breached"].any()
    assert len(claims) > 0