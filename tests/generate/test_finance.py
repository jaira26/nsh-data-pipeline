import pytest

from nsh_pipeline.preprocess.flatten import clean_bill_lines, clean_clients, clean_ledger, clean_reserves


def _num(money):
    return float(money.lstrip("$").replace(",", ""))


def test_bills_are_one_per_client_month_with_string_money(world):
    keys = [(b["appId"], b["billingDate"]) for b in world["bills"]]
    assert len(set(keys)) == len(keys)
    for bill in world["bills"]:
        assert len(bill["billingDate"]) == 7  # "YYYY-MM"
        assert all(isinstance(line["premium"], str) for line in bill["recurring"])


def test_bill_lines_belong_to_the_billed_client(world):
    member_client = {p["_id"]: p["Application"] for p in world["people"]}
    for bill in world["bills"]:
        assert all(member_client[line["employeeId"]] == bill["appId"] for line in bill["recurring"])


def test_only_the_reference_month_is_open(world):
    for bill in world["bills"]:
        assert bill["closed"] == (bill["billingDate"] != "2025-06")


def test_dental_lines_exist_and_are_filtered_out(world):
    all_lines = [line for b in world["bills"] for line in b["recurring"]]
    dental = sum(line["type"] == "dental" for line in all_lines)
    assert dental > 0
    assert len(clean_bill_lines(world["bills"])) <= len(all_lines) - dental


def test_reserve_balances_roll_forward(world):
    for r in world["reserves"]:
        expected = r["previousBalance"] + r["reservesBilled"] - r["claimsPaid"] - r["returnOfReserve"]
        assert r["balance"] == pytest.approx(expected, abs=0.01)


def test_reserves_billed_matches_bill_reserve_charges(world):
    bills = {b["_id"]: b for b in world["bills"]}
    for r in world["reserves"]:
        lines = bills[r["bill"]]["recurring"]
        expected = sum(_num(line["reserveCharge"]) for line in lines if line["type"] == "medical")
        assert r["reservesBilled"] == pytest.approx(expected, abs=0.01)


def test_ledger_is_in_cents_and_matches_swipe_payments(world):
    ledger = world["nse_transaction_daily_logs"]
    assert all(isinstance(row["amount"], int) for row in ledger)
    wex_cents = sum(row["amount"] for row in ledger if row["nseSource"] == "wex")
    paid = sum(_num(s["paid_amt"]) for s in world["wex_claims"])
    assert wex_cents == pytest.approx(paid * 100, abs=len(ledger))  # at most 1 cent rounding per row


def test_finance_runs_through_preprocessing(world):
    clients = clean_clients(world["applications"])
    ledger = clean_ledger(world["nse_transaction_daily_logs"], clients)
    reserves = clean_reserves(world["reserves"])
    assert ledger["client_id"].notna().all()  # every employerCode resolves to a client
    assert len(reserves) > 0 and reserves["month"].notna().all()