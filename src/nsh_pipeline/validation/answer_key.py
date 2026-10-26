"""Compare what the anomaly checks flagged with the injected issues in manifest.json (the answer key).

The generator lists every injected record under `manifest.json -> injected_issues`. Each issue
maps to the check expected to catch it and to that check's record ids:

  manifest issue          manifest id              check                 record id
  duplicate_bill_lines    "<bill _id>:<employeeId>" duplicate_bill_lines  "bill_lines:<member>|<YYYY-MM-01>"
                          (bill _id -> billingDate via raw bills.jsonl)
  cents_as_dollars        wex_claims _id           money_unit_error      "swipes:<swipe_id>"
  orphan_swipes           wex_claims _id           orphan_keys           "swipes:<swipe_id>"
  unparseable_money       wex_claims _id           unparseable_money     "swipes:<swipe_id>"
  future_service_dates    wex_claims _id           future_dates          "swipes:<swipe_id>"
  status_casing_drift     tickets ticketNumber     enum_violations       "tickets:<ticket_number>"
  denial_skew             -                        out of scope: bias slicing (M5)
"""
from nsh_pipeline.validation.anomalies import CheckResult

ISSUE_CHECKS: dict[str, tuple[str, str]] = {
    "duplicate_bill_lines": ("duplicate_bill_lines", "bill_lines"),
    "cents_as_dollars": ("money_unit_error", "swipes"),
    "orphan_swipes": ("orphan_keys", "swipes"),
    "unparseable_money": ("unparseable_money", "swipes"),
    "future_service_dates": ("future_dates", "swipes"),
    "status_casing_drift": ("enum_violations", "tickets"),
}
OUT_OF_SCOPE: dict[str, str] = {"denial_skew": "bias slicing (M5): reports/bias_report.md"}


def expected_ids(issue: str, ids: list[str], table: str, bills: list[dict] | None = None) -> list[str]:
    """Translate manifest ids into the record ids the matching check reports."""
    if issue != "duplicate_bill_lines":
        return [f"{table}:{i}" for i in ids]
    months = {b.get("_id"): b.get("billingDate") for b in bills or []}
    out = []
    for item in ids:
        bill_id, _, member_id = item.partition(":")
        month = months.get(bill_id)
        day = f"{month}-01" if month else f"<unknown bill {bill_id}>"
        out.append(f"bill_lines:{member_id}|{day}")
    return out


def detect_injected(manifest: dict, results: list[CheckResult], bills: list[dict] | None = None) -> dict:
    """Per injected issue: injected count, detected count, missed ids and other ids the check flagged."""
    by_check = {r.check: r for r in results}
    issues = []
    for injected in manifest.get("injected_issues") or []:
        name, ids = injected["issue"], list(injected.get("ids") or [])
        row = {"issue": name, "collection": injected.get("collection"), "injected_count": len(ids)}
        if name in OUT_OF_SCOPE:
            issues.append({**row, "status": "out_of_scope", "handled_by": OUT_OF_SCOPE[name]})
            continue
        if name not in ISSUE_CHECKS:
            issues.append({**row, "status": "unmapped", "detected_count": 0, "missed_ids": ids})
            continue
        check, table = ISSUE_CHECKS[name]
        result = by_check.get(check)
        flagged = set(result.record_ids) if result else set()
        expected = expected_ids(name, ids, table, bills)
        missed = [i for i in expected if i not in flagged]
        detected = len(expected) - len(missed)
        others = sorted(i for i in flagged if i.startswith(f"{table}:") and i not in set(expected))
        status = "detected" if not missed else ("missed" if detected == 0 else "partial")
        issues.append({
            **row, "check": check, "status": status, "detected_count": detected,
            "missed_ids": missed, "other_flagged_in_table": len(others), "other_flagged_sample": others[:10],
        })
    in_scope = [i for i in issues if i["status"] != "out_of_scope"]
    return {
        "dataset": manifest.get("dataset"),
        "seed": manifest.get("seed"),
        "all_in_scope_detected": bool(in_scope) and all(i["status"] == "detected" for i in in_scope),
        "issues": issues,
    }
