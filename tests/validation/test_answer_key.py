from nsh_pipeline.validation.anomalies import CheckResult
from nsh_pipeline.validation.answer_key import detect_injected, expected_ids

BILLS = [{"_id": "f00000000000000000000021", "billingDate": "2025-03"}]
MANIFEST = {
    "dataset": "nsh-synthetic", "seed": 42,
    "injected_issues": [
        {"issue": "denial_skew", "collection": "wex_claims", "count": 2, "ids": ["WEX-1", "WEX-2"]},
        {"issue": "duplicate_bill_lines", "collection": "bills", "count": 1,
         "ids": ["f00000000000000000000021:b00000000000000000000060"]},
        {"issue": "cents_as_dollars", "collection": "wex_claims", "count": 2, "ids": ["WEX-3", "WEX-4"]},
        {"issue": "status_casing_drift", "collection": "tickets", "count": 1, "ids": ["NSET-00051"]},
        {"issue": "orphan_swipes", "collection": "wex_claims", "count": 1, "ids": ["WEX-5"]},
        {"issue": "brand_new_issue", "collection": "x", "count": 1, "ids": ["X-1"]},
    ],
}


def result(check, ids):
    return CheckResult(check, "critical", not ids, len(ids), "", ids)


def test_expected_ids_map_manifest_ids_to_check_record_ids():
    assert expected_ids("cents_as_dollars", ["WEX-3"], "swipes") == ["swipes:WEX-3"]
    assert expected_ids("duplicate_bill_lines", ["f00000000000000000000021:b1"], "bill_lines", BILLS) == [
        "bill_lines:b1|2025-03-01"]
    assert expected_ids("duplicate_bill_lines", ["nope:b1"], "bill_lines", BILLS) == [
        "bill_lines:b1|<unknown bill nope>"]


def test_detection_per_issue():
    results = [
        result("duplicate_bill_lines", ["bill_lines:b00000000000000000000060|2025-03-01"]),
        result("money_unit_error", ["swipes:WEX-3", "swipes:WEX-9", "ledger:T1"]),  # missed WEX-4, extra WEX-9
        result("enum_violations", ["tickets:NSET-00051"]),
        result("orphan_keys", []),
    ]
    report = detect_injected(MANIFEST, results, BILLS)
    issues = {i["issue"]: i for i in report["issues"]}
    assert issues["denial_skew"] == {"issue": "denial_skew", "collection": "wex_claims", "injected_count": 2,
                                     "status": "out_of_scope",
                                     "handled_by": "bias slicing (M5): reports/bias_report.md"}
    assert issues["duplicate_bill_lines"]["status"] == "detected"
    cents = issues["cents_as_dollars"]
    assert (cents["status"], cents["detected_count"], cents["missed_ids"]) == ("partial", 1, ["swipes:WEX-4"])
    assert (cents["other_flagged_in_table"], cents["other_flagged_sample"]) == (1, ["swipes:WEX-9"])
    assert issues["status_casing_drift"]["status"] == "detected"
    assert (issues["orphan_swipes"]["status"], issues["orphan_swipes"]["detected_count"]) == ("missed", 0)
    assert issues["brand_new_issue"]["status"] == "unmapped"
    assert report["all_in_scope_detected"] is False


def test_all_detected_when_every_in_scope_issue_is_found():
    manifest = {"injected_issues": MANIFEST["injected_issues"][:3]}
    results = [result("duplicate_bill_lines", ["bill_lines:b00000000000000000000060|2025-03-01"]),
               result("money_unit_error", ["swipes:WEX-3", "swipes:WEX-4"])]
    assert detect_injected(manifest, results, BILLS)["all_in_scope_detected"] is True


def test_missing_check_result_counts_as_missed():
    manifest = {"injected_issues": [MANIFEST["injected_issues"][2]]}
    report = detect_injected(manifest, [], BILLS)
    assert report["issues"][0]["status"] == "missed" and report["all_in_scope_detected"] is False
