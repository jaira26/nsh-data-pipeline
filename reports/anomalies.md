# Anomaly report

- Processed data: `data/processed`
- As of: 2025-06-15  |  mode: `strict`
- Critical failures: **0**  |  warnings: **1**

| Check | Severity | Result | Count | Message |
|---|---|---|---|---|
| `duplicate_bill_lines` | critical | pass | 0 | every (member_id, billing_month) appears once in bill_lines |
| `money_unit_error` | critical | pass | 0 | no money value above 100x its column median |
| `orphan_keys` | critical | pass | 0 | every member_id / client_id resolves to its parent table |
| `excluded_org_leak` | critical | pass | 0 | NS Demo / NSAI appear only in clients, flagged is_excluded_org |
| `unparseable_money` | critical | pass | 0 | every money value parsed to a number |
| `active_client_count` | warning | pass | 0 | 48 active clients, expected 42-60 for scale 'small' |
| `enum_violations` | warning | pass | 0 | every enum value is in the contract A5 lists |
| `future_dates` | warning | **FAIL** | 5 | 5 records dated after the as-of date 2025-06-15 |
| `negative_amounts` | warning | pass | 0 | no negative paid, requested or billed amounts |
| `term_before_hire` | warning | pass | 0 | no termination date before a hire date |
| `distribution_drift` | warning | pass | 0 | no drift beyond the thresholds |

## `future_dates` (warning)

5 records dated after the as-of date 2025-06-15

Sample records: `claims:CLM-100003`, `claims:CLM-100012`, `claims:CLM-100018`, `claims:CLM-100036`, `claims:CLM-100067`

```
claims.created_at: {'count': 5, 'max': '2025-06-20T00:00:00+00:00'}
```
