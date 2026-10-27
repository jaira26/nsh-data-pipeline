# Anomaly report

- Processed data: `data/processed_injected`
- As of: 2025-06-15  |  mode: `report`
- Critical failures: **4**  |  warnings: **3**

| Check | Severity | Result | Count | Message |
|---|---|---|---|---|
| `duplicate_bill_lines` | critical | **FAIL** | 5 | 5 (member_id, billing_month) keys billed more than once (5 extra rows) |
| `money_unit_error` | critical | **FAIL** | 5 | 5 records have money above 100x the column median (cents read as dollars?) |
| `orphan_keys` | critical | **FAIL** | 5 | 5 records have a null or unresolved member_id / client_id |
| `excluded_org_leak` | critical | pass | 0 | NS Demo / NSAI appear only in clients, flagged is_excluded_org |
| `unparseable_money` | critical | **FAIL** | 5 | 5 records have missing money after parsing (unparseable or empty in the raw data) |
| `active_client_count` | warning | pass | 0 | 48 active clients, expected 42-60 for scale 'small' |
| `enum_violations` | warning | **FAIL** | 5 | 5 values outside the contract enums (1 fields) |
| `future_dates` | warning | **FAIL** | 8 | 8 records dated after the as-of date 2025-06-15 |
| `negative_amounts` | warning | pass | 0 | no negative paid, requested or billed amounts |
| `term_before_hire` | warning | pass | 0 | no termination date before a hire date |
| `distribution_drift` | warning | **FAIL** | 3 | 3 statistics drifted beyond the thresholds |

## `duplicate_bill_lines` (critical)

5 (member_id, billing_month) keys billed more than once (5 extra rows)

Sample records: `bill_lines:b00000000000000000000060|2025-03-01`, `bill_lines:b00000000000000000000327|2025-04-01`, `bill_lines:b00000000000000000000807|2025-03-01`, `bill_lines:b00000000000000000000938|2025-04-01`, `bill_lines:b00000000000000000000965|2025-02-01`

```
duplicate_keys: 5
extra_rows: 5
```

## `money_unit_error` (critical)

5 records have money above 100x the column median (cents read as dollars?)

Sample records: `swipes:WEX-001083`, `swipes:WEX-001599`, `swipes:WEX-002252`, `swipes:WEX-004010`, `swipes:WEX-004674`

```
swipes: {'amount_usd': {'median': 73.59, 'threshold': 7359.0, 'flagged': 5, 'max_multiple': 252.4}, 'paid_usd': {'median': 71.9, 'threshold': 7190.0, 'flagged': 5, 'max_multiple': 258.4}}
```

## `orphan_keys` (critical)

5 records have a null or unresolved member_id / client_id

Sample records: `swipes:WEX-000081`, `swipes:WEX-001280`, `swipes:WEX-001642`, `swipes:WEX-003083`, `swipes:WEX-003846`

```
swipes.member_id -> members: {'null': 0, 'orphan': 5}
```

## `unparseable_money` (critical)

5 records have missing money after parsing (unparseable or empty in the raw data)

Sample records: `swipes:WEX-002536`, `swipes:WEX-002572`, `swipes:WEX-004027`, `swipes:WEX-004554`, `swipes:WEX-004576`

```
swipes: {'paid_usd': 5}
raw wex_claims values: {"'N/A'": 5}
```

## `enum_violations` (warning)

5 values outside the contract enums (1 fields)

Sample records: `tickets:NSET-00051`, `tickets:NSET-00090`, `tickets:NSET-00124`, `tickets:NSET-00125`, `tickets:NSET-00181`

```
tickets.status: {'values': {'closed': 3, 'on hold': 1, 'waiting on member': 1}, 'casing_drift': ['closed', 'on hold', 'waiting on member']}
```

## `future_dates` (warning)

8 records dated after the as-of date 2025-06-15

Sample records: `swipes:WEX-000087`, `swipes:WEX-000568`, `swipes:WEX-001724`, `claims:CLM-100003`, `claims:CLM-100012`, `claims:CLM-100018`, `claims:CLM-100036`, `claims:CLM-100067`

```
swipes.service_date: {'count': 3, 'max': '2025-07-15T00:00:00+00:00'}
claims.created_at: {'count': 5, 'max': '2025-06-20T00:00:00+00:00'}
```

## `distribution_drift` (warning)

3 statistics drifted beyond the thresholds

Sample records: `swipes.denied_usd:mean`, `swipes.status:share of 'Approved'`, `swipes.status:share of 'Denied'`

```
findings: ['swipes.denied_usd mean changed by +78.44% (threshold 20%)', "swipes.status share of 'Approved' changed by -4.55 pts (threshold 2.0 pts)", "swipes.status share of 'Denied' changed by +4.55 pts (threshold 2.0 pts)"]
```
