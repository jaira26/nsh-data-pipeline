# Decisions & assumptions

The synthetic setting means nobody at NSH can confirm these. Each one is a sensible default, documented here,
kept in one place in the code, and covered by tests, so it is easy to change.

| # | Decision | Why | Where in code |
|---|---|---|---|
| 1 | `claims`, `bill_lines`, `reserves` keep **medical rows only** (missing type kept) | Contract A0 rule 6; agreed columns have no `type`, so features could not filter later | `_is_medical` in `flatten.py` |
| 2 | **Bill lines are not deduplicated** | PMPM double-counting (A8 #1) must stay visible for validation to detect | `clean_bill_lines` |
| 3 | 4-bucket **employee status** rule: COBRA + active benefit → COBRA Enrolled; terminated → Other; active benefit → Enrolled; else Not Enrolled | A5 lists the buckets but not the rules | `_employee_status` |
| 4 | Ticket **SLA = 40 wall-clock hours** | A1.6 gives `SLA_HOURS = 40`, not the hour type | `TICKET_SLA_HOURS` |
| 5 | `people.Benefits[].Premium` is **dollars** | Unit not stated in A1.2 | `MEMBER_BENEFIT_MONEY_UNITS` |
| 6 | `employer_code` = `claimReferenceId` (client WEX ID) | A3 join map | `clean_clients` |
| 7 | `nswid` keeps punctuation in last names (e.g. `o'brie1234`) | A3 formula taken literally | `make_nswid` |
| 8 | Excluded orgs matched by **id or alias** (`demo`, `nsais`) | A0 rule 2 names both | `flag_excluded_orgs` |
| 9 | Date formats **declared per field**, never guessed; all UTC; date-only values = midnight UTC | A4; avoids 3/7 vs 7/3 errors and day shifts | `dates.py` |
| 10 | Group size buckets 1-9 / 10-24 / 25-49 / 50-99 / 100-249 / 250+ | Real `GROUP_SIZE_BINS` not in the package | `GROUP_SIZE_BINS` |
| 11 | No `cards` processed table | Not one of the 9 agreed tables (plan §6.2); add if a feature needs card/OOP data | — |
| 12 | Fixed **`reference_date`**, never "today" | Reproducible runs | `params.yaml` |
| 13 | **Faker pinned** to an exact version | Faker output can change between releases | `pyproject.toml` |
| 14 | Injected issues **off by default**, separate RNG (seed + 1) | Clean default pipeline; enabling changes only injected records | `generate/inject.py` |
| 15 | Download **falls back to local generation** | Pipeline never depends on the network; same seed → identical data | `acquire_data` |
| 16 | Bias "performance" per slice = **outcome rates**: denial rate, and substantiation rate among paid debit-card swipes. Average paid is shown, never flagged | No model exists yet; paid amounts depend on the plan, not on fair treatment | `bias/metrics.py` |
| 17 | A slice is **flagged** only if worse off than overall by > 10 pts or > 1.5×, has ≥ 30 swipes, and a one-sided Fisher exact test gives p < 0.01 ÷ number of slices tested (Bonferroni) | Without the correction, clean data from 30 seeds gave 15 chance flags (1 with it). Exact test because substantiation is rare (~2.5%). Lower-than-overall rates are no burden, so not flagged | `flag_slices`, `params.yaml → bias` |
| 18 | **Proxy check:** the flag with the most excess outcomes is the root; flags on other dimensions are re-tested without the root slice and reported as proxies if they vanish | On injected data, Run Out clients and COBRA members look worse only because they sit in the skewed state | `explain_proxies` |
| 19 | **Mitigation = reweighing** (Kamiran & Calders) on the root flag's dimension; weights written to `data/bias/swipe_weights.parquet`; rows and outcomes never changed | The course requires an actual mitigation step; reweighing keeps every row and is easy to drop | `bias/mitigation.py` |
| 20 | Bias join drops excluded orgs and orphan swipes (orphans counted in the report); a missing attribute becomes an `Unknown` slice; `clients.status` is sliced as `client_status` | Orphans have no member attributes; `status` clashes with `swipes.status` | `join_swipes` |
