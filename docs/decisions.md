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