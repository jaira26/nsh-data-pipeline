# Preprocessing (M2)

## Quick start
```bash
python -m nsh_pipeline.preprocess.run --raw-dir data/raw --out-dir data/processed --as-of 2025-06-15
```
From Python (what Airflow / DVC call): `run_preprocess(raw_dir, out_dir, as_of)` → row counts per table.

## What it does
```
data/raw/*.jsonl → load 8 collections → 9 cleaners → assert no PHI → data/processed/*.parquet
```

| Problem in the raw data (data contract) | Handled by |
|---|---|
| Money in cents / numeric strings / dollars (A4: "silent 100× error") | `money.py`: unit declared per field, NaN-safe parsing |
| Member data stored as history arrays (A0 rule 4) | `people_current.py`: last entry not `Deleted` / not inactive |
| Dates as ISO, `M/D/YYYY`, `YYYY-MM` (A4) | `dates.py`: format declared per field, all UTC |
| NS Demo / NSAI are not clients (A0 rule 2) | `exclusions.py`: flagged in `clients`, dropped elsewhere (by id **or** alias) |
| SSN, birthdate, addresses (scoping plan §9.2) | `privacy.py`: masked SSN, age band, raw PHI dropped, `assert_no_phi` before writing |
| Member join key `nswid` (A3) | `utils/keys.py`: one shared `make_nswid` used by generator and preprocessing |

## Output: `data/processed/` (9 Parquet tables)
| Table | Grain |
|---|---|
| `clients` | 1 row / client (excluded orgs kept, `is_excluded_org = True`) |
| `members` | 1 row / member, with 4-bucket `employee_status` |
| `member_benefits` | 1 row / non-deleted enrollment |
| `swipes` | 1 row / debit-card swipe |
| `claims` | 1 row / manual medical claim |
| `tickets` | 1 row / ticket, with `sla_breached` |
| `bill_lines` | 1 row / member / billing month (medical) |
| `reserves` | 1 row / client / month (medical) |
| `ledger` | 1 row / transaction (`client_id` looked up from `employerCode`) |

Exact column lists: `CLIENT_COLUMNS`, `MEMBER_COLUMNS`, … in `src/nsh_pipeline/preprocess/flatten.py`.

**Conventions:** snake_case columns; money columns end in `_usd` (float dollars); dates are UTC timestamps;
IDs are strings; no raw SSN, full birthdate or address in any output.

## The `as_of` date
Age bands, employee status and SLA breaches depend on "as of when?". It's a parameter (from
`params.yaml → reference_date`), so the same input always gives the same output.

## Code and tests
| Code | Tests |
|---|---|
| `src/nsh_pipeline/preprocess/` (`money`, `dates`, `people_current`, `exclusions`, `privacy`, `flatten`, `run`) | `tests/preprocess/` |
| Sample dataset (all quirks, hand-built) | `tests/fixtures/` + `tests/test_sample_fixture.py` |

Run: `pytest --cov=nsh_pipeline`. All modules are at 100% line coverage.