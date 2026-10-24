# Data acquisition & synthetic data generation (M1)

## Why synthetic data
The NSH vendor package gives no access to real data (scoping plan §11): every stage is built and tested on a
**referentially consistent synthetic dataset** shaped exactly like NSH's MongoDB collections
(`01-data-contract-package.md`). This module generates that dataset and publishes it for download.

## Quick start
Download the published dataset (release `data-v1.0`), verify it, and unpack it into `data/raw/`:

```bash
python -m nsh_pipeline.acquire.download --params params.yaml --out-dir data/raw
```

If the download fails, or `dataset.url` is empty, the same dataset is **generated locally from the seed**
(identical files), so the pipeline never depends on the network. To generate directly:

```bash
python -m nsh_pipeline.generate.run --params params.yaml --out-dir data/raw
```

## Output: `data/raw/`
One JSON Lines file per collection (one document per line), field names exactly as in the data contract,
quirks included, plus `manifest.json` (seed, scale, reference date, row count and SHA-256 per file, injected issues).

| File | One record is | Rows at `small` scale |
|---|---|---|
| `applications.jsonl` | a client employer (+ NS Demo, NSAI) | 62 |
| `people.jsonl` | a member, with history arrays | 1,377 |
| `debit_card_status.jsonl` | a WEX card record (several per member) | 1,714 |
| `wex_claims.jsonl` | a debit-card swipe | 4,845 |
| `tickets.jsonl` | a support / substantiation ticket | 183 |
| `claims.jsonl` | a manual reimbursement claim | 83 |
| `bills.jsonl` | a client's monthly invoice | 372 |
| `reserves.jsonl` | a client's monthly reserve balance | 372 |
| `nse_transaction_daily_logs.jsonl` | a ledger transaction (cents) | 4,435 |

## `params.yaml`
| Key | Meaning |
|---|---|
| `seed` | Random seed; same seed → byte-identical files |
| `scale` | `small` (CI, grading) or `full` (~605 clients, the real book size) |
| `reference_date` | Fixed "as of" date for generation and preprocessing (no "today" anywhere) |
| `inject_issues` | Known data problems for M3/M5 to detect (off by default) |
| `dataset.url`, `dataset.sha256` | Published dataset to download, and its checksum |

## How the generator works
Collections are built in dependency order, each from the ones before it, so every key resolves:

```
clients → members → swipes → tickets, claims, cards → bills → reserves → ledger
```

| Behaviour | Rate | Source |
|---|---|---|
| Group sizes | lognormal: many small, few large | realistic book |
| Client status | 85% Active, 8% Terminated, ... | contract A5 enums |
| Name change (history entry) | 5% of members | contract A0 rule 4 |
| Terminated / on COBRA | 8% / 35% of terminated | census buckets |
| Waived medical coverage | 10% | "Not Enrolled" bucket |
| VIP | 0.1% | "~99.9% non-VIP" (scoping plan §11) |
| Swipes needing substantiation | ~2.5% of paid card swipes | NSH cohort ~2.2% (glossary) |
| Denied swipes (POS decline) | 5% | contract A1.3 |

**Contract quirks reproduced:** money as cents, numeric strings (`"1,250.75"`, `"$35.00"`) and dollars;
`M/D/YYYY` string dates; history arrays with `Deleted`/`Active` flags; string `_id` on swipes; misspelled fields
(`needsSubstantion`, `pariticpant_status`); multiple card records per member; lowercase vs capitalised status;
the two non-client orgs.

**Internal consistency (tested):** reserves roll forward (previous + billed − paid); reserves billed = bill reserve
charges; ledger cents = swipe payments; every ledger `employerCode` maps to a client.

## Injected data issues (for validation and bias evidence)
Set `inject_issues.enabled: true` (in a copy of `params.yaml`). A separate RNG (seed + 1) is used, so only the
injected records change. Every injected record id is listed in `manifest.json → injected_issues`: the answer key.

| Issue | What's planted | Detected by |
|---|---|---|
| `duplicate_bill_lines` | member billed twice in a month (PMPM double-counting) | M3 |
| `cents_as_dollars` | swipe amounts 100× too large | M3 |
| `orphan_swipes` | swipe for a non-existent member | M3 |
| `unparseable_money` | `"N/A"` in `paid_amt` | M3 |
| `future_service_dates` | service date after `reference_date` | M3 |
| `status_casing_drift` | ticket status `"closed"` | M3 |
| `denial_skew` | 30% of one state's approved swipes denied | M5 (bias) |

## Download stage
`acquire_data(params, out_dir)` downloads with retries, growing backoff and a timeout; checks the archive
SHA-256 **before** unpacking; extracts safely (`filter="data"`, no path escapes); verifies every file against
`manifest.json`; falls back to local generation on any failure.

## Publishing a new dataset version
```bash
python -m nsh_pipeline.generate.run --params params.yaml --out-dir <tmp>/raw
python -m nsh_pipeline.acquire.publish --raw-dir <tmp>/raw --out <tmp>/nsh-raw-small-vX.Y.tar.gz
```
Create a GitHub Release (tag `data-vX.Y`), attach the archive, then put its download URL and the printed
SHA-256 into `params.yaml → dataset`.

## Code and tests
| Code | Tests |
|---|---|
| `src/nsh_pipeline/generate/` (`clients`, `people`, `activity`, `cards`, `finance`, `inject`, `run`) | `tests/generate/` |
| `src/nsh_pipeline/acquire/` (`download`, `publish`) | `tests/acquire/` |
| `src/nsh_pipeline/config.py` | `tests/test_config.py` |