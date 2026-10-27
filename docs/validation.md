# Validation, statistics, drift & anomaly detection (M3)

## Quick start
```bash
python -m nsh_pipeline.validation.run --step all          # raw, schema, stats, anomalies (after preprocessing)
pytest tests/validation tests/utils/test_alerts.py         # tests only, no data or network needed
```
On the default clean dataset the result is **0 critical; 1 warning check (5 records)**. The warning is a real
generator bug, explained under [Clean run](#clean-run-0-critical-1-warning-check-5-records).

## Where it runs
| Runner | What it calls |
|---|---|
| Airflow | tasks `validate_processed`, `stats`, `anomalies` (and `validate_raw` once `feat/orchestration-validate-raw` merges) run `python -m nsh_pipeline.validation.run --processed-dir data/processed --params params.yaml --step <step>`; commands are registered in `orchestration/stages.py` |
| DVC | the `validate` stage in `dvc.yaml` runs `--step all` |
| Python | `run_raw`, `run_schema`, `run_stats`, `run_anomalies`, `run_step` in `nsh_pipeline.validation.run` |

Only `--step` is required. Every other option has a default:

| Option | Default | Notes |
|---|---|---|
| `--step` | `all` | `raw`, `schema`, `stats`, `anomalies` or `all` |
| `--params` | `params.yaml` | thresholds live under `validation:` |
| `--raw-dir` / `--processed-dir` / `--features-dir` | `data/raw` / `data/processed` / `data/features` | |
| `--report-dir` | `reports` | |
| `--as-of` | `params.yaml → reference_date` | used by the future-date check |
| `--update-baseline` | off | `stats` step: replace the drift baseline |

## Run each step alone
| Step | Command | Checks | Writes (under `--report-dir`) |
|---|---|---|---|
| raw | `--step raw` | each raw file exists and matches `manifest.json` (rows, SHA-256); light raw schemas | `validation/raw_schema.json` |
| schema | `--step schema` | strict schemas on the 9 processed tables; feature schemas if `data/features/` exists (otherwise `skipped`) | `validation/processed_schema.json`, `validation/features_schema.json` |
| stats | `--step stats` | per-table statistics, drift vs `reports/baseline/` | `stats/<table>.json`, `drift.json` |
| anomalies | `--step anomalies` | the 11 checks below; answer-key comparison when the manifest lists injected issues | `anomalies.json`, `anomalies.md`, `validation/anomalies_summary.json`, `injection_detection.json` |

`anomalies.json` is a top-level list of `{check, severity, status, passed, count, sample_record_ids, message, details}`
records: `orchestration/summary.py` sums `count` by severity into `run_summary.json`.

## Failure mode and alerts
`validation.mode` in `params.yaml`:
* **`strict`** (default): a raw or schema failure, or any critical anomaly, exits with code 1, so the Airflow task fails.
* **`report`**: every report and alert is still produced, but the exit code is 0. Injected-issue runs use this
  (`params.injected.yaml`); in strict mode the DAG would stop before bias slicing and DVC versioning.

Reports are written before a strict failure is raised, so they always describe the failing run.

**Alerts** (`nsh_pipeline.utils.alerts.send_alert(title, body, details=None)`) go to a Slack incoming webhook and/or
email over SMTP. Credentials come only from environment variables (`.env`, see `.env.example`): `SLACK_WEBHOOK_URL`,
`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM`. With none set, the alert is
logged as a warning and the pipeline carries on. Sending never raises.

**One alert per failure, not two.** Under Airflow, a failed task triggers the DAG's `on_failure_callback`
(`orchestration/summary.notify_failure`), which alerts with the exception text. That text ends with the stage's last
20 output lines. So when M3 runs inside Airflow (`AIRFLOW_CTX_DAG_ID` is set) in strict mode, it does not alert itself.
It raises one detailed message instead (failed checks, messages and sample record ids, about 11 lines), and that
message reaches the callback's alert. M3 sends the alert itself everywhere else: from the CLI, from `dvc repro`, and in
report mode, where no task fails and the callback never fires. `utils.alerts.on_failure_callback(context)` is also
provided for any DAG that wants M3's alert directly.

## Schemas (pandera 0.34.1)
| Layer | Built from | Enforces |
|---|---|---|
| raw (light) | `schemas/raw.py` | fields preprocessing reads are present; keys (`_id`, `claimNumber`, `ticketNumber`) unique text; money and dates stored as contract A4 says (swipe money as strings, ledger as integer cents, card dates `M/D/YYYY`); the misspelled fields `needsSubstantion` and `pariticpant_status` present, and their correct spellings **absent** (an upstream "fix" would make preprocessing silently read nothing) |
| processed (strict) | `schemas/processed.py`, columns and order from the `*_COLUMNS` lists in `preprocess/flatten.py` | exactly the agreed columns, in order (an extra column such as a leaked `ssn` fails); money `*_usd` = float dollars; dates = timezone-aware UTC; primary keys unique and not null; id formats (`CLM-1xxxxx`, `NSET-xxxxx`, 24-hex ids, `***-**-1234`); ranges (renewal month 1-12, group size ≥ 1); enums derived by our own preprocessing |
| features | `schemas/features.py`, plan section 6.3 | one row per key; `mlr ≥ 0`, `sub_rate_pct` and `sla_breach_pct` in 0-100, counts ≥ 0. `src/nsh_pipeline/features/` is not built yet, so this reports `skipped` |

**Works on pandas 2 and 3.** Local runs and CI use pandas 3.1, where text is the `str` dtype and Parquet timestamps
load as `datetime64[us, UTC]`. The Airflow image uses pandas 2.1.4, where text is `object`. No schema pins a string
dtype or a time unit: checks test "every value is a str" and "timezone-aware UTC". Note that plan section 6 says
`datetime64[ns]`, but the processed Parquet actually stores microseconds. The M3 tests pass on both pandas 3.1 (Python
3.11 and 3.13) and pandas 2.1.4 + pyarrow 16.1.

**Why enums are split.** `validation/enums.py` keeps the contract A5 lists in canonical casing, in two groups:
* **Derived enums** (`employee_status`, `age_band`, `group_size_bucket`) are computed by our own code. A value outside
  the list is a pipeline bug, so the processed schema fails.
* **Source enums** (ticket/claim/swipe statuses, types, priorities, carriers, ...) are copied from NSH's systems. New
  values and casing drift there are a data-quality problem upstream, and plan stage 6 rates enum violations a
  **warning**. If the schema enforced them, ticket-status casing drift would stop the pipeline as a critical failure.
  So `anomalies.enum_violations` reports them as warnings, naming the casing-drift values separately.

Not in the schemas, on purpose: null or orphan join keys, missing money, duplicate bill lines (kept by design,
decisions.md #2) and impossible values. These are anomaly checks, each with its own severity.

## Anomaly checks
| Check | Severity | Rule | Why this severity |
|---|---|---|---|
| `duplicate_bill_lines` | critical | a (`member_id`, `billing_month`) appears more than once in `bill_lines` | PMPM double-counting (contract A8 #1) inflated PMPM 30-47% in production |
| `money_unit_error` | critical | a money value above **100×** its column's median (positive values, columns with ≥ 20 values) | a cents/dollars mix-up is a silent 100× error (A4). Clean data peaks at 69.5× (a $5,000 swipe and its ledger line); 50× would raise false alarms on every clean run |
| `orphan_keys` | critical | a `member_id` / `client_id` is null or matches no row in `members` / `clients` (14 relations) | rows that cannot be joined drop out of every per-member and per-client metric |
| `excluded_org_leak` | critical | an NS Demo / NSAI id in any table other than `clients` (including feature tables), or unflagged in `clients` | contract A0 rule 2: they are not clients |
| `unparseable_money` | critical | a `*_usd` value missing after parsing; for swipes, the raw value is shown (e.g. `'N/A'`) | a NaN is silently skipped by `sum()`, so every total that includes the record is understated with no error. It is the same kind of silent wrong-number hazard as a unit error, so it gets the same severity |
| `active_client_count` | warning | Active, non-excluded clients outside 70-100% of the scale's client count | the dataset is not the size we expect |
| `enum_violations` | warning | a source value outside the A5 lists; also raw `applications` status (preprocessing title-cases it, hiding drift) | upstream data quality; plan rates it a warning |
| `future_dates` | warning | swipe service date, claim or ticket creation date after the as-of date (by calendar day) | impossible, but it does not change money totals. Ledger effective dates are excluded: they legitimately trail the service date by about 2 days |
| `negative_amounts` | warning | negative money in swipes, claims, bill lines or premiums | ledger refunds and reserve deficits can be negative, so those tables are excluded |
| `term_before_hire` | warning | a member's termination date is before their hire date | impossible employment history |
| `distribution_drift` | warning | any drift finding vs the baseline (next section) | a shift is a signal to investigate, not proof of an error |

Each finding has record ids as `<table>:<key>` (e.g. `swipes:WEX-001083`, `bill_lines:<member_id>|2025-03-01`).

## Statistics and drift
`stats/<table>.json` has, per column: kind (`number`, `text`, `bool`, `datetime`), null count and rate,
min/max/mean/std/median, true rate for flags, the date range, and the top 5 values with shares for text columns with
up to 50 distinct values. Output is deterministic (no timestamps, rounded floats, dtype-independent kinds).

**Baseline:** `reports/baseline/` holds the stats of the clean seed-42 run, committed to git (not under
`reports/stats/`, which is a DVC output). `tests/validation/test_baseline.py` regenerates it from the seed and
compares. If a generator or preprocessing change makes that test fail, rebuild and commit the baseline:
```bash
python -m nsh_pipeline.validation.run --step stats --update-baseline
```
Without a baseline, the stats step writes one and reports `no_baseline`, not a failure.

**Thresholds** (`params.yaml → validation.drift`) are set from seed-to-seed noise at small scale (seeds 1-5 vs 42),
not tuned to the injected data:

| Signal | Threshold | Why |
|---|---|---|
| category share (the **headline** is the swipe Denied share) | 2.0 pts | across seeds the Denied share moves at most 0.54 pts |
| numeric mean | 20% | money means move at most 5.5% across seeds |
| null rate | 1.0 pt | |
| row count | 10% | same-seed reruns are byte-identical, so any change is real; 10% is where a volume change matters for PMPM denominators |
| columns or tables added/removed, column kind changed | always | schema drift |

## Clean run: 0 critical; 1 warning check (5 records)
| Step | Result |
|---|---|
| raw | passed (9 files match the manifest) |
| schema | 9/9 processed tables passed; features skipped |
| anomalies | **0 critical; 1 warning check (5 records)**: `future_dates` flags 5 claims |
| drift | `ok`, Denied share 5.38% = baseline |

**The warning is a generator bug, not a validation false alarm.** In `make_claims`
(`src/nsh_pipeline/generate/activity.py:143-144`), `createdAt = service date + 1-10 days` and
`closedAt = createdAt + 1-20 days`, with no cap at the reference date. Claims serviced in the last days before
2025-06-15 are therefore "created" afterwards: `CLM-100003`, `CLM-100012`, `CLM-100018`, `CLM-100036` and `CLM-100067`,
up to 2025-06-20. Reported to M1; once fixed, the clean run will be 0 critical, 0 warnings.
Full report: [docs/validation/clean/anomalies.md](validation/clean/anomalies.md).

## Injected run: detection evidence
Reproduce (about 15 s):
```bash
python -m nsh_pipeline.generate.run --params params.injected.yaml --out-dir data/raw_injected
python -m nsh_pipeline.preprocess.run --raw-dir data/raw_injected --out-dir data/processed_injected --as-of 2025-06-15
python -m nsh_pipeline.validation.run --step all --params params.injected.yaml \
    --raw-dir data/raw_injected --processed-dir data/processed_injected --report-dir reports/injected
```
`params.injected.yaml` equals `params.yaml` except `inject_issues.enabled: true` and `validation.mode: report`
(`tests/validation/test_params_injected.py` keeps them in sync). Both data folders and `reports/injected/` are
git-ignored.

From [`injection_detection.json`](validation/injected/injection_detection.json), which compares flagged record ids
with the generator's answer key (`manifest.json → injected_issues`):

| Injected issue | Caught by | Injected | Detected | Missed | Other records flagged in the table |
|---|---|---|---|---|---|
| `duplicate_bill_lines` | `duplicate_bill_lines` (critical) | 5 | **5** | 0 | 0 |
| `cents_as_dollars` | `money_unit_error` (critical) | 5 | **5** | 0 | 0 |
| `orphan_swipes` | `orphan_keys` (critical) | 5 | **5** | 0 | 0 |
| `unparseable_money` | `unparseable_money` (critical) | 5 | **5** | 0 | 0 |
| `future_service_dates` | `future_dates` (warning) | 3 | **3** | 0 | 0 |
| `status_casing_drift` | `enum_violations` (warning) | 5 | **5** | 0 | 0 |
| `denial_skew` | out of scope: bias slicing (M5) | 219 | - | - | - |

All in-scope issues detected, with no misses. `future_dates` reports 8 records in total: the 3 injected swipes plus
the same 5 generator-bug claims as the clean run (claims are a different table, so they are not counted against the
swipe answer key).

**Drift** ([`drift.json`](validation/injected/drift.json)): headline **Denied share 5.38% → 9.94% (+4.55 pts,
threshold 2.0)**, plus Approved share −4.55 pts and `swipes.denied_usd` mean +78%. All three are effects of
`denial_skew`. The ledger is 4.97% smaller (219 denied swipes never become ledger lines), below the 10% row-count
threshold, so it is not flagged; the Denied share is the signal designed to catch this.

**Answer-key id mapping** (`validation/answer_key.py`):

| Manifest issue | Manifest id | Check record id |
|---|---|---|
| `duplicate_bill_lines` | `<bill _id>:<employeeId>` | `bill_lines:<member_id>\|<YYYY-MM-01>` (bill id → billing month via raw `bills.jsonl`) |
| swipe issues | `wex_claims._id` | `swipes:<swipe_id>` |
| `status_casing_drift` | `tickets.ticketNumber` | `tickets:<ticket_number>` |

**Strict mode on the same data** (`--params params.yaml`) exits with code 1 after writing the reports and calling
`send_alert` with the 4 critical checks and their sample record ids (logged when no Slack/SMTP credentials are set).

## Pipeline integration (Airflow and DVC)
Checked on a scratch merge of `main` + this branch + `feat/orchestration-validate-raw` (2026-10-25):

* **`dvc repro`**: all three stages (`acquire`, `preprocess`, `validate`) ran; `validate` gave 0 critical; 1 warning check (5 records),
  raw passed, drift `ok`, and `dvc.lock` recorded its outputs.
* **Airflow (Docker, `nsh-airflow:2.10.5`: pandas 2.1.4, pyarrow 16.1, pandera 0.34.1)**, clean data: all 10 tasks
  `success`. `validate_raw`, `validate_processed`, `stats` and `anomalies` ran M3's code (stage status `ok`), and
  `run_summary.json` read `anomalies_by_severity = {critical: 0, warning: 5}` (5 flagged records from the one warning
  check). The first M3 task after a container start takes about 70 s (cold imports), and about 2 s after that. Take
  Gantt screenshots from a second run.
* **Airflow, injected data in strict mode**: `anomalies` failed, `dvc_version` and `run_summary` were
  `upstream_failed`, and `bias` (parallel) still succeeded. M3 logged "the task's failure callback sends the alert"
  and sent none itself. `notify_failure` sent exactly one alert, whose `Error:` field carries the 4 critical checks
  with sample record ids.

## Code and tests
| Code | Tests |
|---|---|
| `src/nsh_pipeline/validation/` (`run`, `schemas/`, `stats`, `drift`, `anomalies`, `answer_key`, `enums`, `common`) | `tests/validation/` (mirrors `src`) |
| `src/nsh_pipeline/utils/alerts.py` | `tests/utils/test_alerts.py` (Slack and SMTP mocked, no network) |

For every schema and check, a clean frame passes and a deliberately broken one fails with the expected check and
record ids. Contract tests cover the exact DAG commands, `anomalies.json` being readable by
`orchestration/summary.py`, the injected params file and the committed baseline. All tests use the hand-made sample in
`tests/fixtures/`: no network, Docker or Airflow. Coverage on `validation/` and `alerts.py` is 98-100%.

## Known limitations
* Feature schemas follow plan section 6.3 and are untested against real feature tables until `features/` exists.
* `clients.client_type` is always null: the generator does not emit `clientType` (M1).
* The answer key needs `data/raw/manifest.json`; without it, `injection_detection.json` is not written.
