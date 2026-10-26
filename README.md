# nsh-data-pipeline

[![CI](https://github.com/jaira26/nsh-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/jaira26/nsh-data-pipeline/actions/workflows/ci.yml)

Data pipeline for synthetic NSH reporting data (Airflow + DVC)

## Data acquisition & preprocessing

```bash
pip install -r requirements.txt    # exact versions CI tests with
pip install -e ".[dev]"
python -m nsh_pipeline.acquire.download --params params.yaml --out-dir data/raw          # download (or generate) raw data
python -m nsh_pipeline.preprocess.run --raw-dir data/raw --out-dir data/processed --as-of 2025-06-15
pytest --cov=nsh_pipeline
```

- [Data acquisition & synthetic data generation](docs/data_acquisition.md): download stage, generator, `params.yaml`, injected issues
- [Preprocessing](docs/preprocessing.md): cleaning steps and the 9 processed tables
- [Decisions & assumptions](docs/decisions.md)

## Validation, statistics & anomaly detection

```bash
python -m nsh_pipeline.validation.run --step all     # or --step raw | schema | stats | anomalies
```

Validation runs after preprocessing (Airflow tasks `validate_processed`, `stats`, `anomalies`; DVC stage `validate`):

- **Schemas** (pandera): light checks on the raw files against `manifest.json`, strict checks on the 9 processed
  tables (columns from `flatten.py`, float-dollar money, UTC dates, unique keys, id formats), and the feature tables
  once they exist.
- **Statistics and drift**: per-table stats in `reports/stats/`, compared with the committed clean baseline in
  `reports/baseline/`. The headline signal is the swipe Denied share.
- **11 anomaly checks** for the data contract's hazards. Critical: duplicate bill lines (PMPM double-counting), money
  100× the median (cents read as dollars), orphan join keys, NS Demo/NSAI leaks, unparseable money. Warnings: enum and
  casing drift, future dates, negative amounts, termination before hire, client count, drift. Output goes to
  `reports/anomalies.json` and `anomalies.md`.
- **Alerts** by Slack webhook or SMTP from `.env`; with no credentials they are logged. `validation.mode: strict`
  fails the task on a critical finding; `report` only reports.

**Clean data: 0 critical; 1 warning check (5 records).** The warning is 5 claims created after the as-of date, a generator bug in
`generate/activity.py:143-144` (claim dates are not capped at the reference date). **Injected data**
(`params.injected.yaml`): every injected issue in scope is detected (5/5 duplicate bill lines, 5/5 cents-as-dollars,
5/5 orphans, 5/5 unparseable money, 3/3 future dates, 5/5 casing drift), with none missed. The denial skew is caught
by the drift headline (Denied share 5.4% → 9.9%) and is analysed by bias slicing.

Details, thresholds, how to reproduce the evidence: [docs/validation.md](docs/validation.md).