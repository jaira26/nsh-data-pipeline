# Orchestration: Airflow, Docker, DVC, logging and Gantt (M4)

## What runs

```
acquire_data → validate_raw → preprocess → validate_processed → build_features → ┬ stats     ┬ → dvc_version → run_summary
                                                                                 ├ anomalies ┤
                                                                                 └ bias      ┘
```

| Task | Calls | Owner |
|---|---|---|
| `acquire_data` | `acquire_data(params, data/raw)`; XCom = `"download"` or `"generated"` | M1 |
| `validate_raw` | `python -m nsh_pipeline.validation.run ... --step raw` (files vs manifest, raw schemas) | M3 |
| `preprocess` | `run_preprocess(data/raw, data/processed, reference_date)`; XCom = row counts | M2 |
| `validate_processed` | `python -m nsh_pipeline.validation.run ... --step schema` | M3 |
| `build_features` | `python -m nsh_pipeline.features.run ...` | FE |
| `stats`, `anomalies` | `python -m nsh_pipeline.validation.run ... --step stats / anomalies` | M3 |
| `bias` | `python -m nsh_pipeline.bias.slicing ...` | M5 |
| `dvc_version` | `dvc commit -f <stage>` for every `dvc.yaml` stage, optional `dvc push` | M4 |
| `run_summary` | writes `reports/run_summary.json` | M4 |

The CLI commands live in one place, `src/nsh_pipeline/orchestration/stages.py`. Until a
module is merged, its task logs "not merged yet" and succeeds, so the DAG runs end to end
at every stage of the week. Once merged, a non-zero exit fails the task, and the failure
triggers M3's `send_alert` through `on_failure_callback`.

## Prerequisites

* Docker Desktop with **at least 4 GB RAM** for Docker (Settings → Resources).
  Windows: use the WSL2 backend.
* Git. Clone outside OneDrive.

## Run it with Docker

```bash
cp .env.example .env              # Linux: also set AIRFLOW_UID to the output of `id -u`
mkdir -p logs/airflow             # PowerShell: mkdir logs\airflow
docker compose build              # first time ~5 min
docker compose up airflow-init    # wait for "exited with code 0"
docker compose up -d
docker compose ps                 # webserver and scheduler become "healthy" in ~1 min
```

Open **http://localhost:8080**, log in as `airflow` / `airflow`, switch `nsh_data_pipeline`
on and press ▶ **Trigger DAG**. Watch the **Grid** view; click any task square → **Logs**.

How it fits together: the repo is mounted into the containers at `/opt/airflow/project`,
the DAG folder points at `dags/`, and the image installs `nsh_pipeline` in editable mode
from that mounted `src/`. Code changes apply without rebuilding; rebuild only when
`pyproject.toml` dependencies change (`docker compose build && docker compose up -d`).
Outputs appear in your local `data/` and `reports/` folders.

Useful commands:

```bash
docker compose exec airflow-scheduler airflow dags list-import-errors       # DAG not showing?
docker compose exec airflow-scheduler airflow dags test nsh_data_pipeline   # whole DAG in your terminal
docker compose exec airflow-scheduler airflow tasks test nsh_data_pipeline preprocess 2025-06-15
docker compose exec -w /opt/airflow/project airflow-scheduler pytest tests/orchestration
docker compose down               # stop (add -v to wipe the Airflow database)
```

## Run it without Docker (DVC)

```bash
pip install -e ".[dev]"
dvc repro                         # runs every stage in dvc.yaml whose inputs changed
dvc push                          # upload data to the remote
git add dvc.yaml dvc.lock && git commit -m "data: pipeline outputs"
```

Stages beyond `acquire` and `preprocess` are commented out in `dvc.yaml` until their modules
merge; uncomment each one as it lands.

## Data versioning

* `dvc.yaml` declares each stage's command, dependencies, parameters and outputs.
  `dvc.lock` records the exact hash of every output. Git tracks those two files; the data
  lives in the DVC cache and the remote.
* After an Airflow run, `dvc_version` runs `dvc commit -f` for each stage so `dvc.lock`
  matches what Airflow produced. Commit `dvc.lock` from your machine afterwards.
* `dvc push` from inside the container needs remote credentials, so it is off by default
  (`DVC_PUSH=false`). Run `dvc push` on the host after the DAG instead.
* Going back to an older version of the data:

  ```bash
  git checkout <commit>
  dvc checkout                    # restores data/ to match that commit's dvc.lock
  ```

## Logging and run summary

Every module uses `logging.getLogger(__name__)`. Airflow captures these messages in the
task logs, and the output of each CLI stage is streamed line by line into its task log.
Logs are kept under `logs/airflow/`.

`run_summary` writes `reports/run_summary.json` with the run id, git commit, dataset
source, row counts per table, anomaly counts by severity, each stage's status, task
durations and the DVC result.

## Gantt chart and optimization

1. **Before:** set `NSH_PARALLEL_REPORTS=false` in `.env`, run `docker compose up -d`
   (reloads the setting), trigger the DAG, open the run → **Gantt** tab, save
   `docs/gantt_before.png`.
2. **After:** set it back to `true`, `docker compose up -d`, trigger again, save
   `docs/gantt_after.png`.
3. Notes for the report: _fill in after the runs_. What was the longest bar (likely
   `acquire_data` on the first run, then `preprocess`)? How much did running stats,
   anomalies and bias side by side save? What else could be done, such as caching the
   download or skipping unchanged stages with `dvc repro`?

## Troubleshooting

| Symptom | Fix |
|---|---|
| `env file .env not found` | `cp .env.example .env` |
| DAG missing from the UI | `airflow dags list-import-errors` (command above) |
| Webserver restarts in a loop | Give Docker ≥ 4 GB RAM |
| `PermissionError` on `logs/` or `data/` (Linux) | Set `AIRFLOW_UID` to `id -u` in `.env`, then `docker compose down && docker compose up -d` |
| Port 8080 in use | Change `"8080:8080"` to `"8081:8080"` and open localhost:8081 |
| `pip` conflict during `docker compose build` | Pin the conflicting package to a version Airflow 2.10.5 allows |
| Task log says "not merged yet" | Expected until that teammate's module is on `main` |
| `dvc_version` reports `skipped` | `dvc init` hasn't been committed yet, or DVC failed to install in the image |
