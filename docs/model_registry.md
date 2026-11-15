# Model registry, rollback, CI/CD, Docker & alerts (phase 2, step 5)

Owner: Hiral Hitesh Rana · Guidelines §2.6, §7, §8.1, §8.7 · Task assigner section 3 (person 5) · Code:
`src/nsh_pipeline/registry/`, `Dockerfile.train`, `.github/workflows/train.yml`

This step decides whether the model the pipeline just selected may replace the production model, pushes it to
the registry when it may, keeps production when it may not (rollback), and tells the team either way.

## One command

```bash
dvc repro register          # the whole model pipeline, ending in a registry push or a recorded rollback
# or only the last step, on the reports already in reports/model/:
python -m nsh_pipeline.registry.promote --params params.yaml --reports-dir reports/model --data-dir data/training
```

Manual rollback (a model that passed every gate but misbehaves after release):

```bash
python -m nsh_pipeline.registry.rollback --previous --reason "drift after release"
python -m nsh_pipeline.registry.rollback --to v20261114-153617-2f68d25
```

## Which model is the candidate

The bias stage makes the final selection (guidelines §8.6), and it may pick a **mitigated** model in
`models_mitigated/`, not `models/best/`. The registry therefore registers the folder named by
`reports/model/bias.json → selected_model_dir`.

## Promotion rule

A candidate is promoted only when **all four** checks pass; otherwise production is kept and the reason recorded.

| # | Check | Source |
|---|---|---|
| 1 | `validation_gate`: hold-out validation passed | `reports/model/validation.json → passed` |
| 2 | `bias_gate`: a model was selected and passed the fairness gate on test | `reports/model/bias.json → passed`, `selected_model` |
| 3 | `candidate_metrics`: the *selected* model's test metrics meet the `model_validation` minimums | `bias.json → performance.test` |
| 4 | `beats_production`: candidate `pr_auc` > production `pr_auc` + `min_improvement` | production **re-scored on the current test split** |

Check 3 exists because validation.json describes `models/best`, which is not always the model the bias stage
selected. Check 4 re-scores the production model on the same rows the candidate was scored on: comparing
numbers recorded on different test splits (a new data release) would not be a fair comparison. If the
production bundle cannot be loaded, the recorded metric is used and the decision says so
(`production_metric_source: recorded`).

The DVC stages before `register` already stop the pipeline with exit code 1 when a gate fails, so in practice
checks 1–3 matter when `promote` is run on its own.

## Outputs

| File | Content |
|---|---|
| `registry/production.json` | current production record: `version`, `model_name`, `metrics` (test), `threshold`, `data_version`, `git_sha`, `mlflow_run_id`, `promoted_at`, `previous_version`, `uri`, `backend` |
| `reports/model/registry_decision.json` | this run's decision: `promoted`, `version`, `reason`, every check, both metrics, CI run link |
| `registry/history.jsonl` | one line per promotion, kept production and manual rollback |
| registry version | `bundle.tar.gz` = `manifest.json` (the production record) + `model/` + `reports/validation.json`, `bias.json`, `bias_report.md` |

Versions look like `v20261114-153617-2f68d25` (UTC time + commit), so they sort by time and point to the code.

## Backends

`params.yaml → registry.backend`, overridden by the `REGISTRY_BACKEND` environment variable.

| Backend | Used by | Where |
|---|---|---|
| `local` (default) | local runs, tests, pull requests | `registry/versions/<version>/bundle.tar.gz` + `registry/production.json` (git-ignored) |
| `gcp` | CI on `main` | GCP Artifact Registry, generic-format repository, through the `gcloud` CLI |

On GCP, package `nsh-denial-model` holds one version per promoted model. Artifact Registry versions cannot be
changed, so "which version is production" is a second package, `nsh-denial-model-production`: every promotion
and manual rollback uploads a small `production.json` as a new version, and the newest one wins. Both
packages are a complete audit trail in the console.

### GCP setup (one time)

```bash
# (a project must be linked to a billing account first; a brand-new project can return PERMISSION_DENIED
#  for a minute or two while the owner role propagates: wait and retry)
gcloud services enable artifactregistry.googleapis.com iam.googleapis.com --project "$GCP_PROJECT_ID"
gcloud artifacts repositories create nsh-models --repository-format=generic \
    --location=us-east1 --project "$GCP_PROJECT_ID"
gcloud iam service-accounts create nsh-model-ci --project "$GCP_PROJECT_ID"
gcloud artifacts repositories add-iam-policy-binding nsh-models --location=us-east1 \
    --project "$GCP_PROJECT_ID" --role=roles/artifactregistry.writer \
    --member="serviceAccount:nsh-model-ci@$GCP_PROJECT_ID.iam.gserviceaccount.com"
```

The project in use is `nsh-capstone-hiral` (project number `711808422691`), billed to the course credits.

### CI login without a key: Workload Identity Federation

GitHub Actions does **not** use a service-account key. Each run gets a short-lived OIDC token from GitHub,
which GCP exchanges for temporary `nsh-model-ci` credentials. Nothing secret is stored anywhere; the provider's
attribute condition lets only this repository in, and `nsh-model-ci` can only write to `nsh-models`.

```bash
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project "$GCP_PROJECT_ID"
gcloud iam workload-identity-pools create github --project "$GCP_PROJECT_ID" --location=global \
    --display-name="GitHub"
gcloud iam workload-identity-pools providers create-oidc github-repo --project "$GCP_PROJECT_ID" \
    --location=global --workload-identity-pool=github \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='jaira26/nsh-data-pipeline'"
gcloud iam service-accounts add-iam-policy-binding \
    "nsh-model-ci@$GCP_PROJECT_ID.iam.gserviceaccount.com" --project "$GCP_PROJECT_ID" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/jaira26/nsh-data-pipeline"
```

`train.yml` holds the project id, provider path and service-account email in plain `env:` (identifiers, not
secrets) and requests `id-token: write`. **The repository needs no GitHub secrets at all.**

Manual run (once `train.yml` is on `main`, as GitHub requires): Actions → Model pipeline → Run workflow →
pick a branch and `registry_backend: gcp`. Pull requests always use the local backend.

## Alerts

`nsh_pipeline.utils.alerts.send_alert` (phase 1) sends email (and/or Slack) from environment variables. With
nothing configured, alerts are logged only. Alerts go out from `nshcapestonealerts@gmail.com` (Gmail SMTP,
App Password).

| Setting | Locally | In CI |
|---|---|---|
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `ALERT_EMAIL_FROM` | `.env` | plain `env:` in `train.yml` (not secret) |
| `SMTP_PASSWORD` | `.env` | GCP Secret Manager `smtp-password` |
| `ALERT_EMAIL_TO` | `.env` | GCP Secret Manager `alert-email-to` (keeps addresses out of the repo) |

CI reads the two secrets with the same keyless WIF login (`google-github-actions/get-secretmanager-secrets`);
`nsh-model-ci` has `secretAccessor` on these two secrets only. Pull-request runs use the local registry, so
their alert titles end in `[local test registry]`. Pull requests from forks get no OIDC token and run without
alerts.

One-time setup (the values are piped from `.env`, never typed or printed):

```bash
gcloud services enable secretmanager.googleapis.com --project "$GCP_PROJECT_ID"
for s in smtp-password alert-email-to; do
  gcloud secrets create $s --project "$GCP_PROJECT_ID" --replication-policy=automatic
  gcloud secrets add-iam-policy-binding $s --project "$GCP_PROJECT_ID" \
      --role=roles/secretmanager.secretAccessor \
      --member="serviceAccount:nsh-model-ci@$GCP_PROJECT_ID.iam.gserviceaccount.com"
done
for k in SMTP_PASSWORD:smtp-password ALERT_EMAIL_TO:alert-email-to; do
  python -c "import sys; from dotenv import dotenv_values; sys.stdout.write(dotenv_values('.env')['${k%%:*}'])" \
    | gcloud secrets versions add "${k#*:}" --project "$GCP_PROJECT_ID" --data-file=-
done
```

To change the password or recipients, add a new secret version the same way; CI uses `latest`.

| Event | Sent by |
|---|---|
| training completed + model promoted | `registry.promote` |
| training completed + production kept (rollback) | `registry.promote` |
| validation gate blocked / bias gate blocked / pipeline failed | `registry.notify` (CI `if: failure()` step) |
| manual rollback | `registry.rollback` |

## Docker

`Dockerfile.train` (separate from the Airflow `Dockerfile`): Python 3.11, the pinned `requirements.txt`, the
package, DVC and the gcloud CLI. The repo is mounted, so the image does not change when code changes.

```bash
docker build -f Dockerfile.train -t nsh-train .
docker run --rm -v "$PWD":/app nsh-train                       # dvc repro register, local registry
docker run --rm -v "$PWD":/app --env-file .env nsh-train       # same, with Slack / email alerts
```

## Tests

`tests/registry/`: decision rules (each gate, equal / within-margin metric kept, first version), bundle round
trip and unsafe archive members, both backends (gcloud mocked), and end-to-end on the local backend: a worse or
failing candidate is never promoted, a better one is, production is re-scored on the current split, manual
rollback, failure classification for alerts.
