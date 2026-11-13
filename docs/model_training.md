# Model training, tuning, selection and MLflow (phase 2, step 2)

Owner: Kalpan Pankajbhai Shah · Guidelines §2.2, §3, §4, §8.3

**Task:** will this card swipe be denied? (binary, `denied` = 1). Input = the training-data contract
in `nsh_pipeline.modeling.dataset` (`FEATURE_COLUMNS`, `LABEL_COLUMN`, splits in `data/training/`).

## One command

```bash
python -m nsh_pipeline.modeling.train --data-dir data/training --params params.yaml
# or, as part of the pipeline:
dvc repro train
```

Flags: `--n-trials N` (Optuna trials per model; CI uses 2), `--candidates lightgbm ...`,
`--sample-weights data/bias/swipe_weights.parquet` (bias mitigation by reweighing: `swipe_id`, `weight`),
`--models-dir`, `--reports-dir`, `--stub` (see below).

## What it does

```
train split ──► Optuna TPE search (stratified 5-fold CV, PR-AUC) ──► best params
            └─► refit on the whole train split ──► score on validation split ──► models/candidates/<name>/
all candidates ──► rank by validation PR-AUC ──► models/best/  +  charts + candidates.json
```

| Step | Uses | Never uses |
|---|---|---|
| Tuning (CV) | train split | validation, test |
| Selection | validation split | test |
| Hold-out validation (Raj) | test split | — |

### Candidates (`modeling/candidates.py`)

All three are scikit-learn `Pipeline`s with the same preprocessing, so they are interchangeable for
validation, bias checks and the registry:

| Step | Numeric features | Categorical features |
|---|---|---|
| Preprocess | median impute → standard scale | to text (missing = `Unknown`) → one-hot, unseen values ignored |

| Model | Class imbalance | Fixed settings |
|---|---|---|
| `logistic_regression` | `class_weight="balanced"` | `solver="saga"`, `max_iter=5000` |
| `random_forest` | `class_weight="balanced_subsample"` | `n_jobs=-1` |
| `lightgbm` | `class_weight="balanced"` | `subsample_freq=1`, `n_jobs=1` (faster on small data) |

### Hyperparameter search space (`params.yaml → model.search_space`)

| Model | Hyperparameter | Range |
|---|---|---|
| Logistic regression | `C` | 0.001 – 100 (log) |
| | `l1_ratio` | 0 (L2) – 1 (L1) |
| Random forest | `n_estimators` | 100 – 500, step 50 |
| | `max_depth` | 3 – 20 |
| | `min_samples_leaf` | 1 – 30 |
| | `max_features` | sqrt, log2, 0.5 |
| LightGBM | `n_estimators` | 100 – 600, step 50 |
| | `learning_rate` | 0.01 – 0.3 (log) |
| | `num_leaves` | 8 – 64 |
| | `min_child_samples` | 5 – 60 |
| | `subsample`, `colsample_bytree` | 0.6 – 1.0 |
| | `reg_lambda` | 0.001 – 10 (log) |

Search: Optuna TPE sampler, seeded (`model.seed`), `model.n_trials` trials per model (25 by default),
objective = mean CV PR-AUC. Same seed and data → same best parameters (tested).

### Selection (`modeling/select.py`)

Best **validation PR-AUC**; ties go to the higher CV PR-AUC, then to the simpler model
(logistic regression < random forest < LightGBM). Het's fairness-aware final selection runs after this
and can pick another candidate from `models/candidates/`.

Each candidate's metadata also records validation PR-AUC, ROC-AUC, precision, recall, F1 and the
confusion matrix at the **best-F1 threshold on validation**. That threshold is a starting point; the
official decision threshold and gate belong to the validation step (Raj).

## Outputs (model contract, task assigner §2.2)

```
models/candidates/<model_name>/model.joblib + metadata.json
models/best/model.joblib + metadata.json
reports/model/training/candidates.json        ranking, params, CV + validation metrics, MLflow run ids
reports/model/training/model_comparison.png   grouped bar chart (README)
reports/model/training/confusion_matrices.png per model, validation split (README)
reports/model/training/pr_curves.png          precision-recall curves, validation split
reports/model/training/trials/<model>.csv     every Optuna trial (for hyperparameter sensitivity)
```

`metadata.json` keys: `model_name, params, cv_metrics, validation_metrics, mlflow_run_id, data_version,
git_sha, created_at, seed, n_trials, sample_weighted`.

Load a model (teammates):

```python
from nsh_pipeline.modeling.artifacts import load_model
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, load_split

model, metadata = load_model("models/best")
scores = model.predict_proba(load_split("test")[FEATURE_COLUMNS])[:, 1]
```

### Stub model (for teammates' tests)

`python -m nsh_pipeline.modeling.train --stub --data-dir tests/fixtures/training` writes an untuned
logistic regression in exactly this layout. In tests use the session fixture `stub_model_dir`
(in `tests/conftest.py`): `load_model(stub_model_dir / "best")`. No model file is committed.

## MLflow

Experiment **`nsh-denial-model`**. Run tree for one training command:

```
train-and-select                  (stage=selection)   charts, candidates.json, selected_model tag
├── logistic_regression           (stage=candidate)   best params, cv_* + val_* metrics, model, trials.csv
│   ├── logistic_regression-trial-000   (stage=tuning_trial)  params + cv_* metrics
│   └── ...
├── random_forest ...
└── lightgbm ...
```

Every run carries `data_version` (e.g. `data-v1.1`, from the dataset card) and `git_sha` (`-dirty`
when there are uncommitted changes; `GITHUB_SHA` in CI), plus `data_hash`, `data_scale` and
`injected_issues`.

**Local (default):** runs go to `sqlite:///mlflow.db` and `./mlruns/` in the repo (both git-ignored).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

**Docker server:** set `MLFLOW_TRACKING_URI` and everything else stays the same.

```bash
docker compose -f docker-compose.mlflow.yaml up -d    # http://localhost:5001 (5000 is taken by macOS AirPlay)
export MLFLOW_TRACKING_URI=http://localhost:5001
python -m nsh_pipeline.modeling.train
```

Compare runs in the UI: experiment → filter `tags.stage = "candidate"` → select → Compare.

## Tests

`pytest tests/modeling` (offline, on the sample fixture, throwaway MLflow store in `tmp_path`):
every family fits and predicts; unseen/missing categories are handled; tuning returns params inside
the space, is reproducible, and sees only training rows; MLflow runs carry the required tags;
selection picks the best metric (and breaks ties deterministically); the saved models follow the
contract; sample weights flow through.
