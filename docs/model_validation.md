# Model validation, validation gate & sensitivity analysis (phase 2, step 3)

Owner: Raj Mehta · Guidelines §2.3, §5, §7.2, §8.4 · Task assigner section 3 (person 3) · Code:
`src/nsh_pipeline/modeling/evaluate.py`, `sensitivity.py`, `validation_plots.py`

This step takes the model that training selected (`models/best/`), measures it once on data it has never seen,
blocks it if it is too weak, and explains what it learned.

## One command per part

```bash
# hold-out validation + gate (exit code 1 when the gate fails)
python -m nsh_pipeline.modeling.evaluate --params params.yaml --model-dir models/best \
    --data-dir data/training --out-dir reports/model

# SHAP + hyperparameter sensitivity
python -m nsh_pipeline.modeling.sensitivity --params params.yaml --model-dir models/best \
    --data-dir data/training --out-dir reports/model/sensitivity

# or, as part of the pipeline (after build_training_set and train):
dvc repro validate_model sensitivity
```

| Option | Default | Notes |
|---|---|---|
| `--params` | `params.yaml` | settings under `model_validation:` |
| `--model-dir` | `models/best` | `model.joblib` + `metadata.json` (model contract §2.2) |
| `--data-dir` | `data/training` | splits + `dataset_card.json` (training-data contract §2.1) |
| `--out-dir` | `reports/model` / `reports/model/sensitivity` | `model_validation.out_dir` / `.sensitivity.out_dir` |
| `--validation-report` (sensitivity) | `reports/model/validation.json` | the chosen threshold; chosen again on the validation split if missing |
| `--no-mlflow` | off | write every report, skip MLflow logging |

The DVC stages are named `validate_model` and `sensitivity`, because `validate` is already M3's data validation.
For the same reason the settings live under `model_validation:` in `params.yaml`, not `validation:`.

## What is evaluated, on which split

| Split | Used for | Never used for |
|---|---|---|
| train (Jan–Apr) | fitting and tuning (Kalpan), the SHAP background sample | threshold, reported metrics |
| validation (May) | model selection (Kalpan) and **choosing the decision threshold** (here) | reported metrics |
| test (1–15 June) | **the reported hold-out metrics, the gate, SHAP explanations** | any choice: threshold, tuning, selection |

The test split is scored once at a threshold that is already fixed. Three things in the code enforce this:

* `threshold_from_split` raises `ThresholdLeakError` for any split except `validation`.
* `run_validation` loads the test split only after the threshold is chosen.
* `check_disjoint` fails if a swipe appears in both validation and test.

`tests/modeling/test_evaluate.py::test_threshold_comes_from_validation_not_test` spies on the data loading and the
threshold function to prove it: the two splits imply different thresholds, and the report carries the validation one.

## Metrics

All metrics are for the positive class, **denied = 1**, on the test split.

| Metric | Definition | Why |
|---|---|---|
| **PR-AUC** (primary) | average precision: area under precision vs recall, over all thresholds | Denials are ~13% of swipes. PR-AUC ignores the many easy approvals, and random guessing scores the positive rate (0.13), not 0.5 |
| PR-AUC lift | PR-AUC ÷ test positive rate | "×-better than guessing"; comparable when the denial rate changes |
| ROC-AUC | probability that a random denied swipe scores above a random approved one | threshold-free ranking quality; 0.5 = chance |
| precision | TP / (TP + FP) at the threshold | of the swipes we flag, how many are really denied |
| recall | TP / (TP + FN) at the threshold | of the denied swipes, how many we flag |
| F1 | 2 · precision · recall / (precision + recall) | balance of the two |
| confusion matrix | `[[TN, FP], [FN, TP]]` (rows = actual approved / denied) | the counts behind the four numbers above |

`select.evaluate` (Kalpan's helper) computes these, so training and validation report identical definitions.
`tests/modeling/test_evaluate.py::test_holdout_metrics_match_hand_computed_toy` checks every metric against a toy
prediction worked out by hand.

## Decision threshold

`params.yaml → model_validation.threshold_method`, chosen on the validation split:

* **`best_f1`** (default): the cut-off with the highest F1. No business cost of a false alarm vs a missed denial is
  known for the synthetic setting, so we weigh precision and recall equally. This is also the convention training
  uses for the candidate confusion matrices, so the numbers are comparable.
* `target_precision`: the cut-off with the highest recall among those with validation precision ≥ `target_precision`.
  Use it when a downstream consumer needs "at least X% of flags are real". If no cut-off reaches the target, it falls
  back to best F1 and logs a warning (`threshold_selection.fallback = true`).

## Validation gate

`passed` = every configured minimum is met on the test split (`null` = not checked). When it fails, `validation.json`
and the plots are still written, then the command logs which metrics failed and exits with code 1, so `dvc repro` and
CI stop before bias checking and the registry push.

| Key | Default | Reasoning |
|---|---|---|
| `min_pr_auc` | 0.20 | 1.5× the ~0.13 denial rate. Candidates on data-v1.1: 0.23–0.35 |
| `min_pr_auc_lift` | 1.5 | same bar, but robust if the denial rate of a new data release changes |
| `min_roc_auc` | 0.65 | chance is 0.5; candidates score 0.67–0.71 on the test split |
| `min_recall` | 0.40 | catches a degenerate threshold that misses most denials (best-F1 recall is ~0.6–0.7) |
| `min_precision`, `min_f1` | null | available, not used: precision at best F1 is low by design for a 13% class |

**Why not 0.75?** The dataset card's target ROC-AUC (0.75–0.85) is the design goal for the data-v1.1 signal. On the
small-scale release that every candidate trains on, the best model reaches 0.71 on test and the others ~0.67
(training's own README table: 0.665–0.678 on validation). A 0.75 gate would block every model the pipeline can
currently produce, so it would block nothing useful. The minimums sit between chance and the weakest real candidate,
so a broken or untrained model fails. For example, the untuned stub on the tiny fixture scores ROC-AUC 0.42 and is
blocked on all four checks. Raise them when the full-scale data (`scale: full`) lands and the models improve.

## Hold-out results (data-v1.1, small scale)

Local run on the small-scale data-v1.1 release, from a 10-trials-per-model training run (the selected family,
logistic regression, matches the 25-trial run in the README). Re-run `dvc repro validate_model` to refresh.

| Test split (700 swipes, 13.0% denied) | Value |
|---|---|
| Threshold (best F1 on validation) | 0.530 |
| ROC-AUC | 0.714 |
| **PR-AUC** | **0.351** (2.7× chance) |
| Precision / recall / F1 | 0.237 / 0.692 / 0.353 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[406, 203], [28, 63]]` |
| Gate | **passed** |

The model catches 69% of denials, and about one in four of its flags is a real denial (vs one in eight by chance).

## Outputs

```
reports/model/validation.json                     # contract: roc_auc, pr_auc, precision, recall, f1, threshold,
                                                  #   confusion_matrix, passed (+ informational fields)
reports/model/validation/roc_curve.png            # ROC with chance line and the operating point
reports/model/validation/pr_curve.png             # PR curve with the no-skill line (positive rate)
reports/model/validation/confusion_matrix.png     # counts + row shares at the threshold
reports/model/validation/precision_recall_vs_threshold.png
reports/model/sensitivity/top_features.json       # ranked features (mean |SHAP|), signal check, local cases
reports/model/sensitivity/shap_summary.png        # beeswarm
reports/model/sensitivity/shap_importance.png     # mean |SHAP| bar chart (documented drivers in green)
reports/model/sensitivity/waterfall_{true_positive,false_positive,false_negative}.png
reports/model/sensitivity/hyperparameter_sensitivity_<model>.png
reports/model/sensitivity/hyperparameter_importance.{json,png}
```

Informational fields in `validation.json`: `model_name`, `data_version`, `git_sha` (code that ran the validation),
`model_git_sha`, `evaluated_split`, `n_test`, `positive_rate`, `pr_auc_lift`, `threshold_method`,
`threshold_selection` (split, rows, validation precision/recall, fallback), `gate` (one record per check:
metric, value, minimum, passed) and `mlflow_run_id`.

Reports are git-ignored, because they are generated from real data. DVC tracks `validation.json` as a metric
(`dvc metrics show`) and the plots folder as plots.

## SHAP (feature sensitivity)

**How it works.**
* A fixed-seed sample of the test split (`shap_sample_size: 500`, `seed: 42`) goes through the Pipeline's
  preprocessing step, and the estimator is explained on the transformed columns (`get_feature_names_out`).
* The explainer depends on the estimator type:

  | Estimator | Explainer | Output space |
  |---|---|---|
  | random forest, extra trees, decision tree | `TreeExplainer` | probability |
  | LightGBM, XGBoost, gradient boosting | `TreeExplainer` | log-odds |
  | logistic regression (and other linear models) | `LinearExplainer` | log-odds |
  | anything else | permutation `Explainer` on `predict_proba` | probability |

  The linear and permutation explainers integrate over 100 training rows (`shap_background_size`).
* SHAP values are additive, so the one-hot columns of a categorical (`service_category_Dental`,
  `service_category_Vision`, ...) are **summed back to the original feature**. Every plot and the JSON therefore have
  one row per original feature. Tests check additivity (base value + sum of SHAP = model output) for the linear and
  tree explainers.

**How to read the outputs.**
* `shap_summary.png` (beeswarm): one dot per swipe and feature. Right of zero pushes towards *denied*. For numeric
  features the colour is the feature value (red = high). Categoricals have no order, so they are grey.
* `shap_importance.png`: mean |SHAP|, the average size of a feature's effect. The documented drivers are green.
* `waterfall_*.png`: one swipe each, from the expected model output `E[f(X)]` to this swipe's output `f(x)`, one
  bar per feature, with the swipe's actual values on the left. The cases are the most confident true positive, the most
  confident false positive (highest-scoring approved swipe) and the clearest false negative (lowest-scoring denied
  swipe), at the validation threshold.
* `top_features.json → signal_check`: the documented data-v1.1 drivers (`expected_drivers`: amount, service
  category, tenure, plan deductible, prior denials) should all rank in the top `driver_top_n` (7). If not, the
  command logs a warning, because the model may not have learned the intended signal. It does not fail the stage:
  this is a diagnostic, not a gate.

**Result on data-v1.1 (small scale):** the top four features are all documented drivers: `prior_denials_90d`,
`amount_usd`, `service_category`, `tenure_days`. Two things are worth knowing:
* `plan_deductible` ranks 9th, so the signal check warns. The generator applies the high-deductible effect as a step
  at ≥ $3,000 (+0.5 log-odds). A linear model on the scaled deductible captures that step only weakly.
* `age_band` (5th) and `state` (6th) carry no denial effect by design (no state/age effect in `denial_model`). Their
  SHAP weight is what the model picked up from noise in a few thousand training rows. This is worth a look in the
  bias step (Het), which slices on both.

## Hyperparameter sensitivity

**How it works.** Every Optuna trial in training is an MLflow run tagged `stage = tuning_trial` and `model_name`,
with its hyperparameters as params and its CV metrics (see docs/model_training.md).
* `sensitivity.py` reads them with `MlflowClient.search_runs` (filter `tuning_run_filter`) from the same store and
  experiment as training (`model.mlflow`, overridden by `MLFLOW_TRACKING_URI`), groups them by model family, and
  plots `cv_pr_auc` against every hyperparameter. Numeric ones get a scatter, on a log axis where the search space
  says `log: true`. Categorical ones get a box plot. The best trial is a red star.
* The metric is the CV PR-AUC on the train split, because that is the objective tuning optimises. The validation split
  is not scored per trial.
* **Importance** per hyperparameter, normalised to sum to 1:
  * ≥ `min_runs_for_forest` (10) runs: a random-forest regressor fit on hyperparameters → metric, then permutation
    importance (how much the fit degrades when one hyperparameter is shuffled). It captures non-linear effects and
    interactions.
  * fewer runs: |Spearman ρ| for numeric hyperparameters and the correlation ratio η for categorical ones (the share
    of the metric's spread explained by the category). Both are on a 0–1 scale.
  * the JSON always also carries `spearman_rho` (direction) for numeric hyperparameters, and lists `constant_params`.
* **No runs / too few runs:** a missing store, missing experiment, or no matching runs gives `status: "no_runs"` with
  the reason. A family with fewer than `min_runs` (3) runs gets `status: "too_few_runs"`. Both are logged as a
  warning, written to `hyperparameter_importance.json`, and the stage still succeeds. Reading never creates an MLflow
  store.

**Result on data-v1.1 (10 trials per model):**

| Model | Most influential | Importance | Reading |
|---|---|---|---|
| Logistic regression | `C` | 0.84 | regularisation strength dominates; `l1_ratio` matters little |
| Random forest | `min_samples_leaf` | 0.64 | larger leaves (~20) regularise and win; `max_features` second (0.28) |
| LightGBM | `learning_rate` | 0.86 | low learning rates (~0.02) win; the tree-shape params barely matter |

With 25 trials (the default) the estimates are more stable. Every family's message is the same: on this
small, noisy dataset, **regularisation is what matters**.

## MLflow

Both commands log one run each to experiment `nsh-denial-model` (the same store as training):

| Run | Tags | Params | Metrics | Artifacts |
|---|---|---|---|---|
| `validation` | `stage=validation`, `data_version`, `git_sha`, `model_name`, `model_git_sha`, `model_run_id`, `data_hash`, `gate_passed` | threshold, method, target precision, gate minimums | `test_roc_auc`, `test_pr_auc`, `test_precision`, `test_recall`, `test_f1`, `test_pr_auc_lift`, `threshold`, `gate_passed` | `validation/validation.json`, `validation/plots/*.png` |
| `sensitivity` | `stage=sensitivity`, the same lineage tags, `signal_check_matches`, `hyperparameter_status` | sample size, explainer, `driver_top_n`, tuning metric | drivers found in top N, tuning runs analysed | `sensitivity/*` (every file above) |

`data_version` comes from the model's metadata (else the dataset card), and `git_sha` from `git rev-parse HEAD`
(`-dirty` with uncommitted changes, `GITHUB_SHA` in CI). Turn logging off with `--no-mlflow` or
`model_validation.mlflow.enabled: false`. Everything runs offline against the default local `sqlite:///mlflow.db`.

## Tests

```bash
pytest tests/modeling/test_evaluate.py tests/modeling/test_sensitivity.py    # ~30 s, offline
```

The fixtures (`tests/modeling/conftest.py`, validation block) are self-contained:
* contract-shaped synthetic splits with the documented signal
* a logistic regression and a random forest Pipeline fitted in the fixture
* a throwaway SQLite MLflow store under `tmp_path`, with fake tuning runs logged into it

So the tests don't depend on the training code. Coverage: `evaluate.py` 100%, `sensitivity.py` 98%,
`validation_plots.py` 100%.

Decisions: rows 35–40 in [decisions.md](decisions.md).
