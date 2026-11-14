# Model bias report

Data `data-v1.1` · code `7e3c955f7102` · generated 2026-11-14T15:36:17+00:00

## Verdict

- **Fairness gate passed.** Selected model: **`logistic_regression_reweighed`** (base `logistic_regression`, reweighed on `state`), decision threshold 0.530. Saved in `models_mitigated/candidates/logistic_regression_reweighed`.
- logistic_regression_reweighed has the best validation PR-AUC and passes the fairness gate.
- Test split (test, never used to choose): PR-AUC 0.325, ROC-AUC 0.704, recall 68.1%, precision 24.5%.
- No group is flagged on the test split (23 group × metric tests).

## Candidates

Ranked by validation PR-AUC; the selected model is the first one that passes the gate on the validation split. Gaps are the largest worse-than-overall gap of any judged group.

| # | Candidate | Mitigation | Val PR-AUC | Val worst FPR gap | Val worst recall gap | Val gate | Test PR-AUC | Test gate |
|---|---|---|---|---|---|---|---|---|
| 1 | `logistic_regression_reweighed` ★ | reweighing on `state` | 0.230 | 16.9 pts | 4.3 pts | passed | 0.325 | passed |
| 2 | `logistic_regression` | – | 0.225 | 31.0 pts | 1.4 pts | **FAILED** | 0.351 | passed |
| 3 | `random_forest` | – | 0.215 | 16.9 pts | 0.6 pts | passed | 0.229 | passed |
| 4 | `lightgbm` | – | 0.208 | 16.9 pts | 2.0 pts | passed | 0.233 | passed |

![Fairness against performance](fairness_vs_performance.png)

Flags on the validation split (these decided which candidates need mitigation):

| Candidate | Group | Metric | Group value | Overall | Gap (pts) | n | p |
|---|---|---|---|---|---|---|---|
| `logistic_regression` | state = WA | FPR | 63.0% | 32.6% | +30.3 | 108 | < 0.001 |
| `logistic_regression` | group_size_bucket = 50-99 | FPR | 63.6% | 32.6% | +31.0 | 55 | < 0.001 |

## Mitigation

### `logistic_regression` → `logistic_regression_reweighed` (reweighing on `state`)

- Training rows reweighed so the denial rate is equal in every `state` group: train gap 11.3 → 0.0 pts. Weights 0.63–2.85; effective sample size 2,928 of 2,963 rows (99%).
- Same tuned hyperparameters, refit with the weights; threshold re-chosen on validation.

| Split | Metric | Before | After |
|---|---|---|---|
| validation | PR-AUC | 0.225 | 0.230 |
| validation | recall | 0.584 | 0.613 |
| validation | precision | 0.213 | 0.222 |
| validation | overall FPR | 32.6% | 32.4% |
| validation | worst FPR gap | 31.0 pts | 16.9 pts |
| validation | worst recall gap | 1.4 pts | 4.3 pts |
| validation | fairness gate | **FAILED** | passed |
| test | PR-AUC | 0.351 | 0.325 |
| test | recall | 0.692 | 0.681 |
| test | precision | 0.237 | 0.245 |
| test | overall FPR | 33.3% | 31.4% |
| test | worst FPR gap | 16.7 pts | 14.2 pts |
| test | worst recall gap | 0.0 pts | 0.0 pts |
| test | fairness gate | passed | passed |

![logistic_regression_reweighed before and after](mitigation_logistic_regression_reweighed.png)

**Trade-offs.** Reweighing keeps every row and needs nothing at prediction time, but (1) it costs effective sample size, (2) a model that may no longer use the slice's denial pattern gives up any real signal in it, which shows as lower PR-AUC, and (3) it treats the symptom: if the skew comes from an unfair denial process, that process still needs fixing. Group-specific thresholds (Fairlearn `ThresholdOptimizer`) were not used: they need the group at prediction time and give randomized decisions, which breaks the model contract.

## Per-slice results: `logistic_regression_reweighed`, test split

Decision threshold 0.530. Overall: selection 36.1%, FPR 31.4%, recall 68.1%, precision 24.5%, 700 swipes with 91 denials.

### state

| Group | Swipes | Denials | Selection | FPR | Recall | Precision | Verdict |
|---|---|---|---|---|---|---|---|
| AZ | 31 | 3 | 22.6% | 17.9% | 66.7% | 28.6% | too small for FPR, recall |
| CA | 149 | 17 | 38.3% | 37.1% | 47.1% | 14.0% | too small for recall |
| CO | 15 | 1 | 33.3% | 28.6% | 100.0% | 20.0% | too small for FPR, recall |
| ID | 16 | 0 | 37.5% | 37.5% | – | 0.0% | too small for FPR, recall |
| NV | 25 | 4 | 20.0% | 14.3% | 50.0% | 40.0% | too small for FPR, recall |
| NY | 7 | 0 | 14.3% | 14.3% | – | 0.0% | too small for FPR, recall |
| OR | 347 | 57 | 39.2% | 32.1% | 75.4% | 31.6% | ok |
| TX | 20 | 0 | 35.0% | 35.0% | – | 0.0% | too small for FPR, recall |
| UT | 22 | 1 | 31.8% | 33.3% | 0.0% | 0.0% | too small for FPR, recall |
| WA | 68 | 8 | 32.4% | 26.7% | 75.0% | 27.3% | too small for recall |

Largest gap between two judged groups: selection 16.6 pts (OR vs AZ); FPR 10.4 pts (CA vs WA); precision 17.6 pts (CA vs OR).

![state](slices_state.png)

### group_size_bucket

| Group | Swipes | Denials | Selection | FPR | Recall | Precision | Verdict |
|---|---|---|---|---|---|---|---|
| 1-9 | 63 | 4 | 33.3% | 30.5% | 75.0% | 14.3% | too small for recall |
| 10-24 | 158 | 11 | 24.7% | 21.8% | 63.6% | 17.9% | too small for recall |
| 25-49 | 112 | 12 | 35.7% | 34.0% | 50.0% | 15.0% | too small for recall |
| 50-99 | 24 | 6 | 50.0% | 44.4% | 66.7% | 33.3% | too small for FPR, recall |
| 100-249 | 290 | 52 | 42.8% | 36.6% | 71.2% | 29.8% | ok |
| 250+ | 53 | 6 | 32.1% | 25.5% | 83.3% | 29.4% | too small for recall |

Largest gap between two judged groups: selection 18.1 pts (100-249 vs 10-24); FPR 14.8 pts (100-249 vs 10-24); precision 14.8 pts (25-49 vs 100-249).

![group_size_bucket](slices_group_size_bucket.png)

### age_band

| Group | Swipes | Denials | Selection | FPR | Recall | Precision | Verdict |
|---|---|---|---|---|---|---|---|
| 18-25 | 83 | 15 | 54.2% | 45.6% | 93.3% | 31.1% | too small for recall |
| 26-34 | 140 | 17 | 51.4% | 45.5% | 94.1% | 22.2% | too small for recall |
| 35-44 | 139 | 25 | 41.7% | 35.1% | 72.0% | 31.0% | too small for recall |
| 45-54 | 124 | 16 | 30.6% | 25.0% | 68.8% | 28.9% | too small for recall |
| 55-64 | 147 | 10 | 15.6% | 15.3% | 20.0% | 8.7% | too small for recall |
| 65+ | 67 | 8 | 25.4% | 27.1% | 12.5% | 5.9% | too small for recall |

Largest gap between two judged groups: selection 38.6 pts (18-25 vs 55-64); FPR 30.3 pts (18-25 vs 55-64); precision 8.9 pts (26-34 vs 18-25).

![age_band](slices_age_band.png)

### coverage_tier

| Group | Swipes | Denials | Selection | FPR | Recall | Precision | Verdict |
|---|---|---|---|---|---|---|---|
| EC | 116 | 16 | 37.1% | 31.0% | 75.0% | 27.9% | too small for recall |
| EE | 336 | 35 | 39.9% | 35.5% | 77.1% | 20.1% | ok |
| EF | 118 | 21 | 39.0% | 33.0% | 66.7% | 30.4% | too small for recall |
| ES | 130 | 19 | 23.1% | 18.9% | 47.4% | 30.0% | too small for recall |

Largest gap between two judged groups: selection 16.8 pts (EE vs ES); FPR 16.6 pts (EE vs ES); precision 10.3 pts (EE vs EF).

![coverage_tier](slices_coverage_tier.png)

### is_vip

| Group | Swipes | Denials | Selection | FPR | Recall | Precision | Verdict |
|---|---|---|---|---|---|---|---|
| False | 699 | 91 | 36.2% | 31.4% | 68.1% | 24.5% | ok |
| True | 1 | 0 | 0.0% | 0.0% | – | – | too small for FPR, recall |

![is_vip](slices_is_vip.png)

## Method

- **Decisions:** score ≥ threshold = predicted denied. Threshold = best F1 on the validation split, the same rule as training and model validation.
- **Metrics per group** (Fairlearn `MetricFrame` on each of `state`, `group_size_bucket`, `age_band`, `coverage_tier`, `is_vip`): selection rate, false-positive rate (approved swipes flagged as denials), recall (denials caught), precision, counts.
- **Gate** (`params.yaml → model_bias`): a group is flagged when it is worse off than overall by more than its metric's limit (FPR 20 pts, recall 20 pts), the metric's denominator has at least 30 rows, and a one-sided Fisher exact test against all other rows gives p < 0.01 ÷ number of tests (Bonferroni). Selection rate and precision are reported, not gated: they also move with legitimate differences in denial risk between groups.
- **Selection after bias:** candidates are ranked by validation PR-AUC; the first that passes the gate on the validation split is selected. The test split confirms the choice and is never used to make it.
- Decisions behind these rules: [docs/decisions.md](https://github.com/jaira26/nsh-data-pipeline/blob/main/docs/decisions.md) (rows on model bias).

