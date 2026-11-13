# Training data (model pipeline, phase 2)

Owner: Jairaghavendra Sridhar. Builds the model's training set from the phase-1 processed tables.

## Build it
```bash
dvc repro build_training_set        # runs acquire → preprocess → build_training_set
# or directly:
python -m nsh_pipeline.modeling.build --processed-dir data/processed --raw-dir data/raw --params params.yaml
```

## Output: `data/training/`
| File | Contents |
|---|---|
| `train.parquet`, `validation.parquet`, `test.parquet` | One row per decided swipe, columns = `TRAINING_COLUMNS` |
| `dataset_card.json` | Label rule, features, split dates, rows and denial rate per split, source data (release, seed, generator version, injected issues) |

## Use it
```python
from nsh_pipeline.modeling.dataset import FEATURE_COLUMNS, LABEL_COLUMN, SLICE_COLUMNS, load_split
train = load_split("train")                       # data/training/train.parquet
X, y = train[FEATURE_COLUMNS], train[LABEL_COLUMN]
```
In tests use the sample in `tests/fixtures/training/` through the `training_sample_dir` fixture.

## Task and label
**Will this card swipe be denied?** `denied` = 1 if the swipe was Denied, 0 if Approved or Partial.
Pended swipes have no decision yet and are excluded.

## Features
| Feature | Type | Source | Meaning |
|---|---|---|---|
| `amount_usd` | numeric | swipes | Swipe amount in dollars |
| `plan_deductible` | numeric | member_benefits | Deductible from the plan name's first "deductible/OOP" pair (contract glossary) |
| `tenure_days` | numeric | members | Days from hire date to the swipe |
| `prior_denials_90d` | numeric | swipes | Member's denials 1–90 days **before** this swipe |
| `service_month` | numeric | swipes | Month of service (1–12) |
| `service_category` | categorical | swipes | Medical, Drugs & Medicine, Dental, Vision |
| `coverage_tier` | categorical | members | EE, ES, EC, EF |
| `carrier` | categorical | member_benefits | Plan carrier |
| `age_band` | categorical | members | Age band at the reference date |
| `group_size_bucket` | categorical | clients | Employer size bucket |
| `state` | categorical | clients | Employer state |
| `is_debit_card` | categorical | swipes | Card swipe vs manual claim |

Slice columns for bias checks: `state`, `group_size_bucket`, `age_band`, `coverage_tier`, `is_vip`.

## Split (by service date)
| Split | Dates | Rows (small scale, data-v1.1) | Denied |
|---|---|---|---|
| train | Jan 1 – Apr 30 | 2,963 | 9.2% |
| validation | May | 1,044 | 13.1% |
| test | Jun 1 – 15 | 700 | 13.0% |

Dates are set in `params.yaml → training_data`. The denial rate rises over the year because recent
denials compound; validation and test behave alike, so a threshold tuned on May transfers to June.

## Leakage safeguards
- Split by **time**: the model is trained on the past and judged on the most recent, unseen period.
- `prior_denials_90d` counts only denials on **earlier dates** (never the same day or later).
- No feature uses the outcome or anything recorded after the swipe (paid/denied amounts, tickets).
- Members may appear in several splits (a time-based split, not member-based); documented trade-off.

## Where the signal comes from (data-v1.1)
Denials are generated from documented causes in `params.yaml → denial_model`: higher amounts, Dental and
Vision, high-deductible plans, new members (< 180 days), recent denials and manual claims raise the chance;
Drugs & Medicine lowers it. **State, age, gender and VIP have no effect**, so the clean data is fair by
design; the only bias is the one planted by `inject_issues.denial_skew`.

## Tests
`tests/modeling/`: contract, sample set, features (joins, label, leakage-safe history, orphans),
split (boundaries, order, card) and the build stage end to end.