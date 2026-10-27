# Bias detection & mitigation (M5)

## What it does
`python -m nsh_pipeline.bias.slicing --processed-dir data/processed --params params.yaml` reads the processed
`swipes`, `clients` and `members` tables, joins each swipe to its client and member, and compares outcomes across
subgroups (data slicing with Fairlearn's `MetricFrame`):

| Slice by | From |
|---|---|
| `state`, `group_size_bucket`, `client_status` | `clients` (NS Demo and NSAI removed) |
| `age_band`, `is_vip`, `coverage_tier`, `employee_status`, `gender` | `members` |

Per slice: **denial rate**, **substantiation rate** (paid debit-card swipes) and average paid per swipe. A slice is
flagged when it is worse off than overall by more than 10 points or 1.5×, has at least 30 swipes, and the gap is
significant after a Bonferroni correction. Flags that only appear because their swipes sit inside a stronger flag are
reported as proxies. The strongest flag is mitigated by **reweighing**. Every threshold is in `params.yaml → bias`;
the reasoning is in [decisions #16–#20](decisions.md).

| Output | Contents |
|---|---|
| `reports/bias_report.md` | Summary, method, flags, all slices, interpretation, mitigation and trade-offs, assumptions |
| `reports/bias/slices.csv` | Every slice with rate, gap, p-value and verdict |
| `reports/bias/*.png` | Rate chart for the flagged dimension; before/after reweighing chart |
| `data/bias/swipe_weights.parquet` | `swipe_id`, `weight` for any model trained on swipes (`sample_weight`) |

The stage logs a warning per flag and never fails the pipeline: bias is a finding to explain, not a broken run.

## Evidence
Both reports below were produced by the command above on the published `small` dataset (seed 42, as of 2025-06-15).

| Data | Result | Report |
|---|---|---|
| Clean (default `params.yaml`) | No slice flagged; no reweighing needed | [clean/bias_report.md](bias/clean/bias_report.md) |
| Injected (`inject_issues.enabled: true`) | Only **state = CA** flagged: 30.1% denied vs 5.3% elsewhere, matching the planted `denial_skew`; reweighing closes the 27.6-pt gap | [injected/bias_report.md](bias/injected/bias_report.md) |

![Denial rate by state, injected data](bias/injected/bias/denial_rate_by_state.png)

To check the rule is not tuned to one dataset, it was run on 30 seeds: the planted state was found 30/30 times with
no other unexplained flag, and clean data gave 1 chance flag in 30 runs.

### Reproduce the injected run
```bash
# params_injected.yaml: a copy of params.yaml with inject_issues.enabled: true (keep it outside the repo)
python -m nsh_pipeline.generate.run --params params_injected.yaml --out-dir data/raw_injected
python -m nsh_pipeline.preprocess.run --raw-dir data/raw_injected --out-dir data/processed_injected --as-of 2025-06-15
python -m nsh_pipeline.bias.slicing --processed-dir data/processed_injected --params params_injected.yaml \
    --report-dir reports_injected --out-dir data/bias_injected
```

## Tests
`pytest tests/bias` (about 5 s) covers the join, the flag rules, the proxy check, reweighing, the report, the CLI,
and both answer-key cases: clean data flags nothing, injected data flags exactly the planted state.
