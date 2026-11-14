# Decisions & assumptions

The synthetic setting means nobody at NSH can confirm these. Each one is a sensible default, documented here,
kept in one place in the code, and covered by tests, so it is easy to change.

| # | Decision | Why | Where in code |
|---|---|---|---|
| 1 | `claims`, `bill_lines`, `reserves` keep **medical rows only** (missing type kept) | Contract A0 rule 6; agreed columns have no `type`, so features could not filter later | `_is_medical` in `flatten.py` |
| 2 | **Bill lines are not deduplicated** | PMPM double-counting (A8 #1) must stay visible for validation to detect | `clean_bill_lines` |
| 3 | 4-bucket **employee status** rule: COBRA + active benefit → COBRA Enrolled; terminated → Other; active benefit → Enrolled; else Not Enrolled | A5 lists the buckets but not the rules | `_employee_status` |
| 4 | Ticket **SLA = 40 wall-clock hours** | A1.6 gives `SLA_HOURS = 40`, not the hour type | `TICKET_SLA_HOURS` |
| 5 | `people.Benefits[].Premium` is **dollars** | Unit not stated in A1.2 | `MEMBER_BENEFIT_MONEY_UNITS` |
| 6 | `employer_code` = `claimReferenceId` (client WEX ID) | A3 join map | `clean_clients` |
| 7 | `nswid` keeps punctuation in last names (e.g. `o'brie1234`) | A3 formula taken literally | `make_nswid` |
| 8 | Excluded orgs matched by **id or alias** (`demo`, `nsais`) | A0 rule 2 names both | `flag_excluded_orgs` |
| 9 | Date formats **declared per field**, never guessed; all UTC; date-only values = midnight UTC | A4; avoids 3/7 vs 7/3 errors and day shifts | `dates.py` |
| 10 | Group size buckets 1-9 / 10-24 / 25-49 / 50-99 / 100-249 / 250+ | Real `GROUP_SIZE_BINS` not in the package | `GROUP_SIZE_BINS` |
| 11 | No `cards` processed table | Not one of the 9 agreed tables (plan §6.2); add if a feature needs card/OOP data | — |
| 12 | Fixed **`reference_date`**, never "today" | Reproducible runs | `params.yaml` |
| 13 | **Faker pinned** to an exact version | Faker output can change between releases | `pyproject.toml` |
| 14 | Injected issues **off by default**, separate RNG (seed + 1) | Clean default pipeline; enabling changes only injected records | `generate/inject.py` |
| 15 | Download **falls back to local generation** | Pipeline never depends on the network; same seed → identical data | `acquire_data` |
| 16 | Bias "performance" per slice = **outcome rates**: denial rate, and substantiation rate among paid debit-card swipes. Average paid is shown, never flagged | No model exists yet; paid amounts depend on the plan, not on fair treatment | `bias/metrics.py` |
| 17 | A slice is **flagged** only if worse off than overall by > 10 pts or > 1.5×, has ≥ 30 swipes, and a one-sided Fisher exact test gives p < 0.01 ÷ number of slices tested (Bonferroni) | Without the correction, clean data from 30 seeds gave 15 chance flags (1 with it). Exact test because substantiation is rare (~2.5%). Lower-than-overall rates are no burden, so not flagged | `flag_slices`, `params.yaml → bias` |
| 18 | **Proxy check:** the flag with the most excess outcomes is the root; flags on other dimensions are re-tested without the root slice and reported as proxies if they vanish | On injected data, Run Out clients and COBRA members look worse only because they sit in the skewed state | `explain_proxies` |
| 19 | **Mitigation = reweighing** (Kamiran & Calders) on the root flag's dimension; weights written to `data/bias/swipe_weights.parquet`; rows and outcomes never changed | The course requires an actual mitigation step; reweighing keeps every row and is easy to drop | `bias/mitigation.py` |
| 20 | Bias join drops excluded orgs and orphan swipes (orphans counted in the report); a missing attribute becomes an `Unknown` slice; `clients.status` is sliced as `client_status` | Orphans have no member attributes; `status` clashes with `swipes.status` | `join_swipes` |
| 21 | Model target = **swipe denied**; Pended swipes dropped; Partial counts as not denied | Pended has no decision yet; Partial was still paid | `DECIDED_STATUSES` in `modeling/features.py` |
| 22 | data-v1.1 denials depend on amount, category, deductible, tenure, recent denials and channel; **no** state/age/gender/VIP effect | v1.0 denials were random, so nothing was learnable; clean data stays fair by design | `denial_probability`, `params.yaml → denial_model` |
| 23 | **Time-based split**: train before 1 May, validation May, test from 1 June | The model predicts future swipes; a random split would leak the future into training | `params.yaml → training_data` |
| 24 | `prior_denials_90d` counts only denials on **strictly earlier dates** | Same-day or later denials would leak the outcome | `prior_denials` |
| 25 | `plan_deductible` read from the plan name's first "deductible/OOP" pair; rows without it dropped | Processed tables don't carry the deductible; the contract glossary describes this convention | `plan_deductible` |
| 26 | Members may appear in more than one split | Time-based split matches real use; a member-based split would hide seasonal change | `split_by_time` |
| 27 | **Three candidate families**: logistic regression, random forest, LightGBM, all one scikit-learn Pipeline with shared preprocessing (scale numerics, one-hot categoricals) | Guidelines ask for several model families; one Pipeline type keeps validation, bias and registry code model-agnostic | `modeling/candidates.py` |
| 28 | **Class imbalance via class weights** (`balanced` / `balanced_subsample`), not resampling | Denials are ~9-13%; weights keep every row and need no extra step at inference | `FIXED_PARAMS` |
| 29 | **Tuning = Optuna TPE, stratified 5-fold CV on the train split only**, objective mean PR-AUC | Validation is kept for selection and test for hold-out validation, so neither leaks into tuning. Stratified (not time-series) folds give stable PR-AUC on few positives; the time-based split already tests temporal generalisation | `modeling/tune.py`, `params.yaml → model` |
| 30 | **Selection metric = validation PR-AUC**; ties → CV PR-AUC → simpler model | PR-AUC is the agreed primary metric for a minority class; deterministic tie-break | `modeling/select.py` |
| 31 | Candidate metadata records metrics at the **best-F1 threshold on validation** | Needed for the confusion matrices; the official threshold is the validation step's decision | `best_f1_threshold` |
| 32 | **MLflow store = `sqlite:///mlflow.db` by default**, `MLFLOW_TRACKING_URI` overrides (Docker server) | MLflow 3 no longer accepts the plain `./mlruns` file store; SQLite needs no server for tests or CI | `modeling/tracking.py` |
| 33 | Models logged to MLflow with **cloudpickle** | MLflow's default (skops) refuses the custom `as_text` preprocessing step; same trust model as `model.joblib` (only load models this pipeline made) | `train_candidate` |
| 34 | `data_version` = the card's `data_version`, else the **release tag in the dataset URL** (e.g. `data-v1.1`) | The dataset card records the release URL, not a version field | `tracking.data_version` |
| 41 | Model-bias settings live in **`model_bias:`**, the DVC stage is **`model_bias`**, docs in `docs/model_bias.md`; mitigated models go to **`models_mitigated/`** and `bias.json → selected_model_dir` names the folder to register | `bias:` / `bias` / `docs/bias.md` belong to the phase-1 data-bias report (same reasoning as #37). DVC does not allow one stage to write inside another stage's output (`models/` belongs to `train`) | `params.yaml`, `dvc.yaml`, `fairness.py` |
| 42 | **Gate on equalized odds:** false-positive rate and recall per group are gated; selection rate and precision are reported only | FPR = approved swipes wrongly predicted as denials, recall = denials missed: the two ways the decision can hurt a member. Selection rate and precision also move when a group's true denial risk differs for a legitimate reason | `params.yaml → model_bias.max_gap_pts` |
| 43 | A group is flagged when it is **> 20 pts worse than overall**, the metric's denominator has **≥ 30 rows**, and a one-sided **Fisher exact test** gives p < 0.01 ÷ number of tests (Bonferroni) | On clean data-v1.1 every model family shows a ~17-pt FPR gap for age 18–25 caused by a documented driver (short tenure); the spurious Washington effect in the logistic regression is 30 pts, the planted CA skew 70+. 10–15 pts would block every clean model. Significance rule as #17 | `flag_disparities` |
| 44 | **Selection after bias uses the validation split**: the selected model is the best validation-PR-AUC candidate that passes the gate on validation; the test split only confirms it (`passed`) | Consistent with #35: the test split never makes a choice | `run_model_bias` |
| 45 | **Mitigation = reweighing** (Kamiran & Calders) of the train split on the **root flagged slice** (most excess bad outcomes, as #18), refit with the candidate's **tuned params, no re-tuning**; one round | Keeps the model contract (nothing needed at prediction time) and isolates the effect of the weights; reuses the phase-1 helper | `fairness_mitigation.py` |
| 46 | **No `ThresholdOptimizer`** (group-specific thresholds) | It needs the group at prediction time and gives randomized decisions, so the result is not a contract-shaped Pipeline that can be validated, registered and served | `docs/model_bias.md` |
| 47 | Every candidate's threshold (mitigated ones included) = **best F1 on validation**, recomputed | Same rule as #31 / #35; reweighing shifts the scores, so a mitigated model needs its own threshold | `assess` |
| 48 | **Exit code 1** when no candidate passes or the selected one fails on test, after writing the report | Stops `dvc repro`, CI and registration before an unfair model ships; `--no-fail` for exploratory runs | `fairness.main` |
| 49 | `.dvcignore` ignores **`__pycache__/` and `*.pyc`** | Bytecode is machine- and Python-version-specific; it changed code-dependency hashes, so every other clone saw every stage as modified | `.dvcignore` |
