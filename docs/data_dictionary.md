# Data dictionary

All data is SYNTHETIC and describes a fictional bank (Northbridge Bank). Nothing here is real customer data or a statement about the real UK population. Parameter values are simulation assumptions and live in `config/config.toml`.

**Timeline.** Prediction point T0 = 31 Dec 2023 (close of day). Outcome horizon = 1 Jan 2024 to 31 Dec 2024 inclusive (12 months). "Available at T0" means the value describes the position at or before T0 and can be used as a model input. Outcome fields are known only after T0.

**Order of generation.** Complete underlying values are generated first, then true PD and default outcomes are computed from those complete values, and only then is missingness applied to the observed fields `credit_score` and `employment_tenure_months`.

## customer_data.csv (one row per customer, 10,000 rows)

| Field | Type | Unit and allowed range | Meaning and generation logic | Available at T0 |
| --- | --- | --- | --- | --- |
| customer_id | string | C00001 to C10000, unique | Sequential identifier | Yes (identifier, never a feature) |
| age | integer | Completed years at T0, 19 to 75 | Normal(mean 42, SD 13) truncated to 18.5-75.5 and rounded. The lower bound is 19, not 18, because only age at T0 is stored and an 18-year-old cannot be guaranteed to have been an adult at any eligible origination date (see the age rule below) | Yes |
| annual_income | float | GBP per year, 12,000 to 150,000 | Log-normal by employment status (medians before truncation: full-time 31,000, part-time 20,000, self-employed 28,000, retired 18,000) with a small age effect; out-of-range draws are redrawn, not clipped | Yes |
| employment_status | category | Employed_FT, Employed_PT, Self_employed, Retired | Drawn by age band; Retired only from age 60 | Yes |
| employment_tenure_months | integer, nullable | Months in current employment, 0 to 480; Retired coded 0 (not applicable); about 2% of non-retired values missing | Gamma-distributed, capped at (age - 18) x 12. Missingness is applied last, completely at random | Yes (when observed) |
| credit_score | integer, nullable | Synthetic scale 300 to 900 (not a real bureau scale); about 1.5% missing | 640 + 45 x latent creditworthiness + 12 x standardised age + 10 x z_t + 8 x standardised log income + 25 x noise, rounded and clipped. Missingness applied last, completely at random | Yes (when observed) |
| existing_debt | integer | GBP at T0, 0 or more, multiples of 10, excluding Northbridge loans | 25% are zero; otherwise income times a log-normal share that falls with latent creditworthiness, capped at 1.5 times income | Yes |
| region | category | 12 UK regions | Illustrative weights, not population data | Yes (reporting segment; not a model feature) |

**z_t (bounded log tenure)** = ln( min( max(tenure_months, 3), 240 ) / 60 ), and 0 for Retired customers. It is used identically in the credit-score generation and in the default-risk formula.

## loan_data.csv (one row per loan, 15,000 rows)

| Field | Type | Unit and allowed range | Meaning and generation logic | Available at T0 |
| --- | --- | --- | --- | --- |
| loan_id | string | L000001 to L015000, unique | Sequential identifier | Yes (identifier) |
| customer_id | string | Must exist in customer_data | Exactly 6,000 customers hold 1 loan, 3,000 hold 2 and 1,000 hold 3, assigned randomly | Yes (identifier and split group) |
| origination_date | date | ISO yyyy-mm-dd, 2020-01-01 to 2023-11-30 | Uniform over the allowed window for the loan (see rules below) | Yes |
| loan_amount | integer | GBP, 1,000 to 35,000, multiples of 100, not above annual income | Log-normal in income and purpose; out-of-range draws redrawn | Yes |
| term_months | integer | 24, 36, 48 or 60 | Categorical draw (15/35/20/30%) | Yes |
| interest_rate | float | Fixed APR %, 3.9 to 29.9, one decimal | 12.0 - 3.2 x ((score - 640) / 55) + term premium + noise (SD 1.5), clipped. **Generated from the complete (pre-missingness) credit score of the customer, not from the observed `credit_score` field. If `credit_score` is later missing for a customer, `interest_rate` still reflects the true score. The rate therefore carries information about a missing score, as a bank's own pricing would. Pricing also uses the score at T0 rather than at origination (documented simplification S2-4).** It is never generated from hidden effects or the outcome | Yes |
| loan_purpose | category | debt_consolidation, home_improvement, car_purchase, major_purchase, other | Categorical draw (30/20/20/15/15%) | Yes |
| outstanding_balance | float | GBP at T0, greater than 0 and below loan_amount, 2 decimals | Standard annuity amortisation after the whole months elapsed to T0; APR treated as a nominal annual rate compounded monthly, no fees, no overpayments. **Derived from `interest_rate`, so it inherits the complete-score note above.** Used as exposure (EAD), not as a model feature | Yes |
| default_status | integer | 0 or 1 | 1 if the loan reaches 90 or more days past due in the horizon; drawn as Bernoulli(true PD). The target | No (outcome) |
| default_date | date, nullable | Within the horizon and on or before the scheduled end date; empty exactly when default_status = 0 | Drawn uniformly over the 366 horizon days for defaulted loans; arrears histories are not simulated | No (outcome) |

### Population, date and age rules

- **Scheduled end date** = origination_date plus term_months calendar months (day clamped to the month end if needed). It must be on or after 31 Dec 2024, so a loan maturing on 31 Dec 2024 is included and is active on every day of the horizon.
- **Latest origination** is 30 Nov 2023, so every loan has at least one instalment due on or before T0.
- **Age rule.** Only integer age in completed years at T0 is stored. A loan is valid only if origination_date is on or after 31 Dec of (2023 - (age - 18)). This guarantees the customer was at least 18 on the origination date whatever the unrecorded birthday. For age 19, 31 Dec 2022 is the earliest permitted date (30 Dec 2022 is invalid).
- **Months on book** (used in the default formula, not stored) = whole calendar months from origination to T0.
- Loans that default during the horizon stay in the data as defaults.

**Sample files.** `data/sample/customer_data_sample.csv` and `loan_data_sample.csv` have exactly the columns above and hold the first 100 customers with all of their loans. They are written by the `generate` stage (which asserts that every sampled customer keeps all of their loans) and are tracked in Git, unlike the full data. `data/sample/reporting/<table>_sample.csv` are compact samples of the ten reporting tables with exactly the schemas below; they are written by the `export` stage after the full-table contract passes, are exact copies of rows from the full outputs, and are not complete or reconcilable (see `reporting_tables.md`, section 2).

## Hidden audit file: data/audit/generation_audit.csv (never used as model input)

| Field | Meaning |
| --- | --- |
| loan_id, customer_id | Keys |
| customer_effect | Hidden customer-level log-odds term, 0.5 x (-0.6 x latent creditworthiness + 0.8 x independent factor); shared by all of a customer's loans |
| loan_effect | Hidden loan-level log-odds noise, Normal(0, 0.3) |
| true_pd_12m | Probability used to draw default_status |

`data/audit/generation_run.json` records the seed, the solved intercept and the realised default rate.

## data/processed/loan_analysis_base.csv

A joined table of the loan fields plus the customer fields (one row per loan, 15,000 rows, same columns and types as `loan_data` followed by the customer fields without `customer_id`). It contains no derived features, no train/test split label and no audit columns. Features are built by `src/features.py` at modelling time; the split is in `loan_split.csv`.

## Generation run record and diagnostic

- **Master seed:** 20261002, fixed for the whole project (set in `config/config.toml`). It was chosen because no value was supplied and was approved by the project owner. It was not selected by searching over seeds.
- **Realised portfolio:** 10,000 customers, 15,000 loans, intercept alpha = -4.806402, realised default rate 6.08% against a 6.0% target (a simulation assumption).
- **Oracle AUC diagnostic (REVIEW, accepted):** the AUC of `true_pd_12m` against the simulated outcomes is 0.8667, above the 0.65-0.85 diagnostic band. The band is a diagnostic, not a tuning target, so no parameter was changed and the data was not regenerated. The realised AUC of the observed-driver log-odds alone (the true formula without the hidden effects, using complete values) is 0.8474. This is a reference for how strong the simulated signal is, not a model result. Actual model performance will be assessed in Sprint 3 with the approved validation approach, and a fitted model should not be expected to match either figure.
- **Why it exceeded the planning estimate:** the Sprint 2 plan estimated a true log-odds SD of about 1.0-1.2 and a hidden share of variance of about a quarter. The realised values are 1.524 and 14.3%. This was a planning error in the back-of-envelope estimate, not a coding error: the log-odds recomputed from the formula match the audit file to about 1e-14. The estimate assumed the standardised instalment-to-income and loan-to-income variables had SD of about 1; they are 1.23 and 1.18, and they are correlated at 0.89, so their effects stack. Debt-to-income is also heavy-tailed (SD 0.26, 99th percentile 1.31), and its kink term amplifies the tail. Component SDs on the log-odds: credit score 0.50, debt-to-income 0.48, instalment-to-income 0.43, loan-to-income 0.24, interaction 0.22, hidden customer effect 0.50, hidden loan effect 0.30.
- **Consequence for Sprint 3:** the simulated task is easier than the original design aimed for, so model AUCs should be read with that in mind and not presented as typical of real retail credit.

## Relationships between files

- `customer_id` links `customer_data` (one row) to `loan_data` (one to three rows).
- `loan_id` links every loan-level file: `loan_data`, `loan_analysis_base`, `loan_split`, `model_predictions`, `generation_audit`, `loan_predictions` and `scenario_results`. Each is one row per loan, except `model_predictions` (two models) and `scenario_results` (three scenarios).
- `scenario_id` links `scenario_assumptions`, `scenario_results`, `portfolio_summary`.
- `model_id` links `model_predictions`, `model_metrics`, `calibration_bins`, `segment_performance` and `model_coefficients`.
- Data types below: string, integer, float, boolean. Empty cells mean suppressed or not applicable, never zero.

## Processed files and models (Sprints 3 to 5)

### data/processed/loan_split.csv (15,000 rows)

| Column | Type | Definition |
| --- | --- | --- |
| loan_id | string | Key |
| customer_id | string | Group used for the split |
| data_split | string | `dev` (development, about 75%) or `test` (held-out, about 25%). The test set is the first fold of a 4-fold customer-grouped `StratifiedGroupKFold`; written once and reused |

### data/processed/model_predictions.csv (30,000 rows: 15,000 loans × 2 models)

An intermediate file, not a reporting table.

| Column | Type | Definition |
| --- | --- | --- |
| loan_id | string | Key part |
| model_id | string | `logistic` or `gradient_boosting` |
| data_split | string | `dev_oof`: out-of-fold prediction for a development loan from the 5-fold grouped cross-validation. `test`: prediction from the model refit on all development data |
| cv_fold | integer | 1 to 5 for `dev_oof`; 0 for `test` |
| pd_12m | float | Predicted 12-month PD, 0 to 1 |

### models/*.joblib

Fitted scikit-learn pipelines (preprocessing plus estimator) for `logistic` and `gradient_boosting`, fitted on development loans only. Git-ignored; recreated by the `train` stage.

## Review and evidence outputs (outputs/)

### eda_dev_checks.csv (development data only)

| Column | Type | Definition |
| --- | --- | --- |
| section | string | `univariate_auc` (above 0.5 means higher value, higher default rate), `skewness`, `high_correlation` (absolute Spearman at least 0.8), `default_rate_when_missing` |
| item | string | Feature or pair |
| value | float | The statistic |
| n | integer | Loans used |
| note | string | Context |

### model_selection.csv

| Column | Type | Definition |
| --- | --- | --- |
| section | string | `fold_log_loss`, `fold_roc_auc` (per fold and model), `criterion` (C1, C2, A1, K1, K2), `decision` (challenger eligible) |
| item | string | Fold number, criterion text or decision text |
| model | string | Model, or "gradient_boosting vs logistic" for criteria |
| value | float | Metric value or criterion statistic |
| threshold | float | Threshold from `[model_selection]` (criteria only) |
| met | string | `True` or `False` for criteria and the decision; empty for fold rows |

### validation_summary.csv

| Column | Type | Definition |
| --- | --- | --- |
| check_id | string | Check identifier (S schema, R ranges, M missingness, K keys, C chronology, L logic, T target, A leakage, O oracle, P processed table, X reproducibility) |
| group | string | Check group |
| check | string | Description |
| status | string | `PASS`, `FAIL` (rule broken), `INFO`, or `REVIEW` (a diagnostic outside its expected band: investigate and report) |
| value | string | Observed value |
| expected | string | Expected value or rule |
| detail | string | Additional context |

## Reporting tables (outputs/)

Intended uses, relationships, labelling and Power BI guidance are in `reporting_tables.md`.

### loan_predictions.csv (one row per loan, 15,000 rows; key `loan_id`)

| Column | Type | Definition |
| --- | --- | --- |
| loan_id, customer_id | string | Keys |
| data_split | string | `dev` or `test`. Development-loan PDs are in-sample; test-loan PDs are independent |
| pd_12m | float | Baseline model PD from the final logistic model fitted on development loans (a model estimate, 0 to 1) |
| risk_band | string | `B1` (lowest PD) to `B5`; edges fixed once from the baseline PD quintiles and reused in every scenario |
| ead | float | Exposure at default: `outstanding_balance` at T0, GBP |
| lgd | float | Flat loss given default, 0.70 (synthetic assumption) |
| expected_loss | float | pd_12m × lgd × ead, GBP; illustrative |
| default_status | integer | Observed simulated outcome (a fact), 0 or 1 |
| region | string | Reporting segment (not a model feature) |
| loan_purpose | string | Purpose |
| term_months | integer | Term |
| origination_year | integer | Year of origination |
| employment_status | string | Customer employment status |
| income_band | string | `<20k`, `20-30k`, `30-45k`, `45k+` (annual income at T0) |

Segment fields always use unshocked T0 values.

### portfolio_summary.csv (key: `scenario_id`, `segment_type`, `segment_value`)

| Column | Type | Definition |
| --- | --- | --- |
| scenario_id | string | `baseline`, `adverse`, `severe_adverse` |
| segment_type | string | `portfolio` (all loans), `data_split`, `risk_band`, `loan_purpose`, `term_months`, `origination_year`, `employment_status`, `income_band`, `region`. The views repeat the same loans: filter to one type before summing |
| segment_value | string | Segment label (`all` for `portfolio`) |
| n_loans | integer | Loans in the segment |
| total_ead | float | Sum of EAD, GBP |
| expected_defaults | float | Sum of PD |
| mean_pd | float | Unweighted mean PD |
| ead_weighted_pd | float | Sum of (PD × EAD) / total EAD |
| expected_loss | float | Sum of EL, GBP |
| el_rate | float | expected_loss / total_ead |
| observed_defaults | float | Sum of simulated `default_status` (fact). Baseline rows only; empty for stressed rows |
| simulated_realised_loss | float | Sum of default_status × lgd × ead, GBP (reconciliation aid). Baseline rows only |
| result_type | string | `baseline_estimate` or `hypothetical_stress` |

41 rows per scenario (1 portfolio, 2 splits, 5 bands, 5 purposes, 4 terms, 4 origination years, 4 employment statuses, 4 income bands, 12 regions).

### scenario_assumptions.csv (21 rows; key `scenario_id`, `driver`)

| Column | Type | Definition |
| --- | --- | --- |
| scenario_id, scenario_name | string | Scenario identifier and label |
| scenario_type | string | `baseline_reference` or `hypothetical_stress` |
| driver | string | `annual_income`, `existing_debt` (shocked), and `credit_score`, `employment_status_and_tenure`, `loan_terms_and_interest_rate`, `ead`, `lgd` (held constant) |
| shock_type | string | `relative_change` or `held_constant` |
| shock_value | float | Relative change (for example −0.05 for −5%); 0 for held-constant drivers |
| description | string | Plain-language explanation |

### scenario_results.csv (45,000 rows; key `loan_id`, `scenario_id`)

| Column | Type | Definition |
| --- | --- | --- |
| loan_id, scenario_id | string | Keys |
| pd_12m | float | PD from the unchanged model after the scenario shocks |
| ead, lgd | float | Unchanged from baseline in every scenario |
| expected_loss | float | pd_12m × lgd × ead, GBP |
| risk_band | string | Band assigned with the fixed baseline edges |
| result_type | string | `baseline_estimate` or `hypothetical_stress` |

### sensitivity_results.csv (5 rows; key `multiplier`)

Hypothetical sensitivity: the severe-adverse shock vector scaled linearly (not compounded) by each multiplier, through the same calculation as the scenarios.

| Column | Type | Definition |
| --- | --- | --- |
| multiplier | float | 0.5, 1.0, 1.5, 2.0, 3.0 (1.0 reproduces the severe scenario) |
| income_shock, existing_debt_shock | float | Relative changes applied (multiplier × −0.12 and × 0.25) |
| n_loans, total_ead | integer, float | Loans and EAD (unchanged) |
| expected_defaults, mean_pd, ead_weighted_pd | float | Portfolio PD measures under the shock |
| expected_loss, el_rate | float | Portfolio EL (GBP) and EL / EAD |
| expected_loss_change_vs_baseline | float | EL minus baseline EL, GBP |
| expected_loss_ratio_to_baseline | float | EL / baseline EL |
| result_type | string | `hypothetical_sensitivity` |

### model_metrics.csv (key `model_id`, `dataset`, `metric`)

| Column | Type | Definition |
| --- | --- | --- |
| model_id | string | `logistic`, `gradient_boosting`, or `gradient_boosting_minus_logistic` (paired differences) |
| dataset | string | `cv_oof` (pooled out-of-fold development predictions) or `test` |
| metric | string | `roc_auc`, `gini`, `pr_auc`, `ks`, `brier`, `brier_skill`, `log_loss`, `mean_pred_pd`, `observed_rate`, `calibration_ratio`, `calibration_slope`, `calibration_intercept`, `capture_top10`, `capture_top20`; paired: `auc_difference`, `brier_difference`, `log_loss_difference` (challenger minus baseline). See `metric_definitions.csv` |
| value | float | Metric value |
| ci_lower, ci_upper | float | 95% customer-level bootstrap interval; empty unless status is `reported` |
| n_loans, n_defaults | integer | Loans and defaults behind the value |
| n_valid_resamples | float | Valid bootstrap resamples (empty when no bootstrap was run) |
| interval_status | string | `reported`, `not_applicable` (`cv_oof` rows have no interval), `invalid_resamples`, `not_computable`, `insufficient_data` |

### calibration_bins.csv (key `model_id`, `dataset`, `bin`)

| Column | Type | Definition |
| --- | --- | --- |
| model_id, dataset | string | Model and `test` |
| bin | integer | 1 (lowest predicted PD) to up to 5; fewer if ties force it. Edges are set once from each model's test predictions |
| n_loans, n_defaults | integer | Loans and defaults in the bin |
| mean_pred_pd | float | Mean predicted PD |
| observed_rate | float | Observed default rate (a fact) |
| ci_lower, ci_upper | float | 95% bootstrap interval for `observed_rate` |
| n_valid_resamples, interval_status | integer, string | As above |

### segment_performance.csv (key `model_id`, `segment_type`, `segment_value`)

| Column | Type | Definition |
| --- | --- | --- |
| model_id | string | The selected model (`logistic`) |
| segment_type, segment_value | string | `origination_year`, `loan_purpose`, `term_months`, `employment_status`, `income_band`, `region` and its value |
| n_loans, n_defaults | integer | Test-set counts |
| mean_pd, observed_rate | float | Predicted and observed (fact) rates |
| auc | float | Segment AUC; empty when `interval_status` is `insufficient_data` (fewer than about 1,000 loans or 30 defaults) |
| ci_lower, ci_upper | float | 95% bootstrap interval for `auc`; empty unless reported |
| n_valid_resamples, interval_status | float, string | As above |

### model_coefficients.csv (key `model_id`, `feature`)

| Column | Type | Definition |
| --- | --- | --- |
| model_id | string | `logistic` |
| feature | string | Feature after preprocessing: numeric features as named (standardised), and 0/1 dummies such as `employment_status_Retired`. Reference levels dropped: `Employed_FT` and `car_purchase` |
| coefficient | float | Log-odds change per one standard deviation (numeric) or versus the reference level (dummy) |
| odds_ratio | float | exp(coefficient). See `limitations.md` for collinearity caveats |

### metric_definitions.csv

| Column | Type | Definition |
| --- | --- | --- |
| metric | string | Metric name used in `model_metrics` |
| definition | string | Plain-language definition |
| direction | string | How to read it (higher or lower is better, ideal value) |

## Default-risk formula

The full formula, all standardisation centres and scales and the lookup offsets are in `config/config.toml` under `[default_model]` and `[transforms]`. The intercept is solved so the mean true PD equals the configured prevalence (6.0%, a simulation assumption, not a market statistic). Default outcomes are then drawn at random, so the realised rate differs slightly from the target.
