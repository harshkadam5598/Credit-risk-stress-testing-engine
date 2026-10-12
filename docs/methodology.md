# Methodology

This document describes what each stage does and why, as implemented. Parameter values live in `config/config.toml` and `config/scenarios.toml`; this document explains them without repeating every number. All data is synthetic and the fictional bank is Northbridge Bank.

## 1. Problem framing

| Element | Definition |
| --- | --- |
| Unit of prediction | One performing personal loan |
| Prediction point (T0) | 31 December 2023; all features describe the position at or before T0 |
| Horizon | 1 January to 31 December 2024 (12 months), inclusive |
| Default event | The loan reaches 90 or more days past due in the horizon |
| Target | `default_status` = 1 if the default event occurs in the horizon, else 0 |
| Population | Loans not in default at T0 whose scheduled end date is on or after 31 December 2024; loans that default in the horizon stay in the data as defaults |

The 12-month horizon and 90-days-past-due definition follow common retail-credit conventions. A single snapshot was chosen because it matches a portfolio view and keeps exposure and scenarios simple; the cost is that there is no out-of-time test.

## 2. Synthetic data generation (`src/generate_data.py`)

**Order of generation.** (1) complete customer attributes; (2) loan allocation and complete loan attributes; (3) derived quantities, log-odds, intercept, true PD and default outcomes; (4) the hidden audit file; (5) missingness on `credit_score` and `employment_tenure_months`, applied last; (6) CSV files. True PDs and outcomes therefore use complete values.

**Customers and loans.** 10,000 customers; exactly 6,000 hold one loan, 3,000 hold two and 1,000 hold three (15,000 loans). Age, employment status, income, tenure, a latent creditworthiness, credit score, existing debt and region are drawn from the distributions in the config, with correlations (income with age and status, credit score with age, tenure, income and debt). Out-of-range draws are redrawn, not clipped. Loan purpose, term, amount, origination date and interest rate follow. The interest rate is priced from the complete credit score, never from hidden effects or the outcome. The outstanding balance is the standard annuity balance at T0 (APR as a nominal annual rate compounded monthly, no fees, no overpayments).

**Chronology rules.** A loan is included only if its scheduled end date (origination plus term in calendar months, clamped to month end) is on or after 31 December 2024. The latest origination is 30 November 2023. A conservative age rule guarantees the customer was at least 18 at origination, using only age in completed years at T0: origination must be on or after 31 December of (2023 − (age − 18)). This is why age starts at 19.

**Default model.** The log-odds of default are the sum of a fixed intercept and the terms in `[default_model]`: credit score (linear and a small quadratic), debt-to-income (linear plus a steeper term above 0.5), instalment-to-income, loan-to-income, a low-score by high-loan-to-income interaction, employment, term, purpose and region offsets, bounded log tenure, a weak age U-shape, log income, a months-on-book bell, and two hidden effects. Standardisation uses fixed centres and scales from the config, not sample statistics. Bounded log tenure is `ln(min(max(tenure, 3), 240) / 60)`, with Retired customers set to 0, and is used identically in the credit-score generation and the default formula. The hidden customer effect (SD 0.5, shared by all of a customer's loans) is partly linked to the latent creditworthiness, so the observed score is informative but incomplete; the hidden loan effect has SD 0.3. The intercept is solved numerically so that mean true PD equals the 6.0% target prevalence, a simulation assumption; outcomes are then drawn as Bernoulli(true PD), so the realised rate differs slightly.

**Diagnostics.** True PDs are written to a separate audit file that modelling code never reads. The oracle AUC (true PD against simulated outcomes) is computed only as a diagnostic of how hard the simulated task is. See `docs/limitations.md` for the accepted deviation.

## 3. Features and preprocessing (`src/features.py`)

The feature set is an explicit allow-list in the config. Numeric: age, log annual income, employment tenure, credit score, existing debt, loan amount, term, interest rate, loan-to-income, debt-to-income, instalment-to-income, months on book, and customer active loans (number of the customer's loans in the portfolio). Categorical: employment status and loan purpose. Excluded by design: identifiers, outcomes, `outstanding_balance` (used only as EAD), `region` (a reporting segment only), raw annual income (replaced by its log) and anything in the audit file. A leakage guard raises if a forbidden column reaches a model.

`build_features` is deterministic and row-wise from T0 information, so it can be re-applied to shocked inputs in the stress stage. It is run on the full portfolio and then subset, because `customer_active_loans` is a count over the frame it is given.

Preprocessing is fitted inside scikit-learn pipelines on training data only (including within each cross-validation fold): median imputation for missing numeric values, standardisation for the logistic model only, and one-hot encoding with the first category dropped and unseen categories handled safely. No resampling or class weights are used, because they distort probabilities and PDs must stay calibrated for expected-loss use.

## 4. Train/test split and cross-validation (`src/data_split.py`, `src/train.py`)

**Test hold-out.** `StratifiedGroupKFold` has no test-size setting, so four folds are created with loan-level `default_status` as the target and `customer_id` as the group, and the first fold becomes the test set (about 25%). Whole customers go to one side. Three checks run and the pipeline stops if one fails: no customer on both sides; test share of loans within 25% ± 2 points; test default rate within 0.5 points of the overall rate. There is no seed searching. The label is written once to `data/processed/loan_split.csv`.

**Development cross-validation.** Five customer-grouped, stratified folds on development data, the same folds for both models, with fixed model settings and no tuning in version 1.

## 5. Models and selection (`src/features.py`, `src/train.py`)

| Role | Model | Fixed settings |
| --- | --- | --- |
| Baseline | Logistic regression (L2) on standardised inputs | C = 1.0, up to 1,000 iterations |
| Challenger | Histogram gradient boosting | depth 3, learning rate 0.05, 300 iterations, minimum leaf 50, L2 1.0, no early stopping |

**Selection rule** (thresholds fixed in `[model_selection]` before any model was fitted). Logistic regression is the default. The challenger is only reported as eligible if all five criteria hold, with the log-loss difference per fold defined as baseline minus challenger:

| ID | Criterion |
| --- | --- |
| C1 | Challenger log loss is better in at least 4 of 5 folds |
| C2 | Mean log-loss improvement is at least 1% of the baseline's mean log loss |
| A1 | Challenger mean fold AUC is not lower than the baseline's by more than 0.005 |
| K1 | On pooled out-of-fold predictions, the challenger's distance of calibration slope from 1 exceeds the baseline's by no more than 0.10 |
| K2 | On pooled out-of-fold predictions, the challenger's distance of (mean predicted PD / observed rate) from 1 exceeds the baseline's by no more than 0.05 |

Eligibility is a recommendation only. The model used downstream is `modelling.selected_model` in the config, which the owner controls, and the test set plays no part in selection. Both models are then refit on all development data and the test set is predicted once. Fold-based comparison is coarse: five folds, overlapping training sets, about 135 defaults per fold, fixed settings, a single partition and judgement-based thresholds.

## 6. Validation and calibration (`src/evaluate.py`)

**Metrics.** ROC-AUC and Gini, PR-AUC, KS, Brier score and Brier skill (against always predicting the observed rate), log loss, mean predicted PD versus observed rate (calibration-in-the-large), calibration slope and intercept (logistic regression of outcomes on the log-odds of predicted PD), and the share of defaults captured in the riskiest 10% and 20% of loans. Definitions are in `outputs/metric_definitions.csv`. Accuracy is not a headline metric.

**Intervals: customer-level cluster bootstrap.** Customers are resampled with replacement and all of a sampled customer's loans stay together; a customer drawn k times gives all of their loans weight k (equivalent to duplication). 1,000 resamples are generated once from a fixed stream and reused for every metric, bin, segment and model. The model is not refitted. Intervals are 95% percentile intervals and describe sampling uncertainty of the fixed test set only. An interval is reported only if at least 95% of resamples are valid; otherwise it is suppressed and flagged.

**Calibration bins.** Five quantile bins of predicted PD on the test set, with edges set once from each model's original predictions and reused in every resample. Edges are unique quantiles and bins are right-closed, so tied predictions are never separated and fewer than five bins are kept if ties require. Per bin: observed default rate with its interval, and mean predicted PD.

**Segment stability.** On the test set only, from the same resamples. A segment reports AUC and its interval only if it has about 1,000 loans and 30 defaults, decided once on the original test data; otherwise counts and observed versus predicted PD are shown and the status is `insufficient_data`.

**Paired comparison.** Challenger minus baseline on the same resampled customers: AUC difference (positive favours the challenger), Brier and log-loss differences (negative favours the challenger). Models are not compared by overlap of separate intervals.

## 7. Portfolio risk (`src/portfolio.py`)

| Quantity | Definition |
| --- | --- |
| PD | Model-estimated probability of default within 12 months after T0 |
| EAD | `outstanding_balance` at T0; no amortisation, drawdown or accrued interest over the horizon |
| LGD | Flat 70% of EAD, a synthetic assumption; no recovery timing, costs or discounting |
| EL | PD × LGD × EAD per loan; portfolio EL is the sum |
| EL rate | Portfolio EL divided by total EAD |
| EAD-weighted PD | Sum of (PD × EAD) divided by total EAD |
| Simulated realised loss | Sum of default_status × LGD × EAD (baseline only; a reconciliation aid) |

**Scoring.** The selected final model, fitted on development loans only, scores all 15,000 loans. Development-loan PDs are therefore in-sample, and only the test-loan comparison of expected versus observed defaults is independent evidence. A diagnostic compares development-loan EL from in-sample PDs with EL from out-of-fold PDs (a 5% difference is a review trigger, not an automatic switch). If the gradient-boosting challenger were ever selected, cross-fitted scoring would be required; it is not implemented and the code blocks that selection until the owner approves.

**Risk bands.** Five bands from the quintiles of the baseline PD distribution (B1 lowest). The edges are fixed once and reused in every scenario, so loans visibly migrate between bands under stress.

**Aggregation and checks.** Aggregates are produced for the portfolio, development/test split, risk band, purpose, term, origination year, employment status, income band and region. A reconciliation function checks loan-level definitions (EL = PD × LGD × EAD, EAD equals the balance, PD in (0, 1)) and that every segment view sums to the portfolio totals. Rates are always ratios of sums, never averages of segment rates.

## 8. Stress testing (`src/scenarios.py`, `config/scenarios.toml`)

Scenarios are hypothetical what-if exercises, not forecasts. Magnitudes are illustrative and not calibrated to any regulator's scenario.

| Scenario | Annual income | Existing debt | Held constant |
| --- | --- | --- | --- |
| Baseline | 0% | 0% | everything |
| Adverse | −5% | +10% | credit score, employment, loan terms, interest rate, EAD, LGD |
| Severe adverse | −12% | +25% | credit score, employment, loan terms, interest rate, EAD, LGD |

**Calculation.** For each scenario: (1) apply the relative shocks to raw annual income and existing debt; (2) rebuild the features with the same `build_features` code, so ratios are recomputed from the shocked inputs; (3) re-score with the unchanged logistic model; (4) compute EL with unchanged EAD and LGD; (5) assign risk bands with the fixed edges; (6) aggregate using the unshocked T0 segment attributes. No PD multiplier is added.

**Double-counting rules.** Credit score is not shocked (a bureau score already reflects income and debt behaviour; this probably understates stress). Employment and loan terms are unchanged, and interest rates are unchanged because the loans are fixed-rate.

**Sensitivity.** The severe-adverse shock vector scaled by 0.5, 1.0, 1.5, 2.0 and 3.0 (the relative change is scaled linearly, not compounded), through the same calculation. Results are labelled hypothetical sensitivity. The 1.0 multiplier reproduces the severe scenario exactly.

**Consistency checks executed by the stage.** Zero-shock PDs equal baseline PDs exactly (and match the persisted file within CSV rounding); EAD, LGD and credit score are unchanged; only income, debt and their derived ratios change; fixed band edges are used; every scenario reconciles to its aggregates; portfolio EL increases with severity; the 1.0 multiplier equals the severe scenario.

## 9. Reporting tables (`src/export_reporting.py`)

Stages write the final tables directly. The export step verifies exact columns, allowed values, row counts, reconciliation between loan-level and aggregate tables, that stressed rows carry no observed outcomes, and that suppressed intervals are empty rather than zero. See `docs/reporting_tables.md`.
