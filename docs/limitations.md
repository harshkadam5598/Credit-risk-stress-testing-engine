# Limitations and caveats

Read these before interpreting any result. Figures marked "reference run" come from the run with master seed 20261002 and should be re-checked after regenerating.

## 1. Synthetic data

- All data comes from a seeded simulation of a fictional bank. Distributions, relationships, the 6% default prevalence, the 70% LGD and every coefficient are assumptions, not estimates from real data or market statistics.
- The model learns the simulation's rules. A good score shows it recovered a designed relationship, not that it would work on real borrowers.
- Region effects are twelve small arbitrary offsets and make no statement about real regional risk. Region weights are illustrative, not population data.
- Results are one realisation from a single seed. Variation across seeds has not been measured.

## 2. Accepted oracle AUC deviation

The oracle AUC (true PD against the simulated outcomes) is **0.8667** on the full portfolio and **0.8885** on the test split, above the 0.65 to 0.85 diagnostic band chosen at planning time. The band is a diagnostic, not a tuning target. The project owner reviewed the investigation and accepted the result: the data-generating process, configuration and generated data were kept unchanged.

- **Cause: a planning error, not a coding error.** The planning estimate assumed a true log-odds SD of about 1.0 to 1.2 and a hidden share of variance of about a quarter. The reference run gives 1.524 and 14.3%. The estimate assumed the standardised instalment-to-income and loan-to-income variables had an SD near 1; they are 1.23 and 1.18 and correlate at 0.89, so their effects stack. Debt-to-income is heavy-tailed (99th percentile 1.31) and its kink amplifies the tail. The log-odds recomputed from the formula match the audit file to about 1e-14.
- **Consequence:** the simulated task is easier than intended. The AUC of the observed-driver log-odds alone is 0.8474, and the logistic model's test AUC in the reference run (0.8647, 95% interval 0.8385 to 0.8891) sits close to the oracle value. These scores are **not typical of real retail credit** and must not be presented as such.
- A fitted model is not expected to match the oracle. On any one sample it can fall above or below by sampling noise, and a model that clearly exceeded it would signal leakage or a generator flaw.

## 3. Cross-sectional design and no out-of-time validation

- There is one snapshot (T0 = 31 December 2023) and one 12-month outcome window. Development and test loans share the same period, so results cannot show how the model would hold up in a later period or economic regime.
- Vintage-based or out-of-time validation is out of scope for version 1. Origination year is shown only as a descriptive segment.
- Validation is customer-disjoint, which prevents the same borrower informing both sides, but it is not a substitute for out-of-time testing.

## 4. Model assumptions and coefficient observations

Logistic regression assumes linear effects on the log-odds for the features supplied, and the features were fixed by approval. Observations from the reference run (standardised coefficients, `outputs/model_coefficients.csv`):

- **Collinearity.** Highly correlated pairs on development data: existing debt with debt-to-income (0.948), loan-to-income with instalment-to-income (0.912), and credit score with interest rate (−0.886). Interest rate is generated partly from credit score.
- **Counter-intuitive signs.** The raw levels `existing_debt` (−0.051) and `loan_amount` (−0.090) are slightly negative while their ratios are positive (debt-to-income 0.526, loan-to-income 0.256, instalment-to-income 0.568). `log_annual_income` is positive (0.069) although income is protective in the simulation. These are the usual artefacts of correlated inputs, not behavioural findings. Interest rate is positive (0.180) alongside a strongly negative credit score (−0.892).
- **Retired.** The coefficient for Retired is positive (0.202) although the simulation gives retired customers a small negative offset. Retired tenure is coded 0 (not applicable) and the model is linear in raw tenure, so the status term absorbs part of that.
- **Age.** The age coefficient is near zero (0.001) because the simulation's weak U-shape cannot be captured by a linear term.
- None of these were corrected, by decision: raw features and the tenure treatment were left unchanged. They affect interpretation of individual coefficients and of stress results (see section 7), not the headline validation logic.
- The gradient-boosting challenger uses fixed, conservative settings and was not tuned, so conclusions about it apply only to that configuration.
- Model selection rests on five folds with about 135 defaults each, overlapping training sets, one partition and judgement-based thresholds.

## 5. Evidence strength

- The test set has 3,750 loans and 228 defaults (reference run). Five calibration bins hold about 750 loans each, intervals are wide, and the bin intervals are not simultaneous, so one bin missing the diagonal among five is not unusual. A bin interval containing the diagonal means no detectable miscalibration, not proof of good calibration.
- Bootstrap intervals describe sampling uncertainty of the fixed test set only. They exclude variation from retraining and model selection and the simulation's own assumptions.
- Segment AUCs need about 1,000 loans and 30 defaults. In the reference run only 8 of 33 segments qualify; the rest, including all regions, are reported as `insufficient_data`.
- Development-loan PDs are in-sample (the final model is fitted on development loans and scores all loans). Only the comparison of expected and observed defaults on test loans is independent evidence.

## 6. Portfolio and loss assumptions

- EAD is the outstanding balance at T0, with no amortisation, drawdown or accrued interest over the horizon.
- LGD is a flat, synthetic 70%, with no recovery timing, costs or discounting. EL is exactly linear in LGD.
- EL is a sum of expected values. There is no default correlation, loss distribution, unexpected loss, capital, staging or lifetime expected credit loss.
- Defaults are drawn directly with a date in the horizon; monthly arrears histories are not simulated. A loan maturing on 31 December 2024 can therefore be given a default date late in December.

## 7. Stress-testing limitations

All stress and sensitivity results are **hypothetical what-if outputs, not forecasts**, and must be labelled as such wherever shown.

- **Fixed EAD and LGD.** Both are held constant, so stressed changes come only from PD. Real stress would also raise LGD and could change exposure.
- **Uniform shocks.** Every borrower receives the same relative change in income and existing debt. In reality shocks differ by employment type and circumstance (for example self-employed borrowers), and no unemployment mechanism is modelled.
- **Credit score is not shocked.** This avoids double counting but probably understates stress, because real scores would deteriorate.
- **Loans are fixed-rate and fixed-term**, so rates, terms and balances are unchanged.
- **Scenario magnitudes** are illustrative and not calibrated to any regulator's scenario.
- **The response is the model's, not the economy's.** The model learned cross-sectional associations from synthetic data. Its response to shocked income and debt reflects those associations and the simulation design. In the reference run the effect runs mainly through debt-to-income, instalment-to-income and loan-to-income; the counter-intuitive raw-level coefficients offset about 8% of the gross effect.
- **Extreme sensitivity.** The multipliers (0.5 to 3.0 times the severe vector) are the approved range, but at 3.0 the shocks are income −36% and debt +75%. Shocked incomes can then fall below the 12,000 floor of the generated data and shocked ratios can exceed anything the model saw in training, so behaviour there is less reliable and the convex EL growth should be read as the model's extrapolation. The multiplier scales the relative change linearly, not compounded.
- **No loss distribution.** Scenario EL is a sum of per-loan expectations and says nothing about the spread of outcomes.

## 8. Implementation notes affecting interpretation

- Customer fields and interest-rate pricing use T0 values, while loan terms are fixed at origination. In reality pricing uses the score at origination.
- `interest_rate` is generated from the complete credit score, so it carries information about a score that is later set to missing, as a bank's own pricing would.
- Months on book and term are structurally correlated, because every loan must still be scheduled to be active at the end of 2024.
- Missing values are completely at random and imputed by the median inside the model pipeline.
- Stage outputs depend on each other; re-run from the changed stage onward (see the README).
- Reproducibility is verified only for the pinned library versions on Python 3.12.3 (Linux). The gradient-boosting challenger's predictions and metrics are sensitive to the scikit-learn version (a newer release gave different challenger figures and no change to the logistic, portfolio or scenario results), so the exact pins in `requirements.txt` are required to match the documented challenger values. Other operating systems and Python versions are untested.

## 9. Intended use

This is a teaching and portfolio prototype. It must not be used for real lending, provisioning, capital or regulatory purposes. It was developed as an independent portfolio project with AI assistance and is not affiliated with any bank.
