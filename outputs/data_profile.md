# Data profile (Sprint 2)

All data is SYNTHETIC for a fictional bank. Figures below are computed from the generated files; default prevalence is a simulation assumption, not a market statistic. This profile reads observed fields only (including their missing values), never the hidden audit file.

## 1. Overview

| item | value |
| --- | --- |
| customers | 10,000 |
| loans | 15,000 |
| defaults (12-month horizon) | 912 |
| default rate | 6.08% |
| mean loans per customer | 1.500 |
| loans: 1 / 2 / 3 per customer | 6,000 / 3,000 / 1,000 |

## 2. Distributions

### Customer fields (observed values)

| variable | n_missing | min | mean | sd | p1 | p5 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 0 | 19.00 | 42.96 | 11.91 | 20.00 | 24.00 | 34.00 | 43.00 | 51.00 | 64.00 | 71.00 | 75.00 |
| annual_income | 0 | 12,005.32 | 33,055.99 | 15,961.05 | 12,748.39 | 14,809.72 | 21,630.53 | 29,404.56 | 40,519.10 | 62,985.81 | 87,931.10 | 147,511.20 |
| employment_tenure_months | 180 | 0.00 | 64.68 | 53.50 | 0.00 | 2.00 | 25.00 | 52.00 | 91.00 | 168.00 | 240.00 | 420.00 |
| credit_score | 151 | 442.00 | 638.06 | 54.58 | 510.00 | 549.00 | 601.00 | 638.00 | 675.00 | 729.00 | 765.00 | 871.00 |
| existing_debt | 0 | 0.00 | 7,021.90 | 9,916.40 | 0.00 | 0.00 | 467.50 | 3,940.00 | 9,232.50 | 25,322.00 | 46,170.90 | 211,910.00 |

### Loan fields and ratios (observed values)

| variable | n_missing | min | mean | sd | p1 | p5 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loan_amount | 0 | 1,000.000 | 9,041.673 | 5,574.136 | 1,800.000 | 2,800.000 | 5,000.000 | 7,600.000 | 11,600.000 | 20,300.000 | 28,500.000 | 35,000.000 |
| interest_rate | 0 | 3.900 | 12.387 | 3.510 | 4.100 | 6.500 | 10.000 | 12.400 | 14.800 | 18.100 | 20.600 | 26.200 |
| outstanding_balance | 0 | 428.180 | 6,289.569 | 4,353.623 | 1,026.465 | 1,641.005 | 3,236.062 | 5,114.505 | 8,106.092 | 14,906.270 | 21,907.979 | 33,139.230 |
| months_on_book | 0 | 1.000 | 15.690 | 11.386 | 1.000 | 2.000 | 7.000 | 13.000 | 22.000 | 39.000 | 46.000 | 47.000 |
| debt_to_income | 0 | 0.000 | 0.217 | 0.260 | 0.000 | 0.000 | 0.023 | 0.142 | 0.295 | 0.733 | 1.311 | 1.500 |
| loan_to_income | 0 | 0.024 | 0.300 | 0.177 | 0.061 | 0.093 | 0.171 | 0.258 | 0.385 | 0.667 | 0.875 | 1.000 |
| instalment_to_income | 0 | 0.007 | 0.111 | 0.074 | 0.020 | 0.031 | 0.059 | 0.092 | 0.142 | 0.260 | 0.370 | 0.574 |

### Category shares (customers for employment and region; loans for purpose and term)

| field | category | share |
| --- | --- | --- |
| employment_status | Employed_FT | 68.25% |
| employment_status | Self_employed | 15.13% |
| employment_status | Employed_PT | 12.25% |
| employment_status | Retired | 4.37% |
| region | South East | 14.52% |
| region | London | 13.04% |
| region | North West | 11.25% |
| region | West Midlands | 8.96% |
| region | East of England | 8.74% |
| region | South West | 8.46% |
| region | Scotland | 8.27% |
| region | Yorkshire and The Humber | 7.57% |
| region | East Midlands | 7.04% |
| region | Wales | 4.87% |
| region | North East | 4.14% |
| region | Northern Ireland | 3.14% |
| loan_purpose | debt_consolidation | 29.45% |
| loan_purpose | home_improvement | 20.34% |
| loan_purpose | car_purchase | 20.16% |
| loan_purpose | other | 15.25% |
| loan_purpose | major_purchase | 14.79% |
| term_months | 24 | 15.17% |
| term_months | 36 | 34.61% |
| term_months | 48 | 20.04% |
| term_months | 60 | 30.19% |

### Loans originated by year, and term by months on book

| origination_year | n_loans |
| --- | --- |
| 2020 | 1101 |
| 2021 | 2101 |
| 2022 | 4964 |
| 2023 | 6834 |

| term_months | min | median | max |
| --- | --- | --- | --- |
| 24 | 1 | 6 | 12 |
| 36 | 1 | 12 | 24 |
| 48 | 1 | 17 | 36 |
| 60 | 1 | 23 | 47 |

Months on book and term are structurally related because every loan must still be scheduled to be active at the end of 2024 (approved population rule).

### Missingness

| field | missing_rate |
| --- | --- |
| credit_score | 1.51% |
| employment_tenure_months (all customers) | 1.80% |
| employment_tenure_months (non-retired) | 1.88% |

## 3. Relationships between observed fields

### Spearman correlations (loan level, pairwise complete)

| variable | age | annual_income | credit_score | debt_to_income | loan_to_income | interest_rate | term_months | months_on_book | tenure_non_retired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1.00 | 0.23 | 0.24 | 0.01 | -0.07 | -0.21 | 0.00 | 0.02 | 0.10 |
| annual_income | 0.23 | 1.00 | 0.16 | -0.01 | -0.33 | -0.14 | -0.00 | 0.01 | 0.06 |
| credit_score | 0.24 | 0.16 | 1.00 | -0.18 | -0.05 | -0.89 | -0.00 | -0.00 | 0.17 |
| debt_to_income | 0.01 | -0.01 | -0.18 | 1.00 | 0.00 | 0.16 | 0.01 | 0.00 | -0.00 |
| loan_to_income | -0.07 | -0.33 | -0.05 | 0.00 | 1.00 | 0.05 | 0.01 | 0.00 | -0.02 |
| interest_rate | -0.21 | -0.14 | -0.89 | 0.16 | 0.05 | 1.00 | 0.13 | 0.07 | -0.15 |
| term_months | 0.00 | -0.00 | -0.00 | 0.01 | 0.01 | 0.13 | 1.00 | 0.51 | 0.01 |
| months_on_book | 0.02 | 0.01 | -0.00 | 0.00 | 0.00 | 0.07 | 0.51 | 1.00 | 0.01 |
| tenure_non_retired | 0.10 | 0.06 | 0.17 | -0.00 | -0.02 | -0.15 | 0.01 | 0.01 | 1.00 |

### Credit score versus employment tenure (required review item)

| tenure_band | n_customers | mean_score | sd_score | median_score |
| --- | --- | --- | --- | --- |
| 0-2 | 68 | 607.8 | 55.3 | 607.0 |
| 3-5 | 153 | 604.6 | 51.3 | 598.0 |
| 6-11 | 432 | 623.5 | 50.7 | 627.0 |
| 12-23 | 1135 | 626.2 | 55.2 | 626.0 |
| 24-59 | 3162 | 633.2 | 53.5 | 632.5 |
| 60-119 | 2913 | 643.5 | 53.4 | 644.0 |
| 120-239 | 1278 | 652.0 | 53.9 | 650.0 |
| 240+ | 99 | 657.6 | 48.3 | 658.0 |
| Retired (tenure not applicable) | 432 | 655.0 | 53.5 | 653.5 |

| pair (non-retired, both observed) | Spearman |
| --- | --- |
| z_t (bounded log tenure) vs credit_score | 0.179 |
| age vs credit_score | 0.240 |
| age vs tenure | 0.099 |

The bounded log tenure z_t = ln(min(max(tenure, 3), 240) / 60) is used both in the credit-score generation (weight 10) and the default formula. Tenure is also correlated with age, which also enters the score, so the table shows a marginal association, not an isolated effect.

## 4. Observed default rates by segment

Counts are shown beside every rate; small groups are noisy.

### Credit-score quintile (Q1 lowest score)

| credit_score_quintile | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| Q1 | 3019 | 530 | 17.56% |
| Q2 | 2945 | 184 | 6.25% |
| Q3 | 2969 | 105 | 3.54% |
| Q4 | 2970 | 56 | 1.89% |
| Q5 | 2869 | 28 | 0.98% |

### Debt-to-income (zero, then quartiles of positive values)

| dti_bin | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| positive Q1 | 2831 | 84 | 2.97% |
| positive Q2 | 2831 | 148 | 5.23% |
| positive Q3 | 2830 | 165 | 5.83% |
| positive Q4 | 2831 | 362 | 12.79% |
| zero | 3677 | 153 | 4.16% |

### Loan-to-income quintile

| lti_quintile | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| Q1 | 3000 | 70 | 2.33% |
| Q2 | 3000 | 89 | 2.97% |
| Q3 | 3000 | 131 | 4.37% |
| Q4 | 3000 | 174 | 5.80% |
| Q5 | 3000 | 448 | 14.93% |

### Term (months)

| term_months | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| 24 | 2275 | 164 | 7.21% |
| 36 | 5191 | 336 | 6.47% |
| 48 | 3006 | 152 | 5.06% |
| 60 | 4528 | 260 | 5.74% |

### Loan purpose

| loan_purpose | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| car_purchase | 3024 | 226 | 7.47% |
| debt_consolidation | 4418 | 280 | 6.34% |
| home_improvement | 3051 | 211 | 6.92% |
| major_purchase | 2219 | 100 | 4.51% |
| other | 2288 | 95 | 4.15% |

### Employment status

| employment_status | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| Employed_FT | 10216 | 546 | 5.34% |
| Employed_PT | 1861 | 161 | 8.65% |
| Retired | 642 | 43 | 6.70% |
| Self_employed | 2281 | 162 | 7.10% |

### Region

| region | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| East Midlands | 1057 | 76 | 7.19% |
| East of England | 1335 | 90 | 6.74% |
| London | 1956 | 114 | 5.83% |
| North East | 632 | 30 | 4.75% |
| North West | 1679 | 109 | 6.49% |
| Northern Ireland | 469 | 38 | 8.10% |
| Scotland | 1248 | 69 | 5.53% |
| South East | 2136 | 124 | 5.81% |
| South West | 1265 | 69 | 5.45% |
| Wales | 753 | 49 | 6.51% |
| West Midlands | 1368 | 90 | 6.58% |
| Yorkshire and The Humber | 1102 | 54 | 4.90% |

### Tenure band (non-retired, observed tenure)

| tenure_band | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| 0-2 | 102 | 14 | 13.73% |
| 3-5 | 231 | 37 | 16.02% |
| 6-11 | 654 | 56 | 8.56% |
| 12-23 | 1745 | 134 | 7.68% |
| 24-59 | 4838 | 295 | 6.10% |
| 60-119 | 4436 | 244 | 5.50% |
| 120-239 | 1933 | 76 | 3.93% |
| 240+ | 149 | 3 | 2.01% |

### Months-on-book band

| mob_band | n_loans | n_defaults | default_rate |
| --- | --- | --- | --- |
| 1-6 | 3692 | 235 | 6.37% |
| 7-12 | 3563 | 220 | 6.17% |
| 13-24 | 4746 | 286 | 6.03% |
| 25-36 | 1984 | 109 | 5.49% |
| 37-48 | 1015 | 62 | 6.11% |

## 5. Figures

- figures/fig1_distributions.png
- figures/fig2_relationships.png
