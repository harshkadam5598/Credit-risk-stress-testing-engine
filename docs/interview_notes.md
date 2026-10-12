# Interview preparation notes

Personal study notes, deliberately short. Keep them out of the public repository if you prefer. Numbers come from the reference run (seed 20261002) and must be re-checked after the final test run.

## 60-second summary

"I built an end-to-end credit risk prototype for a fictional UK bank. I generated a synthetic personal-loan book with a documented default process, built a logistic regression PD model with a gradient-boosting challenger, validated it with customer-grouped splits, calibration and bootstrap intervals, aggregated PD, EAD and LGD into expected loss, ran hypothetical income and debt stress scenarios, and exported tables for a three-page Power BI report. It is synthetic and simplified, and I can explain every design choice and its limits. I built it with AI assistance, reviewing and approving the decisions myself."

## Be able to explain simply

- **PD, EAD, LGD, EL.** PD is the probability of default within 12 months; EAD is the balance exposed; LGD is the share lost if default happens (flat 70% here, an assumption); EL = PD × LGD × EAD, summed over loans. It is an expected value, not a loss distribution, and not regulatory.
- **Why logistic regression.** Interpretable, outputs probabilities, standard in credit risk, and its response to stressed inputs is smooth. The gradient-boosting challenger had to beat it on criteria fixed before any model was fitted. It did not (log loss no better in any fold, AUC lower), so logistic stayed.
- **Discrimination versus calibration.** AUC measures ranking; calibration measures whether a predicted 5% means about 5%. EL needs calibration, so I report calibration slope, mean predicted versus observed rate, and fixed-bin reliability tables with intervals.
- **Leakage.** Features come from an explicit allow-list; the hidden generating variables live in a separate audit file that modelling never reads; preprocessing is fitted inside the pipeline on training data only; the split is customer-grouped because a customer's loans share hidden risk.
- **Imbalance.** About 6% defaults. I used metrics suited to it (PR-AUC, KS, capture rates) and did not resample or class-weight, because that distorts the probabilities EL depends on.
- **Cluster bootstrap.** Resample customers, keep all their loans together, 1,000 times; it reflects test-set sampling uncertainty, not retraining.
- **Stress design.** Shock only raw income and existing debt, rebuild the ratios, re-score with the unchanged model, hold credit score, EAD, LGD fixed to avoid double counting. Zero shock reproduces the baseline exactly; fixed risk-band edges make migration visible.

## Likely questions

1. **Why customer-grouped splitting?** Loans from one customer are correlated through a shared hidden effect; a random loan split would leak that and inflate performance.
2. **Why no out-of-time test?** Single-snapshot design for a clean portfolio view; it is a stated limitation, and vintage-based validation was explicitly left out of scope.
3. **Your AUC looks high. Is that realistic?** No. The simulated signal is stronger than I planned (oracle AUC 0.867, test-split oracle 0.889, against a diagnostic band of 0.65 to 0.85). My back-of-envelope estimate was wrong: two correlated burden ratios stack and debt-to-income has a heavy tail. I investigated, confirmed it was not a coding error, and kept the data rather than tune it to look more typical.
4. **How do you know the model is calibrated?** On the test set the slope is about 1.03 and mean predicted over observed is about 1.05, with intervals that include 1; two middle bins miss the diagonal in opposite directions, which is weak evidence with five non-simultaneous bins and about 750 loans each.
5. **Why are some coefficients the "wrong" sign?** Collinearity: raw debt and loan amount are negative while their ratios are positive; I left it as agreed, and quantified that those terms offset roughly 8% of the gross stress effect.
6. **Are development PDs trustworthy?** They are in-sample; only the test comparison is independent (239.5 expected versus 228 observed defaults).
7. **What do the stress results mean?** Hypothetical what-ifs from this model on synthetic data. Severe adverse (income −12%, debt +25%) raises EL by about 44%; they are not forecasts and ignore LGD stress, differing borrower impact and credit-score deterioration.
8. **What would you do next?** Out-of-time validation, tuned challengers with nested cross-validation, LGD and EAD stress, borrower-specific shocks, and a loss distribution.

## Numbers to know (reference run)

15,000 loans, 10,000 customers, EAD about £94.3m; default rate 6.08%; test AUC 0.865 (0.839 to 0.889) logistic versus 0.836 gradient boosting; baseline EL £5.12m (5.42% of EAD); adverse £5.92m (+15.7%); severe adverse £7.37m (+43.9%); sensitivity from 1.19× to 3.15× baseline EL across 0.5× to 3× the severe shock.

## Weaknesses to volunteer honestly

Synthetic data; stronger-than-intended signal; no out-of-time test; in-sample development PDs; coarse calibration evidence; flat LGD; uniform shocks; extrapolation at 3× shocks. Naming limits first is a strength.

## Describing the AI assistance

Say it plainly: Claude assisted with implementation and analysis; you set the scope, reviewed and approved the design decisions and deviations, and can explain and defend the work. Be ready to walk through any module and run it yourself.
