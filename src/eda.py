"""Compact exploratory checks on DEVELOPMENT data only (the test set is never examined).

Purpose: support the modelling choices (skew, collinearity, missingness) with evidence.
Writes outputs/eda_dev_checks.csv (long format: section, item, value, n, note).
The broader distribution and default-rate profile is already in outputs/data_profile.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import build_features
from data_split import SPLIT_FILE


def run(cfg: dict, root: Path) -> pd.DataFrame:
    proc = root / cfg["paths"]["processed_dir"]
    base = pd.read_csv(proc / "loan_analysis_base.csv")
    split = pd.read_csv(proc / SPLIT_FILE)
    X = build_features(base, cfg)              # built on the full portfolio, then subset to development loans
    dev = (split["data_split"] == "dev").to_numpy()
    Xd, yd = X[dev], base.loc[dev, "default_status"]
    num = list(cfg["features"]["numeric"])

    rows = []
    for c in num:                              # univariate discrimination and skew
        ok = Xd[c].notna()
        auc = roc_auc_score(yd[ok], Xd.loc[ok, c])
        rows.append(("univariate_auc", c, round(auc, 4), int(ok.sum()), "above 0.5 means higher value, higher default rate"))
        rows.append(("skewness", c, round(float(Xd[c].skew()), 3), int(ok.sum()), ""))
    corr = Xd[num].corr(method="spearman")
    for i, a in enumerate(num):                # collinearity review (Blueprint 4.2, D7)
        for b in num[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.8:
                rows.append(("high_correlation", f"{a} ~ {b}", round(float(corr.loc[a, b]), 3), len(Xd), "|Spearman| >= 0.8"))
    for c, mask in [("credit_score", Xd["credit_score"].isna()),
                    ("employment_tenure_months", Xd["employment_tenure_months"].isna() & (Xd["employment_tenure_months"].isna() & (base.loc[dev, "employment_status"] != "Retired")))]:
        n_miss = int(mask.sum())
        rows.append(("default_rate_when_missing", c, round(float(yd[mask].mean()), 4), n_miss,
                     f"observed-value default rate {yd[~mask].mean():.4f}"))
    out = pd.DataFrame(rows, columns=["section", "item", "value", "n", "note"])
    out.to_csv(root / cfg["paths"]["output_dir"] / "eda_dev_checks.csv", index=False)
    hc = out[out.section == "high_correlation"]
    print(f"EDA (development data, {int(dev.sum()):,} loans): wrote eda_dev_checks.csv")
    print("  high-correlation pairs:", "; ".join(f"{r.item} ({r.value})" for r in hc.itertuples()) or "none")
    return out
