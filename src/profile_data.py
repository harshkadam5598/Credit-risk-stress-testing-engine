"""Data profile for owner review: distributions, relationships and default rates.

Reads only the observed raw files (customer_data.csv, loan_data.csv). It never reads the
hidden audit file. Writes outputs/data_profile.md and two figures in outputs/figures/.
All numbers are computed from the generated synthetic data; the text adds no findings.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from generate_data import load_config  # noqa: E402

TENURE_BANDS = [(3, 5, "3-5"), (6, 11, "6-11"), (12, 23, "12-23"), (24, 59, "24-59"),
                (60, 119, "60-119"), (120, 239, "120-239"), (240, 10**6, "240+")]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else floatfmt.format(x)
        return str(x)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def tenure_band(months: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=months.index, dtype="object")
    for lo, hi, label in TENURE_BANDS:
        out[(months >= lo) & (months <= hi)] = label
    # tenure of 0-2 months sits below the 3-month floor used by the risk formula
    out[months < 3] = "0-2"
    return out


def quantile_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    q = df[cols].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    q.columns = ["p1", "p5", "p25", "median", "p75", "p95", "p99"]
    q.insert(0, "mean", df[cols].mean())
    q.insert(1, "sd", df[cols].std())
    q.insert(0, "min", df[cols].min())
    q["max"] = df[cols].max()
    q.insert(0, "n_missing", df[cols].isna().sum())
    return q.reset_index(names="variable")


def default_rate_table(df: pd.DataFrame, group: str, label: str) -> pd.DataFrame:
    g = df.groupby(group, observed=True)["default_status"].agg(n_loans="count", n_defaults="sum", default_rate="mean")
    out = g.reset_index().rename(columns={group: label})
    out["n_loans"] = out["n_loans"].astype(int)
    out["n_defaults"] = out["n_defaults"].astype(int)
    return out


def run(config_path: str | Path, root: Path) -> None:
    cfg = load_config(config_path)
    raw = root / cfg["paths"]["raw_dir"]
    out_dir = root / cfg["paths"]["output_dir"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = cfg["dates"]["t0"]

    cust = pd.read_csv(raw / "customer_data.csv")
    loans = pd.read_csv(raw / "loan_data.csv")
    orig = pd.to_datetime(loans["origination_date"])
    loans["months_on_book"] = (t0.year - orig.dt.year) * 12 + (t0.month - orig.dt.month) - (orig.dt.day > t0.day).astype(int)
    d = loans.merge(cust, on="customer_id", how="left")
    d["debt_to_income"] = d["existing_debt"] / d["annual_income"]
    d["loan_to_income"] = d["loan_amount"] / d["annual_income"]
    r = d["interest_rate"] / 1200.0
    pay = d["loan_amount"] * r / (1 - (1 + r) ** (-d["term_months"]))
    d["instalment_to_income"] = 12 * pay / d["annual_income"]
    d["is_retired"] = d["employment_status"] == "Retired"

    parts: list[str] = []
    parts.append("# Data profile (Sprint 2)\n")
    parts.append("All data is SYNTHETIC for a fictional bank. Figures below are computed from the generated files; "
                 "default prevalence is a simulation assumption, not a market statistic. This profile reads observed "
                 "fields only (including their missing values), never the hidden audit file.\n")

    # ---- overview
    parts.append("## 1. Overview\n")
    overview = pd.DataFrame({
        "item": ["customers", "loans", "defaults (12-month horizon)", "default rate", "mean loans per customer",
                 "loans: 1 / 2 / 3 per customer"],
        "value": [f"{len(cust):,}", f"{len(loans):,}", f"{int(loans['default_status'].sum()):,}",
                  f"{loans['default_status'].mean():.2%}", f"{len(loans) / len(cust):.3f}",
                  " / ".join(f"{int((loans.groupby('customer_id').size() == k).sum()):,}" for k in (1, 2, 3))]})
    parts.append(md_table(overview) + "\n")

    # ---- distributions
    parts.append("## 2. Distributions\n")
    parts.append("### Customer fields (observed values)\n")
    parts.append(md_table(quantile_table(cust, ["age", "annual_income", "employment_tenure_months", "credit_score", "existing_debt"]), "{:,.2f}") + "\n")
    parts.append("### Loan fields and ratios (observed values)\n")
    parts.append(md_table(quantile_table(d, ["loan_amount", "interest_rate", "outstanding_balance", "months_on_book",
                                             "debt_to_income", "loan_to_income", "instalment_to_income"]), "{:,.3f}") + "\n")
    shares = []
    for col in ["employment_status", "region", "loan_purpose", "term_months"]:
        base = cust if col in ("employment_status", "region") else loans
        s = base[col].value_counts(normalize=True)
        s = s.sort_index() if col == "term_months" else s
        shares.append(pd.DataFrame({"field": col, "category": s.index.astype(str), "share": s.values}))
    parts.append("### Category shares (customers for employment and region; loans for purpose and term)\n")
    parts.append(md_table(pd.concat(shares), "{:.2%}") + "\n")
    parts.append("### Loans originated by year, and term by months on book\n")
    by_year = d.assign(origination_year=orig.dt.year).groupby("origination_year").size().rename("n_loans").reset_index()
    parts.append(md_table(by_year) + "\n")
    mob = d.groupby("term_months")["months_on_book"].agg(["min", "median", "max"]).reset_index()
    parts.append(md_table(mob, "{:.0f}") + "\n")
    parts.append("Months on book and term are structurally related because every loan must still be scheduled "
                 "to be active at the end of 2024 (approved population rule).\n")

    miss = pd.DataFrame({"field": ["credit_score", "employment_tenure_months (all customers)",
                                   "employment_tenure_months (non-retired)"],
                         "missing_rate": [cust["credit_score"].isna().mean(), cust["employment_tenure_months"].isna().mean(),
                                          cust.loc[cust["employment_status"] != "Retired", "employment_tenure_months"].isna().mean()]})
    parts.append("### Missingness\n")
    parts.append(md_table(miss, "{:.2%}") + "\n")

    # ---- relationships
    parts.append("## 3. Relationships between observed fields\n")
    num = d[["age", "annual_income", "credit_score", "debt_to_income", "loan_to_income", "interest_rate",
             "term_months", "months_on_book"]].copy()
    num["tenure_non_retired"] = d["employment_tenure_months"].where(~d["is_retired"])
    corr = num.corr(method="spearman").round(2)
    parts.append("### Spearman correlations (loan level, pairwise complete)\n")
    parts.append(md_table(corr.reset_index(names="variable"), "{:.2f}") + "\n")

    parts.append("### Credit score versus employment tenure (required review item)\n")
    nr = cust[(cust["employment_status"] != "Retired")].dropna(subset=["credit_score", "employment_tenure_months"]).copy()
    nr["tenure_band"] = tenure_band(nr["employment_tenure_months"])
    order = ["0-2"] + [b[2] for b in TENURE_BANDS]
    st = nr.groupby("tenure_band")["credit_score"].agg(n_customers="count", mean_score="mean", sd_score="std",
                                                       median_score="median").reindex(order).reset_index()
    ret = cust[cust["employment_status"] == "Retired"].dropna(subset=["credit_score"])
    st = pd.concat([st, pd.DataFrame({"tenure_band": ["Retired (tenure not applicable)"], "n_customers": [len(ret)],
                                      "mean_score": [ret["credit_score"].mean()], "sd_score": [ret["credit_score"].std()],
                                      "median_score": [ret["credit_score"].median()]})])
    st["n_customers"] = st["n_customers"].astype(int)
    parts.append(md_table(st, "{:.1f}") + "\n")
    z = np.log(np.clip(nr["employment_tenure_months"], 3, 240) / 60)
    rho_t = spearmanr(z, nr["credit_score"]).statistic
    rho_a = spearmanr(nr["age"], nr["credit_score"]).statistic
    rho_ta = spearmanr(nr["age"], nr["employment_tenure_months"]).statistic
    parts.append(md_table(pd.DataFrame({
        "pair (non-retired, both observed)": ["z_t (bounded log tenure) vs credit_score", "age vs credit_score", "age vs tenure"],
        "Spearman": [rho_t, rho_a, rho_ta]}), "{:.3f}") + "\n")
    parts.append("The bounded log tenure z_t = ln(min(max(tenure, 3), 240) / 60) is used both in the credit-score "
                 "generation (weight 10) and the default formula. Tenure is also correlated with age, which also "
                 "enters the score, so the table shows a marginal association, not an isolated effect.\n")

    # ---- default rates
    parts.append("## 4. Observed default rates by segment\n")
    parts.append("Counts are shown beside every rate; small groups are noisy.\n")
    sc = d.dropna(subset=["credit_score"]).copy()
    sc["credit_score_quintile"] = pd.qcut(sc["credit_score"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    parts.append("### Credit-score quintile (Q1 lowest score)\n")
    parts.append(md_table(default_rate_table(sc, "credit_score_quintile", "credit_score_quintile"), "{:.2%}") + "\n")
    d["dti_bin"] = "zero"
    pos = d["debt_to_income"] > 0
    q = pd.qcut(d.loc[pos, "debt_to_income"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    d.loc[pos, "dti_bin"] = "positive " + q.astype(str)
    parts.append("### Debt-to-income (zero, then quartiles of positive values)\n")
    parts.append(md_table(default_rate_table(d, "dti_bin", "dti_bin"), "{:.2%}") + "\n")
    d["lti_quintile"] = pd.qcut(d["loan_to_income"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    parts.append("### Loan-to-income quintile\n")
    parts.append(md_table(default_rate_table(d, "lti_quintile", "lti_quintile"), "{:.2%}") + "\n")
    for col, label in [("term_months", "Term (months)"), ("loan_purpose", "Loan purpose"),
                       ("employment_status", "Employment status"), ("region", "Region")]:
        parts.append(f"### {label}\n")
        parts.append(md_table(default_rate_table(d, col, col), "{:.2%}") + "\n")
    nr_l = d[~d["is_retired"]].dropna(subset=["employment_tenure_months"]).copy()
    nr_l["tenure_band"] = tenure_band(nr_l["employment_tenure_months"])
    tb = default_rate_table(nr_l, "tenure_band", "tenure_band").set_index("tenure_band").reindex(order).reset_index()
    parts.append("### Tenure band (non-retired, observed tenure)\n")
    parts.append(md_table(tb.dropna(), "{:.2%}") + "\n")
    d["mob_band"] = pd.cut(d["months_on_book"], [0, 6, 12, 24, 36, 48], labels=["1-6", "7-12", "13-24", "25-36", "37-48"])
    parts.append("### Months-on-book band\n")
    parts.append(md_table(default_rate_table(d, "mob_band", "mob_band"), "{:.2%}") + "\n")

    parts.append("## 5. Figures\n")
    parts.append("- figures/fig1_distributions.png\n- figures/fig2_relationships.png\n")
    (out_dir / "data_profile.md").write_text("\n".join(parts))

    # ---- figure 1: distributions
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist(cust["age"], bins=range(19, 77, 2), color="#2F5496"); axes[0, 0].set_title("Age at T0")
    axes[0, 1].hist(np.log10(cust["annual_income"]), bins=40, color="#2F5496"); axes[0, 1].set_title("log10 annual income (GBP)")
    axes[0, 2].hist(cust["credit_score"].dropna(), bins=40, color="#2F5496"); axes[0, 2].set_title("Credit score (observed)")
    axes[1, 0].hist(loans["loan_amount"], bins=40, color="#2F5496"); axes[1, 0].set_title("Loan amount (GBP)")
    axes[1, 1].hist(loans["interest_rate"], bins=40, color="#2F5496"); axes[1, 1].set_title("Interest rate (APR %)")
    axes[1, 2].hist(d["debt_to_income"].clip(upper=1.5), bins=40, color="#2F5496"); axes[1, 2].set_title("Debt-to-income (capped at 1.5 for display)")
    fig.suptitle("Synthetic data: distributions")
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_distributions.png", dpi=110); plt.close(fig)

    # ---- figure 2: relationships
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ts = st[st["tenure_band"].isin(order)]
    axes[0].errorbar(range(len(ts)), ts["mean_score"], yerr=ts["sd_score"] / np.sqrt(ts["n_customers"]) * 1.96,
                     marker="o", color="#2F5496")
    axes[0].set_xticks(range(len(ts))); axes[0].set_xticklabels(ts["tenure_band"], rotation=45)
    axes[0].set_title("Mean credit score by tenure band\n(non-retired; bars = 95% CI of the mean)"); axes[0].set_xlabel("Tenure (months)")
    sq = default_rate_table(sc, "credit_score_quintile", "credit_score_quintile")
    axes[1].bar(sq["credit_score_quintile"].astype(str), sq["default_rate"] * 100, color="#2F5496")
    axes[1].set_title("Default rate by credit-score quintile (%)")
    dq = default_rate_table(d, "dti_bin", "dti_bin")
    dq = dq.set_index("dti_bin").reindex(["zero", "positive Q1", "positive Q2", "positive Q3", "positive Q4"]).reset_index()
    axes[2].bar(dq["dti_bin"], dq["default_rate"] * 100, color="#2F5496"); axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_title("Default rate by debt-to-income bin (%)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_relationships.png", dpi=110); plt.close(fig)
    print(f"Wrote {out_dir / 'data_profile.md'} and two figures.")
