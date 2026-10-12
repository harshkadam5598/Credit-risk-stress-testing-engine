"""Automated data-quality checks for the synthetic customer and loan data.

Several checks (age rule, maturity rule, balance, months on book) are deliberately
re-implemented here with plain pandas/loop logic instead of importing the generator's
helpers, so a bug in the generator cannot silently hide itself in the validator.

Statuses:  PASS | FAIL | INFO | REVIEW
  FAIL   - a rule is broken; the pipeline exits with an error.
  REVIEW - a DIAGNOSTIC fell outside its expected band; investigate and report.
           It never triggers an automatic parameter change.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from generate_data import (AUDIT_COLUMNS, CUSTOMER_COLUMNS, LOAN_COLUMNS, generate_all,
                           loans_per_customer_counts, load_config, sha256_of, write_outputs)

REGION_COUNT = 12


# --------------------------------------------------------------------------- helpers
def _add(results: list, check_id: str, group: str, check: str, status: str,
         value="", expected="", detail="") -> None:
    results.append({"check_id": check_id, "group": group, "check": check, "status": status,
                    "value": value, "expected": expected, "detail": detail})


def _pass_if(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _range(results, check_id, group, label, series, lo, hi, integer=False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    bad = int(((s < lo) | (s > hi)).sum())
    if integer:
        bad += int((s != np.round(s)).sum())
    _add(results, check_id, group, f"{label} within [{lo}, {hi}]" + (" and integer" if integer else ""),
         _pass_if(bad == 0), bad, 0, f"{len(s)} non-missing values checked")


def _categories(results, check_id, group, label, series, allowed):
    bad = int((~series.isin(allowed)).sum())
    _add(results, check_id, group, f"{label} only takes allowed values", _pass_if(bad == 0), bad, 0,
         f"allowed: {len(allowed)} categories")


def read_data(root: Path, cfg: dict):
    raw = root / cfg["paths"]["raw_dir"]
    audit = root / cfg["paths"]["audit_dir"]
    customers = pd.read_csv(raw / "customer_data.csv")
    loans = pd.read_csv(raw / "loan_data.csv")
    audit_df = pd.read_csv(audit / "generation_audit.csv")
    return customers, loans, audit_df


# --------------------------------------------------------------------------- main checks
def validate_frames(customers: pd.DataFrame, loans: pd.DataFrame, audit: pd.DataFrame, cfg: dict) -> list[dict]:
    res: list[dict] = []
    c, L, d, v = cfg["customers"], cfg["loans"], cfg["dates"], cfg["validation"]
    t0, h_start, h_end = pd.Timestamp(d["t0"]), pd.Timestamp(d["horizon_start"]), pd.Timestamp(d["horizon_end"])

    # ---- schema
    _add(res, "S01", "schema", "customer_data columns match specification",
         _pass_if(list(customers.columns) == CUSTOMER_COLUMNS), len(customers.columns), len(CUSTOMER_COLUMNS))
    _add(res, "S02", "schema", "loan_data columns match specification",
         _pass_if(list(loans.columns) == LOAN_COLUMNS), len(loans.columns), len(LOAN_COLUMNS))
    numeric_c = ["age", "annual_income", "employment_tenure_months", "credit_score", "existing_debt"]
    numeric_l = ["loan_amount", "term_months", "interest_rate", "outstanding_balance", "default_status"]
    non_numeric = [col for col in numeric_c if not pd.api.types.is_numeric_dtype(customers[col])] + \
                  [col for col in numeric_l if not pd.api.types.is_numeric_dtype(loans[col])]
    _add(res, "S03", "schema", "numeric fields are numeric", _pass_if(not non_numeric), len(non_numeric), 0,
         ", ".join(non_numeric))

    # ---- ranges and categories
    g = "ranges"
    _range(res, "R01", g, "age", customers["age"], c["age_min"], c["age_max"], integer=True)
    _range(res, "R02", g, "annual_income", customers["annual_income"], c["income_min"], c["income_max"])
    _categories(res, "R03", g, "employment_status", customers["employment_status"], c["employment_categories"])
    _range(res, "R04", g, "employment_tenure_months", customers["employment_tenure_months"], 0, c["tenure_max_months"], integer=True)
    _range(res, "R05", g, "credit_score", customers["credit_score"], c["credit_score_min"], c["credit_score_max"], integer=True)
    _range(res, "R06", g, "existing_debt", customers["existing_debt"], 0, np.inf, integer=True)
    _categories(res, "R07", g, "region", customers["region"], c["region_names"])
    _range(res, "R08", g, "loan_amount", loans["loan_amount"], L["amount_min"], L["amount_max"], integer=True)
    _categories(res, "R09", g, "term_months", loans["term_months"], L["terms"])
    _range(res, "R10", g, "interest_rate", loans["interest_rate"], L["rate_min"], L["rate_max"])
    _categories(res, "R11", g, "loan_purpose", loans["loan_purpose"], L["purposes"])
    _categories(res, "R12", g, "default_status", loans["default_status"], [0, 1])
    bad_mult = int((loans["loan_amount"] % L["amount_round_to"] != 0).sum())
    _add(res, "R13", g, "loan_amount is a multiple of 100", _pass_if(bad_mult == 0), bad_mult, 0)
    bad_mult = int((customers["existing_debt"] % c["debt_round_to"] != 0).sum())
    _add(res, "R14", g, "existing_debt is a multiple of 10", _pass_if(bad_mult == 0), bad_mult, 0)

    # ---- missingness
    g = "missingness"
    tol = v["missingness_tolerance"]
    miss_cols = [col for col in customers.columns if customers[col].isna().any()]
    unexpected = sorted(set(miss_cols) - {"credit_score", "employment_tenure_months"})
    _add(res, "M01", g, "customer fields other than credit_score and tenure have no missing values",
         _pass_if(not unexpected), ", ".join(unexpected) or "none", "none")
    rate = customers["credit_score"].isna().mean()
    _add(res, "M02", g, "credit_score missing rate near configured value",
         _pass_if(abs(rate - cfg["missingness"]["credit_score_rate"]) <= tol), f"{rate:.4%}",
         f"{cfg['missingness']['credit_score_rate']:.2%} +/- {tol:.1%}")
    non_ret = customers["employment_status"] != "Retired"
    rate = customers.loc[non_ret, "employment_tenure_months"].isna().mean()
    _add(res, "M03", g, "tenure missing rate (non-retired) near configured value",
         _pass_if(abs(rate - cfg["missingness"]["tenure_rate_non_retired"]) <= tol), f"{rate:.4%}",
         f"{cfg['missingness']['tenure_rate_non_retired']:.2%} +/- {tol:.1%}")
    ret_missing = int(customers.loc[~non_ret, "employment_tenure_months"].isna().sum())
    _add(res, "M04", g, "Retired customers never have missing tenure (coded 0)", _pass_if(ret_missing == 0), ret_missing, 0)
    loan_missing = [col for col in loans.columns if col != "default_date" and loans[col].isna().any()]
    _add(res, "M05", g, "loan fields other than default_date have no missing values",
         _pass_if(not loan_missing), ", ".join(loan_missing) or "none", "none")

    # ---- keys and uniqueness
    g = "keys"
    _add(res, "K01", g, "customer_id is unique", _pass_if(customers["customer_id"].is_unique),
         int(customers["customer_id"].duplicated().sum()), 0)
    _add(res, "K02", g, "loan_id is unique", _pass_if(loans["loan_id"].is_unique),
         int(loans["loan_id"].duplicated().sum()), 0)
    orphans = int((~loans["customer_id"].isin(customers["customer_id"])).sum())
    _add(res, "K03", g, "every loan links to an existing customer", _pass_if(orphans == 0), orphans, 0)
    no_loan = int((~customers["customer_id"].isin(loans["customer_id"])).sum())
    _add(res, "K04", g, "every customer has at least one loan", _pass_if(no_loan == 0), no_loan, 0)
    per_cust = loans.groupby("customer_id").size().value_counts().sort_index()
    expected_counts = dict(zip(cfg["portfolio"]["loans_per_customer"], loans_per_customer_counts(cfg)))
    actual_counts = {int(k): int(val) for k, val in per_cust.items()}
    _add(res, "K05", g, "loans-per-customer allocation is exact", _pass_if(actual_counts == expected_counts),
         str(actual_counts), str(expected_counts))
    _add(res, "K06", g, "customer and loan totals as configured",
         _pass_if(len(customers) == cfg["run"]["n_customers"] and len(loans) == sum(k * n for k, n in expected_counts.items())),
         f"{len(customers)} customers / {len(loans)} loans",
         f"{cfg['run']['n_customers']} customers / {sum(k * n for k, n in expected_counts.items())} loans")

    # ---- chronology (independent re-implementation)
    g = "chronology"
    orig = pd.to_datetime(loans["origination_date"], format="%Y-%m-%d", errors="coerce")
    dflt = pd.to_datetime(loans["default_date"], format="%Y-%m-%d", errors="coerce")
    bad_dates = int(orig.isna().sum())
    _add(res, "C01", g, "origination_date parses as a valid date", _pass_if(bad_dates == 0), bad_dates, 0)
    out_window = int(((orig < pd.Timestamp(d["origination_floor"])) | (orig > pd.Timestamp(d["origination_latest"]))).sum())
    _add(res, "C02", g, "origination_date within the allowed window", _pass_if(out_window == 0), out_window, 0,
         f"{d['origination_floor']} to {d['origination_latest']}")
    sched_end = pd.Series(pd.NaT, index=loans.index, dtype="datetime64[ns]")
    for term in loans["term_months"].dropna().unique():
        mask = loans["term_months"] == term
        sched_end[mask] = orig[mask] + pd.DateOffset(months=int(term))
    early_end = int((sched_end < h_end).sum())
    _add(res, "C03", g, "scheduled end date is on or after 31 Dec 2024 (inclusive)", _pass_if(early_end == 0), early_end, 0)

    merged = loans.merge(customers[["customer_id", "age", "employment_status", "employment_tenure_months", "annual_income"]],
                         on="customer_id", how="left")
    # age rule: origination_date >= 31 Dec of (T0 year - (age - 18)); computed from integer parts
    req_year = d["t0"].year - (merged["age"] - 18)
    on_year = orig.dt.year == req_year
    before_anniv = (orig.dt.month < d["t0"].month) | ((orig.dt.month == d["t0"].month) & (orig.dt.day < d["t0"].day))
    too_early = (orig.dt.year < req_year) | (on_year & before_anniv)
    n_young = int(too_early.sum())
    _add(res, "C04", g, "customer was at least 18 on the origination date (conservative rule)", _pass_if(n_young == 0),
         n_young, 0, "origination_date >= T0 minus (age - 18) years, birthday unknown")

    has_default = loans["default_status"] == 1
    mismatch = int((has_default & dflt.isna()).sum() + (~has_default & dflt.notna()).sum())
    _add(res, "C05", g, "default_date present exactly when default_status = 1", _pass_if(mismatch == 0), mismatch, 0)
    out_h = int(((dflt < h_start) | (dflt > h_end)).sum())
    _add(res, "C06", g, "default_date inside the 12-month horizon", _pass_if(out_h == 0), out_h, 0)
    after_end = int((dflt > sched_end).sum())
    _add(res, "C07", g, "default_date on or before the scheduled end date", _pass_if(after_end == 0), after_end, 0)

    # ---- logical consistency
    g = "logic"
    over = int((merged["loan_amount"] > merged["annual_income"] * L["amount_max_loan_to_income"]).sum())
    _add(res, "L01", g, "loan_amount does not exceed annual income", _pass_if(over == 0), over, 0)
    bal_ok = (loans["outstanding_balance"] > 0) & (loans["outstanding_balance"] < loans["loan_amount"])
    _add(res, "L02", g, "0 < outstanding_balance < loan_amount", _pass_if(bool(bal_ok.all())), int((~bal_ok).sum()), 0)

    # balance recomputed by month-by-month amortisation (independent of the closed form)
    r = loans["interest_rate"].to_numpy(dtype=float) / 1200.0
    amount = loans["loan_amount"].to_numpy(dtype=float)
    n = loans["term_months"].to_numpy(dtype=float)
    pay = amount * r / (1.0 - (1.0 + r) ** (-n))
    k = ((d["t0"].year - orig.dt.year) * 12 + (d["t0"].month - orig.dt.month) - (orig.dt.day > d["t0"].day)).to_numpy(dtype=int)
    bal = amount.copy()
    for step in range(int(k.max()) if len(k) else 0):
        active = step < k
        bal[active] = bal[active] * (1.0 + r[active]) - pay[active]
    max_gap = float(np.max(np.abs(bal - loans["outstanding_balance"].to_numpy(dtype=float)))) if len(k) else 0.0
    _add(res, "L03", g, "outstanding_balance matches month-by-month amortisation",
         _pass_if(max_gap <= v["balance_tolerance_gbp"] + 1e-9), f"{max_gap:.4f}", f"<= {v['balance_tolerance_gbp']}",
         "max absolute gap in GBP")

    young_retired = int(((customers["employment_status"] == "Retired") & (customers["age"] < 60)).sum())
    _add(res, "L04", g, "Retired status only from age 60", _pass_if(young_retired == 0), young_retired, 0)
    ret_tenure = int(((customers["employment_status"] == "Retired") & (customers["employment_tenure_months"].fillna(0) != 0)).sum())
    _add(res, "L05", g, "Retired customers have tenure coded 0", _pass_if(ret_tenure == 0), ret_tenure, 0)
    too_long = int((customers["employment_tenure_months"] > (customers["age"] - 18) * 12).sum())
    _add(res, "L06", g, "tenure does not exceed working life (age - 18) x 12", _pass_if(too_long == 0), too_long, 0)

    # ---- target and relationships
    g = "target"
    prev = float(loans["default_status"].mean())
    target, ptol = cfg["portfolio"]["target_prevalence"], cfg["portfolio"]["prevalence_tolerance"]
    _add(res, "T01", g, "realised default rate within tolerance of the configured target",
         _pass_if(abs(prev - target) <= ptol), f"{prev:.4%}", f"{target:.2%} +/- {ptol:.1%}",
         "simulation assumption, not a market statistic")

    data = loans.merge(customers, on="customer_id", how="left")
    data["dti"] = data["existing_debt"] / data["annual_income"]
    min_rho = v["trend_min_spearman"]

    sc = data.dropna(subset=["credit_score"]).copy()
    sc["bin"] = pd.qcut(sc["credit_score"], 5, labels=False, duplicates="drop")
    rates = sc.groupby("bin")["default_status"].mean()
    rho = spearmanr(rates.index, rates.values).statistic
    ok = rho <= -min_rho and rates.iloc[0] > rates.iloc[-1]
    _add(res, "T02", g, "default rate falls across credit-score quintiles", _pass_if(bool(ok)),
         "; ".join(f"{x:.2%}" for x in rates.values), f"Spearman <= -{min_rho} and lowest bin > highest bin",
         f"Spearman = {rho:.2f}")

    pos = data["dti"] > 0
    data["dti_bin"] = 0
    data.loc[pos, "dti_bin"] = pd.qcut(data.loc[pos, "dti"], 4, labels=False) + 1
    rates = data.groupby("dti_bin")["default_status"].mean()
    rho = spearmanr(rates.index, rates.values).statistic
    ok = rho >= min_rho and rates.iloc[-1] > rates.iloc[0]
    _add(res, "T03", g, "default rate rises across debt-to-income bins (zero, then quartiles)", _pass_if(bool(ok)),
         "; ".join(f"{x:.2%}" for x in rates.values), f"Spearman >= {min_rho} and highest bin > lowest bin",
         f"Spearman = {rho:.2f}")

    one_class = []
    for col in ["region", "loan_purpose", "term_months", "employment_status"]:
        by = data.groupby(col)["default_status"].agg(["sum", "count"])
        one_class += [f"{col}={idx}" for idx, row in by.iterrows() if row["sum"] == 0 or row["sum"] == row["count"]]
    _add(res, "T04", g, "both outcome classes occur in every region, purpose, term and employment group",
         _pass_if(not one_class), ", ".join(one_class) or "none", "none")

    # ---- audit file and leakage
    g = "leakage"
    audit_cols = set(AUDIT_COLUMNS) - {"loan_id", "customer_id"}
    leaked = sorted(audit_cols & (set(customers.columns) | set(loans.columns)))
    _add(res, "A01", g, "hidden audit columns are absent from customer_data and loan_data",
         _pass_if(not leaked), ", ".join(leaked) or "none", "none")
    aligned = loans["loan_id"].tolist() == audit["loan_id"].tolist()
    in_range = bool(((audit["true_pd_12m"] > 0) & (audit["true_pd_12m"] < 1)).all())
    _add(res, "A02", g, "audit file aligns with loans and true PDs lie strictly in (0, 1)",
         _pass_if(aligned and in_range), f"aligned={aligned}, in_range={in_range}", "True, True")

    # ---- oracle benchmark and generator diagnostics (diagnostic only)
    g = "oracle"
    lo, hi = v["oracle_auc_band"]
    auc = float(roc_auc_score(loans["default_status"], audit["true_pd_12m"]))
    in_band = lo <= auc <= hi
    _add(res, "O01", g, "oracle AUC (true PD vs simulated outcomes), full portfolio",
         "INFO" if in_band else "REVIEW", f"{auc:.4f}", f"diagnostic band {lo}-{hi}",
         "inside the band" if in_band else "OUTSIDE the band: investigate and report; no automatic parameter change")
    p = audit["true_pd_12m"].to_numpy()
    eta = np.log(p / (1 - p))
    hidden = (audit["customer_effect"] + audit["loan_effect"]).to_numpy()
    _add(res, "O02", g, "standard deviation of the true log-odds", "INFO", f"{eta.std():.4f}", "design estimate 1.0-1.2")
    _add(res, "O03", g, "share of log-odds variance from hidden effects", "INFO", f"{hidden.var() / eta.var():.4f}",
         "design estimate about a quarter")
    both = pd.DataFrame({"s": data["credit_score"], "h": audit["customer_effect"]}).dropna()
    _add(res, "O04", g, "correlation of observed credit_score with the hidden customer effect", "INFO",
         f"{both['s'].corr(both['h']):.4f}", "design: negative")
    _add(res, "O05", g, "mean true PD versus realised default rate", "INFO",
         f"{p.mean():.4%} vs {prev:.4%}", "equal in expectation by construction")
    return res


# --------------------------------------------------------------------------- file-level checks
def check_processed_table(root: Path, cfg: dict, loans: pd.DataFrame, customers: pd.DataFrame) -> list[dict]:
    res: list[dict] = []
    path = root / cfg["paths"]["processed_dir"] / "loan_analysis_base.csv"
    if not path.exists():
        _add(res, "P01", "processed", "analysis-ready table exists", "FAIL", "missing", "present")
        return res
    joined = pd.read_csv(path)
    expected_cols = LOAN_COLUMNS + [col for col in CUSTOMER_COLUMNS if col != "customer_id"]
    _add(res, "P01", "processed", "joined table has the loan and customer columns only (no derived features, no split, no audit columns)",
         _pass_if(list(joined.columns) == expected_cols), len(joined.columns), len(expected_cols))
    _add(res, "P02", "processed", "joined table has one row per loan", _pass_if(len(joined) == len(loans) and joined["loan_id"].is_unique),
         len(joined), len(loans))
    return res


def check_oracle_test_split(root: Path, cfg: dict, loans: pd.DataFrame, audit: pd.DataFrame) -> list[dict]:
    """Oracle AUC (true PD vs outcomes) on the held-out test split. Diagnostic only (S2-3).

    Lives here, not in the modelling code, because the hidden audit file is never loaded by modelling.
    """
    res: list[dict] = []
    path = root / cfg["paths"]["processed_dir"] / "loan_split.csv"
    if not path.exists():
        return res
    split = pd.read_csv(path)
    test = (split["data_split"] == "test").to_numpy()
    auc = float(roc_auc_score(loans.loc[test, "default_status"], audit.loc[test, "true_pd_12m"]))
    lo, hi = cfg["validation"]["oracle_auc_band"]
    in_band = lo <= auc <= hi
    _add(res, "O06", "oracle", "oracle AUC (true PD vs simulated outcomes), held-out test split",
         "INFO" if in_band else "REVIEW", f"{auc:.4f}", f"diagnostic band {lo}-{hi}",
         "same accepted deviation as O01; a fitted model should not be expected to match it")
    return res


def check_reproducibility(root: Path, cfg: dict) -> list[dict]:
    """Regenerate everything in a temporary folder and compare file hashes with the files on disk."""
    res: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        hashes = write_outputs(cfg, generate_all(cfg), Path(tmp))
    on_disk = {
        "customer_data.csv": sha256_of(root / cfg["paths"]["raw_dir"] / "customer_data.csv"),
        "loan_data.csv": sha256_of(root / cfg["paths"]["raw_dir"] / "loan_data.csv"),
        "generation_audit.csv": sha256_of(root / cfg["paths"]["audit_dir"] / "generation_audit.csv"),
    }
    diff = [name for name in hashes if hashes[name] != on_disk[name]]
    _add(res, "X01", "reproducibility", "regeneration with the same seed gives byte-identical files",
         _pass_if(not diff), ", ".join(diff) or "identical", "identical", "compared by SHA-256")
    return res


# --------------------------------------------------------------------------- entry point
def run(config_path: str | Path, root: Path) -> int:
    cfg = load_config(config_path)
    customers, loans, audit = read_data(root, cfg)
    results = validate_frames(customers, loans, audit, cfg)
    results += check_processed_table(root, cfg, loans, customers)
    results += check_oracle_test_split(root, cfg, loans, audit)
    results += check_reproducibility(root, cfg)

    out_dir = root / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results)
    summary.to_csv(out_dir / "validation_summary.csv", index=False)

    counts = summary["status"].value_counts().to_dict()
    for row in results:
        print(f"[{row['status']:6}] {row['check_id']} {row['check']} -> {row['value']}")
    print("Summary:", counts)
    return 1 if counts.get("FAIL", 0) else 0
