"""Synthetic data generator for the Northbridge Bank credit-risk case study.

All data is SYNTHETIC. Every parameter lives in config/config.toml.

Order of generation (approved plan):
  1. complete customer attributes
  2. loan allocation and complete loan attributes
  3. derived quantities, log-odds, intercept solve, true PD, default outcomes
  4. audit file (hidden variables)
  5. observed-field missingness (credit_score, employment_tenure_months) - LAST
  6. write CSV files
Each stage draws from its own child random stream spawned from one master seed.
"""
from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import truncnorm

EPOCH_ORDINAL = date(1970, 1, 1).toordinal()

CUSTOMER_COLUMNS = [
    "customer_id", "age", "annual_income", "employment_status",
    "employment_tenure_months", "credit_score", "existing_debt", "region",
]
LOAN_COLUMNS = [
    "loan_id", "customer_id", "origination_date", "loan_amount", "term_months",
    "interest_rate", "loan_purpose", "outstanding_balance", "default_status", "default_date",
]
AUDIT_COLUMNS = ["loan_id", "customer_id", "customer_effect", "loan_effect", "true_pd_12m"]


# --------------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --------------------------------------------------------------------------- date helpers
def add_months(d: date, n: int) -> date:
    """Add n calendar months; if the day does not exist in the target month use its last day."""
    month_index = d.year * 12 + (d.month - 1) + n
    year, month = divmod(month_index, 12)
    month += 1
    first_next = date(year + (month == 12), month % 12 + 1, 1)
    last_day = (first_next - date(year, month, 1)).days
    return date(year, month, min(d.day, last_day))


def earliest_origination_for_term(term_months: int, horizon_end: date) -> date:
    """Earliest origination date whose scheduled end date is on or after horizon_end."""
    candidate = add_months(horizon_end, -term_months)
    while add_months(candidate, term_months) < horizon_end:
        candidate = date.fromordinal(candidate.toordinal() + 1)
    while add_months(date.fromordinal(candidate.toordinal() - 1), term_months) >= horizon_end:
        candidate = date.fromordinal(candidate.toordinal() - 1)
    return candidate


def earliest_origination_for_age(age_t0: int, t0: date) -> date:
    """Conservative age rule: origination_date >= T0 minus (age - 18) calendar years.

    Only integer age in completed years at T0 is stored, so the birthday is unknown.
    The worst case is a birthday on T0; this date guarantees the customer was already 18.
    """
    year = t0.year - (age_t0 - 18)
    day = 28 if (t0.month == 2 and t0.day == 29) else t0.day   # Feb 29 only exists in leap years
    return date(year, t0.month, day)


def months_on_book(origination: pd.Series, t0: date) -> np.ndarray:
    """Whole calendar months from origination to T0."""
    o = pd.to_datetime(origination)
    months = (t0.year - o.dt.year) * 12 + (t0.month - o.dt.month)
    months = months - (t0.day < o.dt.day).astype(int)
    return months.to_numpy(dtype=int)


# --------------------------------------------------------------------------- finance helpers
def annuity_payment(amount, apr_pct, n_months):
    r = np.asarray(apr_pct, dtype=float) / 1200.0
    amount = np.asarray(amount, dtype=float)
    n = np.asarray(n_months, dtype=float)
    return amount * r / (1.0 - (1.0 + r) ** (-n))


def balance_after(amount, apr_pct, n_months, k_payments):
    r = np.asarray(apr_pct, dtype=float) / 1200.0
    pay = annuity_payment(amount, apr_pct, n_months)
    growth = (1.0 + r) ** np.asarray(k_payments, dtype=float)
    return np.asarray(amount, dtype=float) * growth - pay * (growth - 1.0) / r


# --------------------------------------------------------------------------- shared transform
def tenure_z(tenure_months, is_retired, cfg: dict) -> np.ndarray:
    """z_t = ln( min(max(tenure, floor), cap) / reference ); Retired customers get 0."""
    t = cfg["transforms"]
    clipped = np.clip(np.asarray(tenure_months, dtype=float), t["tenure_floor_months"], t["tenure_cap_months"])
    z = np.log(clipped / t["tenure_reference_months"])
    return np.where(np.asarray(is_retired, dtype=bool), 0.0, z)


# --------------------------------------------------------------------------- default model
def default_logit_without_intercept(f: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Observed-driver part of the log-odds (everything except alpha and hidden effects).

    Required columns of f (all COMPLETE, pre-missingness values):
      credit_score, existing_debt, annual_income, loan_amount, interest_rate, term_months,
      employment_status, employment_tenure_months, age, loan_purpose, region, months_on_book
    """
    m = cfg["default_model"]
    income = f["annual_income"].to_numpy(dtype=float)
    dti = f["existing_debt"].to_numpy(dtype=float) / income
    lti = f["loan_amount"].to_numpy(dtype=float) / income
    inst = annuity_payment(f["loan_amount"], f["interest_rate"], f["term_months"])
    iti = 12.0 * inst / income
    z_s = (f["credit_score"].to_numpy(dtype=float) - m["score_centre"]) / m["score_scale"]
    z_iti = (iti - m["iti_centre"]) / m["iti_scale"]
    z_lti = (lti - m["lti_centre"]) / m["lti_scale"]
    retired = (f["employment_status"] == "Retired").to_numpy()
    z_t = tenure_z(f["employment_tenure_months"], retired, cfg)
    age = f["age"].to_numpy(dtype=float)
    mob = f["months_on_book"].to_numpy(dtype=float)

    eta = m["score_linear"] * z_s + m["score_quadratic"] * z_s**2
    eta = eta + m["dti_linear"] * dti / m["dti_scale"]
    eta = eta + m["dti_kink"] * np.maximum(0.0, dti - m["dti_kink_point"]) / m["dti_scale"]
    eta = eta + m["iti_coefficient"] * z_iti + m["lti_coefficient"] * z_lti
    eta = eta + m["interaction_coefficient"] * np.maximum(0.0, -z_s) * np.maximum(0.0, z_lti)
    eta = eta + f["employment_status"].map(m["employment_offset"]).to_numpy(dtype=float)
    eta = eta + f["term_months"].map(lambda t: m["term_offset"][str(int(t))]).to_numpy(dtype=float)
    eta = eta + f["loan_purpose"].map(m["purpose_offset"]).to_numpy(dtype=float)
    eta = eta + f["region"].map(m["region_offset"]).to_numpy(dtype=float)
    eta = eta + m["tenure_coefficient"] * z_t
    eta = eta + m["age_coefficient"] * ((age - m["age_centre"]) / m["age_scale"]) ** 2
    eta = eta + m["income_coefficient"] * np.log(income / m["income_ref"])
    eta = eta + m["mob_amplitude"] * np.exp(-0.5 * ((mob - m["mob_centre"]) / m["mob_width"]) ** 2)
    return eta


def solve_intercept(base_logit: np.ndarray, target: float, bracket) -> float:
    """Find alpha so that mean(expit(base_logit + alpha)) equals the target prevalence."""
    f = lambda a: float(np.mean(expit(base_logit + a))) - target  # noqa: E731
    return brentq(f, bracket[0], bracket[1], xtol=1e-12)


# --------------------------------------------------------------------------- stage 1: customers
@dataclass
class Customers:
    complete: pd.DataFrame          # complete values, no missingness
    z_c: np.ndarray                 # hidden latent creditworthiness


def generate_customers(cfg: dict, rng: np.random.Generator) -> Customers:
    c = cfg["customers"]
    n = cfg["run"]["n_customers"]

    # age: truncated normal on [min-0.5, max+0.5], rounded -> integer support min..max
    lo, hi = c["age_min"] - 0.5, c["age_max"] + 0.5
    a, b = (lo - c["age_mean"]) / c["age_sd"], (hi - c["age_mean"]) / c["age_sd"]
    age = np.rint(truncnorm.rvs(a, b, loc=c["age_mean"], scale=c["age_sd"], size=n, random_state=rng)).astype(int)

    # employment status by age band
    shares = np.array([c["employment_shares_under_60"], c["employment_shares_60_64"], c["employment_shares_65_plus"]])
    band = np.where(age >= 65, 2, np.where(age >= 60, 1, 0))
    cum = np.cumsum(shares, axis=1)
    u = rng.random(n)
    status_idx = np.minimum((u[:, None] > cum[band]).sum(axis=1), len(c["employment_categories"]) - 1)
    status = np.array(c["employment_categories"])[status_idx]
    retired = status == "Retired"

    # income: truncated log-normal by redrawing out-of-range values (not clipping)
    median = np.array(c["income_median_by_status"])[status_idx]
    sigma = np.array(c["income_log_sd_by_status"])[status_idx]
    age_effect = c["income_age_effect_per_year"] * (np.minimum(age, c["income_age_effect_cap_age"]) - c["income_age_effect_ref_age"])
    base_ln = np.log(median) + age_effect
    eps = rng.standard_normal(n)
    for _ in range(500):
        income = np.exp(base_ln + sigma * eps)
        bad = (income < c["income_min"]) | (income > c["income_max"])
        if not bad.any():
            break
        eps[bad] = rng.standard_normal(int(bad.sum()))
    else:  # pragma: no cover
        raise RuntimeError("income redraw did not converge")
    income = np.round(income, 2)

    # tenure (months): gamma, capped by working life, Retired coded 0 (not applicable)
    tenure = np.rint(rng.gamma(c["tenure_gamma_shape"], c["tenure_gamma_scale"], size=n)).astype(int)
    tenure = np.minimum(tenure, np.minimum((age - 18) * 12, c["tenure_max_months"]))
    tenure = np.where(retired, 0, tenure)

    # latent creditworthiness and credit score
    z_c = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    z_t = tenure_z(tenure, retired, cfg)
    std_age = (age - c["generation_age_centre"]) / c["generation_age_scale"]
    std_inc = np.log(income / c["generation_income_ref"]) / c["generation_income_log_scale"]
    score = (c["credit_score_base"] + c["credit_score_w_latent"] * z_c + c["credit_score_w_age"] * std_age
             + c["credit_score_w_tenure"] * z_t + c["credit_score_w_income"] * std_inc
             + c["credit_score_w_noise"] * noise)
    score = np.clip(np.rint(score), c["credit_score_min"], c["credit_score_max"]).astype(int)

    # existing debt: 25% zero, otherwise a log-normal share of income that falls with creditworthiness
    zero_flag = rng.random(n) < c["debt_zero_probability"]
    share = np.exp(np.log(c["debt_median_share_of_income"]) + c["debt_latent_coefficient"] * z_c
                   + c["debt_log_sd"] * rng.standard_normal(n))
    share = np.minimum(share, c["debt_max_share_of_income"])
    debt = np.round(np.where(zero_flag, 0.0, share * income) / c["debt_round_to"]) * c["debt_round_to"]

    region = np.array(c["region_names"])[rng.choice(len(c["region_names"]), size=n, p=c["region_weights"])]

    df = pd.DataFrame({
        "customer_id": [f"C{i + 1:05d}" for i in range(n)],
        "age": age,
        "annual_income": income,
        "employment_status": status,
        "employment_tenure_months": tenure,
        "credit_score": score,
        "existing_debt": debt.astype(int),
        "region": region,
    })
    return Customers(complete=df, z_c=z_c)


# --------------------------------------------------------------------------- stage 2: loans
def loans_per_customer_counts(cfg: dict) -> list[int]:
    """Exact allocation of customers to 1/2/3 loans (6,000/3,000/1,000 at 10,000 customers)."""
    n = cfg["run"]["n_customers"]
    shares = cfg["portfolio"]["loans_per_customer_share"]
    counts = [int(round(n * s)) for s in shares]
    counts[-1] = n - sum(counts[:-1])
    return counts


def generate_loans(cfg: dict, cust: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    L = cfg["loans"]
    d = cfg["dates"]
    t0, h_end = d["t0"], d["horizon_end"]
    n_cust = len(cust)

    # exact loans-per-customer allocation, randomly assigned to customers
    per_customer = np.repeat(cfg["portfolio"]["loans_per_customer"], loans_per_customer_counts(cfg))
    per_customer = rng.permutation(per_customer)
    cidx = np.repeat(np.arange(n_cust), per_customer)
    n = len(cidx)

    purposes = np.array(L["purposes"])
    p_idx = rng.choice(len(purposes), size=n, p=L["purpose_shares"])
    t_idx = rng.choice(len(L["terms"]), size=n, p=L["term_shares"])
    term = np.array(L["terms"])[t_idx]

    income = cust["annual_income"].to_numpy()[cidx]
    age = cust["age"].to_numpy()[cidx]
    score = cust["credit_score"].to_numpy()[cidx]       # COMPLETE score (missingness comes later)

    # loan amount: redraw until inside bounds and affordability cap, rounded to GBP 100
    shift = np.array(L["amount_purpose_shift"])[p_idx]
    base_ln = (np.log(L["amount_median"]) + L["amount_income_elasticity"] * np.log(income / L["amount_income_ref"]) + shift)
    eps = rng.standard_normal(n)
    for _ in range(500):
        amount = np.round(np.exp(base_ln + L["amount_log_sd"] * eps) / L["amount_round_to"]) * L["amount_round_to"]
        bad = (amount < L["amount_min"]) | (amount > L["amount_max"]) | (amount > L["amount_max_loan_to_income"] * income)
        if not bad.any():
            break
        eps[bad] = rng.standard_normal(int(bad.sum()))
    else:  # pragma: no cover
        raise RuntimeError("loan amount redraw did not converge")

    # origination date: uniform over the allowed window for this loan
    term_earliest = {t: earliest_origination_for_term(t, h_end).toordinal() for t in L["terms"]}
    age_earliest = {a: earliest_origination_for_age(int(a), t0).toordinal() for a in np.unique(age)}
    earliest = np.maximum.reduce([
        np.full(n, d["origination_floor"].toordinal()),
        np.array([term_earliest[int(t)] for t in term]),
        np.array([age_earliest[int(a)] for a in age]),
    ])
    latest = d["origination_latest"].toordinal()
    if (earliest > latest).any():
        raise RuntimeError("some loans have no feasible origination window")
    ordinal = earliest + np.floor(rng.random(n) * (latest - earliest + 1)).astype(int)
    orig = pd.Series(np.datetime64("1970-01-01") + (ordinal - EPOCH_ORDINAL).astype("timedelta64[D]")).astype("datetime64[ns]")

    # interest rate from the COMPLETE credit score (never from hidden effects or the outcome)
    z_score = (score - L["rate_score_centre"]) / L["rate_score_scale"]
    rate = (L["rate_base"] - L["rate_score_slope"] * z_score + np.array(L["rate_term_premium"])[t_idx]
            + L["rate_noise_sd"] * rng.standard_normal(n))
    rate = np.round(np.clip(rate, L["rate_min"], L["rate_max"]), 1)

    mob = months_on_book(orig, t0)
    balance = np.round(balance_after(amount, rate, term, mob), 2)

    return pd.DataFrame({
        "loan_id": [f"L{i + 1:06d}" for i in range(n)],
        "customer_id": cust["customer_id"].to_numpy()[cidx],
        "origination_date": orig,
        "loan_amount": amount.astype(int),
        "term_months": term.astype(int),
        "interest_rate": rate,
        "loan_purpose": purposes[p_idx],
        "outstanding_balance": balance,
    })


# --------------------------------------------------------------------------- stage 3: outcomes
def generate_outcomes(cfg: dict, cust: Customers, loans: pd.DataFrame, rng: np.random.Generator):
    m = cfg["default_model"]
    d = cfg["dates"]
    c = cust.complete.set_index("customer_id")
    feats = loans[["loan_id", "customer_id", "loan_amount", "interest_rate", "term_months", "loan_purpose"]].copy()
    for col in ["credit_score", "existing_debt", "annual_income", "employment_status",
                "employment_tenure_months", "age", "region"]:
        feats[col] = c.loc[loans["customer_id"], col].to_numpy()
    feats["months_on_book"] = months_on_book(loans["origination_date"], d["t0"])

    base = default_logit_without_intercept(feats, cfg)

    # hidden effects: customer-level (shared by a customer's loans) and loan-level
    pos = c.index.get_indexer(loans["customer_id"])
    u_c = rng.standard_normal(len(c))
    customer_effect_c = m["customer_effect_sd"] * (m["customer_latent_weight"] * cust.z_c
                                                   + m["customer_independent_weight"] * u_c)
    customer_effect = customer_effect_c[pos]
    loan_effect = m["loan_effect_sd"] * rng.standard_normal(len(loans))

    hidden = customer_effect + loan_effect
    alpha = solve_intercept(base + hidden, cfg["portfolio"]["target_prevalence"], m["alpha_bracket"])
    true_pd = expit(base + hidden + alpha)

    default = (rng.random(len(loans)) < true_pd).astype(int)
    horizon_days = (d["horizon_end"] - d["horizon_start"]).days + 1
    offset = rng.integers(0, horizon_days, size=len(loans))
    start = pd.Timestamp(d["horizon_start"])
    default_date = pd.Series(start + pd.to_timedelta(offset, unit="D")).astype("datetime64[ns]")
    default_date = default_date.where(default == 1)

    out = loans.copy()
    out["default_status"] = default
    out["default_date"] = default_date
    audit = pd.DataFrame({
        "loan_id": loans["loan_id"], "customer_id": loans["customer_id"],
        "customer_effect": customer_effect, "loan_effect": loan_effect, "true_pd_12m": true_pd,
    })
    return out, audit, alpha


# --------------------------------------------------------------------------- stage 5: missingness
def apply_missingness(cfg: dict, complete: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    miss = cfg["missingness"]
    n = len(complete)
    obs = complete.copy()
    retired = (obs["employment_status"] == "Retired").to_numpy()
    tenure_missing = (rng.random(n) < miss["tenure_rate_non_retired"]) & ~retired
    score_missing = rng.random(n) < miss["credit_score_rate"]
    obs["employment_tenure_months"] = obs["employment_tenure_months"].astype("Int64")
    obs["credit_score"] = obs["credit_score"].astype("Int64")
    obs.loc[tenure_missing, "employment_tenure_months"] = pd.NA
    obs.loc[score_missing, "credit_score"] = pd.NA
    return obs


# --------------------------------------------------------------------------- orchestration
@dataclass
class GeneratedData:
    customers_complete: pd.DataFrame
    customers: pd.DataFrame
    loans: pd.DataFrame
    audit: pd.DataFrame
    alpha: float


def generate_all(cfg: dict) -> GeneratedData:
    seeds = np.random.SeedSequence(cfg["run"]["master_seed"]).spawn(4)
    rng_customers, rng_loans, rng_outcomes, rng_missing = (np.random.default_rng(s) for s in seeds)

    cust = generate_customers(cfg, rng_customers)
    loans = generate_loans(cfg, cust.complete, rng_loans)
    loans, audit, alpha = generate_outcomes(cfg, cust, loans, rng_outcomes)
    observed = apply_missingness(cfg, cust.complete, rng_missing)
    return GeneratedData(cust.complete, observed, loans, audit, alpha)


# --------------------------------------------------------------------------- writing
def _fmt_dates(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d")


def loans_for_csv(loans: pd.DataFrame) -> pd.DataFrame:
    out = loans[LOAN_COLUMNS].copy()
    out["origination_date"] = _fmt_dates(out["origination_date"])
    out["default_date"] = _fmt_dates(out["default_date"])
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_outputs(cfg: dict, data: GeneratedData, root: Path) -> dict:
    """Write raw, audit, processed and sample files under root. Returns file hashes."""
    p = cfg["paths"]
    raw, audit_dir, processed, sample = (root / p[k] for k in ("raw_dir", "audit_dir", "processed_dir", "sample_dir"))
    for directory in (raw, audit_dir, processed, sample):
        directory.mkdir(parents=True, exist_ok=True)

    customers = data.customers[CUSTOMER_COLUMNS]
    loans = loans_for_csv(data.loans)
    customers.to_csv(raw / "customer_data.csv", index=False)
    loans.to_csv(raw / "loan_data.csv", index=False)
    data.audit[AUDIT_COLUMNS].to_csv(audit_dir / "generation_audit.csv", index=False, float_format="%.10g")

    # S2-1: joined, typed table with no derived features and no train/test split
    joined = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    joined.to_csv(processed / "loan_analysis_base.csv", index=False)

    # small sample: the first 100 customers and ALL of their loans
    sample_ids = customers["customer_id"].iloc[:100]
    s_cust = customers[customers["customer_id"].isin(sample_ids)]
    s_loans = loans[loans["customer_id"].isin(sample_ids)]
    full_counts = loans.groupby("customer_id").size().reindex(sample_ids)
    sample_counts = s_loans.groupby("customer_id").size().reindex(sample_ids)
    assert (full_counts == sample_counts).all(), "sample must keep all loans of each sampled customer"
    assert s_loans["customer_id"].isin(s_cust["customer_id"]).all()
    s_cust.to_csv(sample / "customer_data_sample.csv", index=False)
    s_loans.to_csv(sample / "loan_data_sample.csv", index=False)

    run_info = {
        "master_seed": cfg["run"]["master_seed"],
        "intercept_alpha": data.alpha,
        "n_customers": int(len(customers)),
        "n_loans": int(len(loans)),
        "target_prevalence": cfg["portfolio"]["target_prevalence"],
        "realised_default_rate": float(data.loans["default_status"].mean()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (audit_dir / "generation_run.json").write_text(json.dumps(run_info, indent=2))
    return {name: sha256_of(path) for name, path in {
        "customer_data.csv": raw / "customer_data.csv",
        "loan_data.csv": raw / "loan_data.csv",
        "generation_audit.csv": audit_dir / "generation_audit.csv",
    }.items()}


def run(config_path: str | Path, root: Path) -> GeneratedData:
    cfg = load_config(config_path)
    data = generate_all(cfg)
    hashes = write_outputs(cfg, data, root)
    print(f"Generated {len(data.customers):,} customers and {len(data.loans):,} loans (seed {cfg['run']['master_seed']}).")
    print(f"Intercept alpha = {data.alpha:.6f}; realised default rate = {data.loans['default_status'].mean():.4%}.")
    for name, h in hashes.items():
        print(f"  sha256 {name}: {h[:16]}...")
    return data
