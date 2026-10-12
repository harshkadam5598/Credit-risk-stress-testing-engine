import copy
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_data as gd  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return gd.load_config(ROOT / "config" / "config.toml")


@pytest.fixture(scope="module")
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["run"]["n_customers"] = 500
    return c


@pytest.fixture(scope="module")
def small_data(small_cfg):
    return gd.generate_all(small_cfg)


# ---- date arithmetic
def test_add_months_clamps_to_month_end():
    assert gd.add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
    assert gd.add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)   # leap year
    assert gd.add_months(date(2022, 12, 31), 24) == date(2024, 12, 31)
    assert gd.add_months(date(2024, 12, 31), -24) == date(2022, 12, 31)


@pytest.mark.parametrize("term,expected", [
    (24, date(2022, 12, 31)), (36, date(2021, 12, 31)), (48, date(2020, 12, 31)), (60, date(2019, 12, 31))])
def test_earliest_origination_by_term(cfg, term, expected):
    h_end = cfg["dates"]["horizon_end"]
    got = gd.earliest_origination_for_term(term, h_end)
    assert got == expected
    assert gd.add_months(got, term) >= h_end                                  # inclusive end rule
    assert gd.add_months(date.fromordinal(got.toordinal() - 1), term) < h_end  # one day earlier fails


def test_earliest_origination_by_age(cfg):
    t0 = cfg["dates"]["t0"]
    assert gd.earliest_origination_for_age(19, t0) == date(2022, 12, 31)
    assert gd.earliest_origination_for_age(20, t0) == date(2021, 12, 31)
    assert gd.earliest_origination_for_age(18, t0) == date(2023, 12, 31)   # after the latest origination: infeasible


def test_months_on_book_whole_months(cfg):
    t0 = cfg["dates"]["t0"]
    s = pd.Series(pd.to_datetime(["2023-11-30", "2023-12-01", "2022-12-31", "2020-01-01"]))
    assert gd.months_on_book(s, t0).tolist() == [1, 0, 12, 47]


# ---- finance helpers
def test_annuity_known_values():
    assert gd.annuity_payment(7000, 12.0, 36) == pytest.approx(232.50, abs=0.005)
    assert gd.balance_after(7000, 12.0, 36, 12) == pytest.approx(4939.09, abs=0.005)
    assert gd.balance_after(7000, 12.0, 36, 0) == pytest.approx(7000.0)
    assert gd.balance_after(7000, 12.0, 36, 36) == pytest.approx(0.0, abs=1e-6)


# ---- tenure transformation (approved)
def test_tenure_transform_values_and_direction(cfg):
    months = np.array([0, 3, 12, 30, 60, 120, 240, 480])
    z = gd.tenure_z(months, np.zeros(len(months), dtype=bool), cfg)
    assert z[0] == pytest.approx(np.log(3 / 60))        # floored at 3
    assert z[1] == pytest.approx(-2.9957, abs=1e-3)
    assert z[4] == pytest.approx(0.0)
    assert z[6] == pytest.approx(np.log(4), abs=1e-9)   # 240 months
    assert z[7] == z[6]                                 # capped at 240
    assert np.all(np.diff(z) >= 0)                      # non-decreasing in tenure
    contribution = cfg["default_model"]["tenure_coefficient"] * z
    assert np.all(np.diff(contribution) <= 1e-12)       # risk never rises with tenure


def test_tenure_retired_is_neutral(cfg):
    z = gd.tenure_z(np.array([0, 0]), np.array([True, False]), cfg)
    assert z[0] == 0.0 and z[1] < 0


# ---- default formula
def _base_row():
    return pd.DataFrame([{
        "credit_score": 640, "existing_debt": 6000.0, "annual_income": 30000.0, "loan_amount": 7000.0,
        "interest_rate": 12.0, "term_months": 36, "employment_status": "Employed_FT",
        "employment_tenure_months": 60, "age": 45, "loan_purpose": "car_purchase", "region": "London",
        "months_on_book": 18}])


def test_formula_directions(cfg):
    base = _base_row()
    eta0 = gd.default_logit_without_intercept(base, cfg)[0]
    worse = {"credit_score": 520, "existing_debt": 20000.0, "loan_amount": 14000.0,
             "employment_tenure_months": 6, "annual_income": 20000.0}
    for col, val in worse.items():
        row = base.copy(); row[col] = val
        assert gd.default_logit_without_intercept(row, cfg)[0] > eta0, col
    better = {"credit_score": 760, "employment_tenure_months": 200}
    for col, val in better.items():
        row = base.copy(); row[col] = val
        assert gd.default_logit_without_intercept(row, cfg)[0] < eta0, col


def test_formula_matches_hand_calculation(cfg):
    row = _base_row()
    row["credit_score"] = 640; row["existing_debt"] = 0.0; row["annual_income"] = 30000.0
    row["loan_amount"] = 6900.0; row["employment_tenure_months"] = 60; row["age"] = 45
    row["months_on_book"] = 18; row["term_months"] = 36; row["loan_purpose"] = "car_purchase"; row["region"] = "London"
    m = cfg["default_model"]
    inst = float(gd.annuity_payment(6900.0, 12.0, 36))
    z_iti = (12 * inst / 30000.0 - m["iti_centre"]) / m["iti_scale"]
    z_lti = (6900.0 / 30000.0 - m["lti_centre"]) / m["lti_scale"]
    expected = (m["iti_coefficient"] * z_iti + m["lti_coefficient"] * z_lti       # z_s = 0, dti = 0, z_t = 0, age term 0
                + m["mob_amplitude"] * 1.0 + 0.0)                                  # offsets all 0 here, income term 0
    assert gd.default_logit_without_intercept(row, cfg)[0] == pytest.approx(expected, abs=1e-9)


def test_intercept_hits_target_prevalence(cfg):
    rng = np.random.default_rng(1)
    base = rng.normal(-1, 1.3, size=5000)
    alpha = gd.solve_intercept(base, 0.06, cfg["default_model"]["alpha_bracket"])
    assert expit(base + alpha).mean() == pytest.approx(0.06, abs=1e-9)


# ---- generation structure and reproducibility
def test_same_seed_same_data(small_cfg, small_data):
    again = gd.generate_all(small_cfg)
    pd.testing.assert_frame_equal(again.customers, small_data.customers)
    pd.testing.assert_frame_equal(again.loans, small_data.loans)
    pd.testing.assert_frame_equal(again.audit, small_data.audit)


def test_different_seed_changes_data(small_cfg, small_data):
    c = copy.deepcopy(small_cfg); c["run"]["master_seed"] += 1
    other = gd.generate_all(c)
    assert not other.customers["annual_income"].equals(small_data.customers["annual_income"])


def test_exact_loans_per_customer(small_cfg, small_data):
    counts = small_data.loans.groupby("customer_id").size().value_counts().sort_index().to_dict()
    assert counts == {1: 300, 2: 150, 3: 50}
    assert gd.loans_per_customer_counts(small_cfg) == [300, 150, 50]


def test_missingness_is_applied_last_and_does_not_change_outcomes(small_cfg, small_data):
    c = copy.deepcopy(small_cfg)
    c["missingness"]["credit_score_rate"] = 0.0
    c["missingness"]["tenure_rate_non_retired"] = 0.0
    no_missing = gd.generate_all(c)
    # loans (incl. interest rates priced from the COMPLETE score), outcomes and audit are identical
    pd.testing.assert_frame_equal(no_missing.loans, small_data.loans)
    pd.testing.assert_frame_equal(no_missing.audit, small_data.audit)
    assert no_missing.alpha == small_data.alpha
    assert small_data.customers["credit_score"].isna().any()
    assert not no_missing.customers["credit_score"].isna().any()
    # observed customers equal the complete customers wherever a value is present
    both = small_data.customers.dropna(subset=["credit_score"])
    assert (both["credit_score"].astype(int).to_numpy()
            == small_data.customers_complete.loc[both.index, "credit_score"].to_numpy()).all()


def test_interest_rate_uses_complete_score_even_when_observed_score_missing(cfg):
    c = copy.deepcopy(cfg)
    c["run"]["n_customers"] = 4000      # enough customers for dozens of loans with a missing observed score
    data = gd.generate_all(c)
    miss_ids = data.customers.loc[data.customers["credit_score"].isna(), "customer_id"]
    comp = data.customers_complete.set_index("customer_id")
    loans = data.loans[data.loans["customer_id"].isin(miss_ids)]
    assert len(loans) >= 50
    complete_score = comp.loc[loans["customer_id"], "credit_score"].to_numpy()
    # the rate still tracks the COMPLETE score strongly (design slope is negative)
    assert np.corrcoef(complete_score, loans["interest_rate"].to_numpy())[0, 1] < -0.7


def test_loans_obey_population_and_age_rules(small_cfg, small_data):
    d = small_cfg["dates"]
    loans = small_data.loans
    cust = small_data.customers_complete.set_index("customer_id")
    for row in loans.itertuples():
        o = row.origination_date.date()
        assert d["origination_floor"] <= o <= d["origination_latest"]
        assert gd.add_months(o, row.term_months) >= d["horizon_end"]
        age = int(cust.loc[row.customer_id, "age"])
        assert o >= gd.earliest_origination_for_age(age, d["t0"])
    assert cust["age"].min() >= small_cfg["customers"]["age_min"]


def test_defaults_and_dates_consistent(small_data, small_cfg):
    loans = small_data.loans
    d = small_cfg["dates"]
    dflt = loans["default_status"] == 1
    assert loans.loc[dflt, "default_date"].notna().all()
    assert loans.loc[~dflt, "default_date"].isna().all()
    dates = loans.loc[dflt, "default_date"].dt.date
    assert (dates >= d["horizon_start"]).all() and (dates <= d["horizon_end"]).all()
    assert ((small_data.audit["true_pd_12m"] > 0) & (small_data.audit["true_pd_12m"] < 1)).all()
