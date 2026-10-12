import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_data as gd  # noqa: E402
import validate_data as vd  # noqa: E402

STATISTICAL_GROUPS = {"target", "oracle"}


@pytest.fixture(scope="module")
def cfg():
    return gd.load_config(ROOT / "config" / "config.toml")


@pytest.fixture(scope="module")
def frames(cfg):
    data = gd.generate_all(cfg)
    return (data.customers[gd.CUSTOMER_COLUMNS].copy(), gd.loans_for_csv(data.loans),
            data.audit[gd.AUDIT_COLUMNS].copy())


def status(cfg, customers, loans, audit):
    return {r["check_id"]: r["status"] for r in vd.validate_frames(customers, loans, audit, cfg)}


def single_loan_customer(customers, loans):
    counts = loans.groupby("customer_id").size()
    cid = counts[counts == 1].index[0]
    return cid, loans.index[loans["customer_id"] == cid][0], customers.index[customers["customer_id"] == cid][0]


def test_clean_data_passes_all_structural_checks(cfg, frames):
    results = vd.validate_frames(*frames, cfg)
    failures = [r for r in results if r["group"] not in STATISTICAL_GROUPS and r["status"] == "FAIL"]
    assert failures == []
    assert {r["check_id"] for r in results if r["group"] in STATISTICAL_GROUPS} >= {"T01", "T02", "T03", "O01"}


# ---- approved age-at-origination examples (age 19 at T0, earliest permitted date is 31 Dec 2022)
@pytest.mark.parametrize("date_str,term,expected_c04", [
    ("2022-12-31", 36, "PASS"),    # boundary: allowed
    ("2022-12-30", 36, "FAIL"),    # one day too early: invalid (approved example)
    ("2021-06-15", 60, "FAIL"),    # clearly invalid; 60 months keeps the scheduled end after 31 Dec 2024
    ("2023-06-01", 36, "PASS"),
])
def test_age_rule_for_age_19(cfg, frames, date_str, term, expected_c04):
    customers, loans, audit = (f.copy() for f in frames)
    cid, li, ci = single_loan_customer(customers, loans)
    customers.loc[ci, "age"] = 19
    loans.loc[li, "term_months"] = term          # chosen so the scheduled end is after 31 Dec 2024: only the age rule can fail
    loans.loc[li, "origination_date"] = date_str
    s = status(cfg, customers, loans, audit)
    assert s["C04"] == expected_c04
    assert s["C03"] == "PASS"                    # the maturity rule is not what triggers the failure


def test_age_18_customer_cannot_hold_an_eligible_loan(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    cid, li, ci = single_loan_customer(customers, loans)
    customers.loc[ci, "age"] = 18
    loans.loc[li, "term_months"] = 36
    loans.loc[li, "origination_date"] = "2023-11-30"
    s = status(cfg, customers, loans, audit)
    assert s["C04"] == "FAIL" and s["R01"] == "FAIL"   # age support is 19-75 and the age rule fails


# ---- other deliberately broken examples
def test_24_month_loan_originated_30_dec_2022_fails_maturity_rule(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    cid, li, ci = single_loan_customer(customers, loans)
    customers.loc[ci, "age"] = 40
    loans.loc[li, "term_months"] = 24
    loans.loc[li, "origination_date"] = "2022-12-30"
    assert status(cfg, customers, loans, audit)["C03"] == "FAIL"
    loans.loc[li, "origination_date"] = "2022-12-31"
    assert status(cfg, customers, loans, audit)["C03"] == "PASS"


def test_duplicate_loan_id_and_orphan_customer(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    loans.loc[1, "loan_id"] = loans.loc[0, "loan_id"]
    assert status(cfg, customers, loans, audit)["K02"] == "FAIL"
    customers2, loans2, audit2 = (f.copy() for f in frames)
    loans2.loc[0, "customer_id"] = "C99999"
    s = status(cfg, customers2, loans2, audit2)
    assert s["K03"] == "FAIL" and s["K05"] == "FAIL"


def test_default_date_rules(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    idx = loans.index[loans["default_status"] == 1][0]
    loans.loc[idx, "default_date"] = "2025-02-01"
    assert status(cfg, customers, loans, audit)["C06"] == "FAIL"
    customers, loans, audit = (f.copy() for f in frames)
    idx0 = loans.index[loans["default_status"] == 0][0]
    loans.loc[idx0, "default_date"] = "2024-05-01"
    assert status(cfg, customers, loans, audit)["C05"] == "FAIL"


def test_tampered_balance_and_oversized_loan(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    loans.loc[5, "outstanding_balance"] = loans.loc[5, "outstanding_balance"] + 50.0
    assert status(cfg, customers, loans, audit)["L03"] == "FAIL"
    customers, loans, audit = (f.copy() for f in frames)
    cid = loans.loc[0, "customer_id"]
    customers.loc[customers["customer_id"] == cid, "annual_income"] = 12000.0
    loans.loc[0, "loan_amount"] = 30000
    assert status(cfg, customers, loans, audit)["L01"] == "FAIL"


def test_retired_rules(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    customers.loc[0, "employment_status"] = "Retired"
    customers.loc[0, "age"] = 50
    assert status(cfg, customers, loans, audit)["L04"] == "FAIL"


def test_audit_column_leak_is_detected(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    loans["true_pd_12m"] = audit["true_pd_12m"]
    s = status(cfg, customers, loans, audit)
    assert s["A01"] == "FAIL" and s["S02"] == "FAIL"


def test_oracle_outside_band_is_review_not_fail(cfg, frames):
    customers, loans, audit = (f.copy() for f in frames)
    audit["true_pd_12m"] = loans["default_status"].astype(float) * 0.98 + 0.01   # absurdly perfect oracle
    s = status(cfg, customers, loans, audit)
    assert s["O01"] == "REVIEW"
