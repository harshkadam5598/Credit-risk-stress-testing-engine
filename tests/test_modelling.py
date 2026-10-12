import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_split as ds  # noqa: E402
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import generate_data as gd  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return gd.load_config(ROOT / "config" / "config.toml")


@pytest.fixture(scope="module")
def joined(cfg):
    c = copy.deepcopy(cfg)
    c["run"]["n_customers"] = 1200
    data = gd.generate_all(c)
    loans = gd.loans_for_csv(data.loans)
    return c, loans.merge(data.customers, on="customer_id", how="left")


# ---- features
def test_features_follow_allow_list_and_hand_calculation(joined):
    c, raw = joined
    X = ft.build_features(raw, c)
    assert list(X.columns) == ft.feature_columns(c)
    r, x = raw.iloc[0], X.iloc[0]
    assert x["loan_to_income"] == pytest.approx(r["loan_amount"] / r["annual_income"])
    assert x["debt_to_income"] == pytest.approx(r["existing_debt"] / r["annual_income"])
    inst = float(gd.annuity_payment(r["loan_amount"], r["interest_rate"], r["term_months"]))
    assert x["instalment_to_income"] == pytest.approx(12 * inst / r["annual_income"])
    assert x["log_annual_income"] == pytest.approx(np.log(r["annual_income"]))
    n_loans = (raw["customer_id"] == r["customer_id"]).sum()
    assert x["customer_active_loans"] == n_loans


def test_leakage_guard_rejects_forbidden_columns(joined):
    c, _ = joined
    for bad in ["default_status", "true_pd_12m", "outstanding_balance", "region", "customer_id"]:
        with pytest.raises(ValueError):
            ft.assert_no_leakage(list(c["features"]["numeric"]) + [bad], c)
    ft.assert_no_leakage(ft.feature_columns(c), c)      # the allow-list itself is clean


def test_features_do_not_use_outcomes(joined):
    c, raw = joined
    flipped = raw.copy()
    flipped["default_status"] = 1 - flipped["default_status"]
    flipped["default_date"] = "2024-06-01"
    pd.testing.assert_frame_equal(ft.build_features(raw, c), ft.build_features(flipped, c))


def test_shocked_income_changes_ratios_consistently(joined):
    c, raw = joined
    shocked = raw.copy()
    shocked["annual_income"] = shocked["annual_income"] * 0.9
    a, b = ft.build_features(raw, c), ft.build_features(shocked, c)
    assert (b["loan_to_income"] > a["loan_to_income"]).all()
    assert (b["instalment_to_income"] > a["instalment_to_income"]).all()


# ---- split
def test_split_is_customer_disjoint_and_balanced(joined):
    c, raw = joined
    split = ds.make_split(raw[["loan_id", "customer_id", "default_status"]], c)   # raises if a check fails
    r = ds.check_split(split, raw["default_status"].to_numpy(), c)
    assert r["customer_overlap"] == 0 and abs(r["test_share"] - 0.25) <= 0.02
    again = ds.make_split(raw[["loan_id", "customer_id", "default_status"]], c)
    assert split.equals(again)


def test_split_check_raises_when_customers_overlap(joined):
    c, raw = joined
    split = ds.make_split(raw[["loan_id", "customer_id", "default_status"]], c)
    multi = raw.groupby("customer_id").size()
    cid = multi[multi > 1].index[0]
    idx = split.index[split["customer_id"] == cid]
    split.loc[idx[0], "data_split"] = "test"
    split.loc[idx[1], "data_split"] = "dev"
    with pytest.raises(RuntimeError):
        ds.check_split(split, raw["default_status"].to_numpy(), c)


# ---- evaluation
def test_metrics_known_cases():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    m = ev.compute_metrics(y, p, None, (0.5,))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["capture_top50"] == 1.0
    assert m["observed_rate"] == 0.5 and m["mean_pred_pd"] == pytest.approx(0.5)


def test_weights_equal_duplicated_rows():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.15).astype(int); p = np.clip(0.1 + 0.3 * y * rng.random(300), 0.01, 0.95)
    w = rng.integers(0, 4, size=300)
    idx = np.repeat(np.arange(300), w)
    a, b = ev.compute_metrics(y, p, w), ev.compute_metrics(y[idx], p[idx])
    assert all(a[k] == pytest.approx(b[k], abs=1e-8) for k in a if not np.isnan(a[k]))


def test_bootstrap_keeps_customers_together_and_is_reproducible(cfg):
    ids = np.repeat(np.arange(50), [1, 2, 3, 2, 1] * 10)
    w = ev.cluster_weights(ids, cfg)
    assert w.shape == (cfg["evaluation"]["n_bootstrap"], len(ids))
    for cid in (0, 7, 33):                                  # all loans of a customer share one weight
        cols = np.where(ids == cid)[0]
        assert (w[:, cols] == w[:, [cols[0]]]).all()
    assert (w.reshape(len(w), -1)[:, np.unique(ids, return_index=True)[1]].sum(axis=1) == 50).all()  # 50 draws
    assert np.array_equal(w, ev.cluster_weights(ids, cfg))


def test_metric_rows_interval_status(cfg):
    rng = np.random.default_rng(5)
    ids = np.repeat(np.arange(150), 2)
    y = (rng.random(300) < 0.2).astype(int); p = np.clip(0.2 + 0.2 * y * rng.random(300), 0.01, 0.95)
    small = copy.deepcopy(cfg); small["evaluation"]["n_bootstrap"] = 100
    tbl = ev.metric_rows("m", "test", y, p, small, ev.cluster_weights(ids, small))
    auc = tbl[tbl.metric == "roc_auc"].iloc[0]
    assert auc["interval_status"] == "reported" and auc["ci_lower"] <= auc["value"] <= auc["ci_upper"]
    pt = ev.metric_rows("m", "cv_oof", y, p, small, None)
    assert (pt["interval_status"] == "not_applicable").all()
