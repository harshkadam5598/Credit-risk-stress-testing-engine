import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate as ev  # noqa: E402
import generate_data as gd  # noqa: E402
import portfolio as pf  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return gd.load_config(ROOT / "config" / "config.toml")


# ---- calibration bins
def test_bin_edges_never_split_tied_predictions(cfg):
    p = np.array([0.01] * 60 + [0.02] * 20 + [0.05] * 10 + [0.30] * 10)
    edges = ev.fixed_bin_edges(p, 5)
    assert len(edges) == len(np.unique(edges))
    bins = ev.assign_bins(p, edges)
    for value in np.unique(p):
        assert len(np.unique(bins[p == value])) == 1          # identical predictions share one bin
    assert len(np.unique(bins)) < 5                            # fewer bins are kept rather than splitting ties


def test_calibration_table_counts_and_status(cfg):
    rng = np.random.default_rng(1)
    ids = np.repeat(np.arange(400), 2)
    p = rng.beta(1, 15, size=800)
    y = (rng.random(800) < p).astype(int)
    small = copy.deepcopy(cfg); small["evaluation"]["n_bootstrap"] = 200
    tbl = ev.calibration_table("m", "test", y, p, ev.cluster_weights(ids, small), small)
    assert tbl["n_loans"].sum() == 800 and tbl["n_defaults"].sum() == y.sum()
    assert (tbl["interval_status"] == "reported").all()


# ---- segments
def test_segment_eligibility_and_suppression(cfg):
    rng = np.random.default_rng(2)
    n = 3000
    seg = pd.DataFrame({"s": np.where(np.arange(n) < 2500, "big", "small")})
    p = rng.beta(1, 12, size=n); y = (rng.random(n) < p).astype(int)
    small = copy.deepcopy(cfg); small["evaluation"]["n_bootstrap"] = 100
    w = ev.cluster_weights(np.arange(n), small)
    tbl = ev.segment_table("m", y, p, seg, w, small).set_index("segment_value")
    assert tbl.loc["big", "interval_status"] == "reported" and not np.isnan(tbl.loc["big", "auc"])
    assert tbl.loc["small", "interval_status"] == "insufficient_data"
    assert np.isnan(tbl.loc["small", "auc"]) and np.isnan(tbl.loc["small", "ci_lower"])
    assert tbl.loc["small", "n_loans"] == 500                  # counts and observed vs predicted stay visible


# ---- paired comparison
def test_paired_rows_signs(cfg):
    rng = np.random.default_rng(4)
    n = 1500
    true_p = rng.beta(1, 12, size=n); y = (rng.random(n) < true_p).astype(int)
    good, bad = true_p, rng.permutation(true_p)                # challenger = shuffled (uninformative)
    small = copy.deepcopy(cfg); small["evaluation"]["n_bootstrap"] = 100
    w = ev.cluster_weights(np.arange(n), small)
    rows = ev.paired_rows(y, good, bad, w, small).set_index("metric")
    assert rows.loc["auc_difference", "value"] < 0             # challenger worse
    assert rows.loc["brier_difference", "value"] > 0 and rows.loc["log_loss_difference", "value"] > 0
    assert rows.loc["auc_difference", "ci_upper"] < 0


# ---- portfolio
def toy(cfg, n=40):
    rng = np.random.default_rng(7)
    raw = pd.DataFrame({"loan_id": [f"L{i}" for i in range(n)], "outstanding_balance": rng.uniform(1000, 20000, n)})
    pdv = rng.beta(1, 15, n) + 1e-4
    edges = pf.risk_band_edges(pdv, cfg)
    ll = pf.loan_level(raw, pdv, edges, cfg)
    loans = ll.assign(customer_id="C", default_status=(rng.random(n) < pdv).astype(int),
                      region=np.where(np.arange(n) % 2, "A", "B"), loan_purpose="x", term_months="36",
                      origination_year="2022", employment_status="e", income_band="b",
                      data_split=np.where(np.arange(n) < 30, "dev", "test"))[pf.LOAN_COLUMNS]
    split = pd.DataFrame({"data_split": loans["data_split"]})
    return raw, loans, split


def test_loan_level_el_and_bands(cfg):
    raw, loans, _ = toy(cfg)
    assert np.allclose(loans["expected_loss"], loans["pd_12m"] * 0.70 * loans["ead"])
    edges = pf.risk_band_edges(loans["pd_12m"], cfg)
    bands = pf.assign_risk_band(loans["pd_12m"], edges)
    order = np.argsort(loans["pd_12m"].to_numpy())
    assert list(bands[order]) == sorted(bands[order])           # higher PD never gets a lower band
    shifted = pf.assign_risk_band(loans["pd_12m"] * 3, edges)   # fixed edges: loans migrate upward under stress
    assert (shifted >= bands).all() and (shifted > bands).any()


def test_reconciliation_passes_and_detects_tampering(cfg):
    raw, loans, split = toy(cfg)
    summary = pf.summarise(loans, "baseline")
    assert pf.reconcile(loans, summary, raw, split, cfg) > 10
    bad = loans.copy(); bad.loc[0, "expected_loss"] += 1.0
    with pytest.raises(AssertionError):
        pf.reconcile(bad, pf.summarise(bad, "baseline"), raw, split, cfg)
    broken = summary.copy()
    broken.loc[(broken.segment_type == "region") & (broken.segment_value == "A"), "n_loans"] += 1
    with pytest.raises(AssertionError):
        pf.reconcile(loans, broken, raw, split, cfg)


def test_non_logistic_selection_is_blocked(cfg):
    c = copy.deepcopy(cfg); c["modelling"]["selected_model"] = "gradient_boosting"
    with pytest.raises(NotImplementedError):
        pf.load_selected_model(c, ROOT)
