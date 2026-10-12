import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as ft  # noqa: E402
import generate_data as gd  # noqa: E402
import portfolio as pf  # noqa: E402
import scenarios as sc  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    cfg = gd.load_config(ROOT / "config" / "config.toml")
    c = copy.deepcopy(cfg); c["run"]["n_customers"] = 1500
    data = gd.generate_all(c)
    raw = gd.loans_for_csv(data.loans).merge(data.customers, on="customer_id", how="left")
    model = ft.make_logistic(c).fit(ft.build_features(raw, c), raw["default_status"])
    return c, raw, model


def test_scenario_file_is_the_approved_set():
    spec = sc.load_scenarios(ROOT)
    got = {s["id"]: (s["income_shock"], s["existing_debt_shock"]) for s in spec["scenario"]}
    assert got == {"baseline": (0.0, 0.0), "adverse": (-0.05, 0.10), "severe_adverse": (-0.12, 0.25)}
    assert spec["sensitivity"]["multipliers"] == [0.5, 1.0, 1.5, 2.0, 3.0]


def test_zero_shock_reproduces_baseline_exactly(setup):
    c, raw, model = setup
    pd_base = pf.score(model, raw, c)
    pd_zero = pf.score(model, sc.apply_shock(raw, 0.0, 0.0), c)
    assert np.array_equal(pd_base, pd_zero)                                 # exact, not just close
    edges = pf.risk_band_edges(pd_base, c)
    a, b = pf.loan_level(raw, pd_base, edges, c), pf.loan_level(raw, pd_zero, edges, c)
    pd.testing.assert_frame_equal(a, b)


def test_shock_changes_only_income_and_debt(setup):
    c, raw, _ = setup
    s = sc.apply_shock(raw, -0.12, 0.25)
    assert np.allclose(s["annual_income"], raw["annual_income"] * 0.88)
    assert np.allclose(s["existing_debt"], raw["existing_debt"] * 1.25)
    for col in ["credit_score", "outstanding_balance", "interest_rate", "term_months", "loan_amount", "age",
                "employment_status", "employment_tenure_months"]:
        assert raw[col].equals(s[col]), col
    assert sc.check_fixed_and_features(raw, s, c, -0.12, 0.25) > 0          # feature consistency check runs and passes


def test_stress_raises_pd_and_el_in_severity_order_with_fixed_ead_lgd_bands(setup):
    c, raw, model = setup
    pd_base = pf.score(model, raw, c)
    edges = pf.risk_band_edges(pd_base, c)
    out = {}
    for sid, (i, d) in {"baseline": (0, 0), "adverse": (-0.05, 0.10), "severe": (-0.12, 0.25)}.items():
        out[sid] = pf.loan_level(raw, pf.score(model, sc.apply_shock(raw, i, d), c), edges, c)
    el = {k: v["expected_loss"].sum() for k, v in out.items()}
    assert el["baseline"] < el["adverse"] < el["severe"]
    for k in ("adverse", "severe"):
        assert np.array_equal(out[k]["ead"], out["baseline"]["ead"]) and np.array_equal(out[k]["lgd"], out["baseline"]["lgd"])
        assert (out[k]["risk_band"] >= out["baseline"]["risk_band"]).mean() > 0.99    # fixed edges: migration is upward
        assert np.allclose(out[k]["expected_loss"], out[k]["pd_12m"] * out[k]["lgd"] * out[k]["ead"], rtol=0, atol=1e-9)


def test_sensitivity_multiplier_one_equals_severe(setup):
    c, raw, model = setup
    sev = pf.score(model, sc.apply_shock(raw, -0.12, 0.25), c)
    one = pf.score(model, sc.apply_shock(raw, 1.0 * -0.12, 1.0 * 0.25), c)
    assert np.array_equal(sev, one)


def test_assumptions_table_labels_held_constant_drivers():
    t = sc.assumptions_table(sc.load_scenarios(ROOT))
    assert list(t.columns) == sc.ASSUMPTION_COLUMNS
    held = t[t.shock_type == "held_constant"]
    assert {"credit_score", "ead", "lgd"} <= set(held["driver"]) and (held["shock_value"] == 0).all()
    assert set(t.loc[t.scenario_type == "hypothetical_stress", "scenario_id"]) == {"adverse", "severe_adverse"}


def test_unapproved_scenario_set_is_rejected(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "scenarios.toml").write_text(
        '[[scenario]]\nid="baseline"\nname="b"\ntype="baseline_reference"\nincome_shock=0.0\nexisting_debt_shock=0.0\ndescription="x"\n')
    with pytest.raises(ValueError):
        sc.load_scenarios(tmp_path)


def test_baseline_summary_never_overwrites_an_existing_file(tmp_path):
    summary = pd.DataFrame({"scenario_id": ["baseline"], "x": [1]})
    assert pf.publish_baseline_summary(summary, tmp_path) is True           # created when absent
    (tmp_path / "portfolio_summary.csv").write_text("scenario_id,x\nbaseline,1\nadverse,2\n")
    assert pf.publish_baseline_summary(summary, tmp_path) is False          # existing file is left alone
    assert "adverse" in (tmp_path / "portfolio_summary.csv").read_text()
