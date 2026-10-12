"""Stress testing (Blueprint 7). HYPOTHETICAL what-if exercises, not forecasts.

For each scenario: shock raw annual_income and existing_debt -> rebuild features with the approved
feature code -> re-score with the UNCHANGED selected logistic model -> EL with unchanged EAD and LGD.
credit_score, employment, loan terms, interest rate, EAD and LGD are never shocked. Risk bands use the
fixed edges derived once from the baseline PDs. Segments always use the unshocked T0 attributes.

Writes: scenario_assumptions.csv, scenario_results.csv, sensitivity_results.csv and rewrites
portfolio_summary.csv with every scenario (baseline rows first).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio as pf
from data_split import SPLIT_FILE
from features import build_features

SCENARIO_FILE = "config/scenarios.toml"
RESULT_COLUMNS = ["loan_id", "scenario_id", "pd_12m", "ead", "lgd", "expected_loss", "risk_band", "result_type"]
ASSUMPTION_COLUMNS = ["scenario_id", "scenario_name", "scenario_type", "driver", "shock_type", "shock_value", "description"]
SENSITIVITY_COLUMNS = ["multiplier", "income_shock", "existing_debt_shock", "n_loans", "total_ead", "expected_defaults",
                       "mean_pd", "ead_weighted_pd", "expected_loss", "el_rate", "expected_loss_change_vs_baseline",
                       "expected_loss_ratio_to_baseline", "result_type"]
HELD_CONSTANT = [("credit_score", "Not shocked: a bureau score already reflects income and debt behaviour, so shocking it too would double count (likely understates stress)."),
                 ("employment_status_and_tenure", "Not changed: avoids a second, unmodelled unemployment mechanism."),
                 ("loan_terms_and_interest_rate", "Not changed: loans are fixed-rate and fixed-term."),
                 ("ead", "Held at the T0 outstanding balance in every scenario."),
                 ("lgd", "Held at the configured flat LGD in every scenario.")]


def load_scenarios(root: Path) -> dict:
    with open(root / SCENARIO_FILE, "rb") as f:
        spec = tomllib.load(f)
    ids = [s["id"] for s in spec["scenario"]]
    if ids != ["baseline", "adverse", "severe_adverse"]:
        raise ValueError(f"unexpected scenario set {ids}; adding or changing scenarios needs the owner's approval")
    base = next(s for s in spec["scenario"] if s["id"] == "baseline")
    if base["income_shock"] != 0 or base["existing_debt_shock"] != 0:
        raise ValueError("the baseline scenario must have zero shocks")
    return spec


def apply_shock(raw: pd.DataFrame, income_shock: float, debt_shock: float) -> pd.DataFrame:
    """Relative shocks to the two raw inputs only. A zero shock multiplies by exactly 1.0 (identity)."""
    shocked = raw.copy()
    shocked["annual_income"] = raw["annual_income"].to_numpy(dtype=float) * (1.0 + income_shock)
    shocked["existing_debt"] = raw["existing_debt"].to_numpy(dtype=float) * (1.0 + debt_shock)
    return shocked


def check_fixed_and_features(raw, shocked, cfg, income_shock, debt_shock) -> int:
    """Feature consistency: only the shocked inputs and their derived ratios may change."""
    f0, f1 = build_features(raw, cfg), build_features(shocked, cfg)
    changed = {"log_annual_income", "loan_to_income", "debt_to_income", "instalment_to_income", "existing_debt"}
    if income_shock == 0:
        changed -= {"log_annual_income", "loan_to_income", "instalment_to_income"}
    if debt_shock == 0:
        changed -= {"debt_to_income", "existing_debt"}
    problems = []
    for c in f0.columns:
        same = f0[c].equals(f1[c])
        if c in changed:
            continue
        if not same:
            problems.append(f"{c} changed but must be fixed")
    if not np.array_equal(raw["credit_score"].to_numpy(dtype=float), shocked["credit_score"].to_numpy(dtype=float), equal_nan=True):
        problems.append("credit_score changed")
    if income_shock != 0:
        problems.append("log income not shifted by ln(1+shock)") if not np.allclose(
            f1["log_annual_income"] - f0["log_annual_income"], np.log1p(income_shock)) else None
        problems.append("loan_to_income not rebuilt from shocked income") if not np.allclose(
            f1["loan_to_income"], raw["loan_amount"] / (raw["annual_income"] * (1 + income_shock))) else None
    if debt_shock != 0:
        problems.append("debt_to_income not rebuilt from shocked inputs") if not np.allclose(
            f1["debt_to_income"], raw["existing_debt"] * (1 + debt_shock) / (raw["annual_income"] * (1 + income_shock))) else None
    if problems:
        raise AssertionError("Feature consistency failed: " + "; ".join(problems))
    return 4 + len(f0.columns)


def assumptions_table(spec: dict) -> pd.DataFrame:
    rows = []
    for s in spec["scenario"]:
        base = {"scenario_id": s["id"], "scenario_name": s["name"], "scenario_type": s["type"]}
        rows.append(dict(base, driver="annual_income", shock_type="relative_change", shock_value=s["income_shock"],
                         description=s["description"]))
        rows.append(dict(base, driver="existing_debt", shock_type="relative_change", shock_value=s["existing_debt_shock"],
                         description=s["description"]))
        for driver, text in HELD_CONSTANT:
            rows.append(dict(base, driver=driver, shock_type="held_constant", shock_value=0.0, description=text))
    return pd.DataFrame(rows)[ASSUMPTION_COLUMNS]


def run(cfg: dict, root: Path) -> dict:
    out = root / cfg["paths"]["output_dir"]
    proc = root / cfg["paths"]["processed_dir"]
    spec = load_scenarios(root)
    raw = pd.read_csv(proc / "loan_analysis_base.csv")
    split = pd.read_csv(proc / SPLIT_FILE)
    name, model = pf.load_selected_model(cfg, root)

    pd_base = pf.score(model, raw, cfg)
    edges = pf.risk_band_edges(pd_base, cfg)             # fixed once from the baseline, reused everywhere
    problems, n_checks = [], 0

    results, summaries, level, band_counts = [], [], {}, {}
    for s in spec["scenario"]:
        shocked = apply_shock(raw, s["income_shock"], s["existing_debt_shock"])
        n_checks += check_fixed_and_features(raw, shocked, cfg, s["income_shock"], s["existing_debt_shock"])
        pdv = pf.score(model, shocked, cfg)
        ll = pf.loan_level(raw, pdv, edges, cfg)          # EAD from the unshocked raw frame, LGD from config
        rtype = "baseline_estimate" if s["type"] == "baseline_reference" else "hypothetical_stress"
        loans = pf.with_attributes(ll, raw, split, cfg)
        summ = pf.summarise(loans, s["id"], rtype)
        if rtype != "baseline_estimate":                   # observed outcomes are facts, not scenario results
            summ[["observed_defaults", "simulated_realised_loss"]] = np.nan
        else:
            n_checks += pf.reconcile(loans, summ, raw, split, cfg)
        level[s["id"]] = ll
        band_counts[s["id"]] = ll["risk_band"].value_counts().sort_index()
        summaries.append(summ)
        results.append(ll.assign(scenario_id=s["id"], result_type=rtype)[RESULT_COLUMNS])

    base_ll = level["baseline"]
    # --- zero-shock equality and fixed variables
    if not np.array_equal(base_ll["pd_12m"].to_numpy(), pd_base):
        problems.append("zero-shock PDs are not identical to the baseline PDs")
    if not (out / "loan_predictions.csv").exists():
        raise FileNotFoundError("loan_predictions.csv is missing: run the portfolio stage before the scenarios stage")
    persisted = pd.read_csv(out / "loan_predictions.csv")
    if not (np.allclose(persisted["pd_12m"], base_ll["pd_12m"], rtol=1e-8, atol=1e-10)
            and np.allclose(persisted["expected_loss"], base_ll["expected_loss"], rtol=1e-8, atol=1e-9)):
        problems.append("baseline scenario differs from persisted loan_predictions.csv beyond CSV rounding")
    n_checks += 2
    for sid, ll in level.items():
        if not (np.array_equal(ll["ead"], base_ll["ead"]) and np.array_equal(ll["lgd"], base_ll["lgd"])):
            problems.append(f"{sid}: EAD or LGD changed")
        if not np.allclose(ll["expected_loss"], ll["pd_12m"] * ll["lgd"] * ll["ead"], rtol=0, atol=1e-9):
            problems.append(f"{sid}: EL != PD x LGD x EAD")
        if not np.array_equal(pf.assign_risk_band(ll["pd_12m"], edges), ll["risk_band"].to_numpy()):
            problems.append(f"{sid}: bands not assigned with the fixed edges")
        n_checks += 3
    if not np.array_equal(pf.risk_band_edges(base_ll["pd_12m"], cfg), edges):
        problems.append("fixed risk-band edges differ from the baseline quantiles")
    n_checks += 1

    # --- aggregation consistency: loan-level scenario results reconcile to the summary rows
    summary = pd.concat(summaries, ignore_index=True)
    res = pd.concat(results, ignore_index=True)
    for sid in level:
        s_sum = summary[summary.scenario_id == sid]
        r = res[res.scenario_id == sid]
        for stype in ["portfolio"] + pf.SEGMENT_TYPES:
            v = s_sum[s_sum.segment_type == stype]
            ok = (int(v["n_loans"].sum()) == len(r) and np.isclose(v["expected_loss"].sum(), r["expected_loss"].sum(), rtol=1e-9)
                  and np.isclose(v["expected_defaults"].sum(), r["pd_12m"].sum(), rtol=1e-9)
                  and np.isclose(v["total_ead"].sum(), r["ead"].sum(), rtol=1e-9))
            if not ok:
                problems.append(f"{sid}/{stype}: summary does not reconcile to loan-level results")
            n_checks += 1

    # --- ordering expected from the design (a violation would be reported, not hidden)
    el = {sid: float(ll["expected_loss"].sum()) for sid, ll in level.items()}
    if not (el["baseline"] < el["adverse"] < el["severe_adverse"]):
        problems.append(f"portfolio EL not increasing with severity: {el}")
    n_checks += 1

    # --- shock-severity sensitivity (hypothetical, approved multipliers only)
    sev = next(s for s in spec["scenario"] if s["id"] == spec["sensitivity"]["base_scenario"])
    sens_rows = []
    for m in spec["sensitivity"]["multipliers"]:
        i_s, d_s = m * sev["income_shock"], m * sev["existing_debt_shock"]
        pdv = pf.score(model, apply_shock(raw, i_s, d_s), cfg)
        ll = pf.loan_level(raw, pdv, edges, cfg)
        e = float(ll["expected_loss"].sum())
        sens_rows.append({"multiplier": m, "income_shock": i_s, "existing_debt_shock": d_s, "n_loans": len(ll),
                          "total_ead": float(ll["ead"].sum()), "expected_defaults": float(ll["pd_12m"].sum()),
                          "mean_pd": float(ll["pd_12m"].mean()),
                          "ead_weighted_pd": float((ll["pd_12m"] * ll["ead"]).sum() / ll["ead"].sum()),
                          "expected_loss": e, "el_rate": e / float(ll["ead"].sum()),
                          "expected_loss_change_vs_baseline": e - el["baseline"],
                          "expected_loss_ratio_to_baseline": e / el["baseline"], "result_type": "hypothetical_sensitivity"})
        if m == 1.0 and not np.array_equal(pdv, level[sev["id"]]["pd_12m"].to_numpy()):
            problems.append("multiplier 1.0 does not reproduce the severe-adverse scenario")
        n_checks += 1
    sens = pd.DataFrame(sens_rows)[SENSITIVITY_COLUMNS]
    if not sens["expected_loss"].is_monotonic_increasing:
        problems.append("sensitivity EL is not increasing with the multiplier")
    n_checks += 1

    if problems:
        raise AssertionError("Scenario checks failed: " + "; ".join(problems))

    assumptions_table(spec).to_csv(out / "scenario_assumptions.csv", index=False)
    res.to_csv(out / "scenario_results.csv", index=False, float_format="%.10g")
    summary.to_csv(out / "portfolio_summary.csv", index=False, float_format="%.10g")
    sens.to_csv(out / "sensitivity_results.csv", index=False, float_format="%.10g")

    # ---- inline report
    print(f"Scenarios (HYPOTHETICAL; {len(raw):,} loans; checks executed and passed: {n_checks}). Fixed band edges: {np.round(edges, 4).tolist()}")
    t = summary[summary.segment_type == "portfolio"].set_index("scenario_id")
    for sid in level:
        r = t.loc[sid]
        delta = r.expected_loss - t.loc["baseline", "expected_loss"]
        print(f"  {sid:15} expected defaults {r.expected_defaults:7.1f}  mean PD {r.mean_pd:.4f}  EAD-wtd PD {r.ead_weighted_pd:.4f}  "
              f"EL GBP {r.expected_loss:,.0f}  EL rate {r.el_rate:.4%}  change vs baseline GBP {delta:+,.0f} ({delta / t.loc['baseline', 'expected_loss']:+.1%})")
    base_pd = level["baseline"]["pd_12m"].to_numpy()
    for sid in ("adverse", "severe_adverse"):
        diff = level[sid]["pd_12m"].to_numpy() - base_pd
        print(f"  {sid}: PD fell for {(diff < 0).mean():.1%} of loans; median change {np.median(diff):+.4f}; "
              f"loans moving up at least one band: {(level[sid]['risk_band'] > level['baseline']['risk_band']).mean():.1%}")
    print("  risk-band loan counts (B1 lowest PD):")
    bc = pd.DataFrame(band_counts).fillna(0).astype(int)
    print("   " + bc.to_string().replace("\n", "\n   "))
    print("  shock-severity sensitivity (multiples of the severe-adverse shock vector; hypothetical):")
    for r in sens.itertuples():
        print(f"    x{r.multiplier}: income {r.income_shock:+.0%}, debt {r.existing_debt_shock:+.0%} -> expected defaults {r.expected_defaults:.1f}, "
              f"EL GBP {r.expected_loss:,.0f} ({r.expected_loss_ratio_to_baseline:.2f}x baseline)")
    return {"summary": summary, "sensitivity": sens}
