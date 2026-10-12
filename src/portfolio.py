"""Portfolio risk: PD, EAD, LGD and illustrative expected loss (Blueprint 6).

Definitions (illustrative, not regulatory):
  PD  - model-estimated probability of reaching 90+ days past due within 12 months after T0
  EAD - outstanding_balance at T0 (no amortisation, drawdown or accrued interest over the horizon)
  LGD - flat 70% of EAD (synthetic assumption; no recovery timing, costs or discounting)
  EL  - PD x LGD x EAD per loan; portfolio EL is the sum.

v1 scoring: the selected final model (fitted on development data only) scores ALL loans.
Development-loan PDs are therefore in-sample; only the test-loan comparison of expected versus
observed defaults is independent evidence. Risk-band edges are fixed once from the baseline PDs
and reused in every scenario so loans visibly migrate between bands under stress.
The same helpers are reused by the scenario stage.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate import segment_frame
from features import build_features
from data_split import SPLIT_FILE

SEGMENT_TYPES = ["data_split", "risk_band", "loan_purpose", "term_months", "origination_year",
                 "employment_status", "income_band", "region"]
SUMMARY_COLUMNS = ["scenario_id", "segment_type", "segment_value", "n_loans", "total_ead", "expected_defaults",
                   "mean_pd", "ead_weighted_pd", "expected_loss", "el_rate", "observed_defaults",
                   "simulated_realised_loss", "result_type"]
LOAN_COLUMNS = ["loan_id", "customer_id", "data_split", "pd_12m", "risk_band", "ead", "lgd", "expected_loss",
                "default_status", "region", "loan_purpose", "term_months", "origination_year",
                "employment_status", "income_band"]


def load_selected_model(cfg: dict, root: Path):
    name = cfg["modelling"]["selected_model"]
    if name != "logistic":
        raise NotImplementedError(
            "Selected model is not the logistic baseline. The cross-fitted scoring contingency (Blueprint 6.2) "
            "is not implemented and needs the owner's approval before use.")
    return name, joblib.load(root / cfg["modelling"]["models_dir"] / f"{name}.joblib")


def score(model, raw: pd.DataFrame, cfg: dict) -> np.ndarray:
    """PD for every row of a raw joined frame (features are rebuilt from the raw fields)."""
    return model.predict_proba(build_features(raw, cfg))[:, 1]


def risk_band_edges(pd_values, cfg: dict) -> np.ndarray:
    """Inner edges from the BASELINE PD distribution (quantiles, unique values). Fixed once, reused."""
    return np.unique(np.quantile(np.asarray(pd_values, dtype=float), cfg["portfolio_risk"]["risk_band_quantiles"]))


def assign_risk_band(pd_values, edges) -> np.ndarray:
    """Right-closed bands B1 (lowest PD) ... Bk. Identical PDs always share a band."""
    idx = np.searchsorted(edges, np.asarray(pd_values, dtype=float), side="left")
    return np.array([f"B{i + 1}" for i in idx])


def loan_level(raw: pd.DataFrame, pd_values, edges, cfg: dict) -> pd.DataFrame:
    """Loan-level PD, band, EAD, LGD and expected loss for one set of PDs (baseline or a scenario)."""
    lgd = cfg["portfolio_risk"]["lgd"]
    ead = raw["outstanding_balance"].to_numpy(dtype=float)
    pdv = np.asarray(pd_values, dtype=float)
    return pd.DataFrame({"loan_id": raw["loan_id"].to_numpy(), "pd_12m": pdv,
                         "risk_band": assign_risk_band(pdv, edges), "ead": ead, "lgd": lgd,
                         "expected_loss": pdv * lgd * ead})


def with_attributes(ll: pd.DataFrame, raw: pd.DataFrame, split: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach customer, outcome, split and reporting-segment fields (always the UNSHOCKED T0 values)."""
    seg = segment_frame(raw, cfg)
    loans = pd.concat([ll.reset_index(drop=True), raw[["customer_id", "default_status", "region"]].reset_index(drop=True),
                       seg.drop(columns=["region"]).reset_index(drop=True)], axis=1)
    loans["data_split"] = split["data_split"].to_numpy()
    return loans[LOAN_COLUMNS]


def summarise(loans: pd.DataFrame, scenario_id: str, result_type: str = "baseline_estimate") -> pd.DataFrame:
    """Aggregate to portfolio and segment level. loans needs the LOAN_COLUMNS fields.

    result_type labels what the rows are: baseline_estimate (model PDs on the T0 book, not observed
    outcomes) or hypothetical_stress. observed_defaults and simulated_realised_loss are facts about the
    simulated outcomes and are only meaningful for the baseline rows.
    """
    d = loans.copy()
    d["portfolio"] = "all"
    d["pd_x_ead"] = d["pd_12m"] * d["ead"]
    d["realised_loss"] = d["default_status"] * d["lgd"] * d["ead"]
    parts = []
    for stype in ["portfolio"] + SEGMENT_TYPES:
        g = d.groupby(stype, sort=True).agg(
            n_loans=("loan_id", "size"), total_ead=("ead", "sum"), expected_defaults=("pd_12m", "sum"),
            mean_pd=("pd_12m", "mean"), pd_x_ead=("pd_x_ead", "sum"), expected_loss=("expected_loss", "sum"),
            observed_defaults=("default_status", "sum"), simulated_realised_loss=("realised_loss", "sum"))
        g["ead_weighted_pd"] = g["pd_x_ead"] / g["total_ead"]
        g["el_rate"] = g["expected_loss"] / g["total_ead"]
        g = g.reset_index().rename(columns={stype: "segment_value"})
        g["segment_type"] = stype
        g["scenario_id"] = scenario_id
        g["result_type"] = result_type
        g["segment_value"] = g["segment_value"].astype(str)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)[SUMMARY_COLUMNS]


def reconcile(loans: pd.DataFrame, summary: pd.DataFrame, raw: pd.DataFrame, split: pd.DataFrame, cfg: dict) -> int:
    """Consistency checks against the approved definitions. Raises AssertionError listing any failure."""
    problems, n = [], 0

    def check(ok: bool, text: str):
        nonlocal n
        n += 1
        if not ok:
            problems.append(text)

    lgd = cfg["portfolio_risk"]["lgd"]
    check(bool(loans[["pd_12m", "ead", "expected_loss"]].notna().all().all()), "missing values in loan-level PD, EAD or EL")
    check(bool(((loans["pd_12m"] > 0) & (loans["pd_12m"] < 1)).all()), "PD outside (0, 1)")
    check(bool((loans["ead"] > 0).all()), "non-positive EAD")
    check(np.allclose(loans["ead"].to_numpy(), raw["outstanding_balance"].to_numpy(dtype=float)), "EAD differs from outstanding_balance")
    check(bool((loans["lgd"] == lgd).all()), "LGD differs from the configured value")
    check(np.allclose(loans["expected_loss"], loans["pd_12m"] * lgd * loans["ead"], rtol=0, atol=1e-9), "EL != PD x LGD x EAD")
    check(len(loans) == loans["loan_id"].nunique() == len(raw), "loan count or uniqueness")

    tot = summary[summary.segment_type == "portfolio"].iloc[0]
    check(int(tot["n_loans"]) == len(loans), "portfolio n_loans")
    check(np.isclose(tot["total_ead"], loans["ead"].sum()), "portfolio total EAD")
    check(np.isclose(tot["expected_loss"], loans["expected_loss"].sum()), "portfolio EL")
    check(np.isclose(tot["expected_defaults"], loans["pd_12m"].sum()), "portfolio expected defaults")
    check(np.isclose(tot["el_rate"], tot["expected_loss"] / tot["total_ead"]), "portfolio EL rate")
    check(np.isclose(tot["ead_weighted_pd"], (loans["pd_12m"] * loans["ead"]).sum() / loans["ead"].sum()), "EAD-weighted PD")
    check(int(tot["observed_defaults"]) == int(loans["default_status"].sum()), "observed defaults")
    check(np.isclose(tot["simulated_realised_loss"], (loans["default_status"] * lgd * loans["ead"]).sum()), "simulated realised loss")
    for stype in SEGMENT_TYPES:                                    # every view must reconcile to the total
        s = summary[summary.segment_type == stype]
        check(int(s["n_loans"].sum()) == len(loans), f"{stype}: n_loans does not sum to the portfolio")
        for col in ("total_ead", "expected_loss", "expected_defaults", "simulated_realised_loss"):
            check(np.isclose(s[col].sum(), tot[col], rtol=1e-9), f"{stype}: {col} does not sum to the portfolio")
        check(int(s["observed_defaults"].sum()) == int(tot["observed_defaults"]), f"{stype}: observed defaults")
    counts = split["data_split"].value_counts()
    ds = summary[summary.segment_type == "data_split"].set_index("segment_value")["n_loans"]
    check(all(int(ds[k]) == int(counts[k]) for k in counts.index), "data_split counts differ from the split file")
    if problems:
        raise AssertionError("Portfolio reconciliation failed: " + "; ".join(problems))
    return n


def publish_baseline_summary(summary: pd.DataFrame, out_dir: Path) -> bool:
    """Write the baseline-only portfolio_summary.csv ONLY if no summary exists yet.

    portfolio_summary.csv is owned by the scenarios stage, which writes baseline plus every scenario. An
    existing file is never overwritten here, so re-running this stage cannot silently remove scenario rows.
    Returns True if the file was created.
    """
    path = out_dir / "portfolio_summary.csv"
    if path.exists():
        return False
    summary.to_csv(path, index=False, float_format="%.10g")
    return True


def run(cfg: dict, root: Path) -> dict:
    """Baseline portfolio tables (loan_predictions.csv and the reconciled baseline summary).

    portfolio_summary.csv is only created here when absent; the scenarios stage owns it afterwards."""
    proc, out = root / cfg["paths"]["processed_dir"], root / cfg["paths"]["output_dir"]
    raw = pd.read_csv(proc / "loan_analysis_base.csv")
    split = pd.read_csv(proc / SPLIT_FILE)
    assert (raw["loan_id"].to_numpy() == split["loan_id"].to_numpy()).all(), "split file not aligned"
    name, model = load_selected_model(cfg, root)

    pd_base = score(model, raw, cfg)
    edges = risk_band_edges(pd_base, cfg)
    loans = with_attributes(loan_level(raw, pd_base, edges, cfg), raw, split, cfg)
    summary = summarise(loans, "baseline")
    n_checks = reconcile(loans, summary, raw, split, cfg)

    loans.to_csv(out / "loan_predictions.csv", index=False, float_format="%.10g")
    created = publish_baseline_summary(summary, out)

    tot = summary[summary.segment_type == "portfolio"].iloc[0]
    print(f"Portfolio (baseline, model {name}, LGD {cfg['portfolio_risk']['lgd']:.0%}): {int(tot.n_loans):,} loans, "
          f"EAD GBP {tot.total_ead:,.0f}")
    print(f"  expected defaults {tot.expected_defaults:,.1f} vs observed {int(tot.observed_defaults)}; mean PD {tot.mean_pd:.4f}, "
          f"EAD-weighted PD {tot.ead_weighted_pd:.4f}")
    print(f"  expected loss GBP {tot.expected_loss:,.0f} (EL rate {tot.el_rate:.4%}); simulated realised loss GBP "
          f"{tot.simulated_realised_loss:,.0f}")
    print("  portfolio_summary.csv: " + ("created (baseline only; run the scenarios stage to add scenarios)" if created else
          "already exists and belongs to the scenarios stage; left unchanged (re-run scenarios to refresh it)"))
    print(f"  risk-band edges (baseline PD quintiles): {np.round(edges, 4).tolist()}; reconciliation checks executed and passed: {n_checks}")
    for r in summary[summary.segment_type == "data_split"].itertuples():
        print(f"  {r.segment_value}: {int(r.n_loans):,} loans, expected defaults {r.expected_defaults:,.1f} vs observed "
              f"{int(r.observed_defaults)}, EL GBP {r.expected_loss:,.0f}")
    # diagnostic: development-loan EL from in-sample PDs versus out-of-fold PDs (5% review trigger, not an automatic switch)
    oof = pd.read_csv(proc / "model_predictions.csv")
    oof = oof[(oof.model_id == name) & (oof.data_split == "dev_oof")].set_index("loan_id")["pd_12m"]
    dev = loans[loans["data_split"] == "dev"]
    el_in = float(dev["expected_loss"].sum())
    el_oof = float((oof.loc[dev["loan_id"]].to_numpy() * dev["lgd"] * dev["ead"]).sum())
    rel = (el_in - el_oof) / el_oof
    print(f"  diagnostic: development-loan EL in-sample GBP {el_in:,.0f} vs out-of-fold GBP {el_oof:,.0f}: "
          f"relative difference {rel:+.2%} ({'ABOVE' if abs(rel) > 0.05 else 'within'} the 5% review trigger)")
    return {"loans": loans, "summary": summary, "edges": edges, "rel_diff": rel}
