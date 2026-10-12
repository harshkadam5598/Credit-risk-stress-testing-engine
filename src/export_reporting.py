"""Reporting-table contract for the Power BI stage (Blueprint 8.4).

The stages already write the final tables into outputs/. This step verifies that every table exists with
the exact agreed columns, allowed category values and consistent totals, so Power BI can import them without
surprises and re-runs stay stable. It raises if any check fails. Only after the contract passes does it write
a compact, verified SAMPLE of each table to data/sample/reporting/ (the full tables are git-ignored; the
samples are tracked). The sample rows are exact copies of rows in the full outputs and are not a complete
dataset: they do not reconcile to full-portfolio totals.

Conventions: CSV, UTF-8, snake_case, ISO dates, probabilities as 0-1 decimals, GBP as plain numbers.
Empty cells mean "suppressed or not applicable" (never 0): see interval_status. Relationships for Power BI:
loan_predictions (one row per loan) is related one-to-many to scenario_results on loan_id; scenario_id links
scenario_results, portfolio_summary, scenario_assumptions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
import portfolio as pf
import scenarios as sc

TABLES = {
    "loan_predictions": pf.LOAN_COLUMNS,
    "scenario_results": sc.RESULT_COLUMNS,
    "scenario_assumptions": sc.ASSUMPTION_COLUMNS,
    "portfolio_summary": pf.SUMMARY_COLUMNS,
    "sensitivity_results": sc.SENSITIVITY_COLUMNS,
    "model_metrics": ev.ROW_COLUMNS,
    "calibration_bins": ["model_id", "dataset", "bin", "n_loans", "n_defaults", "mean_pred_pd", "observed_rate",
                         "ci_lower", "ci_upper", "n_valid_resamples", "interval_status"],
    "segment_performance": ["model_id", "segment_type", "segment_value", "n_loans", "n_defaults", "mean_pd",
                            "observed_rate", "auc", "ci_lower", "ci_upper", "n_valid_resamples", "interval_status"],
    "model_coefficients": ["model_id", "feature", "coefficient", "odds_ratio"],
    "metric_definitions": ["metric", "definition", "direction"],
}
SCENARIOS = {"baseline", "adverse", "severe_adverse"}
STATUSES = {"reported", "insufficient_data", "invalid_resamples", "not_computable", "not_applicable"}
RESULT_TYPES = {"baseline_estimate", "hypothetical_stress", "hypothetical_sensitivity"}


def run(cfg: dict, root: Path) -> int:
    out = root / cfg["paths"]["output_dir"]
    problems, n = [], 0

    def check(ok, text):
        nonlocal n
        n += 1
        if not bool(ok):
            problems.append(text)

    t = {}
    for name, cols in TABLES.items():
        path = out / f"{name}.csv"
        check(path.exists(), f"{name}.csv is missing")
        if not path.exists():
            continue
        t[name] = pd.read_csv(path)
        check(list(t[name].columns) == cols, f"{name}: columns differ from the contract")
        check(len(t[name]) > 0, f"{name}: empty")
    if problems:
        raise AssertionError("Reporting contract failed: " + "; ".join(problems))

    lp, sr, sm, ass, sens = (t[k] for k in ("loan_predictions", "scenario_results", "portfolio_summary",
                                            "scenario_assumptions", "sensitivity_results"))
    n_loans = len(lp)
    check(lp["loan_id"].is_unique, "loan_predictions: loan_id not unique")
    check(((lp["pd_12m"] > 0) & (lp["pd_12m"] < 1)).all(), "loan_predictions: PD outside (0, 1)")
    check(set(sr["scenario_id"]) == SCENARIOS and len(sr) == 3 * n_loans, "scenario_results: scenario set or row count")
    check(not sr.duplicated(["loan_id", "scenario_id"]).any(), "scenario_results: duplicate loan/scenario rows")
    check(set(sr["loan_id"]) == set(lp["loan_id"]), "scenario_results: loan_ids differ from loan_predictions")
    check(((sr["pd_12m"] > 0) & (sr["pd_12m"] < 1)).all(), "scenario_results: PD outside (0, 1)")
    check(set(sm["scenario_id"]) == SCENARIOS and set(ass["scenario_id"]) == SCENARIOS,
          "portfolio_summary or scenario_assumptions lacks the full scenario set (a baseline-only summary means the "
          "scenarios stage has not run since the portfolio stage created it: run the scenarios stage)")
    if problems:                                   # stop here: later checks assume the full scenario set
        raise AssertionError("Reporting contract failed: " + "; ".join(problems))
    check(set(sr["result_type"]) | set(sm["result_type"]) | set(sens["result_type"]) <= RESULT_TYPES, "unknown result_type")
    check(not sm.duplicated(["scenario_id", "segment_type", "segment_value"]).any(), "portfolio_summary: duplicate keys")
    # facts versus hypothetical results: observed outcomes appear only on baseline rows
    stressed = sm[sm["scenario_id"] != "baseline"]
    check(stressed[["observed_defaults", "simulated_realised_loss"]].isna().all().all(), "stressed rows carry observed outcomes")
    check(sm[sm["scenario_id"] == "baseline"][["observed_defaults", "simulated_realised_loss"]].notna().all().all(), "baseline rows miss observed outcomes")
    # baseline scenario equals the baseline loan table (within CSV rounding); EAD and LGD fixed
    b = sr[sr.scenario_id == "baseline"].set_index("loan_id").loc[lp["loan_id"]]
    check(np.allclose(b["pd_12m"].to_numpy(), lp["pd_12m"].to_numpy(), rtol=1e-8, atol=1e-10), "baseline scenario PD differs from loan_predictions")
    check(np.allclose(b["expected_loss"].to_numpy(), lp["expected_loss"].to_numpy(), rtol=1e-8, atol=1e-9), "baseline scenario EL differs from loan_predictions")
    for sid in SCENARIOS:
        r = sr[sr.scenario_id == sid].set_index("loan_id").loc[lp["loan_id"]]
        check(np.allclose(r["ead"].to_numpy(), lp["ead"].to_numpy()) and np.array_equal(r["lgd"].to_numpy(), lp["lgd"].to_numpy()), f"{sid}: EAD or LGD differs from baseline")
        tot = sm[(sm.scenario_id == sid) & (sm.segment_type == "portfolio")].iloc[0]
        check(np.isclose(tot["expected_loss"], r["expected_loss"].sum(), rtol=1e-8), f"{sid}: summary EL differs from loan-level")
        check(int(tot["n_loans"]) == n_loans, f"{sid}: summary loan count")
    # interval fields: suppressed values are empty, never zero or filled
    for name in ("model_metrics", "calibration_bins", "segment_performance"):
        df = t[name]
        check(set(df["interval_status"]) <= STATUSES, f"{name}: unknown interval_status")
        rep = df["interval_status"] == "reported"
        check(df.loc[rep, ["ci_lower", "ci_upper"]].notna().all().all(), f"{name}: reported rows lack intervals")
        check(df.loc[~rep, ["ci_lower", "ci_upper"]].isna().all().all(), f"{name}: suppressed rows carry interval values")
    seg = t["segment_performance"]
    check(seg.loc[seg.interval_status == "insufficient_data", "auc"].isna().all(), "segment_performance: AUC shown for insufficient data")
    mm = t["model_metrics"]
    paired = mm[mm["model_id"] == "gradient_boosting_minus_logistic"]
    check(set(paired["metric"]) == {"auc_difference", "brier_difference", "log_loss_difference"},
          "model_metrics: paired-comparison rows are missing (the train stage rewrites this file; re-run evaluate, then portfolio and scenarios)")
    check(set(mm.loc[mm["model_id"] != "gradient_boosting_minus_logistic", "dataset"]) <= {"cv_oof", "test"}, "model_metrics: unexpected dataset value")
    check(set(sens["multiplier"]) == {0.5, 1.0, 1.5, 2.0, 3.0}, "sensitivity: multipliers outside the approved set")
    if problems:
        raise AssertionError("Reporting contract failed: " + "; ".join(problems))
    print(f"Reporting contract: {len(TABLES)} tables present with the agreed columns; {n} checks executed and passed.")
    return n + write_and_verify_sample(cfg, root)


# ----------------------------------------------------------------------------- compact verified sample
SAMPLE_SUBDIR = ("data", "sample", "reporting")
SAMPLE_METRICS = ["roc_auc", "brier", "log_loss", "calibration_slope", "calibration_ratio"]
KEYS = {"loan_predictions": ["loan_id"], "scenario_results": ["loan_id", "scenario_id"],
        "scenario_assumptions": ["scenario_id", "driver"],
        "portfolio_summary": ["scenario_id", "segment_type", "segment_value"], "sensitivity_results": ["multiplier"],
        "model_metrics": ["model_id", "dataset", "metric"], "calibration_bins": ["model_id", "dataset", "bin"],
        "segment_performance": ["model_id", "segment_type", "segment_value"],
        "model_coefficients": ["model_id", "feature"], "metric_definitions": ["metric"]}


def _read_text(path: Path) -> pd.DataFrame:
    """Read as text so that sample rows are exact copies of the full-table rows."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _select_sample(name: str, full: pd.DataFrame, loan_ids: set) -> pd.DataFrame:
    if name in ("loan_predictions", "scenario_results"):
        return full[full["loan_id"].isin(loan_ids)]
    if name == "portfolio_summary":
        return full[full["segment_type"].isin(["portfolio", "data_split", "risk_band"])]
    if name == "model_metrics":
        paired = full["model_id"].str.contains("_minus_")
        return full[paired | full["metric"].isin(SAMPLE_METRICS)]
    if name == "segment_performance":
        rep = full[full["interval_status"] == "reported"]
        insufficient = full[full["interval_status"] == "insufficient_data"].head(4)
        return pd.concat([rep, insufficient]).loc[full.index.intersection(pd.concat([rep, insufficient]).index)]
    return full                                   # small tables are included whole


def write_and_verify_sample(cfg: dict, root: Path) -> int:
    out = root / cfg["paths"]["output_dir"]
    sdir = root.joinpath(*SAMPLE_SUBDIR)
    sdir.mkdir(parents=True, exist_ok=True)
    raw_sample = pd.read_csv(root / cfg["paths"]["sample_dir"] / "loan_data_sample.csv", usecols=["loan_id"])
    loan_ids = set(raw_sample["loan_id"])
    problems, n = [], 0

    def check(ok, text):
        nonlocal n
        n += 1
        if not bool(ok):
            problems.append(text)

    full_tables, samples = {}, {}
    for name, cols in TABLES.items():
        full = _read_text(out / f"{name}.csv")
        sample = _select_sample(name, full, loan_ids)
        sample.to_csv(sdir / f"{name}_sample.csv", index=False)
        full_tables[name], samples[name] = full, _read_text(sdir / f"{name}_sample.csv")   # re-read what was written

    for name, cols in TABLES.items():
        s, f = samples[name], full_tables[name]
        check(list(s.columns) == cols, f"{name}_sample: columns differ from the contract")
        check(len(s) > 0 and len(s) <= len(f), f"{name}_sample: size")
        check(len(s.merge(f, how="left", indicator=True).query("_merge != 'both'")) == 0, f"{name}_sample: rows are not exact copies of the full table")
        check(not s.duplicated(KEYS[name]).any(), f"{name}_sample: duplicate keys")
    lp, sr = samples["loan_predictions"], samples["scenario_results"]
    check(set(lp["loan_id"]) == loan_ids, "loan_predictions_sample must cover exactly the loans of data/sample/loan_data_sample.csv")
    check(set(sr["loan_id"]) == loan_ids and len(sr) == 3 * len(loan_ids), "scenario_results_sample: each sampled loan needs three scenario rows")
    check(set(sr["scenario_id"]) == SCENARIOS, "scenario_results_sample: scenario set")
    base = sr[sr["scenario_id"] == "baseline"].set_index("loan_id").loc[lp["loan_id"]]
    check(np.allclose(base["pd_12m"].astype(float), lp["pd_12m"].astype(float), rtol=1e-8, atol=1e-10), "sample baseline PD differs from loan_predictions_sample")
    for df, label in ((lp, "loan_predictions_sample"), (sr, "scenario_results_sample")):
        el = df["pd_12m"].astype(float) * df["lgd"].astype(float) * df["ead"].astype(float)
        check(np.allclose(el, df["expected_loss"].astype(float), rtol=1e-8, atol=1e-9), f"{label}: EL != PD x LGD x EAD")
    ps = samples["portfolio_summary"]
    check(set(ps["scenario_id"]) == SCENARIOS and set(ps["segment_type"]) == {"portfolio", "data_split", "risk_band"}, "portfolio_summary_sample: scenarios or segment types")
    check(set(samples["scenario_assumptions"]["scenario_id"]) == SCENARIOS, "scenario_assumptions_sample: scenario set")
    check(set(samples["sensitivity_results"]["multiplier"].astype(float)) == {0.5, 1.0, 1.5, 2.0, 3.0}, "sensitivity_results_sample: multipliers")
    for name in ("model_metrics", "calibration_bins", "segment_performance"):
        s = samples[name]
        rep = s["interval_status"] == "reported"
        check((s.loc[rep, ["ci_lower", "ci_upper"]] != "").all().all(), f"{name}_sample: reported rows lack intervals")
        check((s.loc[~rep, ["ci_lower", "ci_upper"]] == "").all().all(), f"{name}_sample: suppressed rows carry interval values")
    mm = samples["model_metrics"]
    check(mm["model_id"].str.contains("_minus_").sum() == 3, "model_metrics_sample: paired rows")
    seg = samples["segment_performance"]
    check((seg["interval_status"] == "insufficient_data").any() and (seg["interval_status"] == "reported").any(),
          "segment_performance_sample should show both reported and insufficient_data rows")
    if problems:
        raise AssertionError("Reporting sample failed: " + "; ".join(problems))
    kb = sum(p.stat().st_size for p in sdir.glob("*_sample.csv")) / 1024
    print(f"Reporting sample: {len(TABLES)} files in {'/'.join(SAMPLE_SUBDIR)} ({kb:.0f} KB; {len(lp)} loans, {len(sr)} scenario rows); {n} checks executed and passed.")
    return n
