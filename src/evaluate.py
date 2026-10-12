"""Model evaluation: metrics and the customer-level cluster bootstrap (Blueprint 5.3).

Every metric accepts optional sample weights. A bootstrap resample is represented by integer
customer counts expanded to loan weights: a customer drawn k times gives all of their loans
weight k (identical to duplicating them k times) and a customer not drawn gets weight 0. The
1,000 weight vectors are generated once from a fixed stream and reused for every metric and
every model, so intervals are consistent and paired comparisons (Sprint 4) are possible.
The model is never refitted inside the bootstrap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12
ROW_COLUMNS = ["model_id", "dataset", "metric", "value", "ci_lower", "ci_upper", "n_loans", "n_defaults",
               "n_valid_resamples", "interval_status"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _w(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, dtype=float)


def weighted_ks(y, p, w=None) -> float:
    w = _w(y, w)
    _, inv = np.unique(p, return_inverse=True)
    pos = np.bincount(inv, weights=w * (y == 1))
    neg = np.bincount(inv, weights=w * (y == 0))
    if pos.sum() <= 0 or neg.sum() <= 0:
        return np.nan
    return float(np.max(np.abs(np.cumsum(pos) / pos.sum() - np.cumsum(neg) / neg.sum())))


def capture_rate(y, p, frac, w=None) -> float:
    """Share of defaults captured in the top `frac` of loans (by weight) ranked by predicted PD.

    Tie-safe: loans with identical predictions are never separated. The group of tied loans that
    straddles the cut-off is included proportionally, so exactly `frac` of the weight is selected.
    """
    w = _w(y, w)
    uniq, inv = np.unique(-np.asarray(p, dtype=float), return_inverse=True)     # descending PD groups
    gw = np.bincount(inv, weights=w)
    gd = np.bincount(inv, weights=w * (y == 1))
    if gd.sum() <= 0:
        return np.nan
    target = frac * gw.sum()
    before = np.cumsum(gw) - gw
    share = np.clip((target - before) / np.where(gw > 0, gw, 1.0), 0.0, 1.0)
    return float((share * gd).sum() / gd.sum())


def calibration_fit(y, p, w=None):
    """Calibration slope and intercept: unpenalised logistic regression of y on logit(p)."""
    w = _w(y, w)
    if (w * (y == 1)).sum() <= 0 or (w * (y == 0)).sum() <= 0:
        return np.nan, np.nan
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)   # C very large = effectively unpenalised
    lr.fit(_logit(p).reshape(-1, 1), y, sample_weight=w)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def compute_metrics(y, p, w=None, fractions=(0.10, 0.20)) -> dict:
    """All headline metrics. y in {0,1}, p predicted PD, w optional weights."""
    y = np.asarray(y); p = np.asarray(p, dtype=float); w = _w(y, w)
    tot = w.sum()
    both = (w * (y == 1)).sum() > 0 and (w * (y == 0)).sum() > 0
    prev = (w * y).sum() / tot
    brier = float((w * (p - y) ** 2).sum() / tot)
    pc = np.clip(p, EPS, 1 - EPS)
    out = {"mean_pred_pd": float((w * p).sum() / tot), "observed_rate": float(prev)}
    out["calibration_ratio"] = out["mean_pred_pd"] / prev if prev > 0 else np.nan
    out["brier"] = brier
    ref = prev * (1 - prev)
    out["brier_skill"] = 1 - brier / ref if ref > 0 else np.nan
    out["log_loss"] = float(-(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum() / tot)
    if both:
        auc = roc_auc_score(y, p, sample_weight=w)
        out["roc_auc"], out["gini"] = float(auc), float(2 * auc - 1)
        out["pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
        out["ks"] = weighted_ks(y, p, w)
    else:
        out.update({"roc_auc": np.nan, "gini": np.nan, "pr_auc": np.nan, "ks": np.nan})
    out["calibration_slope"], out["calibration_intercept"] = calibration_fit(y, p, w)
    for f in fractions:
        out[f"capture_top{int(round(f * 100))}"] = capture_rate(y, p, f, w)
    return out


def cluster_weights(customer_ids, cfg: dict) -> np.ndarray:
    """(n_bootstrap, n_loans) integer weights from resampling CUSTOMERS with replacement."""
    e = cfg["evaluation"]
    ids = np.asarray(customer_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng([cfg["run"]["master_seed"], e["bootstrap_stream"]])
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=e["n_bootstrap"])
    return counts[:, inv].astype(np.int16)


def metric_rows(model_id: str, dataset: str, y, p, cfg: dict, weights: np.ndarray | None = None) -> pd.DataFrame:
    """One row per metric. With `weights` (bootstrap) intervals are 95% percentile intervals."""
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    fr = cfg["evaluation"]["capture_fractions"]
    point = compute_metrics(y, p, None, fr)
    base = {"model_id": model_id, "dataset": dataset, "n_loans": int(len(y)), "n_defaults": int(y.sum())}
    boot = None
    if weights is not None:
        boot = pd.DataFrame([compute_metrics(y, p, w, fr) for w in weights])
    rows = []
    for name, value in point.items():
        row = dict(base, metric=name, value=value, ci_lower=np.nan, ci_upper=np.nan,
                   n_valid_resamples=np.nan, interval_status="not_applicable")
        if np.isnan(value):
            row["interval_status"] = "not_computable"
        elif boot is not None:
            vals = boot[name].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row["n_valid_resamples"] = int(len(vals))
            if len(vals) >= cfg["evaluation"]["valid_resample_threshold"] * len(weights):
                row["ci_lower"], row["ci_upper"] = np.percentile(vals, [2.5, 97.5])
                row["interval_status"] = "reported"
            else:
                row["interval_status"] = "invalid_resamples"
        rows.append(row)
    return pd.DataFrame(rows)[ROW_COLUMNS]


# ----------------------------------------------------------------------------- calibration bins
def fixed_bin_edges(p, n_bins: int) -> np.ndarray:
    """Inner edges = quantiles of the ORIGINAL test predictions, reduced to unique values (ties never split)."""
    return np.unique(np.quantile(np.asarray(p, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1]))


def assign_bins(p, edges) -> np.ndarray:
    """Right-closed bins: a prediction equal to an edge goes to the lower bin, so tied loans share a bin."""
    return np.searchsorted(edges, np.asarray(p, dtype=float), side="left")


def _interval(vals: np.ndarray, n_boot: int, cfg: dict):
    vals = vals[np.isfinite(vals)]
    if len(vals) >= cfg["evaluation"]["valid_resample_threshold"] * n_boot:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return lo, hi, len(vals), "reported"
    return np.nan, np.nan, len(vals), "invalid_resamples"


def calibration_table(model_id: str, dataset: str, y, p, weights: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Observed default rate per fixed bin with a customer-level bootstrap interval.

    Edges are set once from the original predictions and reused in every resample. A bin that is
    empty in a resample gives no value for that resample (excluded from the percentiles).
    """
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    edges = fixed_bin_edges(p, cfg["evaluation"]["calibration_bins"])
    b = assign_bins(p, edges)
    rows = []
    for k in range(len(edges) + 1):
        m = b == k
        wk = weights[:, m].astype(float)
        n_b = wk.sum(axis=1)
        d_b = (wk * y[m]).sum(axis=1)
        rate = np.where(n_b > 0, d_b / np.where(n_b > 0, n_b, 1), np.nan)
        lo, hi, n_valid, status = _interval(rate, len(weights), cfg)
        rows.append({"model_id": model_id, "dataset": dataset, "bin": k + 1, "n_loans": int(m.sum()),
                     "n_defaults": int(y[m].sum()), "mean_pred_pd": float(p[m].mean()),
                     "observed_rate": float(y[m].mean()), "ci_lower": lo, "ci_upper": hi,
                     "n_valid_resamples": n_valid, "interval_status": status})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- segments
def segment_frame(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reporting segments. Region is a reporting segment only, never a model feature."""
    edges = cfg["evaluation_segments"]["income_band_edges"]
    labels = ([f"<{edges[0] // 1000}k"] + [f"{a // 1000}-{b // 1000}k" for a, b in zip(edges[:-1], edges[1:])]
              + [f"{edges[-1] // 1000}k+"])
    seg = pd.DataFrame(index=base.index)
    seg["origination_year"] = pd.to_datetime(base["origination_date"]).dt.year.astype(str)
    seg["loan_purpose"] = base["loan_purpose"].astype(str)
    seg["term_months"] = base["term_months"].astype(str)
    seg["employment_status"] = base["employment_status"].astype(str)
    seg["income_band"] = pd.cut(base["annual_income"], [-np.inf] + list(edges) + [np.inf],
                                labels=labels, right=False).astype(str)
    seg["region"] = base["region"].astype(str)
    return seg


def segment_table(model_id: str, y, p, seg: pd.DataFrame, weights: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Test-set segment results from the SAME customer resamples as the whole test set.

    Eligibility (about 1,000 loans and 30 defaults) is decided once on the original test data.
    Ineligible segments keep counts and observed versus predicted PD, with AUC and interval suppressed.
    """
    e = cfg["evaluation"]
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    rows = []
    for stype in seg.columns:
        for value in sorted(seg[stype].unique()):
            m = (seg[stype] == value).to_numpy()
            row = {"model_id": model_id, "segment_type": stype, "segment_value": value, "n_loans": int(m.sum()),
                   "n_defaults": int(y[m].sum()), "mean_pd": float(p[m].mean()), "observed_rate": float(y[m].mean()),
                   "auc": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_valid_resamples": np.nan,
                   "interval_status": "insufficient_data"}
            if m.sum() >= e["segment_min_loans"] and y[m].sum() >= e["segment_min_defaults"]:
                row["auc"] = float(roc_auc_score(y[m], p[m]))
                ym = y[m]
                vals = np.array([roc_auc_score(ym, p[m], sample_weight=w)
                                 if (w[ym == 1].sum() > 0 and w[ym == 0].sum() > 0) else np.nan
                                 for w in weights[:, m]])
                row["ci_lower"], row["ci_upper"], row["n_valid_resamples"], row["interval_status"] = \
                    _interval(vals, len(weights), cfg)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- paired comparison
def paired_rows(y, p_base, p_chal, weights: np.ndarray, cfg: dict,
                base_id: str = "logistic", chal_id: str = "gradient_boosting") -> pd.DataFrame:
    """Challenger minus baseline on the SAME resampled customers (interval of the difference).

    auc_difference: positive favours the challenger. brier_difference and log_loss_difference:
    negative favours the challenger. Models are never compared by overlap of separate intervals.
    """
    y = np.asarray(y)
    names = ["auc_difference", "brier_difference", "log_loss_difference"]

    def stats(p, w):
        if (w * (y == 1)).sum() <= 0 or (w * (y == 0)).sum() <= 0:
            return np.full(3, np.nan)
        pc = np.clip(p, EPS, 1 - EPS)
        tot = w.sum()
        return np.array([roc_auc_score(y, p, sample_weight=w), (w * (p - y) ** 2).sum() / tot,
                         -(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum() / tot])

    ones = np.ones(len(y))
    point = stats(p_chal, ones) - stats(p_base, ones)
    boot = np.array([stats(p_chal, w) - stats(p_base, w) for w in weights])
    rows = []
    for i, name in enumerate(names):
        lo, hi, n_valid, status = _interval(boot[:, i], len(weights), cfg)
        rows.append({"model_id": f"{chal_id}_minus_{base_id}", "dataset": "test", "metric": name,
                     "value": float(point[i]), "ci_lower": lo, "ci_upper": hi, "n_loans": int(len(y)),
                     "n_defaults": int(y.sum()), "n_valid_resamples": n_valid, "interval_status": status})
    return pd.DataFrame(rows)[ROW_COLUMNS]


# ----------------------------------------------------------------------------- metric dictionary
METRIC_DEFINITIONS = [
    ("roc_auc", "Probability a random defaulter is ranked above a random non-defaulter", "higher is better"),
    ("gini", "2 x ROC-AUC - 1", "higher is better"),
    ("pr_auc", "Average precision: ranking quality among the rare defaults (depends on prevalence)", "higher is better"),
    ("ks", "Maximum gap between the score distributions of defaulters and non-defaulters", "higher is better"),
    ("brier", "Mean squared error of predicted PD against the 0/1 outcome", "lower is better"),
    ("brier_skill", "1 - Brier / Brier of always predicting the observed default rate", "higher is better"),
    ("log_loss", "Average negative log-likelihood of the outcomes; penalises confident errors", "lower is better"),
    ("mean_pred_pd", "Average predicted PD", "compare with observed_rate"),
    ("observed_rate", "Observed default rate", "reference"),
    ("calibration_ratio", "Mean predicted PD / observed default rate (calibration-in-the-large)", "1.0 is ideal"),
    ("calibration_slope", "Slope of a logistic regression of outcomes on logit(predicted PD)", "1.0 is ideal; below 1 means predictions too extreme"),
    ("calibration_intercept", "Intercept of the same regression", "0.0 is ideal"),
    ("capture_top10", "Share of all defaults found in the riskiest 10% of loans", "higher is better"),
    ("capture_top20", "Share of all defaults found in the riskiest 20% of loans", "higher is better"),
    ("auc_difference", "Challenger AUC minus baseline AUC on shared resamples", "positive favours the challenger"),
    ("brier_difference", "Challenger Brier minus baseline Brier on shared resamples", "negative favours the challenger"),
    ("log_loss_difference", "Challenger log loss minus baseline log loss on shared resamples", "negative favours the challenger"),
]


# ----------------------------------------------------------------------------- stage entry point
def run(cfg: dict, root: Path) -> None:
    """Calibration bins, segment stability and paired comparison, all on the held-out test set."""
    proc, out = root / cfg["paths"]["processed_dir"], root / cfg["paths"]["output_dir"]
    base = pd.read_csv(proc / "loan_analysis_base.csv").set_index("loan_id")
    pred = pd.read_csv(proc / "model_predictions.csv")
    test = pred[pred["data_split"] == "test"]
    order = test[test["model_id"] == "logistic"]["loan_id"].to_numpy()
    t = base.loc[order]
    y = t["default_status"].to_numpy()
    p = {m: test[test["model_id"] == m].set_index("loan_id").loc[order, "pd_12m"].to_numpy()
         for m in ("logistic", "gradient_boosting")}
    weights = cluster_weights(t["customer_id"].to_numpy(), cfg)

    cal = pd.concat([calibration_table(m, "test", y, p[m], weights, cfg) for m in p], ignore_index=True)
    cal.to_csv(out / "calibration_bins.csv", index=False, float_format="%.6g")
    selected = cfg["modelling"]["selected_model"]
    seg = segment_table(selected, y, p[selected], segment_frame(t.reset_index(), cfg), weights, cfg)
    seg.to_csv(out / "segment_performance.csv", index=False, float_format="%.6g")
    paired = paired_rows(y, p["logistic"], p["gradient_boosting"], weights, cfg)
    metrics = pd.read_csv(out / "model_metrics.csv")
    metrics = pd.concat([metrics[~metrics["model_id"].str.contains("_minus_")], paired], ignore_index=True)
    metrics.to_csv(out / "model_metrics.csv", index=False, float_format="%.6g")
    pd.DataFrame(METRIC_DEFINITIONS, columns=["metric", "definition", "direction"]).to_csv(
        out / "metric_definitions.csv", index=False)

    print("Calibration, test set, fixed quantile bins (observed rate with 95% customer-level bootstrap interval):")
    for m in p:
        c = cal[cal.model_id == m]
        print(f"  {m}: {len(c)} bins")
        for r in c.itertuples():
            print(f"    bin {int(r.bin)}: n={r.n_loans}, defaults={r.n_defaults}, predicted {r.mean_pred_pd:.4f}, "
                  f"observed {r.observed_rate:.4f} ({r.ci_lower:.4f} to {r.ci_upper:.4f})")
    elig = seg[seg.interval_status == "reported"]
    print(f"Segment stability ({selected}, test set): {len(elig)} of {len(seg)} segments eligible for AUC "
          f"(>= {cfg['evaluation']['segment_min_loans']} loans and >= {cfg['evaluation']['segment_min_defaults']} defaults).")
    for r in elig.itertuples():
        print(f"  {r.segment_type}={r.segment_value}: n={r.n_loans}, defaults={r.n_defaults}, AUC {r.auc:.3f} "
              f"({r.ci_lower:.3f} to {r.ci_upper:.3f}), mean PD {r.mean_pd:.3f} vs observed {r.observed_rate:.3f}")
    print("Paired comparison, gradient_boosting minus logistic, test set:")
    for r in paired.itertuples():
        print(f"  {r.metric}: {r.value:+.4f} (95% interval {r.ci_lower:+.4f} to {r.ci_upper:+.4f})")
