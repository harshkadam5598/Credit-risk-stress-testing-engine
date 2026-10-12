"""Fit, compare and evaluate the PD models (Blueprint 5.1-5.3).

1. Development-set 5-fold grouped CV for the baseline (logistic) and the challenger
   (gradient boosting, fixed settings, no tuning) on the SAME folds.
2. Pre-specified selection criteria C1, C2, A1, K1, K2 (thresholds fixed in config before fitting).
   The challenger is only reported as ELIGIBLE; the model used downstream is cfg modelling.selected_model.
3. Refit both models on all development data, predict the held-out test set once.
4. Test metrics with customer-level cluster-bootstrap intervals.
Writes: models/*.joblib, data/processed/model_predictions.csv, outputs/model_selection.csv,
        outputs/model_metrics.csv, outputs/model_coefficients.csv.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import evaluate as ev
from data_split import SPLIT_FILE
from features import MODEL_FACTORIES, build_features

PRED_FILE = "model_predictions.csv"


def load_frame(cfg: dict, root: Path):
    proc = root / cfg["paths"]["processed_dir"]
    base = pd.read_csv(proc / "loan_analysis_base.csv")
    split = pd.read_csv(proc / SPLIT_FILE)
    if not (base["loan_id"].to_numpy() == split["loan_id"].to_numpy()).all():
        raise RuntimeError("split file is not aligned with the analysis table")
    X = build_features(base, cfg)                       # full portfolio first, then subset
    return base, split, X


def cross_validate(factory, X, y, groups, folds):
    """Out-of-fold predictions plus per-fold log loss and AUC."""
    oof = np.zeros(len(y))
    fold_rows = []
    for k, (tr, va) in enumerate(folds):
        model = factory()
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        m = ev.compute_metrics(y[va], oof[va])
        fold_rows.append({"fold": k + 1, "log_loss": m["log_loss"], "roc_auc": m["roc_auc"]})
    return oof, pd.DataFrame(fold_rows)


def selection_table(cfg, folds_b, folds_c, pooled_b, pooled_c) -> tuple[pd.DataFrame, bool]:
    s = cfg["model_selection"]
    d_ll = folds_b["log_loss"].to_numpy() - folds_c["log_loss"].to_numpy()      # positive favours the challenger
    d_auc = folds_c["roc_auc"].mean() - folds_b["roc_auc"].mean()
    dist = lambda x: abs(x - 1.0)  # noqa: E731
    crit = [
        ("C1", "folds where challenger log loss is better (of 5)", int((d_ll > 0).sum()), s["c1_min_folds_better"],
         int((d_ll > 0).sum()) >= s["c1_min_folds_better"]),
        ("C2", "mean log-loss improvement / baseline mean log loss", d_ll.mean() / folds_b["log_loss"].mean(),
         s["c2_min_relative_improvement"], d_ll.mean() / folds_b["log_loss"].mean() >= s["c2_min_relative_improvement"]),
        ("A1", "challenger minus baseline mean fold AUC", d_auc, -s["a1_max_auc_drop"], d_auc >= -s["a1_max_auc_drop"]),
        ("K1", "increase in |calibration slope - 1| (challenger minus baseline)",
         dist(pooled_c["calibration_slope"]) - dist(pooled_b["calibration_slope"]),
         s["k1_max_slope_distance_increase"],
         dist(pooled_c["calibration_slope"]) - dist(pooled_b["calibration_slope"]) <= s["k1_max_slope_distance_increase"]),
        ("K2", "increase in |mean PD / observed rate - 1| (challenger minus baseline)",
         dist(pooled_c["calibration_ratio"]) - dist(pooled_b["calibration_ratio"]),
         s["k2_max_ratio_distance_increase"],
         dist(pooled_c["calibration_ratio"]) - dist(pooled_b["calibration_ratio"]) <= s["k2_max_ratio_distance_increase"]),
    ]
    rows = []
    for m, df in (("logistic", folds_b), ("gradient_boosting", folds_c)):
        for r in df.itertuples():
            rows.append({"section": "fold_log_loss", "item": f"fold {r.fold}", "model": m, "value": r.log_loss, "threshold": np.nan, "met": ""})
            rows.append({"section": "fold_roc_auc", "item": f"fold {r.fold}", "model": m, "value": r.roc_auc, "threshold": np.nan, "met": ""})
    for cid, text, val, thr, met in crit:
        rows.append({"section": "criterion", "item": f"{cid}: {text}", "model": "gradient_boosting vs logistic",
                     "value": float(val), "threshold": float(thr), "met": bool(met)})
    eligible = all(c[4] for c in crit)
    rows.append({"section": "decision", "item": "challenger eligible (all of C1, C2, A1, K1, K2 met)",
                 "model": "gradient_boosting", "value": float(eligible), "threshold": np.nan, "met": bool(eligible)})
    return pd.DataFrame(rows), eligible


def run(cfg: dict, root: Path) -> dict:
    out_dir = root / cfg["paths"]["output_dir"]
    models_dir = root / cfg["modelling"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    base, split, X = load_frame(cfg, root)
    dev = (split["data_split"] == "dev").to_numpy()
    Xd, Xt = X[dev].reset_index(drop=True), X[~dev].reset_index(drop=True)
    yd, yt = base.loc[dev, "default_status"].to_numpy(), base.loc[~dev, "default_status"].to_numpy()
    gd, gt = base.loc[dev, "customer_id"].to_numpy(), base.loc[~dev, "customer_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=cfg["modelling"]["cv_folds"], shuffle=True, random_state=cfg["run"]["master_seed"])
    folds = list(sgkf.split(np.zeros(len(yd)), yd, groups=gd))
    names = ["logistic", "gradient_boosting"]
    oof, fold_tables, pooled = {}, {}, {}
    for name in names:
        factory = lambda n=name: MODEL_FACTORIES[n](cfg)  # noqa: E731
        oof[name], fold_tables[name] = cross_validate(factory, Xd, yd, gd, folds)
        pooled[name] = ev.compute_metrics(yd, oof[name], None, cfg["evaluation"]["capture_fractions"])

    sel, eligible = selection_table(cfg, fold_tables["logistic"], fold_tables["gradient_boosting"],
                                    pooled["logistic"], pooled["gradient_boosting"])
    sel.to_csv(out_dir / "model_selection.csv", index=False)

    # refit on all development data; the test set is predicted once
    fitted, test_pred = {}, {}
    for name in names:
        fitted[name] = MODEL_FACTORIES[name](cfg).fit(Xd, yd)
        test_pred[name] = fitted[name].predict_proba(Xt)[:, 1]
        joblib.dump(fitted[name], models_dir / f"{name}.joblib")

    pred_rows = []
    dev_ids, test_ids = base.loc[dev, "loan_id"].to_numpy(), base.loc[~dev, "loan_id"].to_numpy()
    fold_of = np.zeros(len(yd), dtype=int)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k + 1
    for name in names:
        pred_rows.append(pd.DataFrame({"loan_id": dev_ids, "model_id": name, "data_split": "dev_oof",
                                       "cv_fold": fold_of, "pd_12m": oof[name]}))
        pred_rows.append(pd.DataFrame({"loan_id": test_ids, "model_id": name, "data_split": "test",
                                       "cv_fold": 0, "pd_12m": test_pred[name]}))
    pd.concat(pred_rows).to_csv(root / cfg["paths"]["processed_dir"] / PRED_FILE, index=False, float_format="%.10g")

    # metrics: pooled out-of-fold (no interval) and held-out test (cluster bootstrap)
    weights = ev.cluster_weights(gt, cfg)
    metric_tables = []
    for name in names:
        metric_tables.append(ev.metric_rows(name, "cv_oof", yd, oof[name], cfg, None))
        metric_tables.append(ev.metric_rows(name, "test", yt, test_pred[name], cfg, weights))
    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics.to_csv(out_dir / "model_metrics.csv", index=False, float_format="%.6g")

    # standardised logistic coefficients (inputs are standardised, so magnitudes are comparable)
    lr = fitted["logistic"]
    coef = pd.DataFrame({"model_id": "logistic", "feature": lr.named_steps["prep"].get_feature_names_out(),
                         "coefficient": lr.named_steps["model"].coef_[0]})
    coef["odds_ratio"] = np.exp(coef["coefficient"])
    coef.to_csv(out_dir / "model_coefficients.csv", index=False, float_format="%.6g")

    # ---- inline summary
    print(f"Development CV ({cfg['modelling']['cv_folds']} grouped folds), pooled out-of-fold: "
          f"{len(yd):,} loans, {int(yd.sum())} defaults")
    for name in names:
        t = fold_tables[name]
        print(f"  {name:18} fold log loss {t.log_loss.mean():.4f} (sd {t.log_loss.std():.4f}), "
              f"fold AUC {t.roc_auc.mean():.4f} (sd {t.roc_auc.std():.4f}); pooled slope {pooled[name]['calibration_slope']:.3f}, "
              f"pooled mean PD / observed {pooled[name]['calibration_ratio']:.3f}")
    print("Selection criteria (challenger vs baseline):")
    for r in sel[sel.section == "criterion"].itertuples():
        print(f"  {'MET ' if r.met else 'NOT met'} {r.item} = {r.value:.4f} (threshold {r.threshold})")
    print(f"  challenger eligible: {eligible}; model used downstream: {cfg['modelling']['selected_model']} (owner-controlled)")
    t = metrics[(metrics.dataset == "test") & metrics.metric.isin(["roc_auc", "brier", "log_loss", "calibration_slope", "calibration_ratio"])]
    print("Held-out test (customer-level bootstrap 95% intervals):")
    for r in t.itertuples():
        print(f"  {r.model_id:18} {r.metric:18} {r.value:.4f} ({r.ci_lower:.4f} to {r.ci_upper:.4f}) [{r.interval_status}]")
    return {"eligible": eligible, "metrics": metrics, "coefficients": coef}
