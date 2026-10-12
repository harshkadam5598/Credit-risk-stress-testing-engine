"""Feature construction and preprocessing for the PD models.

build_features() is deterministic and row-wise from T0 information, so it can be re-applied
to SHOCKED raw inputs in the stress-testing stage. Call it on the FULL portfolio frame and
subset afterwards: customer_active_loans is a count over the frame it is given.
Imputation, scaling and encoding are fitted inside scikit-learn Pipelines on training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_data import annuity_payment, months_on_book

RAW_REQUIRED = ["customer_id", "origination_date", "loan_amount", "term_months", "interest_rate", "loan_purpose",
                "age", "annual_income", "employment_status", "employment_tenure_months", "credit_score",
                "existing_debt"]


def feature_columns(cfg: dict) -> list[str]:
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])


def assert_no_leakage(columns, cfg: dict) -> None:
    bad = sorted(set(columns) & set(cfg["features"]["forbidden"]))
    if bad:
        raise ValueError(f"forbidden columns in the feature matrix: {bad}")


def build_features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw joined loan/customer frame -> allow-listed feature frame (same index)."""
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw frame is missing columns: {missing}")
    income = raw["annual_income"].astype(float)
    f = pd.DataFrame(index=raw.index)
    f["age"] = raw["age"].astype(float)
    f["log_annual_income"] = np.log(income)
    f["employment_tenure_months"] = pd.to_numeric(raw["employment_tenure_months"], errors="coerce")
    f["credit_score"] = pd.to_numeric(raw["credit_score"], errors="coerce")
    f["existing_debt"] = raw["existing_debt"].astype(float)
    f["loan_amount"] = raw["loan_amount"].astype(float)
    f["term_months"] = raw["term_months"].astype(float)
    f["interest_rate"] = raw["interest_rate"].astype(float)
    f["loan_to_income"] = f["loan_amount"] / income
    f["debt_to_income"] = f["existing_debt"] / income
    instalment = annuity_payment(raw["loan_amount"], raw["interest_rate"], raw["term_months"])
    f["instalment_to_income"] = 12.0 * instalment / income
    f["months_on_book"] = months_on_book(raw["origination_date"], cfg["dates"]["t0"]).astype(float)
    f["customer_active_loans"] = raw.groupby("customer_id")["customer_id"].transform("size").astype(float)
    f["employment_status"] = raw["employment_status"].astype(str)
    f["loan_purpose"] = raw["loan_purpose"].astype(str)
    out = f[feature_columns(cfg)]
    assert_no_leakage(out.columns, cfg)
    return out


def _preprocessor(cfg: dict, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), list(cfg["features"]["numeric"])),
         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
          list(cfg["features"]["categorical"]))],
        verbose_feature_names_out=False)


def make_logistic(cfg: dict) -> Pipeline:
    p = cfg["modelling"]["logistic"]
    return Pipeline([("prep", _preprocessor(cfg, scale=True)),
                     ("model", LogisticRegression(C=p["C"], max_iter=p["max_iter"]))])   # no class weights


def make_gradient_boosting(cfg: dict) -> Pipeline:
    p = cfg["modelling"]["gradient_boosting"]
    return Pipeline([("prep", _preprocessor(cfg, scale=False)),
                     ("model", HistGradientBoostingClassifier(
                         max_depth=p["max_depth"], learning_rate=p["learning_rate"], max_iter=p["max_iter"],
                         min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
                         early_stopping=False, random_state=cfg["run"]["master_seed"]))])


MODEL_FACTORIES = {"logistic": make_logistic, "gradient_boosting": make_gradient_boosting}
