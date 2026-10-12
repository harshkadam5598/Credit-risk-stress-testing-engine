"""Customer-grouped 75/25 development/test split (Blueprint 5.2).

StratifiedGroupKFold has no test_size setting, so we use 4 folds and take the FIRST
(train, test) pair: its test fold (about 25% of loans) is the held-out test set.
Whole customers go to one side; default counts are balanced greedily, so the 25% share and
the default rate are only approximate. Three checks run after the split; if one fails we stop.
The label is written once and reused by every later stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_FILE = "loan_split.csv"


def make_split(loans: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """loans needs loan_id, customer_id, default_status. Returns loan_id, customer_id, data_split."""
    m = cfg["modelling"]
    sgkf = StratifiedGroupKFold(n_splits=m["test_folds"], shuffle=True, random_state=cfg["run"]["master_seed"])
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(loans)), loans["default_status"], groups=loans["customer_id"]))
    split = np.full(len(loans), "dev", dtype=object)
    split[test_idx] = "test"
    out = pd.DataFrame({"loan_id": loans["loan_id"].to_numpy(), "customer_id": loans["customer_id"].to_numpy(),
                        "data_split": split})
    check_split(out, loans["default_status"].to_numpy(), cfg)
    return out


def check_split(split: pd.DataFrame, y: np.ndarray, cfg: dict) -> dict:
    """Three split checks. Raises RuntimeError (stop and discuss) if any fails. Returns realised values."""
    m = cfg["modelling"]
    is_test = (split["data_split"] == "test").to_numpy()
    overlap = set(split.loc[is_test, "customer_id"]) & set(split.loc[~is_test, "customer_id"])
    share = float(is_test.mean())
    rate_gap = abs(float(y[is_test].mean()) - float(y.mean()))
    realised = {"customer_overlap": len(overlap), "test_share": share,
                "test_default_rate": float(y[is_test].mean()), "overall_default_rate": float(y.mean()),
                "test_loans": int(is_test.sum()), "test_defaults": int(y[is_test].sum()),
                "dev_loans": int((~is_test).sum()), "dev_defaults": int(y[~is_test].sum())}
    problems = []
    if overlap:
        problems.append(f"{len(overlap)} customers appear on both sides")
    if abs(share - 1 / m["test_folds"]) > m["split_test_share_tolerance"]:
        problems.append(f"test share {share:.3%} outside 25% +/- {m['split_test_share_tolerance']:.0%}")
    if rate_gap > m["split_default_rate_tolerance"]:
        problems.append(f"test default rate differs from overall by {rate_gap:.3%}")
    if problems:
        raise RuntimeError("Split check failed (stop and discuss; do not search over seeds): " + "; ".join(problems))
    return realised


def run(cfg: dict, root: Path) -> pd.DataFrame:
    proc = root / cfg["paths"]["processed_dir"]
    loans = pd.read_csv(proc / "loan_analysis_base.csv", usecols=["loan_id", "customer_id", "default_status"])
    split = make_split(loans, cfg)
    split.to_csv(proc / SPLIT_FILE, index=False)
    r = check_split(split, loans["default_status"].to_numpy(), cfg)
    print(f"Split written: dev {r['dev_loans']:,} loans / {r['dev_defaults']} defaults; test {r['test_loans']:,} loans / "
          f"{r['test_defaults']} defaults. Test share {r['test_share']:.2%}; test default rate "
          f"{r['test_default_rate']:.3%} vs overall {r['overall_default_rate']:.3%}; customer overlap {r['customer_overlap']}.")
    return split
