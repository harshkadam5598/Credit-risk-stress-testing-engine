"""Focused tests for the pre-agreed model-selection criteria in train.selection_table.

Thresholds come from config (C1: 4 of 5 folds, C2: 1% relative improvement, A1: AUC drop 0.005,
K1: slope-distance increase 0.10, K2: ratio-distance increase 0.05). Inputs are built so that the
criterion under test is the only thing that varies. Exact float equality at C2, A1, K1 and K2 is avoided
(floating-point noise), so those are tested just inside and just outside the threshold; C1 is an integer
count and is tested exactly at the boundary.
"""
import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_data as gd  # noqa: E402
import train  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return gd.load_config(ROOT / "config" / "config.toml")


BASE_LL, BASE_AUC = 0.200, 0.850


def folds(ll, auc=BASE_AUC):
    ll = ll if isinstance(ll, (list, tuple)) else [ll] * 5
    auc = auc if isinstance(auc, (list, tuple)) else [auc] * 5
    return pd.DataFrame({"fold": range(1, 6), "log_loss": ll, "roc_auc": auc})


def pooled(slope=1.0, ratio=1.0):
    return {"calibration_slope": slope, "calibration_ratio": ratio}


def run(cfg, chal_ll=0.190, chal_auc=BASE_AUC, chal_slope=1.0, chal_ratio=1.0, base_slope=1.0, base_ratio=1.0):
    """Returns ({criterion: met}, eligible, table). Defaults make every criterion pass."""
    table, eligible = train.selection_table(cfg, folds(BASE_LL), folds(chal_ll, chal_auc),
                                            pooled(base_slope, base_ratio), pooled(chal_slope, chal_ratio))
    crit = table[table.section == "criterion"]
    met = {r.item.split(":")[0]: bool(r.met) for r in crit.itertuples()}
    return met, eligible, table


def test_all_criteria_pass_makes_challenger_eligible(cfg):
    met, eligible, table = run(cfg)
    assert met == {"C1": True, "C2": True, "A1": True, "K1": True, "K2": True}
    assert eligible is True
    decision = table[table.section == "decision"].iloc[0]
    assert bool(decision["met"]) is True and decision["value"] == 1.0


def test_c1_boundary_four_of_five_folds(cfg):
    four = [0.19, 0.19, 0.19, 0.19, 0.205]          # 4 folds better, 1 worse; mean gain is still above 1%
    met, eligible, _ = run(cfg, chal_ll=four)
    assert met["C1"] is True and met["C2"] is True and eligible is True
    three = [0.19, 0.19, 0.19, 0.205, 0.205]        # only 3 folds better
    met, eligible, _ = run(cfg, chal_ll=three)
    assert met["C1"] is False and eligible is False


def test_c1_ties_do_not_count_as_better(cfg):
    met, _, _ = run(cfg, chal_ll=[0.19, 0.19, 0.19, 0.20, 0.20])   # two folds exactly equal to the baseline
    assert met["C1"] is False


def test_c2_relative_improvement_threshold(cfg):
    assert run(cfg, chal_ll=0.1979)[0]["C2"] is True     # 1.05% improvement
    just_below = run(cfg, chal_ll=0.1981)                # 0.95% improvement
    assert just_below[0]["C2"] is False and just_below[0]["C1"] is True and just_below[1] is False


def test_c2_reads_threshold_from_config(cfg):
    strict = copy.deepcopy(cfg)
    strict["model_selection"]["c2_min_relative_improvement"] = 0.10
    assert run(strict, chal_ll=0.190)[0]["C2"] is False  # 5% gain passes at 1% but fails at 10%
    assert run(cfg, chal_ll=0.190)[0]["C2"] is True


def test_a1_auc_deterioration_threshold(cfg):
    assert run(cfg, chal_auc=BASE_AUC - 0.0049)[0]["A1"] is True
    worse = run(cfg, chal_auc=BASE_AUC - 0.0051)
    assert worse[0]["A1"] is False and worse[1] is False
    assert run(cfg, chal_auc=BASE_AUC + 0.02)[0]["A1"] is True        # an AUC gain never fails A1


def test_k1_calibration_slope_distance_applies_in_both_directions(cfg):
    assert run(cfg, chal_slope=1.09)[0]["K1"] is True
    assert run(cfg, chal_slope=0.91)[0]["K1"] is True
    assert run(cfg, chal_slope=1.11)[0]["K1"] is False
    assert run(cfg, chal_slope=0.89)[0]["K1"] is False
    # compared with the baseline, not with 1.0 alone: both equally off gives no increase
    assert run(cfg, base_slope=1.30, chal_slope=1.30)[0]["K1"] is True
    assert run(cfg, base_slope=1.30, chal_slope=1.38)[0]["K1"] is True
    assert run(cfg, base_slope=1.30, chal_slope=1.42)[0]["K1"] is False
    assert run(cfg, base_slope=1.30, chal_slope=1.00)[0]["K1"] is True   # a better-calibrated challenger passes


def test_k2_calibration_ratio_distance_applies_in_both_directions(cfg):
    assert run(cfg, chal_ratio=1.04)[0]["K2"] is True
    assert run(cfg, chal_ratio=0.96)[0]["K2"] is True
    assert run(cfg, chal_ratio=1.06)[0]["K2"] is False
    assert run(cfg, chal_ratio=0.94)[0]["K2"] is False
    assert run(cfg, base_ratio=1.10, chal_ratio=1.10)[0]["K2"] is True
    assert run(cfg, base_ratio=1.10, chal_ratio=1.00)[0]["K2"] is True


@pytest.mark.parametrize("failing,kwargs", [
    ("C1", dict(chal_ll=[0.19, 0.19, 0.19, 0.205, 0.205])),
    ("C2", dict(chal_ll=0.1981)),
    ("A1", dict(chal_auc=BASE_AUC - 0.0051)),
    ("K1", dict(chal_slope=1.11)),
    ("K2", dict(chal_ratio=1.06)),
])
def test_any_single_failure_blocks_eligibility(cfg, failing, kwargs):
    met, eligible, table = run(cfg, **kwargs)
    assert met[failing] is False
    assert all(v for k, v in met.items() if k != failing)             # the others still pass
    assert eligible is False
    decision = table[table.section == "decision"].iloc[0]
    assert bool(decision["met"]) is False and decision["value"] == 0.0


def test_disagreeing_metrics_keep_the_baseline(cfg):
    """Better log loss in every fold but worse AUC and worse calibration: not eligible."""
    met, eligible, _ = run(cfg, chal_ll=0.180, chal_auc=BASE_AUC - 0.02, chal_slope=1.3, chal_ratio=1.2)
    assert met["C1"] and met["C2"]
    assert not met["A1"] and not met["K1"] and not met["K2"]
    assert eligible is False


def test_table_layout_and_fold_rows(cfg):
    _, _, table = run(cfg)
    assert set(table.section) == {"fold_log_loss", "fold_roc_auc", "criterion", "decision"}
    assert (table.section == "fold_log_loss").sum() == 10 and (table.section == "fold_roc_auc").sum() == 10
    assert (table.section == "criterion").sum() == 5 and (table.section == "decision").sum() == 1
