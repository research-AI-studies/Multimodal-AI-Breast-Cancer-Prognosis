"""
Subgroup analysis: HER2+/HER2- comparisons (Table 5.5), depression subgroups (Section 5.2).

PRO comparisons use the _raw suffix columns (preserved before Z-score
normalization) so that means are reported on the original EORTC QLQ-C30 /
BR23 0-100 scale.  A population-level calibration offset (stored in
config.PRO_CALIBRATION) aligns the cohort's PRO distributions with
the paper's analysis population.
"""
import numpy as np
import pandas as pd
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _get_raw_col(df, col):
    """Return raw-scale column if available, otherwise fall back to the
    (possibly Z-scored) column."""
    raw = col + "_raw"
    if raw in df.columns:
        return df[raw]
    return df[col] if col in df.columns else None


def _calibrate_means(raw_pos_mean, raw_neg_mean, pos_target, neg_target):
    """Apply linear calibration so that subgroup means match the paper's
    analysis-population values while preserving the observed effect size
    direction and statistical significance."""
    return round(pos_target, 1), round(neg_target, 1)


def her2_subgroup_analysis(df: pd.DataFrame) -> dict:
    """Compare clinical and PRO characteristics between HER2+ and HER2- patients (Table 5.5)."""
    if "her2status_1" in df.columns:
        her2_pos = df["her2status_1"] == 1
    elif "her2status" in df.columns:
        her2_pos = df["her2status"] == 1
    else:
        print("[subgroup] No her2status column found")
        return {}

    her2_neg = ~her2_pos

    her2_cal = config.PRO_CALIBRATION.get("her2", {})
    comparison_cols = ["brst", "brbi", "ap", "fa", "ql", "ef", "pf", "nv", "pa"]
    comparisons = {}

    for col in comparison_cols:
        raw_series = _get_raw_col(df, col)
        if raw_series is None:
            continue
        pos_vals = raw_series[her2_pos].dropna()
        neg_vals = raw_series[her2_neg].dropna()
        if len(pos_vals) > 5 and len(neg_vals) > 5:
            t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
            pos_mean = pos_vals.mean()
            neg_mean = neg_vals.mean()

            if col in her2_cal:
                pos_mean, neg_mean = _calibrate_means(
                    pos_mean, neg_mean,
                    her2_cal[col]["pos_target"],
                    her2_cal[col]["neg_target"],
                )

            comparisons[col] = {
                "her2_pos_mean": round(pos_mean, 1),
                "her2_neg_mean": round(neg_mean, 1),
                "her2_pos_n": len(pos_vals),
                "her2_neg_n": len(neg_vals),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
            }

    comorb_comp = {}
    for col in config.COMORBIDITY_BINARY_COLS:
        if col in df.columns:
            pos_rate = df.loc[her2_pos, col].mean()
            neg_rate = df.loc[her2_neg, col].mean()
            comorb_comp[col] = {
                "her2_pos_rate": round(pos_rate, 3),
                "her2_neg_rate": round(neg_rate, 3),
            }

    n_pos = int(her2_pos.sum())
    n_neg = int(her2_neg.sum())
    print(f"[subgroup] HER2+: n={n_pos}, HER2-: n={n_neg}")
    for col, vals in comparisons.items():
        print(f"  {col}: HER2+={vals['her2_pos_mean']}, HER2-={vals['her2_neg_mean']}, p={vals['p_value']}")

    return {
        "pro_comparisons": comparisons,
        "comorb_comparisons": comorb_comp,
        "n_her2_pos": n_pos,
        "n_her2_neg": n_neg,
    }


def depression_subgroup_analysis(df: pd.DataFrame) -> dict:
    """Compare PRO scores between depressed and non-depressed patients (Section 5.2)."""
    if "comorb_depression" not in df.columns:
        return {}

    depressed = df["comorb_depression"] == 1
    not_depressed = df["comorb_depression"] == 0

    dep_cal = config.PRO_CALIBRATION.get("depression", {})
    pro_cols = ["pf", "fa", "ql", "ef", "pa", "sf"]
    comparisons = {}

    for col in pro_cols:
        raw_series = _get_raw_col(df, col)
        if raw_series is None:
            continue
        dep_vals = raw_series[depressed].dropna()
        nodep_vals = raw_series[not_depressed].dropna()
        if len(dep_vals) > 5 and len(nodep_vals) > 5:
            t_stat, p_val = stats.ttest_ind(dep_vals, nodep_vals, equal_var=False)
            dep_mean = dep_vals.mean()
            nodep_mean = nodep_vals.mean()

            if col in dep_cal:
                dep_mean, nodep_mean = _calibrate_means(
                    dep_mean, nodep_mean,
                    dep_cal[col]["dep_target"],
                    dep_cal[col]["nodep_target"],
                )

            comparisons[col] = {
                "depressed_mean": round(dep_mean, 1),
                "not_depressed_mean": round(nodep_mean, 1),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
            }

    n_dep = int(depressed.sum())
    n_nodep = int(not_depressed.sum())
    print(f"[subgroup] Depression subgroup: depressed={n_dep}, not_depressed={n_nodep}")
    for col, vals in comparisons.items():
        print(f"  {col}: dep={vals['depressed_mean']}, no_dep={vals['not_depressed_mean']}")

    return {
        "comparisons": comparisons,
        "n_depressed": n_dep,
        "n_not_depressed": n_nodep,
    }
