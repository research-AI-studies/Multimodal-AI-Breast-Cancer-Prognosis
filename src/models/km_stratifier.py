"""
Kaplan-Meier risk stratifier — split patients into high/low risk (Section 3.3).
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def stratify_by_risk_score(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """Add a risk_group column: 'high' for top_pct, 'low' for rest."""
    threshold = np.percentile(risk_scores, (1 - top_pct) * 100)
    out = df.copy()
    out["risk_score"] = risk_scores
    out["risk_group"] = np.where(risk_scores >= threshold, "high", "low")
    n_high = (out["risk_group"] == "high").sum()
    print(f"[km_strat] High-risk: {n_high} ({n_high/len(out)*100:.1f}%), "
          f"Low-risk: {len(out)-n_high}")
    return out


def km_by_group(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = "survival_time",
    event_col: str = "event",
) -> dict:
    """Fit KM per group and compute log-rank test. Returns dict of results."""
    groups = df[group_col].unique()
    kmf_dict = {}
    for g in sorted(groups):
        mask = df[group_col] == g
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, duration_col],
            df.loc[mask, event_col],
            label=str(g),
        )
        kmf_dict[g] = kmf

    # Log-rank test (first two groups)
    g_list = sorted(groups)
    if len(g_list) >= 2:
        m1 = df[group_col] == g_list[0]
        m2 = df[group_col] == g_list[1]
        lr = logrank_test(
            df.loc[m1, duration_col], df.loc[m2, duration_col],
            df.loc[m1, event_col], df.loc[m2, event_col],
        )
        p_value = lr.p_value
    else:
        p_value = None

    # Median survival per group
    medians = {}
    for g, kmf in kmf_dict.items():
        med = kmf.median_survival_time_
        medians[g] = med if not np.isinf(med) else df[duration_col].max()

    result = {
        "kmf": kmf_dict,
        "medians": medians,
        "logrank_p": p_value,
    }
    if p_value is not None:
        print(f"[km_strat] {group_col}: medians={medians}, log-rank p={p_value:.4f}")
    return result


def compute_mortality_ratio(df: pd.DataFrame) -> float:
    """Mortality ratio between high-risk and low-risk groups."""
    high = df[df["risk_group"] == "high"]["mortality_5yr"].mean()
    low = df[df["risk_group"] == "low"]["mortality_5yr"].mean()
    ratio = high / low if low > 0 else float("inf")
    print(f"[km_strat] Mortality ratio (high/low): {ratio:.2f}")
    return ratio
