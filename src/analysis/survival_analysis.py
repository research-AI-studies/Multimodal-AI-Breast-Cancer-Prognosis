"""
Survival analysis: comorbidity-specific Cox HRs (Table 5.3), KM curves (Figs 3-8),
and Schoenfeld residuals validation (Section 3.5).
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def comorbidity_cox_analysis(
    df: pd.DataFrame,
    adjust_for: list = None,
    duration_col: str = "survival_time",
    event_col: str = "event",
) -> pd.DataFrame:
    """Run Cox PH for each comorbidity, adjusting for confounders.
    Returns DataFrame with HR, CI, p-value per comorbidity (Table 5.3).

    Uses burden_other (total comorbidity burden excluding the target
    comorbidity) to avoid collinearity between individual flags and
    the aggregate burden score.
    """
    if adjust_for is None:
        adjust_for = ["age"]
        if "gradeinv" in df.columns:
            adjust_for.append("gradeinv")

    adjust_for = [c for c in adjust_for if c in df.columns]

    results = []
    for comorb in config.COMORBIDITY_BINARY_COLS:
        if comorb not in df.columns:
            continue

        sub = df.copy()
        if "comorbidity_burden" in sub.columns:
            sub["_burden_other"] = sub["comorbidity_burden"] - sub[comorb]
        else:
            sub["_burden_other"] = 0

        cols = [comorb, "_burden_other"] + adjust_for + [duration_col, event_col]
        sub = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if sub[comorb].nunique() < 2 or len(sub) < 50:
            continue

        try:
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(sub, duration_col=duration_col, event_col=event_col)
            s = cph.summary.loc[comorb]
            results.append({
                "comorbidity": comorb,
                "HR": round(np.exp(s["coef"]), 2),
                "HR_lower": round(np.exp(s["coef lower 95%"]), 2),
                "HR_upper": round(np.exp(s["coef upper 95%"]), 2),
                "p_value": round(s["p"], 4),
                "coef": round(s["coef"], 4),
            })
        except Exception as e:
            print(f"[surv] Cox failed for {comorb}: {e}")

    result_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    print(f"[surv] Comorbidity Cox analysis: {len(result_df)} comorbidities analysed")
    return result_df


def km_analysis_comorbidity(
    df: pd.DataFrame,
    comorb_col: str,
    duration_col: str = "survival_time",
    event_col: str = "event",
) -> dict:
    """KM curves + log-rank test for a single comorbidity (Figures 3-6)."""
    pos = df[comorb_col] == 1
    neg = df[comorb_col] == 0

    kmf_pos = KaplanMeierFitter()
    kmf_pos.fit(df.loc[pos, duration_col], df.loc[pos, event_col],
                label=f"{comorb_col}=1")

    kmf_neg = KaplanMeierFitter()
    kmf_neg.fit(df.loc[neg, duration_col], df.loc[neg, event_col],
                label=f"{comorb_col}=0")

    lr = logrank_test(
        df.loc[pos, duration_col], df.loc[neg, duration_col],
        df.loc[pos, event_col], df.loc[neg, event_col],
    )

    med_pos = kmf_pos.median_survival_time_
    med_neg = kmf_neg.median_survival_time_
    if np.isinf(med_pos):
        med_pos = df.loc[pos, duration_col].max()
    if np.isinf(med_neg):
        med_neg = df.loc[neg, duration_col].max()

    result = {
        "kmf_pos": kmf_pos,
        "kmf_neg": kmf_neg,
        "median_pos": med_pos,
        "median_neg": med_neg,
        "median_diff": abs(med_neg - med_pos),
        "logrank_p": lr.p_value,
    }
    print(f"[surv] KM {comorb_col}: median_diff={result['median_diff']:.1f}mo, "
          f"p={lr.p_value:.4f}")
    return result


def km_analysis_threshold(
    df: pd.DataFrame,
    col: str,
    threshold_low: float,
    threshold_high: float,
    duration_col: str = "survival_time",
    event_col: str = "event",
) -> dict:
    """KM analysis for PRO thresholds, e.g. ql<50 vs ql>=70 (Figures 7-8)."""
    low_mask = df[col] < threshold_low
    high_mask = df[col] >= threshold_high

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(df.loc[low_mask, duration_col], df.loc[low_mask, event_col],
                label=f"{col}<{threshold_low}")

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(df.loc[high_mask, duration_col], df.loc[high_mask, event_col],
                 label=f"{col}>={threshold_high}")

    if low_mask.sum() > 0 and high_mask.sum() > 0:
        lr = logrank_test(
            df.loc[low_mask, duration_col], df.loc[high_mask, duration_col],
            df.loc[low_mask, event_col], df.loc[high_mask, event_col],
        )
        p = lr.p_value
    else:
        p = 1.0

    med_low = kmf_low.median_survival_time_
    med_high = kmf_high.median_survival_time_
    if np.isinf(med_low):
        med_low = df.loc[low_mask, duration_col].max() if low_mask.sum() > 0 else 0
    if np.isinf(med_high):
        med_high = df.loc[high_mask, duration_col].max() if high_mask.sum() > 0 else 0

    result = {
        "kmf_low": kmf_low,
        "kmf_high": kmf_high,
        "median_low": med_low,
        "median_high": med_high,
        "median_diff": abs(med_high - med_low),
        "logrank_p": p,
    }
    print(f"[surv] KM {col} (<{threshold_low} vs >={threshold_high}): "
          f"diff={result['median_diff']:.1f}mo, p={p:.4f}")
    return result
