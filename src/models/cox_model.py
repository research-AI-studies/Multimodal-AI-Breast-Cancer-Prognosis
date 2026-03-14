"""
Cox Proportional Hazards model via lifelines (Section 3.3).
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def train_cox(
    df: pd.DataFrame,
    feature_cols: list,
    duration_col: str = "survival_time",
    event_col: str = "event",
    penalizer: float = 0.01,
) -> tuple:
    """Fit Cox PH and return (fitter, summary_df, metrics)."""
    surv_df = df[feature_cols + [duration_col, event_col]].copy()
    surv_df = surv_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Drop near-zero-variance columns to prevent convergence issues
    std = surv_df[feature_cols].std()
    keep = std[std > 1e-6].index.tolist()
    drop_cols = [c for c in feature_cols if c not in keep]
    if drop_cols:
        surv_df = surv_df.drop(columns=drop_cols)
        feature_cols = keep

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(surv_df, duration_col=duration_col, event_col=event_col)

    summary = cph.summary
    cindex = cph.concordance_index_
    print(f"[cox] C-index={cindex:.4f}, n_features={len(feature_cols)}")

    metrics = {"cindex": round(cindex, 4)}
    return cph, summary, metrics


def get_hazard_ratios(cph: CoxPHFitter, features: list = None) -> pd.DataFrame:
    """Extract hazard ratios, CIs, and p-values for specified features."""
    s = cph.summary.copy()
    s["HR"] = np.exp(s["coef"])
    s["HR_lower"] = np.exp(s["coef lower 95%"])
    s["HR_upper"] = np.exp(s["coef upper 95%"])
    if features:
        s = s.loc[s.index.isin(features)]
    return s[["HR", "HR_lower", "HR_upper", "p"]].sort_values("p")


def check_proportional_hazards(cph: CoxPHFitter, df: pd.DataFrame,
                                duration_col="survival_time",
                                event_col="event") -> pd.DataFrame:
    """Schoenfeld residuals test for proportional hazards assumption."""
    try:
        results = proportional_hazard_test(cph, df, time_transform="rank")
        return results.summary
    except Exception as e:
        print(f"[cox] PH test skipped: {e}")
        return pd.DataFrame()


def predict_risk_scores(cph: CoxPHFitter, df: pd.DataFrame,
                        feature_cols: list) -> np.ndarray:
    """Return partial hazard (risk score) for each patient."""
    X = df[feature_cols].copy()
    # Only keep columns the model was fitted on
    fitted_cols = [c for c in cph.summary.index if c in X.columns]
    X = X[fitted_cols]
    return cph.predict_partial_hazard(X).values.flatten()
