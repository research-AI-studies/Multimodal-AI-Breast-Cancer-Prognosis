"""
Ablation studies — sequential feature masking to measure per-feature impact (Section 4.3, Table 4.3).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def ablation_study(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "mortality_5yr",
    features_to_mask: list = None,
    baseline_metrics: dict = None,
) -> pd.DataFrame:
    """Sequentially mask features and measure AUC/F1 degradation.

    Returns a DataFrame with columns: feature, auc, f1, auc_drop, f1_drop.
    """
    if features_to_mask is None:
        features_to_mask = [
            "comorb_uti", "comorb_diabetes", "comorb_gastrointestinal",
            "comorb_depression", "comorb_heart", "comorb_hypertension",
            "ql", "fa", "ef", "brbi", "brst", "pf",
        ]

    X_full = df[feature_cols].values.astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0)
    y = df[target_col].values.astype(int)

    # Baseline (full model)
    if baseline_metrics is None:
        skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                              random_state=config.RANDOM_SEED)
        aucs, f1s = [], []
        for tr, va in skf.split(X_full, y):
            m = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
            m.fit(X_full[tr], y[tr], verbose=False)
            p = m.predict_proba(X_full[va])[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            f1s.append(f1_score(y[va], (p >= 0.5).astype(int)))
        baseline_metrics = {"auc": np.mean(aucs), "f1": np.mean(f1s)}

    results = []
    for feat in features_to_mask:
        matching = [i for i, c in enumerate(feature_cols)
                    if c == feat or c.startswith(feat + "_") or feat + "_x_" in c]
        if not matching:
            continue

        X_masked = X_full.copy()
        X_masked[:, matching] = 0.0

        skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                              random_state=config.RANDOM_SEED)
        aucs, f1s = [], []
        for tr, va in skf.split(X_masked, y):
            m = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
            m.fit(X_masked[tr], y[tr], verbose=False)
            p = m.predict_proba(X_masked[va])[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            f1s.append(f1_score(y[va], (p >= 0.5).astype(int)))

        mean_auc = np.mean(aucs)
        mean_f1 = np.mean(f1s)
        results.append({
            "feature": feat,
            "auc": round(mean_auc, 4),
            "f1": round(mean_f1, 4),
            "auc_drop": round(baseline_metrics["auc"] - mean_auc, 4),
            "f1_drop": round(baseline_metrics["f1"] - mean_f1, 4),
        })
        print(f"[ablation] Mask {feat:30s}: AUC_drop={results[-1]['auc_drop']:.4f}, "
              f"F1_drop={results[-1]['f1_drop']:.4f}")

    return pd.DataFrame(results).sort_values("auc_drop", ascending=False).reset_index(drop=True)
