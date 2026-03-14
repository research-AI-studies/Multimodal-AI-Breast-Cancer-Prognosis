"""
5-fold stratified cross-validation and baseline model comparison (Section 4.1, Table 4.2).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
import xgboost as xgb
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _get_column_groups(all_cols: list) -> dict:
    """Partition feature columns into clinical-only, clinical+comorbidity, and full sets."""
    clinical_indicators = (
        ["age", "bmi", "bust", "cupsize", "menstruation_firsttime_age",
         "menopause_yn", "pregnancy_number", "birth_number", "pre_op",
         "weight", "height", "alcohol", "smokingstatus"]
    )
    clinical_ohe_prefixes = ["diagnosis_", "histotype_", "gradeinv",
                             "erstatus_", "prstatus_", "her2status_",
                             "marital_status_", "education"]

    comorbidity_prefixes = ["comorb_", "comorbidity_burden"]
    pro_prefixes = (
        ["ql", "pf", "rf", "ef", "cf", "sf", "fa", "nv", "pa",
         "dy", "sl", "ap", "co", "di", "fi",
         "brst", "brbi", "brbs", "brfu", "brsee", "brsef", "bras", "brhl"]
    )
    interaction_marker = "_x_"

    clinical_cols = []
    comorbidity_cols = []
    pro_cols = []

    for c in all_cols:
        is_clinical = (c in clinical_indicators or
                       any(c.startswith(p) for p in clinical_ohe_prefixes))
        is_comorb = any(c.startswith(p) for p in comorbidity_prefixes)
        is_pro = any(c == p or c.startswith(p + "_") for p in pro_prefixes)

        if interaction_marker in c:
            # Interaction terms go to full model only
            continue
        elif is_clinical:
            clinical_cols.append(c)
        elif is_comorb:
            comorbidity_cols.append(c)
        elif is_pro:
            pro_cols.append(c)

    interactions = [c for c in all_cols if interaction_marker in c]

    return {
        "model1": clinical_cols,
        "model2": clinical_cols + comorbidity_cols,
        "model3": clinical_cols + comorbidity_cols + pro_cols + interactions,
    }


def cross_validate_models(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "mortality_5yr",
) -> dict:
    """Run 5-fold stratified CV for Model 1/2/3 and return metrics dict."""
    exclude = {"id", "survival_time", "event", "mortality_5yr", "recurrence",
               "raw_survival_time", "risk_score", "risk_group"}
    all_features = [c for c in feature_cols if c not in exclude]

    col_groups = _get_column_groups(all_features)
    y = df[target_col].values.astype(int)
    skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)

    results = {}
    for model_name, cols in col_groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"[cv] {model_name}: no columns found, skipping")
            continue

        X = df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        fold_aucs, fold_f1s, fold_briers = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
            model.fit(X_tr, y_tr, verbose=False)

            proba = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)

            fold_aucs.append(roc_auc_score(y_val, proba))
            fold_f1s.append(f1_score(y_val, preds))
            fold_briers.append(brier_score_loss(y_val, proba))

        mean_auc = np.mean(fold_aucs)
        mean_f1 = np.mean(fold_f1s)
        mean_brier = np.mean(fold_briers)

        results[model_name] = {
            "auc": round(mean_auc, 4),
            "f1": round(mean_f1, 4),
            "brier": round(mean_brier, 4),
            "n_features": len(cols),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
        }
        print(f"[cv] {model_name}: AUC={mean_auc:.4f}, F1={mean_f1:.4f}, "
              f"Brier={mean_brier:.4f} ({len(cols)} features)")

    return results
