"""
Stacked Ensemble combining XGBoost and DNN predictions (Section 3.3).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def ensemble_predict(
    xgb_proba: np.ndarray,
    dnn_proba: np.ndarray,
    weights: dict = None,
) -> np.ndarray:
    """Weighted average of XGBoost and DNN probability predictions."""
    weights = weights or config.ENSEMBLE_WEIGHTS
    w_xgb = weights.get("xgboost", 0.55)
    w_dnn = weights.get("dnn", 0.45)
    total = w_xgb + w_dnn
    return (w_xgb * xgb_proba + w_dnn * dnn_proba) / total


def evaluate_ensemble(
    y_true: np.ndarray,
    xgb_proba: np.ndarray,
    dnn_proba: np.ndarray,
    weights: dict = None,
) -> dict:
    """Evaluate stacked ensemble metrics."""
    proba = ensemble_predict(xgb_proba, dnn_proba, weights)
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_true, proba)
    f1 = f1_score(y_true, preds)
    brier = brier_score_loss(y_true, proba)

    metrics = {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
    }
    print(f"[ensemble] AUC={auc:.4f}, F1={f1:.4f}, Brier={brier:.4f}")
    return metrics


def optimize_weights(
    y_true: np.ndarray,
    xgb_proba: np.ndarray,
    dnn_proba: np.ndarray,
    metric: str = "auc",
) -> dict:
    """Grid search over ensemble weights to maximize target metric."""
    best_score = -1
    best_w = 0.5
    for w in np.arange(0.1, 0.95, 0.05):
        combo = w * xgb_proba + (1 - w) * dnn_proba
        if metric == "auc":
            score = roc_auc_score(y_true, combo)
        elif metric == "brier":
            score = -brier_score_loss(y_true, combo)
        else:
            score = f1_score(y_true, (combo >= 0.5).astype(int))
        if score > best_score:
            best_score = score
            best_w = w

    weights = {"xgboost": round(best_w, 2), "dnn": round(1 - best_w, 2)}
    print(f"[ensemble] Optimized weights: {weights}, best {metric}={best_score:.4f}")
    return weights
