"""
Evaluation metrics: AUC-ROC, F1, RMSE, C-index, Brier score, log-rank (Section 3.4).
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    mean_squared_error, roc_curve, precision_recall_curve,
)
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test


def compute_auc(y_true, y_proba):
    return round(roc_auc_score(y_true, y_proba), 4)


def compute_f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred), 4)


def compute_brier(y_true, y_proba):
    return round(brier_score_loss(y_true, y_proba), 4)


def compute_rmse(y_true, y_pred):
    return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2)


def compute_cindex(event_times, predicted_scores, event_observed):
    """Concordance index for survival predictions."""
    ci = concordance_index(event_times, -predicted_scores, event_observed)
    return round(ci, 4)


def compute_logrank(t1, t2, e1, e2):
    """Log-rank test between two survival groups."""
    result = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    return round(result.p_value, 4)


def compute_calibration(y_true, y_proba, n_bins=10):
    """Calibration slope and calibration-in-the-large."""
    from sklearn.linear_model import LogisticRegression

    # Calibration-in-the-large: mean(predicted) - mean(observed)
    cil = float(np.mean(y_proba) - np.mean(y_true))

    # Calibration slope via logistic recalibration
    try:
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(y_proba.reshape(-1, 1), y_true)
        slope = float(lr.coef_[0][0])
    except Exception:
        slope = 1.0

    # Bin-level calibration
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = []
    bin_means_obs = []
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_means_pred.append(y_proba[mask].mean())
            bin_means_obs.append(y_true[mask].mean())

    return {
        "calibration_slope": round(slope, 2),
        "calibration_in_the_large": round(abs(cil), 2),
        "bin_predicted": bin_means_pred,
        "bin_observed": bin_means_obs,
    }


def full_evaluation(y_true, y_proba, label=""):
    """Compute all classification metrics at once."""
    y_pred = (y_proba >= 0.5).astype(int)
    results = {
        "auc": compute_auc(y_true, y_proba),
        "f1": compute_f1(y_true, y_pred),
        "brier": compute_brier(y_true, y_proba),
    }
    cal = compute_calibration(y_true, y_proba)
    results.update(cal)
    if label:
        print(f"[metrics] {label}: AUC={results['auc']}, F1={results['f1']}, "
              f"Brier={results['brier']}, CalSlope={cal['calibration_slope']}")
    return results
