"""
RMSE Optimizer — PRO score regression for fatigue (fa) and brst (Section 3.3).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def train_pro_regressor(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "fa",
) -> tuple:
    """Gradient boosting regressor for continuous PRO score prediction."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X, y)
    preds = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    metrics = {"rmse": round(rmse, 2), "target": target_col}
    print(f"[rmse_opt] {target_col} RMSE={rmse:.2f}")
    return model, metrics


def evaluate_pro_regression(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    label: str = "",
) -> dict:
    X = np.nan_to_num(X, nan=0.0)
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    print(f"[rmse_opt] {label} RMSE={rmse:.2f}")
    return {"rmse": round(rmse, 2)}
