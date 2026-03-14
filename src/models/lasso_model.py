"""
LASSO regression for feature selection — top-K predictors (Section 3.3).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def train_lasso(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "mortality_5yr",
    alpha: float = config.LASSO_ALPHA,
    top_k: int = config.LASSO_TOP_K,
) -> tuple:
    """L1-regularized logistic regression. Returns (model, top_features_df, metrics)."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    X = np.nan_to_num(X, nan=0.0)

    model = LogisticRegression(
        penalty="l1",
        C=1.0 / alpha,
        solver="saga",
        max_iter=5000,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X, y)

    coefs = model.coef_.flatten()
    feature_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    top_features = feature_df.head(top_k)
    print(f"[lasso] Top {top_k} features:")
    for _, row in top_features.iterrows():
        print(f"  {row['feature']:30s} coef={row['coefficient']:.4f}")

    metrics = {"n_nonzero": int((coefs != 0).sum())}
    return model, top_features, metrics
