"""
XGBoost classifier with Bayesian hyperparameter tuning via Optuna (Section 3.3).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (exclude targets, id, and raw-scale copies)."""
    exclude = {"id", "survival_time", "event", "mortality_5yr", "recurrence",
               "raw_survival_time", "max_followup"}
    return [c for c in df.columns
            if c not in exclude and not c.endswith("_raw")]


def train_xgboost(
    df: pd.DataFrame,
    target_col: str = "mortality_5yr",
    feature_cols: list = None,
    tune: bool = False,
    n_trials: int = 30,
) -> tuple:
    """Train XGBoost and return (model, metrics_dict)."""
    feature_cols = feature_cols or get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(int)

    params = dict(config.XGBOOST_PARAMS)

    if tune and HAS_OPTUNA:
        params = _optuna_tune(X, y, n_trials=n_trials)

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, preds)

    metrics = {"auc": round(auc, 4), "f1": round(f1, 4)}
    print(f"[xgboost] Train AUC={auc:.4f}, F1={f1:.4f}")
    return model, metrics, feature_cols


def _optuna_tune(X, y, n_trials=30):
    """Bayesian optimisation over config search space."""
    def objective(trial):
        p = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": config.RANDOM_SEED,
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", *config.XGBOOST_SEARCH_SPACE["learning_rate"], log=True),
            "max_depth": trial.suggest_int("max_depth", *config.XGBOOST_SEARCH_SPACE["max_depth"]),
            "subsample": trial.suggest_float("subsample", *config.XGBOOST_SEARCH_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *config.XGBOOST_SEARCH_SPACE["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *config.XGBOOST_SEARCH_SPACE["min_child_weight"]),
            "gamma": trial.suggest_float("gamma", *config.XGBOOST_SEARCH_SPACE["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *config.XGBOOST_SEARCH_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *config.XGBOOST_SEARCH_SPACE["reg_lambda"]),
        }
        model = xgb.XGBClassifier(**p)
        cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_SEED)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": config.RANDOM_SEED,
        "n_estimators": 500,
    })
    print(f"[xgboost] Optuna best AUC={study.best_value:.4f}")
    return best


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    imp = model.feature_importances_
    total = imp.sum()
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
        "importance_pct": imp / total if total > 0 else imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp
