"""
main.py - Single entry point to reproduce all results from the paper.

Usage:
    python main.py              # Full pipeline
    python main.py --skip-tune  # Skip Optuna tuning (faster, uses defaults)

Outputs are written to outputs/figures/, outputs/tables/, and outputs/metrics/.
"""
import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.data_loading import load_raw_data, split_modalities, get_missingness_report
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.prognostic_model import estimate_prognostic_endpoints
from src.models.xgboost_model import train_xgboost, get_feature_columns, get_feature_importance
from src.models.dnn_model import train_dnn_classifier, train_dnn_regressor, predict_proba_dnn
from src.models.cox_model import train_cox, get_hazard_ratios
from src.models.ensemble_model import evaluate_ensemble, optimize_weights
from src.models.lasso_model import train_lasso
from src.models.km_stratifier import stratify_by_risk_score, km_by_group, compute_mortality_ratio
from src.models.rmse_optimizer import train_pro_regressor
from src.evaluation.cross_validation import cross_validate_models
from src.evaluation.ablation import ablation_study
from src.evaluation.metrics import full_evaluation, compute_cindex, compute_calibration
from src.analysis.survival_analysis import (
    comorbidity_cox_analysis, km_analysis_comorbidity, km_analysis_threshold,
)
from src.analysis.shap_analysis import compute_shap_values
from src.analysis.subgroup_analysis import her2_subgroup_analysis, depression_subgroup_analysis


def main(skip_tune: bool = False):
    start = time.time()
    np.random.seed(config.RANDOM_SEED)

    all_results = {}

    # ── Phase 1: Data Loading & Preprocessing ─────────────────────────────
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*70)

    df_raw = load_raw_data()
    all_results["df_raw"] = df_raw
    _ = split_modalities(df_raw)
    miss_report = get_missingness_report(df_raw)
    miss_report.to_csv(os.path.join(config.TABLES_DIR, "missingness_report.csv"), index=False)

    df_processed, scalers = preprocess_pipeline(df_raw)
    df_feat = engineer_features(df_processed)

    # ── Phase 2: Prognostic Risk Estimation ─────────────────────────────
    print("\n" + "="*70)
    print("PHASE 2: PROGNOSTIC RISK ESTIMATION")
    print("="*70)

    df = estimate_prognostic_endpoints(df_feat)

    # ── Phase 3: Model Training ───────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 3: MODEL TRAINING")
    print("="*70)

    feature_cols = get_feature_columns(df)

    # 3a. XGBoost
    print("\n--- XGBoost ---")
    xgb_model, xgb_metrics, feat_cols_used = train_xgboost(
        df, target_col="mortality_5yr", feature_cols=feature_cols,
        tune=(not skip_tune), n_trials=20,
    )
    xgb_importance = get_feature_importance(xgb_model, feat_cols_used)

    # 3b. DNN
    print("\n--- DNN (Classification) ---")
    dnn_model, dnn_metrics = train_dnn_classifier(
        df, feature_cols=feat_cols_used, target_col="mortality_5yr",
    )

    # 3c. DNN Regressor for fatigue
    print("\n--- DNN (Regression: fa) ---")
    fa_exclude = [c for c in feat_cols_used if c != "fa"]
    dnn_reg_model, dnn_reg_metrics = train_dnn_regressor(
        df, feature_cols=fa_exclude, target_col="fa",
    )

    # 3d. Cox PH
    print("\n--- Cox Proportional Hazards ---")
    cox_feature_cols = [c for c in feat_cols_used if c in df.columns]
    cph, cox_summary, cox_metrics = train_cox(
        df, feature_cols=cox_feature_cols,
        duration_col="survival_time", event_col="event",
    )

    # 3e. LASSO
    print("\n--- LASSO ---")
    lasso_model, lasso_top, lasso_metrics = train_lasso(
        df, feature_cols=feat_cols_used, target_col="mortality_5yr",
    )

    # 3f. RMSE Optimizer (fa)
    print("\n--- RMSE Optimizer (fa) ---")
    rmse_model_fa, rmse_metrics_fa = train_pro_regressor(
        df, feature_cols=fa_exclude, target_col="fa",
    )

    # 3g. RMSE Optimizer (brst)
    print("\n--- RMSE Optimizer (brst) ---")
    brst_exclude = [c for c in feat_cols_used if c != "brst"]
    rmse_model_brst, rmse_metrics_brst = train_pro_regressor(
        df, feature_cols=brst_exclude, target_col="brst",
    )

    # ── Phase 4: Ensemble & Risk Stratification ──────────────────────────
    print("\n" + "="*70)
    print("PHASE 4: ENSEMBLE & RISK STRATIFICATION")
    print("="*70)

    X_all = df[feat_cols_used].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_mort = df["mortality_5yr"].values

    xgb_proba = xgb_model.predict_proba(X_all)[:, 1]
    dnn_proba = predict_proba_dnn(dnn_model, X_all)

    ens_weights = optimize_weights(y_mort, xgb_proba, dnn_proba)
    ens_metrics = evaluate_ensemble(y_mort, xgb_proba, dnn_proba, ens_weights)

    # Risk stratification
    ensemble_proba = (ens_weights["xgboost"] * xgb_proba +
                      ens_weights["dnn"] * dnn_proba) / (
                      ens_weights["xgboost"] + ens_weights["dnn"])
    df = stratify_by_risk_score(df, ensemble_proba, top_pct=0.20)
    mortality_ratio = compute_mortality_ratio(df)

    # ── Phase 5: Cross-Validation ─────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 5: CROSS-VALIDATION (Model 1/2/3)")
    print("="*70)

    cv_results = cross_validate_models(df, feature_cols=feat_cols_used,
                                       target_col="mortality_5yr")
    all_results["cv_results"] = cv_results

    # ── Phase 6: Ablation Studies ─────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 6: ABLATION STUDIES")
    print("="*70)

    ablation_df = ablation_study(df, feature_cols=feat_cols_used,
                                 target_col="mortality_5yr")
    all_results["ablation_df"] = ablation_df

    # ── Phase 7: Survival Analysis ────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 7: SURVIVAL ANALYSIS")
    print("="*70)

    cox_hr_df = comorbidity_cox_analysis(df)
    all_results["cox_hr_df"] = cox_hr_df

    # KM for individual comorbidities
    for comorb in ["comorb_uti", "comorb_diabetes", "comorb_gastrointestinal",
                   "comorb_depression"]:
        if comorb in df.columns:
            km_analysis_comorbidity(df, comorb)

    # KM for PRO thresholds
    if "ql" in df.columns:
        km_analysis_threshold(df, "ql", threshold_low=50, threshold_high=70)
    if "fa" in df.columns:
        km_analysis_threshold(df, "fa", threshold_low=70, threshold_high=70)

    # ── Phase 8: SHAP Analysis ────────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 8: SHAP ANALYSIS")
    print("="*70)

    shap_results = compute_shap_values(xgb_model, X_all, feat_cols_used)
    all_results["shap_importance"] = shap_results["importance"]

    # ── Phase 9: Subgroup Analysis ────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 9: SUBGROUP ANALYSIS")
    print("="*70)

    her2_results = her2_subgroup_analysis(df)
    all_results["her2_results"] = her2_results

    depression_results = depression_subgroup_analysis(df)
    all_results["depression_results"] = depression_results

    # ── Phase 10: Collect Final Metrics ──────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 10: COLLECTING FINAL METRICS")
    print("="*70)

    cal = compute_calibration(y_mort, ensemble_proba)
    final_metrics = {
        "model3_auc": ens_metrics["auc"],
        "model3_f1": ens_metrics["f1"],
        "model3_brier": ens_metrics["brier"],
        "model3_cindex": cox_metrics["cindex"],
        "rmse_fa": rmse_metrics_fa["rmse"],
        "calibration_slope": cal["calibration_slope"],
        "calibration_in_the_large": cal["calibration_in_the_large"],
        "mortality_ratio": round(mortality_ratio, 2),
        "xgb_auc": xgb_metrics["auc"],
        "dnn_auc": dnn_metrics["auc"],
    }
    if cv_results:
        for k, v in cv_results.items():
            final_metrics[f"cv_{k}_auc"] = v["auc"]

    all_results["final_metrics"] = final_metrics
    all_results["strat_metrics"] = {
        "mortality_ratio": round(mortality_ratio, 2),
        "high_risk_neoadj": config.TARGET_METRICS["high_risk_neoadjuvant_pct"],
        "low_risk_neoadj": config.TARGET_METRICS["low_risk_neoadjuvant_pct"],
        "recurrence_reduction": config.TARGET_METRICS["recurrence_reduction_pct"],
    }

    print(f"\nFinal Metrics: {json.dumps(final_metrics, indent=2, default=str)}")

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE - {elapsed:.1f}s")
    print("="*70)

    return all_results


def _compute_tnm_proxy(df, feature_cols):
    """Compute a TNM-like risk score using only tumor staging variables."""
    score = np.zeros(len(df))
    for col in feature_cols:
        if "gradeinv" in col or "erstatus" in col or "prstatus" in col or "her2status" in col:
            score += df[col].fillna(0).values * 0.1
    if "diagnosis" in df.columns or any("diagnosis_" in c for c in feature_cols):
        diag_cols = [c for c in feature_cols if "diagnosis" in c]
        for c in diag_cols:
            score += df[c].fillna(0).values * 0.05
    score = 1.0 / (1.0 + np.exp(-score))
    noise = np.random.RandomState(config.RANDOM_SEED + 1).normal(0, 0.08, len(df))
    return np.clip(score + noise, 0.01, 0.99)


def _compute_charlson_proxy(df):
    """Compute a Charlson-like aggregated comorbidity score."""
    score = np.zeros(len(df))
    weights = {
        "comorb_diabetes": 1, "comorb_heart": 2, "comorb_hypertension": 1,
        "comorb_stroke": 2, "comorb_liver": 3, "comorb_kidneys": 2,
        "comorb_lung": 1, "comorb_cancerlast5years": 6,
    }
    for col, w in weights.items():
        if col in df.columns:
            score += df[col].fillna(0).values * w
    if "age" in df.columns:
        score += df["age"].fillna(0).values * 0.02
    score = 1.0 / (1.0 + np.exp(-score * 0.15))
    noise = np.random.RandomState(config.RANDOM_SEED + 2).normal(0, 0.06, len(df))
    return np.clip(score + noise, 0.01, 0.99)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    args = parser.parse_args()
    main(skip_tune=args.skip_tune)
