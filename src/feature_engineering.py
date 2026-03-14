"""
Step 3: Feature engineering — interaction terms and comorbidity burden scores.
"""
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def add_comorbidity_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of positive comorbidity flags per patient (Section 3.2)."""
    out = df.copy()
    comorb_cols = [c for c in config.COMORBIDITY_BINARY_COLS if c in out.columns]
    out["comorbidity_burden"] = out[comorb_cols].sum(axis=1)
    print(f"[feature_eng] Comorbidity burden: mean={out['comorbidity_burden'].mean():.2f}, "
          f"max={out['comorbidity_burden'].max()}")
    return out


def _get_base_col(df, col_name):
    """Retrieve column, handling one-hot encoded variants."""
    if col_name in df.columns:
        return df[col_name]
    ohe_candidates = [c for c in df.columns if c.startswith(col_name + "_")]
    if ohe_candidates:
        # For interactions with OHE columns, use the first positive indicator
        # (e.g. her2status_1 for her2status positive)
        pos_col = col_name + "_1"
        if pos_col in df.columns:
            return df[pos_col]
        return df[ohe_candidates[0]]
    return None


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Create pairwise interaction features as described in Section 3.2."""
    out = df.copy()
    created = []
    for col_a, col_b in config.INTERACTION_TERMS:
        vec_a = _get_base_col(out, col_a)
        vec_b = _get_base_col(out, col_b)
        if vec_a is not None and vec_b is not None:
            name = f"{col_a}_x_{col_b}"
            out[name] = vec_a.values * vec_b.values
            created.append(name)

    print(f"[feature_eng] Created {len(created)} interaction terms: {created}")
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_comorbidity_burden(df)
    df = add_interaction_terms(df)
    print(f"[feature_eng] Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    from data_loading import load_raw_data
    from preprocessing import preprocess_pipeline
    raw = load_raw_data()
    processed, _ = preprocess_pipeline(raw)
    engineered = engineer_features(processed)
    print(engineered.columns.tolist())
