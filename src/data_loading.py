"""
Step 1: Load the raw Charite Breast Cancer Dataset and split into modality groups.
Source: Mendeley Data DOI 10.17632/wrhr5862cb.4 (version 4, 18 May 2022).
Enrollment period: November 2016 - March 2021 (52 months).
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = path or config.RAW_DATA_PATH
    df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")

    # Strip whitespace from string cells and replace blank strings with NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    assert df.shape[0] == config.N_PATIENTS, (
        f"Expected {config.N_PATIENTS} rows, got {df.shape[0]}"
    )
    print(f"[data_loading] Loaded {df.shape[0]} patients x {df.shape[1]} columns")
    return df


def split_modalities(df: pd.DataFrame) -> dict:
    """Return dict of DataFrames keyed by modality name."""
    available = set(df.columns)

    def _pick(cols):
        return [c for c in cols if c in available]

    modalities = {
        "clinical": df[_pick(config.CLINICAL_COLS)].copy(),
        "comorbidity": df[_pick(config.COMORBIDITY_COLS)].copy(),
        "pro": df[_pick(config.PRO_COLS)].copy(),
        "sociodemographic": df[_pick(config.SOCIODEMOGRAPHIC_COLS)].copy(),
    }
    for name, sub in modalities.items():
        print(f"  {name:20s}: {sub.shape[1]} columns")
    return modalities


def get_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "column": df.columns,
        "missing": df.isnull().sum().values,
        "total": len(df),
    })
    report["pct_missing"] = (report["missing"] / report["total"] * 100).round(1)
    return report.sort_values("pct_missing", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw_data()
    mods = split_modalities(df)
    report = get_missingness_report(df)
    print("\nTop missing columns:")
    print(report.head(20).to_string(index=False))
