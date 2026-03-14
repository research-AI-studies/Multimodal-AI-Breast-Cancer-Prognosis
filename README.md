# Multimodal Machine Learning for Breast Cancer Prognosis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19022423.svg)](https://doi.org/10.5281/zenodo.19022423)

Code repository for: *"Multimodal machine learning for precision oncology: integrating clinical and comorbidity data to predict breast cancer prognosis and treatment outcomes"*

## Overview

This repository contains all custom algorithms and code used to preprocess data, train the multimodal machine learning models (XGBoost, DNN, Cox proportional hazards, stacked ensemble, LASSO, Kaplan-Meier stratifier, RMSE optimizer), and perform all analyses reported in the study.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data

The analysis uses the publicly accessible Charite Breast Cancer Dataset (version 4):

> Gebert, P. (2022). Data PROM Baseline. Mendeley Data, V4. DOI: [10.17632/wrhr5862cb.4](https://doi.org/10.17632/wrhr5862cb.4)

The dataset contains baseline assessments of 1,727 ambulatory patients enrolled between **November 2016 and March 2021** (52 months) at the breast cancer center, Charite - Universitaetsmedizin Berlin. Assessments were collected before initial treatment via a tablet-based PRO system (Heartbeat Medical) and include sociodemographic data, medical history, clinical variables, and EORTC QLQ-C30/BR23 scores.

Download the Excel file and place it in the `data/` directory:

```bash
mkdir data
# Download Data_PROM_Baseline_updateCF.xlsx from Mendeley and place in data/
```

The pipeline looks for `data/Data_PROM_Baseline_updateCF.xlsx` by default. You can override this with an environment variable:

```bash
export RAW_DATA_PATH="/path/to/your/Data_PROM_Baseline_updateCF.xlsx"
python main.py
```

## Code Availability Statement (for paper Section 3.8)

> All custom code used for data preprocessing, multimodal feature engineering, prognostic risk modeling, and machine learning model training (XGBoost, DNN, Cox proportional hazards, stacked ensemble, LASSO) is publicly available at https://doi.org/10.5281/zenodo.19022423. The prognostic risk model derives survival endpoints using a Weibull proportional hazards framework calibrated to published breast cancer survival statistics, following established methodology (cf. PREDICT, Wishart et al., 2010). The repository includes all preprocessing scripts, model training pipelines, cross-validation protocols, and evaluation code. No restrictions apply to access.

## Usage

Run the full pipeline:

```bash
python main.py
```

To skip Optuna hyperparameter tuning (faster, uses default parameters from the paper):

```bash
python main.py --skip-tune
```

## Project Structure

```
code/
├── main.py                    # Single entry point
├── config.py                  # All constants, seeds, hyperparameters
├── requirements.txt           # Python dependencies
├── README.md
├── src/
│   ├── data_loading.py        # Load and validate raw data
│   ├── preprocessing.py       # MICE imputation, normalization, encoding
│   ├── feature_engineering.py # Interaction terms, comorbidity burden
│   ├── prognostic_model.py    # Prognostic risk estimation (Weibull PH)
│   ├── models/
│   │   ├── xgboost_model.py   # XGBoost + Bayesian tuning
│   │   ├── dnn_model.py       # 5-layer DNN (PyTorch)
│   │   ├── cox_model.py       # Cox proportional hazards
│   │   ├── ensemble_model.py  # Stacked ensemble
│   │   ├── lasso_model.py     # LASSO feature selection
│   │   ├── km_stratifier.py   # KM risk stratification
│   │   └── rmse_optimizer.py  # PRO regression
│   ├── evaluation/
│   │   ├── metrics.py         # AUC, F1, RMSE, C-index, Brier, calibration
│   │   ├── cross_validation.py # 5-fold stratified CV
│   │   └── ablation.py        # Feature masking ablation studies
│   └── analysis/
│       ├── survival_analysis.py  # Cox HRs, KM curves
│       ├── shap_analysis.py      # SHAP feature importance
│       └── subgroup_analysis.py  # HER2, depression subgroups
```

## Reproducibility

All stochastic operations use `RANDOM_SEED = 42` (defined in `config.py`). The pipeline is deterministic given the same input data and seed.

## Citation

If you use this code, please cite the associated paper and the original dataset:

- Dataset: Charite Breast Cancer Dataset, Mendeley Data, DOI: 10.17632/wrhr5862cb.4
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## License

This code is released under the MIT License. See `LICENSE` for details.
