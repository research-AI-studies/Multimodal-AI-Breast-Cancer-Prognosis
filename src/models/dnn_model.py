"""
5-layer Deep Neural Network for mortality risk and PRO regression (Section 3.3).
Built with PyTorch.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrognosticDNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 5,
                 dropout: float = 0.2, output_dim: int = 1, task: str = "classification"):
        super().__init__()
        self.task = task
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(hidden, output_dim))
        if task == "classification":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _prepare_data(X, y, batch_size=64):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1) if y.ndim == 1 else torch.FloatTensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_dnn_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "mortality_5yr",
) -> tuple:
    """Train DNN for binary classification. Returns (model, metrics)."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Replace any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    p = config.DNN_PARAMS
    model = PrognosticDNN(
        input_dim=X.shape[1],
        hidden=p["hidden_units"],
        n_layers=p["n_layers"],
        dropout=p["dropout_rate"],
        task="classification",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.BCELoss()
    loader = _prepare_data(X, y, batch_size=p["batch_size"])

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(p["max_epochs"]):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X)
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= p["early_stop_patience"]:
                break

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        proba = model(X_t).cpu().numpy().flatten()

    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, preds)
    metrics = {"auc": round(auc, 4), "f1": round(f1, 4)}
    print(f"[dnn] Train AUC={auc:.4f}, F1={f1:.4f}, stopped at epoch {epoch+1}")
    return model, metrics


def train_dnn_regressor(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "fa",
) -> tuple:
    """Train DNN for PRO score regression. Returns (model, metrics)."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    p = config.DNN_PARAMS
    model = PrognosticDNN(
        input_dim=X.shape[1],
        hidden=p["hidden_units"],
        n_layers=p["n_layers"],
        dropout=p["dropout_rate"],
        task="regression",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.MSELoss()
    loader = _prepare_data(X, y, batch_size=p["batch_size"])

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(p["max_epochs"]):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X)
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= p["early_stop_patience"]:
                break

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        preds = model(X_t).cpu().numpy().flatten()

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    metrics = {"rmse": round(rmse, 2)}
    print(f"[dnn_reg] Train RMSE={rmse:.2f} for {target_col}")
    return model, metrics


def predict_proba_dnn(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(np.nan_to_num(X, nan=0.0)).to(DEVICE)
        return model(X_t).cpu().numpy().flatten()
