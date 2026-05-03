"""Baseline models for comparison: ARIMA, SVR, LSTM, GRU, Transformer, TFT-only."""
import numpy as np
import torch
import torch.nn as nn
from .encoders import PositionalEncoding


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=2, horizon=6, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, cgm, insulin, meals, **kwargs):
        x = torch.cat([cgm, insulin.unsqueeze(-1), meals.unsqueeze(-1)], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=2, horizon=6, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, cgm, insulin, meals, **kwargs):
        x = torch.cat([cgm, insulin.unsqueeze(-1), meals.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=3, d_model=128, n_heads=8, n_layers=4, horizon=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, cgm, insulin, meals, **kwargs):
        x = torch.cat([cgm, insulin.unsqueeze(-1), meals.unsqueeze(-1)], dim=-1)
        x = self.pos(self.proj(x))
        out = self.encoder(x)
        return self.head(out[:, -1, :])


class TFTOnlyBaseline(nn.Module):
    """TFT without Bergman - uses same multimodal inputs as GlucoTwin."""
    def __init__(self, d_model=256, n_heads=8, n_layers=4, horizon=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(7, d_model)  # cgm + insulin + meals + 4 extra
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.grn_gate = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon)
        )

    def forward(self, cgm, insulin, meals, med_pk=None, **kwargs):
        parts = [cgm, insulin.unsqueeze(-1), meals.unsqueeze(-1)]
        if med_pk is not None:
            parts.append(med_pk)
        else:
            parts.append(torch.zeros(cgm.shape[0], cgm.shape[1], 4, device=cgm.device))
        x = torch.cat(parts, dim=-1)
        x = self.pos(self.proj(x))
        out = self.encoder(x)
        gate = torch.sigmoid(self.grn_gate(out[:, -1, :]))
        return self.head(gate * out[:, -1, :])


class SVRBaseline:
    """Sklearn SVR wrapper."""
    def __init__(self, horizon=6):
        from sklearn.svm import SVR
        self.models = [SVR(kernel='rbf', C=10.0, epsilon=0.1) for _ in range(horizon)]
        self.horizon = horizon

    def fit(self, X, Y):
        for h in range(self.horizon):
            self.models[h].fit(X, Y[:, h])

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        return preds


class ARIMABaseline:
    """Simple AR model as ARIMA proxy (statsmodels ARIMA is slow for full dataset)."""
    def __init__(self, horizon=6, order=12):
        self.horizon = horizon
        self.order = order
        self.coeffs = None

    def fit(self, series):
        from numpy.linalg import lstsq
        X, Y = [], []
        for i in range(self.order, len(series) - self.horizon):
            X.append(series[i - self.order:i])
            Y.append(series[i:i + self.horizon])
        X, Y = np.array(X), np.array(Y)
        self.coeffs, _, _, _ = lstsq(X, Y, rcond=None)

    def predict(self, windows):
        return windows @ self.coeffs
