"""Simplified Temporal Fusion Transformer for residual glucose prediction."""
import torch
import torch.nn as nn
from .fusion import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Selects and weights input features via learned importance."""

    def __init__(self, d_input, d_model, n_vars, dropout=0.1):
        super().__init__()
        self.flattened_grn = GatedResidualNetwork(n_vars * d_input, d_hidden=d_model, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_input, d_hidden=d_model, dropout=dropout)
            for _ in range(n_vars)
        ])
        self.d_input = d_input
        self.n_vars = n_vars
        self.proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()

    def forward(self, x):
        # x: (batch, seq, n_vars * d_input)
        B, T, _ = x.shape
        flat = x.reshape(B * T, -1)
        weights = self.softmax(self.flattened_grn(flat, flat))
        weights = weights.reshape(B, T, -1)

        # weight average impossible with different dims; use simple gating
        return self.proj(x[:, :, :self.d_input]) * weights[:, :, :1].expand_as(x[:, :, :self.d_input])


class TemporalFusionTransformer(nn.Module):
    """TFT for learning residual corrections to the Bergman model."""

    def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=1024,
                 dropout=0.1, horizon=6):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model

        self.input_grn = GatedResidualNetwork(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.phys_proj = nn.Linear(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_grn = GatedResidualNetwork(d_model, dropout=dropout)

        self.point_head = nn.Linear(d_model, 1)
        self.quantile_head = nn.Linear(d_model, 3)  # q10, q50, q90

    def forward(self, fused_state, phys_pred):
        """
        Args:
            fused_state: (batch, lookback, d_model) multimodal fused representation
            phys_pred: (batch, horizon) Bergman model glucose prediction
        Returns:
            residual: (batch, horizon) point estimate residual
            quantiles: (batch, horizon, 3) quantile estimates
        """
        x = self.input_grn(fused_state)
        memory = self.encoder(x)

        phys_emb = self.phys_proj(phys_pred.unsqueeze(-1))
        decoded = self.decoder(phys_emb, memory)
        decoded = self.output_grn(decoded)

        residual = self.point_head(decoded).squeeze(-1)
        quantiles = self.quantile_head(decoded)

        return residual, quantiles
