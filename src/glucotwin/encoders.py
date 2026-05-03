"""Modality-specific encoders for CGM, EHR, Medication PK, and Lifestyle data."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with circadian awareness."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CausalDilatedConv(nn.Module):
    """Single causal dilated conv layer with residual connection."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (batch, seq, channels)
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.padding, 0))
        x_t = self.conv(x_t)
        x = x_t.transpose(1, 2)
        x = self.norm(x)
        return F.relu(x) + residual


class CGMEncoder(nn.Module):
    """Encodes CGM time-series via dilated causal convolutions."""

    def __init__(self, d_model=256, n_layers=4, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CausalDilatedConv(d_model, kernel_size, dilation=2**i)
            for i in range(n_layers)
        ])

    def forward(self, cgm):
        """
        Args:
            cgm: (batch, seq_len, 1) normalized glucose values
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(cgm)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x


class EHREncoder(nn.Module):
    """TabNet-inspired encoder for structured EHR features."""

    def __init__(self, input_dim, d_model=256, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.shared_fc = nn.Linear(input_dim, d_model)
        self.step_attns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, input_dim), nn.Softmax(dim=-1))
            for _ in range(n_steps)
        ])
        self.step_fcs = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
            for _ in range(n_steps)
        ])
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x_ehr):
        """
        Args:
            x_ehr: (batch, input_dim)
        Returns:
            (batch, d_model)
        """
        aggregated = torch.zeros(x_ehr.size(0), self.shared_fc.out_features, device=x_ehr.device)
        prior_scales = torch.ones_like(x_ehr)

        for i in range(self.n_steps):
            h = self.shared_fc(x_ehr * prior_scales)
            attn = self.step_attns[i](h)
            masked = x_ehr * attn
            step_out = self.step_fcs[i](masked)
            aggregated = aggregated + step_out
            prior_scales = prior_scales * (1.0 - attn)

        return self.output_proj(aggregated)


class MedicationPKEncoder(nn.Module):
    """Encodes medication pharmacokinetic action profiles."""

    def __init__(self, n_med_types=4, d_model=256):
        super().__init__()
        self.input_proj = nn.Linear(n_med_types, d_model)
        self.conv1 = CausalDilatedConv(d_model, kernel_size=3, dilation=1)
        self.conv2 = CausalDilatedConv(d_model, kernel_size=3, dilation=2)

    def forward(self, pk_profiles):
        """
        Args:
            pk_profiles: (batch, seq_len, n_med_types) PK action curves
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(pk_profiles)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LifestyleEncoder(nn.Module):
    """Encodes lifestyle events (meals, exercise, sleep) as sparse time-series."""

    def __init__(self, n_event_types=5, d_model=256):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types, d_model // 2)
        self.value_proj = nn.Linear(1, d_model // 2)
        self.temporal_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, event_types, event_values, event_mask):
        """
        Args:
            event_types: (batch, seq_len) integer event type indices
            event_values: (batch, seq_len, 1) event magnitudes (e.g., carbs in grams)
            event_mask: (batch, seq_len) binary mask for active events
        Returns:
            (batch, seq_len, d_model)
        """
        type_emb = self.event_embed(event_types)
        val_emb = self.value_proj(event_values)
        combined = torch.cat([type_emb, val_emb], dim=-1)
        combined = self.temporal_proj(combined)
        combined = combined * event_mask.unsqueeze(-1)
        return self.norm(combined)
