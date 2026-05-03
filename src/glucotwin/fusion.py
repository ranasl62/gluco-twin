"""Cross-attention multimodal fusion and gated residual network."""
import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """GRN as used in the Temporal Fusion Transformer."""

    def __init__(self, d_model, d_hidden=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(h))
        return self.norm(gate * h + (1 - gate) * residual)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of multimodal representations.

    Each modality attends to all others, then results are concatenated
    and projected through a GRN.
    """

    def __init__(self, d_model=256, n_heads=8, n_modalities=4, dropout=0.1):
        super().__init__()
        self.n_modalities = n_modalities
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_modalities)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modalities)])
        self.fusion_proj = nn.Linear(d_model * n_modalities, d_model)
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)

    def forward(self, modality_list):
        """
        Args:
            modality_list: list of (batch, seq_len, d_model) tensors, one per modality.
                           Non-temporal modalities should be broadcast to seq_len.
        Returns:
            fused: (batch, seq_len, d_model)
        """
        attended = []
        for i, (attn, norm) in enumerate(zip(self.cross_attns, self.norms)):
            query = modality_list[i]
            keys_values = torch.cat([modality_list[j] for j in range(self.n_modalities) if j != i], dim=1)
            out, _ = attn(query, keys_values, keys_values)
            attended.append(norm(out + query))

        concat = torch.cat(attended, dim=-1)
        projected = self.fusion_proj(concat)
        return self.grn(projected)


class ConcatFusion(nn.Module):
    """Baseline fusion: concatenate modality tensors and project (no cross-attention)."""

    def __init__(self, d_model=256, n_modalities=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model * n_modalities, d_model)
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)

    def forward(self, modality_list):
        x = torch.cat(modality_list, dim=-1)
        return self.grn(self.proj(x))
