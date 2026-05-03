"""GlucoTwin: Full hybrid physiological-neural model."""
import torch
import torch.nn as nn
from .bergman import BergmanMinimalModel
from .encoders import CGMEncoder, EHREncoder, MedicationPKEncoder, LifestyleEncoder
from .fusion import CrossAttentionFusion, ConcatFusion
from .tft import TemporalFusionTransformer


class GlucoTwinModel(nn.Module):
    """End-to-end GlucoTwin prediction model.

    Combines Bergman minimal model (physics) with Temporal Fusion Transformer
    (residual learning) through cross-attention multimodal fusion.
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=1024,
                 n_ehr_features=47, n_med_types=4, n_event_types=5,
                 horizon=6, lookback=48, dropout=0.1,
                 fusion_mode='cross',
                 ablate_bergman=False,
                 use_ehr=True, use_med=True, use_life=True):
        super().__init__()
        self.horizon = horizon
        self.lookback = lookback
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        self.ablate_bergman = ablate_bergman
        self.use_ehr = use_ehr
        self.use_med = use_med
        self.use_life = use_life

        self.bergman = BergmanMinimalModel()
        self.cgm_encoder = CGMEncoder(d_model, n_layers=4)
        self.ehr_encoder = EHREncoder(n_ehr_features, d_model)
        self.med_encoder = MedicationPKEncoder(n_med_types, d_model)
        self.life_encoder = LifestyleEncoder(n_event_types, d_model)
        if fusion_mode == 'cross':
            self.fusion = CrossAttentionFusion(d_model, n_heads, n_modalities=4, dropout=dropout)
        elif fusion_mode == 'concat':
            self.fusion = ConcatFusion(d_model, n_modalities=4, dropout=dropout)
        else:
            raise ValueError(f'Unknown fusion_mode: {fusion_mode}')
        self.tft = TemporalFusionTransformer(d_model, n_heads, n_layers, d_ff, dropout, horizon)

    def forward(self, cgm, ehr, med_pk, life_types, life_values, life_mask,
                meal_rates_future, insulin_rates_future):
        """
        Args:
            cgm: (batch, lookback, 1) normalized CGM
            ehr: (batch, n_ehr_features)
            med_pk: (batch, lookback, n_med_types) PK profiles
            life_types: (batch, lookback) event type indices
            life_values: (batch, lookback, 1) event values
            life_mask: (batch, lookback) event activity mask
            meal_rates_future: (batch, horizon) known future meal rates
            insulin_rates_future: (batch, horizon) known future insulin
        Returns:
            dict with 'glucose_pred', 'phys_pred', 'residual', 'quantiles'
        """
        h_cgm = self.cgm_encoder(cgm)
        h_ehr = self.ehr_encoder(ehr).unsqueeze(1).expand(-1, self.lookback, -1)
        h_med = self.med_encoder(med_pk)
        h_life = self.life_encoder(life_types, life_values, life_mask)
        if not self.use_ehr:
            h_ehr = torch.zeros_like(h_ehr)
        if not self.use_med:
            h_med = torch.zeros_like(h_med)
        if not self.use_life:
            h_life = torch.zeros_like(h_life)

        fused = self.fusion([h_cgm, h_ehr, h_med, h_life])

        G0 = cgm[:, -1, 0]
        G0_denorm = G0  # caller passes raw mg/dL for Bergman
        if self.ablate_bergman:
            phys_pred = G0_denorm.unsqueeze(1).expand(-1, self.horizon).contiguous()
        else:
            phys_pred = self.bergman(G0_denorm, meal_rates_future, insulin_rates_future, self.horizon)

        residual, quantiles = self.tft(fused, phys_pred)
        glucose_pred = phys_pred + residual

        return {
            'glucose_pred': glucose_pred,
            'phys_pred': phys_pred,
            'residual': residual,
            'quantiles': quantiles,
        }


class GlucoTwinLoss(nn.Module):
    """Combined loss: MSE + physics + quantile."""

    def __init__(self, lambda_phys=0.05, lambda_quant=0.1):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.lambda_quant = lambda_quant
        self.quantiles_target = [0.1, 0.5, 0.9]

    def forward(self, outputs, targets):
        pred = outputs['glucose_pred']
        phys = outputs['phys_pred']
        quant = outputs['quantiles']

        mse = torch.mean((pred - targets) ** 2)

        # Physics residual: penalize large corrections
        phys_loss = torch.mean((pred - phys) ** 2) * 0.01 + \
                    torch.mean(torch.relu(torch.abs(outputs['residual']) - 50.0))

        # Quantile loss
        q_loss = torch.tensor(0.0, device=pred.device)
        for i, q in enumerate(self.quantiles_target):
            errors = targets - quant[..., i]
            q_loss = q_loss + torch.mean(torch.max(q * errors, (q - 1) * errors))

        return mse + self.lambda_phys * phys_loss + self.lambda_quant * q_loss, {
            'mse': mse.item(),
            'phys': phys_loss.item(),
            'quant': q_loss.item(),
        }
