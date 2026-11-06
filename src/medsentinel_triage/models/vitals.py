"""PyTorch 1D-CNN + Bi-LSTM + MultiheadAttention 24-hour ICU vitals trajectory inference module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from medsentinel_triage.base import BaseInferenceModel, InputValidationError
from medsentinel_triage.config import ICU_NUM_VITAL_CHANNELS, ICU_SEQUENCE_LENGTH, resolve_torch_device
from medsentinel_triage.preprocessing import VitalsInputType, validate_and_preprocess_vitals


class VitalSignsBiLSTMAttentionNet(nn.Module):
    """PyTorch 1D-CNN + Bidirectional LSTM + Multi-Head Temporal Self-Attention Network.

    Exact architecture from Notebook 03 & Notebook 06 (`vitals_bilstm_attention.pth`).
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.bilstm(x_conv)
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        temporal_importance = attn_weights.mean(dim=1)
        return logits, temporal_importance


class PyTorchVitalsTrainer(BaseInferenceModel[VitalsInputType, np.ndarray, Dict[str, Any]]):
    """Runs 24-hour ICU vitals trajectory forecasting and temporal attention attribution."""

    model_name: str = "vitals_bilstm_attention"

    def __init__(
        self,
        epochs: int = 25,
        lr: float = 2e-3,
        in_channels: int = ICU_NUM_VITAL_CHANNELS,
        hidden_dim: int = 32,
        num_heads: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or resolve_torch_device()
        self.epochs = epochs
        self.lr = lr
        self.net = VitalSignsBiLSTMAttentionNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        ).to(self.device)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def validate_input(self, raw_input: VitalsInputType) -> None:
        if self.mean_ is None or self.std_ is None:
            raise InputValidationError(
                "Vitals normalization statistics (`mean_`, `std_`) are not loaded on PyTorchVitalsTrainer."
            )
        validate_and_preprocess_vitals(
            raw_input,
            seq_len=ICU_SEQUENCE_LENGTH,
            num_channels=ICU_NUM_VITAL_CHANNELS,
        )

    def preprocess(self, raw_input: VitalsInputType) -> np.ndarray:
        return validate_and_preprocess_vitals(
            raw_input,
            seq_len=ICU_SEQUENCE_LENGTH,
            num_channels=ICU_NUM_VITAL_CHANNELS,
        )

    def forecast_trajectory(self, vitals_24h: VitalsInputType) -> Dict[str, Any]:
        """Forecasts 6-hour ICU deterioration probability, peak attention hours, and projected vitals."""
        arr = (
            vitals_24h
            if isinstance(vitals_24h, np.ndarray)
            and vitals_24h.shape == (ICU_SEQUENCE_LENGTH, ICU_NUM_VITAL_CHANNELS)
            and vitals_24h.dtype == np.float32
            else self.preprocess(vitals_24h)
        )
        if self.mean_ is None or self.std_ is None:
            raise InputValidationError("Normalization stats (`mean_`, `std_`) must be loaded before inference.")

        self.net.eval()
        x_norm = (arr[None, :, :] - self.mean_) / self.std_
        with torch.no_grad():
            tx = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
            logit, attn_map = self.net(tx)
            prob = float(torch.sigmoid(logit).item())
            hour_weights = attn_map[0].cpu().numpy()

        peak_hours = [int(h) for h in np.argsort(hour_weights)[-3:][::-1]]
        slope = (arr[-1] - arr[-6]) / 5.0
        proj_6h = arr[-1] + 6.0 * 0.5 * slope

        return {
            "deterioration_6h_prob": round(prob, 4),
            "peak_attention_hours": peak_hours,
            "projected_vitals_6h": {
                "heart_rate_bpm": round(float(proj_6h[0]), 1),
                "spo2_pct": round(float(np.clip(proj_6h[1], 70.0, 100.0)), 1),
                "systolic_bp_mmhg": round(float(proj_6h[2]), 1),
                "resp_rate_bpm": round(float(proj_6h[3]), 1),
                "temp_c": round(float(proj_6h[4]), 1),
            },
        }

    def predict(self, preprocessed_input: np.ndarray) -> Dict[str, Any]:
        return self.forecast_trajectory(preprocessed_input)
