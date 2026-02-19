"""Minimal Transformer-like model for DeepSEA multi-label prediction."""

from __future__ import annotations

import torch
from torch import nn


class DeepSEATransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 1000,
        in_channels: int = 4,
        num_labels: int = 919,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1000, 4]
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        h = self.encoder(h)
        h = self.norm(h)
        h = h.mean(dim=1)  # global average pooling over sequence length
        logits = self.head(h)
        return logits
