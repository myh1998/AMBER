"""Model definitions for DeepSEA multi-label prediction."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class DeepSEATransformer(nn.Module):
    """Local TransformerEncoder baseline (no external pretrained model)."""

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
        h = h.mean(dim=1)
        logits = self.head(h)
        return logits


class HFSequenceClassifier(nn.Module):
    """HuggingFace DNA model wrapper + multi-label classification head.

    This supports model IDs such as DNABERT-2 and Nucleotide Transformer.
    """

    def __init__(self, model_name: str, num_labels: int = 919, dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for model_id=dnabert2/nucleotide_transformer. "
                "Install with: pip install transformers"
            ) from e

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Cannot infer hidden size from model config: {self.backbone.config}")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(model_id: str, num_labels: int = 919, **kwargs) -> nn.Module:
    """Factory for selectable model backbones.

    Supported model_id:
    - local_transformer
    - dnabert2
    - nucleotide_transformer
    """
    model_id = model_id.lower()
    if model_id == "local_transformer":
        return DeepSEATransformer(num_labels=num_labels, **kwargs)

    if model_id == "dnabert2":
        model_name = kwargs.pop("hf_model_name", "zhihan1996/DNABERT-2-117M")
        return HFSequenceClassifier(model_name=model_name, num_labels=num_labels, dropout=kwargs.pop("dropout", 0.1))

    if model_id == "nucleotide_transformer":
        model_name = kwargs.pop("hf_model_name", "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
        return HFSequenceClassifier(model_name=model_name, num_labels=num_labels, dropout=kwargs.pop("dropout", 0.1))

    raise ValueError(
        f"Unknown model_id={model_id}. Supported: local_transformer, dnabert2, nucleotide_transformer"
    )


def tokenize_dna_sequences(
    sequences: List[str],
    model_name: str | None = None,
    tokenizer=None,
    max_length: int = 1024,
    device: torch.device | None = None,
):
    """Tokenize DNA strings for HuggingFace backbones."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for model_id=dnabert2/nucleotide_transformer. "
            "Install with: pip install transformers"
        ) from e

    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either model_name or tokenizer must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    batch = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
