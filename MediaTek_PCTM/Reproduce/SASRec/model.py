"""PyTorch implementation of SASRec for sequential recommendation."""
from __future__ import annotations

import math

import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    """The position-wise two-layer Conv1d block used by the SASRec reference."""

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(1, 2))
        outputs = self.dropout1(self.relu(outputs))
        outputs = self.dropout2(self.conv2(outputs)).transpose(1, 2)
        return outputs + inputs


class SASRecBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            hidden_units,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.feed_forward = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, inputs: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        outputs = inputs + attended
        return self.feed_forward(self.feed_forward_norm(outputs))


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation with sampled binary loss."""

    def __init__(
        self,
        num_items: int,
        maxlen: int = 100,
        hidden_units: int = 128,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if num_items < 1:
            raise ValueError("num_items must be positive")
        if hidden_units % num_heads:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.num_items = num_items
        self.maxlen = maxlen
        # Item 0 is padding; dataset item IDs are shifted by one at the boundary.
        self.item_embedding = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, hidden_units)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape [batch, length]")
        length = sequences.size(1)
        if length > self.maxlen:
            raise ValueError(f"sequence length {length} exceeds maxlen={self.maxlen}")
        padding = sequences.eq(0)
        positions = torch.arange(length, device=sequences.device).unsqueeze(0)
        features = self.item_embedding(sequences) * math.sqrt(self.item_embedding.embedding_dim)
        features = self.embedding_dropout(features + self.position_embedding(positions))
        features = features.masked_fill(padding.unsqueeze(-1), 0.0)
        causal_mask = self.causal_mask[:length, :length]
        for block in self.blocks:
            features = block(features, causal_mask)
            features = features.masked_fill(padding.unsqueeze(-1), 0.0)
        return self.final_norm(features).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(
        self,
        sequences: torch.Tensor,
        positive: torch.Tensor | None = None,
        negative: torch.Tensor | None = None,
    ):
        features = self.encode(sequences)
        if positive is None and negative is None:
            # All evaluation sequences are right-aligned and non-empty.
            return features[:, -1] @ self.item_embedding.weight[1:].transpose(0, 1)
        if positive is None or negative is None:
            raise ValueError("positive and negative must either both be set or both be omitted")
        positive_logits = (features * self.item_embedding(positive)).sum(dim=-1)
        negative_logits = (features * self.item_embedding(negative)).sum(dim=-1)
        return positive_logits, negative_logits
