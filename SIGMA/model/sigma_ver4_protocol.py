"""SIGMA model adapted to the MediaTek ver4 data/evaluation protocol.

The network equations intentionally follow ``gated_mamba.py`` from the cloned
repository. RecBole is not used because it would filter and split matched data
a second time.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _mamba_class():
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:  # pragma: no cover - CUDA environment dependent
        raise RuntimeError(
            "mamba-ssm is required for a research run; no surrogate sequence "
            "layer is used."
        ) from exc
    return Mamba


class FeedForward(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dimension, 4 * dimension)
        self.w2 = nn.Linear(4 * dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension, eps=1e-12)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.gelu(self.w1(values)))
        hidden = self.dropout(self.w2(hidden))
        return self.norm(hidden + values)


class GMambaBlock(nn.Module):
    """PF-Mamba + dense selective gates + FE-GRU from the released code.

    ``flip_prefix`` is 45 for the paper/repository setting of length 50. It is
    exposed solely to make the previously hard-coded constant auditable.
    """

    def __init__(
        self,
        dimension: int,
        state_dimension: int,
        convolution_width: int,
        expansion: int,
        dropout: float,
        flip_prefix: int,
    ) -> None:
        super().__init__()
        Mamba = _mamba_class()
        self.flip_prefix = flip_prefix
        self.combining_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.8]))
        self.dense1 = nn.Linear(dimension, dimension)
        self.dense2 = nn.Linear(dimension, dimension)
        self.projection = nn.Linear(dimension, dimension)
        # The clone shares one Mamba module between forward and flipped paths.
        self.mamba = Mamba(
            d_model=dimension,
            d_state=state_dimension,
            d_conv=convolution_width,
            expand=expansion,
        )
        self.gru = nn.GRU(dimension, dimension, bias=False, batch_first=True)
        # Activation-before-linear ordering matches the released implementation.
        self.selective_gate_sig = nn.Sequential(nn.Sigmoid(), nn.Linear(dimension, dimension))
        self.selective_gate_silu = nn.Sequential(nn.SiLU(), nn.Linear(dimension, dimension))
        self.gate_dropout = nn.Dropout(dropout)
        self.conv1d = nn.Conv1d(dimension, dimension, kernel_size=3, padding=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        h1 = input_tensor + self.dense1(input_tensor)
        gru_input = self.conv1d(input_tensor.transpose(1, 2)).transpose(1, 2)

        flipped_input = input_tensor.clone()
        prefix = min(self.flip_prefix, input_tensor.size(1))
        flipped_input[:, :prefix, :] = input_tensor[:, :prefix, :].flip(dims=[1])
        h2 = self.dense2(flipped_input) + flipped_input

        forward_mamba = self.mamba(input_tensor)
        flipped_mamba = self.mamba(flipped_input)
        h1 = self.gate_dropout(self.selective_gate_silu(h1) + self.selective_gate_sig(h1))
        # This cross-path sigmoid term is present in the released clone.
        h2 = self.gate_dropout(self.selective_gate_silu(h2) + self.selective_gate_sig(h1))
        forward_mamba = forward_mamba * h1 + forward_mamba
        flipped_mamba = flipped_mamba * h2 + flipped_mamba
        gru_output, _ = self.gru(gru_input)

        combined = (
            self.combining_weights[2] * forward_mamba
            + self.combining_weights[1] * flipped_mamba
            + self.combining_weights[0] * gru_output
        )
        return self.projection(combined)


class SIGMALayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        state_dimension: int,
        convolution_width: int,
        expansion: int,
        dropout: float,
        flip_prefix: int,
        layer_count: int,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.gmamba = GMambaBlock(
            dimension, state_dimension, convolution_width, expansion, dropout, flip_prefix
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension, eps=1e-12)
        self.feed_forward = FeedForward(dimension, dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.gmamba(values)
        if self.layer_count == 1:
            hidden = self.norm(self.dropout(hidden))
        else:
            hidden = self.norm(self.dropout(hidden) + values)
        return self.feed_forward(hidden)


class SIGMA(nn.Module):
    """Standalone form of the repository model with identical item scoring."""

    def __init__(
        self,
        num_items: int,
        max_history: int = 50,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        retained_tail: int = 5,
    ) -> None:
        super().__init__()
        if retained_tail < 0 or retained_tail >= max_history:
            raise ValueError("retained_tail must be in [0, max_history)")
        self.num_items = num_items
        self.max_history = max_history
        # Zero is padding; MediaTek item IDs are shifted by one at model ingress.
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            SIGMALayer(
                hidden_size,
                d_state,
                d_conv,
                expand,
                dropout,
                max_history - retained_tail,
                num_layers,
            )
            for _ in range(num_layers)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, item_sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.dropout(self.item_embedding(item_sequence)))
        for layer in self.layers:
            hidden = layer(hidden)
        row = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[row, lengths - 1]

    def full_sort_scores(self, item_sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state = self.encode(item_sequence, lengths)
        return state @ self.item_embedding.weight[1:].T

