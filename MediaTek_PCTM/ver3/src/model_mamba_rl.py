"""Efficient multi-agent, LoRA-adapted selective-state recommender.

The expensive pretrained Mamba is used once to cache item semantics.  Online
training and ranking then use linear-time selective recurrences over those item
vectors.  Long, short, and coordinator agents own disjoint LoRA parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.model import LightGCN


class LoRAUpdate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(input_dim, rank, bias=False)
        self.b = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(self.dropout(inputs))) * self.scale


class SelectiveMambaAgent(nn.Module):
    """Mamba-style diagonal selective SSM with an agent-owned LoRA delta."""

    def __init__(self, dim: int, rank: int, alpha: float, dropout: float, initial_timescale: float):
        super().__init__()
        self.candidate_lora = LoRAUpdate(dim, dim, rank, alpha, dropout)
        self.delta_lora = LoRAUpdate(dim, dim, rank, alpha, dropout)
        self.gate_lora = LoRAUpdate(dim, dim, rank, alpha, dropout)
        self.output_lora = LoRAUpdate(dim, dim, rank, alpha, dropout)
        self.delta_bias = nn.Parameter(torch.full((dim,), math.log(math.expm1(initial_timescale))))
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width, dim = sequence.shape
        state = sequence.new_zeros(batch, dim)
        output = state
        for index in range(width):
            inputs = sequence[:, index]
            delta = F.softplus(self.delta_bias + self.delta_lora(inputs)).clamp(max=12.0)
            decay = torch.exp(-delta)
            proposal = torch.tanh(inputs + self.candidate_lora(inputs))
            updated = decay * state + (1.0 - decay) * proposal
            active = (index < lengths).unsqueeze(1)
            state = torch.where(active, updated, state)
            gate = torch.sigmoid(self.gate_lora(inputs))
            current = gate * state + (1.0 - gate) * inputs
            output = torch.where(active, current, output)
        return F.normalize(output + self.output_lora(output), dim=-1)

    def logits(self, state: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(max=20.0)
        if candidate_vectors.ndim == 2:
            return state @ candidate_vectors.T * temperature
        return torch.einsum("bd,bcd->bc", state, candidate_vectors) * temperature


class CoordinatorAgent(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.state_lora = LoRAUpdate(dim * 2, dim, rank, alpha, dropout)
        self.mix_lora = LoRAUpdate(dim * 2, 2, rank, alpha, dropout)
        self.score_lora = LoRAUpdate(dim, dim, rank, alpha, dropout)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, long_state: torch.Tensor, short_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((long_state, short_state), dim=-1)
        weights = torch.softmax(self.mix_lora(combined), dim=-1)
        base = weights[:, :1] * long_state + weights[:, 1:] * short_state
        state = F.normalize(base + self.state_lora(combined), dim=-1)
        return state, weights

    def logits(self, state: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        adapted = F.normalize(state + self.score_lora(state), dim=-1)
        temperature = self.log_temperature.exp().clamp(max=20.0)
        if candidate_vectors.ndim == 2:
            return adapted @ candidate_vectors.T * temperature
        return torch.einsum("bd,bcd->bc", adapted, candidate_vectors) * temperature


class MultiAgentMambaRecommender(nn.Module):
    """Long/short selective-state specialists plus a LoRA coordinator."""

    def __init__(
        self,
        item_features: torch.Tensor,
        dim: int = 128,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        short_window: int = 10,
        graph_edges: torch.Tensor | None = None,
        graph_users: int | None = None,
        use_graph_embeddings: bool = True,
    ):
        super().__init__()
        if item_features.ndim != 2:
            raise ValueError("item_features must have shape [items, feature_dim]")
        if use_graph_embeddings and (graph_edges is None or graph_users is None):
            raise ValueError("graph_edges and graph_users are required when graph embeddings are enabled")
        self.register_buffer("item_features", item_features, persistent=False)
        self.item_projection = nn.Linear(item_features.size(1), dim, bias=False)
        nn.init.xavier_uniform_(self.item_projection.weight)
        self.use_graph_embeddings = use_graph_embeddings
        self._show_graph_progress = use_graph_embeddings
        if use_graph_embeddings:
            self.graph = LightGCN(graph_users, item_features.size(0), dim)
            self.register_buffer("graph_edges", graph_edges, persistent=False)
        self.long_agent = SelectiveMambaAgent(dim, lora_rank, lora_alpha, lora_dropout, initial_timescale=0.08)
        self.short_agent = SelectiveMambaAgent(dim, lora_rank, lora_alpha, lora_dropout, initial_timescale=0.5)
        self.coordinator = CoordinatorAgent(dim, lora_rank, lora_alpha, lora_dropout)
        self.short_window = short_window

    @property
    def num_items(self) -> int:
        return self.item_features.size(0)

    def place_devices(self, main_device: str, graph_device: str | None = None):
        """Place the dense agents and LightGCN on separate devices when requested."""
        main = torch.device(main_device)
        self.item_features = self.item_features.to(main)
        for module in (
            self.item_projection,
            self.long_agent,
            self.short_agent,
            self.coordinator,
        ):
            module.to(main)
        if self.use_graph_embeddings:
            graph = torch.device(graph_device or main_device)
            self.graph.to(graph)
            self.graph_edges = self.graph_edges.to(graph)
            self._graph_device = graph
        else:
            self._graph_device = None
        return self

    def graph_item_vectors(self) -> torch.Tensor | None:
        """Propagate the graph once so all item lookups in a batch can share it."""
        if not self.use_graph_embeddings:
            return None
        if self._show_graph_progress:
            with tqdm(
                total=self.graph.layers + 1,
                desc="GCN item embeddings (initial pass)",
                unit="layer",
                leave=True,
                dynamic_ncols=True,
            ) as progress:
                _, graph_items = self.graph(self.graph_edges, progress=progress)
            self._show_graph_progress = False
            return graph_items
        return self.graph(self.graph_edges)[1]

    def project_ids(
        self, item_ids: torch.Tensor, graph_items: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.item_features[item_ids]
        projected = self.item_projection(features.to(self.item_projection.weight.dtype))
        projected = F.normalize(projected, dim=-1)
        if not self.use_graph_embeddings:
            return projected
        if graph_items is None:
            graph_items = self.graph_item_vectors()
        assert graph_items is not None
        graph_ids = item_ids.to(graph_items.device)
        graph_projected = F.normalize(graph_items[graph_ids], dim=-1).to(
            projected.device, non_blocking=True
        )
        return F.normalize(projected + graph_projected, dim=-1)

    def project_all(self, graph_items: torch.Tensor | None = None) -> torch.Tensor:
        return self.project_ids(
            torch.arange(self.num_items, device=self.item_features.device), graph_items
        )

    @staticmethod
    def _short_histories(histories: torch.Tensor, lengths: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        short_lengths = lengths.clamp(max=window)
        positions = torch.arange(window, device=histories.device).unsqueeze(0)
        starts = (lengths - short_lengths).unsqueeze(1)
        indices = starts + positions
        indices = indices.clamp(min=0, max=max(histories.size(1) - 1, 0))
        gathered = histories.gather(1, indices)
        gathered = torch.where(positions < short_lengths.unsqueeze(1), gathered, torch.zeros_like(gathered))
        return gathered, short_lengths

    def encode_states(
        self, histories: torch.Tensor, lengths: torch.Tensor,
        graph_items: torch.Tensor | None = None,
    ):
        if self.use_graph_embeddings and graph_items is None:
            graph_items = self.graph_item_vectors()
        long_sequence = self.project_ids(histories, graph_items)
        long_state = self.long_agent.encode(long_sequence, lengths)
        short_ids, short_lengths = self._short_histories(histories, lengths, min(self.short_window, histories.size(1)))
        short_sequence = self.project_ids(short_ids, graph_items)
        short_state = self.short_agent.encode(short_sequence, short_lengths)
        coordinator_state, weights = self.coordinator(long_state, short_state)
        return long_state, short_state, coordinator_state, weights

    def logits_from_states(self, states, candidate_vectors: torch.Tensor):
        long_state, short_state, coordinator_state, weights = states
        long_logits = self.long_agent.logits(long_state, candidate_vectors)
        short_logits = self.short_agent.logits(short_state, candidate_vectors)
        coordinator_logits = self.coordinator.logits(coordinator_state, candidate_vectors)
        final_logits = coordinator_logits + weights[:, :1] * long_logits + weights[:, 1:] * short_logits
        return {
            "long": long_logits,
            "short": short_logits,
            "coordinator": final_logits,
            "states": (long_state, short_state, coordinator_state),
            "weights": weights,
        }

    def forward(self, histories: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor):
        graph_items = self.graph_item_vectors()
        states = self.encode_states(histories, lengths, graph_items)
        candidate_vectors = self.project_ids(candidates, graph_items)
        return self.logits_from_states(states, candidate_vectors)

    def full_catalog_scores(
        self, histories: torch.Tensor, lengths: torch.Tensor,
        item_vectors: torch.Tensor | None = None,
        graph_items: torch.Tensor | None = None,
    ):
        if self.use_graph_embeddings and graph_items is None:
            graph_items = self.graph_item_vectors()
        states = self.encode_states(histories, lengths, graph_items)
        items = self.project_all(graph_items) if item_vectors is None else item_vectors
        return self.logits_from_states(states, items)

    def set_stage(self, stage: str) -> None:
        if stage not in {"specialists", "coordinator", "joint"}:
            raise ValueError(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        graph_modules = (self.graph,) if self.use_graph_embeddings else ()
        modules = (
            (self.long_agent, self.short_agent, self.item_projection, *graph_modules)
            if stage == "specialists"
            else (self.coordinator,)
            if stage == "coordinator"
            else (self.long_agent, self.short_agent, self.coordinator, self.item_projection, *graph_modules)
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def agent_parameter_counts(self) -> dict[str, int]:
        return {
            "long": sum(p.numel() for p in self.long_agent.parameters()),
            "short": sum(p.numel() for p in self.short_agent.parameters()),
            "coordinator": sum(p.numel() for p in self.coordinator.parameters()),
            "shared_projection": sum(p.numel() for p in self.item_projection.parameters()),
            "graph": sum(p.numel() for p in self.graph.parameters()) if self.use_graph_embeddings else 0,
        }
