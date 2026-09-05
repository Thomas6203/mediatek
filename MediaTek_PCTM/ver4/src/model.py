from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

MAMBA_MODEL_ID = "state-spaces/mamba-2.8b-hf"


class LightGCN(nn.Module):
    def __init__(self, users: int, items: int, dim: int, layers: int = 2):
        super().__init__()
        self.users, self.items, self.layers = users, items, layers
        self.embedding = nn.Embedding(users + items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edges: torch.Tensor, progress=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding.weight
        src, dst = edges
        degree = torch.bincount(src, minlength=x.size(0)).clamp_min(1).float().to(x.device)
        all_layers = [x]
        if progress is not None:
            progress.update(1)
        for _ in range(self.layers):
            out = torch.zeros_like(x)
            weight = (degree[src] * degree[dst]).rsqrt().unsqueeze(1)
            out.index_add_(0, dst, x[src] * weight)
            x = out
            all_layers.append(x)
            if progress is not None:
                progress.update(1)
        x = torch.stack(all_layers).mean(0)
        return x[: self.users], x[self.users :]


class MambaTextEncoder:
    """Frozen product-text encoder backed by the official HF Mamba 2.8B checkpoint."""
    def __init__(self, device: str, cache_dir: str, max_tokens: int = 48):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.max_tokens = device, max_tokens
        # The original mamba-2.8b repository has no complete HF tokenizer files.
        # Its official -hf companion keeps the same checkpoint and adds GPT-NeoX
        # tokenizer/config assets required by AutoTokenizer and Transformers.
        with tqdm(total=1, desc="Loading Mamba tokenizer", unit="component") as progress:
            self.tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, cache_dir=cache_dir)
            progress.update(1)
        with tqdm(total=1, desc="Loading Mamba 2.8B weights", unit="model") as progress:
            self.model = AutoModelForCausalLM.from_pretrained(
                MAMBA_MODEL_ID, cache_dir=cache_dir,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            ).to(device).eval()
            progress.update(1)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(
        self, texts: list[str], batch_size: int = 4, prompt_prefix: str = ""
    ) -> torch.Tensor:
        vectors = []
        prefix_length = 0
        if prompt_prefix:
            prefix_length = len(self.tokenizer(
                prompt_prefix, add_special_tokens=False
            )["input_ids"])
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding item text with Mamba", unit="batch"):
            prompted = [
                prompt_prefix + text for text in texts[start : start + batch_size]
            ]
            batch = self.tokenizer(prompted, padding=True, truncation=True,
                                   max_length=self.max_tokens, return_tensors="pt").to(self.device)
            # The recommender only needs the backbone representation. Calling the
            # CausalLM wrapper also materialises full-vocabulary logits that are
            # immediately discarded; the backbone hidden state is numerically
            # identical and substantially cheaper for large item catalogs.
            output = self.model.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            content_mask = batch["attention_mask"].clone()
            if prefix_length:
                content_mask[:, :min(prefix_length, content_mask.size(1))] = 0
                empty = content_mask.sum(1) == 0
                content_mask[empty] = batch["attention_mask"][empty]
            mask = content_mask.unsqueeze(-1)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
        return torch.cat(vectors)


class HybridRecommender(nn.Module):
    def __init__(self, users: int, items: int, dim: int, text_vectors: torch.Tensor | None):
        super().__init__()
        self.graph = LightGCN(users, items, dim)
        self.item_fallback = nn.Embedding(items, dim)
        self.text_projection = nn.Linear(text_vectors.size(1), dim, bias=False) if text_vectors is not None else None
        if text_vectors is not None:
            self.register_buffer("text_vectors", text_vectors.float())
        self.history = nn.GRU(dim, dim, batch_first=True)
        self.mix = nn.Parameter(torch.tensor(0.5))

    def item_vectors(self, graph_items: torch.Tensor) -> torch.Tensor:
        if self.text_projection is None:
            return graph_items + self.item_fallback.weight
        return graph_items + self.text_projection(self.text_vectors)

    def score(self, users: torch.Tensor, candidates: torch.Tensor, edges: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        graph_users, graph_items = self.graph(edges)
        items = self.item_vectors(graph_items)
        sequence, _ = self.history(items[histories])
        state = sequence[:, -1]
        user = graph_users[users]
        candidate = items[candidates]
        graph_score = (user.unsqueeze(1) * candidate).sum(-1)
        text_score = (state.unsqueeze(1) * candidate).sum(-1)
        alpha = torch.sigmoid(self.mix)
        return alpha * graph_score + (1 - alpha) * text_score

    def forward(self, users: torch.Tensor, candidates: torch.Tensor, edges: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        """DDP-compatible entry point for candidate ranking."""
        return self.score(users, candidates, edges, histories)


def load_or_encode_text(
    texts: list[str], artifact: str, device: str, skip_mamba: bool, cache_dir: str,
    batch_size: int = 4, max_tokens: int = 48, prompt_prefix: str = "",
) -> torch.Tensor | None:
    path = Path(artifact)
    if skip_mamba:
        return None
    if path.exists():
        with tqdm(total=1, desc="Loading cached Mamba item vectors", unit="artifact") as progress:
            vectors = torch.load(path, map_location="cpu", weights_only=True)
            progress.update(1)
        return vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = MambaTextEncoder(device, cache_dir, max_tokens=max_tokens).encode(
        texts, batch_size=batch_size, prompt_prefix=prompt_prefix
    )
    with tqdm(total=1, desc="Saving Mamba item-vector cache", unit="artifact") as progress:
        torch.save(vectors, path)
        progress.update(1)
    return vectors
