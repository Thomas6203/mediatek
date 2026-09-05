"""Independent REINFORCE + LoRA trainer for Mamba item recommendation.

This module intentionally does not modify the DL baseline.  A policy observes a
user's item-text history, samples an item from a candidate slate, receives a
binary next-item reward, and updates only LoRA adapters on Mamba.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from src.data import build_data, load_amazon, synthetic_events
from src.model import MAMBA_MODEL_ID, load_or_encode_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_CACHE_ROOT = PROJECT_ROOT.parents[1] / "cache"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distributed_setup(enabled: bool) -> tuple[int, int, int]:
    if not enabled:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def configure_logging(directory: Path, run_id: str, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("mamba_rl")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if enabled:
        directory.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.FileHandler(directory / f"rl_{run_id}.log", encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def state_prompts(data, histories: list[list[int]]) -> list[str]:
    """Convert interaction histories into compact Mamba policy prompts."""
    prompts = []
    for history in histories:
        recent = history[-20:]
        text = " \n ".join(data.item_texts[item] for item in recent)
        prompts.append(f"User interaction history: {text}\nNext item:")
    return prompts


class MambaRLPolicy(nn.Module):
    def __init__(self, cache_dir: str, item_vectors: torch.Tensor, device: str):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        show_progress = os.environ.get("RANK", "0") == "0"
        with tqdm(total=1, desc="RL loading Mamba tokenizer", unit="component", disable=not show_progress) as progress:
            self.tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, cache_dir=cache_dir)
            progress.update(1)
        with tqdm(total=1, desc="RL loading trainable Mamba 2.8B", unit="model", disable=not show_progress) as progress:
            self.mamba = AutoModelForCausalLM.from_pretrained(
                MAMBA_MODEL_ID, cache_dir=cache_dir, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
            )
            progress.update(1)
        self.mamba.config.use_cache = False
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            # PEFT bypasses LoRA wrappers for Mamba's fused out_proj/conv1d
            # kernel path.  Keep only projections whose forward path invokes
            # the adapter: in_proj, x_proj, and dt_proj.
            target_modules=["in_proj", "x_proj", "dt_proj"],
        )
        self.mamba = get_peft_model(self.mamba, lora)
        # Static base-Mamba item vectors; do not broadcast this large buffer each DDP step.
        item_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.register_buffer("item_vectors", item_vectors.to(dtype=item_dtype), persistent=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_state(self, prompts: list[str], device: str) -> torch.Tensor:
        batch = self.tokenizer(prompts, padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
        output = self.mamba(**batch, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        mask = batch["attention_mask"].unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

    def forward(self, prompts: list[str], candidate_ids: torch.Tensor, device: str) -> torch.Tensor:
        state = self.encode_state(prompts, device)
        candidates = self.item_vectors[candidate_ids]
        state = state.to(dtype=candidates.dtype)
        return (state.unsqueeze(1) * candidates).sum(-1) * self.log_temperature.exp().clamp(max=20)

    @torch.inference_mode()
    def score_all(self, prompts: list[str], device: str) -> torch.Tensor:
        state = self.encode_state(prompts, device)
        state = state.to(dtype=self.item_vectors.dtype)
        return state @ self.item_vectors.T

    @torch.inference_mode()
    def generate_reason(self, history_texts: list[str], recommendation_text: str, device: str) -> str:
        """Generate a short explanation with the trained Mamba policy."""
        history = " | ".join(history_texts[-10:])
        prompt = (
            "Explain this recommendation in one short sentence. "
            f"The user previously liked: {history}. "
            f"Recommended item: {recommendation_text}. Reason:"
        )
        batch = self.tokenizer(prompt, truncation=True, max_length=192, return_tensors="pt").to(device)
        generated = self.mamba.generate(
            **batch, max_new_tokens=40, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        reason = self.tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return reason or "This item is related to products in the user history."


def make_candidate_slates(positive: torch.Tensor, item_count: int, candidates_per_step: int) -> torch.Tensor:
    negatives = torch.randint(item_count, (positive.size(0), candidates_per_step - 1), device=positive.device)
    while torch.any(negatives == positive.unsqueeze(1)):
        clashes = negatives == positive.unsqueeze(1)
        negatives[clashes] = torch.randint(item_count, (int(clashes.sum()),), device=positive.device)
    return torch.cat((positive.unsqueeze(1), negatives), dim=1)


def build_rl_transitions(data) -> list[tuple[int, list[int], int]]:
    transitions = []
    for user, history in tqdm(data.train_by_user.items(), total=len(data.train_by_user), desc="Building RL next-item transitions", unit="user", disable=os.environ.get("RANK", "0") != "0"):
        for index in range(1, len(history)):
            transitions.append((user, history[:index], history[index]))
    if not transitions:
        raise RuntimeError("RL needs at least two training interactions per user; increase min-user-events or use more data.")
    return transitions


@torch.inference_mode()
def evaluate_full_catalog(policy, data, device: str, batch_size: int, split: str, enabled: bool) -> tuple[dict[str, float], float]:
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    totals = {"recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), batch_size), total=math.ceil(len(users) / batch_size), desc=f"RL full-catalog {split}", unit="batch", disable=not enabled):
        user_batch = users[start:start + batch_size]
        histories = [data.train_by_user[user] if split == "valid" else data.train_by_user[user] + [data.valid_target[user]] for user in user_batch]
        scores = policy.score_all(state_prompts(data, histories), device)
        target = torch.tensor([targets[user] for user in user_batch], device=device)
        for row, history in enumerate(histories):
            seen = set(history) - {int(target[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, target.unsqueeze(1))).sum(1)
        for cutoff in (5, 10):
            totals[f"recall@{cutoff}"] += (ranks <= cutoff).sum().item()
            totals[f"ndcg@{cutoff}"] += torch.where(ranks <= cutoff, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)).sum().item()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    seconds = max(time.perf_counter() - started, 1e-9)
    return {name: value / len(users) for name, value in totals.items()}, len(users) / seconds


def save_rl_outputs(losses: list[float], policy, data, device: str, fig_path: Path, json_path: Path, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", label="REINFORCE loss")
    ax.set(xlabel="Epoch", ylabel="Policy loss", title="RL policy training loss")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(fig_path, dpi=160); plt.close(fig)

    users = sorted(data.test_target)
    sampled_users = random.Random(seed).sample(users, k=min(100, len(users)))
    rows = []
    for user in tqdm(sampled_users, desc="Generating recommendation reasons", unit="user"):
        history = data.train_by_user[user] + [data.valid_target[user]]
        scores = policy.score_all(state_prompts(data, [history]), device)[0]
        scores[list(set(history))] = -torch.inf
        item = int(torch.argmax(scores))
        history_texts = [data.item_texts[index] for index in history]
        rows.append({
            "user_index": user,
            "previously_liked": [{"item_index": index, "item_text": data.item_texts[index]} for index in history],
            "recommendation": {"item_index": item, "item_text": data.item_texts[item], "score": round(float(scores[item]), 6)},
            "recommendation_reason": policy.generate_reason(history_texts, data.item_texts[item], device),
        })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE + LoRA Mamba recommender (independent RL version)")
    parser.add_argument("--subset", default="raw_review_All_Beauty")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates-per-step", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--validate-every-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=str(SHARED_CACHE_ROOT))
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "log"))
    parser.add_argument("--fig-output-dir", default=str(PROJECT_ROOT / "fig_outputs"))
    parser.add_argument("--json-output-dir", default=str(PROJECT_ROOT / "json_outputs"))
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.candidates_per_step < 2 or args.validate_every_steps < 1:
        parser.error("--batch-size and --validate-every-steps must be positive; --candidates-per-step must be at least 2")

    rank, world_size, local_rank = distributed_setup(args.distributed)
    main_process = rank == 0
    if args.distributed:
        args.device = f"cuda:{local_rank}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(Path(args.log_dir), run_id, main_process)
    seed_everything(args.seed + rank)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    event_tag = "full" if args.max_events is None else f"max{args.max_events}"
    artifact = cache_dir / f"mamba_item_vectors_{args.subset}_{event_tag}.pt"

    events = synthetic_events() if args.synthetic else load_amazon(args.subset, args.max_events, str(cache_dir))
    data = build_data(events)
    if main_process:
        item_vectors = load_or_encode_text(data.item_texts, str(artifact), args.device, False, str(cache_dir))
    if args.distributed:
        dist.barrier()
    if not main_process:
        item_vectors = torch.load(artifact, map_location="cpu", weights_only=True)
    assert item_vectors is not None
    policy = MambaRLPolicy(str(cache_dir), item_vectors, args.device).to(args.device)
    if args.distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    transitions = build_rl_transitions(data)
    logger.info("RL run_id=%s transitions=%d items=%d world_size=%d", run_id, len(transitions), data.num_items, world_size)
    baseline = 0.0
    epoch_losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(transitions)
        local = transitions[rank::world_size]
        if args.distributed:
            if not local:
                raise RuntimeError("Number of RL transitions must be at least the number of selected GPUs.")
            steps = math.ceil(len(transitions) / (world_size * args.batch_size))
            local = (local * math.ceil(steps * args.batch_size / len(local)))[:steps * args.batch_size]
        progress = tqdm(range(0, len(local), args.batch_size), total=math.ceil(len(local) / args.batch_size), desc=f"RL epoch {epoch}/{args.epochs} | REINFORCE policy", unit="step", disable=not main_process, dynamic_ncols=True)
        losses = []
        for step, start in enumerate(progress, start=1):
            batch = local[start:start + args.batch_size]
            prompts = state_prompts(data, [history for _, history, _ in batch])
            positive = torch.tensor([target for _, _, target in batch], device=args.device)
            slates = make_candidate_slates(positive, data.num_items, args.candidates_per_step)
            logits = policy(prompts, slates, args.device)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            reward = torch.where(action == 0, torch.ones_like(action, dtype=torch.float), torch.full_like(action, -0.05, dtype=torch.float))
            baseline = 0.95 * baseline + 0.05 * reward.mean().item()
            advantage = reward - baseline
            loss = -(advantage.detach() * distribution.log_prob(action)).mean() - args.entropy_coef * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            global_step += 1
            losses.append(loss.item())
            if main_process:
                progress.set_postfix_str(f"step={step}/{progress.total} | sample-action+REINFORCE | reward={reward.mean().item():.3f} | loss={loss.item():.5f}")
            if global_step % args.validate_every_steps == 0:
                if args.distributed:
                    dist.barrier()
                if main_process:
                    model = policy.module if isinstance(policy, DDP) else policy
                    metrics, throughput = evaluate_full_catalog(model, data, args.device, args.batch_size, "valid", True)
                    logger.info(
                        "RL validation step=%d Recall@5=%.4f Recall@10=%.4f NDCG@5=%.4f NDCG@10=%.4f inference=%.2f users/s",
                        global_step, metrics["recall@5"], metrics["recall@10"], metrics["ndcg@5"], metrics["ndcg@10"], throughput,
                    )
                if args.distributed:
                    dist.barrier()
        stats = torch.tensor([sum(losses), len(losses)], device=args.device, dtype=torch.float64)
        if args.distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        average_loss = (stats[0] / stats[1].clamp_min(1)).item()
        if main_process:
            epoch_losses.append(average_loss)
            logger.info("RL epoch=%d loss=%.5f global_step=%d", epoch, average_loss, global_step)
        if args.distributed:
            dist.barrier()

    if main_process:
        model = policy.module if isinstance(policy, DDP) else policy
        metrics, throughput = evaluate_full_catalog(model, data, args.device, args.batch_size, "test", True)
        adapter_dir = cache_dir / f"mamba_rl_lora_{run_id}"
        model.mamba.save_pretrained(adapter_dir)
        save_rl_outputs(epoch_losses, model, data, args.device, Path(args.fig_output_dir) / f"rl_loss_{run_id}.png", Path(args.json_output_dir) / "run_rl" / f"rl_test_samples_{run_id}.json", args.seed)
        logger.info(
            "RL TEST full-catalog Recall@5=%.4f Recall@10=%.4f NDCG@5=%.4f NDCG@10=%.4f inference=%.2f users/s",
            metrics["recall@5"], metrics["recall@10"], metrics["ndcg@5"], metrics["ndcg@10"], throughput,
        )
        logger.info("saved RL LoRA adapter: %s", adapter_dir)
    if args.distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
