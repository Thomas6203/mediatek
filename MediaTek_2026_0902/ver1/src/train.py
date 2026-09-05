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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.data import build_data, edge_index, load_amazon, synthetic_events
from src.model import HybridRecommender, load_or_encode_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_CACHE_ROOT = PROJECT_ROOT.parents[1] / "cache"


def seed_everything(seed: int) -> None:
    """Set all local random generators before loading data or creating a model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(log_dir: Path, run_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recommender")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_dir / f"train_{run_id}.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_distributed(enabled: bool) -> tuple[int, int, int]:
    """Initialise one NCCL process per selected GPU when launched by torchrun."""
    if not enabled:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("--distributed requires CUDA GPUs.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_loss_curve(losses: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", label="BPR loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training loss", xticks=range(1, len(losses) + 1))
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_test_samples(model, data, edges, device, path: Path, count: int = 5) -> None:
    """Save a small, human-readable set of held-out ranking examples as JSON."""
    model.eval()
    rows = []
    for user in sorted(data.test_target)[:count]:
        target = data.test_target[user]
        user_tensor = torch.tensor([user], device=device)
        all_items = torch.arange(data.num_items, device=device).unsqueeze(0)
        scores = model.score(user_tensor, all_items, edges, padded_histories(data, user_tensor, device))[0]
        top_ids = torch.topk(scores, k=min(10, data.num_items)).indices.tolist()
        rows.append({
            "user_index": user,
            "history_item_indices": data.train_by_user[user],
            "held_out_item_index": target,
            "held_out_item_text": data.item_texts[target],
            "top_recommendations": [
                {"item_index": item, "item_text": data.item_texts[item], "score": round(float(scores[item]), 6)}
                for item in top_ids
            ],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def padded_histories(data, users, device):
    histories = [data.train_by_user[int(u)] for u in users.tolist()]
    width = max(map(len, histories))
    # Left-pad with the first observed item; GRU uses the last actual item either way.
    return torch.tensor([[h[0]] * (width - len(h)) + h for h in histories], device=device)


def candidates_with_target(targets, num_items, negatives, device):
    candidates = []
    for target in targets.tolist():
        pool = [i for i in range(num_items) if i != target]
        sampled = random.sample(pool, k=min(negatives, len(pool)))
        candidates.append([target] + sampled)
    return torch.tensor(candidates, device=device)


@torch.no_grad()
def evaluate(model, data, edges, targets, device, batch_size: int, k=10, progress_desc: str | None = None):
    """Evaluate every held-out user against the complete item catalogue."""
    model.eval()
    user_ids = sorted(targets)
    all_items = torch.arange(data.num_items, device=device)
    recall_sum = ndcg_sum = 0.0
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    starts = range(0, len(user_ids), batch_size)
    for start in tqdm(
        starts,
        total=math.ceil(len(user_ids) / batch_size),
        disable=progress_desc is None,
        dynamic_ncols=True,
        desc=progress_desc,
        unit="batch",
    ):
        batch_user_ids = user_ids[start:start + batch_size]
        users = torch.tensor(batch_user_ids, device=device)
        gold = torch.tensor([targets[user] for user in batch_user_ids], device=device)
        candidates = all_items.unsqueeze(0).expand(len(batch_user_ids), -1)
        scores = model.score(users, candidates, edges, padded_histories(data, users, device))
        target_scores = scores.gather(1, gold.unsqueeze(1))
        ranks = (scores >= target_scores).sum(1)
        recall_sum += (ranks <= k).sum().item()
        ndcg_sum += torch.where(ranks <= k, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)).sum().item()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return recall_sum / len(user_ids), ndcg_sum / len(user_ids), len(user_ids) / elapsed, len(user_ids) * data.num_items / elapsed


def gpu_memory_mb(device: str) -> tuple[float, float]:
    """Return peak allocated and reserved GPU memory in MiB for this process."""
    if not str(device).startswith("cuda"):
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated(device) / 2**20, torch.cuda.max_memory_reserved(device) / 2**20


def main():
    p = argparse.ArgumentParser(description="Mamba + bipartite graph recommender demo")
    p.add_argument("--subset", default="raw_review_All_Beauty")
    p.add_argument("--max-events", type=int, default=None, help="Optional review cap; omit to use the complete review split")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=str(SHARED_CACHE_ROOT), help="Hugging Face dataset and Mamba cache directory")
    p.add_argument("--artifact", default=None, help="Optional item-vector cache path; defaults under --cache-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=str(PROJECT_ROOT / "log"))
    p.add_argument("--fig-output-dir", default=str(PROJECT_ROOT / "fig_outputs"))
    p.add_argument("--json-output-dir", default=str(PROJECT_ROOT / "json_outputs"))
    p.add_argument("--test-sample-count", type=int, default=5)
    p.add_argument("--distributed", action="store_true", help="Enable DDP; set by run.sh for two or more GPUs")
    p.add_argument("--skip-mamba", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    rank, world_size, local_rank = setup_distributed(args.distributed)
    is_main_process = rank == 0
    if args.distributed:
        args.device = f"cuda:{local_rank}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(Path(args.log_dir), run_id) if is_main_process else logging.getLogger("recommender.worker")
    if not is_main_process:
        logger.addHandler(logging.NullHandler())
    seed_everything(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    event_tag = "full" if args.max_events is None else f"max{args.max_events}"
    artifact = args.artifact or str(cache_dir / f"mamba_item_vectors_{args.subset}_{event_tag}.pt")
    logger.info("run_id=%s seed=%d device=%s cache_dir=%s world_size=%d", run_id, args.seed, args.device, cache_dir, world_size)

    events = synthetic_events() if args.synthetic else load_amazon(args.subset, args.max_events, str(cache_dir))
    data = build_data(events)
    logger.info("users=%d items=%d train_edges=%d", data.num_users, data.num_items, sum(map(len, data.train_by_user.values())))
    # Only rank 0 ever loads the 2.8B model for encoding. Other ranks wait, then
    # read the completed item-vector cache, avoiding duplicate model memory.
    if is_main_process:
        text = load_or_encode_text(data.item_texts, artifact, args.device, args.skip_mamba, str(cache_dir))
        mamba_peak_allocated, mamba_peak_reserved = gpu_memory_mb(args.device)
    if args.distributed:
        dist.barrier()
    if not is_main_process:
        text = load_or_encode_text(data.item_texts, artifact, args.device, args.skip_mamba, str(cache_dir))
    # Restore identical initial recommender weights across DDP ranks after rank 0's encoding.
    seed_everything(args.seed)
    model = HybridRecommender(data.num_users, data.num_items, args.dim, text).to(args.device)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    edges = edge_index(data).to(args.device)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    train_pairs = []
    for user, history in tqdm(data.train_by_user.items(), total=len(data.train_by_user), disable=not is_main_process, desc="Creating full training interaction list", unit="user"):
        train_pairs.extend((user, item) for item in history)
    logger.info("full train interactions per epoch=%d; full test users=%d; catalog items=%d", len(train_pairs), len(data.test_target), data.num_items)

    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_pairs); losses = []
        local_pairs = train_pairs[rank::world_size]
        # DDP requires every rank to execute the same number of backward calls.
        # Pad the final shard by repeating local pairs, rather than dropping data.
        if args.distributed:
            if not local_pairs:
                raise RuntimeError("Number of train interactions must be at least the number of selected GPUs.")
            batches_per_rank = math.ceil(len(train_pairs) / (world_size * args.batch_size))
            required_users = batches_per_rank * args.batch_size
            repeats = math.ceil(required_users / len(local_pairs))
            local_pairs = (local_pairs * repeats)[:required_users]
        batch_starts = range(0, len(local_pairs), args.batch_size)
        progress = tqdm(
            batch_starts,
            total=math.ceil(len(local_pairs) / args.batch_size),
            disable=not is_main_process,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}/{args.epochs} | BPR ranking",
            unit="step",
        )
        for step, start in enumerate(progress, start=1):
            pairs = local_pairs[start:start + args.batch_size]
            users = torch.tensor([user for user, _ in pairs], device=args.device)
            positive = torch.tensor([item for _, item in pairs], device=args.device)
            negative = torch.randint(data.num_items, positive.shape, device=args.device)
            # BPR needs a genuinely unobserved comparison item for each draw.
            while torch.any(negative == positive):
                clash = negative == positive
                negative[clash] = torch.randint(data.num_items, (int(clash.sum()),), device=args.device)
            candidates = torch.stack((positive, negative), 1)
            scores = model(users, candidates, edges, padded_histories(data, users, args.device))
            loss = F.softplus(-(scores[:, 0] - scores[:, 1])).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
            if is_main_process:
                progress.set_postfix_str(
                    f"step={step}/{progress.total} | backward+optimizer | loss={loss.item():.5f}"
                )
        loss_stats = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=args.device)
        if args.distributed:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        average_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        if is_main_process:
            epoch_losses.append(average_loss)
            recall, ndcg, users_per_second, _ = evaluate(unwrap_model(model), data, edges, data.valid_target, args.device, args.batch_size, progress_desc=f"Epoch {epoch}/{args.epochs} | Full-catalog validation")
            logger.info("epoch=%d loss=%.4f full-catalog valid Recall@10=%.4f NDCG@10=%.4f inference=%.2f users/s", epoch, average_loss, recall, ndcg, users_per_second)
        if args.distributed:
            dist.barrier()
    if is_main_process:
        recommender = unwrap_model(model)
        recall, ndcg, users_per_second, item_scores_per_second = evaluate(recommender, data, edges, data.test_target, args.device, args.batch_size, progress_desc="Full-catalog test evaluation")
        train_peak_allocated, train_peak_reserved = gpu_memory_mb(args.device)
        total_required_memory = max(mamba_peak_allocated, train_peak_allocated)
        logger.info("TEST full-catalog Recall@10=%.4f NDCG@10=%.4f", recall, ndcg)
        logger.info("full-catalog inference=%.2f users/s, %.2f user-item scores/s", users_per_second, item_scores_per_second)
        logger.info("GPU memory peak: Mamba encoding allocated=%.1f MiB reserved=%.1f MiB; training allocated=%.1f MiB reserved=%.1f MiB; required peak allocated=%.1f MiB", mamba_peak_allocated, mamba_peak_reserved, train_peak_allocated, train_peak_reserved, total_required_memory)
        loss_path = Path(args.fig_output_dir) / f"loss_{run_id}.png"
        samples_path = Path(args.json_output_dir) / f"test_samples_{run_id}.json"
        save_loss_curve(epoch_losses, loss_path)
        save_test_samples(recommender, data, edges, args.device, samples_path, args.test_sample_count)
        logger.info("loss curve: %s", loss_path)
        logger.info("test samples: %s", samples_path)
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
