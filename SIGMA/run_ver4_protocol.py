#!/usr/bin/env python3
"""Train released SIGMA under the MediaTek ver4 split/evaluation protocol."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
VER4_ROOT = ROOT.parent / "MediaTek_2026_0902" / "ver4"
VENDOR_ROOT = ROOT / "vendor"
if VENDOR_ROOT.exists():
    sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(VER4_ROOT))
from src.train_mamba_rl import build_transitions  # noqa: E402
from prepare_ver4_data import COMPLETED_VER4_STATS, split_signature, statistics  # noqa: E402
from model.sigma_ver4_protocol import SIGMA  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("sigma_ver4_protocol")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def amp_context(device: str):
    return nullcontext()


def stable_full_sort_scores(model, sequences, lengths, device: str, include_padding: bool = False):
    """Keep the large catalog projection in FP32 to prevent FP16 overflow."""
    with amp_context(device):
        state = model.encode(sequences, lengths)
    embeddings = model.item_embedding.weight if include_padding else model.item_embedding.weight[1:]
    return state.float() @ embeddings.float().T


def history_batch(data, transitions, max_history: int, device: str):
    sequences = torch.zeros((len(transitions), max_history), dtype=torch.long)
    lengths = torch.empty(len(transitions), dtype=torch.long)
    targets = torch.empty(len(transitions), dtype=torch.long)
    for row, transition in enumerate(transitions):
        history = data.train_by_user[transition.user][
            max(0, transition.end - max_history):transition.end
        ]
        lengths[row] = len(history)
        sequences[row, :len(history)] = torch.tensor(history, dtype=torch.long) + 1
        targets[row] = transition.target
    return sequences.to(device), lengths.to(device), targets.to(device)


def evaluation_batch(data, users: list[int], split: str, max_history: int, device: str):
    sequences = torch.zeros((len(users), max_history), dtype=torch.long)
    lengths = torch.empty(len(users), dtype=torch.long)
    histories: list[list[int]] = []
    for row, user in enumerate(users):
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        history = history[-max_history:]
        histories.append(history)
        lengths[row] = len(history)
        sequences[row, :len(history)] = torch.tensor(history, dtype=torch.long) + 1
    return sequences.to(device), lengths.to(device), histories


@torch.inference_mode()
def evaluate(model, data, split: str, batch_size: int, max_history: int, device: str):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    totals = {
        "recall@5": 0.0,
        "recall@10": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
    }
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in tqdm(
        range(0, len(users), batch_size), desc=f"full-catalog {split}", unit="batch"
    ):
        batch_users = users[start:start + batch_size]
        sequences, lengths, histories = evaluation_batch(
            data, batch_users, split, max_history, device
        )
        gold = torch.tensor([targets[user] for user in batch_users], device=device)
        scores = stable_full_sort_scores(model, sequences, lengths, device)
        if not torch.isfinite(scores.gather(1, gold.unsqueeze(1))).all():
            raise FloatingPointError("non-finite target score during evaluation")
        for row, history in enumerate(histories):
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, gold.unsqueeze(1))).sum(1)
        for cutoff in (5, 10):
            hits = ranks <= cutoff
            totals[f"recall@{cutoff}"] += hits.sum().item()
            totals[f"ndcg@{cutoff}"] += torch.where(
                hits,
                1 / torch.log2(ranks.float() + 1),
                torch.zeros_like(ranks, dtype=torch.float),
            ).sum().item()
        totals["mrr@10"] += torch.where(
            ranks <= 10,
            1 / ranks.float(),
            torch.zeros_like(ranks, dtype=torch.float),
        ).sum().item()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {key: value / len(users) for key, value in totals.items()}
    metrics.update(
        {
            "hit@5": metrics["recall@5"],
            "hit@10": metrics["recall@10"],
            "evaluated_users": len(users),
            "total_users": len(users),
            "users_per_second": len(users) / seconds,
            "scores_per_second": len(users) * data.num_items / seconds,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("all_beauty", "baby_products", "sports_and_outdoors", "toys_and_games"),
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "ver4_protocol"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs_ver4_protocol"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--max-transitions", type=int, default=500_000)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--retained-tail", type=int, default=5)
    parser.add_argument(
        "--save-model-weights", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.max_history,
        args.hidden_size,
    ) < 1:
        parser.error("epochs, batch sizes, max history, and hidden size must be positive")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    artifact = Path(args.data_dir) / f"{args.dataset}.pkl"
    manifest_path = Path(args.data_dir) / f"{args.dataset}.json"
    if not artifact.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Prepare the matched split first: {artifact}")
    with artifact.open("rb") as stream:
        data = pickle.load(stream)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_stats = statistics(data)
    if (
        actual_stats != manifest["statistics"]
        or split_signature(data) != manifest["split_sha256"]
    ):
        raise RuntimeError("Matched data artifact failed statistics/fingerprint verification")
    expected = COMPLETED_VER4_STATS.get(args.dataset)
    if expected is not None and actual_stats != expected:
        raise RuntimeError(
            f"Completed ver4 statistics mismatch: expected={expected}, actual={actual_stats}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_dir) / args.dataset / run_id
    output.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(output / "train.log")
    logger.info("DATA_STATS %s", json.dumps(actual_stats, sort_keys=True))
    logger.info("DATA_SPLIT_SHA256 %s", manifest["split_sha256"])
    logger.info("CONFIG %s", json.dumps(vars(args), sort_keys=True))
    logger.info(
        "METHOD repo_commit=a3f63751503eca869338fc9c8b24a05fa128c6ef "
        "architecture=SIGMA loss=full_catalog_CE precision=fp32"
    )

    model = SIGMA(
        data.num_items,
        max_history=args.max_history,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        retained_tail=args.retained_tail,
    ).to(args.device)
    logger.info("MODEL parameters=%d", sum(parameter.numel() for parameter in model.parameters()))
    base_transitions = build_transitions(data, args.max_transitions, args.seed)
    logger.info("TRAINING_TRANSITIONS %d", len(base_transitions))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    checks_without_improvement = 0
    history = []
    global_step = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        transitions = build_transitions(
            data, len(base_transitions), args.seed + epoch * 1009 + global_step
        )
        model.train()
        loss_total = 0.0
        started = time.perf_counter()
        progress = tqdm(
            range(0, len(transitions), args.batch_size),
            desc=f"train {epoch}/{args.epochs}",
            unit="step",
        )
        completed_steps = 0
        for start in progress:
            batch = transitions[start:start + args.batch_size]
            sequences, lengths, targets = history_batch(
                data, batch, args.max_history, args.device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = stable_full_sort_scores(model, sequences, lengths, args.device, include_padding=True)
            loss = F.cross_entropy(logits, targets + 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            completed_steps += 1
            loss_total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        train_seconds = time.perf_counter() - started
        valid = evaluate(
            model, data, "valid", args.eval_batch_size, args.max_history, args.device
        )
        score = valid["ndcg@10"]
        improved = score > best_score
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss_total / max(completed_steps, 1),
                "train_seconds": train_seconds,
                "valid": valid,
                "improved": improved,
            }
        )
        logger.info(
            "EPOCH %d loss=%.6f valid=%s improved=%s",
            epoch,
            history[-1]["loss"],
            json.dumps(valid, sort_keys=True),
            improved,
        )
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if (
            args.early_stopping_patience
            and checks_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "EARLY_STOP epoch=%d best_epoch=%d best_ndcg@10=%.8f",
                epoch,
                best_epoch,
                best_score,
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation state")
    model.load_state_dict(best_state)
    valid = evaluate(
        model, data, "valid", args.eval_batch_size, args.max_history, args.device
    )
    test = evaluate(
        model, data, "test", args.eval_batch_size, args.max_history, args.device
    )
    result = {
        "dataset": args.dataset,
        "run_id": run_id,
        "data": manifest,
        "valid": valid,
        "test": test,
        "training": {
            "best_epoch": best_epoch,
            "best_ndcg@10": best_score,
            "global_steps": global_step,
            "stopped_early": stopped_early,
            "history": history,
        },
        "config": vars(args),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_model_weights:
        torch.save({"model": best_state, "metrics": result}, output / "best_validation.pt")
    logger.info("VALID_BEST %s", json.dumps(valid, sort_keys=True))
    logger.info("TEST_FINAL %s", json.dumps(test, sort_keys=True))
    logger.info("OUTPUT %s", output)


if __name__ == "__main__":
    main()

