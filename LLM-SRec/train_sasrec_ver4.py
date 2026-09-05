#!/usr/bin/env python3
"""Train LLM-SRec's frozen SASRec teacher under the MediaTek ver4 protocol."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from SeqRec.sasrec.model import SASRec
from time_control import (
    RunTimer,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_path,
    restore_rng_state,
)
from ver4_protocol import DATASETS, DEFAULT_SOURCE_ROOT, ROOT, build_transitions, load_verified_data


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger("llmsrec_sasrec_ver4")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def padded_histories(data, rows, maxlen: int) -> np.ndarray:
    sequences = np.zeros((len(rows), maxlen), dtype=np.int64)
    for index, row in enumerate(rows):
        history = data.train_by_user[row.user][max(0, row.end - maxlen):row.end]
        sequences[index, -len(history):] = np.asarray(history, dtype=np.int64) + 1
    return sequences


def item_embedding(model: SASRec, ids: torch.Tensor) -> torch.Tensor:
    if model.nn_parameter:
        return model.item_emb[ids]
    return model.item_emb(ids)


def sample_negatives(data, rows, rng: random.Random) -> np.ndarray:
    result = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        seen = set(data.train_by_user[row.user])
        candidate = rng.randrange(data.num_items)
        while candidate in seen:
            candidate = rng.randrange(data.num_items)
        result[index] = candidate + 1
    return result


@torch.inference_mode()
def evaluate(model: SASRec, data, split: str, batch_size: int, maxlen: int, device: str):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    totals = {"recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    catalog = item_embedding(
        model, torch.arange(1, data.num_items + 1, dtype=torch.long, device=device)
    ).float()
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), batch_size), desc=f"full-catalog {split}", unit="batch"):
        batch_users = users[start:start + batch_size]
        sequences = np.zeros((len(batch_users), maxlen), dtype=np.int64)
        histories = []
        for row, user in enumerate(batch_users):
            history = list(data.train_by_user[user])
            if split == "test":
                history.append(data.valid_target[user])
            history = history[-maxlen:]
            histories.append(history)
            sequences[row, -len(history):] = np.asarray(history) + 1
        states = model.log2feats(sequences)[:, -1].float()
        scores = states @ catalog.T
        gold = torch.tensor([targets[user] for user in batch_users], device=device)
        gold_scores = scores.gather(1, gold.unsqueeze(1))
        for row, history in enumerate(histories):
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= gold_scores).sum(1)
        for cutoff in (5, 10):
            hits = ranks <= cutoff
            totals[f"recall@{cutoff}"] += hits.sum().item()
            totals[f"ndcg@{cutoff}"] += torch.where(
                hits,
                1 / torch.log2(ranks.float() + 1),
                torch.zeros_like(ranks, dtype=torch.float),
            ).sum().item()
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {key: value / len(users) for key, value in totals.items()}
    metrics.update(
        {
            "hit@5": metrics["recall@5"],
            "hit@10": metrics["recall@10"],
            "evaluated_users": len(users),
            "users_per_second": len(users) / elapsed,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(DATASETS))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs_ver4_protocol" / "sasrec")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval-hours", type=float, default=1.0)
    parser.add_argument("--shutdown-buffer-minutes", type=float, default=2.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, manifest = load_verified_data(args.dataset, args.source_root)
    maximum = args.max_transitions or DATASETS[args.dataset]["max_transitions"]
    training_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "maxlen": args.maxlen,
        "hidden_units": args.hidden_units,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "max_transitions": maximum,
    }
    summary = {
        "dataset": args.dataset,
        "statistics": manifest["statistics"],
        "split_sha256": manifest["split_sha256"],
        "max_transitions": maximum,
        "device": args.device,
        "max_hours": args.max_hours,
        "checkpoint_interval_hours": args.checkpoint_interval_hours,
        "shutdown_buffer_minutes": args.shutdown_buffer_minutes,
        "resume": str(args.resume) if args.resume else None,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("SASRec training requires an available CUDA device")

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_state.get("kind") != "sasrec_ver4_progress":
            raise RuntimeError("The resume file is not a SASRec ver4 progress checkpoint")
        if resume_state.get("dataset") != args.dataset:
            raise RuntimeError("Resume checkpoint dataset mismatch")
        if resume_state.get("split_sha256") != manifest["split_sha256"]:
            raise RuntimeError("Resume checkpoint split fingerprint mismatch")
        if resume_state.get("training_config") != training_config:
            raise RuntimeError("Resume checkpoint training configuration mismatch")
        run_dir = Path(resume_state["run_dir"])
    else:
        run_dir = args.output_root / args.dataset / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)

    seed_everything(args.seed)
    logger = logger_for(run_dir / "train.log")
    logger.info("CONFIG %s", json.dumps(vars(args), sort_keys=True, default=str))
    logger.info("DATA %s", json.dumps(summary, sort_keys=True))

    model_args = Namespace(
        device=args.device,
        hidden_units=args.hidden_units,
        maxlen=args.maxlen,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        nn_parameter=False,
    )
    model = SASRec(data.num_users, data.num_items, model_args).to(args.device)
    for parameter in model.parameters():
        if parameter.dim() > 1:
            torch.nn.init.xavier_normal_(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    start_epoch = 1
    batch_offset = 0
    epoch_loss = 0.0
    epoch_steps = 0
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    previous_elapsed = 0.0
    saved_negative_rng_state = None
    if resume_state:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = int(resume_state["epoch"])
        batch_offset = int(resume_state["batch_offset"])
        epoch_loss = float(resume_state["epoch_loss"])
        epoch_steps = int(resume_state["epoch_steps"])
        best_score = float(resume_state["best_score"])
        best_epoch = int(resume_state["best_epoch"])
        stale = int(resume_state["stale"])
        history = list(resume_state["history"])
        previous_elapsed = float(resume_state["elapsed_seconds"])
        saved_negative_rng_state = resume_state.get("negative_rng_state")
        restore_rng_state(resume_state["rng_state"])
        logger.info("RESUMED checkpoint=%s epoch=%d batch_offset=%d", args.resume, start_epoch, batch_offset)

    timer = RunTimer(
        args.max_hours,
        args.checkpoint_interval_hours,
        previous_elapsed,
        args.shutdown_buffer_minutes,
    )
    best_path = run_dir / "best_model_state.pt"

    def save_progress(epoch: int, offset: int, loss_total: float, steps: int, rng, reason: str) -> Path:
        path = checkpoint_path(run_dir, timer, stopped=reason == "time_limit")
        payload = {
            "kind": "sasrec_ver4_progress",
            "version": 2,
            "dataset": args.dataset,
            "split_sha256": manifest["split_sha256"],
            "run_dir": str(run_dir.resolve()),
            "training_config": training_config,
            "epoch": epoch,
            "batch_offset": offset,
            "epoch_loss": loss_total,
            "epoch_steps": steps,
            "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
            "elapsed_seconds": timer.total_elapsed_seconds,
            "rng_state": capture_rng_state(),
            "negative_rng_state": rng.getstate(),
            "reason": reason,
        }
        atomic_torch_save(payload, path)
        logger.info("CHECKPOINT reason=%s path=%s elapsed_seconds=%.1f", reason, path, timer.total_elapsed_seconds)
        return path

    for epoch in range(start_epoch, args.epochs + 1):
        transitions = build_transitions(data, maximum, args.seed + epoch * 1009)
        rng = random.Random(args.seed + epoch * 7919)
        if epoch == start_epoch and batch_offset and saved_negative_rng_state is not None:
            rng.setstate(saved_negative_rng_state)
        model.train()
        total_loss = epoch_loss if epoch == start_epoch else 0.0
        steps = epoch_steps if epoch == start_epoch else 0
        first_offset = batch_offset if epoch == start_epoch else 0
        for start in tqdm(range(first_offset, len(transitions), args.batch_size), desc=f"train {epoch}/{args.epochs}"):
            rows = transitions[start:start + args.batch_size]
            sequences = padded_histories(data, rows, args.maxlen)
            targets = torch.tensor([row.target + 1 for row in rows], device=args.device)
            negatives = torch.from_numpy(sample_negatives(data, rows, rng)).to(args.device)
            states = model.log2feats(sequences)[:, -1]
            positive = (states * item_embedding(model, targets)).sum(-1)
            negative = (states * item_embedding(model, negatives)).sum(-1)
            loss = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
            loss += F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            next_offset = start + len(rows)
            stopped = timer.time_limit_reached
            if timer.checkpoint_due or stopped:
                saved = save_progress(epoch, next_offset, total_loss, steps, rng, "time_limit" if stopped else "hourly")
                if stopped:
                    status = {
                        **summary,
                        "status": "time_limit_reached",
                        "resume_checkpoint": str(saved.resolve()),
                        "elapsed_seconds": timer.total_elapsed_seconds,
                        "epoch": epoch,
                        "batch_offset": next_offset,
                    }
                    (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                    logger.info("STOP %s", json.dumps(status, sort_keys=True))
                    return

        batch_offset = 0
        epoch_loss = 0.0
        epoch_steps = 0
        valid = evaluate(model, data, "valid", args.eval_batch_size, args.maxlen, args.device)
        improved = valid["ndcg@10"] > best_score
        row = {"epoch": epoch, "loss": total_loss / max(steps, 1), "valid": valid, "improved": improved}
        history.append(row)
        logger.info("EPOCH %s", json.dumps(row, sort_keys=True))
        if improved:
            best_score = valid["ndcg@10"]
            best_epoch = epoch
            atomic_torch_save(
                {"model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}},
                best_path,
            )
            stale = 0
        else:
            stale += 1
        if timer.time_limit_reached:
            saved = save_progress(epoch + 1, 0, 0.0, 0, rng, "time_limit")
            status = {
                **summary,
                "status": "time_limit_reached_after_validation",
                "resume_checkpoint": str(saved.resolve()),
                "elapsed_seconds": timer.total_elapsed_seconds,
                "next_epoch": epoch + 1,
            }
            (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            logger.info("STOP %s", json.dumps(status, sort_keys=True))
            return
        if timer.checkpoint_due:
            save_progress(epoch + 1, 0, 0.0, 0, rng, "hourly")
        if stale >= args.early_stopping_patience:
            break

    if not best_path.exists():
        raise RuntimeError("No SASRec checkpoint was selected")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(best_state)
    test = evaluate(model, data, "test", args.eval_batch_size, args.maxlen, args.device)
    checkpoint_dir = ROOT / "SeqRec" / "sasrec" / args.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint_path = checkpoint_dir / "ver4_best.pth"
    torch.save([model.kwargs, best_state], teacher_checkpoint_path)
    result = {
        **summary,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_ndcg@10": best_score,
        "test": test,
        "history": history,
        "checkpoint": str(teacher_checkpoint_path.resolve()),
        "elapsed_seconds": timer.total_elapsed_seconds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("RESULT %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
