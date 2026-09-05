#!/usr/bin/env python3
"""Train and evaluate LLM-SRec with MediaTek ver4 controls.

The upstream model implementation is retained.  This runner replaces its
sampling, split selection, user subsampling, candidate evaluation, checkpoint
selection, and experiment bookkeeping.
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
from tqdm.auto import tqdm

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
    logger = logging.getLogger("llmsrec_ver4")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def phase2_loss(model, batch, epoch: int, epochs: int, step: int, total_steps: int):
    """Upstream LLM-SRec phase-2 objective, with optimization left to the runner."""
    users, sequences, positives, negatives = batch
    with torch.no_grad():
        teacher_states = model.recsys.model(users, sequences, positives, negatives, mode="log_only")

    prompts = []
    candidate_prompts = []
    history_embeddings = []
    candidate_embeddings = []
    for row in range(len(users)):
        target = positives[row][-1]
        target_title = model.find_item_text_single(target, title_flag=True, description_flag=False)
        history_text, history_ids = model.make_interact_text(
            sequences[row][sequences[row] > 0], 10, users[row]
        )
        texts, candidate_ids = model.make_candidate_text(
            sequences[row][sequences[row] > 0], 4, target, target_title, task="RecTask"
        )
        prompts.append(
            "This user has made a series of purchases in the following order: "
            + history_text
            + ". Based on this sequence of purchases, generate user representation token:[UserOut]"
        )
        candidate_prompts.extend(texts)
        history_embeddings.append(model.item_emb_proj(model.get_item_emb(history_ids)))
        candidate_embeddings.append(
            model.item_emb_proj(model.get_item_emb([candidate_ids])).squeeze(0)
        )

    samples = {
        "text_input": prompts,
        "log_emb": teacher_states,
        "candidates_pos": candidate_prompts,
        "interact": history_embeddings,
        "candidate_embs": torch.cat(candidate_embeddings),
    }
    loss, rec_loss, match_loss = model.llm(samples, mode=0)
    return loss, rec_loss, match_loss


def training_batch(data, transitions, maxlen: int, rng: random.Random):
    users = np.asarray([row.user + 1 for row in transitions], dtype=np.int64)
    sequences = np.zeros((len(transitions), maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    for index, row in enumerate(transitions):
        history = data.train_by_user[row.user][max(0, row.end - maxlen):row.end]
        sequences[index, -len(history):] = np.asarray(history) + 1
        positives[index, -1] = row.target + 1
        seen = set(data.train_by_user[row.user])
        negative = rng.randrange(data.num_items)
        while negative in seen:
            negative = rng.randrange(data.num_items)
        negatives[index, -1] = negative + 1
    return users, sequences, positives, negatives


@torch.inference_mode()
def encode_catalog(model, batch_size: int) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in tqdm(range(1, model.item_num + 1, batch_size), desc="encoding full catalog"):
        ids = list(range(start, min(start + batch_size, model.item_num + 1)))
        texts = [
            "The item title and item embedding are as follows: "
            + model.find_item_text_single(item, title_flag=True, description_flag=False)
            + "[HistoryEmb], then generate item representation token:[ItemOut]"
            for item in ids
        ]
        tokens = model.llm.llm_tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True, max_length=1024
        ).to(model.device)
        input_embeddings = model.llm.llm_model.get_input_embeddings()(tokens["input_ids"])
        projected = model.item_emb_proj(model.get_item_emb(ids))
        input_embeddings = model.llm.replace_out_token_all_infer(
            tokens,
            input_embeddings,
            token=["[ItemOut]", "[HistoryEmb]"],
            embs={"[HistoryEmb]": projected},
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model.llm.llm_model(
                inputs_embeds=input_embeddings, output_hidden_states=True
            )
            positions = model.llm.get_embeddings(tokens, "[ItemOut]")
            states = torch.cat(
                [
                    result.hidden_states[-1][row, position].mean(0).unsqueeze(0)
                    for row, position in enumerate(positions)
                ]
            )
            outputs.append(model.llm.pred_item(states))
    return torch.cat(outputs)


@torch.inference_mode()
def encode_users(model, data, users: list[int], split: str, maxlen: int) -> torch.Tensor:
    prompts = []
    history_embeddings = []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        history = history[-maxlen:]
        one_based = np.asarray(history, dtype=np.int64) + 1
        text, used_ids = model.make_interact_text(one_based, 10, user + 1)
        prompts.append(
            "This user has made a series of purchases in the following order: "
            + text
            + ". Based on this sequence of purchases, generate user representation token:[UserOut]"
        )
        history_embeddings.append(model.item_emb_proj(model.get_item_emb(used_ids)))
    tokens = model.llm.llm_tokenizer(
        prompts, return_tensors="pt", padding="longest", truncation=True, max_length=1024
    ).to(model.device)
    input_embeddings = model.llm.llm_model.get_input_embeddings()(tokens["input_ids"])
    input_embeddings = model.llm.replace_out_token_all(
        tokens,
        input_embeddings,
        token=["[UserOut]", "[HistoryEmb]"],
        embs={"[HistoryEmb]": history_embeddings},
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = model.llm.llm_model(inputs_embeds=input_embeddings, output_hidden_states=True)
        positions = model.llm.get_embeddings(tokens, "[UserOut]")
        states = torch.cat(
            [
                result.hidden_states[-1][row, position].mean(0).unsqueeze(0)
                for row, position in enumerate(positions)
            ]
        )
        return model.llm.pred_user(states)


@torch.inference_mode()
def evaluate(
    model,
    data,
    catalog: torch.Tensor,
    split: str,
    user_batch_size: int,
    catalog_chunk_size: int,
    maxlen: int,
):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)  # Never sample or truncate evaluation users.
    totals = {"recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), user_batch_size), desc=f"full-catalog {split}"):
        batch_users = users[start:start + user_batch_size]
        user_states = encode_users(model, data, batch_users, split, maxlen).float()
        gold = torch.tensor([targets[user] for user in batch_users], device=model.device)
        gold_scores = (user_states * catalog[gold].float()).sum(1)
        ranks = torch.zeros(len(batch_users), dtype=torch.long, device=model.device)
        histories = []
        for user in batch_users:
            history = list(data.train_by_user[user])
            if split == "test":
                history.append(data.valid_target[user])
            histories.append(set(history) - {targets[user]})
        for item_start in range(0, data.num_items, catalog_chunk_size):
            item_end = min(item_start + catalog_chunk_size, data.num_items)
            scores = user_states @ catalog[item_start:item_end].float().T
            for row, seen in enumerate(histories):
                local = [item - item_start for item in seen if item_start <= item < item_end]
                if local:
                    scores[row, local] = -torch.inf
            ranks += (scores >= gold_scores.unsqueeze(1)).sum(1)
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
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs_ver4_protocol" / "llmsrec")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm", choices=("llama-3b", "llama"), default="llama-3b")
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--catalog-encode-batch-size", type=int, default=128)
    parser.add_argument("--catalog-chunk-size", type=int, default=32768)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval-hours", type=float, default=1.0)
    parser.add_argument("--shutdown-buffer-minutes", type=float, default=2.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--nn-parameter", action="store_true")
    parser.add_argument("--token", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, manifest = load_verified_data(args.dataset, args.source_root)
    maximum = args.max_transitions or DATASETS[args.dataset]["max_transitions"]
    data_dir = ROOT / "SeqRec" / f"data_{args.dataset}"
    metadata_path = data_dir / "text_name_dict.json.gz"
    metadata_manifest = data_dir / "ver4_manifest.json"
    teacher_dir = ROOT / "SeqRec" / "sasrec" / args.dataset
    teacher_checkpoints = list(teacher_dir.glob("*.pth"))
    training_config = {
        "llm": args.llm,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "catalog_encode_batch_size": args.catalog_encode_batch_size,
        "catalog_chunk_size": args.catalog_chunk_size,
        "maxlen": args.maxlen,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "max_transitions": maximum,
        "nn_parameter": args.nn_parameter,
        "token": args.token,
        "teacher_checkpoint": str(teacher_checkpoints[0].resolve()) if len(teacher_checkpoints) == 1 else None,
    }
    preflight = {
        "dataset": args.dataset,
        "statistics": manifest["statistics"],
        "split_sha256": manifest["split_sha256"],
        "max_transitions": maximum,
        "max_hours": args.max_hours,
        "checkpoint_interval_hours": args.checkpoint_interval_hours,
        "shutdown_buffer_minutes": args.shutdown_buffer_minutes,
        "resume": str(args.resume) if args.resume else None,
        "metadata_ready": metadata_path.exists() and metadata_manifest.exists(),
        "teacher_checkpoint_count": len(teacher_checkpoints),
        "teacher_checkpoint_dir": str(teacher_dir),
        "llm_model": args.llm,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return
    if not preflight["metadata_ready"]:
        raise FileNotFoundError(f"Run prepare_ver4_data.py first: {metadata_path}")
    prepared = json.loads(metadata_manifest.read_text(encoding="utf-8"))
    if prepared.get("placeholders_only"):
        raise RuntimeError("Placeholder metadata is forbidden for an experiment run")
    if len(teacher_checkpoints) != 1:
        raise RuntimeError(f"Exactly one SASRec teacher checkpoint is required in {teacher_dir}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("LLM-SRec training requires an available CUDA device")

    os.chdir(ROOT)  # Preserve paths used inside the released implementation.
    from models.seqllm_model import llmrec_model

    seed_everything(args.seed)
    args.recsys = "sasrec"
    args.rec_pre_trained_data = args.dataset
    args.save_dir = "ver4_protocol"
    args.train = True
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_state.get("kind") != "llmsrec_ver4_progress":
            raise RuntimeError("The resume file is not an LLM-SRec ver4 progress checkpoint")
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
    logger = logger_for(run_dir / "train.log")
    logger.info("CONFIG %s", json.dumps(vars(args), sort_keys=True, default=str))
    logger.info("PREFLIGHT %s", json.dumps(preflight, sort_keys=True))

    model = llmrec_model(args).to(args.device)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)

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
        model.load_state_dict(resume_state["model_state"], strict=False)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
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
    best_path = run_dir / "best_trainable.pt"

    def save_progress(epoch: int, offset: int, loss_total: float, steps: int, rng, reason: str) -> Path:
        path = checkpoint_path(run_dir, timer, stopped=reason == "time_limit")
        payload = {
            "kind": "llmsrec_ver4_progress",
            "version": 2,
            "dataset": args.dataset,
            "split_sha256": manifest["split_sha256"],
            "run_dir": str(run_dir.resolve()),
            "training_config": training_config,
            "epoch": epoch,
            "batch_offset": offset,
            "epoch_loss": loss_total,
            "epoch_steps": steps,
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if name in trainable
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
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
        total_loss = epoch_loss if epoch == start_epoch else 0.0
        steps = epoch_steps if epoch == start_epoch else 0
        first_offset = batch_offset if epoch == start_epoch else 0
        model.train()
        for start in tqdm(range(first_offset, len(transitions), args.batch_size), desc=f"train {epoch}/{args.epochs}"):
            rows = transitions[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            effective_loss = 0.0
            for micro_start in range(0, len(rows), args.micro_batch_size):
                micro_rows = rows[micro_start:micro_start + args.micro_batch_size]
                batch = training_batch(data, micro_rows, args.maxlen, rng)
                loss, _, _ = phase2_loss(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    steps,
                    math.ceil(len(transitions) / args.batch_size),
                )
                (loss * (len(micro_rows) / len(rows))).backward()
                effective_loss += loss.item() * len(micro_rows) / len(rows)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
            )
            optimizer.step()
            total_loss += effective_loss
            steps += 1
            next_offset = start + len(rows)
            stopped = timer.time_limit_reached
            if timer.checkpoint_due or stopped:
                saved = save_progress(epoch, next_offset, total_loss, steps, rng, "time_limit" if stopped else "hourly")
                if stopped:
                    status = {
                        **preflight,
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
        scheduler.step()
        # Item encodings depend on trainable projection/output heads and must be refreshed.
        catalog = encode_catalog(model, args.catalog_encode_batch_size)
        valid = evaluate(
            model, data, catalog, "valid", args.eval_batch_size, args.catalog_chunk_size, args.maxlen
        )
        improved = valid["ndcg@10"] > best_score
        row = {"epoch": epoch, "loss": total_loss / max(steps, 1), "valid": valid, "improved": improved}
        history.append(row)
        logger.info("EPOCH %s", json.dumps(row, sort_keys=True))
        if improved:
            best_score = valid["ndcg@10"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if name in trainable
            }
            atomic_torch_save(best_state, best_path)
            stale = 0
        else:
            stale += 1
        if timer.time_limit_reached:
            saved = save_progress(epoch + 1, 0, 0.0, 0, rng, "time_limit")
            status = {
                **preflight,
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
        raise RuntimeError("No LLM-SRec checkpoint was selected")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state, strict=False)
    catalog = encode_catalog(model, args.catalog_encode_batch_size)
    test = evaluate(
        model, data, catalog, "test", args.eval_batch_size, args.catalog_chunk_size, args.maxlen
    )
    result = {
        **preflight,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_ndcg@10": best_score,
        "test": test,
        "history": history,
        "checkpoint": str(best_path.resolve()),
        "elapsed_seconds": timer.total_elapsed_seconds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("RESULT %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
