"""Independent three-agent Mamba-RL training entry point."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.data import edge_index
from src.data_mamba_rl import load_recommendation_data
from src.model import MAMBA_MODEL_ID, load_or_encode_text
from src.model_mamba_rl import MultiAgentMambaRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_CACHE_ROOT = PROJECT_ROOT.parents[1] / "cache"


@dataclass(frozen=True)
class Transition:
    user: int
    end: int
    target: int


@dataclass
class TrainingMonitor:
    """Track periodic validation and retain the best deployable policy."""
    metric_name: str
    checkpoint_path: Path | None
    global_step: int = 0
    best_score: float = -math.inf
    best_step: int = 0
    best_stage: str = ""
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    checks_without_improvement: int = 0
    stopped_early: bool = False
    history: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogPriors:
    """Training-only catalog statistics used to complement semantic policy scores."""
    popularity: torch.Tensor
    transitions: dict[int, dict[int, float]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_agent_mamba_rl")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_recommendation_data_cached(args, logger):
    """Cache the normalized split so repeated tuning does not rescan raw datasets."""
    identity = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "max_events": args.max_events,
        "min_rating": args.min_rating,
        "min_user_events": args.min_user_events,
        "schema_version": 1,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    artifact = Path(args.cache_dir) / "mamba_multi_agent_data" / f"{safe_dataset}_{digest}.pkl"
    if artifact.exists() and not args.refresh_data_cache:
        with artifact.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", artifact)
        return data, artifact

    data = load_recommendation_data(
        args.dataset, args.data_path, args.cache_dir, args.max_events,
        args.min_rating, args.min_user_events,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(artifact)
    logger.info("INTERACTION_CACHE miss_saved path=%s", artifact)
    return data, artifact


def build_transitions(data, maximum: int | None, seed: int) -> list[Transition]:
    """Bound training samples while maximising eligible-user coverage."""
    rng = random.Random(seed)
    eligible = [
        (user, len(history) - 1)
        for user, history in data.train_by_user.items()
        if len(history) > 1
    ]
    total = sum(count for _, count in eligible)
    budget = total if maximum is None or maximum <= 0 else min(maximum, total)
    rng.shuffle(eligible)

    # Give as many users as possible one randomly selected prefix before
    # spending the remaining budget on additional interactions.
    selected_ends: dict[int, int] = {}
    result: list[Transition] = []
    for user, count in eligible[:budget]:
        end = rng.randint(1, count)
        selected_ends[user] = end
        result.append(Transition(user, end, data.train_by_user[user][end]))

    remaining_budget = budget - len(result)
    reservoir: list[Transition] = []
    seen_remaining = 0
    if remaining_budget > 0:
        for user, count in eligible:
            selected = selected_ends.get(user)
            history = data.train_by_user[user]
            for end in range(1, count + 1):
                if end == selected:
                    continue
                transition = Transition(user, end, history[end])
                seen_remaining += 1
                if len(reservoir) < remaining_budget:
                    reservoir.append(transition)
                else:
                    replacement = rng.randrange(seen_remaining)
                    if replacement < remaining_budget:
                        reservoir[replacement] = transition
        result.extend(reservoir)
    if not result:
        raise RuntimeError("No train prefixes exist; load more interactions or lower --min-user-events.")
    rng.shuffle(result)
    return result


def history_batch(data, transitions: list[Transition], max_history: int, device: str):
    histories = [data.train_by_user[row.user][max(0, row.end - max_history):row.end] for row in transitions]
    lengths = torch.tensor([len(history) for history in histories], device=device)
    padded = torch.zeros((len(histories), max(int(lengths.max()), 1)), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        padded[row, :len(history)] = torch.tensor(history, device=device)
    return padded, lengths, torch.tensor([row.target for row in transitions], device=device)


def evaluation_history_batch(data, users: list[int], split: str, max_history: int, device: str):
    histories = []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        histories.append(history[-max_history:])
    lengths = torch.tensor([len(history) for history in histories], device=device)
    padded = torch.zeros((len(users), max(int(lengths.max()), 1)), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        padded[row, :len(history)] = torch.tensor(history, device=device)
    return padded, lengths, histories


def candidate_slates(targets: torch.Tensor, num_items: int, count: int) -> torch.Tensor:
    negatives = torch.randint(num_items, (targets.size(0), count - 1), device=targets.device)
    while torch.any(negatives == targets.unsqueeze(1)):
        clashes = negatives == targets.unsqueeze(1)
        negatives[clashes] = torch.randint(num_items, (int(clashes.sum()),), device=targets.device)
    return torch.cat((targets.unsqueeze(1), negatives), dim=1)


def build_catalog_priors(data, device: str) -> CatalogPriors:
    """Build popularity and first-order transition priors from training histories only."""
    popularity = torch.zeros(data.num_items, dtype=torch.float32)
    transition_counts: dict[int, dict[int, int]] = {}
    for history in data.train_by_user.values():
        for item in history:
            popularity[item] += 1
        for previous, following in zip(history, history[1:]):
            row = transition_counts.setdefault(previous, {})
            row[following] = row.get(following, 0) + 1
    popularity = torch.log1p(popularity)
    popularity = (popularity - popularity.mean()) / popularity.std().clamp_min(1e-6)
    transitions = {
        previous: {following: math.log1p(count) for following, count in row.items()}
        for previous, row in transition_counts.items()
    }
    return CatalogPriors(popularity.to(device), transitions)


def amp_context(device: str):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()


def record_validation(model, metrics, stage, epoch, monitor, logger) -> bool:
    """Record a validation check and atomically persist a newly best model."""
    score = float(metrics[monitor.metric_name])
    monitor.history.append({
        "global_step": monitor.global_step,
        "stage": stage,
        "epoch": epoch,
        **{name: float(value) for name, value in metrics.items()},
    })
    improved = score > monitor.best_score
    logger.info(
        "VALID_STEP step=%d stage=%s epoch=%d %s=%.6f recall@10=%.6f improved=%s",
        monitor.global_step, stage, epoch, monitor.metric_name, score,
        metrics["recall@10"], improved,
    )
    if improved:
        monitor.best_score = score
        monitor.best_step = monitor.global_step
        monitor.best_stage = stage
        monitor.best_metrics = dict(metrics)
        monitor.best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        monitor.checks_without_improvement = 0
        if monitor.checkpoint_path is not None:
            payload = {
                "model": monitor.best_state,
                "valid_metrics": monitor.best_metrics,
                "monitor_metric": monitor.metric_name,
                "best_score": monitor.best_score,
                "best_step": monitor.best_step,
                "best_stage": monitor.best_stage,
            }
            temporary = monitor.checkpoint_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(monitor.checkpoint_path)
            logger.info(
                "BEST_CHECKPOINT step=%d stage=%s %s=%.6f path=%s",
                monitor.best_step, monitor.best_stage, monitor.metric_name,
                monitor.best_score, monitor.checkpoint_path,
            )
        else:
            logger.info(
                "BEST_MODEL_IN_MEMORY step=%d stage=%s %s=%.6f",
                monitor.best_step, monitor.best_stage, monitor.metric_name,
                monitor.best_score,
            )
    elif stage == "joint":
        monitor.checks_without_improvement += 1
    return improved


def log_metric_block(logger, label, metrics, stage, epoch, step) -> None:
    """Log the six ranking metrics as one readable multi-line record."""
    logger.info(
        "\n========== %s ==========\n"
        "stage=%s | epoch=%d | step=%d | users=%d/%d\n"
        "NDCG  | @5 %.6f | @10 %.6f\n"
        "Recall| @5 %.6f | @10 %.6f\n"
        "Hit   | @5 %.6f | @10 %.6f\n"
        "================================",
        label, stage, epoch, step,
        int(metrics.get("evaluated_users", 0)), int(metrics.get("total_users", 0)),
        metrics["ndcg@5"], metrics["ndcg@10"],
        metrics["recall@5"], metrics["recall@10"],
        metrics["hit@5"], metrics["hit@10"],
    )


def train_stage(model, data, transitions, stage, epochs, batch_size, candidates, max_history,
                learning_rate, entropy_coef, supervised_coef, specialization_coef, device, logger,
                monitor, validate_every_steps, eval_batch_size, early_stopping_patience,
                lr_patience, full_catalog_supervised, catalog_priors, popularity_alpha,
                transition_beta, validation_user_limit, periodic_test_user_limit):
    if epochs == 0:
        return []
    model.set_stage(stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    scheduler = None
    if stage in {"coordinator", "joint"} and validate_every_steps > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=lr_patience, min_lr=1e-6
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    baselines = {"long": 0.0, "short": 0.0, "coordinator": 0.0}
    step_losses = []
    for epoch in range(1, epochs + 1):
        random.shuffle(transitions)
        model.train()
        total = 0.0
        completed_steps = 0
        completed_examples = 0
        validation_seconds = 0.0
        steps = math.ceil(len(transitions) / batch_size)
        started = time.perf_counter()
        progress = tqdm(range(0, len(transitions), batch_size), total=steps,
                        desc=f"{stage} {epoch}/{epochs}", unit="step", dynamic_ncols=True)
        for start in progress:
            batch = transitions[start:start + batch_size]
            histories, lengths, targets = history_batch(data, batch, max_history, device)
            slates = candidate_slates(targets, data.num_items, candidates)
            with amp_context(device):
                graph_items = model.graph_item_vectors()
                states = model.encode_states(histories, lengths, graph_items)
                if full_catalog_supervised and stage != "joint":
                    output = model.logits_from_states(states, model.project_all(graph_items))
                    labels = targets
                else:
                    output = model.logits_from_states(
                        states, model.project_ids(slates, graph_items)
                    )
                    labels = torch.zeros(len(batch), dtype=torch.long, device=device)
                reward_mean = 0.0
                if stage == "specialists":
                    loss = F.cross_entropy(output["long"], labels) + F.cross_entropy(output["short"], labels)
                elif stage == "coordinator":
                    loss = F.cross_entropy(output["coordinator"], labels)
                else:
                    distributions = {name: Categorical(logits=output[name])
                                     for name in ("long", "short", "coordinator")}
                    actions = {name: distribution.sample() for name, distribution in distributions.items()}
                    rewards = {name: torch.where(action == 0, torch.ones_like(action, dtype=torch.float),
                                                 torch.full_like(action, -0.05, dtype=torch.float))
                               for name, action in actions.items()}
                    policy_loss = 0.0
                    for name, distribution in distributions.items():
                        baselines[name] = 0.95 * baselines[name] + 0.05 * rewards[name].mean().item()
                        advantage = rewards[name] - baselines[name]
                        policy_loss -= (advantage.detach() * distribution.log_prob(actions[name])).mean()
                        policy_loss -= entropy_coef * distribution.entropy().mean()
                    long_state, short_state, _ = output["states"]
                    specialization = F.cosine_similarity(long_state, short_state).square().mean()
                    if full_catalog_supervised:
                        full_output = model.logits_from_states(
                            states, model.project_all(graph_items)
                        )
                        supervised = sum(
                            F.cross_entropy(full_output[name], targets)
                            for name in ("long", "short", "coordinator")
                        ) / 3
                    else:
                        supervised = sum(F.cross_entropy(output[name], labels)
                                         for name in ("long", "short", "coordinator")) / 3
                    loss = policy_loss + supervised_coef * supervised + specialization_coef * specialization
                    reward_mean = rewards["coordinator"].mean().item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            monitor.global_step += 1
            completed_steps += 1
            completed_examples += len(batch)
            total += loss.item()
            step_losses.append(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}", reward=f"{reward_mean:.3f}",
                step=monitor.global_step,
            )

            if validate_every_steps > 0 and monitor.global_step % validate_every_steps == 0:
                validation_started = time.perf_counter()
                valid_metrics, _ = evaluate(
                    model, data, "valid", eval_batch_size, max_history, device,
                    priors=catalog_priors, popularity_alpha=popularity_alpha,
                    transition_beta=transition_beta, user_limit=validation_user_limit,
                )
                log_metric_block(
                    logger, "PERIODIC VALIDATION", valid_metrics,
                    stage, epoch, monitor.global_step,
                )
                if periodic_test_user_limit >= 0:
                    test_metrics, _ = evaluate(
                        model, data, "test", eval_batch_size, max_history, device,
                        priors=catalog_priors, popularity_alpha=popularity_alpha,
                        transition_beta=transition_beta, user_limit=periodic_test_user_limit,
                    )
                    log_metric_block(
                        logger, "PERIODIC TEST", test_metrics,
                        stage, epoch, monitor.global_step,
                    )
                validation_seconds += time.perf_counter() - validation_started
                record_validation(model, valid_metrics, stage, epoch, monitor, logger)
                if scheduler is not None:
                    scheduler.step(valid_metrics[monitor.metric_name])
                    logger.info(
                        "LEARNING_RATE step=%d stage=%s lr=%.8g",
                        monitor.global_step, stage, optimizer.param_groups[0]["lr"],
                    )
                model.train()
                if (
                    stage == "joint"
                    and early_stopping_patience > 0
                    and monitor.checks_without_improvement >= early_stopping_patience
                ):
                    monitor.stopped_early = True
                    logger.info(
                        "EARLY_STOP step=%d checks_without_improvement=%d best_step=%d best_%s=%.6f",
                        monitor.global_step, monitor.checks_without_improvement,
                        monitor.best_step, monitor.metric_name, monitor.best_score,
                    )
                    break

        average = total / max(completed_steps, 1)
        training_seconds = max(
            time.perf_counter() - started - validation_seconds, 1e-9
        )
        logger.info(
            "stage=%s epoch=%d loss=%.6f steps=%d transitions/s=%.2f",
            stage, epoch, average, completed_steps,
            completed_examples / training_seconds,
        )
        if monitor.stopped_early:
            break
    return step_losses


@torch.inference_mode()
def evaluate(
    model, data, split, batch_size, max_history, device, sample_count=0,
    priors: CatalogPriors | None = None, popularity_alpha: float = 0.0,
    transition_beta: float = 0.0, user_limit: int = 0,
):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    total_users = len(users)
    if user_limit > 0 and total_users > user_limit:
        # Evenly cover the stable sorted user list without relying on global RNG state.
        users = [users[index * total_users // user_limit] for index in range(user_limit)]
    totals = {
        "recall@5": 0.0, "recall@10": 0.0,
        "ndcg@5": 0.0, "ndcg@10": 0.0,
        "hit@5": 0.0, "hit@10": 0.0,
    }
    samples = []
    with amp_context(device):
        graph_items = model.graph_item_vectors()
        projected_items = model.project_all(graph_items)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), batch_size), desc=f"full-catalog {split}", unit="batch"):
        batch_users = users[start:start + batch_size]
        histories, lengths, raw_histories = evaluation_history_batch(data, batch_users, split, max_history, device)
        with amp_context(device):
            output = model.full_catalog_scores(
                histories, lengths, projected_items, graph_items
            )
        scores = output["coordinator"].float()
        gold = torch.tensor([targets[user] for user in batch_users], device=device)
        if priors is not None:
            scores += popularity_alpha * priors.popularity.unsqueeze(0)
        for row, history in enumerate(raw_histories):
            if priors is not None and transition_beta != 0.0:
                transition_row = priors.transitions.get(history[-1], {})
                if transition_row:
                    item_ids = list(transition_row)
                    values = torch.tensor(
                        list(transition_row.values()), device=device, dtype=scores.dtype
                    )
                    scores[row, item_ids] += transition_beta * values
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, gold.unsqueeze(1))).sum(1)
        for cutoff in (5, 10):
            hits = (ranks <= cutoff).sum().item()
            # There is one held-out target per user, so Recall and Hit are
            # numerically equal. Keep both names for standard reports.
            totals[f"recall@{cutoff}"] += hits
            totals[f"hit@{cutoff}"] += hits
            totals[f"ndcg@{cutoff}"] += torch.where(
                ranks <= cutoff, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)
            ).sum().item()
        remaining = max(sample_count - len(samples), 0)
        if remaining:
            top = torch.topk(scores, k=min(10, data.num_items), dim=1).indices
            for row in range(min(remaining, len(batch_users))):
                samples.append({
                    "user_index": batch_users[row], "history": raw_histories[row], "target": int(gold[row]),
                    "top_items": top[row].tolist(),
                    "agent_weights": {"long": round(float(output["weights"][row, 0]), 6),
                                      "short": round(float(output["weights"][row, 1]), 6)},
                })
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update({"users_per_second": len(users) / seconds,
                    "scores_per_second": len(users) * data.num_items / seconds,
                    "evaluated_users": len(users), "total_users": total_users})
    return metrics, samples


def generate_reasons(samples, data, cache_dir, device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, cache_dir=cache_dir)
    generator = AutoModelForCausalLM.from_pretrained(
        MAMBA_MODEL_ID, cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    for sample in tqdm(samples, desc="Generating recommendation reasons", unit="user"):
        history = " | ".join(data.item_texts[item] for item in sample["history"][-10:])
        item = sample["top_items"][0]
        prompt = (
            "Explain this recommendation in one concise sentence using only the supplied history. "
            f"Long-term agent weight={sample['agent_weights']['long']}; "
            f"short-term agent weight={sample['agent_weights']['short']}. "
            f"History: {history}. Recommended item: {data.item_texts[item]}. Reason:"
        )
        encoded = tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = generator.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
        reason = tokenizer.decode(output[0, encoded["input_ids"].size(1):], skip_special_tokens=True).strip()
        sample["recommendation"] = {
            "item_index": item, "item_text": data.item_texts[item],
            "reason": reason or "The item matches the user's recent and long-term interaction patterns.",
        }


def save_loss_curve(losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    offset = 0
    for stage, values in losses.items():
        xs = list(range(offset + 1, offset + len(values) + 1))
        axis.plot(xs, values, marker="o", markersize=2, linewidth=1, label=stage)
        offset += len(values)
    axis.set(xlabel="Optimizer step", ylabel="Loss", title="Multi-agent Mamba-RL training loss per step")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Three-agent LoRA selective-Mamba recommender")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--cache-dir", default=str(SHARED_CACHE_ROOT))
    parser.add_argument("--refresh-data-cache", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=500_000)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-events", type=int, default=5)
    parser.add_argument("--specialist-epochs", type=int, default=3)
    parser.add_argument("--coordinator-epochs", type=int, default=2)
    parser.add_argument("--rl-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--validate-every-steps", type=int, default=1000)
    parser.add_argument("--monitor-metric", choices=("ndcg@10", "recall@10"), default="ndcg@10")
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument(
        "--use-graph-embeddings", action=argparse.BooleanOptionalAction, default=True,
        help="Fuse trainable LightGCN item embeddings with Mamba item vectors.",
    )
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--specialist-lr", type=float, default=2e-4)
    parser.add_argument("--coordinator-lr", type=float, default=2e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--supervised-coef", type=float, default=0.1)
    parser.add_argument("--specialization-coef", type=float, default=0.01)
    parser.add_argument("--full-catalog-supervised", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--popularity-alpha", type=float, default=0.0)
    parser.add_argument("--transition-beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-mamba", action="store_true")
    parser.add_argument(
        "--graph-device", default=None,
        help="Optional device for LightGCN parameters/edges; use cuda:1 for two-GPU model parallelism.",
    )
    parser.add_argument("--mamba-encode-batch-size", type=int, default=4)
    parser.add_argument("--mamba-max-tokens", type=int, default=48)
    parser.add_argument("--validation-user-limit", type=int, default=0)
    parser.add_argument(
        "--periodic-test-user-limit", type=int, default=-1,
        help="Users in each periodic test; 0 means all and -1 disables periodic test.",
    )
    parser.add_argument("--generate-reasons", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-model-weights", action=argparse.BooleanOptionalAction, default=True,
        help="Persist checkpoints/model weights; loss.png is always written.",
    )
    parser.add_argument("--reason-count", type=int, default=20)
    parser.add_argument("--reason-max-new-tokens", type=int, default=40)
    parser.add_argument("--item-vector-artifact", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--experiment-note", default="No experiment note supplied.")
    parser.add_argument("--target-recall-at-10", type=float, default=0.15)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_mamba_rl"))
    parser.add_argument(
        "--output-run-dir", default=None,
        help="Write directly to this directory instead of dataset/run_id nesting.",
    )
    parser.add_argument(
        "--score-file", default=None,
        help="Optional JSON path for validation/test ranking scores.",
    )
    args = parser.parse_args()
    if min(args.specialist_epochs, args.coordinator_epochs, args.rl_epochs) < 0:
        parser.error("stage epoch counts cannot be negative")
    if min(args.validate_every_steps, args.early_stopping_patience, args.lr_patience) < 0:
        parser.error("validation interval and patience values cannot be negative")
    if (
        args.candidates < 2 or args.batch_size < 1 or args.max_history < 1
        or args.mamba_encode_batch_size < 1 or args.mamba_max_tokens < 1
        or args.validation_user_limit < 0 or args.periodic_test_user_limit < -1
    ):
        parser.error("candidate count must be >=2 and batch/history sizes must be positive")
    if not 0.0 <= args.target_recall_at_10 <= 1.0:
        parser.error("--target-recall-at-10 must be in [0, 1]")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    output = Path(args.output_run_dir) if args.output_run_dir else Path(args.output_dir) / safe_dataset / run_id
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output / f"train_{run_id}.log")
    logger.info("run_id=%s dataset=%s device=%s", run_id, args.dataset, args.device)
    logger.info("EXPERIMENT_NOTE %s", args.experiment_note)
    logger.info("EXPERIMENT_CONFIG %s", json.dumps(vars(args), sort_keys=True))
    data, interaction_artifact = load_recommendation_data_cached(args, logger)
    logger.info("users=%d items=%d train_interactions=%d",
                data.num_users, data.num_items, sum(map(len, data.train_by_user.values())))
    catalog_priors = (
        build_catalog_priors(data, args.device)
        if args.popularity_alpha != 0.0 or args.transition_beta != 0.0
        else None
    )
    logger.info(
        "CATALOG_PRIORS popularity_alpha=%.6f transition_beta=%.6f "
        "source=train_histories_only enabled=%s",
        args.popularity_alpha, args.transition_beta, catalog_priors is not None,
    )

    fingerprint = hashlib.sha1("\n".join(data.item_texts).encode("utf-8")).hexdigest()[:12]
    artifact = (
        Path(args.item_vector_artifact)
        if args.item_vector_artifact
        else Path(args.cache_dir) / "mamba_multi_agent" / (
            f"{safe_dataset}_{data.num_items}_{fingerprint}_tok{args.mamba_max_tokens}.pt"
        )
    )
    if args.skip_mamba:
        generator = torch.Generator().manual_seed(args.seed)
        item_features = torch.randn(data.num_items, args.dim, generator=generator)
        logger.warning("--skip-mamba uses random item features; it is only a smoke-test mode")
    else:
        item_features = load_or_encode_text(
            data.item_texts, str(artifact), args.device, False, args.cache_dir,
            batch_size=args.mamba_encode_batch_size, max_tokens=args.mamba_max_tokens,
        )
        assert item_features is not None
    if item_features.size(0) != data.num_items:
        raise ValueError(
            f"Item-vector rows ({item_features.size(0)}) do not match catalog size ({data.num_items}); "
            "the explicit artifact is incompatible with this data split."
        )
    logger.info("ITEM_VECTOR_ARTIFACT path=%s fingerprint=%s", artifact, fingerprint)
    item_features = item_features.to(dtype=torch.float16 if args.device.startswith("cuda") else torch.float32)
    graph_edges = edge_index(data) if args.use_graph_embeddings else None
    logger.info(
        "GRAPH_EMBEDDINGS enabled=%s source=train_histories_only main_device=%s graph_device=%s",
        args.use_graph_embeddings, args.device,
        args.graph_device or args.device,
    )
    model = MultiAgentMambaRecommender(
        item_features, args.dim, args.lora_rank, args.lora_alpha, args.lora_dropout, args.short_window,
        graph_edges=graph_edges, graph_users=data.num_users,
        use_graph_embeddings=args.use_graph_embeddings,
    )
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        resume_state = resume_payload.get("model", resume_payload)
        model.load_state_dict(resume_state)
        logger.info("RESUME_CHECKPOINT path=%s", resume_path)
    model.place_devices(args.device, args.graph_device)
    logger.info("agent_parameters=%s", model.agent_parameter_counts())
    transitions = build_transitions(data, args.max_transitions, args.seed)
    logger.info("training_transitions=%d", len(transitions))
    logger.info(
        "schedule specialist_epochs=%d coordinator_epochs=%d rl_epochs=%d "
        "validate_every_steps=%d validation_users=%d periodic_test_users=%d "
        "monitor=%s early_stopping_patience=%d",
        args.specialist_epochs, args.coordinator_epochs, args.rl_epochs,
        args.validate_every_steps, args.validation_user_limit,
        args.periodic_test_user_limit, args.monitor_metric, args.early_stopping_patience,
    )
    monitor = TrainingMonitor(
        metric_name=args.monitor_metric,
        checkpoint_path=(output / "best_validation.pt") if args.save_model_weights else None,
    )
    if args.resume_checkpoint:
        resumed_metrics, _ = evaluate(
            model, data, "valid", args.eval_batch_size, args.max_history, args.device,
            priors=catalog_priors, popularity_alpha=args.popularity_alpha,
            transition_beta=args.transition_beta, user_limit=args.validation_user_limit,
        )
        log_metric_block(logger, "RESUME VALIDATION", resumed_metrics, "resume", 0, monitor.global_step)
        record_validation(model, resumed_metrics, "resume", 0, monitor, logger)
    common = dict(
        model=model, data=data, transitions=transitions, batch_size=args.batch_size,
        candidates=args.candidates, max_history=args.max_history, entropy_coef=args.entropy_coef,
        supervised_coef=args.supervised_coef, specialization_coef=args.specialization_coef,
        device=args.device, logger=logger, monitor=monitor,
        validate_every_steps=args.validate_every_steps, eval_batch_size=args.eval_batch_size,
        early_stopping_patience=args.early_stopping_patience, lr_patience=args.lr_patience,
        full_catalog_supervised=args.full_catalog_supervised,
        catalog_priors=catalog_priors, popularity_alpha=args.popularity_alpha,
        transition_beta=args.transition_beta,
        validation_user_limit=args.validation_user_limit,
        periodic_test_user_limit=args.periodic_test_user_limit,
    )
    losses = {
        "specialists": train_stage(stage="specialists", epochs=args.specialist_epochs,
                                   learning_rate=args.specialist_lr, **common),
        "coordinator": train_stage(stage="coordinator", epochs=args.coordinator_epochs,
                                   learning_rate=args.coordinator_lr, **common),
        "joint": train_stage(stage="joint", epochs=args.rl_epochs,
                             learning_rate=args.joint_lr, **common),
    }

    final_current_metrics, _ = evaluate(
        model, data, "valid", args.eval_batch_size, args.max_history, args.device,
        priors=catalog_priors, popularity_alpha=args.popularity_alpha,
        transition_beta=args.transition_beta, user_limit=args.validation_user_limit,
    )
    log_metric_block(
        logger, "FINAL CURRENT VALIDATION", final_current_metrics,
        "final", 0, monitor.global_step,
    )
    record_validation(model, final_current_metrics, "final", 0, monitor, logger)
    if monitor.best_state is None:
        raise RuntimeError("Training finished without producing a validation checkpoint.")
    model.load_state_dict(monitor.best_state)
    logger.info(
        "RESTORE_BEST step=%d stage=%s %s=%.6f",
        monitor.best_step, monitor.best_stage, monitor.metric_name, monitor.best_score,
    )
    valid_metrics, _ = evaluate(
        model, data, "valid", args.eval_batch_size, args.max_history, args.device,
        priors=catalog_priors, popularity_alpha=args.popularity_alpha,
        transition_beta=args.transition_beta, user_limit=args.validation_user_limit,
    )
    test_metrics, samples = evaluate(
        model, data, "test", args.eval_batch_size, args.max_history, args.device, args.reason_count,
        priors=catalog_priors, popularity_alpha=args.popularity_alpha,
        transition_beta=args.transition_beta,
    )
    log_metric_block(
        logger, "BEST CHECKPOINT VALIDATION", valid_metrics,
        monitor.best_stage, 0, monitor.best_step,
    )
    log_metric_block(
        logger, "FINAL FULL TEST", test_metrics,
        monitor.best_stage, 0, monitor.best_step,
    )
    logger.info("VALID_BEST %s", json.dumps(valid_metrics, sort_keys=True))
    logger.info("TEST_FINAL %s", json.dumps(test_metrics, sort_keys=True))
    target_achieved = test_metrics["recall@10"] > args.target_recall_at_10
    logger.info(
        "TARGET_RESULT metric=recall@10 target=>%.6f actual=%.6f achieved=%s",
        args.target_recall_at_10, test_metrics["recall@10"], target_achieved,
    )
    training_summary = {
        "monitor_metric": monitor.metric_name,
        "best_score": monitor.best_score,
        "best_step": monitor.best_step,
        "best_stage": monitor.best_stage,
        "global_steps": monitor.global_step,
        "stopped_early": monitor.stopped_early,
        "target_recall_at_10": args.target_recall_at_10,
        "target_achieved": target_achieved,
        "experiment_note": args.experiment_note,
        "catalog_prior": {
            "popularity_alpha": args.popularity_alpha,
            "transition_beta": args.transition_beta,
            "source": "train_histories_only",
        },
        "validation_history": monitor.history,
    }
    if args.save_model_weights:
        checkpoint = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": vars(args), "valid_metrics": valid_metrics, "test_metrics": test_metrics,
            "training": training_summary, "item_vector_artifact": str(artifact),
            "interaction_artifact": str(interaction_artifact),
        }
        torch.save(checkpoint, output / "multi_agent_lora.pt")
    save_loss_curve(losses, output / "loss.png")
    if args.generate_reasons and samples:
        model.to("cpu")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        generate_reasons(samples, data, args.cache_dir, args.device, args.reason_max_new_tokens)
    else:
        for sample in samples:
            item = sample["top_items"][0]
            sample["recommendation"] = {"item_index": item, "item_text": data.item_texts[item], "reason": None}
    (output / "recommendations.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(
        json.dumps({"valid": valid_metrics, "test": test_metrics, "training": training_summary}, indent=2), encoding="utf-8"
    )
    if args.score_file:
        requested_names = ("ndcg@5", "ndcg@10", "recall@5", "recall@10", "hit@5", "hit@10")
        score_path = Path(args.score_file)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "run_id": run_id,
                    "valid": {name: valid_metrics[name] for name in requested_names},
                    "test": {name: test_metrics[name] for name in requested_names},
                    "config": vars(args),
                    "artifacts": str(output),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("scores=%s", score_path)
    logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()
