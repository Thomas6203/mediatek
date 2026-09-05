#!/usr/bin/env python3
"""Validation-only grid search for the existing catalog calibration weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data_mamba_rl import load_recommendation_data
from src.model import load_or_encode_text
from src.model_mamba_rl import MultiAgentMambaRecommender
from src.train_mamba_rl import build_catalog_priors, evaluate, seed_everything


def floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--item-vector-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", default="/workspace/P78123011/cache")
    parser.add_argument("--alphas", type=floats, default=floats("-2,-1,-0.5,-0.25,0,0.25,0.5"))
    parser.add_argument("--betas", type=floats, default=floats("0,0.25,0.5,1,2,3,4"))
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    data = load_recommendation_data(
        args.dataset, args.data_path, args.cache_dir, min_rating=4.0, min_user_events=5
    )
    item_features = load_or_encode_text(
        data.item_texts, str(args.item_vector_artifact), args.device, False, args.cache_dir
    ).to(dtype=torch.float16 if args.device.startswith("cuda") else torch.float32)
    model = MultiAgentMambaRecommender(
        item_features, args.dim, args.lora_rank, args.lora_alpha,
        args.lora_dropout, args.short_window,
    ).to(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    priors = build_catalog_priors(data, args.device)

    results = []
    for alpha in args.alphas:
        for beta in args.betas:
            metrics, _ = evaluate(
                model, data, "valid", args.eval_batch_size, args.max_history,
                args.device, priors=priors, popularity_alpha=alpha,
                transition_beta=beta,
            )
            row = {"popularity_alpha": alpha, "transition_beta": beta, **metrics}
            results.append(row)
            print(json.dumps(row, sort_keys=True))
    results.sort(key=lambda row: (row["recall@10"], row["ndcg@10"]), reverse=True)
    report = {
        "selection_split": "valid",
        "test_metrics_used": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "item_vector_artifact": str(args.item_vector_artifact.resolve()),
        "best": results[0],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BEST", json.dumps(results[0], sort_keys=True))


if __name__ == "__main__":
    main()
