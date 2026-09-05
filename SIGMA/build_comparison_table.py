#!/usr/bin/env python3
"""Build the comparable ver4/SIGMA main table from verified completed runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("all_beauty", "baby_products", "sports_and_outdoors", "toys_and_games")
VER4_RUNS = {
    "all_beauty": ROOT.parent / "MediaTek_2026_0902/ver4/outputs_mamba_rl/amazons_lora/Full_Beauty/rl_20260830_175202/metrics.json",
    "baby_products": ROOT.parent / "MediaTek_2026_0902/ver4/outputs_mamba_rl/amazons_lora/Baby_Products/rl_20260831_023533/metrics.json",
    "sports_and_outdoors": ROOT.parent / "MediaTek_2026_0902/ver4/outputs_mamba_rl/amazons_lora/Sports_and_Outdoors/rl_20260831_100146/metrics.json",
}


def latest_sigma(dataset: str) -> tuple[Path, dict] | None:
    paths = sorted((ROOT / "outputs_ver4_protocol" / dataset).glob("*/metrics.json"))
    return (paths[-1], json.loads(paths[-1].read_text())) if paths else None


def result_row(
    dataset: str,
    method: str,
    result: dict,
    manifest: dict,
    metrics_path: Path,
) -> dict:
    test = result["test"]
    training = result["training"]
    return {
        "dataset": dataset,
        "method": method,
        "seed": result.get("config", {}).get("seed", 25252),
        "best_checkpoint": training.get("best_epoch", training.get("best_stage", "")),
        "recall@5": test["recall@5"],
        "recall@10": test["recall@10"],
        "ndcg@5": test["ndcg@5"],
        "ndcg@10": test["ndcg@10"],
        "mrr@10": test.get("mrr@10", ""),
        "evaluation_users": test["evaluated_users"],
        "split_sha256": manifest["split_sha256"],
        "metrics_path": metrics_path.relative_to(ROOT.parent),
    }


def main() -> None:
    fields = (
        "dataset",
        "method",
        "seed",
        "best_checkpoint",
        "recall@5",
        "recall@10",
        "ndcg@5",
        "ndcg@10",
        "mrr@10",
        "evaluation_users",
        "split_sha256",
        "metrics_path",
    )
    rows = []
    for dataset in DATASETS:
        manifest = json.loads(
            (ROOT / "data" / "ver4_protocol" / f"{dataset}.json").read_text()
        )
        ver4_path = VER4_RUNS.get(dataset)
        if ver4_path is not None:
            ver4 = json.loads(ver4_path.read_text())
            if ver4["test"]["evaluated_users"] != manifest["statistics"]["evaluation_users"]:
                raise RuntimeError(f"{dataset}: refusing mismatched ver4 main-table row")
            rows.append(result_row(dataset, "MediaTek_ver4_Multi-LoRA", ver4, manifest, ver4_path))
        sigma_pair = latest_sigma(dataset)
        if sigma_pair is not None:
            sigma_path, sigma = sigma_pair
            if sigma["data"]["statistics"] != manifest["statistics"]:
                raise RuntimeError(f"{dataset}: refusing mismatched SIGMA main-table row")
            rows.append(result_row(dataset, "SIGMA", sigma, manifest, sigma_path))

    with (ROOT / "main_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
