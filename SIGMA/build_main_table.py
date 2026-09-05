#!/usr/bin/env python3
"""Build audited data-statistics and main-result tables from completed runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("all_beauty", "baby_products", "sports_and_outdoors", "toys_and_games")


def latest_metrics(dataset: str) -> dict | None:
    paths = sorted((ROOT / "outputs_ver4_protocol" / dataset).glob("*/metrics.json"))
    return json.loads(paths[-1].read_text()) if paths else None


def main() -> None:
    manifests = {
        name: json.loads((ROOT / "data" / "ver4_protocol" / f"{name}.json").read_text())
        for name in DATASETS
    }
    with (ROOT / "dataset_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dataset",
                "num_users",
                "evaluation_users",
                "num_items",
                "train_interactions",
                "split_sha256",
                "completed_ver4_counts_match",
            ),
        )
        writer.writeheader()
        for name, manifest in manifests.items():
            reference = manifest["completed_ver4_reference"]
            writer.writerow(
                {
                    "dataset": name,
                    **manifest["statistics"],
                    "split_sha256": manifest["split_sha256"],
                    "completed_ver4_counts_match": (
                        reference == manifest["statistics"] if reference is not None else "N/A"
                    ),
                }
            )

    fields = (
        "dataset",
        "method",
        "seed",
        "best_epoch",
        "recall@5",
        "recall@10",
        "ndcg@5",
        "ndcg@10",
        "mrr@10",
        "evaluation_users",
        "split_sha256",
        "metrics_path",
    )
    with (ROOT / "main_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in DATASETS:
            result = latest_metrics(name)
            if result is None:
                continue
            test = result["test"]
            path = sorted(
                (ROOT / "outputs_ver4_protocol" / name).glob("*/metrics.json")
            )[-1]
            writer.writerow(
                {
                    "dataset": name,
                    "method": "SIGMA",
                    "seed": result["config"]["seed"],
                    "best_epoch": result["training"]["best_epoch"],
                    "recall@5": test["recall@5"],
                    "recall@10": test["recall@10"],
                    "ndcg@5": test["ndcg@5"],
                    "ndcg@10": test["ndcg@10"],
                    "mrr@10": test["mrr@10"],
                    "evaluation_users": test["evaluated_users"],
                    "split_sha256": result["data"]["split_sha256"],
                    "metrics_path": path.relative_to(ROOT),
                }
            )


if __name__ == "__main__":
    main()
