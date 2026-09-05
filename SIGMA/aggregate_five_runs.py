#!/usr/bin/env python3
"""彙整四個資料集各五次 independent runs 的結果。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
DATASETS = ("all_beauty", "baby_products", "sports_and_outdoors", "toys_and_games")
METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs_ver4_protocol_5runs")
    )
    parser.add_argument("--base-seed", type=int, default=25252)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    expected_seeds = tuple(range(args.base_seed, args.base_seed + args.repeats))
    individual_rows: list[dict] = []

    for dataset in DATASETS:
        manifest = json.loads(
            (ROOT / "data" / "ver4_protocol" / f"{dataset}.json").read_text()
        )
        by_seed: dict[int, tuple[Path, dict]] = {}
        for path in sorted((output_dir / dataset).glob("*/metrics.json")):
            result = json.loads(path.read_text())
            seed = int(result["config"]["seed"])
            if seed not in expected_seeds:
                continue
            if result["config"]["epochs"] != 18:
                continue
            if result["data"]["statistics"] != manifest["statistics"]:
                raise RuntimeError(f"{dataset}: data statistics mismatch in {path}")
            checkpoint = path.with_name("best_validation.pt")
            if not checkpoint.exists():
                raise RuntimeError(f"{dataset}: missing checkpoint {checkpoint}")
            by_seed[seed] = (path, result)

        missing = sorted(set(expected_seeds) - set(by_seed))
        if missing:
            raise RuntimeError(f"{dataset}: missing completed seeds {missing}")

        for seed in expected_seeds:
            path, result = by_seed[seed]
            row = {
                "dataset": dataset,
                "seed": seed,
                "best_epoch": result["training"]["best_epoch"],
                "stopped_early": result["training"]["stopped_early"],
                **{metric: result["test"][metric] for metric in METRICS},
                "evaluation_users": result["test"]["evaluated_users"],
                "split_sha256": result["data"]["split_sha256"],
                "run_directory": str(path.parent.resolve()),
            }
            individual_rows.append(row)

    individual_path = output_dir / "five_run_results.csv"
    with individual_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(individual_rows[0]))
        writer.writeheader()
        writer.writerows(individual_rows)

    summary_rows = []
    for dataset in DATASETS:
        rows = [row for row in individual_rows if row["dataset"] == dataset]
        summary = {"dataset": dataset, "runs": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
        summary_rows.append(summary)

    summary_path = output_dir / "five_run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(individual_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
