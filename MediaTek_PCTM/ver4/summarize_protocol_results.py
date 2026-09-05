#!/usr/bin/env python3
"""Aggregate completed matched-protocol runs and reject mixed configurations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.data_contract import assert_same_data_contract


PCTM_NDCG_AT_10 = {
    "amazon-beauty-pctm": 0.0635,
    "amazon-sports-pctm": 0.0368,
    "amazon-toys-pctm": 0.0738,
    "movielens-1m-pctm": 0.1815,
}
METRICS = ("ndcg@10", "recall@10")
MUTABLE_CONFIG = {
    "cache_dir",
    "checkpoint_every_minutes",
    "data_path",
    "experiment_note",
    "generate_reasons",
    "item_vector_artifact",
    "output_dir",
    "output_run_dir",
    "reason_count",
    "reason_max_new_tokens",
    "resume_checkpoint",
    "run_checkpoint_path",
    "run_hours",
    "score_file",
    "seed",
}


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_completed_run(metrics_path: Path) -> dict[str, object] | None:
    status_path = metrics_path.with_name("run_status.json")
    if not status_path.is_file():
        print(f"SKIP missing status: {metrics_path}", file=sys.stderr)
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        print(f"SKIP non-completed run: {metrics_path.parent}", file=sys.stderr)
        return None

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    contract = payload.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Missing data contract in {metrics_path}")
    dataset = str(contract.get("dataset", ""))
    if dataset not in PCTM_NDCG_AT_10:
        print(f"SKIP non-official-four dataset {dataset!r}: {metrics_path}", file=sys.stderr)
        return None
    test = payload.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing test metrics in {metrics_path}")

    score_paths = sorted(metrics_path.parent.glob("*_scores.json"))
    if len(score_paths) != 1:
        raise ValueError(
            f"Expected exactly one *_scores.json beside {metrics_path}, found {len(score_paths)}"
        )
    score = json.loads(score_paths[0].read_text(encoding="utf-8"))
    config = score.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing config in {score_paths[0]}")
    fixed_config = {
        key: value for key, value in config.items() if key not in MUTABLE_CONFIG
    }
    return {
        "dataset": dataset,
        "metrics_path": metrics_path,
        "contract": contract,
        "test": {name: float(test[name]) for name in METRICS},
        "config_sha256": canonical_hash(fixed_config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root")
    parser.add_argument("--output", help="Optional CSV output path")
    parser.add_argument(
        "--require-runs", type=int, default=1,
        help="Require at least this many completed seeds per dataset (paper default: 5).",
    )
    args = parser.parse_args()
    if args.require_runs < 1:
        parser.error("--require-runs must be positive")

    root = Path(args.results_root).expanduser().resolve()
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for metrics_path in sorted(root.rglob("metrics.json")):
        run = load_completed_run(metrics_path)
        if run is not None:
            groups[str(run["dataset"])].append(run)
    if not groups:
        raise RuntimeError(f"No completed official-four metrics found under {root}")

    rows: list[dict[str, object]] = []
    for dataset in sorted(groups):
        runs = groups[dataset]
        if len(runs) < args.require_runs:
            raise RuntimeError(
                f"{dataset} has {len(runs)} completed runs; required {args.require_runs}"
            )
        reference = runs[0]
        config_hashes = {str(run["config_sha256"]) for run in runs}
        if len(config_hashes) != 1:
            raise RuntimeError(
                f"{dataset} contains mixed training configurations: {sorted(config_hashes)}"
            )
        for run in runs[1:]:
            assert_same_data_contract(
                reference["contract"], run["contract"],
                context=f"aggregating {dataset}",
            )

        values = {
            metric: [float(run["test"][metric]) for run in runs]
            for metric in METRICS
        }
        ndcg_mean = statistics.mean(values["ndcg@10"])
        row = {
            "dataset": dataset,
            "runs": len(runs),
            "evaluation_contract_sha256": reference["contract"][
                "evaluation_contract_sha256"
            ],
            "config_sha256": reference["config_sha256"],
            "ndcg@10_mean": ndcg_mean,
            "ndcg@10_std": statistics.stdev(values["ndcg@10"]) if len(runs) > 1 else 0.0,
            "recall@10_mean": statistics.mean(values["recall@10"]),
            "recall@10_std": statistics.stdev(values["recall@10"]) if len(runs) > 1 else 0.0,
            "pctm_ndcg@10": PCTM_NDCG_AT_10[dataset],
            "ndcg@10_delta_vs_pctm": ndcg_mean - PCTM_NDCG_AT_10[dataset],
        }
        rows.append(row)

    fieldnames = list(rows[0])
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
