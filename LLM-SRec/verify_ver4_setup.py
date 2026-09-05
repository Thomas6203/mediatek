#!/usr/bin/env python3
"""Verify LLM-SRec's prepared ver4 data without importing model code."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from prepare_ver4_data import sha256_file
from ver4_protocol import DATASETS, DEFAULT_SOURCE_ROOT, ROOT, load_verified_data


def count_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def verify_dataset(dataset: str, source_root: Path) -> dict:
    data, source_manifest = load_verified_data(dataset, source_root)
    output = ROOT / "SeqRec" / f"data_{dataset}"
    manifest_path = output / "ver4_manifest.json"
    report = {
        "dataset": dataset,
        "canonical_statistics": source_manifest["statistics"],
        "canonical_split_sha256": source_manifest["split_sha256"],
        "prepared": manifest_path.exists(),
        "data_valid": False,
        "metadata_valid": False,
        "teacher_checkpoint_count": len(list((ROOT / "SeqRec" / "sasrec" / dataset).glob("*.pth"))),
        "errors": [],
    }
    if not manifest_path.exists():
        report["errors"].append(f"missing {manifest_path}")
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("statistics") != DATASETS[dataset]["expected"]:
        report["errors"].append("prepared manifest statistics mismatch")
    if manifest.get("split_sha256") != source_manifest["split_sha256"]:
        report["errors"].append("prepared manifest split fingerprint mismatch")

    expected_rows = {
        "train": source_manifest["statistics"]["train_interactions"],
        "valid": source_manifest["statistics"]["evaluation_users"],
        "test": source_manifest["statistics"]["evaluation_users"],
    }
    actual_rows = {}
    for split, expected in expected_rows.items():
        path = output / f"{dataset}_{split}.txt"
        if not path.exists():
            report["errors"].append(f"missing {path}")
            continue
        actual_rows[split] = count_rows(path)
        if actual_rows[split] != expected:
            report["errors"].append(
                f"{split} row mismatch: expected={expected}, actual={actual_rows[split]}"
            )
        expected_hash = manifest.get("split_file_sha256", {}).get(split)
        if expected_hash != sha256_file(path):
            report["errors"].append(f"{split} SHA-256 mismatch")
    report["split_rows"] = actual_rows
    report["data_valid"] = not report["errors"]

    metadata_path = output / "text_name_dict.json.gz"
    if not metadata_path.exists():
        report["errors"].append(f"missing {metadata_path}")
        return report
    if manifest.get("text_metadata_sha256") != sha256_file(metadata_path):
        report["errors"].append("text metadata SHA-256 mismatch")
        return report
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    title_count = len(metadata.get("title", {}))
    description_count = len(metadata.get("description", {}))
    report["metadata"] = {
        "titles": title_count,
        "descriptions": description_count,
        "placeholders_only": bool(manifest.get("placeholders_only")),
        "amazon_items_matched": manifest.get("metadata_items_matched"),
    }
    if title_count != data.num_items or description_count != data.num_items:
        report["errors"].append("metadata item count mismatch")
    if manifest.get("placeholders_only"):
        report["errors"].append("placeholder metadata cannot be used for experiments")
    report["metadata_valid"] = not any(
        "metadata" in error or "placeholder" in error for error in report["errors"]
    )
    report["data_valid"] = not any(
        "row" in error or "split" in error or "missing" in error for error in report["errors"]
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-teachers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [verify_dataset(dataset, args.source_root) for dataset in DATASETS]
    data_ready = all(report["data_valid"] and report["metadata_valid"] for report in reports)
    teachers_ready = all(report["teacher_checkpoint_count"] == 1 for report in reports)
    ready = data_ready and (teachers_ready or not args.require_teachers)
    result = {
        "data_ready_for_all_datasets": data_ready,
        "sasrec_teachers_ready_for_all_datasets": teachers_ready,
        "requested_checks_passed": ready,
        "note": "A Hugging Face token with Llama-3.2-3B-Instruct access is checked only when the LLM runner starts.",
        "datasets": reports,
    }
    print(json.dumps(result, indent=2))
    if not ready and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
