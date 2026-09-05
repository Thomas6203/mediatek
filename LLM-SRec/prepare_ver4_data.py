#!/usr/bin/env python3
"""Export canonical MediaTek ver4 splits in LLM-SRec's legacy file format."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from ver4_protocol import (
    DATASETS,
    DEFAULT_SOURCE_ROOT,
    ROOT,
    load_verified_data,
    protocol_manifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sequence(data, user: int) -> list[int]:
    sequence = list(data.train_by_user[user])
    if user in data.valid_target:
        sequence.extend((data.valid_target[user], data.test_target[user]))
    return sequence


def export_split_files(dataset: str, data, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    split_rows = {
        "train": ((user, item) for user, items in data.train_by_user.items() for item in items),
        "valid": data.valid_target.items(),
        "test": data.test_target.items(),
    }
    hashes = {}
    for split, rows in split_rows.items():
        path = output / f"{dataset}_{split}.txt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for user, item in rows:
                # LLM-SRec reserves zero for padding.
                stream.write(f"{user + 1} {item + 1}\n")
        temporary.replace(path)
        hashes[split] = sha256_file(path)
    return hashes


def _description(row: dict) -> str:
    value = row.get("description")
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).strip() if value else "Empty description"


def build_item_text(data, config: dict, cache_dir: Path, placeholders_only: bool):
    item_index = {asin: index + 1 for index, asin in enumerate(data.item_texts)}
    titles = {index: asin for asin, index in item_index.items()}
    descriptions = {index: "Empty description" for index in item_index.values()}
    if placeholders_only:
        return item_index, titles, descriptions, 0

    from datasets import load_dataset  # Imported only when metadata is requested.

    metadata = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config["metadata_config"],
        split="full",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    matched = 0
    for row in tqdm(metadata, total=len(metadata), desc="Joining Amazon metadata", unit="item"):
        asin = str(row.get("parent_asin") or "")
        index = item_index.get(asin)
        if index is None:
            continue
        title = str(row.get("title") or "").strip()
        if title:
            titles[index] = title
        descriptions[index] = _description(row)
        matched += 1
    return item_index, titles, descriptions, matched


def reconstruct_times(data, config: dict, item_index: dict[str, int], cache_dir: Path):
    """Recover canonical user IDs and interaction timestamps from raw reviews."""
    from datasets import load_dataset

    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config["amazon_config"],
        split="full",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    raw_user_counts: dict[str, int] = defaultdict(int)
    retained: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in tqdm(reviews, total=len(reviews), desc="Recovering canonical timestamps", unit="review"):
        user = str(row.get("user_id") or row.get("user") or "")
        asin = str(row.get("parent_asin") or row.get("asin") or row.get("item_id") or "")
        if not user or not asin:
            continue
        raw_user_counts[user] += 1
        item = item_index.get(asin)
        if item is not None:
            retained[user].append((int(row.get("timestamp") or 0), item))

    users = sorted(
        user for user, entries in retained.items() if raw_user_counts[user] >= 5 and entries
    )
    if len(users) != data.num_users:
        raise RuntimeError(
            f"Canonical user reconstruction mismatch: expected={data.num_users}, actual={len(users)}"
        )

    times: dict[int, dict[int, int]] = defaultdict(dict)
    pairs = 0
    for zero_user, user in enumerate(tqdm(users, desc="Verifying reconstructed histories", unit="user")):
        entries = sorted(retained[user], key=lambda entry: entry[0])  # Python sort is stable.
        actual = [item - 1 for _, item in entries]
        expected = canonical_sequence(data, zero_user)
        if actual != expected:
            raise RuntimeError(
                f"Canonical sequence mismatch for zero-based user {zero_user}: "
                f"expected_length={len(expected)}, actual_length={len(actual)}"
            )
        one_user = zero_user + 1
        for timestamp, one_item in entries:
            # LLM-SRec stores one timestamp per (user, item) pair.
            times[one_item][one_user] = timestamp
            pairs += 1
    return times, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "cache")
    parser.add_argument(
        "--placeholders-only",
        action="store_true",
        help="Use ASIN titles and zero timestamps; intended only for offline structural tests.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_one(dataset: str, args: argparse.Namespace) -> None:
    data, source_manifest = load_verified_data(dataset, args.source_root)
    output = ROOT / "SeqRec" / f"data_{dataset}"
    manifest_path = output / "ver4_manifest.json"
    metadata_path = output / "text_name_dict.json.gz"  # Upstream name; content is pickle.
    required = [
        output / f"{dataset}_{split}.txt" for split in ("train", "valid", "test")
    ] + [metadata_path, manifest_path]
    if all(path.exists() for path in required) and not args.force:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    split_hashes = export_split_files(dataset, data, output)
    config = DATASETS[dataset]
    item_index, titles, descriptions, matched = build_item_text(
        data, config, args.cache_dir, args.placeholders_only
    )
    if args.placeholders_only:
        times: dict[int, dict[int, int]] = defaultdict(dict)
        for user in range(data.num_users):
            for item in canonical_sequence(data, user):
                times[item + 1][user + 1] = 0
        timestamp_rows = sum(len(canonical_sequence(data, user)) for user in range(data.num_users))
    else:
        times, timestamp_rows = reconstruct_times(data, config, item_index, args.cache_dir)

    text_data = {"time": times, "description": descriptions, "title": titles}
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(text_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(metadata_path)

    manifest = protocol_manifest(dataset, data, source_manifest)
    manifest.update(
        {
            "source_commit": "b81019ca655fb759cee895924b8b6c7cc0f0cce9",
            "split_file_sha256": split_hashes,
            "text_metadata_sha256": sha256_file(metadata_path),
            "metadata_items_matched": matched,
            "timestamp_interactions_verified": timestamp_rows,
            "placeholders_only": args.placeholders_only,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        prepare_one(dataset, args)


if __name__ == "__main__":
    main()
