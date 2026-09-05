#!/usr/bin/env python3
"""Build compact SIGMA inputs with MediaTek ver4's exact ID split semantics.

Only review IDs and timestamps are retained because SIGMA learns ID embeddings;
item metadata cannot affect the requested counts or the split.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

from datasets import load_dataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
VER4_ROOT = ROOT.parent / "MediaTek_2026_0902" / "ver4"
sys.path.insert(0, str(VER4_ROOT))
from src.data import InteractionData  # noqa: E402


DATASETS = {
    "all_beauty": ("amazon-all-beauty", "raw_review_All_Beauty"),
    "baby_products": ("amazon:Baby_Products", "raw_review_Baby_Products"),
    "sports_and_outdoors": ("amazon-sports-and-outdoors", "raw_review_Sports_and_Outdoors"),
    "toys_and_games": ("amazon:Toys_and_Games", "raw_review_Toys_and_Games"),
}

# Values from completed sasrec_filtering=True ver4 logs. The old Toys run used
# amazon-toys (a metadata-keyword subgroup), not full Toys_and_Games.
COMPLETED_VER4_STATS = {
    "all_beauty": {
        "num_users": 1603,
        "evaluation_users": 1457,
        "num_items": 4090,
        "train_interactions": 8445,
    },
    "baby_products": {
        "num_users": 184834,
        "evaluation_users": 184584,
        "num_items": 77324,
        "train_interactions": 1121805,
    },
    "sports_and_outdoors": {
        "num_users": 631803,
        "evaluation_users": 625563,
        "num_items": 401572,
        "train_interactions": 3678023,
    },
}


def split_signature(data: InteractionData) -> str:
    digest = hashlib.sha256()
    for user in range(data.num_users):
        history = data.train_by_user[user]
        digest.update(array("I", (user, len(history), *history)).tobytes())
        digest.update(
            array(
                "i",
                (data.valid_target.get(user, -1), data.test_target.get(user, -1)),
            ).tobytes()
        )
    return digest.hexdigest()


def statistics(data: InteractionData) -> dict[str, int]:
    return {
        "num_users": data.num_users,
        "evaluation_users": len(data.test_target),
        "num_items": data.num_items,
        "train_interactions": sum(map(len, data.train_by_user.values())),
    }


def build_id_only_ver4_split(rows) -> InteractionData:
    """Equivalent to ver4 ``build_data(..., sasrec_filtering=True)``.

    This preserves raw one-pass counts, stable timestamp sorting, user ordering,
    item first-occurrence ordering, short training-only users, and leave-two-out.
    """
    per_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    item_counts: dict[str, int] = defaultdict(int)
    for row in tqdm(rows, total=len(rows), desc="Scanning raw reviews", unit="review"):
        user = row.get("user_id") or row.get("user")
        item = row.get("parent_asin") or row.get("asin") or row.get("item_id")
        if not user or not item:
            continue
        user, item = str(user), str(item)
        per_user[user].append((int(row.get("timestamp") or 0), item))
        item_counts[item] += 1

    filtered: dict[str, list[tuple[int, str]]] = {}
    for user, entries in tqdm(
        per_user.items(), total=len(per_user), desc="Applying ver4 one-pass 5-core", unit="user"
    ):
        if len(entries) < 5:
            continue
        kept = [entry for entry in entries if item_counts[entry[1]] >= 5]
        if kept:
            filtered[user] = sorted(kept, key=lambda entry: entry[0])
    if not filtered:
        raise RuntimeError("No users remain after ver4 filtering")

    user_map = {user: index for index, user in enumerate(sorted(filtered))}
    item_map: dict[str, int] = {}
    for entries in filtered.values():
        for _, item in entries:
            item_map.setdefault(item, len(item_map))

    train_by_user: dict[int, list[int]] = {}
    valid_target: dict[int, int] = {}
    test_target: dict[int, int] = {}
    for user, entries in tqdm(
        filtered.items(), total=len(filtered), desc="Chronological leave-two-out", unit="user"
    ):
        ids = [item_map[item] for _, item in entries]
        uid = user_map[user]
        if len(ids) < 3:
            train_by_user[uid] = ids
        else:
            train_by_user[uid] = ids[:-2]
            valid_target[uid], test_target[uid] = ids[-2], ids[-1]

    item_texts = [""] * len(item_map)
    for item, index in item_map.items():
        item_texts[index] = item
    return InteractionData(
        train_by_user,
        valid_target,
        test_target,
        item_texts,
        len(user_map),
        len(item_map),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(DATASETS))
    parser.add_argument("--cache-dir", default=str(ROOT.parent / "cache"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "ver4_protocol"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{args.dataset}.pkl"
    manifest_path = output_dir / f"{args.dataset}.json"
    if artifact.exists() and manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    ver4_name, config_name = DATASETS[args.dataset]
    rows = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        split="full",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    data = build_id_only_ver4_split(rows)
    stats = statistics(data)
    expected = COMPLETED_VER4_STATS.get(args.dataset)
    if expected is not None and stats != expected:
        raise RuntimeError(f"ver4 statistics mismatch: expected={expected}, actual={stats}")
    manifest = {
        "dataset_key": args.dataset,
        "ver4_dataset_argument": ver4_name,
        "amazon_2023_config": config_name,
        "filter": "one-pass raw user>=5 and raw item>=5 (ver4 --sasrec-filtering)",
        "split": "per-user chronological leave-two-out; post-filter length<3 is training-only",
        "statistics": stats,
        "completed_ver4_reference": expected,
        "split_sha256": split_signature(data),
        "artifact": str(artifact.resolve()),
    }
    temporary = artifact.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(artifact)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

