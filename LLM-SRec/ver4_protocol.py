"""Shared MediaTek ver4 data and sampling contract for LLM-SRec.

This module intentionally has no model imports.  It is safe to use for data
preparation, verification, and launcher preflight checks without loading an
LLM or neural-network checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
VER4_ROOT = WORKSPACE / "MediaTek_2026_0902" / "ver4"
DEFAULT_DATA_ROOT = ROOT / "data" / "ver4_protocol"
DEFAULT_SOURCE_ROOT = WORKSPACE / "SIGMA" / "data" / "ver4_protocol"

# The keys are deliberately shared by all launchers and output paths.
DATASETS = {
    "all_beauty": {
        "amazon_config": "raw_review_All_Beauty",
        "metadata_config": "raw_meta_All_Beauty",
        "max_transitions": 500_000,
        "expected": {
            "num_users": 1_603,
            "evaluation_users": 1_457,
            "num_items": 4_090,
            "train_interactions": 8_445,
        },
    },
    "baby_products": {
        "amazon_config": "raw_review_Baby_Products",
        "metadata_config": "raw_meta_Baby_Products",
        "max_transitions": 1_500_000,
        "expected": {
            "num_users": 184_834,
            "evaluation_users": 184_584,
            "num_items": 77_324,
            "train_interactions": 1_121_805,
        },
    },
    "sports_and_outdoors": {
        "amazon_config": "raw_review_Sports_and_Outdoors",
        "metadata_config": "raw_meta_Sports_and_Outdoors",
        "max_transitions": 1_000_000,
        "expected": {
            "num_users": 631_803,
            "evaluation_users": 625_563,
            "num_items": 401_572,
            "train_interactions": 3_678_023,
        },
    },
    "toys_and_games": {
        "amazon_config": "raw_review_Toys_and_Games",
        "metadata_config": "raw_meta_Toys_and_Games",
        "max_transitions": 1_500_000,
        "expected": {
            "num_users": 568_262,
            "evaluation_users": 564_666,
            "num_items": 333_160,
            "train_interactions": 3_734_374,
        },
    },
}


@dataclass(frozen=True)
class Transition:
    user: int
    end: int
    target: int


def _enable_interaction_data_unpickle() -> None:
    """Make the class stored in the canonical SIGMA pickle importable."""
    path = str(VER4_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    from src.data import InteractionData  # noqa: F401, PLC0415


def statistics(data) -> dict[str, int]:
    return {
        "num_users": data.num_users,
        "evaluation_users": len(data.test_target),
        "num_items": data.num_items,
        "train_interactions": sum(map(len, data.train_by_user.values())),
    }


def split_signature(data) -> str:
    """Match the canonical SIGMA/ver4 fingerprint byte-for-byte."""
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


def load_verified_data(
    dataset: str,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {tuple(DATASETS)}")
    source_root = Path(source_root)
    artifact = source_root / f"{dataset}.pkl"
    manifest_path = source_root / f"{dataset}.json"
    if not artifact.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Canonical ver4 split is missing: {artifact} and {manifest_path} are required"
        )
    _enable_interaction_data_unpickle()
    with artifact.open("rb") as stream:
        data = pickle.load(stream)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = statistics(data)
    expected = DATASETS[dataset]["expected"]
    signature = split_signature(data)
    if actual != expected:
        raise RuntimeError(
            f"{dataset} statistics mismatch: expected={expected}, actual={actual}"
        )
    if actual != manifest.get("statistics"):
        raise RuntimeError(f"{dataset} artifact and source manifest statistics differ")
    if signature != manifest.get("split_sha256"):
        raise RuntimeError(f"{dataset} split fingerprint mismatch")
    return data, manifest


def build_transitions(data, maximum: int | None, seed: int) -> list[Transition]:
    """Use ver4's coverage-first reservoir sampling exactly."""
    rng = random.Random(seed)
    eligible = [
        (user, len(history) - 1)
        for user, history in data.train_by_user.items()
        if len(history) > 1
    ]
    total = sum(count for _, count in eligible)
    budget = total if maximum is None or maximum <= 0 else min(maximum, total)
    rng.shuffle(eligible)

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
        raise RuntimeError("No training prefixes exist in the canonical split")
    rng.shuffle(result)
    return result


def protocol_manifest(dataset: str, data, source_manifest: dict) -> dict:
    return {
        "dataset_key": dataset,
        "amazon_2023_config": DATASETS[dataset]["amazon_config"],
        "metadata_config": DATASETS[dataset]["metadata_config"],
        "filter": "one-pass raw user>=5 and raw item>=5",
        "split": "stable chronological leave-two-out; post-filter length<3 is training-only",
        "statistics": statistics(data),
        "split_sha256": split_signature(data),
        "source_split_sha256": source_manifest["split_sha256"],
        "protocol": {
            "seed": 25252,
            "epochs": 18,
            "batch_size": 128,
            "evaluation_batch_size": 64,
            "max_history": 50,
            "early_stopping_patience": 6,
            "selection_metric": "validation ndcg@10",
            "evaluation": "all eligible users, full catalog, seen-item masking",
            "max_transitions": DATASETS[dataset]["max_transitions"],
        },
    }
