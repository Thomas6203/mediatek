from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Iterable

from tqdm.auto import tqdm


def _show_progress() -> bool:
    """Only rank 0 renders preprocessing bars during DDP runs."""
    return os.environ.get("RANK", "0") == "0"


@dataclass
class InteractionData:
    """Integer-indexed implicit interactions and text associated with every item."""

    train_by_user: dict[int, list[int]]
    valid_target: dict[int, int]
    test_target: dict[int, int]
    item_texts: list[str]
    num_users: int
    num_items: int


def _normalise_row(row: dict, item_texts: dict[str, str] | None = None) -> tuple[str, str, int, str] | None:
    user = row.get("user_id") or row.get("user")
    item = row.get("parent_asin") or row.get("asin") or row.get("item_id")
    if not user or not item:
        return None
    # Use item metadata when available; a review title/text is only a fallback.
    text = (item_texts or {}).get(str(item)) or str(row.get("title") or row.get("text") or item)
    timestamp = int(row.get("timestamp") or 0)
    return str(user), str(item), timestamp, text


def _metadata_text(row: dict) -> str:
    """Make a compact LLM input from the Amazon item-metadata fields."""
    parts = [row.get("title"), row.get("subtitle"), row.get("main_category")]
    parts.extend(row.get("features") or [])
    parts.extend(row.get("description") or [])
    return " ".join(str(part).strip() for part in parts if part and str(part).strip())


_GAME_METADATA_TERMS = (
    "game", "gaming", "puzzle", "chess", "checkers", "domino", "dice",
    "playing card", "board game", "tabletop", "mahjong", "bingo",
)


def _amazon_toys_and_games_group(row: dict) -> str:
    """Split the combined catalog deterministically using stable metadata fields."""
    categories = row.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    haystack = " ".join(
        str(value).lower()
        for value in (row.get("title"), row.get("subtitle"), row.get("main_category"), *categories)
        if value
    )
    return "games" if any(term in haystack for term in _GAME_METADATA_TERMS) else "toys"


def load_amazon(
    subset: str,
    max_events: int | None,
    cache_dir: str | None = None,
    item_group: str | None = None,
) -> list[tuple[str, str, int, str]]:
    """Load reviews and item metadata from matching Amazon Reviews 2023 configs.

    The raw review configuration has names such as ``raw_review_All_Beauty``;
    its matching metadata configuration is ``raw_meta_All_Beauty``.
    """
    import datasets
    from datasets import load_dataset

    major_version = int(datasets.__version__.split(".", 1)[0])
    if major_version >= 4:
        raise RuntimeError(
            "McAuley-Lab/Amazon-Reviews-2023 uses a dataset loading script, which datasets 4.x no "
            "longer supports. Install the project requirements (datasets==3.6.0), then retry."
        )

    if not subset.startswith("raw_review_"):
        raise ValueError("--subset must start with 'raw_review_', e.g. raw_review_All_Beauty")
    if item_group not in {None, "games", "toys"}:
        raise ValueError(f"Unsupported Amazon item group: {item_group!r}")
    if item_group is not None and subset != "raw_review_Toys_and_Games":
        raise ValueError("Games/Toys item grouping is only valid for raw_review_Toys_and_Games")
    # Keep the DatasetDict form used in the official review-loading example.
    with tqdm(total=1, desc="Loading full review dataset", unit="dataset", disable=not _show_progress()) as progress:
        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", subset,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        progress.update(1)
    review_split = reviews["full"]
    limited_reviews = review_split if max_events is None else review_split.select(range(min(max_events, len(review_split))))

    metadata_subset = subset.replace("raw_review_", "raw_meta_", 1)
    with tqdm(total=1, desc="Loading full item metadata dataset", unit="dataset", disable=not _show_progress()) as progress:
        metadata = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", metadata_subset, split="full",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        progress.update(1)
    required_items = set()
    for row in tqdm(limited_reviews, total=len(limited_reviews), desc="Scanning review item IDs", unit="review", disable=not _show_progress()):
        if row.get("parent_asin"):
            required_items.add(str(row["parent_asin"]))
    item_texts = {}
    for row in tqdm(metadata, total=len(metadata), desc="Joining item metadata by parent_asin", unit="item", disable=not _show_progress()):
        parent_asin = row.get("parent_asin")
        if (
            parent_asin in required_items
            and (item_group is None or _amazon_toys_and_games_group(row) == item_group)
            and (text := _metadata_text(row))
        ):
            item_texts[str(parent_asin)] = text

    rows = []
    for row in tqdm(limited_reviews, total=len(limited_reviews), desc="Normalising review interactions", unit="review", disable=not _show_progress()):
        if item_group is not None and str(row.get("parent_asin") or "") not in item_texts:
            continue
        normal = _normalise_row(row, item_texts)
        if normal is not None:
            rows.append(normal)
    if not rows:
        raise RuntimeError(f"No usable user/item rows found in subset {subset!r}.")
    return rows


def synthetic_events() -> list[tuple[str, str, int, str]]:
    """A deterministic tiny dataset for a complete local smoke test."""
    rows = []
    titles = ["skin cleanser", "vitamin serum", "face cream", "hair shampoo", "body lotion"]
    for user in range(24):
        for step in range(5):
            item = (user + step * 2) % len(titles)
            rows.append((f"u{user}", f"i{item}", step + user * 10, titles[item]))
    return rows


SASREC_MIN_INTERACTIONS = 5


def build_data(
    events: Iterable[tuple[str, str, int, str]],
    min_user_events: int = 3,
    sasrec_filtering: bool = False,
) -> InteractionData:
    per_user: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    item_counts: dict[str, int] = defaultdict(int)
    total_events = len(events) if hasattr(events, "__len__") else None
    for user, item, ts, text in tqdm(events, total=total_events, desc="Grouping interactions by user", unit="interaction", disable=not _show_progress()):
        per_user[user].append((ts, item, text))
        item_counts[item] += 1

    if sasrec_filtering:
        # Match SASRec/data/DataProcessing.py: count on the raw interactions,
        # then apply user/item >= 5 once (this is not an iterative k-core).
        per_user = {
            user: sorted(
                (
                    entry for entry in entries
                    if item_counts[entry[1]] >= SASREC_MIN_INTERACTIONS
                ),
                key=lambda entry: entry[0],
            )
            for user, entries in per_user.items()
            if len(entries) >= SASREC_MIN_INTERACTIONS
        }
        per_user = {user: entries for user, entries in per_user.items() if entries}
    else:
        per_user = {
            user: sorted(entries)
            for user, entries in per_user.items()
            if len(entries) >= min_user_events
        }
    if not per_user:
        raise RuntimeError("No users have enough events after filtering.")

    user_map = {u: idx for idx, u in enumerate(sorted(per_user))}
    item_map: dict[str, int] = {}
    item_text: dict[str, str] = {}
    for entries in tqdm(per_user.values(), total=len(per_user), desc="Building item index and text table", unit="user", disable=not _show_progress()):
        for _, item, text in entries:
            item_map.setdefault(item, len(item_map))
            item_text.setdefault(item, text)

    train_by_user, valid, test = {}, {}, {}
    for user, entries in tqdm(per_user.items(), total=len(per_user), desc="Creating chronological train/valid/test split", unit="user", disable=not _show_progress()):
        ids = [item_map[item] for _, item, _ in entries]
        uid = user_map[user]
        if sasrec_filtering and len(ids) < 3:
            # SASRec retains these users for training but skips evaluation
            # because they do not receive validation/test targets.
            train_by_user[uid] = ids
        else:
            train_by_user[uid] = ids[:-2]
            valid[uid], test[uid] = ids[-2], ids[-1]
    item_texts = [""] * len(item_map)
    for item, iid in tqdm(item_map.items(), total=len(item_map), desc="Finalising item text array", unit="item", disable=not _show_progress()):
        item_texts[iid] = item_text[item]
    return InteractionData(train_by_user, valid, test, item_texts, len(user_map), len(item_map))


def edge_index(data: InteractionData) -> torch.Tensor:
    """Return directed edges, with users offset by zero and items by num_users."""
    import torch

    src, dst = [], []
    for user, history in tqdm(data.train_by_user.items(), total=len(data.train_by_user), desc="Building bipartite graph edges", unit="user", disable=not _show_progress()):
        for item in set(history):
            src.extend((user, data.num_users + item))
            dst.extend((data.num_users + item, user))
    return torch.tensor([src, dst], dtype=torch.long)
