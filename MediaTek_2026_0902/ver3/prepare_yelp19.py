#!/usr/bin/env python3
"""Convert the published MAPLE/NETE Yelp19 pickle to safe JSONL.

The source does not contain review timestamps.  The conversion therefore uses
the stable source-row index as the ordering key consumed by the existing
leave-two-out pipeline.  It also builds one compact item profile from the
categories and aspects in the training corpus so the frozen Mamba text encoder
receives useful restaurant semantics rather than an opaque item identifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle

from tqdm.auto import tqdm


ALLOWED_GLOBALS = {
    ("builtins", "dict"): dict,
    ("collections", "defaultdict"): defaultdict,
}


class RestrictedUnpickler(pickle.Unpickler):
    """Only permit the two benign constructors present in the release."""

    def find_class(self, module: str, name: str):
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terms(row: dict) -> tuple[list[str], list[str]]:
    categories = [part.strip() for part in str(row.get("category") or "").split(",") if part.strip()]
    aspects: list[str] = []
    for triplet in row.get("triplets") or ():
        if isinstance(triplet, (list, tuple)) and triplet and triplet[0]:
            aspects.append(str(triplet[0]).strip())
    template = row.get("template")
    if isinstance(template, (list, tuple)) and template and template[0]:
        aspects.append(str(template[0]).strip())
    return categories, [aspect for aspect in aspects if aspect]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--top-categories", type=int, default=8)
    parser.add_argument("--top-aspects", type=int, default=12)
    args = parser.parse_args()

    source_hash = sha256(args.source)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise RuntimeError(f"source SHA-256 mismatch: {source_hash}")

    with args.source.open("rb") as stream:
        rows = RestrictedUnpickler(stream).load()
    if not isinstance(rows, list) or not rows:
        raise TypeError("expected a non-empty list of Yelp review dictionaries")

    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    aspect_counts: dict[str, Counter[str]] = defaultdict(Counter)
    ratings = Counter()
    users = set()
    items = set()
    for row in tqdm(rows, desc="Building Yelp19 item profiles", unit="review"):
        if not isinstance(row, dict) or "user" not in row or "item" not in row or "rating" not in row:
            raise TypeError("unexpected Yelp19 review schema")
        user, item = str(row["user"]), str(row["item"])
        users.add(user)
        items.add(item)
        ratings[str(row["rating"])] += 1
        categories, aspects = terms(row)
        category_counts[item].update(categories)
        aspect_counts[item].update(aspects)

    profiles = {}
    for item in items:
        categories = [term for term, _ in category_counts[item].most_common(args.top_categories)]
        aspects = [term for term, _ in aspect_counts[item].most_common(args.top_aspects)]
        parts = ["Yelp restaurant"]
        if categories:
            parts.append("Categories: " + ", ".join(categories))
        if aspects:
            parts.append("Popular aspects: " + ", ".join(aspects))
        profiles[item] = ". ".join(parts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(tqdm(rows, desc="Writing normalized Yelp19", unit="review")):
            item = str(row["item"])
            record = {
                "user_id": str(row["user"]),
                "item_id": item,
                "rating": float(row["rating"]),
                "timestamp": index,
                "item_text": profiles[item],
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_rows": len(rows),
        "users": len(users),
        "items": len(items),
        "rating_counts": dict(sorted(ratings.items())),
        "ordering": "stable source-row index (source release has no review timestamps)",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "item_profile": {
            "top_categories": args.top_categories,
            "top_aspects": args.top_aspects,
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
