"""Dataset adapters for the independent multi-agent Mamba-RL pipeline.

This module deliberately does not change :mod:`src.data`.  All loaders return
the event tuple consumed by ``build_data``: ``(user, item, timestamp, text)``.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Iterable, Iterator
from urllib.request import urlopen
import zipfile

from tqdm.auto import tqdm

from src.data import InteractionData, build_data, load_amazon, synthetic_events


MOVIELENS_URLS = {
    "movielens-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "movielens-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "movielens-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "movielens-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "movielens-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "movielens-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


def _limit(rows: Iterable[tuple[str, str, int, str]], maximum: int | None):
    if maximum is None:
        yield from rows
        return
    for index, row in enumerate(rows):
        if index >= maximum:
            break
        yield row


def _download_movielens(name: str, cache_dir: str) -> Path:
    target = Path(cache_dir) / "raw" / name
    if target.exists() and any(target.rglob("ratings*")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{name}.zip"
    url = MOVIELENS_URLS[name]
    with tqdm(total=1, desc=f"Downloading {name}", unit="archive") as progress:
        with urlopen(url) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        progress.update(1)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink()
    return target


def _find_one(root: Path, names: tuple[str, ...]) -> Path | None:
    if root.is_file():
        return root
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _movie_texts(root: Path) -> dict[str, str]:
    csv_path = _find_one(root, ("movies.csv",))
    if csv_path:
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as stream:
            return {
                str(row["movieId"]): f'{row.get("title", "")} Genres: {row.get("genres", "")}'.strip()
                for row in csv.DictReader(stream)
            }
    dat_path = _find_one(root, ("movies.dat",))
    if dat_path:
        result = {}
        with dat_path.open(encoding="latin-1", errors="replace") as stream:
            for line in stream:
                movie, title, genres = line.rstrip("\n").split("::", 2)
                result[movie] = f"{title} Genres: {genres}"
        return result
    item_path = _find_one(root, ("u.item",))
    if item_path:
        genres = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        result = {}
        with item_path.open(encoding="latin-1", errors="replace") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("|")
                active = [genre for genre, flag in zip(genres, fields[5:24]) if flag == "1"]
                result[fields[0]] = f'{fields[1]} Genres: {"|".join(active)}'
        return result
    raise FileNotFoundError(f"Could not find movies.csv, movies.dat, or u.item below {root}")


def load_movielens(
    name: str,
    data_path: str | None,
    cache_dir: str,
    max_events: int | None,
    min_rating: float,
) -> list[tuple[str, str, int, str]]:
    root = Path(data_path) if data_path else _download_movielens(name, cache_dir)
    texts = _movie_texts(root)
    ratings = _find_one(root, ("ratings.csv", "ratings.dat", "u.data"))
    if ratings is None:
        raise FileNotFoundError(f"Could not find MovieLens ratings below {root}")

    def rows() -> Iterator[tuple[str, str, int, str]]:
        if ratings.name == "ratings.csv":
            with ratings.open(encoding="utf-8", errors="replace", newline="") as stream:
                for row in csv.DictReader(stream):
                    if float(row["rating"]) >= min_rating:
                        item = str(row["movieId"])
                        yield str(row["userId"]), item, int(float(row["timestamp"])), texts.get(item, item)
        elif ratings.name == "ratings.dat":
            with ratings.open(encoding="latin-1", errors="replace") as stream:
                for line in stream:
                    user, item, rating, timestamp = line.rstrip("\n").split("::")
                    if float(rating) >= min_rating:
                        yield user, item, int(timestamp), texts.get(item, item)
        else:
            with ratings.open(encoding="latin-1", errors="replace") as stream:
                for line in stream:
                    user, item, rating, timestamp = line.rstrip("\n").split("\t")[:4]
                    if float(rating) >= min_rating:
                        yield user, item, int(timestamp), texts.get(item, item)

    return list(_limit(rows(), max_events))


def _json_lines(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def _timestamp(value) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())


def _business_text(row: dict) -> str:
    categories = row.get("categories") or ""
    attributes = row.get("attributes") or {}
    useful_attributes = " ".join(f"{key}: {value}" for key, value in list(attributes.items())[:20])
    return " ".join(
        str(value).strip()
        for value in (row.get("name"), categories, row.get("city"), row.get("state"), useful_attributes)
        if value and str(value).strip()
    )


def _load_yelp_json(
    root: Path, max_events: int | None, min_rating: float
) -> list[tuple[str, str, int, str]]:
    reviews = _find_one(root, ("yelp_academic_dataset_review.json", "review.json", "reviews.jsonl"))
    businesses = _find_one(root, ("yelp_academic_dataset_business.json", "business.json", "businesses.jsonl"))
    if reviews is None:
        raise FileNotFoundError(f"Could not find a Yelp review JSON file below {root}")
    interactions: list[tuple[str, str, int]] = []
    required = set()
    for row in tqdm(_json_lines(reviews), desc="Reading Yelp reviews", unit="review"):
        if float(row.get("stars", row.get("rating", 0))) < min_rating:
            continue
        user = row.get("user_id") or row.get("user")
        item = row.get("business_id") or row.get("item_id")
        if user and item:
            interactions.append((str(user), str(item), _timestamp(row.get("date") or row.get("timestamp"))))
            required.add(str(item))
            if max_events is not None and len(interactions) >= max_events:
                break
    texts: dict[str, str] = {}
    if businesses:
        for row in tqdm(_json_lines(businesses), desc="Joining Yelp businesses", unit="business"):
            item = str(row.get("business_id") or row.get("item_id") or "")
            if item in required:
                texts[item] = _business_text(row) or item
    return [(user, item, timestamp, texts.get(item, item)) for user, item, timestamp in interactions]


def _load_generic_table(
    path: Path, max_events: int | None, min_rating: float
) -> list[tuple[str, str, int, str]]:
    aliases = {
        "user": ("user_id", "userId", "user"),
        "item": ("business_id", "item_id", "itemId", "movieId", "item"),
        "rating": ("stars", "rating", "score"),
        "time": ("timestamp", "date", "time"),
        "text": ("item_text", "business_text", "title", "name", "text"),
    }

    def value(row: dict, kind: str, default=None):
        return next((row[key] for key in aliases[kind] if key in row and row[key] not in (None, "")), default)

    if path.suffix.lower() in {".json", ".jsonl"}:
        source = _json_lines(path)
    elif path.suffix.lower() == ".csv":
        stream = path.open(encoding="utf-8", errors="replace", newline="")
        source = csv.DictReader(stream)
    elif path.suffix.lower() == ".parquet":
        from datasets import load_dataset
        source = load_dataset("parquet", data_files=str(path), split="train")
    else:
        raise ValueError("Generic data files must be .csv, .json, .jsonl, or .parquet")
    result = []
    try:
        for row in source:
            rating = float(value(row, "rating", min_rating))
            if rating < min_rating:
                continue
            user, item = value(row, "user"), value(row, "item")
            if user is None or item is None:
                continue
            result.append((str(user), str(item), _timestamp(value(row, "time", 0)), str(value(row, "text", item))))
            if max_events is not None and len(result) >= max_events:
                break
    finally:
        if path.suffix.lower() == ".csv":
            stream.close()
    return result


def load_yelp(
    data_path: str | None, max_events: int | None, min_rating: float
) -> list[tuple[str, str, int, str]]:
    if not data_path:
        raise ValueError(
            "Yelp snapshots are license-gated. Download Yelp Open Dataset 2019/2023, then pass "
            "its directory or a normalized CSV/JSONL/Parquet file with --data-path."
        )
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        return _load_generic_table(path, max_events, min_rating)
    if _find_one(path, ("yelp_academic_dataset_review.json", "review.json", "reviews.jsonl")):
        return _load_yelp_json(path, max_events, min_rating)
    tables = [
        candidate
        for suffix in ("*.csv", "*.jsonl", "*.parquet")
        for candidate in path.rglob(suffix)
    ]
    if tables:
        return _load_generic_table(tables[0], max_events, min_rating)
    raise FileNotFoundError(f"No Yelp review JSON or normalized table found below {path}")


def amazon_subset(name: str) -> str:
    if name.lower() in {"amazon-games", "amazon-toys"}:
        return "raw_review_Toys_and_Games"
    if name.startswith("amazon:"):
        subset = name.split(":", 1)[1]
        return subset if subset.startswith("raw_review_") else f"raw_review_{subset}"
    category = name.removeprefix("amazon-")
    special = {"and": "and", "tv": "TV", "cds": "CDs", "dvd": "DVD", "mp3": "MP3"}
    category = "_".join(
        special.get(part.lower(), part.capitalize())
        for part in category.replace("_", "-").split("-")
        if part
    )
    return f"raw_review_{category}"


def amazon_item_group(name: str) -> str | None:
    """Return the deterministic split for aliases of Amazon's combined category."""
    return {"amazon-games": "games", "amazon-toys": "toys"}.get(name.lower())


def load_recommendation_data(
    dataset: str,
    data_path: str | None,
    cache_dir: str,
    max_events: int | None = None,
    min_rating: float = 4.0,
    min_user_events: int = 5,
    sasrec_filtering: bool = False,
) -> InteractionData:
    name = dataset.lower()
    if name == "synthetic":
        events = synthetic_events()
    elif name in MOVIELENS_URLS:
        events = load_movielens(name, data_path, cache_dir, max_events, min_rating)
    elif name.startswith("amazon-") or name.startswith("amazon:"):
        events = load_amazon(
            amazon_subset(dataset), max_events, cache_dir, item_group=amazon_item_group(dataset)
        )
    elif name in {"yelp19", "yelp-2019", "yelp23", "yelp-2023"}:
        events = load_yelp(data_path, max_events, min_rating)
    else:
        supported = ", ".join((*MOVIELENS_URLS, "amazon-<category>", "yelp19", "yelp23", "synthetic"))
        raise ValueError(f"Unknown dataset {dataset!r}. Supported values: {supported}")
    if not events:
        raise RuntimeError(f"No positive interactions loaded for {dataset!r}; check --min-rating and --data-path")
    return build_data(
        events,
        min_user_events=min_user_events,
        sasrec_filtering=sasrec_filtering,
    )
