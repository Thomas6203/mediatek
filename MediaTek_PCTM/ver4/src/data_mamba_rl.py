"""Dataset adapters for the independent multi-agent Mamba-RL pipeline.

This module deliberately does not change :mod:`src.data`.  All loaders return
the event tuple consumed by ``build_data``: ``(user, item, timestamp, text)``.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import shutil
from typing import Iterable, Iterator
from urllib.request import urlopen
import zipfile

from tqdm.auto import tqdm

from src.data import InteractionData, build_data, load_amazon, synthetic_events
from src.data_contract import (
    PCTM_ML20M_DATASET,
    PROTOCOL_DATASETS,
    build_evaluation_contract,
    inspect_protocol_split,
    is_protocol_dataset,
    protocol_dataset_spec,
    resolve_protocol_split,
    sha256_and_rows,
    verify_protocol_split,
)


MOVIELENS_URLS = {
    "movielens-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "movielens-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "movielens-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "movielens-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "movielens-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "movielens-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}

# Backwards-compatible export used by existing tests/callers.
PCTM_ML20M_SPLIT = protocol_dataset_spec(PCTM_ML20M_DATASET)["files"]


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


def resolve_pctm_ml20m_split(data_path: str | None, cache_dir: str) -> Path:
    """Compatibility wrapper for the generalized protocol split resolver."""
    return resolve_protocol_split(PCTM_ML20M_DATASET, data_path, cache_dir)


def _sha256_and_rows(path: Path) -> tuple[str, int]:
    return sha256_and_rows(path)


def verify_pctm_ml20m_split(directory: Path) -> None:
    """Compatibility wrapper for the generalized fail-closed verifier."""
    verify_protocol_split(PCTM_ML20M_DATASET, directory)


_PCTM_ISO_TIME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[ T](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})?)?$"
)


def _pctm_time_key(value: str) -> Decimal:
    """Return an exact sortable timestamp for official PCTM CSV values.

    The Amazon builders serialize nanosecond-resolution pandas timestamps such
    as ``1970-01-01 00:00:00.000000001``.  CPython 3.10's
    :meth:`datetime.fromisoformat` accepts at most microsecond precision, so the
    fractional component is parsed separately and retained as a ``Decimal``.
    """
    value = value.strip()
    try:
        numeric = Decimal(value)
    except InvalidOperation:
        numeric = None
    if numeric is not None:
        if numeric.is_finite():
            return numeric
        raise ValueError(f"Unsupported PCTM datetime value: {value!r}")

    match = _PCTM_ISO_TIME.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported PCTM datetime value: {value!r}")

    time = match.group("time") or "00:00:00"
    zone = match.group("zone") or ""
    if zone == "Z":
        zone = "+00:00"
    try:
        parsed = datetime.fromisoformat(f"{match.group('date')}T{time}{zone}")
    except ValueError as error:
        raise ValueError(f"Unsupported PCTM datetime value: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = Decimal(delta.days * 86400 + delta.seconds)
    fraction = match.group("fraction")
    if fraction:
        seconds += Decimal(f"0.{fraction}")
    return seconds


def _read_pctm_train(path: Path) -> tuple[dict[int, list[int]], set[int]]:
    """Read the source-grouped official CSV while preserving timestamp tie order."""
    histories: dict[int, list[int]] = {}
    catalog: set[int] = set()
    current_user: int | None = None
    current_rows: list[tuple[Decimal, int, int]] = []

    def flush() -> None:
        if current_user is None:
            return
        if current_user in histories:
            raise ValueError(
                f"Rows for user_id={current_user} are not contiguous in {path}; "
                "this is not the released eSASRec split ordering."
            )
        current_rows.sort(key=lambda entry: (entry[0], entry[1]))
        histories[current_user] = [item for _, _, item in current_rows]

    with path.open(encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"user_id", "item_id", "datetime"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for source_order, row in enumerate(reader):
            user, item = int(row["user_id"]), int(row["item_id"])
            if current_user is None:
                current_user = user
            elif user != current_user:
                flush()
                current_user, current_rows = user, []
            current_rows.append((_pctm_time_key(row["datetime"]), source_order, item))
            catalog.add(item)
    flush()
    if not histories:
        raise RuntimeError(f"No interactions found in {path}")
    return histories, catalog


def _read_pctm_holdout(path: Path) -> dict[int, int]:
    targets: dict[int, int] = {}
    with path.open(encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"user_id", "item_id", "datetime"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            user, item = int(row["user_id"]), int(row["item_id"])
            if user in targets:
                raise ValueError(f"Expected one outer holdout row for user_id={user} in {path}")
            targets[user] = item
    if not targets:
        raise RuntimeError(f"No holdout interactions found in {path}")
    return targets


def _pctm_metadata_texts(split_dir: Path, metadata_path: str | None) -> dict[str, str]:
    if metadata_path:
        root = Path(metadata_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(root)
        return _movie_texts(root)
    for ancestor in (split_dir, *split_dir.parents):
        for relative in (
            Path("movies.csv"),
            Path("data/raw/ml_20m/movies.csv"),
            Path(".external/transformer_benchmark/data/raw/ml_20m/movies.csv"),
        ):
            candidate = ancestor / relative
            if candidate.is_file():
                return _movie_texts(candidate)
    raise FileNotFoundError(
        "--pctm-item-text-mode metadata requires MovieLens movies.csv. "
        "Pass its file or parent directory with --pctm-metadata-path."
    )


def load_protocol_recommendation_data(
    dataset: str,
    data_path: str | None,
    cache_dir: str,
    *,
    verify_split: bool = True,
    item_text_mode: str = "id",
    metadata_path: str | None = None,
    file_contract: dict[str, object] | None = None,
) -> InteractionData:
    """Load a frozen outer split and derive a training-only inner LOO split.

    The returned training histories exclude each user's last outer-train event.
    ``valid_target`` contains only warm, unseen inner targets.  Test histories are
    retained separately so final evaluation sees the complete untouched outer
    training sequence, including users that were ineligible for inner validation.
    """
    if item_text_mode not in {"id", "metadata"}:
        raise ValueError("PCTM item text mode must be 'id' or 'metadata'")
    split_dir = resolve_protocol_split(dataset, data_path, cache_dir)
    if file_contract is None:
        file_contract = (
            verify_protocol_split(dataset, split_dir)
            if verify_split
            else inspect_protocol_split(dataset, split_dir)
        )
    if file_contract.get("dataset") != dataset.lower():
        raise ValueError(
            f"Precomputed file contract belongs to {file_contract.get('dataset')!r}, "
            f"not {dataset!r}"
        )
    external_histories, external_catalog = _read_pctm_train(split_dir / "train.csv")
    external_test = _read_pctm_holdout(split_dir / "holdout.csv")
    data_contract = build_evaluation_contract(
        file_contract, external_histories, external_test, external_catalog
    )

    external_items = sorted(external_catalog)
    item_map = {item: index for index, item in enumerate(external_items)}
    external_users = sorted(external_histories)
    user_map = {user: index for index, user in enumerate(external_users)}
    full_histories = {
        user_map[user]: [item_map[item] for item in external_histories[user]]
        for user in external_users
    }

    inner_item_counts = [0] * len(external_items)
    train_by_user: dict[int, list[int]] = {}
    for user, full_history in full_histories.items():
        inner_history = full_history[:-1]
        train_by_user[user] = inner_history
        for item in inner_history:
            inner_item_counts[item] += 1
    train_candidates = [item for item, count in enumerate(inner_item_counts) if count]

    valid: dict[int, int] = {}
    for user, full_history in full_histories.items():
        if len(full_history) < 2:
            continue
        target, history = full_history[-1], train_by_user[user]
        if inner_item_counts[target] and target not in history:
            valid[user] = target

    test: dict[int, int] = {}
    for external_user, external_item in external_test.items():
        if external_user not in user_map:
            raise ValueError(f"Cold outer holdout user_id={external_user} in {split_dir}")
        if external_item not in item_map:
            raise ValueError(f"Cold outer holdout item_id={external_item} in {split_dir}")
        user, item = user_map[external_user], item_map[external_item]
        if item in full_histories[user]:
            raise ValueError(
                f"Already-seen outer holdout pair user_id={external_user}, "
                f"item_id={external_item} in {split_dir}"
            )
        test[user] = item

    metadata = (
        _pctm_metadata_texts(split_dir, metadata_path)
        if item_text_mode == "metadata"
        else {}
    )
    item_texts = [metadata.get(str(item), str(item)) for item in external_items]
    return InteractionData(
        train_by_user=train_by_user,
        valid_target=valid,
        test_target=test,
        item_texts=item_texts,
        num_users=len(external_users),
        num_items=len(external_items),
        test_history_by_user=full_histories,
        train_candidate_items=train_candidates,
        valid_candidate_items=train_candidates,
        data_contract=data_contract,
    )


def load_pctm_movielens20m(
    data_path: str | None,
    cache_dir: str,
    *,
    verify_split: bool = True,
    item_text_mode: str = "id",
    metadata_path: str | None = None,
) -> InteractionData:
    """Compatibility wrapper for the original ML-20M dataset name."""
    return load_protocol_recommendation_data(
        PCTM_ML20M_DATASET,
        data_path,
        cache_dir,
        verify_split=verify_split,
        item_text_mode=item_text_mode,
        metadata_path=metadata_path,
    )


def pctm_outer_train_data(data: InteractionData) -> InteractionData:
    """Create the refit view that restores all outer-train interactions."""
    if data.test_history_by_user is None:
        raise ValueError("Outer-train refit is only available for a protocol-aware dataset")
    return InteractionData(
        train_by_user=data.test_history_by_user,
        valid_target={},
        test_target=data.test_target,
        item_texts=data.item_texts,
        num_users=data.num_users,
        num_items=data.num_items,
        test_history_by_user=data.test_history_by_user,
        data_contract=data.data_contract,
    )


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
    pctm_verify_split: bool = True,
    pctm_item_text_mode: str = "id",
    pctm_metadata_path: str | None = None,
    pctm_file_contract: dict[str, object] | None = None,
) -> InteractionData:
    name = dataset.lower()
    if name == "synthetic":
        events = synthetic_events()
    elif is_protocol_dataset(name):
        if max_events is not None:
            raise ValueError(
                f"--max-events is incompatible with the frozen {name} protocol"
            )
        return load_protocol_recommendation_data(
            name,
            data_path,
            cache_dir,
            verify_split=pctm_verify_split,
            item_text_mode=pctm_item_text_mode,
            metadata_path=pctm_metadata_path,
            file_contract=pctm_file_contract,
        )
    elif name in MOVIELENS_URLS:
        events = load_movielens(name, data_path, cache_dir, max_events, min_rating)
    elif name.startswith("amazon-") or name.startswith("amazon:"):
        events = load_amazon(
            amazon_subset(dataset), max_events, cache_dir, item_group=amazon_item_group(dataset)
        )
    elif name in {"yelp19", "yelp-2019", "yelp23", "yelp-2023"}:
        events = load_yelp(data_path, max_events, min_rating)
    else:
        supported = ", ".join(
            (
                *MOVIELENS_URLS,
                *sorted(PROTOCOL_DATASETS),
                "amazon-<category>",
                "yelp19",
                "yelp23",
                "synthetic",
            )
        )
        raise ValueError(f"Unknown dataset {dataset!r}. Supported values: {supported}")
    if not events:
        raise RuntimeError(f"No positive interactions loaded for {dataset!r}; check --min-rating and --data-path")
    return build_data(
        events,
        min_user_events=min_user_events,
        sasrec_filtering=sasrec_filtering,
    )
