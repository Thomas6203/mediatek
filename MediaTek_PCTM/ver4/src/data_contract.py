"""Fail-closed data contracts for protocol-aware recommendation datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DATA_CONTRACT_SCHEMA_VERSION = 1

PROTOCOL_DATASETS: dict[str, dict[str, object]] = {
    "amazon-beauty-pctm": {
        "directory": "s3_beauty",
        "max_history": 50,
        "files": {
            "train.csv": {
                "rows": 176_139,
                "sha256": "956357e585c8c4ddf50ff4dd4331d7e046c1b42b8eaa13cc21c8b9dccee89795",
            },
            "holdout.csv": {
                "rows": 22_311,
                "sha256": "38dbedf2dedc6951be1d8a3d4f6b2d93aa7129ce5d4f0fcf0d17bee0e315f43a",
            },
        },
    },
    "amazon-sports-pctm": {
        "directory": "s3_sports",
        "max_history": 50,
        "files": {
            "train.csv": {
                "rows": 260_739,
                "sha256": "e90df6401241237709a9386d91fd9c2f24eb132665e99e74d6f5af5e12b8b9a6",
            },
            "holdout.csv": {
                "rows": 35_539,
                "sha256": "2fb358ca824c8df1e655275ba13d106b1c1b2dc1bb131810cc8519919abc1139",
            },
        },
    },
    "amazon-toys-pctm": {
        "directory": "s3_toys",
        "max_history": 50,
        "files": {
            "train.csv": {
                "rows": 148_185,
                "sha256": "ca9191ea48da1bc4b9ed812354ae3b58a7b93f2417d4794142c9c9dc0230b356",
            },
            "holdout.csv": {
                "rows": 19_365,
                "sha256": "ff18781f3e2784b1d4438bdd5a54be7fb75aa0f2d7d08f552271f135bf1ee7b8",
            },
        },
    },
    "movielens-1m-pctm": {
        "directory": "ml_1m",
        "max_history": 200,
        "files": {
            "train.csv": {
                "rows": 994_169,
                "sha256": "74afd9fcafba3e694195fe26c8ea486ec12dd16f22f0aaaa88cbcd66c960176a",
            },
            "holdout.csv": {
                "rows": 6_038,
                "sha256": "83bf1681a6dc94a5e567fcca9bd5de44618ff4097f9a2172df840bdbcb15f478",
            },
        },
    },
    "movielens-20m-pctm": {
        "directory": "ml_20m",
        "max_history": 200,
        "files": {
            "train.csv": {
                "rows": 19_861_770,
                "sha256": "5392d0bd6cb40724a141f0419fd039064058f7a154f9a8c31d9659a221c16f82",
            },
            "holdout.csv": {
                "rows": 138_456,
                "sha256": "b5da7afee4b9d24b7e6f6bcaea33d7d961ad63db19cf9a8df8faf63563e676d7",
            },
        },
    },
    # Video Games is an extension, not a paper dataset.  Its expected bytes
    # must be frozen explicitly in data_contract.json next to the split.
    "amazon-video-games-pctm": {
        "directory": "s3_video_games",
        "max_history": 50,
        "files": None,
    },
}

PCTM_ML20M_DATASET = "movielens-20m-pctm"


def is_protocol_dataset(dataset: str) -> bool:
    return dataset.lower() in PROTOCOL_DATASETS


def protocol_dataset_spec(dataset: str) -> dict[str, object]:
    try:
        return PROTOCOL_DATASETS[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"Not a protocol-aware dataset: {dataset!r}") from error


def protocol_max_history(dataset: str) -> int:
    return int(protocol_dataset_spec(dataset)["max_history"])


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_and_rows(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    last_byte = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            lines += block.count(b"\n")
            last_byte = block[-1:]
    # CSV files with or without a final newline have the same logical row
    # count.  Subtract the header after accounting for the final line.
    physical_lines = lines if not last_byte or last_byte == b"\n" else lines + 1
    return digest.hexdigest(), max(physical_lines - 1, 0)


def resolve_protocol_split(
    dataset: str, data_path: str | None, cache_dir: str
) -> Path:
    """Resolve a train/holdout directory without creating or modifying it."""
    spec = protocol_dataset_spec(dataset)
    directory = str(spec["directory"])
    roots: list[Path] = []
    if data_path:
        roots.append(Path(data_path).expanduser())
    else:
        cache_parent = Path(cache_dir).expanduser().resolve().parent
        roots.extend(
            (
                cache_parent / "sequential-capacity-probes",
                Path.cwd().parent / "sequential-capacity-probes",
                Path.cwd() / "sequential-capacity-probes",
            )
        )
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root,
                root / "leave_one_out",
                root / directory / "leave_one_out",
                root / "data" / "processed" / directory / "leave_one_out",
            )
        )
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (
            candidate / "holdout.csv"
        ).is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {directory}/leave_one_out train.csv and holdout.csv for "
        f"{dataset}. Checked:\n  {checked}"
    )


def _validate_file_details(filename: str, details: object) -> dict[str, object]:
    if not isinstance(details, dict):
        raise ValueError(f"Invalid contract entry for {filename}: expected an object")
    try:
        rows = int(details["rows"])
        sha256 = str(details["sha256"]).lower()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid contract entry for {filename}: rows and sha256 are required"
        ) from error
    if rows < 0 or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError(f"Invalid rows/SHA-256 in contract entry for {filename}")
    return {"rows": rows, "sha256": sha256}


def _sidecar_expectations(dataset: str, split_dir: Path) -> dict[str, dict[str, object]]:
    candidates = (split_dir / "data_contract.json", split_dir.parent / "data_contract.json")
    sidecar = next((path for path in candidates if path.is_file()), None)
    if sidecar is None:
        raise FileNotFoundError(
            f"{dataset} is a custom benchmark and requires data_contract.json next to "
            f"the split. Expected one of: {', '.join(str(path) for path in candidates)}"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    declared = payload.get("dataset")
    if declared and str(declared).lower() != dataset.lower():
        raise ValueError(
            f"Dataset identity mismatch in {sidecar}: {declared!r} != {dataset!r}"
        )
    files = payload.get("files", payload)
    return {
        filename: _validate_file_details(filename, files.get(filename))
        for filename in ("train.csv", "holdout.csv")
    }


def inspect_protocol_split(dataset: str, split_dir: Path) -> dict[str, object]:
    """Hash the actual split bytes and return a path-independent contract."""
    spec = protocol_dataset_spec(dataset)
    files: dict[str, dict[str, object]] = {}
    for filename in ("train.csv", "holdout.csv"):
        digest, rows = sha256_and_rows(split_dir / filename)
        files[filename] = {"rows": rows, "sha256": digest}
    base: dict[str, object] = {
        "schema_version": DATA_CONTRACT_SCHEMA_VERSION,
        "dataset": dataset.lower(),
        "directory": str(spec["directory"]),
        "files": files,
    }
    base["file_contract_sha256"] = _canonical_sha256(base)
    return base


def verify_protocol_split(dataset: str, split_dir: Path) -> dict[str, object]:
    """Reject a split that differs from the official or frozen custom contract."""
    spec = protocol_dataset_spec(dataset)
    expected_raw = spec["files"]
    expected = (
        {
            filename: _validate_file_details(filename, details)
            for filename, details in expected_raw.items()
        }
        if isinstance(expected_raw, dict)
        else _sidecar_expectations(dataset, split_dir)
    )
    actual = inspect_protocol_split(dataset, split_dir)
    for filename, wanted in expected.items():
        observed = actual["files"][filename]
        if observed != wanted:
            raise RuntimeError(
                f"Protocol split verification failed for {split_dir / filename}: "
                f"rows={observed['rows']} sha256={observed['sha256']}; expected "
                f"rows={wanted['rows']} sha256={wanted['sha256']}. the dataset was "
                "not modified and the experiment was refused."
            )
    actual["verification"] = "official-pinned" if isinstance(expected_raw, dict) else "custom-sidecar"
    return actual


def build_evaluation_contract(
    file_contract: Mapping[str, object],
    histories: Mapping[object, Sequence[object]],
    targets: Mapping[object, object],
    catalogue: Iterable[object],
) -> dict[str, object]:
    """Fingerprint the exact external-ID evaluation population after loading."""
    external_targets = sorted(
        ((str(user), str(item)) for user, item in targets.items()),
        key=lambda row: (row[0], row[1]),
    )
    external_histories = sorted(
        (
            (str(user), [str(item) for item in histories[user]])
            for user in targets
        ),
        key=lambda row: row[0],
    )
    external_catalogue = sorted(str(item) for item in catalogue)
    semantic = {
        "eligible_test_users": len(external_targets),
        "catalogue_items": len(external_catalogue),
        "targets_sha256": _canonical_sha256(external_targets),
        "histories_sha256": _canonical_sha256(external_histories),
        "catalogue_sha256": _canonical_sha256(external_catalogue),
        "protocol": {"full_catalogue": True, "filter_seen": True, "k": 10},
    }
    combined = {
        "schema_version": DATA_CONTRACT_SCHEMA_VERSION,
        "dataset": file_contract["dataset"],
        "directory": file_contract["directory"],
        "files": file_contract["files"],
        "file_contract_sha256": file_contract["file_contract_sha256"],
        "verification": file_contract.get("verification", "inspection-only"),
        "evaluation": semantic,
    }
    combined["evaluation_contract_sha256"] = _canonical_sha256(combined)
    return combined


def assert_same_data_contract(expected: object, actual: object, *, context: str) -> None:
    """Fail closed when a resume/result belongs to another evaluation population."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise RuntimeError(f"Missing data contract while {context}; refusing to continue")
    left = expected.get("evaluation_contract_sha256")
    right = actual.get("evaluation_contract_sha256")
    if not left or left != right:
        raise RuntimeError(
            f"Data contract mismatch while {context}: stored={left!r} current={right!r}; "
            "refusing to continue"
        )
