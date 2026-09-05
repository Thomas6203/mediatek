#!/usr/bin/env python3
"""Verify a frozen split or compare contracts embedded in result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_contract import (
    assert_same_data_contract,
    inspect_protocol_split,
    protocol_dataset_spec,
    resolve_protocol_split,
)
from src.data_mamba_rl import load_protocol_recommendation_data


def _contract_from(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("data_contract", payload)
    if not isinstance(contract, dict) or "evaluation_contract_sha256" not in contract:
        raise ValueError(f"No evaluation data contract found in {path}")
    return contract


def verify(args: argparse.Namespace) -> int:
    data = load_protocol_recommendation_data(
        args.dataset,
        args.data_path,
        args.cache_dir,
        verify_split=True,
        item_text_mode="id",
    )
    if data.data_contract is None:
        raise RuntimeError("Protocol loader did not produce a data contract")
    rendered = json.dumps(data.data_contract, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"wrote verified contract to {output}")
    else:
        print(rendered, end="")
    print(
        "VERIFIED "
        f"dataset={args.dataset} "
        f"sha256={data.data_contract['evaluation_contract_sha256']}"
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    reference_path = Path(args.reference)
    reference = _contract_from(reference_path)
    for candidate_name in args.candidates:
        candidate_path = Path(candidate_name)
        candidate = _contract_from(candidate_path)
        assert_same_data_contract(
            reference, candidate, context=f"comparing {reference_path} and {candidate_path}"
        )
    print(
        "MATCH "
        f"files={1 + len(args.candidates)} "
        f"sha256={reference['evaluation_contract_sha256']}"
    )
    return 0


def freeze_custom(args: argparse.Namespace) -> int:
    spec = protocol_dataset_spec(args.dataset)
    if spec["files"] is not None:
        raise ValueError(
            f"{args.dataset} already has an official pinned contract; refusing to replace it"
        )
    split_dir = resolve_protocol_split(args.dataset, args.data_path, args.cache_dir)
    inspected = inspect_protocol_split(args.dataset, split_dir)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset.lower(),
        "files": inspected["files"],
    }
    output = Path(args.output).expanduser().resolve()
    allowed = {
        (split_dir / "data_contract.json").resolve(),
        (split_dir.parent / "data_contract.json").resolve(),
    }
    if output not in allowed:
        raise ValueError(
            "Custom contract must be written next to the split so runtime verification "
            f"can find it. Allowed: {', '.join(sorted(str(path) for path in allowed))}"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing contract: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"FROZEN dataset={args.dataset} output={output} "
        f"file_contract_sha256={inspected['file_contract_sha256']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verifier = subparsers.add_parser("verify", help="verify and fingerprint one split")
    verifier.add_argument("--dataset", required=True)
    verifier.add_argument("--data-path", required=True)
    verifier.add_argument("--cache-dir", default="cache")
    verifier.add_argument("--output")
    verifier.set_defaults(handler=verify)

    comparator = subparsers.add_parser(
        "compare", help="require result/contract JSON files to name the same data"
    )
    comparator.add_argument("reference")
    comparator.add_argument("candidates", nargs="+")
    comparator.set_defaults(handler=compare)

    freezer = subparsers.add_parser(
        "freeze-custom",
        help="create a non-overwriting sidecar for a reviewed custom split",
    )
    freezer.add_argument("--dataset", required=True)
    freezer.add_argument("--data-path", required=True)
    freezer.add_argument("--cache-dir", default="cache")
    freezer.add_argument("--output", required=True)
    freezer.set_defaults(handler=freeze_custom)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
