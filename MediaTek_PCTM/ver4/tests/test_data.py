import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.data import build_data, edge_index, synthetic_events
from src.data_contract import (
    PROTOCOL_DATASETS,
    assert_same_data_contract,
    sha256_and_rows,
    verify_protocol_split,
)
from src.data_mamba_rl import (
    load_protocol_recommendation_data,
    load_pctm_movielens20m,
    pctm_outer_train_data,
    verify_pctm_ml20m_split,
)


class DataTest(unittest.TestCase):
    def test_protocol_registry_covers_requested_datasets(self):
        self.assertTrue(
            {
                "amazon-beauty-pctm",
                "amazon-sports-pctm",
                "amazon-toys-pctm",
                "amazon-video-games-pctm",
                "movielens-1m-pctm",
            }.issubset(PROTOCOL_DATASETS)
        )

    def test_chronological_split(self):
        data = build_data(synthetic_events())
        self.assertEqual(data.num_users, 24)
        self.assertTrue(all(len(history) == 3 for history in data.train_by_user.values()))

    def test_sasrec_filtering_matches_one_pass_five_core_and_split(self):
        events = []
        common_items = [f"i{index}" for index in range(5)]
        for user_index in range(5):
            user = f"u{user_index}"
            for timestamp, item in enumerate(common_items):
                events.append((user, item, timestamp, item))
            events.append((user, f"rare_{user}", 10, f"rare_{user}"))

        # Passes the raw user threshold, but only two interactions survive the
        # item filter, so SASRec keeps this user for training only.
        events.extend([
            ("u_short", "i0", 0, "i0"),
            ("u_short", "i1", 1, "i1"),
            ("u_short", "short_rare_0", 2, "short_rare_0"),
            ("u_short", "short_rare_1", 3, "short_rare_1"),
            ("u_short", "short_rare_2", 4, "short_rare_2"),
        ])
        # Four raw interactions are insufficient despite using common items.
        events.extend(("u_low", item, index, item) for index, item in enumerate(common_items[:4]))

        data = build_data(events, min_user_events=5, sasrec_filtering=True)

        self.assertEqual(data.num_users, 6)
        self.assertEqual(data.num_items, 5)
        self.assertEqual(len(data.valid_target), 5)
        self.assertEqual(len(data.test_target), 5)
        training_only_users = set(data.train_by_user) - set(data.valid_target)
        self.assertEqual(len(training_only_users), 1)
        self.assertEqual(len(data.train_by_user[training_only_users.pop()]), 2)

    def test_pctm_outer_split_builds_filtered_inner_loo_and_full_test_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ["user_id", "item_id", "datetime", "rating", "weight"]
            train_rows = [
                (1, 20, "1970-01-01 00:00:00.000000002", 3.0, 1),
                (1, 10, "1970-01-01 00:00:00.000000001", 1.0, 1),
                (1, 30, "1970-01-01 00:00:00.000000003", 5.0, 1),
                (2, 10, "2020-01-01", 2.0, 1),
                (2, 40, "2020-01-02", 4.0, 1),
                (2, 20, "2020-01-03", 1.0, 1),
                (3, 50, "2020-01-01", 0.5, 1),
                (3, 60, "2020-01-02", 5.0, 1),
            ]
            holdout_rows = [
                (1, 40, "2020-01-04", 5.0, 1),
                (2, 30, "2020-01-04", 5.0, 1),
                (3, 10, "2020-01-03", 5.0, 1),
            ]
            for filename, rows in (("train.csv", train_rows), ("holdout.csv", holdout_rows)):
                with (root / filename).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(rows)

            data = load_pctm_movielens20m(
                str(root), str(root / "cache"), verify_split=False
            )

            # External item IDs are indexed in ascending order: 10..60.
            self.assertEqual(data.train_by_user[0], [0, 1])
            self.assertEqual(data.train_by_user[1], [0, 3])
            self.assertEqual(data.train_by_user[2], [4])
            # Only user 2 has a warm, not-already-seen inner target.
            self.assertEqual(data.valid_target, {1: 1})
            self.assertEqual(data.valid_candidate_items, [0, 1, 3, 4])
            self.assertEqual(data.train_candidate_items, [0, 1, 3, 4])
            self.assertEqual(data.test_target, {0: 3, 1: 2, 2: 0})
            self.assertEqual(data.test_history_by_user[0], [0, 1, 2])

            refit = pctm_outer_train_data(data)
            self.assertIs(refit.train_by_user, data.test_history_by_user)
            self.assertEqual(refit.train_by_user[1], [0, 3, 1])
            self.assertEqual(refit.valid_target, {})
            with self.assertRaisesRegex(RuntimeError, "dataset was not modified"):
                verify_pctm_ml20m_split(root)

    def test_custom_video_games_contract_is_required_verified_and_semantic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ["user_id", "item_id", "datetime", "weight"]
            train_rows = [
                (1, 10, 1, 1),
                (1, 20, 2, 1),
                (2, 20, 1, 1),
                (2, 30, 2, 1),
            ]
            holdout_rows = [(1, 30, 3, 1), (2, 10, 3, 1)]
            for filename, rows in (("train.csv", train_rows), ("holdout.csv", holdout_rows)):
                with (root / filename).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(rows)

            with self.assertRaisesRegex(FileNotFoundError, "requires data_contract.json"):
                verify_protocol_split("amazon-video-games-pctm", root)

            files = {}
            for filename in ("train.csv", "holdout.csv"):
                digest, rows = sha256_and_rows(root / filename)
                files[filename] = {"sha256": digest, "rows": rows}
            (root / "data_contract.json").write_text(
                json.dumps(
                    {"dataset": "amazon-video-games-pctm", "files": files}, indent=2
                ),
                encoding="utf-8",
            )

            verified = verify_protocol_split("amazon-video-games-pctm", root)
            self.assertEqual(verified["verification"], "custom-sidecar")
            data = load_protocol_recommendation_data(
                "amazon-video-games-pctm",
                str(root),
                str(root / "cache"),
                verify_split=False,
                file_contract=verified,
            )
            self.assertEqual(data.data_contract["verification"], "custom-sidecar")
            self.assertEqual(data.data_contract["evaluation"]["eligible_test_users"], 2)
            self.assertEqual(data.data_contract["evaluation"]["catalogue_items"], 3)
            self.assertEqual(len(data.data_contract["evaluation_contract_sha256"]), 64)

            with (root / "holdout.csv").open("a", encoding="utf-8") as stream:
                stream.write("3,10,4,1\n")
            with self.assertRaisesRegex(RuntimeError, "experiment was refused"):
                verify_protocol_split("amazon-video-games-pctm", root)

    def test_data_contract_comparison_fails_closed(self):
        contract = {"evaluation_contract_sha256": "a" * 64}
        assert_same_data_contract(contract, dict(contract), context="unit test")
        with self.assertRaisesRegex(RuntimeError, "Data contract mismatch"):
            assert_same_data_contract(
                contract,
                {"evaluation_contract_sha256": "b" * 64},
                context="unit test",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_edges(self):
        data = build_data(synthetic_events())
        edges = edge_index(data)
        self.assertEqual(edges.shape[0], 2)
        self.assertGreater(edges.shape[1], 0)
