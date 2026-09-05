import importlib.util
import unittest

from src.data import build_data, edge_index, synthetic_events


class DataTest(unittest.TestCase):
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_edges(self):
        data = build_data(synthetic_events())
        edges = edge_index(data)
        self.assertEqual(edges.shape[0], 2)
        self.assertGreater(edges.shape[1], 0)
