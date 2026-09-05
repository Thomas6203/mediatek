import importlib.util
import unittest

from src.data import build_data, edge_index, synthetic_events


class DataTest(unittest.TestCase):
    def test_chronological_split(self):
        data = build_data(synthetic_events())
        self.assertEqual(data.num_users, 24)
        self.assertTrue(all(len(history) == 3 for history in data.train_by_user.values()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_edges(self):
        data = build_data(synthetic_events())
        edges = edge_index(data)
        self.assertEqual(edges.shape[0], 2)
        self.assertGreater(edges.shape[1], 0)
