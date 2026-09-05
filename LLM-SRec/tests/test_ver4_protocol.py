from __future__ import annotations

import unittest
from types import SimpleNamespace

from SeqRec.sasrec.utils import data_partition
from ver4_protocol import DATASETS, ROOT, build_transitions, load_verified_data, statistics


class Ver4ProtocolTest(unittest.TestCase):
    def test_canonical_all_beauty_contract(self):
        data, manifest = load_verified_data("all_beauty")
        self.assertEqual(statistics(data), DATASETS["all_beauty"]["expected"])
        self.assertEqual(
            manifest["split_sha256"],
            "0a0349ae56194bbfa44cee2d21feee3a0983ef83ea8bbc06147033295ac14ad4",
        )

    def test_coverage_first_sampler_is_deterministic(self):
        data = SimpleNamespace(
            train_by_user={0: [0, 1, 2], 1: [1, 2, 3], 2: [3, 4]},
        )
        first = build_transitions(data, maximum=3, seed=25252)
        second = build_transitions(data, maximum=3, seed=25252)
        self.assertEqual(first, second)
        self.assertEqual({row.user for row in first}, {0, 1, 2})

    def test_exported_files_are_accepted_by_upstream_loader(self):
        dataset = "all_beauty"
        prefix = ROOT / "SeqRec" / f"data_{dataset}" / dataset
        partition = data_partition(
            dataset,
            SimpleNamespace(dataset=dataset),
            path=str(prefix),
        )
        train, valid, test, users, items, _ = partition
        expected = DATASETS[dataset]["expected"]
        self.assertEqual(users, expected["num_users"])
        self.assertEqual(items, expected["num_items"])
        self.assertEqual(sum(map(len, train.values())), expected["train_interactions"])
        self.assertEqual(len(valid), expected["evaluation_users"])
        self.assertEqual(len(test), expected["evaluation_users"])


if __name__ == "__main__":
    unittest.main()
