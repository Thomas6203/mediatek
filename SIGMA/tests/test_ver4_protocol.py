from __future__ import annotations

import importlib.util

import torch
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / "vendor").exists():
    sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT.parent / "MediaTek_2026_0902" / "ver4"))

from prepare_ver4_data import build_id_only_ver4_split, statistics
from src.data import build_data


class MatchedPreprocessingTest(unittest.TestCase):
    def test_id_only_implementation_matches_ver4(self):
        events = []
        common = [f"i{index}" for index in range(5)]
        for user_index in range(5):
            for timestamp, item in enumerate(common):
                events.append((f"u{user_index}", item, timestamp, item))
            events.append((f"u{user_index}", f"rare_{user_index}", 10, "unused"))
        events.extend(
            ("u_short", item, index, item) for index, item in enumerate(common[:2])
        )
        events.extend(
            ("u_short", f"rare_short_{index}", index + 2, "unused")
            for index in range(3)
        )

        rows = [
            {"user_id": user, "parent_asin": item, "timestamp": timestamp}
            for user, item, timestamp, _ in events
        ]
        expected = build_data(events, min_user_events=5, sasrec_filtering=True)
        actual = build_id_only_ver4_split(rows)
        self.assertEqual(statistics(actual), statistics(expected))
        self.assertEqual(actual.train_by_user, expected.train_by_user)
        self.assertEqual(actual.valid_target, expected.valid_target)
        self.assertEqual(actual.test_target, expected.test_target)

    @unittest.skipUnless(
        importlib.util.find_spec("mamba_ssm") and torch.cuda.is_available(), "CUDA mamba_ssm is unavailable"
    )
    def test_model_full_catalog_shape(self):
        from model.sigma_ver4_protocol import SIGMA

        model = SIGMA(
            17, max_history=50, hidden_size=8, d_state=4, d_conv=2, expand=1
        ).cuda()
        sequence = torch.zeros((2, 50), dtype=torch.long, device="cuda")
        sequence[0, :3] = torch.tensor([1, 2, 3], device="cuda")
        sequence[1, :2] = torch.tensor([4, 5], device="cuda")
        scores = model.full_sort_scores(sequence, torch.tensor([3, 2], device="cuda"))
        self.assertEqual(scores.shape, (2, 17))


if __name__ == "__main__":
    unittest.main()

