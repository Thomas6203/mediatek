from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

import torch


SASREC_ROOT = Path(__file__).resolve().parents[1]
VER4_ROOT = SASREC_ROOT.parents[1] / "ver4"
for path in (SASREC_ROOT, VER4_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model import SASRec
from src.data import build_data, synthetic_events
from train import evaluate, sample_training_batch


class SASRecTest(unittest.TestCase):
    def setUp(self):
        self.data = build_data(synthetic_events(), min_user_events=5, sasrec_filtering=True)

    def test_forward_shapes_and_finite_logits(self):
        model = SASRec(
            num_items=self.data.num_items,
            maxlen=4,
            hidden_units=8,
            num_blocks=1,
            num_heads=1,
            dropout_rate=0.0,
        )
        sequences = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 4]])
        positive = torch.tensor([[0, 2, 3, 4], [0, 0, 4, 5]])
        negative = torch.tensor([[0, 4, 5, 1], [0, 0, 1, 2]])
        positive_logits, negative_logits = model(sequences, positive, negative)
        self.assertEqual(positive_logits.shape, sequences.shape)
        self.assertEqual(negative_logits.shape, sequences.shape)
        self.assertTrue(torch.isfinite(positive_logits).all())
        self.assertEqual(model(sequences).shape, (2, self.data.num_items))

    def test_sampler_uses_padding_zero_and_shifted_item_ids(self):
        sequences, positives, negatives = sample_training_batch(
            self.data, batch_size=8, maxlen=4, rng=random.Random(1), device="cpu"
        )
        self.assertEqual(sequences.shape, (8, 4))
        self.assertTrue(torch.all(sequences >= 0))
        self.assertTrue(torch.all(sequences <= self.data.num_items))
        self.assertTrue(torch.all(positives[sequences == 0] == 0))
        active = positives > 0
        self.assertTrue(torch.all(negatives[active] > 0))
        self.assertTrue(torch.all(positives[active] != negatives[active]))

    def test_full_catalog_evaluation_protocol(self):
        model = SASRec(
            num_items=self.data.num_items,
            maxlen=4,
            hidden_units=8,
            num_blocks=1,
            num_heads=1,
            dropout_rate=0.0,
        )
        metrics = evaluate(model, self.data, "test", batch_size=8, maxlen=4, device="cpu")
        self.assertEqual(metrics["recall@5"], metrics["hit@5"])
        self.assertEqual(metrics["recall@10"], metrics["hit@10"])
        self.assertEqual(metrics["evaluated_users"], len(self.data.test_target))


if __name__ == "__main__":
    unittest.main()
