from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / "vendor").exists():
    sys.path.insert(0, str(ROOT / "vendor"))

from run_ver4_protocol import stable_full_sort_scores


@unittest.skipUnless(
    importlib.util.find_spec("mamba_ssm") and torch.cuda.is_available(),
    "CUDA is unavailable",
)
class StableCatalogScoringTest(unittest.TestCase):
    def test_catalog_projection_is_float32_and_finite(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.item_embedding = nn.Embedding(3, 64)
                self.item_embedding.weight.data.fill_(500.0)

            def encode(self, sequences, lengths):
                return torch.full(
                    (sequences.size(0), 64),
                    500.0,
                    dtype=torch.float16,
                    device=sequences.device,
                )

        model = Dummy().cuda()
        sequences = torch.ones((2, 2), dtype=torch.long, device="cuda")
        lengths = torch.full((2,), 2, dtype=torch.long, device="cuda")
        scores = stable_full_sort_scores(model, sequences, lengths, "cuda")
        self.assertEqual(scores.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
