import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = PROJECT_ROOT / "summarize_protocol_results.py"


class SummarizeProtocolResultsTest(unittest.TestCase):
    def write_run(self, root: Path, seed: int, *, candidates: int = 64) -> None:
        run_dir = root / "amazon-beauty-pctm" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        contract = {
            "dataset": "amazon-beauty-pctm",
            "evaluation_contract_sha256": "a" * 64,
        }
        (run_dir / "run_status.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "data_contract": contract,
                    "test": {"ndcg@10": 0.07 + seed / 1_000_000, "recall@10": 0.12},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "amazon-beauty-pctm_scores.json").write_text(
            json.dumps(
                {
                    "config": {
                        "dataset": "amazon-beauty-pctm",
                        "candidates": candidates,
                        "seed": seed,
                        "output_run_dir": str(run_dir),
                        "run_hours": 6,
                    }
                }
            ),
            encoding="utf-8",
        )

    def run_summarizer(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SUMMARIZER),
                str(root),
                "--require-runs",
                "2",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_aggregates_seed_only_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(root, 25252)
            self.write_run(root, 25253)
            result = self.run_summarizer(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("amazon-beauty-pctm,2,", result.stdout)

    def test_rejects_mixed_training_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(root, 25252, candidates=64)
            self.write_run(root, 25253, candidates=128)
            result = self.run_summarizer(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixed training configurations", result.stderr)


if __name__ == "__main__":
    unittest.main()
