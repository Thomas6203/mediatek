import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from time_control import (
    RunTimer,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_path,
    restore_rng_state,
)


class RunTimerTests(unittest.TestCase):
    def test_hourly_checkpoint_uses_latest_crossed_boundary(self):
        clock = [100.0]
        with patch("time_control.time.monotonic", side_effect=lambda: clock[0]):
            timer = RunTimer(max_hours=0, checkpoint_interval_hours=1)
            clock[0] = 3 * 3600 + 101
            self.assertTrue(timer.checkpoint_due)
            self.assertEqual(timer.consume_checkpoint_index(), 3)
            self.assertFalse(timer.checkpoint_due)

    def test_resume_preserves_cumulative_checkpoint_number(self):
        clock = [50.0]
        with patch("time_control.time.monotonic", side_effect=lambda: clock[0]):
            timer = RunTimer(
                max_hours=2,
                checkpoint_interval_hours=1,
                previous_elapsed_seconds=3900,
            )
            clock[0] = 3351.0
            self.assertTrue(timer.checkpoint_due)
            self.assertEqual(timer.consume_checkpoint_index(), 2)

    def test_shutdown_buffer_stops_before_session_limit(self):
        clock = [20.0]
        with patch("time_control.time.monotonic", side_effect=lambda: clock[0]):
            timer = RunTimer(
                max_hours=2,
                checkpoint_interval_hours=1,
                shutdown_buffer_minutes=2,
            )
            clock[0] = 20 + 7079
            self.assertFalse(timer.time_limit_reached)
            clock[0] = 20 + 7080
            self.assertTrue(timer.time_limit_reached)

    def test_atomic_checkpoint_and_rng_round_trip(self):
        original = capture_rng_state()
        try:
            random.seed(77)
            np.random.seed(77)
            torch.manual_seed(77)
            saved = capture_rng_state()
            expected = (random.random(), np.random.random(), torch.rand(1).item())
            restore_rng_state(saved)
            actual = (random.random(), np.random.random(), torch.rand(1).item())
            self.assertEqual(expected, actual)

            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                clock = [0.0]
                with patch("time_control.time.monotonic", side_effect=lambda: clock[0]):
                    timer = RunTimer(0, 1)
                    clock[0] = 3601.0
                    path = checkpoint_path(run_dir, timer, stopped=False)
                atomic_torch_save({"value": 7}, path)
                self.assertEqual(torch.load(path, weights_only=False)["value"], 7)
                self.assertFalse(path.with_suffix(".pt.tmp").exists())
        finally:
            restore_rng_state(original)


if __name__ == "__main__":
    unittest.main()
