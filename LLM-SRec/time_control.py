"""Wall-clock budgets and resumable checkpoint helpers for ver4 runners."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch


class RunTimer:
    def __init__(
        self,
        max_hours: float,
        checkpoint_interval_hours: float,
        previous_elapsed_seconds: float = 0.0,
        shutdown_buffer_minutes: float = 2.0,
    ) -> None:
        if max_hours < 0:
            raise ValueError("max_hours must be non-negative")
        if checkpoint_interval_hours <= 0:
            raise ValueError("checkpoint_interval_hours must be positive")
        if shutdown_buffer_minutes < 0:
            raise ValueError("shutdown_buffer_minutes must be non-negative")
        self.started = time.monotonic()
        self.previous_elapsed_seconds = float(previous_elapsed_seconds)
        self.session_budget_seconds = max_hours * 3600 if max_hours else math.inf
        requested_buffer = shutdown_buffer_minutes * 60
        self.shutdown_buffer_seconds = min(
            requested_buffer,
            self.session_budget_seconds * 0.1 if math.isfinite(self.session_budget_seconds) else requested_buffer,
        )
        self.interval_seconds = checkpoint_interval_hours * 3600
        self.next_checkpoint_index = (
            math.floor(self.previous_elapsed_seconds / self.interval_seconds) + 1
        )

    @property
    def session_elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def total_elapsed_seconds(self) -> float:
        return self.previous_elapsed_seconds + self.session_elapsed_seconds

    @property
    def time_limit_reached(self) -> bool:
        return self.session_elapsed_seconds >= (
            self.session_budget_seconds - self.shutdown_buffer_seconds
        )

    @property
    def checkpoint_due(self) -> bool:
        return self.total_elapsed_seconds >= self.next_checkpoint_index * self.interval_seconds

    def consume_checkpoint_index(self) -> int:
        index = math.floor(self.total_elapsed_seconds / self.interval_seconds)
        self.next_checkpoint_index = index + 1
        return index


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_path(run_dir: Path, timer: RunTimer, stopped: bool) -> Path:
    if timer.checkpoint_due:
        index = timer.consume_checkpoint_index()
        name = f"checkpoint_hour_{index:04d}.pt"
    elif stopped:
        minutes = round(timer.total_elapsed_seconds / 60)
        name = f"checkpoint_stop_{minutes:06d}m.pt"
    else:
        name = "checkpoint_latest.pt"
    return run_dir / "checkpoints" / name
