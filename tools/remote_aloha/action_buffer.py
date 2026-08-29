from __future__ import annotations

from collections import deque

import numpy as np


class ActionBuffer:
    def __init__(self, horizon: int) -> None:
        if not 1 <= horizon <= 50:
            raise ValueError("action horizon must be between 1 and 50")
        self._horizon = horizon
        self._actions: deque[np.ndarray] = deque()

    def __len__(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._actions.clear()

    def peek(self) -> np.ndarray | None:
        return self._actions[0] if self._actions else None

    def replace(self, actions: np.ndarray, elapsed_steps: int, *, crossfade_steps: int = 0) -> int:
        if elapsed_steps < 0:
            raise ValueError("elapsed steps must be nonnegative")
        if isinstance(crossfade_steps, bool) or not isinstance(crossfade_steps, int) or crossfade_steps not in {0, 5}:
            raise ValueError("crossfade steps must be exactly 0 or 5")
        action_array = np.asarray(actions)
        if (
            action_array.ndim != 2
            or action_array.shape[1] != 14
            or not np.issubdtype(action_array.dtype, np.floating)
            or not np.isfinite(action_array).all()
        ):
            raise ValueError("actions must contain finite 14D vectors")

        fresh = action_array[elapsed_steps : elapsed_steps + self._horizon].astype(np.float64, copy=True)
        applied = min(crossfade_steps, len(self._actions), len(fresh))
        if applied:
            old = np.asarray(list(self._actions)[:applied])
            if old.shape != (applied, 14) or not np.issubdtype(old.dtype, np.floating) or not np.isfinite(old).all():
                raise ValueError("buffered actions must contain finite 14D vectors")
            alpha = np.arange(1, applied + 1, dtype=np.float64)[:, None] / applied
            with np.errstate(invalid="ignore", over="ignore"):
                fresh[:applied] = old * (1 - alpha) + fresh[:applied] * alpha
            if not np.isfinite(fresh[:applied]).all():
                raise ValueError("crossfaded actions must contain finite 14D vectors")

        self._actions = deque(fresh)
        return applied

    def pop(self) -> np.ndarray:
        if not self._actions:
            raise RuntimeError("action buffer is empty")
        return self._actions.popleft()
