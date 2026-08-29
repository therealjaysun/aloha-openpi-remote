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

    def replace(self, actions: np.ndarray, elapsed_steps: int) -> None:
        if elapsed_steps < 0:
            raise ValueError("elapsed steps must be nonnegative")
        self._actions = deque(actions[elapsed_steps : elapsed_steps + self._horizon])

    def pop(self) -> np.ndarray:
        if not self._actions:
            raise RuntimeError("action buffer is empty")
        return self._actions.popleft()
